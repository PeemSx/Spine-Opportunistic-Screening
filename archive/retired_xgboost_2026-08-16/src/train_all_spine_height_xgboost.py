from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.build_masked_morphology_dataset import wrap_axial_deg
from src.train_masked_height_xgboost import height_metrics, height_targets
from src.train_masked_morphology_xgboost import load_dataset


SPLITS = ("train", "val", "test")
EVALUATION_SPLITS = ("val", "test")
ALL_SPINE_FEATURES = (
    "left_height_norm",
    "right_height_norm",
    "superior_width_norm",
    "inferior_width_norm",
    "orientation_relative_deg",
    "center_dx_norm",
    "center_dy_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost for masked height using every visible vertebra."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/processed/masked_morphology_coco_nih_lumos"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/masked_morphology/xgboost_all_vertebrae_height_v1"),
    )
    parser.add_argument("--max-offset", type=int, default=23)
    parser.add_argument("--n-estimators", type=int, default=1800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-child-weight", type=float, default=4.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def relative_offsets(max_offset: int) -> tuple[int, ...]:
    if max_offset < 2:
        raise ValueError("max_offset must be at least 2")
    return tuple(offset for offset in range(-max_offset, max_offset + 1) if offset)


def all_spine_feature_columns(max_offset: int) -> list[str]:
    return [
        f"x_{offset:+d}_{feature}"
        for offset in relative_offsets(max_offset)
        for feature in ALL_SPINE_FEATURES
    ]


def load_vertebrae(dataset_root: Path, split: str) -> dict[str, pd.DataFrame]:
    path = dataset_root / split / "vertebrae.csv"
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[frame["geometry_valid"].astype(bool)].copy()
    return {
        str(image_key): group.sort_values("chain_rank")
        for image_key, group in frame.groupby("image_key", sort=False)
    }


def build_all_spine_features(
    samples: pd.DataFrame,
    chains: dict[str, pd.DataFrame],
    *,
    max_offset: int,
) -> pd.DataFrame:
    offsets = relative_offsets(max_offset)
    offset_start = {
        offset: index * len(ALL_SPINE_FEATURES)
        for index, offset in enumerate(offsets)
    }
    columns = all_spine_feature_columns(max_offset)
    values = np.full((len(samples), len(columns)), np.nan, dtype=np.float32)

    for row_index, sample in enumerate(samples.itertuples(index=False)):
        chain = chains.get(str(sample.image_key))
        if chain is None:
            raise ValueError(f"No vertebral chain for {sample.sample_id}")
        target_matches = chain["coco_annotation_id"].eq(
            int(sample.target_annotation_id)
        )
        if int(target_matches.sum()) != 1:
            raise ValueError(f"Target lookup failed for {sample.sample_id}")

        for vertebra in chain.itertuples(index=False):
            offset = int(vertebra.chain_rank) - int(sample.target_chain_rank)
            if offset == 0:
                continue
            if offset not in offset_start:
                raise ValueError(
                    f"Relative offset {offset} exceeds max_offset={max_offset}"
                )
            start = offset_start[offset]
            values[row_index, start : start + len(ALL_SPINE_FEATURES)] = (
                float(vertebra.left_height) / float(sample.reference_height_px),
                float(vertebra.right_height) / float(sample.reference_height_px),
                float(vertebra.superior_width) / float(sample.reference_width_px),
                float(vertebra.inferior_width) / float(sample.reference_width_px),
                float(
                    wrap_axial_deg(
                        float(vertebra.orientation_deg)
                        - float(sample.baseline_orientation_deg)
                    )
                ),
                (
                    float(vertebra.center_x) - float(sample.baseline_center_x_px)
                )
                / float(sample.reference_width_px),
                (
                    float(vertebra.center_y) - float(sample.baseline_center_y_px)
                )
                / float(sample.reference_height_px),
            )
    return pd.DataFrame(values, columns=columns, index=samples.index)


def build_model(args: argparse.Namespace) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "XGBoost is not installed. Run: python -m pip install -r requirement.txt"
        ) from error
    return XGBRegressor(
        objective="reg:pseudohubererror",
        eval_metric="mae",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        tree_method="hist",
        device=args.device,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )


def prepare_output_directory(path: Path, *, force: bool) -> Path:
    result = path.resolve()
    if result.exists() and any(result.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {result}. Use a new path or --force."
        )
    result.mkdir(parents=True, exist_ok=True)
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve(strict=True)
    frames, schema, _ = load_dataset(
        dataset_root,
        limit_per_split=args.limit_per_split,
    )
    features = {}
    targets = {}
    for split in SPLITS:
        chains = load_vertebrae(dataset_root, split)
        features[split] = build_all_spine_features(
            frames[split], chains, max_offset=args.max_offset
        )
        targets[split] = height_targets(frames[split])

    output_dir = prepare_output_directory(args.output_dir, force=args.force)
    model = build_model(args)
    model.fit(
        features["train"],
        targets["train"]["mean_height_residual"],
        sample_weight=frames["train"]["sample_weight"].to_numpy(),
        eval_set=[(features["val"], targets["val"]["mean_height_residual"])],
        sample_weight_eval_set=[frames["val"]["sample_weight"].to_numpy()],
        verbose=False,
    )
    model.save_model(output_dir / "mean_height_all_vertebrae.json")

    metric_rows = []
    prediction_tables = []
    for split in EVALUATION_SPLITS:
        predicted_residual = model.predict(features[split])
        predicted_height = (
            targets[split]["baseline_mean_height_norm"].to_numpy()
            + predicted_residual
        )
        metric_rows.append(
            {
                "split": split,
                "method": "xgboost_all_visible_vertebrae",
                "output": "mean_height",
                "unit": "fraction_of_context_median_height",
                **height_metrics(
                    targets[split]["actual_mean_height_norm"].to_numpy(),
                    targets[split]["baseline_mean_height_norm"].to_numpy(),
                    predicted_height,
                    frames[split]["sample_weight"].to_numpy(),
                ),
            }
        )
        metadata = [
            "sample_id",
            "group_id",
            "image_path",
            "source_dataset",
            "target_annotation_id",
            "target_chain_rank",
            "chain_count",
            "sample_weight",
        ]
        table = frames[split][metadata].copy()
        table.insert(1, "split", split)
        table["actual_mean_height_norm"] = targets[split][
            "actual_mean_height_norm"
        ]
        table["baseline_mean_height_norm"] = targets[split][
            "baseline_mean_height_norm"
        ]
        table["predicted_mean_height_norm"] = predicted_height
        table["absolute_error_norm"] = np.abs(
            table["actual_mean_height_norm"] - table["predicted_mean_height_norm"]
        )
        prediction_tables.append(table)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    importance = pd.DataFrame(
        {
            "feature": features["train"].columns,
            "importance": model.feature_importances_.astype(float),
            "observed_train_fraction": features["train"].notna().mean().to_numpy(),
        }
    ).sort_values("importance", ascending=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(model.evals_result(), file, indent=2)
        file.write("\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "research_use_only": True,
                "task": "masked mean vertebral height using all visible vertebrae",
                "rows": metric_rows,
            },
            file,
            indent=2,
        )
        file.write("\n")

    import xgboost

    manifest = {
        "task": "all-visible-vertebrae masked mean height residual regression",
        "research_use_only": True,
        "dataset_root": str(dataset_root),
        "feature_schema_version": schema.get("schema_version"),
        "max_offset": args.max_offset,
        "feature_count": features["train"].shape[1],
        "feature_columns": list(features["train"].columns),
        "missing_value_contract": "NaN means no visible vertebra at that relative offset",
        "target_exclusion_contract": "relative offset 0 is never an input feature",
        "normalization": "immediate four-context median height and width",
        "best_iteration": int(model.best_iteration),
        "split_rows": {split: len(frame) for split, frame in frames.items()},
        "hyperparameters": vars(args)
        | {
            "dataset_root": str(args.dataset_root),
            "output_dir": str(args.output_dir),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, default=str)
        file.write("\n")
    return {"output_dir": output_dir, "metrics": metrics, "manifest": manifest}


def main() -> None:
    result = train(parse_args())
    print(f"Artifacts: {result['output_dir']}")
    print(
        result["metrics"][
            [
                "split",
                "baseline_mae",
                "model_mae",
                "model_within_5pct_reference_percent",
                "model_within_10pct_reference_percent",
                "model_r2",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
