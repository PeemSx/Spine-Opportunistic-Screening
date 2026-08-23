from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.train_masked_morphology_xgboost import load_dataset


SPLITS = ("train", "val", "test")
EVALUATION_SPLITS = ("val", "test")
CONTEXT_NAMES = ("m2", "m1", "p1", "p2")
HEIGHT_FEATURES = tuple(f"x_{name}_mean_height_norm" for name in CONTEXT_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one XGBoost model for masked mean vertebral height."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/processed/masked_morphology_coco_nih_lumos"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/masked_morphology/xgboost_height_only_v1"),
    )
    parser.add_argument(
        "--feature-set",
        choices=("full_context", "height_only"),
        default="full_context",
        help=(
            "full_context uses all surrounding morphology to predict height; "
            "height_only is the four-mean-height ablation."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=("reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"),
        default="reg:pseudohubererror",
    )
    parser.add_argument("--n-estimators", type=int, default=1600)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=4)
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


def build_height_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for context in CONTEXT_NAMES:
        left = frame[f"x_{context}_left_height_norm"].to_numpy(dtype=np.float64)
        right = frame[f"x_{context}_right_height_norm"].to_numpy(dtype=np.float64)
        result[f"x_{context}_mean_height_norm"] = 0.5 * (left + right)
    return result.loc[:, HEIGHT_FEATURES]


def height_targets(frame: pd.DataFrame) -> pd.DataFrame:
    actual = 0.5 * (
        frame["y_left_height_norm"].to_numpy(dtype=np.float64)
        + frame["y_right_height_norm"].to_numpy(dtype=np.float64)
    )
    baseline = 0.5 * (
        frame["baseline_left_height_norm"].to_numpy(dtype=np.float64)
        + frame["baseline_right_height_norm"].to_numpy(dtype=np.float64)
    )
    return pd.DataFrame(
        {
            "actual_mean_height_norm": actual,
            "baseline_mean_height_norm": baseline,
            "mean_height_residual": actual - baseline,
        },
        index=frame.index,
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def weighted_r2(
    actual: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    actual_mean = weighted_mean(actual, weights)
    residual_sum = float(np.sum(weights * np.square(actual - predicted)))
    total_sum = float(np.sum(weights * np.square(actual - actual_mean)))
    if total_sum <= np.finfo(np.float64).eps:
        return 1.0 if residual_sum <= np.finfo(np.float64).eps else 0.0
    return 1.0 - residual_sum / total_sum


def weighted_within_percent(
    errors: np.ndarray,
    threshold: float,
    weights: np.ndarray,
) -> float:
    return 100.0 * weighted_mean(
        (np.asarray(errors) <= float(threshold)).astype(np.float64),
        weights,
    )


def height_metrics(
    actual: np.ndarray,
    baseline: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    baseline_error = np.abs(actual - baseline)
    model_error = np.abs(actual - predicted)
    baseline_mae = weighted_mean(baseline_error, weights)
    model_mae = weighted_mean(model_error, weights)
    return {
        "samples": len(actual),
        "baseline_mae": baseline_mae,
        "model_mae": model_mae,
        "baseline_rmse": float(
            np.sqrt(weighted_mean(np.square(baseline_error), weights))
        ),
        "model_rmse": float(
            np.sqrt(weighted_mean(np.square(model_error), weights))
        ),
        "baseline_r2": weighted_r2(actual, baseline, weights),
        "model_r2": weighted_r2(actual, predicted, weights),
        "model_median_absolute_error": float(np.median(model_error)),
        "model_p90_absolute_error": float(np.quantile(model_error, 0.90)),
        "model_p95_absolute_error": float(np.quantile(model_error, 0.95)),
        "baseline_within_5pct_reference_percent": weighted_within_percent(
            baseline_error, 0.05, weights
        ),
        "model_within_5pct_reference_percent": weighted_within_percent(
            model_error, 0.05, weights
        ),
        "baseline_within_10pct_reference_percent": weighted_within_percent(
            baseline_error, 0.10, weights
        ),
        "model_within_10pct_reference_percent": weighted_within_percent(
            model_error, 0.10, weights
        ),
        "relative_mae_improvement_percent": (
            100.0 * (baseline_mae - model_mae) / baseline_mae
            if baseline_mae > 0.0
            else 0.0
        ),
    }


def build_model(args: argparse.Namespace) -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "XGBoost is not installed. Run: python -m pip install -r requirement.txt"
        ) from error
    return XGBRegressor(
        objective=args.objective,
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


def metric_row(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    predicted_residual: np.ndarray,
    *,
    split: str,
    source_dataset: str = "all",
) -> dict[str, Any]:
    predicted = (
        targets["baseline_mean_height_norm"].to_numpy()
        + np.asarray(predicted_residual)
    )
    return {
        "split": split,
        "source_dataset": source_dataset,
        "output": "mean_height",
        "unit": "fraction_of_context_median_height",
        **height_metrics(
            targets["actual_mean_height_norm"].to_numpy(),
            targets["baseline_mean_height_norm"].to_numpy(),
            predicted,
            frame["sample_weight"].to_numpy(),
        ),
    }


def prediction_frame(
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    predicted_residual: np.ndarray,
    *,
    split: str,
) -> pd.DataFrame:
    metadata = [
        "sample_id",
        "group_id",
        "image_path",
        "source_dataset",
        "view",
        "target_annotation_id",
        "target_chain_rank",
        "chain_count",
        "sample_weight",
        "reference_height_px",
    ]
    result = frame[metadata].copy()
    result.insert(1, "split", split)
    result["actual_mean_height_norm"] = targets["actual_mean_height_norm"]
    result["baseline_mean_height_norm"] = targets["baseline_mean_height_norm"]
    result["predicted_mean_height_norm"] = (
        targets["baseline_mean_height_norm"].to_numpy()
        + np.asarray(predicted_residual)
    )
    result["absolute_error_norm"] = np.abs(
        result["actual_mean_height_norm"] - result["predicted_mean_height_norm"]
    )
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    frames, schema, full_context_features = load_dataset(
        args.dataset_root,
        limit_per_split=args.limit_per_split,
    )
    if args.feature_set == "height_only":
        feature_columns = list(HEIGHT_FEATURES)
        features = {
            split: build_height_features(frame) for split, frame in frames.items()
        }
    else:
        feature_columns = list(full_context_features)
        features = {
            split: frame.loc[:, feature_columns] for split, frame in frames.items()
        }
    targets = {split: height_targets(frame) for split, frame in frames.items()}
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
    model.save_model(output_dir / "mean_height.json")

    predictions = {
        split: model.predict(features[split]) for split in EVALUATION_SPLITS
    }
    metric_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    prediction_tables = []
    for split in EVALUATION_SPLITS:
        frame = frames[split]
        metric_rows.append(
            metric_row(frame, targets[split], predictions[split], split=split)
        )
        prediction_tables.append(
            prediction_frame(frame, targets[split], predictions[split], split=split)
        )
        for source in sorted(frame["source_dataset"].unique()):
            mask = frame["source_dataset"].eq(source).to_numpy()
            source_rows.append(
                metric_row(
                    frame.loc[mask],
                    targets[split].loc[mask],
                    predictions[split][mask],
                    split=split,
                    source_dataset=str(source),
                )
            )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output_dir / "metrics_by_source.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_.astype(float),
        }
    ).to_csv(output_dir / "feature_importance.csv", index=False)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(model.evals_result(), file, indent=2)
        file.write("\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "research_use_only": True,
                "task": "masked mean vertebral height reconstruction",
                "rows": metric_rows,
            },
            file,
            indent=2,
        )
        file.write("\n")

    import xgboost

    manifest = {
        "task": "four-context masked mean vertebral height residual regression",
        "research_use_only": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "feature_schema_version": schema.get("schema_version"),
        "feature_set": args.feature_set,
        "feature_columns": feature_columns,
        "target_formula": "0.5 * (left_height_norm + right_height_norm)",
        "normalization": "median mean height of m2, m1, p1, p2 context vertebrae",
        "baseline": "mean height of m1 and p1",
        "split_rows": {split: len(frame) for split, frame in frames.items()},
        "best_iteration": int(model.best_iteration),
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
