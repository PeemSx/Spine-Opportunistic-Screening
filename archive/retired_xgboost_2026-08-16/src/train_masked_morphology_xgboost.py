from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class OutputContract:
    name: str
    residual_column: str
    actual_column: str
    baseline_column: str
    scale_column: str | None
    unit: str


OUTPUT_CONTRACTS = {
    name: OutputContract(
        name=name,
        residual_column=f"y_{name}_residual",
        actual_column=f"y_{name}_norm",
        baseline_column=f"baseline_{name}_norm",
        scale_column=(
            "reference_height_px" if "height" in name else "reference_width_px"
        ),
        unit="normalized",
    )
    for name in (
        "left_height",
        "right_height",
        "superior_width",
        "inferior_width",
    )
}
OUTPUT_CONTRACTS["orientation"] = OutputContract(
    name="orientation",
    residual_column="y_orientation_residual_deg",
    actual_column="y_orientation_deg",
    baseline_column="baseline_orientation_deg",
    scale_column=None,
    unit="degrees",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost models for masked frontal vertebral morphology."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/processed/masked_morphology_coco_nih_lumos"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/masked_morphology/xgboost_four_context"),
    )
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-child-weight", type=float, default=4.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Development-only row limit; omit for real training.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def wrap_axial_deg(angle: np.ndarray | Sequence[float] | float) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float64) + 90.0) % 180.0 - 90.0


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
    within = np.asarray(errors, dtype=np.float64) <= float(threshold)
    return 100.0 * weighted_mean(within.astype(np.float64), weights)


def absolute_residual_error(
    contract: OutputContract,
    actual_residual: np.ndarray,
    predicted_residual: np.ndarray,
) -> np.ndarray:
    difference = np.asarray(actual_residual) - np.asarray(predicted_residual)
    if contract.name == "orientation":
        return np.abs(wrap_axial_deg(difference))
    return np.abs(difference)


def regression_metrics(
    contract: OutputContract,
    actual_residual: np.ndarray,
    predicted_residual: np.ndarray,
    weights: np.ndarray,
    *,
    scale: np.ndarray | None = None,
) -> dict[str, float | int]:
    model_error = absolute_residual_error(
        contract, actual_residual, predicted_residual
    )
    baseline_error = absolute_residual_error(
        contract, actual_residual, np.zeros_like(actual_residual)
    )
    baseline_mae = weighted_mean(baseline_error, weights)
    model_mae = weighted_mean(model_error, weights)
    metrics: dict[str, float | int] = {
        "samples": len(model_error),
        "baseline_mae": baseline_mae,
        "model_mae": model_mae,
        "baseline_rmse": float(
            np.sqrt(weighted_mean(np.square(baseline_error), weights))
        ),
        "model_rmse": float(
            np.sqrt(weighted_mean(np.square(model_error), weights))
        ),
        "baseline_residual_r2": weighted_r2(
            actual_residual,
            np.zeros_like(actual_residual),
            weights,
        ),
        "model_residual_r2": weighted_r2(
            actual_residual,
            predicted_residual,
            weights,
        ),
        "unweighted_model_mae": float(np.mean(model_error)),
        "model_median_absolute_error": float(np.median(model_error)),
        "model_p90_absolute_error": float(np.quantile(model_error, 0.90)),
        "model_p95_absolute_error": float(np.quantile(model_error, 0.95)),
        "relative_mae_improvement_percent": (
            100.0 * (baseline_mae - model_mae) / baseline_mae
            if baseline_mae > 0.0
            else 0.0
        ),
    }
    if scale is not None:
        pixel_error = model_error * np.asarray(scale)
        baseline_pixel_error = baseline_error * np.asarray(scale)
        metrics.update(
            {
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
                "baseline_mae_px": weighted_mean(baseline_pixel_error, weights),
                "model_mae_px": weighted_mean(pixel_error, weights),
                "baseline_rmse_px": float(
                    np.sqrt(weighted_mean(np.square(baseline_pixel_error), weights))
                ),
                "model_rmse_px": float(
                    np.sqrt(weighted_mean(np.square(pixel_error), weights))
                ),
                "model_p95_absolute_error_px": float(
                    np.quantile(pixel_error, 0.95)
                ),
                "baseline_within_5px_percent": weighted_within_percent(
                    baseline_pixel_error, 5.0, weights
                ),
                "model_within_5px_percent": weighted_within_percent(
                    pixel_error, 5.0, weights
                ),
                "baseline_within_10px_percent": weighted_within_percent(
                    baseline_pixel_error, 10.0, weights
                ),
                "model_within_10px_percent": weighted_within_percent(
                    pixel_error, 10.0, weights
                ),
            }
        )
    else:
        metrics.update(
            {
                "baseline_within_2deg_percent": weighted_within_percent(
                    baseline_error, 2.0, weights
                ),
                "model_within_2deg_percent": weighted_within_percent(
                    model_error, 2.0, weights
                ),
                "baseline_within_5deg_percent": weighted_within_percent(
                    baseline_error, 5.0, weights
                ),
                "model_within_5deg_percent": weighted_within_percent(
                    model_error, 5.0, weights
                ),
            }
        )
    return metrics


def load_dataset(
    dataset_root: Path,
    *,
    limit_per_split: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[str]]:
    root = dataset_root.resolve(strict=True)
    schema = load_json(root / "feature_schema.json")
    features = [str(value) for value in schema.get("numeric_input_columns", [])]
    if not features:
        raise ValueError("feature_schema.json contains no numeric input columns")
    if any(name.startswith(("y_", "baseline_")) for name in features):
        raise ValueError("Target or baseline leakage exists in numeric_input_columns")

    required = set(features)
    required.update(
        {
            "sample_id",
            "group_id",
            "sample_weight",
            "source_dataset",
            "source_dataset_key",
            "view",
            "image_path",
            "target_annotation_id",
            "target_chain_rank",
            "chain_count",
            "reference_height_px",
            "reference_width_px",
        }
    )
    for contract in OUTPUT_CONTRACTS.values():
        required.update(
            {
                contract.residual_column,
                contract.actual_column,
                contract.baseline_column,
            }
        )

    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = root / split / "masked_samples.csv"
        frame = pd.read_csv(path, low_memory=False)
        if limit_per_split is not None:
            frame = frame.head(limit_per_split).copy()
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{path} contains no rows")
        numeric = features + [
            contract.residual_column for contract in OUTPUT_CONTRACTS.values()
        ]
        numeric += ["sample_weight", "reference_height_px", "reference_width_px"]
        if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{path} contains non-finite model values")
        if (frame["sample_weight"] <= 0.0).any():
            raise ValueError(f"{path} contains nonpositive sample weights")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate sample IDs")
        frames[split] = frame

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(frames[first]["group_id"]) & set(frames[second]["group_id"])
        if overlap:
            raise ValueError(
                f"Group leakage between {first} and {second}: {len(overlap)} groups"
            )
    return frames, schema, features


def xgb_regressor_class() -> Any:
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "XGBoost is not installed. Run: python -m pip install -r requirement.txt"
        ) from error
    return XGBRegressor


def build_model(args: argparse.Namespace) -> Any:
    return xgb_regressor_class()(
        objective="reg:squarederror",
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
    (result / "models").mkdir(exist_ok=True)
    return result


def metric_rows(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    *,
    split: str,
    source_dataset: str = "all",
) -> list[dict[str, Any]]:
    rows = []
    for name, contract in OUTPUT_CONTRACTS.items():
        scale = (
            frame[contract.scale_column].to_numpy()
            if contract.scale_column is not None
            else None
        )
        rows.append(
            {
                "split": split,
                "source_dataset": source_dataset,
                "output": name,
                "unit": contract.unit,
                **regression_metrics(
                    contract,
                    frame[contract.residual_column].to_numpy(),
                    predictions[name],
                    frame["sample_weight"].to_numpy(),
                    scale=scale,
                ),
            }
        )
    return rows


def prediction_frame(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
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
    ]
    result = frame[metadata].copy()
    result.insert(1, "split", split)
    for name, contract in OUTPUT_CONTRACTS.items():
        actual_residual = frame[contract.residual_column].to_numpy()
        predicted_residual = np.asarray(predictions[name])
        baseline = frame[contract.baseline_column].to_numpy()
        predicted_value = baseline + predicted_residual
        if name == "orientation":
            predicted_value = wrap_axial_deg(predicted_value)
        result[f"{name}_actual"] = frame[contract.actual_column].to_numpy()
        result[f"{name}_baseline"] = baseline
        result[f"{name}_predicted"] = predicted_value
        result[f"{name}_absolute_error"] = absolute_residual_error(
            contract, actual_residual, predicted_residual
        )
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    frames, schema, features = load_dataset(
        args.dataset_root, limit_per_split=args.limit_per_split
    )
    output_dir = prepare_output_directory(args.output_dir, force=args.force)
    X_train = frames["train"][features]
    X_val = frames["val"][features]
    X_test = frames["test"][features]
    train_weights = frames["train"]["sample_weight"].to_numpy()
    val_weights = frames["val"]["sample_weight"].to_numpy()
    models: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {"val": {}, "test": {}}
    histories: dict[str, Any] = {}
    importance_rows = []

    for name, contract in OUTPUT_CONTRACTS.items():
        print(f"Training {name}...")
        model = build_model(args)
        model.fit(
            X_train,
            frames["train"][contract.residual_column],
            sample_weight=train_weights,
            eval_set=[(X_val, frames["val"][contract.residual_column])],
            sample_weight_eval_set=[val_weights],
            verbose=False,
        )
        model.save_model(output_dir / "models" / f"{name}.json")
        models[name] = model
        predictions["val"][name] = model.predict(X_val)
        predictions["test"][name] = model.predict(X_test)
        histories[name] = model.evals_result()
        for feature, importance in zip(features, model.feature_importances_):
            importance_rows.append(
                {"output": name, "feature": feature, "importance": float(importance)}
            )
        print(f"  best_iteration={model.best_iteration}")

    metrics = []
    source_metrics = []
    prediction_tables = []
    for split in ("val", "test"):
        frame = frames[split]
        metrics.extend(metric_rows(frame, predictions[split], split=split))
        prediction_tables.append(prediction_frame(frame, predictions[split], split=split))
        for source in sorted(frame["source_dataset"].unique()):
            mask = frame["source_dataset"].eq(source).to_numpy()
            source_metrics.extend(
                metric_rows(
                    frame.loc[mask],
                    {name: values[mask] for name, values in predictions[split].items()},
                    split=split,
                    source_dataset=str(source),
                )
            )

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(source_metrics).to_csv(
        output_dir / "metrics_by_source.csv", index=False
    )
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False
    )
    pd.DataFrame(importance_rows).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "research_use_only": True,
                "task": "observed masked morphology reconstruction",
                "rows": metrics,
            },
            file,
            indent=2,
        )
        file.write("\n")
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(histories, file, indent=2)
        file.write("\n")

    import xgboost

    manifest = {
        "task": "four-context masked vertebral morphology residual regression",
        "research_use_only": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "feature_schema_version": schema.get("schema_version"),
        "feature_count": len(features),
        "feature_columns": features,
        "outputs": list(OUTPUT_CONTRACTS),
        "split_rows": {split: len(frame) for split, frame in frames.items()},
        "best_iterations": {
            name: int(model.best_iteration) for name, model in models.items()
        },
        "hyperparameters": vars(args) | {"dataset_root": str(args.dataset_root), "output_dir": str(args.output_dir)},
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    return {"output_dir": output_dir, "metrics": metrics_frame, "manifest": manifest}


def main() -> None:
    result = train(parse_args())
    print(f"Artifacts: {result['output_dir']}")
    print(
        result["metrics"][
            [
                "split",
                "output",
                "baseline_mae",
                "model_mae",
                "relative_mae_improvement_percent",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
