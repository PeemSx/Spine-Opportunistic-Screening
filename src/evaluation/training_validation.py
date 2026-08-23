from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.inference_geometry import (
    TransformMeta,
    map_points_to_original,
    valid_center_mask,
)
from src.evaluation.center_metrics import CenterMetricAccumulator
from src.evaluation.centernet_decode import decode_centernet_outputs
from src.evaluation.scorecard import (
    ImageEvaluation,
    VERTEBRA_SCALE_BINS,
    evaluate_prediction,
    scale_summary_rows,
    source_summary_rows,
)
from src.postprocessing.spine_chain import (
    candidates_to_prediction,
    prediction_to_candidates,
    select_spine_chain,
)


@dataclass(frozen=True)
class ValidationLandmarkResult:
    flat_metrics: dict[str, float]
    report: dict[str, Any]
    instance_rows: list[dict[str, Any]]


def _value_at(value: Any, index: int) -> Any:
    item = value[index]
    return item.item() if torch.is_tensor(item) else item


def _transform_meta_from_batch(batch: dict[str, Any], index: int) -> TransformMeta:
    return TransformMeta(
        original_width=int(_value_at(batch["original_width"], index)),
        original_height=int(_value_at(batch["original_height"], index)),
        resized_width=int(_value_at(batch["resized_width"], index)),
        resized_height=int(_value_at(batch["resized_height"], index)),
        pad_left=int(_value_at(batch["pad_left"], index)),
        pad_top=int(_value_at(batch["pad_top"], index)),
        scale=float(_value_at(batch["scale"], index)),
        input_size=int(_value_at(batch["input_size"], index)),
    )


def _prediction_to_original(
    prediction: dict[str, np.ndarray],
    meta: TransformMeta,
) -> dict[str, np.ndarray]:
    centers = map_points_to_original(prediction["centers"], meta)
    corners = map_points_to_original(prediction["corners"], meta)
    mask = valid_center_mask(centers, meta)
    return {
        "scores": np.asarray(prediction["scores"][mask], dtype=np.float32),
        "centers": np.asarray(centers[mask], dtype=np.float32),
        "corners": np.asarray(corners[mask], dtype=np.float32),
    }


def _chain_prediction(
    prediction: dict[str, np.ndarray],
    chain_settings: dict[str, float | int],
) -> dict[str, np.ndarray]:
    _, selected, _ = select_spine_chain(
        prediction_to_candidates(prediction),
        duplicate_iou_threshold=float(chain_settings["duplicate_iou"]),
        duplicate_center_scale=float(chain_settings["duplicate_center_scale"]),
        score_threshold=float(chain_settings["score_thresh"]),
        score_weight=float(chain_settings["score_weight"]),
        min_chain_len=int(chain_settings["min_len"]),
    )
    return candidates_to_prediction(selected)


def _flatten_summary(
    target: dict[str, float],
    prefix: str,
    summary: dict[str, Any],
) -> None:
    for key, value in summary.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not np.isfinite(float(value)):
            continue
        target[f"{prefix}_{key}"] = float(value)


@torch.no_grad()
def evaluate_validation_landmarks(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    down_ratio: int,
    peak_thresh: float,
    topk: int,
    chain_settings: dict[str, float | int],
    max_batches: int | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_every: int = 25,
) -> ValidationLandmarkResult:
    model.eval()
    evaluations: dict[str, list[ImageEvaluation]] = {
        "raw": [],
        "spine_chain": [],
    }
    legacy_centers = CenterMetricAccumulator()
    batches = 0
    images = 0
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    epoch_label = (
        f"epoch {epoch}/{total_epochs} landmark-eval"
        if epoch is not None and total_epochs is not None
        else "landmark-eval"
    )

    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        inputs = batch["input"].to(device, non_blocking=True)
        outputs = model(inputs)
        raw_padded = decode_centernet_outputs(
            outputs,
            down_ratio=down_ratio,
            peak_thresh=peak_thresh,
            topk=topk,
        )
        legacy_centers.update(raw_padded, batch["gt_centers"], batch["gt_count"])

        gt_centers_original = batch["gt_centers_original"].detach().cpu().numpy()
        gt_corners_original = batch["gt_corners_original"].detach().cpu().numpy()
        gt_counts = batch["gt_count"].detach().cpu().numpy().astype(int)
        for sample_index, padded_prediction in enumerate(raw_padded):
            count = int(gt_counts[sample_index])
            meta = _transform_meta_from_batch(batch, sample_index)
            raw_original = _prediction_to_original(padded_prediction, meta)
            chain_original = _chain_prediction(raw_original, chain_settings)
            gt_centers = gt_centers_original[sample_index, :count]
            gt_corners = gt_corners_original[sample_index, :count]
            common = {
                "gt_centers": gt_centers,
                "gt_corners": gt_corners,
                "image": str(batch["file_name"][sample_index]),
                "image_id": int(_value_at(batch["image_id"], sample_index)),
                "source_dataset": str(batch["source_dataset"][sample_index]),
                "cluster_id": str(batch["patient_cluster_id"][sample_index]),
                "image_width": meta.original_width,
                "image_height": meta.original_height,
            }
            evaluations["raw"].append(
                evaluate_prediction(model="raw", prediction=raw_original, **common)
            )
            evaluations["spine_chain"].append(
                evaluate_prediction(
                    model="spine_chain",
                    prediction=chain_original,
                    **common,
                )
            )
            images += 1

        batches += 1
        if progress_every > 0 and (
            batches % progress_every == 0 or batches == total_batches
        ):
            print(
                f"{epoch_label} batch {batches}/{total_batches} | images {images}",
                flush=True,
            )

    flat_metrics = legacy_centers.compute()
    models_report: dict[str, Any] = {}
    instance_rows: list[dict[str, Any]] = []
    for model_name in ("raw", "spine_chain"):
        overall, per_source, source_macro, worst_source = source_summary_rows(
            evaluations[model_name],
            model_name,
        )
        by_scale, by_source_scale = scale_summary_rows(
            evaluations[model_name],
            model_name,
        )
        models_report[model_name] = {
            "overall": overall,
            "source_macro": source_macro,
            "worst_source": worst_source,
            "by_source": per_source,
            "by_scale": by_scale,
            "by_source_scale": by_source_scale,
        }
        _flatten_summary(flat_metrics, model_name, overall)
        _flatten_summary(flat_metrics, f"{model_name}_source_macro", source_macro)
        _flatten_summary(flat_metrics, f"{model_name}_worst_source", worst_source)
        instance_rows.extend(
            row
            for item in evaluations[model_name]
            for row in item.instance_rows
        )

    report = {
        "schema_version": 1,
        "epoch": epoch,
        "images": images,
        "coordinate_space": "original_image_pixels",
        "signed_residual_convention": (
            "prediction_minus_ground_truth; positive dx is right and positive dy is down"
        ),
        "metric_protocol": {
            "matching": "one_to_one_hungarian_normalized_center_distance",
            "normalization": "gt_vertebra_bounding_box_diagonal",
            "primary_detection_gate": "0.20D",
            "pck_thresholds": [0.05, 0.10, 0.20],
            "usable_vertebra": (
                "accepted center match, finite non-self-intersecting in-image "
                "quadrilateral, and NME <= 0.10"
            ),
            "padding_policy": (
                "predictions with centers outside the original image are removed "
                "before matching and spine-chain selection"
            ),
        },
        "scale_bins": [
            {
                "label": label,
                "diagonal_fraction_min": lower,
                "diagonal_fraction_max": upper if np.isfinite(upper) else None,
            }
            for label, lower, upper in VERTEBRA_SCALE_BINS
        ],
        "chain_settings": dict(chain_settings),
        "medical_scope": (
            "Corner-derived geometric agreement for research screening support; "
            "these metrics are not diagnostic accuracy."
        ),
        "models": models_report,
    }
    return ValidationLandmarkResult(
        flat_metrics=flat_metrics,
        report=report,
        instance_rows=instance_rows,
    )
