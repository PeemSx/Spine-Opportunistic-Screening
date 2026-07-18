from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from src.evaluation.corner_refiner_decode import transform_points_batch


CORNER_ORDER = ("TL", "TR", "BL", "BR")


def map_crop_predictions_to_original(
    points_crop: torch.Tensor,
    crop_to_original: torch.Tensor,
) -> torch.Tensor:
    return transform_points_batch(points_crop, crop_to_original)


def _line_angle(points: np.ndarray, first: int, second: int) -> np.ndarray:
    vector = points[:, second] - points[:, first]
    return np.degrees(np.arctan2(vector[:, 1], vector[:, 0]))


def _angle_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs((predicted - target + 90.0) % 180.0 - 90.0)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1e-6)


def compute_corner_metrics(
    predicted_original: np.ndarray,
    target_original: np.ndarray,
    bbox_diagonals: np.ndarray,
    *,
    boundary_peaks: np.ndarray | None = None,
) -> dict[str, Any]:
    predicted = np.asarray(predicted_original, dtype=np.float64)
    target = np.asarray(target_original, dtype=np.float64)
    diagonals = np.asarray(bbox_diagonals, dtype=np.float64).reshape(-1)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[1:] != (4, 2):
        raise ValueError("predicted and target points must have shape [N, 4, 2]")
    if len(diagonals) != len(predicted):
        raise ValueError("bbox_diagonals length must match samples")

    pixel_errors = np.linalg.norm(predicted - target, axis=-1)
    normalized_corner_errors = pixel_errors / np.maximum(diagonals[:, None], 1e-6)
    sample_nme = normalized_corner_errors.mean(axis=1)

    pred_superior = _line_angle(predicted, 0, 1)
    gt_superior = _line_angle(target, 0, 1)
    pred_inferior = _line_angle(predicted, 2, 3)
    gt_inferior = _line_angle(target, 2, 3)
    pred_left = np.linalg.norm(predicted[:, 2] - predicted[:, 0], axis=-1)
    gt_left = np.linalg.norm(target[:, 2] - target[:, 0], axis=-1)
    pred_right = np.linalg.norm(predicted[:, 3] - predicted[:, 1], axis=-1)
    gt_right = np.linalg.norm(target[:, 3] - target[:, 1], axis=-1)
    ratio_error = np.abs(_safe_ratio(pred_left, pred_right) - _safe_ratio(gt_left, gt_right))

    metrics: dict[str, Any] = {
        "sample_count": int(len(predicted)),
        "corner_count": int(pixel_errors.size),
        "mean_nme": float(sample_nme.mean()),
        "median_nme": float(np.median(sample_nme)),
        "p95_nme": float(np.percentile(sample_nme, 95)),
        "mean_original_pixel_error": float(pixel_errors.mean()),
        "median_original_pixel_error": float(np.median(pixel_errors)),
        "p95_original_pixel_error": float(np.percentile(pixel_errors, 95)),
        "pck_0.02": float((normalized_corner_errors <= 0.02).mean()),
        "pck_0.05": float((normalized_corner_errors <= 0.05).mean()),
        "pck_0.10": float((normalized_corner_errors <= 0.10).mean()),
        "superior_endplate_angle_mae_deg": float(_angle_error(pred_superior, gt_superior).mean()),
        "inferior_endplate_angle_mae_deg": float(_angle_error(pred_inferior, gt_inferior).mean()),
        "left_height_mae_px": float(np.abs(pred_left - gt_left).mean()),
        "right_height_mae_px": float(np.abs(pred_right - gt_right).mean()),
        "height_ratio_mae": float(ratio_error.mean()),
        "per_corner_original_pixel_error": {
            name: float(pixel_errors[:, index].mean()) for index, name in enumerate(CORNER_ORDER)
        },
        "per_corner_nme": {
            name: float(normalized_corner_errors[:, index].mean())
            for index, name in enumerate(CORNER_ORDER)
        },
    }
    if boundary_peaks is not None:
        boundary = np.asarray(boundary_peaks, dtype=bool)
        metrics["crop_boundary_peak_rate"] = float(boundary.mean())
    return metrics


class CornerMetricAccumulator:
    def __init__(self) -> None:
        self.predicted: list[np.ndarray] = []
        self.target: list[np.ndarray] = []
        self.diagonals: list[np.ndarray] = []
        self.boundary: list[np.ndarray] = []
        self.sources: list[str] = []
        self.image_ids: list[int] = []
        self.annotation_ids: list[int] = []

    def update(
        self,
        predicted_original: torch.Tensor,
        target_original: torch.Tensor,
        bbox_diagonals: torch.Tensor,
        sources: list[str] | tuple[str, ...],
        boundary_peaks: torch.Tensor,
        image_ids: torch.Tensor | None = None,
        annotation_ids: torch.Tensor | None = None,
    ) -> None:
        self.predicted.append(predicted_original.detach().cpu().numpy())
        self.target.append(target_original.detach().cpu().numpy())
        self.diagonals.append(bbox_diagonals.detach().cpu().numpy())
        self.boundary.append(boundary_peaks.detach().cpu().numpy())
        self.sources.extend(str(source) for source in sources)
        if image_ids is not None and annotation_ids is not None:
            self.image_ids.extend(int(value) for value in image_ids.detach().cpu().tolist())
            self.annotation_ids.extend(int(value) for value in annotation_ids.detach().cpu().tolist())

    @staticmethod
    def _identity_metrics(
        predicted: np.ndarray,
        target: np.ndarray,
        sources: list[str],
        image_ids: list[int],
        annotation_ids: list[int],
        selected_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        indices = selected_indices if selected_indices is not None else list(range(len(predicted)))
        group_members: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index in range(len(predicted)):
            group_members[(sources[index], image_ids[index])].append(index)
        predicted_centers = predicted.mean(axis=1)
        target_centers = target.mean(axis=1)
        correct: list[bool] = []
        margins: list[float] = []
        for index in indices:
            candidates = group_members[(sources[index], image_ids[index])]
            if len(candidates) < 2:
                continue
            distances = np.linalg.norm(target_centers[candidates] - predicted_centers[index], axis=1)
            nearest_global = candidates[int(np.argmin(distances))]
            correct.append(annotation_ids[nearest_global] == annotation_ids[index])
            target_distance = float(np.linalg.norm(predicted_centers[index] - target_centers[index]))
            other_distances = [
                float(np.linalg.norm(predicted_centers[index] - target_centers[other]))
                for other in candidates
                if annotation_ids[other] != annotation_ids[index]
            ]
            if other_distances:
                margins.append(min(other_distances) - target_distance)
        accuracy = float(np.mean(correct)) if correct else 1.0
        return {
            "identity_evaluable_count": int(len(correct)),
            "identity_accuracy": accuracy,
            "identity_switch_rate": 1.0 - accuracy,
            "mean_identity_margin_px": float(np.mean(margins)) if margins else 0.0,
        }

    def compute(self) -> dict[str, Any]:
        if not self.predicted:
            raise RuntimeError("No metric samples accumulated")
        predicted = np.concatenate(self.predicted)
        target = np.concatenate(self.target)
        diagonals = np.concatenate(self.diagonals)
        boundary = np.concatenate(self.boundary)
        aggregate = compute_corner_metrics(predicted, target, diagonals, boundary_peaks=boundary)
        source_indices: dict[str, list[int]] = defaultdict(list)
        for index, source in enumerate(self.sources):
            source_indices[source].append(index)
        has_identity = len(self.image_ids) == len(predicted) and len(self.annotation_ids) == len(predicted)
        if has_identity:
            aggregate.update(
                self._identity_metrics(
                    predicted, target, self.sources, self.image_ids, self.annotation_ids
                )
            )
        per_source = {}
        for source, indices in sorted(source_indices.items()):
            source_metrics = compute_corner_metrics(
                predicted[indices], target[indices], diagonals[indices], boundary_peaks=boundary[indices]
            )
            if has_identity:
                source_metrics.update(
                    self._identity_metrics(
                        predicted,
                        target,
                        self.sources,
                        self.image_ids,
                        self.annotation_ids,
                        selected_indices=indices,
                    )
                )
            per_source[source] = source_metrics
        aggregate["per_source"] = per_source
        return aggregate
