from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.evaluation.cobb_angle import CobbResult, calculate_cobb_angles
from src.evaluation.config import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_ACCEPTANCE_GATES,
    MATCH_THRESHOLDS,
    NUMERICAL_TOLERANCE,
    PCK_THRESHOLDS,
    PRIMARY_MATCH_THRESHOLD,
    USABLE_NME_THRESHOLD,
)


def threshold_label(value: float) -> str:
    return f"{float(value):.2f}"


def vertebral_diagonals(gt_corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(gt_corners, dtype=np.float32)
    if len(corners) == 0:
        return np.zeros((0,), dtype=np.float32)
    minimum = corners.min(axis=1)
    maximum = corners.max(axis=1)
    sizes = maximum - minimum
    return np.maximum(np.linalg.norm(sizes, axis=1), 1e-6).astype(np.float32)


@dataclass(frozen=True)
class CenterMatch:
    pred_index: int
    gt_index: int
    distance_px: float
    normalized_distance: float


def hungarian_center_match(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    gt_corners: np.ndarray,
    max_normalized_distance: float,
) -> list[CenterMatch]:
    pred_centers = np.asarray(pred_centers, dtype=np.float32)
    gt_centers = np.asarray(gt_centers, dtype=np.float32)
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return []

    diagonals = vertebral_diagonals(gt_corners)
    distances = np.linalg.norm(
        pred_centers[:, None, :] - gt_centers[None, :, :],
        axis=2,
    )
    normalized = distances / diagonals[None, :]
    boundary = float(max_normalized_distance) + NUMERICAL_TOLERANCE
    gated_cost = normalized.astype(np.float64)
    gated_cost[gated_cost > boundary] = 1e6
    pred_indices, gt_indices = linear_sum_assignment(gated_cost)

    matches = []
    for pred_index, gt_index in zip(pred_indices, gt_indices):
        normalized_distance = float(normalized[pred_index, gt_index])
        if normalized_distance > boundary:
            continue
        matches.append(
            CenterMatch(
                pred_index=int(pred_index),
                gt_index=int(gt_index),
                distance_px=float(distances[pred_index, gt_index]),
                normalized_distance=normalized_distance,
            )
        )
    return sorted(matches, key=lambda match: (match.gt_index, match.pred_index))


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    return float(
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _segments_cross(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    first_side = _orientation(first_start, first_end, second_start)
    second_side = _orientation(first_start, first_end, second_end)
    third_side = _orientation(second_start, second_end, first_start)
    fourth_side = _orientation(second_start, second_end, first_end)
    return first_side * second_side < 0.0 and third_side * fourth_side < 0.0


def valid_quadrilateral(corners: np.ndarray) -> bool:
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False
    polygon = points[[0, 1, 3, 2]]
    x_values = polygon[:, 0]
    y_values = polygon[:, 1]
    area = 0.5 * abs(
        float(
            np.dot(x_values, np.roll(y_values, -1))
            - np.dot(y_values, np.roll(x_values, -1))
        )
    )
    if area <= 1e-6:
        return False
    if _segments_cross(polygon[0], polygon[1], polygon[2], polygon[3]):
        return False
    if _segments_cross(polygon[1], polygon[2], polygon[3], polygon[0]):
        return False
    return True


def corners_inside_image(corners: np.ndarray, width: int, height: int) -> bool:
    points = np.asarray(corners, dtype=np.float32)
    return bool(
        points.shape == (4, 2)
        and np.isfinite(points).all()
        and (points[:, 0] >= 0.0).all()
        and (points[:, 0] < float(width)).all()
        and (points[:, 1] >= 0.0).all()
        and (points[:, 1] < float(height)).all()
    )


def _longest_contiguous_fraction(gt_indices: Iterable[int], gt_count: int) -> float:
    if gt_count <= 0:
        return 1.0
    ordered = sorted(set(int(index) for index in gt_indices))
    if not ordered:
        return 0.0
    longest = current = 1
    for previous, following in zip(ordered, ordered[1:]):
        if following == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return float(longest / gt_count)


def _cobb_pair(result: CobbResult) -> tuple[int, int] | None:
    if result.major_lines is None:
        return None
    return tuple(
        sorted(
            (
                int(result.major_lines[0].original_index),
                int(result.major_lines[1].original_index),
            )
        )
    )


def _angle_band(angle: float) -> str:
    if angle < 5.0:
        return "lt_5deg"
    if angle < 10.0:
        return "5_to_lt_10deg"
    if angle < 20.0:
        return "10_to_lt_20deg"
    return "ge_20deg"


@dataclass
class ImageEvaluation:
    model: str
    image: str
    image_id: int
    source_dataset: str
    cluster_id: str
    gt_count: int
    pred_count: int
    detection: dict[str, dict[str, int]]
    count_signed_error: int
    nmes: list[float] = field(default_factory=list)
    normalized_corner_errors: list[float] = field(default_factory=list)
    usable_count: int = 0
    invalid_geometry_count: int = 0
    out_of_image_count: int = 0
    full_coverage: bool = False
    clean_chain: bool = False
    longest_contiguous_fraction: float = 0.0
    gt_cobb_valid: bool = False
    pred_cobb_valid: bool = False
    gt_cobb_deg: float | None = None
    pred_cobb_deg: float | None = None
    cobb_signed_error_deg: float | None = None
    cobb_endpoint_exact: bool = False
    cobb_endpoint_within_one: bool = False
    instance_rows: list[dict[str, Any]] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        matched_corner_count = len(self.normalized_corner_errors)
        row: dict[str, Any] = {
            "model": self.model,
            "image": self.image,
            "image_id": self.image_id,
            "source_dataset": self.source_dataset,
            "patient_cluster_id": self.cluster_id,
            "gt_count": self.gt_count,
            "pred_count": self.pred_count,
            "count_signed_error": self.count_signed_error,
            "count_absolute_error": abs(self.count_signed_error),
            "usable_count": self.usable_count,
            "usable_vertebra_recall": self.usable_count / max(self.gt_count, 1),
            "corner_nme_mean": (
                float(np.mean(self.nmes))
                if self.nmes
                else None
            ),
            "invalid_geometry_count": self.invalid_geometry_count,
            "invalid_geometry_rate": self.invalid_geometry_count / max(self.pred_count, 1),
            "out_of_image_count": self.out_of_image_count,
            "out_of_image_prediction_rate": (
                self.out_of_image_count / max(self.pred_count, 1)
            ),
            "full_coverage": self.full_coverage,
            "clean_chain": self.clean_chain,
            "longest_contiguous_fraction": self.longest_contiguous_fraction,
            "gt_cobb_valid": self.gt_cobb_valid,
            "pred_cobb_valid": self.pred_cobb_valid,
            "gt_cobb_deg": self.gt_cobb_deg,
            "pred_cobb_deg": self.pred_cobb_deg,
            "cobb_signed_error_deg": self.cobb_signed_error_deg,
            "cobb_absolute_error_deg": (
                abs(self.cobb_signed_error_deg)
                if self.cobb_signed_error_deg is not None
                else None
            ),
            "cobb_at_5deg": bool(
                self.gt_cobb_valid
                and self.pred_cobb_valid
                and self.cobb_signed_error_deg is not None
                and abs(self.cobb_signed_error_deg) <= 5.0
            ),
            "cobb_at_10deg": bool(
                self.gt_cobb_valid
                and self.pred_cobb_valid
                and self.cobb_signed_error_deg is not None
                and abs(self.cobb_signed_error_deg) <= 10.0
            ),
            "cobb_endpoint_exact": self.cobb_endpoint_exact,
            "cobb_endpoint_within_one": self.cobb_endpoint_within_one,
        }
        for label, counts in self.detection.items():
            row[f"tp_{label}d"] = counts["tp"]
            row[f"fp_{label}d"] = counts["fp"]
            row[f"fn_{label}d"] = counts["fn"]
        for threshold in PCK_THRESHOLDS:
            label = threshold_label(threshold)
            correct = sum(
                value <= threshold + NUMERICAL_TOLERANCE
                for value in self.normalized_corner_errors
            )
            row[f"pck_{label}"] = _safe_rate(correct, matched_corner_count)
            row[f"end_to_end_pck_{label}"] = _safe_rate(
                correct,
                self.gt_count * 4,
            )
        return row


def evaluate_prediction(
    *,
    model: str,
    prediction: dict[str, np.ndarray],
    gt_centers: np.ndarray,
    gt_corners: np.ndarray,
    image: str,
    image_id: int,
    source_dataset: str,
    cluster_id: str,
    image_width: int,
    image_height: int,
) -> ImageEvaluation:
    pred_centers = np.asarray(prediction["centers"], dtype=np.float32)
    pred_corners = np.asarray(prediction["corners"], dtype=np.float32)
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    gt_centers = np.asarray(gt_centers, dtype=np.float32)
    gt_corners = np.asarray(gt_corners, dtype=np.float32)
    gt_count = len(gt_centers)
    pred_count = len(pred_centers)

    matches_by_threshold: dict[str, list[CenterMatch]] = {}
    detection: dict[str, dict[str, int]] = {}
    for threshold in MATCH_THRESHOLDS:
        label = threshold_label(threshold)
        matches = hungarian_center_match(
            pred_centers,
            gt_centers,
            gt_corners,
            max_normalized_distance=threshold,
        )
        matches_by_threshold[label] = matches
        detection[label] = {
            "tp": len(matches),
            "fp": pred_count - len(matches),
            "fn": gt_count - len(matches),
        }

    primary_label = threshold_label(PRIMARY_MATCH_THRESHOLD)
    primary_matches = matches_by_threshold[primary_label]
    match_pairs_by_threshold = {
        label: {
            (match.pred_index, match.gt_index)
            for match in matches
        }
        for label, matches in matches_by_threshold.items()
    }
    matched_pred_indices = {match.pred_index for match in primary_matches}
    matched_gt_indices = {match.gt_index for match in primary_matches}
    diagonals = vertebral_diagonals(gt_corners)
    nmes: list[float] = []
    normalized_corner_errors: list[float] = []
    instance_rows: list[dict[str, Any]] = []
    usable_count = 0

    pred_geometry_valid = [
        valid_quadrilateral(corners)
        for corners in pred_corners
    ]
    pred_inside_image = [
        corners_inside_image(corners, image_width, image_height)
        for corners in pred_corners
    ]

    for match in primary_matches:
        corner_errors = np.linalg.norm(
            pred_corners[match.pred_index] - gt_corners[match.gt_index],
            axis=1,
        )
        normalized_errors = corner_errors / float(diagonals[match.gt_index])
        nme = float(np.mean(normalized_errors))
        geometry_valid = pred_geometry_valid[match.pred_index]
        inside_image = pred_inside_image[match.pred_index]
        usable = (
            geometry_valid
            and inside_image
            and nme <= USABLE_NME_THRESHOLD + NUMERICAL_TOLERANCE
        )
        usable_count += int(usable)
        nmes.append(nme)
        normalized_corner_errors.extend(float(value) for value in normalized_errors)
        row: dict[str, Any] = {
            "model": model,
            "image": image,
            "image_id": image_id,
            "source_dataset": source_dataset,
            "status": "matched",
            "prediction_index": match.pred_index,
            "gt_index": match.gt_index,
            "score": float(scores[match.pred_index]),
            "center_error_px": match.distance_px,
            "normalized_center_error": match.normalized_distance,
            "gt_diagonal_px": float(diagonals[match.gt_index]),
            "corner_nme": nme,
            "valid_quadrilateral": geometry_valid,
            "inside_image": inside_image,
            "usable": usable,
        }
        for corner_index, name in enumerate(("tl", "tr", "bl", "br")):
            row[f"{name}_error_px"] = float(corner_errors[corner_index])
            row[f"{name}_normalized_error"] = float(normalized_errors[corner_index])
        for threshold in PCK_THRESHOLDS:
            label = threshold_label(threshold)
            correct = normalized_errors <= threshold + NUMERICAL_TOLERANCE
            row[f"pck_{label}"] = float(np.mean(correct))
            row[f"all_corners_pck_{label}"] = bool(correct.all())
        for threshold in MATCH_THRESHOLDS:
            label = threshold_label(threshold)
            row[f"center_matched_{label}d"] = (
                match.pred_index,
                match.gt_index,
            ) in match_pairs_by_threshold[label]
        instance_rows.append(row)

    for pred_index in sorted(set(range(pred_count)) - matched_pred_indices):
        row = {
            "model": model,
            "image": image,
            "image_id": image_id,
            "source_dataset": source_dataset,
            "status": "false_positive",
            "prediction_index": pred_index,
            "gt_index": None,
            "score": float(scores[pred_index]),
            "valid_quadrilateral": pred_geometry_valid[pred_index],
            "inside_image": pred_inside_image[pred_index],
            "usable": False,
        }
        for threshold in MATCH_THRESHOLDS:
            label = threshold_label(threshold)
            row[f"center_matched_{label}d"] = any(
                match.pred_index == pred_index
                for match in matches_by_threshold[label]
            )
        for threshold in PCK_THRESHOLDS:
            row[f"pck_{threshold_label(threshold)}"] = None
        instance_rows.append(row)
    for gt_index in sorted(set(range(gt_count)) - matched_gt_indices):
        row = {
            "model": model,
            "image": image,
            "image_id": image_id,
            "source_dataset": source_dataset,
            "status": "missed_ground_truth",
            "prediction_index": None,
            "gt_index": gt_index,
            "score": None,
            "gt_diagonal_px": float(diagonals[gt_index]),
            "usable": False,
        }
        for threshold in MATCH_THRESHOLDS:
            label = threshold_label(threshold)
            row[f"center_matched_{label}d"] = any(
                match.gt_index == gt_index
                for match in matches_by_threshold[label]
            )
        for threshold in PCK_THRESHOLDS:
            row[f"pck_{threshold_label(threshold)}"] = 0.0
        instance_rows.append(row)

    gt_cobb = calculate_cobb_angles(gt_corners)
    pred_cobb = calculate_cobb_angles(pred_corners)
    gt_cobb_deg = gt_cobb.cobb_1_deg if gt_cobb.valid else None
    pred_cobb_deg = pred_cobb.cobb_1_deg if pred_cobb.valid else None
    cobb_signed_error = (
        float(pred_cobb_deg - gt_cobb_deg)
        if gt_cobb_deg is not None and pred_cobb_deg is not None
        else None
    )

    endpoint_exact = False
    endpoint_within_one = False
    gt_pair = _cobb_pair(gt_cobb)
    pred_pair = _cobb_pair(pred_cobb)
    if gt_pair is not None and pred_pair is not None:
        pred_to_gt = {
            match.pred_index: match.gt_index
            for match in primary_matches
        }
        mapped_pair = (
            tuple(sorted(pred_to_gt[index] for index in pred_pair))
            if all(index in pred_to_gt for index in pred_pair)
            else None
        )
        if mapped_pair is not None:
            endpoint_exact = mapped_pair == gt_pair
            endpoint_within_one = all(
                abs(predicted - target) <= 1
                for predicted, target in zip(mapped_pair, gt_pair)
            )

    primary_counts = detection[primary_label]
    return ImageEvaluation(
        model=model,
        image=image,
        image_id=image_id,
        source_dataset=source_dataset,
        cluster_id=cluster_id,
        gt_count=gt_count,
        pred_count=pred_count,
        detection=detection,
        count_signed_error=pred_count - gt_count,
        nmes=nmes,
        normalized_corner_errors=normalized_corner_errors,
        usable_count=usable_count,
        invalid_geometry_count=sum(not value for value in pred_geometry_valid),
        out_of_image_count=sum(not value for value in pred_inside_image),
        full_coverage=primary_counts["fn"] == 0,
        clean_chain=primary_counts["fn"] == 0 and primary_counts["fp"] == 0,
        longest_contiguous_fraction=_longest_contiguous_fraction(
            matched_gt_indices,
            gt_count,
        ),
        gt_cobb_valid=gt_cobb.valid,
        pred_cobb_valid=pred_cobb.valid,
        gt_cobb_deg=float(gt_cobb_deg) if gt_cobb_deg is not None else None,
        pred_cobb_deg=float(pred_cobb_deg) if pred_cobb_deg is not None else None,
        cobb_signed_error_deg=cobb_signed_error,
        cobb_endpoint_exact=endpoint_exact,
        cobb_endpoint_within_one=endpoint_within_one,
        instance_rows=instance_rows,
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def aggregate_evaluations(
    evaluations: list[ImageEvaluation],
    *,
    model: str,
    scope: str,
    source_dataset: str | None = None,
) -> dict[str, Any]:
    gt_total = sum(item.gt_count for item in evaluations)
    pred_total = sum(item.pred_count for item in evaluations)
    image_count = len(evaluations)
    row: dict[str, Any] = {
        "model": model,
        "scope": scope,
        "source_dataset": source_dataset,
        "images": image_count,
        "gt_vertebrae": gt_total,
        "predictions": pred_total,
    }

    for threshold in MATCH_THRESHOLDS:
        label = threshold_label(threshold)
        tp = sum(item.detection[label]["tp"] for item in evaluations)
        fp = sum(item.detection[label]["fp"] for item in evaluations)
        fn = sum(item.detection[label]["fn"] for item in evaluations)
        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)
        f1 = _safe_rate(2.0 * precision * recall, precision + recall)
        row[f"center_tp_{label}d"] = tp
        row[f"center_fp_{label}d"] = fp
        row[f"center_fn_{label}d"] = fn
        row[f"center_precision_{label}d"] = precision
        row[f"center_recall_{label}d"] = recall
        row[f"center_f1_{label}d"] = f1

    primary_label = threshold_label(PRIMARY_MATCH_THRESHOLD)
    row["false_positives_per_image"] = _safe_rate(
        row[f"center_fp_{primary_label}d"],
        image_count,
    )
    signed_count_errors = [item.count_signed_error for item in evaluations]
    absolute_count_errors = [abs(value) for value in signed_count_errors]
    row["count_mae"] = float(np.mean(absolute_count_errors)) if evaluations else 0.0
    row["count_bias"] = float(np.mean(signed_count_errors)) if evaluations else 0.0
    row["count_exact_rate"] = _safe_rate(
        sum(value == 0 for value in signed_count_errors),
        image_count,
    )
    row["count_within_one_rate"] = _safe_rate(
        sum(abs(value) <= 1 for value in signed_count_errors),
        image_count,
    )
    row["count_p90_absolute_error"] = _percentile(absolute_count_errors, 90.0)

    nmes = [value for item in evaluations for value in item.nmes]
    normalized_corner_errors = [
        value
        for item in evaluations
        for value in item.normalized_corner_errors
    ]
    row["corner_nme_mean"] = float(np.mean(nmes)) if nmes else None
    row["corner_nme_median"] = _percentile(nmes, 50.0)
    row["corner_nme_p95"] = _percentile(nmes, 95.0)
    matched_corner_count = len(normalized_corner_errors)
    for threshold in PCK_THRESHOLDS:
        label = threshold_label(threshold)
        correct = sum(
            value <= threshold + NUMERICAL_TOLERANCE
            for value in normalized_corner_errors
        )
        row[f"pck_{label}"] = _safe_rate(correct, matched_corner_count)
        row[f"end_to_end_pck_{label}"] = _safe_rate(correct, gt_total * 4)

    row["usable_vertebra_recall"] = _safe_rate(
        sum(item.usable_count for item in evaluations),
        gt_total,
    )
    row["invalid_geometry_rate"] = _safe_rate(
        sum(item.invalid_geometry_count for item in evaluations),
        pred_total,
    )
    row["out_of_image_prediction_rate"] = _safe_rate(
        sum(item.out_of_image_count for item in evaluations),
        pred_total,
    )
    row["full_coverage_rate"] = _safe_rate(
        sum(item.full_coverage for item in evaluations),
        image_count,
    )
    row["clean_chain_rate"] = _safe_rate(
        sum(item.clean_chain for item in evaluations),
        image_count,
    )
    row["longest_contiguous_fraction"] = (
        float(np.mean([item.longest_contiguous_fraction for item in evaluations]))
        if evaluations
        else 0.0
    )

    eligible = [item for item in evaluations if item.gt_cobb_valid]
    valid = [item for item in eligible if item.pred_cobb_valid]
    cobb_errors = [
        float(item.cobb_signed_error_deg)
        for item in valid
        if item.cobb_signed_error_deg is not None
    ]
    cobb_absolute_errors = [abs(value) for value in cobb_errors]
    row["cobb_eligible_images"] = len(eligible)
    row["cobb_valid_predictions"] = len(valid)
    row["cobb_coverage"] = _safe_rate(len(valid), len(eligible))
    row["cobb_mae_deg"] = (
        float(np.mean(cobb_absolute_errors))
        if cobb_absolute_errors
        else None
    )
    row["cobb_median_absolute_error_deg"] = _percentile(
        cobb_absolute_errors,
        50.0,
    )
    row["cobb_p95_absolute_error_deg"] = _percentile(
        cobb_absolute_errors,
        95.0,
    )
    row["cobb_rmse_deg"] = (
        float(np.sqrt(np.mean(np.square(cobb_errors))))
        if cobb_errors
        else None
    )
    row["cobb_bias_deg"] = float(np.mean(cobb_errors)) if cobb_errors else None
    row["cobb_at_5deg"] = _safe_rate(
        sum(abs(value) <= 5.0 for value in cobb_errors),
        len(eligible),
    )
    row["cobb_at_10deg"] = _safe_rate(
        sum(abs(value) <= 10.0 for value in cobb_errors),
        len(eligible),
    )
    zero_errors = [
        abs(float(item.gt_cobb_deg))
        for item in eligible
        if item.gt_cobb_deg is not None
    ]
    row["zero_cobb_baseline_mae_deg"] = (
        float(np.mean(zero_errors))
        if zero_errors
        else None
    )
    row["zero_cobb_baseline_at_5deg"] = _safe_rate(
        sum(value <= 5.0 for value in zero_errors),
        len(eligible),
    )
    row["zero_cobb_baseline_at_10deg"] = _safe_rate(
        sum(value <= 10.0 for value in zero_errors),
        len(eligible),
    )
    row["cobb_endpoint_exact_rate"] = _safe_rate(
        sum(item.cobb_endpoint_exact for item in eligible),
        len(eligible),
    )
    row["cobb_endpoint_within_one_rate"] = _safe_rate(
        sum(item.cobb_endpoint_within_one for item in eligible),
        len(eligible),
    )

    for band in ("lt_5deg", "5_to_lt_10deg", "10_to_lt_20deg", "ge_20deg"):
        band_items = [
            item
            for item in eligible
            if item.gt_cobb_deg is not None and _angle_band(item.gt_cobb_deg) == band
        ]
        band_errors = [
            abs(float(item.cobb_signed_error_deg))
            for item in band_items
            if item.pred_cobb_valid and item.cobb_signed_error_deg is not None
        ]
        row[f"cobb_{band}_images"] = len(band_items)
        row[f"cobb_{band}_valid_predictions"] = len(band_errors)
        row[f"cobb_{band}_coverage"] = _safe_rate(
            len(band_errors),
            len(band_items),
        )
        row[f"cobb_{band}_mae_deg"] = (
            float(np.mean(band_errors))
            if band_errors
            else None
        )
        row[f"cobb_{band}_at_5deg"] = _safe_rate(
            sum(value <= 5.0 for value in band_errors),
            len(band_items),
        )
        row[f"cobb_{band}_at_10deg"] = _safe_rate(
            sum(value <= 10.0 for value in band_errors),
            len(band_items),
        )
    return row


def source_summary_rows(
    evaluations: list[ImageEvaluation],
    model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    overall = aggregate_evaluations(evaluations, model=model, scope="overall_micro")
    sources = sorted({item.source_dataset for item in evaluations})
    per_source = [
        aggregate_evaluations(
            [item for item in evaluations if item.source_dataset == source],
            model=model,
            scope="source",
            source_dataset=source,
        )
        for source in sources
    ]
    numeric_keys = [
        key
        for key, value in overall.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    macro: dict[str, Any] = {
        "model": model,
        "scope": "source_macro",
        "source_dataset": None,
        "sources": len(per_source),
    }
    worst: dict[str, Any] = {
        "model": model,
        "scope": "worst_source",
        "source_dataset": None,
        "sources": len(per_source),
    }
    lower_is_better_tokens = (
        "mae",
        "rmse",
        "p95",
        "false_positives",
        "invalid_",
        "out_of_image",
        "error",
    )
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in per_source
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ]
        if not values:
            macro[key] = None
            worst[key] = None
            continue
        macro[key] = float(np.mean(values))
        worst[key] = (
            max(values)
            if any(token in key for token in lower_is_better_tokens)
            else min(values)
        )
    return overall, per_source, macro, worst


BOOTSTRAP_METRICS = (
    "center_precision_0.20d",
    "center_recall_0.20d",
    "center_f1_0.20d",
    "false_positives_per_image",
    "count_mae",
    "count_bias",
    "corner_nme_mean",
    "pck_0.10",
    "end_to_end_pck_0.10",
    "usable_vertebra_recall",
    "invalid_geometry_rate",
    "cobb_coverage",
    "cobb_mae_deg",
    "cobb_at_5deg",
    "cobb_at_10deg",
)


def bootstrap_confidence_intervals(
    evaluations: list[ImageEvaluation],
    *,
    model: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    if not evaluations or samples <= 0:
        return {}
    by_cluster: dict[str, list[ImageEvaluation]] = {}
    for evaluation in evaluations:
        by_cluster.setdefault(evaluation.cluster_id, []).append(evaluation)
    cluster_ids = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(int(samples)):
        sampled_ids = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled = [
            evaluation
            for cluster_id in sampled_ids
            for evaluation in by_cluster[str(cluster_id)]
        ]
        aggregate = aggregate_evaluations(
            sampled,
            model=model,
            scope="bootstrap",
        )
        for metric in BOOTSTRAP_METRICS:
            value = aggregate.get(metric)
            if value is not None and np.isfinite(float(value)):
                values[metric].append(float(value))
    return {
        metric: {
            "lower_95": float(np.percentile(metric_values, 2.5)),
            "upper_95": float(np.percentile(metric_values, 97.5)),
        }
        for metric, metric_values in values.items()
        if metric_values
    }


def postprocessing_comparison(
    raw: dict[str, Any],
    chain: dict[str, Any],
) -> dict[str, Any]:
    primary_label = threshold_label(PRIMARY_MATCH_THRESHOLD)
    raw_fp = int(raw[f"center_fp_{primary_label}d"])
    chain_fp = int(chain[f"center_fp_{primary_label}d"])
    fp_reduction = (
        float((raw_fp - chain_fp) / raw_fp)
        if raw_fp > 0
        else (0.0 if chain_fp == 0 else -float(chain_fp))
    )
    return {
        "f1_change_0.20d": float(
            chain["center_f1_0.20d"] - raw["center_f1_0.20d"]
        ),
        "recall_change_0.20d": float(
            chain["center_recall_0.20d"] - raw["center_recall_0.20d"]
        ),
        "recall_loss_0.20d": float(
            raw["center_recall_0.20d"] - chain["center_recall_0.20d"]
        ),
        "false_positive_reduction": fp_reduction,
        "count_mae_change": float(chain["count_mae"] - raw["count_mae"]),
        "usable_vertebra_recall_change": float(
            chain["usable_vertebra_recall"] - raw["usable_vertebra_recall"]
        ),
        "cobb_at_5deg_change": float(
            chain["cobb_at_5deg"] - raw["cobb_at_5deg"]
        ),
    }


def evaluate_acceptance_gates(
    *,
    chain_overall: dict[str, Any],
    chain_source_macro: dict[str, Any],
    chain_worst_source: dict[str, Any],
    comparison: dict[str, Any],
    gates: dict[str, float] = DEFAULT_ACCEPTANCE_GATES,
) -> dict[str, Any]:
    def absolute_or_none(value: Any) -> float | None:
        if value is None:
            return None
        numeric = float(value)
        return abs(numeric) if np.isfinite(numeric) else None

    checks = {
        "source_macro_f1_0.20d": (
            chain_source_macro.get("center_f1_0.20d"),
            ">=",
            gates["source_macro_f1_0.20d_min"],
        ),
        "worst_source_recall_0.20d": (
            chain_worst_source.get("center_recall_0.20d"),
            ">=",
            gates["worst_source_recall_0.20d_min"],
        ),
        "mean_nme": (
            chain_overall.get("corner_nme_mean"),
            "<=",
            gates["mean_nme_max"],
        ),
        "pck_0.10": (
            chain_overall.get("pck_0.10"),
            ">=",
            gates["pck_0.10_min"],
        ),
        "usable_vertebra_recall": (
            chain_overall.get("usable_vertebra_recall"),
            ">=",
            gates["usable_vertebra_recall_min"],
        ),
        "count_mae": (
            chain_overall.get("count_mae"),
            "<=",
            gates["count_mae_max"],
        ),
        "absolute_count_bias": (
            absolute_or_none(chain_overall.get("count_bias")),
            "<=",
            gates["absolute_count_bias_max"],
        ),
        "chain_false_positive_reduction": (
            comparison.get("false_positive_reduction"),
            ">=",
            gates["chain_fp_reduction_min"],
        ),
        "chain_recall_loss": (
            comparison.get("recall_loss_0.20d"),
            "<=",
            gates["chain_recall_loss_max"],
        ),
        "cobb_coverage": (
            chain_overall.get("cobb_coverage"),
            ">=",
            gates["cobb_coverage_min"],
        ),
        "cobb_mae_deg": (
            chain_overall.get("cobb_mae_deg"),
            "<=",
            gates["cobb_mae_deg_max"],
        ),
        "cobb_at_5deg": (
            chain_overall.get("cobb_at_5deg"),
            ">=",
            gates["cobb_at_5deg_min"],
        ),
        "cobb_at_10deg": (
            chain_overall.get("cobb_at_10deg"),
            ">=",
            gates["cobb_at_10deg_min"],
        ),
    }
    results: dict[str, Any] = {}
    for name, (observed, operator, threshold) in checks.items():
        passed = bool(
            observed is not None
            and np.isfinite(float(observed))
            and (
                float(observed) >= float(threshold)
                if operator == ">="
                else float(observed) <= float(threshold)
            )
        )
        results[name] = {
            "observed": observed,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
        }
    return {
        "passed": all(result["passed"] for result in results.values()),
        "checks": results,
    }
