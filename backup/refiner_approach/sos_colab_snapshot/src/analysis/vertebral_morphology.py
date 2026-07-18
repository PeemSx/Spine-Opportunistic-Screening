from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from src.postprocessing.spine_chain import SpineCandidate


MORPHOLOGY_DISCLAIMER = (
    "Research screening-support measurements only. These features are not a diagnosis "
    "and require clinical interpretation."
)

MORPHOLOGY_FIELDNAMES = [
    "image",
    "rank",
    "candidate_index",
    "landmark_source",
    "fallback_reason",
    "detector_score",
    "center_x_px",
    "center_y_px",
    "superior_width_px",
    "inferior_width_px",
    "left_height_px",
    "right_height_px",
    "mean_height_px",
    "left_right_height_ratio",
    "height_asymmetry_fraction",
    "width_height_ratio",
    "superior_endplate_angle_deg",
    "inferior_endplate_angle_deg",
    "endplate_nonparallel_deg",
    "neighbor_reference_height_px",
    "height_ratio_to_neighbors",
    "relative_height_deviation",
    "previous_center_spacing_px",
    "next_center_spacing_px",
]


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(second) - np.asarray(first)))


def _line_angle(first: np.ndarray, second: np.ndarray) -> float:
    vector = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def _angle_difference(first: float, second: float) -> float:
    return float(abs((first - second + 90.0) % 180.0 - 90.0))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(float(denominator), 1e-6))


def _landmark_provenance(
    candidate_index: int,
    diagnostics_by_index: dict[int, dict[str, Any]],
) -> tuple[str, str]:
    diagnostic = diagnostics_by_index.get(int(candidate_index))
    if diagnostic is None:
        return "centernet", ""
    if bool(diagnostic.get("accepted")):
        return "refiner", ""
    return "centernet_fallback", str(diagnostic.get("reason", "refiner_rejected"))


def extract_chain_morphology(
    image: str,
    candidates: Iterable[SpineCandidate],
    refiner_diagnostics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract original-pixel and dimensionless features from an ordered spine chain."""
    ordered = sorted(list(candidates), key=lambda candidate: (candidate.y, candidate.x))
    diagnostics_by_index = {
        int(item["candidate_index"]): item for item in (refiner_diagnostics or [])
    }
    base_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ordered, start=1):
        points = np.asarray(candidate.corners, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            continue
        tl, tr, bl, br = points
        superior_width = _distance(tl, tr)
        inferior_width = _distance(bl, br)
        left_height = _distance(tl, bl)
        right_height = _distance(tr, br)
        mean_height = 0.5 * (left_height + right_height)
        mean_width = 0.5 * (superior_width + inferior_width)
        superior_angle = _line_angle(tl, tr)
        inferior_angle = _line_angle(bl, br)
        landmark_source, fallback_reason = _landmark_provenance(
            candidate.index, diagnostics_by_index
        )
        base_rows.append(
            {
                "image": str(image),
                "rank": int(rank),
                "candidate_index": int(candidate.index),
                "landmark_source": landmark_source,
                "fallback_reason": fallback_reason,
                "detector_score": float(candidate.score),
                "center_x_px": float(candidate.center[0]),
                "center_y_px": float(candidate.center[1]),
                "superior_width_px": superior_width,
                "inferior_width_px": inferior_width,
                "left_height_px": left_height,
                "right_height_px": right_height,
                "mean_height_px": mean_height,
                "left_right_height_ratio": _safe_ratio(left_height, right_height),
                "height_asymmetry_fraction": _safe_ratio(
                    abs(left_height - right_height), mean_height
                ),
                "width_height_ratio": _safe_ratio(mean_width, mean_height),
                "superior_endplate_angle_deg": superior_angle,
                "inferior_endplate_angle_deg": inferior_angle,
                "endplate_nonparallel_deg": _angle_difference(
                    superior_angle, inferior_angle
                ),
            }
        )

    for index, row in enumerate(base_rows):
        neighbor_heights = []
        if index > 0:
            neighbor_heights.append(float(base_rows[index - 1]["mean_height_px"]))
        if index + 1 < len(base_rows):
            neighbor_heights.append(float(base_rows[index + 1]["mean_height_px"]))
        reference_height = float(np.mean(neighbor_heights)) if neighbor_heights else None
        current_height = float(row["mean_height_px"])
        row["neighbor_reference_height_px"] = reference_height
        row["height_ratio_to_neighbors"] = (
            _safe_ratio(current_height, reference_height) if reference_height is not None else None
        )
        row["relative_height_deviation"] = (
            _safe_ratio(abs(current_height - reference_height), reference_height)
            if reference_height is not None
            else None
        )
        center = np.asarray([row["center_x_px"], row["center_y_px"]], dtype=np.float32)
        if index > 0:
            previous = np.asarray(
                [base_rows[index - 1]["center_x_px"], base_rows[index - 1]["center_y_px"]],
                dtype=np.float32,
            )
            row["previous_center_spacing_px"] = _distance(previous, center)
        else:
            row["previous_center_spacing_px"] = None
        if index + 1 < len(base_rows):
            following = np.asarray(
                [base_rows[index + 1]["center_x_px"], base_rows[index + 1]["center_y_px"]],
                dtype=np.float32,
            )
            row["next_center_spacing_px"] = _distance(center, following)
        else:
            row["next_center_spacing_px"] = None
    return base_rows


def morphology_json_payload(
    image: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "image": str(image),
        "vertebra_count": int(len(rows)),
        "coordinate_units": "original_image_pixels",
        "disclaimer": MORPHOLOGY_DISCLAIMER,
        "features": rows,
    }
