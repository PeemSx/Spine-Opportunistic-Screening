from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def validate_four_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Expected four 2-D points, got shape {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Landmark points must be finite")
    return points


def make_square_roi(
    points: np.ndarray,
    crop_scale: float = 1.5,
    *,
    center_offset_fraction: np.ndarray | tuple[float, float] | None = None,
    scale_multiplier: float = 1.0,
) -> np.ndarray:
    """Build an unclipped square ROI in original-image coordinates."""
    points = validate_four_points(points)
    crop_scale = float(crop_scale)
    scale_multiplier = float(scale_multiplier)
    if crop_scale <= 0.0 or scale_multiplier <= 0.0:
        raise ValueError("crop_scale and scale_multiplier must be positive")

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = points.mean(axis=0).astype(np.float32)
    side = max(float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]), 1.0)
    side *= crop_scale * scale_multiplier

    if center_offset_fraction is not None:
        offset = np.asarray(center_offset_fraction, dtype=np.float32)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            raise ValueError("center_offset_fraction must contain two finite values")
        center = center + offset * side

    half = side * 0.5
    return np.asarray(
        [center[0] - half, center[1] - half, center[0] + half, center[1] + half],
        dtype=np.float32,
    )


def roi_contains_points(roi_xyxy: np.ndarray, points: np.ndarray) -> bool:
    roi = np.asarray(roi_xyxy, dtype=np.float32)
    points = validate_four_points(points)
    if roi.shape != (4,) or not np.isfinite(roi).all():
        return False
    x0, y0, x1, y1 = roi
    return bool(
        np.all(points[:, 0] >= x0)
        and np.all(points[:, 0] < x1)
        and np.all(points[:, 1] >= y0)
        and np.all(points[:, 1] < y1)
    )


def sample_contained_square_roi(
    points: np.ndarray,
    *,
    crop_scale: float = 1.5,
    rng: np.random.Generator | None = None,
    center_jitter_fraction: float = 0.08,
    scale_jitter_range: tuple[float, float] = (0.90, 1.10),
    max_retries: int = 10,
) -> tuple[np.ndarray, int, bool]:
    """Sample a jittered ROI, falling back to the deterministic ROI if needed.

    Returns ``(roi_xyxy, attempts, used_fallback)``.
    """
    points = validate_four_points(points)
    deterministic = make_square_roi(points, crop_scale=crop_scale)
    if rng is None:
        return deterministic, 0, False

    low, high = [float(value) for value in scale_jitter_range]
    if low <= 0.0 or high < low:
        raise ValueError("scale_jitter_range must be positive and ordered")
    jitter = float(center_jitter_fraction)
    if jitter < 0.0:
        raise ValueError("center_jitter_fraction cannot be negative")

    for attempt in range(1, int(max_retries) + 1):
        roi = make_square_roi(
            points,
            crop_scale=crop_scale,
            center_offset_fraction=rng.uniform(-jitter, jitter, size=2),
            scale_multiplier=float(rng.uniform(low, high)),
        )
        if roi_contains_points(roi, points):
            return roi, attempt, False
    return deterministic, int(max_retries), True


def affine_matrices_for_roi(roi_xyxy: np.ndarray, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    roi = np.asarray(roi_xyxy, dtype=np.float32)
    if roi.shape != (4,) or not np.isfinite(roi).all():
        raise ValueError("roi_xyxy must contain four finite values")
    output_size = int(output_size)
    if output_size <= 1:
        raise ValueError("output_size must be greater than one")

    x0, y0, x1, y1 = [float(value) for value in roi]
    side_x = x1 - x0
    side_y = y1 - y0
    if side_x <= 0.0 or side_y <= 0.0 or not np.isclose(side_x, side_y, rtol=1e-5, atol=1e-4):
        raise ValueError("ROI must be a non-empty isotropic square")

    scale = float(output_size) / side_x
    original_to_crop = np.asarray(
        [
            [scale, 0.0, -x0 * scale],
            [0.0, scale, -y0 * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    crop_to_original = np.linalg.inv(original_to_crop).astype(np.float32)
    return original_to_crop, crop_to_original


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must have shape [N, 2] with finite values")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a finite 3x3 transform")
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)],
        axis=1,
    )
    mapped = homogeneous @ matrix.T
    return (mapped[:, :2] / mapped[:, 2:3]).astype(np.float32)


def points_inside_crop(points: np.ndarray, output_size: int) -> bool:
    points = np.asarray(points, dtype=np.float32)
    output_size = int(output_size)
    return bool(
        np.isfinite(points).all()
        and np.all(points[:, 0] >= 0.0)
        and np.all(points[:, 0] < float(output_size))
        and np.all(points[:, 1] >= 0.0)
        and np.all(points[:, 1] < float(output_size))
    )


def crop_from_roi(
    image: np.ndarray,
    points: np.ndarray,
    roi_xyxy: np.ndarray,
    output_size: int = 256,
) -> dict[str, Any]:
    """Crop an ROI and return the exact bidirectional coordinate transforms."""
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {image.shape}")
    points = validate_four_points(points)
    roi = np.asarray(roi_xyxy, dtype=np.float32)
    original_to_crop, crop_to_original = affine_matrices_for_roi(roi, output_size)

    border_value: int | tuple[int, int, int]
    border_value = 0 if image.ndim == 2 else (0, 0, 0)
    crop = cv2.warpAffine(
        image,
        original_to_crop[:2],
        (int(output_size), int(output_size)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    crop_points = transform_points(points, original_to_crop)
    round_trip_points = transform_points(crop_points, crop_to_original)
    round_trip_error = float(np.linalg.norm(round_trip_points - points, axis=1).max())

    height, width = image.shape[:2]
    uses_padding = bool(
        roi[0] < 0.0
        or roi[1] < 0.0
        or roi[2] > float(width)
        or roi[3] > float(height)
    )
    return {
        "crop": crop,
        "crop_points": crop_points,
        "roi_xyxy": roi,
        "original_to_crop": original_to_crop,
        "crop_to_original": crop_to_original,
        "inside_crop": points_inside_crop(crop_points, output_size),
        "uses_padding": uses_padding,
        "round_trip_error": round_trip_error,
    }

