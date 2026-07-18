from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.data.corner_refiner_dataset import IMAGENET_MEAN, IMAGENET_STD
from src.data.roi_geometry import affine_matrices_for_roi, transform_points
from src.evaluation.corner_refiner_decode import decode_corner_heatmaps
from src.models.corner_refiner import CornerRefinerHRNetW18


@dataclass(frozen=True)
class RefinerRuntimeConfig:
    crop_size: int = 256
    crop_scale: float = 1.5
    batch_size: int = 16
    min_base_side: float = 8.0
    median_side_min_factor: float = 0.50
    median_side_max_factor: float = 2.00
    max_center_offset_fraction: float = 0.20
    reject_boundary_peaks: bool = True
    require_nearest_anchor: bool = True


def load_corner_refiner(
    checkpoint_path: Path | str,
    device: torch.device,
) -> tuple[CornerRefinerHRNetW18, dict[str, Any], RefinerRuntimeConfig]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model_args = checkpoint.get("model_args", checkpoint.get("metadata", {}).get("model_arguments", {}))
    model = CornerRefinerHRNetW18(
        pretrained=False,
        projection_channels=int(model_args.get("projection_channels", 32)),
        head_channels=int(model_args.get("head_channels", 128)),
        output_size=int(model_args.get("output_size", 64)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    train_args = checkpoint.get("args", {})
    crop_policy = checkpoint.get("metadata", {}).get("crop_policy", {})
    runtime = RefinerRuntimeConfig(
        crop_size=int(train_args.get("input_size", crop_policy.get("output_size", 256))),
        crop_scale=float(train_args.get("crop_scale", crop_policy.get("base_scale", 1.5))),
    )
    return model, checkpoint, runtime


def _valid_preliminary_side(corners: np.ndarray, minimum: float) -> float | None:
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    extent = points.max(axis=0) - points.min(axis=0)
    side = max(float(extent[0]), float(extent[1]))
    return side if side >= float(minimum) else None


def estimate_detector_roi_sides(
    centers: np.ndarray,
    corners: np.ndarray,
    config: RefinerRuntimeConfig,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    corners = np.asarray(corners, dtype=np.float32)
    raw_sides = np.asarray(
        [
            value if (value := _valid_preliminary_side(points, config.min_base_side)) is not None else np.nan
            for points in corners
        ],
        dtype=np.float32,
    )
    valid = raw_sides[np.isfinite(raw_sides)]
    if len(valid):
        median_side = float(np.median(valid))
    elif len(centers) >= 2:
        y_gaps = np.diff(np.sort(centers[:, 1]))
        positive = y_gaps[y_gaps > 1.0]
        median_side = float(np.median(positive)) if len(positive) else 32.0
    else:
        median_side = 32.0
    median_side = max(median_side, float(config.min_base_side))
    base_sides = np.where(np.isfinite(raw_sides), raw_sides, median_side)
    base_sides = np.clip(
        base_sides,
        config.median_side_min_factor * median_side,
        config.median_side_max_factor * median_side,
    )
    return (base_sides * float(config.crop_scale)).astype(np.float32)


def detector_rois(
    centers: np.ndarray,
    corners: np.ndarray,
    config: RefinerRuntimeConfig,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape [N, 2]")
    sides = estimate_detector_roi_sides(centers, corners, config)
    half = sides[:, None] * 0.5
    return np.concatenate((centers - half, centers + half), axis=1).astype(np.float32)


def _crop_tensor(
    image_rgb: np.ndarray,
    roi: np.ndarray,
    crop_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    original_to_crop, crop_to_original = affine_matrices_for_roi(roi, crop_size)
    crop = cv2.warpAffine(
        image_rgb,
        original_to_crop[:2],
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    normalized = np.repeat(gray[None], 3, axis=0)
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)), crop_to_original


def _valid_corner_geometry(points: np.ndarray, roi_side: float) -> bool:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False
    if not (
        points[0, 0] < points[1, 0]
        and points[2, 0] < points[3, 0]
        and points[:2, 1].mean() < points[2:, 1].mean()
    ):
        return False
    polygon = points[[0, 1, 3, 2]]
    area = 0.5 * abs(
        float(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)))
        - float(np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))
    )
    return area >= 0.005 * float(roi_side) ** 2


@torch.no_grad()
def refine_centernet_prediction(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    prediction: dict[str, np.ndarray],
    device: torch.device,
    config: RefinerRuntimeConfig,
    *,
    use_amp: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Refine all detector candidates and retain CenterNet corners on rejection."""
    scores = np.asarray(prediction["scores"], dtype=np.float32)
    detector_centers = np.asarray(prediction["centers"], dtype=np.float32)
    fallback_corners = np.asarray(prediction["corners"], dtype=np.float32)
    count = len(detector_centers)
    if count == 0:
        return {
            "scores": scores.copy(),
            "centers": detector_centers.copy(),
            "corners": fallback_corners.copy(),
        }, []
    if fallback_corners.shape != (count, 4, 2):
        raise ValueError("prediction corners must have shape [N, 4, 2]")

    rois = detector_rois(detector_centers, fallback_corners, config)
    crop_tensors: list[torch.Tensor] = []
    crop_to_original: list[np.ndarray] = []
    for roi in rois:
        tensor, inverse = _crop_tensor(image_rgb, roi, config.crop_size)
        crop_tensors.append(tensor)
        crop_to_original.append(inverse)

    refined_original = np.empty((count, 4, 2), dtype=np.float32)
    entropies = np.empty((count, 4), dtype=np.float32)
    peaks = np.empty((count, 4), dtype=np.float32)
    boundaries = np.empty((count, 4), dtype=bool)
    amp_enabled = bool(use_amp and device.type == "cuda")
    for start in range(0, count, max(int(config.batch_size), 1)):
        stop = min(start + max(int(config.batch_size), 1), count)
        inputs = torch.stack(crop_tensors[start:stop]).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(inputs)["heatmap_logits"]
        decoded = decode_corner_heatmaps(logits.float(), crop_size=config.crop_size)
        crop_points = decoded["points_crop"].cpu().numpy()
        for local_index, points in enumerate(crop_points):
            index = start + local_index
            refined_original[index] = transform_points(points, crop_to_original[index])
        entropies[start:stop] = decoded["entropy"].cpu().numpy()
        peaks[start:stop] = decoded["peak_values"].cpu().numpy()
        boundaries[start:stop] = decoded["boundary_peaks"].cpu().numpy()

    refined_centers = refined_original.mean(axis=1)
    roi_sides = rois[:, 2] - rois[:, 0]
    offsets = np.linalg.norm(refined_centers - detector_centers, axis=1) / np.maximum(roi_sides, 1e-6)
    distances = np.linalg.norm(
        refined_centers[:, None, :] - detector_centers[None, :, :],
        axis=-1,
    )
    nearest_anchor = distances.argmin(axis=1)

    output_corners = fallback_corners.copy()
    output_centers = detector_centers.copy()
    diagnostics: list[dict[str, Any]] = []
    for index in range(count):
        reasons: list[str] = []
        if offsets[index] > config.max_center_offset_fraction:
            reasons.append("center_offset")
        if config.require_nearest_anchor and int(nearest_anchor[index]) != index:
            reasons.append("identity_switch")
        if not _valid_corner_geometry(refined_original[index], float(roi_sides[index])):
            reasons.append("corner_geometry")
        if config.reject_boundary_peaks and bool(boundaries[index].any()):
            reasons.append("boundary_peak")
        accepted = not reasons
        if accepted:
            output_corners[index] = refined_original[index]
            output_centers[index] = refined_centers[index]
        diagnostics.append(
            {
                "candidate_index": int(index),
                "accepted": bool(accepted),
                "reason": "accepted" if accepted else "+".join(reasons),
                "roi_original_xyxy": rois[index].astype(float).tolist(),
                "identity_offset_fraction": float(offsets[index]),
                "nearest_detector_index": int(nearest_anchor[index]),
                "mean_entropy": float(entropies[index].mean()),
                "mean_peak": float(peaks[index].mean()),
                "boundary_peak_rate": float(boundaries[index].mean()),
            }
        )

    return {
        "scores": scores.copy(),
        "centers": output_centers,
        "corners": output_corners,
    }, diagnostics
