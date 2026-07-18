from __future__ import annotations

import numpy as np
import torch


def crop_points_to_heatmap(
    points_crop: np.ndarray,
    crop_size: int = 256,
    heatmap_size: int = 64,
) -> np.ndarray:
    points = np.asarray(points_crop, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Expected [4, 2] crop points, got {points.shape}")
    scale = float(heatmap_size - 1) / float(crop_size - 1)
    return (points * scale).astype(np.float32)


def heatmap_points_to_crop(
    points_heatmap: np.ndarray,
    crop_size: int = 256,
    heatmap_size: int = 64,
) -> np.ndarray:
    points = np.asarray(points_heatmap, dtype=np.float32)
    scale = float(crop_size - 1) / float(heatmap_size - 1)
    return (points * scale).astype(np.float32)


def crop_points_to_normalized(points_crop: torch.Tensor, crop_size: int = 256) -> torch.Tensor:
    return points_crop * (2.0 / float(crop_size - 1)) - 1.0


def normalized_points_to_crop(points_normalized: torch.Tensor, crop_size: int = 256) -> torch.Tensor:
    return (points_normalized + 1.0) * (float(crop_size - 1) * 0.5)


def build_gaussian_heatmaps(
    points_crop: np.ndarray,
    crop_size: int = 256,
    heatmap_size: int = 64,
    sigma: float = 1.0,
) -> np.ndarray:
    """Create four normalized Gaussian probability maps."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    points_heatmap = crop_points_to_heatmap(points_crop, crop_size, heatmap_size)
    ys = np.arange(heatmap_size, dtype=np.float32)[:, None]
    xs = np.arange(heatmap_size, dtype=np.float32)[None, :]
    heatmaps = np.empty((4, heatmap_size, heatmap_size), dtype=np.float32)
    for index, (x_coord, y_coord) in enumerate(points_heatmap):
        heatmap = np.exp(
            -((xs - float(x_coord)) ** 2 + (ys - float(y_coord)) ** 2)
            / (2.0 * float(sigma) ** 2)
        ).astype(np.float32)
        heatmap /= max(float(heatmap.sum()), 1e-12)
        heatmaps[index] = heatmap
    return heatmaps

