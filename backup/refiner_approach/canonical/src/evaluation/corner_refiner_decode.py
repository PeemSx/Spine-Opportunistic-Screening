from __future__ import annotations

import torch

from src.data.corner_refiner_targets import normalized_points_to_crop


def spatial_softmax_2d(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"Expected [B, K, H, W] logits, got {tuple(logits.shape)}")
    return torch.softmax(logits.flatten(start_dim=-2), dim=-1).reshape_as(logits)


def decode_corner_heatmaps(
    logits: torch.Tensor,
    *,
    crop_size: int = 256,
    boundary_cells: int = 1,
) -> dict[str, torch.Tensor]:
    """Decode heatmap logits with DSNT and expose distribution diagnostics."""
    probabilities = spatial_softmax_2d(logits)
    batch, corners, height, width = probabilities.shape
    dtype, device = probabilities.dtype, probabilities.device
    x_axis = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
    y_axis = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")

    expected_x = (probabilities * grid_x).sum(dim=(-2, -1))
    expected_y = (probabilities * grid_y).sum(dim=(-2, -1))
    points_normalized = torch.stack((expected_x, expected_y), dim=-1)
    points_crop = normalized_points_to_crop(points_normalized, crop_size=crop_size)
    points_heatmap = torch.stack(
        (
            (expected_x + 1.0) * (width - 1) * 0.5,
            (expected_y + 1.0) * (height - 1) * 0.5,
        ),
        dim=-1,
    )

    dx = grid_x[None, None] - expected_x[:, :, None, None]
    dy = grid_y[None, None] - expected_y[:, :, None, None]
    covariance = torch.stack(
        (
            torch.stack(((probabilities * dx * dx).sum((-2, -1)), (probabilities * dx * dy).sum((-2, -1))), dim=-1),
            torch.stack(((probabilities * dx * dy).sum((-2, -1)), (probabilities * dy * dy).sum((-2, -1))), dim=-1),
        ),
        dim=-2,
    )

    flat = probabilities.flatten(start_dim=-2)
    peak_values, peak_indices = flat.max(dim=-1)
    peak_x = peak_indices.remainder(width)
    peak_y = torch.div(peak_indices, width, rounding_mode="floor")
    peak_points_heatmap = torch.stack((peak_x, peak_y), dim=-1)
    boundary_cells = max(int(boundary_cells), 1)
    boundary_peaks = (
        (peak_x < boundary_cells)
        | (peak_x >= width - boundary_cells)
        | (peak_y < boundary_cells)
        | (peak_y >= height - boundary_cells)
    )
    entropy = -(flat * flat.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / torch.log(torch.as_tensor(float(height * width), dtype=dtype, device=device))

    return {
        "probability_maps": probabilities,
        "points_normalized": points_normalized,
        "points_heatmap": points_heatmap,
        "points_crop": points_crop,
        "peak_values": peak_values,
        "peak_points_heatmap": peak_points_heatmap,
        "entropy": entropy,
        "covariance": covariance,
        "boundary_peaks": boundary_peaks,
        "boundary_peak_rate": boundary_peaks.float().mean(),
    }


def transform_points_batch(points: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("points must have shape [B, K, 2]")
    if matrices.ndim != 3 or matrices.shape[-2:] != (3, 3):
        raise ValueError("matrices must have shape [B, 3, 3]")
    ones = torch.ones_like(points[..., :1])
    homogeneous = torch.cat((points, ones), dim=-1)
    mapped = torch.bmm(homogeneous, matrices.transpose(1, 2))
    return mapped[..., :2] / mapped[..., 2:3].clamp_min(1e-12)
