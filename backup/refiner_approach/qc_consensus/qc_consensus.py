"""Archived opt-in QC and seven-ROI consensus helpers.

This module is intentionally outside ``src`` and is not used by the active
pipeline.  It preserves the removed behavior so it can be restored without
reconstructing the rules from memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QcConsensusConfig:
    max_center_offset_fraction: float = 0.20
    reject_boundary_peaks: bool = True
    require_nearest_anchor: bool = True
    consensus_enabled: bool = False
    consensus_scale_delta: float = 0.10
    consensus_shift_fraction: float = 0.05
    max_consensus_p95_fraction: float = 0.04


def detector_roi_variants(
    base_rois: np.ndarray,
    config: QcConsensusConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Recreate the archived base/scale/x-shift/y-shift ROI variants."""
    base_rois = np.asarray(base_rois, dtype=np.float32)
    if not config.consensus_enabled:
        return base_rois[:, None, :], ("base",)

    scale_delta = float(config.consensus_scale_delta)
    shift_fraction = float(config.consensus_shift_fraction)
    if not 0.0 <= scale_delta < 1.0:
        raise ValueError("consensus_scale_delta must be in [0, 1)")
    if shift_fraction < 0.0:
        raise ValueError("consensus_shift_fraction must be non-negative")
    if float(config.max_consensus_p95_fraction) < 0.0:
        raise ValueError("max_consensus_p95_fraction must be non-negative")

    names = ("base", "smaller", "larger", "left", "right", "up", "down")
    policies = (
        (1.0, 0.0, 0.0),
        (1.0 - scale_delta, 0.0, 0.0),
        (1.0 + scale_delta, 0.0, 0.0),
        (1.0, -shift_fraction, 0.0),
        (1.0, shift_fraction, 0.0),
        (1.0, 0.0, -shift_fraction),
        (1.0, 0.0, shift_fraction),
    )
    base_centers = 0.5 * (base_rois[:, :2] + base_rois[:, 2:])
    base_sides = base_rois[:, 2] - base_rois[:, 0]
    variants = np.empty((len(base_rois), len(policies), 4), dtype=np.float32)
    for variant_index, (scale_multiplier, shift_x, shift_y) in enumerate(policies):
        side = base_sides * float(scale_multiplier)
        shifted_centers = base_centers + base_sides[:, None] * np.asarray(
            [shift_x, shift_y], dtype=np.float32
        )
        half = side[:, None] * 0.5
        variants[:, variant_index] = np.concatenate(
            (shifted_centers - half, shifted_centers + half), axis=1
        )
    return variants, names


def compute_corner_consensus(
    variant_points: np.ndarray,
    base_roi_sides: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median corners plus RMS and p95 disagreement/ROI-side."""
    points = np.asarray(variant_points, dtype=np.float32)
    sides = np.asarray(base_roi_sides, dtype=np.float32)
    if points.ndim != 4 or points.shape[2:] != (4, 2):
        raise ValueError("variant_points must have shape [N, V, 4, 2]")
    if sides.shape != (len(points),):
        raise ValueError("base_roi_sides must have shape [N]")
    consensus = np.median(points, axis=1).astype(np.float32)
    distances = np.linalg.norm(points - consensus[:, None], axis=-1)
    denominator = np.maximum(sides, 1e-6)
    rms = np.sqrt(np.mean(np.square(distances), axis=(1, 2))) / denominator
    p95 = np.percentile(distances, 95, axis=(1, 2)) / denominator
    return consensus, rms.astype(np.float32), p95.astype(np.float32)


def valid_corner_geometry(points: np.ndarray, roi_side: float) -> bool:
    """Archived TL/TR/BL/BR ordering and minimum-area rule."""
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


def candidate_qc_reasons(
    *,
    candidate_index: int,
    refined_points: np.ndarray,
    center_offset_fraction: float,
    nearest_detector_index: int,
    base_boundary_peaks: np.ndarray,
    consensus_p95_fraction: float,
    roi_side: float,
    config: QcConsensusConfig,
) -> list[str]:
    """Return the exact archived rejection reasons in their original order."""
    reasons: list[str] = []
    if center_offset_fraction > config.max_center_offset_fraction:
        reasons.append("center_offset")
    if config.require_nearest_anchor and nearest_detector_index != candidate_index:
        reasons.append("identity_switch")
    if not valid_corner_geometry(refined_points, roi_side):
        reasons.append("corner_geometry")
    if config.reject_boundary_peaks and bool(np.asarray(base_boundary_peaks).any()):
        reasons.append("boundary_peak")
    if (
        config.consensus_enabled
        and consensus_p95_fraction > config.max_consensus_p95_fraction
    ):
        reasons.append("unstable_consensus")
    return reasons
