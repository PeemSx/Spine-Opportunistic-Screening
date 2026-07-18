from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CobbLine:
    start: tuple[float, float]
    end: tuple[float, float]
    sorted_index: int
    original_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_x": self.start[0],
            "start_y": self.start[1],
            "end_x": self.end[0],
            "end_y": self.end[1],
            "sorted_index": self.sorted_index,
            "original_index": self.original_index,
        }


@dataclass(frozen=True)
class CobbResult:
    valid: bool
    vertebra_count: int
    cobb_1_deg: float | None
    cobb_2_deg: float | None
    cobb_3_deg: float | None
    major_lines: tuple[CobbLine, CobbLine] | None
    cobb_2_lines: tuple[CobbLine, CobbLine] | None
    cobb_3_lines: tuple[CobbLine, CobbLine] | None

    def angles(self) -> list[float | None]:
        return [self.cobb_1_deg, self.cobb_2_deg, self.cobb_3_deg]

    def measurements(self) -> list[tuple[str, float | None, tuple[CobbLine, CobbLine] | None]]:
        return [
            ("Cobb 1", self.cobb_1_deg, self.major_lines),
            ("Cobb 2", self.cobb_2_deg, self.cobb_2_lines),
            ("Cobb 3", self.cobb_3_deg, self.cobb_3_lines),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "vertebra_count": self.vertebra_count,
            "cobb_1_deg": self.cobb_1_deg,
            "cobb_2_deg": self.cobb_2_deg,
            "cobb_3_deg": self.cobb_3_deg,
            "major_lines": [line.to_dict() for line in self.major_lines] if self.major_lines else [],
            "cobb_2_lines": [line.to_dict() for line in self.cobb_2_lines] if self.cobb_2_lines else [],
            "cobb_3_lines": [line.to_dict() for line in self.cobb_3_lines] if self.cobb_3_lines else [],
        }


def invalid_cobb_result(vertebra_count: int = 0) -> CobbResult:
    return CobbResult(
        valid=False,
        vertebra_count=int(vertebra_count),
        cobb_1_deg=None,
        cobb_2_deg=None,
        cobb_3_deg=None,
        major_lines=None,
        cobb_2_lines=None,
        cobb_3_lines=None,
    )


def _prepare_corners(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = np.asarray(corners, dtype=np.float32)
    if corners.ndim != 3 or corners.shape[1:] != (4, 2):
        return np.zeros((0, 4, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    finite_mask = np.isfinite(corners).all(axis=(1, 2))
    corners = corners[finite_mask]
    original_indices = np.flatnonzero(finite_mask).astype(np.int64)
    if len(corners) == 0:
        return corners, original_indices

    centers = corners.mean(axis=1)
    order = np.argsort(centers[:, 1], kind="stable")
    return corners[order], original_indices[order]


def _axis_lines(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_midpoints = (corners[:, 0, :] + corners[:, 2, :]) * 0.5
    right_midpoints = (corners[:, 1, :] + corners[:, 3, :]) * 0.5
    vectors = right_midpoints - left_midpoints
    return left_midpoints, right_midpoints, vectors


def _pairwise_line_angles(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    denom = norms[:, None] * norms[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosines = np.divide(
            vectors @ vectors.T,
            denom,
            out=np.zeros_like(denom, dtype=np.float32),
            where=denom > 1e-6,
        )
    cosines = np.clip(np.abs(cosines), 0.0, 1.0)
    return np.degrees(np.arccos(cosines)).astype(np.float32)


def _line(
    left_midpoints: np.ndarray,
    right_midpoints: np.ndarray,
    original_indices: np.ndarray,
    index: int,
) -> CobbLine:
    return CobbLine(
        start=(float(left_midpoints[index, 0]), float(left_midpoints[index, 1])),
        end=(float(right_midpoints[index, 0]), float(right_midpoints[index, 1])),
        sorted_index=int(index),
        original_index=int(original_indices[index]),
    )


def _lines_for_pair(
    left_midpoints: np.ndarray,
    right_midpoints: np.ndarray,
    original_indices: np.ndarray,
    first_index: int,
    second_index: int,
) -> tuple[CobbLine, CobbLine]:
    return (
        _line(left_midpoints, right_midpoints, original_indices, int(first_index)),
        _line(left_midpoints, right_midpoints, original_indices, int(second_index)),
    )


def _secondary_angles(
    angles: np.ndarray,
    first_index: int,
    second_index: int,
) -> tuple[float | None, tuple[int, int] | None, float | None, tuple[int, int] | None]:
    vertebra_count = int(angles.shape[0])
    if vertebra_count < 3:
        return None, None, None, None

    top_index = min(first_index, second_index)
    bottom_index = max(first_index, second_index)
    top_angle = None
    top_pair = None
    if top_index > 0:
        top_slice = angles[:top_index, top_index]
        top_partner = int(np.argmax(top_slice))
        top_angle = float(top_slice[top_partner])
        top_pair = (top_partner, top_index)

    bottom_angle = None
    bottom_pair = None
    if bottom_index < vertebra_count - 1:
        bottom_slice = angles[bottom_index, bottom_index + 1 :]
        bottom_partner = int(bottom_index + 1 + np.argmax(bottom_slice))
        bottom_angle = float(angles[bottom_index, bottom_partner])
        bottom_pair = (bottom_index, bottom_partner)

    return top_angle, top_pair, bottom_angle, bottom_pair


def calculate_cobb_angles(corners: np.ndarray) -> CobbResult:
    sorted_corners, original_indices = _prepare_corners(corners)
    vertebra_count = int(len(sorted_corners))
    if vertebra_count < 2:
        return invalid_cobb_result(vertebra_count)

    left_midpoints, right_midpoints, vectors = _axis_lines(sorted_corners)
    valid_lines = np.linalg.norm(vectors, axis=1) > 1e-6
    if int(valid_lines.sum()) < 2:
        return invalid_cobb_result(vertebra_count)

    if not valid_lines.all():
        sorted_corners = sorted_corners[valid_lines]
        original_indices = original_indices[valid_lines]
        left_midpoints = left_midpoints[valid_lines]
        right_midpoints = right_midpoints[valid_lines]
        vectors = vectors[valid_lines]
        vertebra_count = int(len(sorted_corners))

    angles = _pairwise_line_angles(vectors)
    candidate_angles = angles.copy()
    np.fill_diagonal(candidate_angles, -1.0)
    first_index, second_index = np.unravel_index(int(np.argmax(candidate_angles)), candidate_angles.shape)
    major_angle = float(angles[first_index, second_index])
    cobb_2, cobb_2_pair, cobb_3, cobb_3_pair = _secondary_angles(angles, int(first_index), int(second_index))
    major_lines = _lines_for_pair(left_midpoints, right_midpoints, original_indices, int(first_index), int(second_index))
    cobb_2_lines = (
        _lines_for_pair(left_midpoints, right_midpoints, original_indices, cobb_2_pair[0], cobb_2_pair[1])
        if cobb_2_pair is not None
        else None
    )
    cobb_3_lines = (
        _lines_for_pair(left_midpoints, right_midpoints, original_indices, cobb_3_pair[0], cobb_3_pair[1])
        if cobb_3_pair is not None
        else None
    )

    return CobbResult(
        valid=True,
        vertebra_count=vertebra_count,
        cobb_1_deg=major_angle,
        cobb_2_deg=cobb_2,
        cobb_3_deg=cobb_3,
        major_lines=major_lines,
        cobb_2_lines=cobb_2_lines,
        cobb_3_lines=cobb_3_lines,
    )


def cobb_smape(gt_angle: float, pred_angle: float) -> float:
    denom = abs(float(gt_angle)) + abs(float(pred_angle))
    if denom <= 1e-6:
        return 0.0
    return abs(float(gt_angle) - float(pred_angle)) / denom * 100.0
