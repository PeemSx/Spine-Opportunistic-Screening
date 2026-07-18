from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpineCandidate:
    index: int
    score: float
    center: np.ndarray
    corners: np.ndarray
    box: np.ndarray

    @property
    def x(self) -> float:
        return float(self.center[0])

    @property
    def y(self) -> float:
        return float(self.center[1])

    @property
    def h(self) -> float:
        return float(max(self.box[3] - self.box[1], 1.0))


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def box_from_corners(corners: np.ndarray, center: np.ndarray) -> np.ndarray:
    if corners.shape == (4, 2) and np.isfinite(corners).all():
        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)
        if np.all(max_xy > min_xy):
            return np.asarray([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32)

    x_coord, y_coord = float(center[0]), float(center[1])
    return np.asarray([x_coord - 0.5, y_coord - 0.5, x_coord + 0.5, y_coord + 0.5], dtype=np.float32)


def prediction_to_candidates(prediction: dict[str, np.ndarray]) -> list[SpineCandidate]:
    scores = prediction["scores"]
    centers = prediction["centers"]
    corners = prediction["corners"]
    candidates: list[SpineCandidate] = []

    for index, (score, center, points) in enumerate(zip(scores, centers, corners)):
        center = np.asarray(center, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)
        if center.shape != (2,) or points.shape != (4, 2):
            continue
        if not np.isfinite(center).all() or not np.isfinite(points).all():
            continue

        candidates.append(
            SpineCandidate(
                index=int(index),
                score=float(score),
                center=center,
                corners=points,
                box=box_from_corners(points, center),
            )
        )
    return candidates


def candidates_to_prediction(candidates: list[SpineCandidate]) -> dict[str, np.ndarray]:
    if not candidates:
        return {
            "scores": np.zeros((0,), dtype=np.float32),
            "centers": np.zeros((0, 2), dtype=np.float32),
            "corners": np.zeros((0, 4, 2), dtype=np.float32),
        }

    return {
        "scores": np.asarray([candidate.score for candidate in candidates], dtype=np.float32),
        "centers": np.asarray([candidate.center for candidate in candidates], dtype=np.float32),
        "corners": np.asarray([candidate.corners for candidate in candidates], dtype=np.float32),
    }


def suppress_duplicate_candidates(
    candidates: list[SpineCandidate],
    iou_threshold: float = 0.18,
    center_scale: float = 0.35,
) -> list[SpineCandidate]:
    kept: list[SpineCandidate] = []
    for candidate in sorted(candidates, key=lambda item: -item.score):
        duplicate = False
        for previous in kept:
            iou = box_iou(candidate.box, previous.box)
            dist = math.hypot(candidate.x - previous.x, candidate.y - previous.y)
            radius = float(center_scale) * min(candidate.h, previous.h)
            if iou >= float(iou_threshold) or dist <= radius:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: (item.y, item.x))


def estimate_spacing(candidates: list[SpineCandidate]) -> tuple[float, float, float]:
    if not candidates:
        return 32.0, 16.0, 80.0

    heights = np.asarray([candidate.h for candidate in candidates], dtype=np.float32)
    median_h = float(np.median(heights)) if len(heights) else 32.0
    ys = np.asarray(sorted(candidate.y for candidate in candidates), dtype=np.float32)
    gaps = np.diff(ys)
    plausible = gaps[(gaps >= 0.35 * median_h) & (gaps <= 2.20 * median_h)]

    if len(plausible) >= 2:
        target_dy = float(np.median(plausible))
    else:
        target_dy = max(0.85 * median_h, 12.0)

    min_dy = max(0.35 * target_dy, 8.0)
    max_dy = max(2.10 * target_dy, min_dy + 8.0)
    return target_dy, min_dy, max_dy


def transition_score(
    previous: SpineCandidate,
    current: SpineCandidate,
    target_dy: float,
    min_dy: float,
    max_dy: float,
) -> float | None:
    dy = current.y - previous.y
    if dy <= 0.0 or dy < min_dy or dy > max_dy:
        return None

    dx = current.x - previous.x
    if abs(dx) > 1.35 * max(target_dy, 1.0):
        return None

    spacing_penalty = ((dy - target_dy) / max(target_dy, 1.0)) ** 2
    lateral_penalty = (dx / max(target_dy, 1.0)) ** 2
    size_penalty = abs(math.log(max(current.h, 1.0) / max(previous.h, 1.0)))
    return -(0.55 * spacing_penalty + 0.35 * lateral_penalty + 0.20 * size_penalty)


def dynamic_spine_chain(
    candidates: list[SpineCandidate],
    score_threshold: float = 0.18,
    score_weight: float = 3.0,
    min_chain_len: int = 3,
) -> tuple[list[SpineCandidate], dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: (item.y, item.x))
    if not ordered:
        return [], {
            "score": 0.0,
            "target_dy": 0.0,
            "min_dy": 0.0,
            "max_dy": 0.0,
            "node_score_threshold": float(score_threshold),
            "num_candidates": 0,
            "num_selected": 0,
        }

    target_dy, min_dy, max_dy = estimate_spacing(ordered)
    count = len(ordered)
    best = np.full(count, -np.inf, dtype=np.float64)
    length = np.ones(count, dtype=np.int32)
    parent = np.full(count, -1, dtype=np.int32)
    node_values = np.asarray(
        [float(score_weight) * (candidate.score - float(score_threshold)) for candidate in ordered],
        dtype=np.float64,
    )
    best[:] = node_values

    for current_index in range(count):
        for previous_index in range(current_index):
            edge = transition_score(ordered[previous_index], ordered[current_index], target_dy, min_dy, max_dy)
            if edge is None:
                continue
            value = best[previous_index] + node_values[current_index] + edge
            if value > best[current_index]:
                best[current_index] = value
                length[current_index] = length[previous_index] + 1
                parent[current_index] = previous_index

    valid_ends = np.where(length >= int(min_chain_len))[0]
    if len(valid_ends) == 0:
        valid_ends = np.arange(count)

    end_index = int(valid_ends[np.argmax(best[valid_ends])])
    chain_indices = []
    current = end_index
    while current >= 0:
        chain_indices.append(current)
        current = int(parent[current])
    chain_indices.reverse()

    chain = [ordered[index] for index in chain_indices]
    debug = {
        "score": float(best[end_index]),
        "target_dy": float(target_dy),
        "min_dy": float(min_dy),
        "max_dy": float(max_dy),
        "node_score_threshold": float(score_threshold),
        "num_candidates": int(count),
        "num_selected": int(len(chain)),
    }
    return chain, debug


def select_spine_chain(
    candidates: list[SpineCandidate],
    duplicate_iou_threshold: float = 0.18,
    duplicate_center_scale: float = 0.35,
    score_threshold: float = 0.18,
    score_weight: float = 3.0,
    min_chain_len: int = 3,
) -> tuple[list[SpineCandidate], list[SpineCandidate], dict[str, Any]]:
    deduplicated = suppress_duplicate_candidates(
        candidates,
        iou_threshold=duplicate_iou_threshold,
        center_scale=duplicate_center_scale,
    )
    chain, debug = dynamic_spine_chain(
        deduplicated,
        score_threshold=score_threshold,
        score_weight=score_weight,
        min_chain_len=min_chain_len,
    )
    debug["num_raw"] = int(len(candidates))
    debug["num_deduplicated"] = int(len(deduplicated))
    return deduplicated, chain, debug

