from __future__ import annotations

import math
from typing import Any

import numpy as np


def keypoints_to_points(annotation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
    return keypoints[:, :2], keypoints[:, 2] > 0


def sort_vertebra_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(annotation: dict[str, Any]) -> tuple[float, float]:
        points, visible = keypoints_to_points(annotation)
        if len(points) != 4 or not visible.all():
            return (float("inf"), float("inf"))
        center = points.mean(axis=0)
        return (float(center[1]), float(center[0]))

    return sorted(annotations, key=sort_key)


def draw_gaussian(heatmap: np.ndarray, center_xy: np.ndarray, sigma: float) -> None:
    height, width = heatmap.shape
    cx, cy = float(center_xy[0]), float(center_xy[1])
    radius = max(int(math.ceil(3.0 * sigma)), 1)

    x0 = max(int(math.floor(cx)) - radius, 0)
    x1 = min(int(math.floor(cx)) + radius + 1, width)
    y0 = max(int(math.floor(cy)) - radius, 0)
    y1 = min(int(math.floor(cy)) + radius + 1, height)
    if x1 <= x0 or y1 <= y0:
        return

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    gaussian = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    width, height = max_xy - min_xy
    return np.asarray([min_xy[0], min_xy[1], width, height], dtype=np.float32)


def build_centernet_targets(
    image_shape: tuple[int, int] | tuple[int, int, int],
    annotations: list[dict[str, Any]],
    down_ratio: int = 4,
    max_objects: int = 64,
) -> dict[str, Any]:
    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    out_h = int(math.ceil(image_h / down_ratio))
    out_w = int(math.ceil(image_w / down_ratio))

    hm = np.zeros((1, out_h, out_w), dtype=np.float32)
    reg = np.zeros((max_objects, 2), dtype=np.float32)
    wh = np.zeros((max_objects, 8), dtype=np.float32)
    ind = np.zeros((max_objects,), dtype=np.int64)
    reg_mask = np.zeros((max_objects,), dtype=np.float32)
    gt_centers = np.zeros((max_objects, 2), dtype=np.float32)
    gt_corners = np.zeros((max_objects, 4, 2), dtype=np.float32)
    records: list[dict[str, Any]] = []
    occupied: set[int] = set()
    collisions = 0
    truncated = 0

    for annotation in sort_vertebra_annotations(annotations):
        points, visible = keypoints_to_points(annotation)
        if len(points) != 4 or not visible.all():
            continue
        if len(records) >= max_objects:
            truncated += 1
            continue

        center = points.mean(axis=0)
        center_out = center / float(down_ratio)
        ix = int(np.floor(center_out[0]))
        iy = int(np.floor(center_out[1]))
        if ix < 0 or ix >= out_w or iy < 0 or iy >= out_h:
            continue

        bbox = np.asarray(annotation.get("bbox") or _bbox_from_points(points), dtype=np.float32)
        bbox_w = max(float(bbox[2]), 1.0)
        bbox_h = max(float(bbox[3]), 1.0)
        sigma = max(1.0, min(bbox_w, bbox_h) / float(down_ratio) * 0.12)

        draw_gaussian(hm[0], center_out, sigma)
        hm[0, iy, ix] = 1.0

        flat_index = iy * out_w + ix
        if flat_index in occupied:
            collisions += 1
        occupied.add(flat_index)

        object_index = len(records)
        ind[object_index] = flat_index
        reg[object_index] = center_out - np.asarray([ix, iy], dtype=np.float32)
        points_out = points / float(down_ratio)
        wh[object_index] = (center_out[None, :] - points_out).reshape(-1).astype(np.float32)
        reg_mask[object_index] = 1.0
        gt_centers[object_index] = center.astype(np.float32)
        gt_corners[object_index] = points.astype(np.float32)

        records.append(
            {
                "annotation_id": annotation.get("id"),
                "center_px": center.astype(np.float32),
                "center_out": center_out.astype(np.float32),
                "grid_xy": np.asarray([ix, iy], dtype=np.int32),
                "points_px": points.astype(np.float32),
                "points_out": points_out.astype(np.float32),
                "sigma": float(sigma),
                "bbox": bbox,
            }
        )

    return {
        "hm": hm,
        "reg": reg,
        "wh": wh,
        "ind": ind,
        "reg_mask": reg_mask,
        "gt_centers": gt_centers,
        "gt_corners": gt_corners,
        "gt_count": np.asarray(len(records), dtype=np.int64),
        "collisions": np.asarray(collisions, dtype=np.int64),
        "truncated": np.asarray(truncated, dtype=np.int64),
        "records": records,
        "out_shape": (out_h, out_w),
    }
