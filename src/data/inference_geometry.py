from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TransformMeta:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale: float
    input_size: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def resize_pad_image(
    image_rgb: np.ndarray,
    input_size: int,
) -> tuple[np.ndarray, TransformMeta]:
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    original_height, original_width = image_rgb.shape[:2]
    scale = float(input_size) / float(max(original_height, original_width))
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_left = (input_size - resized_width) // 2
    pad_top = (input_size - resized_height) // 2
    padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    padded[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    return padded, TransformMeta(
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        scale=scale,
        input_size=input_size,
    )


def map_points_to_original(points: np.ndarray, meta: TransformMeta) -> np.ndarray:
    mapped = np.asarray(points, dtype=np.float32).copy()
    mapped[..., 0] = (mapped[..., 0] - float(meta.pad_left)) / float(meta.scale)
    mapped[..., 1] = (mapped[..., 1] - float(meta.pad_top)) / float(meta.scale)
    return mapped


def map_points_to_model(points: np.ndarray, meta: TransformMeta) -> np.ndarray:
    mapped = np.asarray(points, dtype=np.float32).copy()
    mapped[..., 0] = mapped[..., 0] * float(meta.scale) + float(meta.pad_left)
    mapped[..., 1] = mapped[..., 1] * float(meta.scale) + float(meta.pad_top)
    return mapped


def valid_center_mask(centers: np.ndarray, meta: TransformMeta) -> np.ndarray:
    centers = np.asarray(centers)
    if len(centers) == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (centers[:, 0] >= 0.0)
        & (centers[:, 0] < float(meta.original_width))
        & (centers[:, 1] >= 0.0)
        & (centers[:, 1] < float(meta.original_height))
    )


def transform_annotations_to_model(
    annotations: list[dict[str, Any]],
    meta: TransformMeta,
) -> list[dict[str, Any]]:
    transformed = copy.deepcopy(annotations)
    for annotation in transformed:
        keypoints = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
        points = map_points_to_model(keypoints[:, :2], meta)
        keypoints[:, :2] = points
        annotation["keypoints"] = keypoints.reshape(-1).tolist()
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        size = max_xy - min_xy
        annotation["bbox"] = [
            float(min_xy[0]),
            float(min_xy[1]),
            float(size[0]),
            float(size[1]),
        ]
        annotation["area"] = float(max(size[0], 0.0) * max(size[1], 0.0))
    return transformed
