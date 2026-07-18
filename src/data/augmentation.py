from __future__ import annotations

import copy
import inspect
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# set an Albumentations environment variable to avoid update checks
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1") 

import albumentations as A
import cv2
import numpy as np
import yaml

PRECISION = 3
KEYPOINT_NAMES = ["top_left", "top_right", "bottom_left", "bottom_right"]
POLYGON_ORDER = [0, 1, 3, 2]
REQUIRED_CONFIG_KEYS = {
    "image_size",
    "seed",
    "horizontal_flip",
    "preview_count",
}


# Configuration is loaded from configs/config.yaml; this class only gives typed access.
@dataclass(frozen=True) # Configs can't be changed after creation
class AugmentationConfig:
    image_size: int
    seed: int
    horizontal_flip: bool
    preview_count: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AugmentationConfig":
        missing = sorted(REQUIRED_CONFIG_KEYS - set(data))
        if missing:
            raise ValueError(
                "Missing augmentation config keys: " + ", ".join(missing)
            )

        return cls(
            image_size=int(data["image_size"]),
            seed=int(data["seed"]),
            horizontal_flip=bool(data["horizontal_flip"]),
            preview_count=int(data["preview_count"]),
        )


class AugmentationError(RuntimeError):
    pass


def load_augmentation_config(path: Path | str = Path("configs/config.yaml")) -> AugmentationConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    augmentation = config.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ValueError("configs/config.yaml must contain an 'augmentation' mapping")

    return AugmentationConfig.from_mapping(augmentation)


def seed_augmentation(seed: int) -> None:
    """Seed common random sources used by preview and augmentation transforms."""
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(A, "set_seed"):
        A.set_seed(seed)


# Geometry helpers rebuild derived COCO fields from transformed corner points.
def round_float(value: float) -> float:
    return round(float(value), PRECISION)


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def bbox_from_points(points: np.ndarray) -> list[float]:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    width, height = max_xy - min_xy
    return [
        round_float(min_xy[0]),
        round_float(min_xy[1]),
        round_float(width),
        round_float(height),
    ]


def segmentation_from_points(points: np.ndarray) -> list[list[float]]:
    polygon = points[POLYGON_ORDER]
    return [[round_float(value) for point in polygon for value in point]]


def validate_points(points: np.ndarray, width: int, height: int) -> bool:
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] > width):
        return False
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] > height):
        return False

    bbox = bbox_from_points(points)
    if bbox[2] <= 0 or bbox[3] <= 0:
        return False
    return polygon_area(points[POLYGON_ORDER]) > 0


# Albumentations changed a few argument names across versions, so these builders adapt.
def _pad_to_square(image_size: int) -> A.BasicTransform:
    params = inspect.signature(A.PadIfNeeded).parameters
    kwargs: dict[str, Any] = {
        "min_height": image_size,
        "min_width": image_size,
        "border_mode": cv2.BORDER_CONSTANT,
        "p": 1.0,
    }
    if "fill" in params:
        kwargs["fill"] = 0
    elif "value" in params:
        kwargs["value"] = 0
    return A.PadIfNeeded(**kwargs)


def _safe_affine() -> A.BasicTransform:
    params = inspect.signature(A.Affine).parameters
    kwargs: dict[str, Any] = {
        "scale": (0.9, 1.1),
        "translate_percent": (-0.04, 0.04),
        "rotate": (-7, 7),
        "shear": 0,
        "p": 0.85,
    }
    if "border_mode" in params:
        kwargs["border_mode"] = cv2.BORDER_CONSTANT
    elif "mode" in params:
        kwargs["mode"] = cv2.BORDER_CONSTANT
    if "fill" in params:
        kwargs["fill"] = 0
    elif "cval" in params:
        kwargs["cval"] = 0
    return A.Affine(**kwargs)


def _compose(transforms: list[A.BasicTransform]) -> A.Compose:
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


# Transform builders are split so training can be random while val/test stay deterministic.
def build_train_transform(config: AugmentationConfig | None = None) -> A.Compose:
    config = config or load_augmentation_config()
    if config.horizontal_flip:
        raise ValueError("horizontal_flip requires keypoint permutation support first")

    return _compose(
        [
            A.LongestMaxSize(max_size=config.image_size, p=1.0),
            _pad_to_square(config.image_size),
            _safe_affine(),
            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.08,
                p=0.35,
            ),
            A.RandomGamma(gamma_limit=(85, 115), p=0.25),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                    A.GaussNoise(p=1.0),
                ],
                p=0.12,
            ),
        ]
    )


def build_eval_transform(config: AugmentationConfig | None = None) -> A.Compose:
    config = config or load_augmentation_config()
    return _compose(
        [
            A.LongestMaxSize(max_size=config.image_size, p=1.0),
            _pad_to_square(config.image_size),
        ]
    )


# COCO conversion keeps landmark order unchanged and transforms only real keypoints.
def flatten_annotation_keypoints(
    annotations: list[dict[str, Any]],
) -> tuple[list[tuple[float, float]], list[list[int]]]:
    keypoints: list[tuple[float, float]] = []
    visibility: list[list[int]] = []

    for annotation in annotations:
        raw = np.asarray(annotation["keypoints"], dtype=float).reshape(4, 3)
        keypoints.extend((float(x), float(y)) for x, y in raw[:, :2])
        visibility.append([int(value) for value in raw[:, 2]])

    return keypoints, visibility


def rebuild_annotation(
    annotation: dict[str, Any],
    points: np.ndarray,
    visibility: list[int],
    width: int,
    height: int,
) -> dict[str, Any] | None:
    if not validate_points(points, width=width, height=height):
        return None

    rebuilt = copy.deepcopy(annotation)
    rebuilt["bbox"] = bbox_from_points(points)
    rebuilt["area"] = round_float(polygon_area(points[POLYGON_ORDER]))
    rebuilt["segmentation"] = segmentation_from_points(points)
    rebuilt["num_keypoints"] = 4
    rebuilt["keypoints"] = [
        round_float(value) if index % 3 != 2 else int(value)
        for index, value in enumerate(
            np.column_stack([points, visibility]).reshape(-1)
        )
    ]
    return rebuilt


def transform_coco_sample(
    image: np.ndarray,
    annotations: list[dict[str, Any]],
    transform: A.Compose,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not annotations:
        raise AugmentationError("sample has no annotations")

    keypoints, visibility = flatten_annotation_keypoints(annotations)
    result = transform(image=image, keypoints=keypoints)

    transformed_keypoints = np.asarray(result["keypoints"], dtype=float)
    if transformed_keypoints.shape != (len(annotations) * 4, 2):
        raise AugmentationError("transformed keypoint count does not match annotations")

    transformed_image = result["image"]
    height, width = transformed_image.shape[:2]
    rebuilt_annotations: list[dict[str, Any]] = []

    for index, annotation in enumerate(annotations):
        start = index * 4
        points = transformed_keypoints[start : start + 4]
        rebuilt = rebuild_annotation(
            annotation=annotation,
            points=points,
            visibility=visibility[index],
            width=width,
            height=height,
        )
        if rebuilt is not None:
            rebuilt_annotations.append(rebuilt)

    if not rebuilt_annotations:
        raise AugmentationError("all annotations were invalid after augmentation")

    return transformed_image, rebuilt_annotations
