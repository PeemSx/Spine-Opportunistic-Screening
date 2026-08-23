from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import (
    AugmentationConfig,
    AugmentationError,
    build_eval_transform,
    build_train_transform,
    load_augmentation_config,
    seed_augmentation,
    transform_coco_sample,
)
from src.data.centernet_targets import build_centernet_targets, keypoints_to_points
from src.data.inference_geometry import (
    TransformMeta,
    resize_pad_image,
    transform_annotations_to_model,
)


def _load_config(config_path: Path | None, image_size: int | None) -> AugmentationConfig:
    if config_path is not None and config_path.exists():
        config = load_augmentation_config(config_path)
    else:
        config = AugmentationConfig(
            image_size=image_size or 1024,
            seed=20260627,
            horizontal_flip=False,
            preview_count=24,
        )

    if image_size is not None and config.image_size != image_size:
        config = AugmentationConfig(
            image_size=image_size,
            seed=config.seed,
            horizontal_flip=config.horizontal_flip,
            preview_count=config.preview_count,
        )
    return config


def load_coco_samples(split_dir: Path) -> list[dict[str, Any]]:
    annotation_path = split_dir / "_annotations.keypoints.coco.json"
    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    samples = []
    for image in coco["images"]:
        annotations = annotations_by_image.get(int(image["id"]), [])
        if annotations:
            samples.append({"image": image, "annotations": annotations})
    return samples


def image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0 - 0.5
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(image))


class CenterNetCocoDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path | str,
        split: str,
        config_path: Path | str | None = "configs/config.yaml",
        image_size: int | None = None,
        down_ratio: int = 4,
        max_objects: int = 64,
        augment: bool | None = None,
        limit: int | None = None,
        source_dataset: str | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.split_dir = self.dataset_root / split
        self.down_ratio = int(down_ratio)
        self.max_objects = int(max_objects)
        self.config = _load_config(Path(config_path) if config_path is not None else None, image_size)
        self.augment = split == "train" if augment is None else bool(augment)
        self.samples = load_coco_samples(self.split_dir)
        self.source_dataset = source_dataset.strip() if source_dataset is not None else None
        if self.source_dataset:
            requested_source = self.source_dataset.casefold()
            available_sources = sorted(
                {
                    str(sample["image"].get("source_dataset", "unknown"))
                    for sample in self.samples
                }
            )
            self.samples = [
                sample
                for sample in self.samples
                if str(sample["image"].get("source_dataset", "unknown")).casefold()
                == requested_source
            ]
            if not self.samples:
                raise ValueError(
                    f"No {split!r} samples found for source_dataset={self.source_dataset!r}. "
                    f"Available sources: {available_sources}"
                )
        if limit is not None:
            self.samples = self.samples[: int(limit)]

        seed_augmentation(self.config.seed)
        self.train_transform = build_train_transform(self.config)
        self.eval_transform = build_eval_transform(self.config)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_info: dict[str, Any]) -> np.ndarray:
        image_path = self.split_dir / image_info["file_name"]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _transform_sample(
        self,
        image: np.ndarray,
        annotations: list[dict[str, Any]],
    ) -> tuple[np.ndarray, list[dict[str, Any]], TransformMeta]:
        if not self.augment:
            image, meta = resize_pad_image(image, input_size=self.config.image_size)
            return image, transform_annotations_to_model(annotations, meta), meta

        transform = self.train_transform
        try:
            transformed_image, transformed_annotations = transform_coco_sample(
                image=image,
                annotations=annotations,
                transform=transform,
            )
        except AugmentationError:
            transformed_image, transformed_annotations = transform_coco_sample(
                image=image,
                annotations=annotations,
                transform=self.eval_transform,
            )
        _, meta = resize_pad_image(image, input_size=self.config.image_size)
        return transformed_image, transformed_annotations, meta

    def _original_targets(
        self,
        annotations: list[dict[str, Any]],
        records: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        centers = np.zeros((self.max_objects, 2), dtype=np.float32)
        corners = np.zeros((self.max_objects, 4, 2), dtype=np.float32)
        annotations_by_id = {
            int(annotation["id"]): annotation
            for annotation in annotations
            if annotation.get("id") is not None
        }
        for index, record in enumerate(records):
            annotation_id = record.get("annotation_id")
            annotation = (
                annotations_by_id.get(int(annotation_id))
                if annotation_id is not None
                else None
            )
            if annotation is None:
                continue
            points, visible = keypoints_to_points(annotation)
            if len(points) != 4 or not visible.all():
                continue
            corners[index] = points.astype(np.float32)
            centers[index] = points.mean(axis=0).astype(np.float32)
        return centers, corners

    @staticmethod
    def _patient_cluster_id(image_info: dict[str, Any]) -> str:
        for field in ("patient_id", "nih_patient_id"):
            explicit = image_info.get(field)
            if explicit not in (None, ""):
                return str(explicit)
        lumos_case_id = image_info.get("lumos_case_id")
        if lumos_case_id not in (None, ""):
            return f"lumos:{lumos_case_id}"
        source = str(image_info.get("source_dataset", "unknown")).casefold()
        file_name = Path(str(image_info["file_name"])).stem
        if "nih" in source and file_name.startswith("NIH_"):
            parts = file_name.split("_")
            if len(parts) >= 2:
                return "_".join(parts[:2])
        return f"image:{int(image_info['id'])}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_info = copy.deepcopy(sample["image"])
        annotations = copy.deepcopy(sample["annotations"])

        original_annotations = copy.deepcopy(annotations)
        image = self._load_image(image_info)
        image, annotations, transform_meta = self._transform_sample(image, annotations)
        targets = build_centernet_targets(
            image_shape=image.shape,
            annotations=annotations,
            down_ratio=self.down_ratio,
            max_objects=self.max_objects,
        )
        gt_centers_original, gt_corners_original = self._original_targets(
            original_annotations,
            targets["records"],
        )

        return {
            "input": image_to_tensor(image),
            "hm": torch.from_numpy(targets["hm"]),
            "reg": torch.from_numpy(targets["reg"]),
            "wh": torch.from_numpy(targets["wh"]),
            "ind": torch.from_numpy(targets["ind"]),
            "reg_mask": torch.from_numpy(targets["reg_mask"]),
            "gt_centers": torch.from_numpy(targets["gt_centers"]),
            "gt_corners": torch.from_numpy(targets["gt_corners"]),
            "gt_centers_original": torch.from_numpy(gt_centers_original),
            "gt_corners_original": torch.from_numpy(gt_corners_original),
            "gt_count": torch.as_tensor(targets["gt_count"], dtype=torch.long),
            "collisions": torch.as_tensor(targets["collisions"], dtype=torch.long),
            "truncated": torch.as_tensor(targets["truncated"], dtype=torch.long),
            "file_name": image_info["file_name"],
            "source_dataset": image_info.get("source_dataset", "unknown"),
            "image_id": int(image_info["id"]),
            "patient_cluster_id": self._patient_cluster_id(image_info),
            "original_width": int(transform_meta.original_width),
            "original_height": int(transform_meta.original_height),
            "resized_width": int(transform_meta.resized_width),
            "resized_height": int(transform_meta.resized_height),
            "pad_left": int(transform_meta.pad_left),
            "pad_top": int(transform_meta.pad_top),
            "scale": float(transform_meta.scale),
            "input_size": int(transform_meta.input_size),
        }
