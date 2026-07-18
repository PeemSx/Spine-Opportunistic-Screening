from __future__ import annotations

import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data.corner_refiner_targets import build_gaussian_heatmaps
from src.data.roi_geometry import crop_from_roi, sample_contained_square_roi


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class CornerRefinerRecord:
    dataset_index: int
    image_id: int
    annotation_id: int
    vertebra_rank: int
    file_name: str
    source_dataset: str
    points: np.ndarray


def _annotation_points(annotation: dict[str, Any]) -> np.ndarray | None:
    keypoints = np.asarray(annotation.get("keypoints", []), dtype=np.float32)
    if keypoints.size != 12:
        return None
    keypoints = keypoints.reshape(4, 3)
    if not np.all(keypoints[:, 2] > 0) or not np.isfinite(keypoints).all():
        return None
    return keypoints[:, :2].astype(np.float32)


def _bbox_diagonal(points: np.ndarray) -> float:
    size = points.max(axis=0) - points.min(axis=0)
    return max(float(np.linalg.norm(size)), 1e-6)


def _canonical_source_name(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if "buu" in normalized:
        return "buu_ap"
    if "mendeley" in normalized:
        return "mendeley_pa"
    if "miccai" in normalized:
        return "miccai_2019"
    if "nih" in normalized or "chestx" in normalized:
        return "nih_chestxray14"
    return normalized


class CornerRefinerCocoDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path | str,
        split: str,
        *,
        crop_size: int = 256,
        heatmap_size: int = 64,
        crop_scale: float = 1.5,
        heatmap_sigma: float = 1.0,
        augment: bool | None = None,
        center_jitter_fraction: float = 0.08,
        scale_jitter_range: tuple[float, float] = (0.90, 1.10),
        jitter_retries: int = 10,
        cache_size: int = 4,
        limit: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = str(split)
        self.split_dir = self.dataset_root / self.split
        self.crop_size = int(crop_size)
        self.heatmap_size = int(heatmap_size)
        self.crop_scale = float(crop_scale)
        self.heatmap_sigma = float(heatmap_sigma)
        self.augment = self.split == "train" if augment is None else bool(augment)
        self.center_jitter_fraction = float(center_jitter_fraction)
        self.scale_jitter_range = tuple(float(value) for value in scale_jitter_range)
        self.jitter_retries = int(jitter_retries)
        self.cache_size = max(int(cache_size), 0)
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._rng: np.random.Generator | None = None
        self._rng_seed: int | None = None

        annotation_path = self.split_dir / "_annotations.keypoints.coco.json"
        with annotation_path.open("r", encoding="utf-8") as file:
            coco = json.load(file)

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        self.records: list[CornerRefinerRecord] = []
        images = sorted(coco["images"], key=lambda item: str(item["file_name"]).lower())
        for dataset_index, image_info in enumerate(images):
            candidates = []
            for annotation in annotations_by_image.get(int(image_info["id"]), []):
                points = _annotation_points(annotation)
                if points is None:
                    continue
                candidates.append((float(points[:, 1].mean()), float(points[:, 0].mean()), annotation, points))
            candidates.sort(key=lambda item: (item[0], item[1]))
            for vertebra_rank, (_, _, annotation, points) in enumerate(candidates, start=1):
                self.records.append(
                    CornerRefinerRecord(
                        dataset_index=int(dataset_index),
                        image_id=int(image_info["id"]),
                        annotation_id=int(annotation.get("id", -1)),
                        vertebra_rank=int(vertebra_rank),
                        file_name=str(image_info["file_name"]),
                        source_dataset=_canonical_source_name(
                            image_info.get("source_dataset")
                            or annotation.get("source_dataset")
                            or "unknown"
                        ),
                        points=points,
                    )
                )
        if limit is not None:
            self.records = self.records[: int(limit)]
        if not self.records:
            raise ValueError(f"No valid four-corner annotations found in {annotation_path}")

        self.image_record_indices: dict[int, list[int]] = defaultdict(list)
        self.image_sources: dict[int, str] = {}
        for record_index, record in enumerate(self.records):
            self.image_record_indices[record.image_id].append(record_index)
            self.image_sources[record.image_id] = record.source_dataset

    def __len__(self) -> int:
        return len(self.records)

    def _get_rng(self) -> np.random.Generator:
        seed = int(torch.initial_seed() % (2**32))
        if self._rng is None or self._rng_seed != seed:
            self._rng = np.random.default_rng(seed)
            self._rng_seed = seed
        return self._rng

    def _load_rgb(self, file_name: str) -> np.ndarray:
        cached = self._image_cache.get(file_name)
        if cached is not None:
            self._image_cache.move_to_end(file_name)
            return cached
        image_path = self.split_dir / file_name
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.cache_size > 0:
            self._image_cache[file_name] = image_rgb
            self._image_cache.move_to_end(file_name)
            while len(self._image_cache) > self.cache_size:
                self._image_cache.popitem(last=False)
        return image_rgb

    @staticmethod
    def _photometric_augment(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = gray.astype(np.float32)
        if rng.random() < 0.35:
            contrast = float(rng.uniform(0.85, 1.15))
            brightness = float(rng.uniform(-0.15, 0.15)) * 255.0
            image = image * contrast + brightness
        image = np.clip(image, 0.0, 255.0)
        if rng.random() < 0.25:
            gamma = float(rng.uniform(0.8, 1.2))
            image = 255.0 * np.power(image / 255.0, gamma)
        if rng.random() < 0.12:
            if rng.random() < 0.5:
                image = cv2.GaussianBlur(image, (3, 3), 0)
            else:
                image = image + rng.normal(0.0, 4.0, size=image.shape).astype(np.float32)
        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _image_to_tensor(gray: np.ndarray) -> torch.Tensor:
        image = gray.astype(np.float32) / 255.0
        image = np.repeat(image[None, :, :], 3, axis=0)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        image_rgb = self._load_rgb(record.file_name)
        rng = self._get_rng() if self.augment else None
        roi_xyxy, jitter_attempts, used_fallback = sample_contained_square_roi(
            record.points,
            crop_scale=self.crop_scale,
            rng=rng,
            center_jitter_fraction=self.center_jitter_fraction,
            scale_jitter_range=self.scale_jitter_range,
            max_retries=self.jitter_retries,
        )
        transformed = crop_from_roi(
            image=image_rgb,
            points=record.points,
            roi_xyxy=roi_xyxy,
            output_size=self.crop_size,
        )
        if not transformed["inside_crop"]:
            raise RuntimeError(f"Positive refiner crop lost a corner: annotation {record.annotation_id}")

        crop_rgb = transformed["crop"]
        crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        if self.augment and rng is not None:
            crop_gray = self._photometric_augment(crop_gray, rng)
        target_heatmaps = build_gaussian_heatmaps(
            transformed["crop_points"],
            crop_size=self.crop_size,
            heatmap_size=self.heatmap_size,
            sigma=self.heatmap_sigma,
        )

        return {
            "input": self._image_to_tensor(crop_gray),
            "target_heatmaps": torch.from_numpy(target_heatmaps),
            "target_points_crop": torch.from_numpy(transformed["crop_points"].astype(np.float32)),
            "gt_points_original": torch.from_numpy(record.points.astype(np.float32)),
            "roi_original_xyxy": torch.from_numpy(roi_xyxy.astype(np.float32)),
            "original_to_crop": torch.from_numpy(transformed["original_to_crop"].astype(np.float32)),
            "crop_to_original": torch.from_numpy(transformed["crop_to_original"].astype(np.float32)),
            "gt_bbox_diagonal": torch.as_tensor(_bbox_diagonal(record.points), dtype=torch.float32),
            "round_trip_error": torch.as_tensor(transformed["round_trip_error"], dtype=torch.float32),
            "uses_padding": torch.as_tensor(transformed["uses_padding"], dtype=torch.bool),
            "jitter_attempts": torch.as_tensor(jitter_attempts, dtype=torch.long),
            "jitter_fallback": torch.as_tensor(used_fallback, dtype=torch.bool),
            "dataset_index": int(record.dataset_index),
            "image_id": int(record.image_id),
            "annotation_id": int(record.annotation_id),
            "vertebra_rank": int(record.vertebra_rank),
            "file_name": record.file_name,
            "source_dataset": record.source_dataset,
        }


class SourceBalancedImageBatchSampler(Sampler[list[int]]):
    """Pack image-contiguous annotations while balancing expected source crop mass."""

    def __init__(
        self,
        dataset: CornerRefinerCocoDataset,
        batch_size: int = 16,
        seed: int = 20260627,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.image_ids = np.asarray(sorted(dataset.image_record_indices), dtype=np.int64)
        source_counts: dict[str, int] = defaultdict(int)
        for image_id, indices in dataset.image_record_indices.items():
            source_counts[dataset.image_sources[image_id]] += len(indices)
        weights = [1.0 / float(source_counts[dataset.image_sources[int(image_id)]]) for image_id in self.image_ids]
        self.image_probabilities = np.asarray(weights, dtype=np.float64)
        self.image_probabilities /= self.image_probabilities.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        target_count = len(self.dataset)
        sampled: list[int] = []
        while len(sampled) < target_count:
            image_id = int(rng.choice(self.image_ids, p=self.image_probabilities))
            indices = list(self.dataset.image_record_indices[image_id])
            rng.shuffle(indices)
            remaining = target_count - len(sampled)
            sampled.extend(indices[:remaining])

        for start in range(0, len(sampled), self.batch_size):
            batch = sampled[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
