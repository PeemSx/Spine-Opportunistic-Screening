from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from src.data.build_coco_nih import (
    CANONICAL_CATEGORY,
    CORNER_ORDER,
    NIH_SOURCE_KEY,
    NIH_SOURCE_NAME,
    SPLIT_RATIOS,
    SPLITS,
    canonical_source_key,
    category_corner_indices,
    largest_remainder_targets,
    load_json,
    normalize_keypoints,
    polygon_area,
    write_json_atomic,
)


LUMOS_SOURCE_NAME = "Lumos AP"
LUMOS_SOURCE_KEY = "lumos_ap"
DATASET_VERSION = "coco_nih_lumos"
DEFAULT_SEED = 20260810
MIN_ANNOTATION_AREA_FRACTION = 1e-5
NIH_ID_PATTERN = re.compile(r"(?P<patient>\d{8})_(?P<study>\d{3})", re.IGNORECASE)
LUMOS_ID_PATTERN = re.compile(r"lumos_AP_(?P<case>\d+)", re.IGNORECASE)
EXCLUDED_INCOMPLETE_IMAGES = {
    (LUMOS_SOURCE_KEY, "213"),
    (LUMOS_SOURCE_KEY, "274"),
    (LUMOS_SOURCE_KEY, "277"),
    (LUMOS_SOURCE_KEY, "278"),
}


@dataclass(frozen=True)
class IncrementRecord:
    identity: tuple[str, ...]
    source_key: str
    source_name: str
    group_id: str
    clean_name: str
    source_path: Path
    source_image: dict[str, Any]
    source_annotations: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class NormalizationResult:
    annotation: dict[str, Any] | None
    corner_order_repaired: bool
    exclusion_reason: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_identity(file_name: str) -> tuple[str, ...]:
    lumos_match = LUMOS_ID_PATTERN.search(str(file_name))
    if lumos_match is not None:
        return (LUMOS_SOURCE_KEY, str(int(lumos_match.group("case"))))
    nih_match = NIH_ID_PATTERN.search(str(file_name))
    if nih_match is not None:
        return (
            NIH_SOURCE_KEY,
            nih_match.group("patient"),
            nih_match.group("study"),
        )
    raise ValueError(f"Unsupported incremental image name: {file_name}")


def clean_image_name(identity: tuple[str, ...]) -> str:
    if identity[0] == NIH_SOURCE_KEY:
        return f"NIH_{identity[1]}_{identity[2]}.png"
    if identity[0] == LUMOS_SOURCE_KEY:
        return f"lumos_AP_{int(identity[1]):03d}.png"
    raise ValueError(f"Unsupported identity: {identity}")


def source_name_for_identity(identity: tuple[str, ...]) -> str:
    if identity[0] == NIH_SOURCE_KEY:
        return NIH_SOURCE_NAME
    if identity[0] == LUMOS_SOURCE_KEY:
        return LUMOS_SOURCE_NAME
    raise ValueError(f"Unsupported identity: {identity}")


def group_id_for_identity(identity: tuple[str, ...]) -> str:
    if identity[0] == NIH_SOURCE_KEY:
        return identity[1]
    if identity[0] == LUMOS_SOURCE_KEY:
        return identity[1]
    raise ValueError(f"Unsupported identity: {identity}")


def corners_have_canonical_semantics(keypoints: np.ndarray) -> bool:
    points = keypoints[:, :2]
    top_left, top_right, bottom_left, bottom_right = points
    return bool(
        top_left[0] < top_right[0]
        and bottom_left[0] < bottom_right[0]
        and (top_left[1] + top_right[1]) < (bottom_left[1] + bottom_right[1])
    )


def repair_corner_semantics(keypoints: np.ndarray) -> tuple[np.ndarray, bool]:
    repaired = np.asarray(keypoints, dtype=np.float64).copy()
    if corners_have_canonical_semantics(repaired):
        return repaired, False

    if repaired[0, 0] >= repaired[1, 0]:
        repaired[[0, 1]] = repaired[[1, 0]]
    if repaired[2, 0] >= repaired[3, 0]:
        repaired[[2, 3]] = repaired[[3, 2]]
    if repaired[0:2, 1].mean() >= repaired[2:4, 1].mean():
        repaired[[0, 1, 2, 3]] = repaired[[2, 3, 0, 1]]
    if repaired[0, 0] >= repaired[1, 0]:
        repaired[[0, 1]] = repaired[[1, 0]]
    if repaired[2, 0] >= repaired[3, 0]:
        repaired[[2, 3]] = repaired[[3, 2]]

    if not corners_have_canonical_semantics(repaired):
        raise ValueError(f"Could not repair TL/TR/BL/BR semantics: {repaired[:, :2].tolist()}")
    return repaired, True


def normalize_annotation(
    source_annotation: dict[str, Any],
    *,
    order_by_category: dict[int, tuple[int, ...]],
    annotation_id: int,
    image_id: int,
    image_width: int,
    image_height: int,
) -> NormalizationResult:
    keypoints = normalize_keypoints(source_annotation, order_by_category)
    keypoints, repaired = repair_corner_semantics(keypoints)
    if not (
        np.all((keypoints[:, 0] >= 0.0) & (keypoints[:, 0] < float(image_width)))
        and np.all((keypoints[:, 1] >= 0.0) & (keypoints[:, 1] < float(image_height)))
    ):
        raise ValueError(f"Annotation {source_annotation.get('id')} has out-of-bounds keypoints")

    points = keypoints[:, :2]
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    size = max_xy - min_xy
    if np.any(size <= 0.0):
        raise ValueError(f"Annotation {source_annotation.get('id')} has a degenerate box")
    area = polygon_area(points)
    image_area = float(image_width) * float(image_height)
    if area / max(image_area, 1.0) < MIN_ANNOTATION_AREA_FRACTION:
        return NormalizationResult(
            annotation=None,
            corner_order_repaired=repaired,
            exclusion_reason="tiny_annotation_below_1e-5_image_area",
        )

    polygon = points[[0, 1, 3, 2]].reshape(-1)
    normalized = dict(source_annotation)
    normalized.update(
        {
            "id": int(annotation_id),
            "image_id": int(image_id),
            "category_id": 1,
            "bbox": [
                round(float(min_xy[0]), 3),
                round(float(min_xy[1]), 3),
                round(float(size[0]), 3),
                round(float(size[1]), 3),
            ],
            "area": round(float(area), 3),
            "iscrowd": 0,
            "num_keypoints": 4,
            "keypoints": [
                value
                for x_coord, y_coord, visibility in keypoints
                for value in (
                    round(float(x_coord), 3),
                    round(float(y_coord), 3),
                    int(round(float(visibility))),
                )
            ],
            "segmentation": [[round(float(value), 3) for value in polygon]],
        }
    )
    normalized.setdefault("source_category_id", int(source_annotation["category_id"]))
    if repaired:
        normalized["corner_order_repaired"] = True
        normalized["corner_order_repair"] = "geometry_normalized_to_TL_TR_BL_BR"
    return NormalizationResult(
        annotation=normalized,
        corner_order_repaired=repaired,
        exclusion_reason=None,
    )


def annotation_signature(
    annotations: Iterable[dict[str, Any]],
    order_by_category: dict[int, tuple[int, ...]],
) -> tuple[tuple[float, ...], ...]:
    signatures = []
    for annotation in annotations:
        keypoints = normalize_keypoints(annotation, order_by_category)
        signatures.append(
            tuple(round(float(value), 3) for value in keypoints.reshape(-1))
        )
    return tuple(sorted(signatures))


def annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        grouped[int(annotation["image_id"])].append(annotation)
    return grouped


def base_identity_index(
    base_payloads: dict[str, dict[str, Any]],
    base_root: Path,
) -> tuple[
    dict[tuple[str, ...], tuple[str, dict[str, Any], tuple[dict[str, Any], ...], Path]],
    dict[str, str],
]:
    index: dict[
        tuple[str, ...], tuple[str, dict[str, Any], tuple[dict[str, Any], ...], Path]
    ] = {}
    patient_splits: dict[str, str] = {}
    for split, coco in base_payloads.items():
        grouped = annotations_by_image(coco)
        for image in coco.get("images", []):
            if canonical_source_key(image.get("source_dataset")) != NIH_SOURCE_KEY:
                continue
            identity = image_identity(str(image["file_name"]))
            if identity in index:
                raise ValueError(f"Duplicate base NIH identity: {identity}")
            patient_id = identity[1]
            previous_split = patient_splits.setdefault(patient_id, split)
            if previous_split != split:
                raise ValueError(f"Base NIH patient leakage for {patient_id}")
            index[identity] = (
                split,
                image,
                tuple(grouped.get(int(image["id"]), [])),
                base_root / split / str(image["file_name"]),
            )
    return index, patient_splits


def load_increment_records(
    increment_root: Path,
    base_index: dict[
        tuple[str, ...], tuple[str, dict[str, Any], tuple[dict[str, Any], ...], Path]
    ],
    base_payloads: dict[str, dict[str, Any]],
) -> tuple[
    list[IncrementRecord],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, tuple[int, ...]],
    dict[str, Any],
]:
    annotation_path = increment_root / "_annotations.coco.json"
    coco = load_json(annotation_path)
    order_by_category = category_corner_indices(coco.get("categories", []))
    if not order_by_category:
        raise ValueError(f"No valid four-corner categories found in {annotation_path}")
    grouped = annotations_by_image(coco)
    base_orders = {
        split: category_corner_indices(payload.get("categories", []))
        for split, payload in base_payloads.items()
    }

    records: list[IncrementRecord] = []
    duplicates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    identities: set[tuple[str, ...]] = set()
    for image in coco.get("images", []):
        image_id = int(image["id"])
        identity = image_identity(str(image["file_name"]))
        if identity in identities:
            raise ValueError(f"Duplicate incremental identity: {identity}")
        identities.add(identity)
        source_path = increment_root / str(image["file_name"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        decoded = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"Could not decode image: {source_path}")
        actual_height, actual_width = decoded.shape[:2]
        expected_width = int(image["width"])
        expected_height = int(image["height"])
        if (actual_width, actual_height) != (expected_width, expected_height):
            raise ValueError(
                f"Dimension mismatch for {source_path.name}: COCO {expected_width}x{expected_height}, "
                f"decoded {actual_width}x{actual_height}"
            )
        source_annotations = tuple(grouped.get(image_id, []))
        if not source_annotations:
            raise ValueError(f"Incremental image has no annotations: {source_path.name}")
        for annotation in source_annotations:
            normalize_keypoints(annotation, order_by_category)

        if identity in EXCLUDED_INCOMPLETE_IMAGES:
            excluded.append(
                {
                    "identity": list(identity),
                    "source_file_name": str(image["file_name"]),
                    "annotations": len(source_annotations),
                    "reason": "incomplete_vertebra_annotation_set_excluded_by_review",
                }
            )
            continue

        if identity in base_index:
            split, base_image, base_annotations, base_path = base_index[identity]
            if sha256_file(source_path) != sha256_file(base_path):
                raise ValueError(f"Duplicate identity has different image content: {identity}")
            incoming_signature = annotation_signature(source_annotations, order_by_category)
            existing_signature = annotation_signature(base_annotations, base_orders[split])
            if incoming_signature != existing_signature:
                raise ValueError(f"Duplicate identity has different annotations: {identity}")
            duplicates.append(
                {
                    "identity": list(identity),
                    "source_file_name": str(image["file_name"]),
                    "base_file_name": str(base_image["file_name"]),
                    "base_split": split,
                    "annotations": len(source_annotations),
                    "match": "exact_image_and_canonical_keypoints",
                }
            )
            continue

        source_key = identity[0]
        records.append(
            IncrementRecord(
                identity=identity,
                source_key=source_key,
                source_name=source_name_for_identity(identity),
                group_id=group_id_for_identity(identity),
                clean_name=clean_image_name(identity),
                source_path=source_path,
                source_image=dict(image),
                source_annotations=source_annotations,
            )
        )

    records.sort(key=lambda record: record.identity)
    duplicates.sort(key=lambda item: tuple(item["identity"]))
    excluded.sort(key=lambda item: tuple(item["identity"]))
    return records, duplicates, excluded, order_by_category, coco


def usable_annotation_count(
    record: IncrementRecord,
    order_by_category: dict[int, tuple[int, ...]],
) -> int:
    count = 0
    for annotation in record.source_annotations:
        result = normalize_annotation(
            annotation,
            order_by_category=order_by_category,
            annotation_id=int(annotation["id"]),
            image_id=int(record.source_image["id"]),
            image_width=int(record.source_image["width"]),
            image_height=int(record.source_image["height"]),
        )
        count += int(result.annotation is not None)
    return count


def _greedy_exact_group_subset(
    groups: list[tuple[str, list[IncrementRecord]]],
    target_images: int,
    rng: random.Random,
) -> set[str] | None:
    order = list(groups)
    rng.shuffle(order)
    selected: set[str] = set()
    image_count = 0
    for group_id, records in order:
        group_size = len(records)
        if image_count + group_size <= target_images:
            selected.add(group_id)
            image_count += group_size
        if image_count == target_images:
            return selected
    return None


def assign_source_records(
    records: list[IncrementRecord],
    *,
    existing_image_counts: dict[str, int],
    existing_annotation_counts: dict[str, int],
    existing_group_splits: dict[str, str],
    order_by_category: dict[int, tuple[int, ...]],
    seed: int,
    search_trials: int = 5000,
) -> tuple[dict[str, list[IncrementRecord]], dict[str, int], dict[str, int]]:
    total_images = sum(existing_image_counts.values()) + len(records)
    total_annotations = sum(existing_annotation_counts.values()) + sum(
        usable_annotation_count(record, order_by_category) for record in records
    )
    image_targets = largest_remainder_targets(total_images, SPLIT_RATIOS)
    annotation_targets = largest_remainder_targets(total_annotations, SPLIT_RATIOS)
    addition_targets = {
        split: image_targets[split] - int(existing_image_counts.get(split, 0))
        for split in SPLITS
    }
    if any(target < 0 for target in addition_targets.values()):
        raise ValueError(f"Preserved base exceeds final source targets: {addition_targets}")
    if sum(addition_targets.values()) != len(records):
        raise RuntimeError("Incremental image targets do not sum to the record count")

    grouped: dict[str, list[IncrementRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    fixed_assignment: dict[str, str] = {
        group_id: existing_group_splits[group_id]
        for group_id in grouped
        if group_id in existing_group_splits
    }
    fixed_counts = Counter()
    for group_id, split in fixed_assignment.items():
        fixed_counts[split] += len(grouped[group_id])
    remaining_targets = {
        split: addition_targets[split] - fixed_counts[split]
        for split in SPLITS
    }
    if any(target < 0 for target in remaining_targets.values()):
        raise ValueError(f"Existing-group additions exceed a split target: {remaining_targets}")

    remaining_groups = sorted(
        (group_id, group_records)
        for group_id, group_records in grouped.items()
        if group_id not in fixed_assignment
    )
    rng = random.Random(int(seed))
    best_assignment: dict[str, str] | None = None
    best_score: tuple[Any, ...] | None = None
    annotation_count_by_record = {
        record.identity: usable_annotation_count(record, order_by_category)
        for record in records
    }
    for _ in range(max(int(search_trials), 1)):
        test_groups = _greedy_exact_group_subset(
            remaining_groups, remaining_targets["test"], rng
        )
        if test_groups is None:
            continue
        after_test = [item for item in remaining_groups if item[0] not in test_groups]
        val_groups = _greedy_exact_group_subset(
            after_test, remaining_targets["val"], rng
        )
        if val_groups is None:
            continue
        assignment = dict(fixed_assignment)
        for group_id, _ in remaining_groups:
            assignment[group_id] = (
                "test" if group_id in test_groups else "val" if group_id in val_groups else "train"
            )
        image_counts = Counter()
        annotation_counts = Counter(existing_annotation_counts)
        for record in records:
            split = assignment[record.group_id]
            image_counts[split] += 1
            annotation_counts[split] += annotation_count_by_record[record.identity]
        if any(image_counts[split] != addition_targets[split] for split in SPLITS):
            continue
        annotation_error = sum(
            abs(annotation_counts[split] - annotation_targets[split])
            / max(float(annotation_targets[split]), 1.0)
            for split in SPLITS
        )
        tie_breaker = (
            tuple(sorted(test_groups)),
            tuple(sorted(val_groups)),
        )
        score: tuple[Any, ...] = (annotation_error, *tie_breaker)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError(f"Could not find exact grouped split targets: {remaining_targets}")

    split_records = {split: [] for split in SPLITS}
    for record in records:
        split_records[best_assignment[record.group_id]].append(record)
    for split in SPLITS:
        split_records[split].sort(key=lambda record: record.identity)
    return split_records, image_targets, annotation_targets


def base_source_counts(
    base_payloads: dict[str, dict[str, Any]],
    source_key: str,
) -> tuple[dict[str, int], dict[str, int]]:
    image_counts: dict[str, int] = {}
    annotation_counts: dict[str, int] = {}
    for split, coco in base_payloads.items():
        grouped = annotations_by_image(coco)
        order_by_category = category_corner_indices(coco.get("categories", []))
        source_images = [
            image
            for image in coco.get("images", [])
            if canonical_source_key(image.get("source_dataset")) == source_key
        ]
        image_counts[split] = len(source_images)
        usable_count = 0
        for image in source_images:
            for annotation in grouped.get(int(image["id"]), []):
                result = normalize_annotation(
                    annotation,
                    order_by_category=order_by_category,
                    annotation_id=int(annotation["id"]),
                    image_id=int(image["id"]),
                    image_width=int(image["width"]),
                    image_height=int(image["height"]),
                )
                usable_count += int(result.annotation is not None)
        annotation_counts[split] = usable_count
    return image_counts, annotation_counts


def cleanup_base_payload(
    coco: dict[str, Any],
    *,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    order_by_category = category_corner_indices(coco.get("categories", []))
    if not order_by_category:
        raise ValueError(f"Base split {split} has no valid four-corner category")
    images = [dict(image) for image in coco.get("images", [])]
    image_by_id = {int(image["id"]): image for image in images}
    annotations: list[dict[str, Any]] = []
    repaired_events: list[dict[str, Any]] = []
    excluded_events: list[dict[str, Any]] = []
    for source_annotation in coco.get("annotations", []):
        image_id = int(source_annotation["image_id"])
        image = image_by_id[image_id]
        result = normalize_annotation(
            source_annotation,
            order_by_category=order_by_category,
            annotation_id=int(source_annotation["id"]),
            image_id=image_id,
            image_width=int(image["width"]),
            image_height=int(image["height"]),
        )
        event = {
            "dataset_role": "base_checkpoint",
            "split": split,
            "file_name": str(image["file_name"]),
            "annotation_id": int(source_annotation["id"]),
        }
        if result.annotation is None:
            excluded_events.append({**event, "reason": result.exclusion_reason})
            continue
        annotations.append(result.annotation)
        if result.corner_order_repaired:
            repaired_events.append(event)
    payload = {
        "info": dict(coco.get("info") or {}),
        "licenses": list(coco.get("licenses") or []),
        "categories": [dict(CANONICAL_CATEGORY)],
        "images": images,
        "annotations": annotations,
    }
    return payload, repaired_events, excluded_events


def append_records_to_split(
    coco: dict[str, Any],
    *,
    split: str,
    records: list[IncrementRecord],
    target_split_dir: Path,
    order_by_category: dict[int, tuple[int, ...]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    next_image_id = max((int(image["id"]) for image in coco["images"]), default=0) + 1
    next_annotation_id = max(
        (int(annotation["id"]) for annotation in coco["annotations"]), default=0
    ) + 1
    repaired_events: list[dict[str, Any]] = []
    excluded_events: list[dict[str, Any]] = []
    for record in records:
        image_id = next_image_id
        next_image_id += 1
        target_dir = target_split_dir / "images" / record.source_key
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / record.clean_name
        if target_path.exists():
            raise FileExistsError(target_path)
        shutil.copy2(record.source_path, target_path)
        width = int(record.source_image["width"])
        height = int(record.source_image["height"])
        image_item: dict[str, Any] = {
            "id": image_id,
            "file_name": f"images/{record.source_key}/{record.clean_name}",
            "width": width,
            "height": height,
            "source_dataset": record.source_name,
            "source_image_id": int(record.source_image["id"]),
            "source_file_name": str(record.source_image["file_name"]),
        }
        if record.source_key == NIH_SOURCE_KEY:
            image_item.update(
                {
                    "nih_patient_id": record.identity[1],
                    "nih_study_id": record.identity[2],
                }
            )
        elif record.source_key == LUMOS_SOURCE_KEY:
            image_item["lumos_case_id"] = record.identity[1]
        coco["images"].append(image_item)

        normalized_annotations: list[dict[str, Any]] = []
        for source_annotation in record.source_annotations:
            result = normalize_annotation(
                source_annotation,
                order_by_category=order_by_category,
                annotation_id=next_annotation_id,
                image_id=image_id,
                image_width=width,
                image_height=height,
            )
            event = {
                "dataset_role": "increment",
                "split": split,
                "file_name": image_item["file_name"],
                "source_file_name": str(record.source_image["file_name"]),
                "source_annotation_id": int(source_annotation["id"]),
            }
            if result.annotation is None:
                excluded_events.append({**event, "reason": result.exclusion_reason})
                continue
            result.annotation["source_dataset"] = record.source_name
            result.annotation["source_annotation_id"] = int(source_annotation["id"])
            normalized_annotations.append(result.annotation)
            if result.corner_order_repaired:
                repaired_events.append(event)
            next_annotation_id += 1
        normalized_annotations.sort(
            key=lambda annotation: (
                float(np.asarray(annotation["keypoints"]).reshape(4, 3)[:, 1].mean()),
                float(np.asarray(annotation["keypoints"]).reshape(4, 3)[:, 0].mean()),
                int(annotation["id"]),
            )
        )
        for vertebra_index, annotation in enumerate(normalized_annotations, start=1):
            annotation["vertebra_index"] = vertebra_index
            coco["annotations"].append(annotation)
    return repaired_events, excluded_events


def validate_split(coco: dict[str, Any], split_dir: Path) -> None:
    if coco.get("categories") != [CANONICAL_CATEGORY]:
        raise ValueError(f"Noncanonical category schema in {split_dir}")
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    image_ids = {int(image["id"]) for image in images}
    annotation_ids = {int(annotation["id"]) for annotation in annotations}
    if len(image_ids) != len(images):
        raise ValueError(f"Duplicate image IDs in {split_dir}")
    if len(annotation_ids) != len(annotations):
        raise ValueError(f"Duplicate annotation IDs in {split_dir}")
    image_by_id = {int(image["id"]): image for image in images}
    counts = Counter(int(annotation["image_id"]) for annotation in annotations)
    for image in images:
        image_path = split_dir / str(image["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if counts[int(image["id"])] == 0:
            raise ValueError(f"Zero-annotation image in {split_dir}: {image['file_name']}")
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in image_ids:
            raise ValueError(f"Annotation {annotation.get('id')} references a missing image")
        if int(annotation.get("category_id", -1)) != 1:
            raise ValueError(f"Annotation {annotation.get('id')} has a noncanonical category")
        keypoints = np.asarray(annotation.get("keypoints", []), dtype=np.float64)
        if keypoints.size != 12 or not np.isfinite(keypoints).all():
            raise ValueError(f"Annotation {annotation.get('id')} has invalid keypoints")
        keypoints = keypoints.reshape(4, 3)
        if not np.all(keypoints[:, 2] > 0) or not corners_have_canonical_semantics(keypoints):
            raise ValueError(f"Annotation {annotation.get('id')} violates TL/TR/BL/BR semantics")
        image = image_by_id[image_id]
        if not (
            np.all((keypoints[:, 0] >= 0.0) & (keypoints[:, 0] < float(image["width"])))
            and np.all((keypoints[:, 1] >= 0.0) & (keypoints[:, 1] < float(image["height"])))
        ):
            raise ValueError(f"Annotation {annotation.get('id')} is out of bounds")
        points = keypoints[:, :2]
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        expected_bbox = np.concatenate((min_xy, max_xy - min_xy))
        if not np.allclose(annotation["bbox"], expected_bbox, atol=0.002, rtol=0.0):
            raise ValueError(f"Annotation {annotation.get('id')} bbox is not keypoint-derived")
        expected_area = polygon_area(points)
        if not math.isclose(float(annotation["area"]), expected_area, abs_tol=0.01, rel_tol=0.0):
            raise ValueError(f"Annotation {annotation.get('id')} area is not keypoint-derived")
        if expected_area / (float(image["width"]) * float(image["height"])) < MIN_ANNOTATION_AREA_FRACTION:
            raise ValueError(f"Annotation {annotation.get('id')} is below the minimum area")


def validate_cross_split_constraints(payloads: dict[str, dict[str, Any]]) -> None:
    identities: dict[tuple[str, ...], str] = {}
    nih_patient_splits: dict[str, str] = {}
    lumos_case_splits: dict[str, str] = {}
    for split, coco in payloads.items():
        for image in coco.get("images", []):
            source_key = canonical_source_key(image.get("source_dataset"))
            if source_key not in {NIH_SOURCE_KEY, LUMOS_SOURCE_KEY}:
                continue
            identity = image_identity(str(image["file_name"]))
            previous = identities.setdefault(identity, split)
            if previous != split:
                raise ValueError(f"Image identity leakage for {identity}: {previous}, {split}")
            if source_key == NIH_SOURCE_KEY:
                patient_id = identity[1]
                previous = nih_patient_splits.setdefault(patient_id, split)
                if previous != split:
                    raise ValueError(f"NIH patient leakage for {patient_id}: {previous}, {split}")
            else:
                case_id = identity[1]
                previous = lumos_case_splits.setdefault(case_id, split)
                if previous != split:
                    raise ValueError(f"Lumos case leakage for {case_id}: {previous}, {split}")


def dataset_summary(
    *,
    staging_root: Path,
    seed: int,
    base_root: Path,
    increment_root: Path,
    raw_coco: dict[str, Any],
    records: list[IncrementRecord],
    duplicates: list[dict[str, Any]],
    excluded_images: list[dict[str, Any]],
    retained_base_identities: list[tuple[str, ...]],
    repaired_events: list[dict[str, Any]],
    excluded_annotations: list[dict[str, Any]],
    source_image_targets: dict[str, dict[str, int]],
    source_annotation_targets: dict[str, dict[str, int]],
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    source_summaries: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for split in SPLITS:
        coco = load_json(staging_root / split / "_annotations.keypoints.coco.json")
        grouped = Counter(int(annotation["image_id"]) for annotation in coco["annotations"])
        images_by_source = Counter()
        annotations_by_source = Counter()
        groups_by_source: dict[str, set[str]] = defaultdict(set)
        for image in coco["images"]:
            source = canonical_source_key(image.get("source_dataset"))
            images_by_source[source] += 1
            annotations_by_source[source] += grouped[int(image["id"])]
            if source in {NIH_SOURCE_KEY, LUMOS_SOURCE_KEY}:
                groups_by_source[source].add(group_id_for_identity(image_identity(image["file_name"])))
        split_summaries[split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "images_by_source": dict(sorted(images_by_source.items())),
            "annotations_by_source": dict(sorted(annotations_by_source.items())),
            "groups_by_source": {
                source: len(groups) for source, groups in sorted(groups_by_source.items())
            },
        }
        for source in sorted(images_by_source):
            source_summaries[source][split] = {
                "images": images_by_source[source],
                "annotations": annotations_by_source[source],
            }

    input_category_counts = Counter(
        int(annotation["category_id"]) for annotation in raw_coco.get("annotations", [])
    )
    return {
        "version": DATASET_VERSION,
        "seed": int(seed),
        "ratios": dict(SPLIT_RATIOS),
        "base_dataset_root": base_root.as_posix(),
        "increment_source_root": increment_root.as_posix(),
        "increment_source_annotation_file": (increment_root / "_annotations.coco.json").as_posix(),
        "corner_order": list(CORNER_ORDER),
        "category_normalization": {
            "input_categories": raw_coco.get("categories", []),
            "input_annotation_counts_by_category": {
                str(category_id): count for category_id, count in sorted(input_category_counts.items())
            },
            "output_category": dict(CANONICAL_CATEGORY),
            "policy": "all valid source labels normalized to one vertebra category and TL/TR/BL/BR",
        },
        "split_policy": {
            "base_assignments": "preserved",
            "nih": "patient_grouped_deterministic_with_existing_patient_assignments_preserved",
            "lumos": "case_grouped_deterministic",
        },
        "increment_audit": {
            "raw_images": len(raw_coco.get("images", [])),
            "raw_annotations": len(raw_coco.get("annotations", [])),
            "exact_duplicate_images_skipped": len(duplicates),
            "new_images_before_review_exclusions": len(records) + len(excluded_images),
            "new_images_added": len(records),
            "new_images_added_by_source": dict(Counter(record.source_key for record in records)),
            "retained_base_images_absent_from_increment": [
                list(identity) for identity in sorted(retained_base_identities)
            ],
            "excluded_incomplete_images": excluded_images,
        },
        "cleanup": {
            "corner_order_repairs": repaired_events,
            "corner_order_repair_count": len(repaired_events),
            "excluded_tiny_annotations": excluded_annotations,
            "excluded_tiny_annotation_count": len(excluded_annotations),
            "bbox_area_policy": "regenerated_from_normalized_keypoints_for_every_annotation",
        },
        "source_image_targets": source_image_targets,
        "source_annotation_targets_for_balancing": source_annotation_targets,
        "splits": split_summaries,
        "sources": dict(sorted(source_summaries.items())),
    }


def build_dataset(
    *,
    base_root: Path,
    increment_root: Path,
    output_root: Path,
    seed: int,
) -> dict[str, Any]:
    base_root = base_root.resolve()
    increment_root = increment_root.resolve()
    output_root = output_root.resolve()
    if not base_root.is_dir():
        raise FileNotFoundError(base_root)
    if not increment_root.is_dir():
        raise FileNotFoundError(increment_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    staging_root = output_root.with_name(output_root.name + ".building")
    if staging_root.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_root}")

    base_payloads = {
        split: load_json(base_root / split / "_annotations.keypoints.coco.json")
        for split in SPLITS
    }
    base_index, existing_nih_patient_splits = base_identity_index(base_payloads, base_root)
    records, duplicates, excluded_images, increment_orders, raw_coco = load_increment_records(
        increment_root,
        base_index,
        base_payloads,
    )
    incoming_identities = {
        image_identity(str(image["file_name"])) for image in raw_coco.get("images", [])
    }
    retained_base_identities = sorted(set(base_index) - incoming_identities)

    records_by_source: dict[str, list[IncrementRecord]] = defaultdict(list)
    for record in records:
        records_by_source[record.source_key].append(record)
    assigned_records = {split: [] for split in SPLITS}
    source_image_targets: dict[str, dict[str, int]] = {}
    source_annotation_targets: dict[str, dict[str, int]] = {}
    for source_key in (NIH_SOURCE_KEY, LUMOS_SOURCE_KEY):
        source_records = records_by_source.get(source_key, [])
        existing_images, existing_annotations = base_source_counts(base_payloads, source_key)
        existing_group_splits = existing_nih_patient_splits if source_key == NIH_SOURCE_KEY else {}
        split_records, image_targets, annotation_targets = assign_source_records(
            source_records,
            existing_image_counts=existing_images,
            existing_annotation_counts=existing_annotations,
            existing_group_splits=existing_group_splits,
            order_by_category=increment_orders,
            seed=int(seed) + (0 if source_key == NIH_SOURCE_KEY else 1),
        )
        source_image_targets[source_key] = image_targets
        source_annotation_targets[source_key] = annotation_targets
        for split in SPLITS:
            assigned_records[split].extend(split_records[split])
            assigned_records[split].sort(key=lambda record: record.identity)

    repaired_events: list[dict[str, Any]] = []
    excluded_annotations: list[dict[str, Any]] = []
    final_payloads: dict[str, dict[str, Any]] = {}
    staging_root.mkdir(parents=True)
    try:
        for split in SPLITS:
            shutil.copytree(base_root / split, staging_root / split)
            payload, base_repairs, base_exclusions = cleanup_base_payload(
                base_payloads[split], split=split
            )
            repaired_events.extend(base_repairs)
            excluded_annotations.extend(base_exclusions)
            increment_repairs, increment_exclusions = append_records_to_split(
                payload,
                split=split,
                records=assigned_records[split],
                target_split_dir=staging_root / split,
                order_by_category=increment_orders,
            )
            repaired_events.extend(increment_repairs)
            excluded_annotations.extend(increment_exclusions)
            info = dict(payload.get("info") or {})
            info.update(
                {
                    "description": "Merged spine vertebra keypoint COCO dataset with COCO, NIH, and Lumos AP",
                    "version": DATASET_VERSION,
                    "split": split,
                    "split_seed": int(seed),
                    "split_ratios": dict(SPLIT_RATIOS),
                    "base_dataset_root": base_root.as_posix(),
                    "increment_source_annotation_file": (
                        increment_root / "_annotations.coco.json"
                    ).as_posix(),
                    "corner_order": list(CORNER_ORDER),
                    "category_name": CANONICAL_CATEGORY["name"],
                    "category_policy": "single canonical vertebra category",
                }
            )
            payload["info"] = info
            validate_split(payload, staging_root / split)
            write_json_atomic(
                staging_root / split / "_annotations.keypoints.coco.json", payload
            )
            final_payloads[split] = payload

        validate_cross_split_constraints(final_payloads)
        summary = dataset_summary(
            staging_root=staging_root,
            seed=seed,
            base_root=base_root,
            increment_root=increment_root,
            raw_coco=raw_coco,
            records=records,
            duplicates=duplicates,
            excluded_images=excluded_images,
            retained_base_identities=retained_base_identities,
            repaired_events=sorted(
                repaired_events,
                key=lambda item: (
                    str(item.get("split")),
                    str(item.get("file_name")),
                    int(item.get("annotation_id", item.get("source_annotation_id", -1))),
                ),
            ),
            excluded_annotations=sorted(
                excluded_annotations,
                key=lambda item: (
                    str(item.get("split")),
                    str(item.get("file_name")),
                    int(item.get("annotation_id", item.get("source_annotation_id", -1))),
                ),
            ),
            source_image_targets=source_image_targets,
            source_annotation_targets=source_annotation_targets,
        )
        write_json_atomic(staging_root / "split_summary.json", summary)
        if output_root.exists():
            output_root.rmdir()
        staging_root.replace(output_root)
        return summary
    except Exception:
        # Retain partial staging output for forensic inspection.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend coco_nih with a reviewed Roboflow export while preserving checkpoint splits."
    )
    parser.add_argument("--base-root", type=Path, default=Path("dataset/processed/coco_nih"))
    parser.add_argument(
        "--increment-root", type=Path, default=Path("dataset/raw/Roboflow/10 AUG 2026")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("dataset/processed/coco_nih_lumos")
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        base_root=args.base_root,
        increment_root=args.increment_root,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(f"saved: {args.output_root}")
    audit = summary["increment_audit"]
    print(f"duplicates skipped: {audit['exact_duplicate_images_skipped']}")
    print(f"new images added: {audit['new_images_added']}")
    print(f"incomplete images excluded: {len(audit['excluded_incomplete_images'])}")
    for split in SPLITS:
        item = summary["splits"][split]
        print(f"{split}: {item['images']} images / {item['annotations']} annotations")


if __name__ == "__main__":
    main()
