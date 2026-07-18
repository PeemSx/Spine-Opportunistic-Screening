from __future__ import annotations

import argparse
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


SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
CORNER_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
CORNER_ALIASES = {
    "tl": "top_left",
    "top_left": "top_left",
    "tr": "top_right",
    "top_right": "top_right",
    "bl": "bottom_left",
    "bottom_left": "bottom_left",
    "br": "bottom_right",
    "bottom_right": "bottom_right",
}
CANONICAL_CATEGORY = {
    "id": 1,
    "name": "vertebra",
    "supercategory": "spine",
    "keypoints": list(CORNER_ORDER),
    "skeleton": [[1, 2], [3, 4], [1, 3], [2, 4]],
}
NIH_SOURCE_NAME = "NIH ChestX-ray14"
NIH_SOURCE_KEY = "nih_chestxray14"
NIH_ID_PATTERN = re.compile(r"(?P<patient>\d{8})_(?P<study>\d{3})", re.IGNORECASE)


@dataclass(frozen=True)
class NihRecord:
    patient_id: str
    study_id: str
    clean_name: str
    source_path: Path
    source_image: dict[str, Any]
    source_annotations: tuple[dict[str, Any], ...]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    temporary_path.replace(path)


def canonical_source_key(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if "buu" in normalized:
        return "buu_ap"
    if "mendeley" in normalized:
        return "mendeley_pa"
    if "miccai" in normalized:
        return "miccai_2019"
    if "nih" in normalized or "chestxray" in normalized:
        return NIH_SOURCE_KEY
    return normalized


def largest_remainder_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: float(total) * float(ratios[split]) for split in SPLITS}
    targets = {split: int(math.floor(raw[split])) for split in SPLITS}
    remainder = int(total) - sum(targets.values())
    priority = {split: index for index, split in enumerate(SPLITS)}
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - targets[split]), priority[split]))
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def category_corner_indices(categories: Iterable[dict[str, Any]]) -> dict[int, tuple[int, ...]]:
    indices_by_category: dict[int, tuple[int, ...]] = {}
    for category in categories:
        names = category.get("keypoints") or []
        normalized = [CORNER_ALIASES.get(str(name).strip().lower()) for name in names]
        if len(normalized) != 4 or set(normalized) != set(CORNER_ORDER):
            continue
        indices_by_category[int(category["id"])] = tuple(normalized.index(name) for name in CORNER_ORDER)
    return indices_by_category


def normalize_keypoints(
    annotation: dict[str, Any],
    order_by_category: dict[int, tuple[int, ...]],
) -> np.ndarray:
    category_id = int(annotation["category_id"])
    if category_id not in order_by_category:
        raise ValueError(
            f"Annotation {annotation.get('id')} uses category {category_id} without a valid four-corner schema"
        )
    keypoints = np.asarray(annotation.get("keypoints", []), dtype=np.float64)
    if keypoints.size != 12:
        raise ValueError(f"Annotation {annotation.get('id')} does not contain four COCO keypoints")
    keypoints = keypoints.reshape(4, 3)[list(order_by_category[category_id])]
    if not np.isfinite(keypoints).all():
        raise ValueError(f"Annotation {annotation.get('id')} contains non-finite keypoints")
    if not np.all(keypoints[:, 2] > 0):
        raise ValueError(f"Annotation {annotation.get('id')} contains absent corner landmarks")
    return keypoints


def polygon_area(points: np.ndarray) -> float:
    polygon = points[[0, 1, 3, 2]]
    x_coord = polygon[:, 0]
    y_coord = polygon[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x_coord, np.roll(y_coord, -1))
            - np.dot(y_coord, np.roll(x_coord, -1))
        )
    )


def normalized_annotation(
    source_annotation: dict[str, Any],
    *,
    annotation_id: int,
    image_id: int,
    vertebra_index: int,
    image_width: int,
    image_height: int,
    order_by_category: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    keypoints = normalize_keypoints(source_annotation, order_by_category)
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
        raise ValueError(f"Annotation {source_annotation.get('id')} has a degenerate vertebral box")
    polygon = points[[0, 1, 3, 2]].reshape(-1)
    return {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": 1,
        "bbox": [
            round(float(min_xy[0]), 3),
            round(float(min_xy[1]), 3),
            round(float(size[0]), 3),
            round(float(size[1]), 3),
        ],
        "area": round(polygon_area(points), 3),
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
        "vertebra_index": int(vertebra_index),
        "source_dataset": NIH_SOURCE_NAME,
        "source_annotation_id": int(source_annotation.get("id", -1)),
        "source_category_id": int(source_annotation["category_id"]),
    }


def extract_nih_identity(image: dict[str, Any]) -> tuple[str, str, str]:
    candidates = [
        str((image.get("extra") or {}).get("name") or ""),
        str(image.get("file_name") or ""),
    ]
    match = next((NIH_ID_PATTERN.search(value) for value in candidates if NIH_ID_PATTERN.search(value)), None)
    if match is None:
        raise ValueError(f"Could not extract NIH patient/study ID from image {image.get('id')}: {candidates}")
    patient_id = match.group("patient")
    study_id = match.group("study")
    return patient_id, study_id, f"NIH_{patient_id}_{study_id}.png"


def load_nih_records(
    nih_root: Path,
) -> tuple[list[NihRecord], list[dict[str, Any]], dict[int, tuple[int, ...]]]:
    annotation_path = nih_root / "_annotations.coco.json"
    coco = load_json(annotation_path)
    order_by_category = category_corner_indices(coco.get("categories", []))
    if not order_by_category:
        raise ValueError(f"No valid four-corner categories found in {annotation_path}")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    records: list[NihRecord] = []
    excluded: list[dict[str, Any]] = []
    clean_names: set[str] = set()
    for image in coco.get("images", []):
        image_id = int(image["id"])
        source_path = nih_root / str(image["file_name"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        decoded = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"Could not decode NIH image: {source_path}")
        actual_height, actual_width = decoded.shape[:2]
        expected_width = int(image["width"])
        expected_height = int(image["height"])
        if (actual_width, actual_height) != (expected_width, expected_height):
            raise ValueError(
                f"NIH dimension mismatch for {source_path.name}: COCO {expected_width}x{expected_height}, "
                f"decoded {actual_width}x{actual_height}"
            )

        patient_id, study_id, clean_name = extract_nih_identity(image)
        if clean_name.lower() in clean_names:
            raise ValueError(f"Duplicate normalized NIH filename: {clean_name}")
        clean_names.add(clean_name.lower())
        source_annotations = annotations_by_image.get(image_id, [])
        if not source_annotations:
            excluded.append(
                {
                    "source_image_id": image_id,
                    "source_file_name": str(image["file_name"]),
                    "normalized_file_name": clean_name,
                    "reason": "zero_annotations_excluded_to_avoid_false_negative_training",
                }
            )
            continue

        for annotation in source_annotations:
            normalize_keypoints(annotation, order_by_category)
        records.append(
            NihRecord(
                patient_id=patient_id,
                study_id=study_id,
                clean_name=clean_name,
                source_path=source_path,
                source_image=dict(image),
                source_annotations=tuple(dict(annotation) for annotation in source_annotations),
            )
        )
    if not records:
        raise ValueError(f"No usable annotated NIH images found in {nih_root}")
    return records, excluded, order_by_category


def _greedy_exact_subset(
    groups: list[tuple[str, list[NihRecord]]],
    target_images: int,
    rng: random.Random,
) -> set[str] | None:
    order = list(groups)
    rng.shuffle(order)
    chosen: set[str] = set()
    image_count = 0
    for patient_id, records in order:
        group_size = len(records)
        if image_count + group_size <= target_images:
            chosen.add(patient_id)
            image_count += group_size
        if image_count == target_images:
            return chosen
    return None


def split_nih_by_patient(
    records: list[NihRecord],
    *,
    seed: int,
    ratios: dict[str, float] | None = None,
    search_trials: int = 5000,
) -> tuple[dict[str, list[NihRecord]], dict[str, int], dict[str, int]]:
    ratios = dict(ratios or SPLIT_RATIOS)
    grouped: dict[str, list[NihRecord]] = defaultdict(list)
    for record in records:
        grouped[record.patient_id].append(record)
    groups = sorted(grouped.items())

    image_targets = largest_remainder_targets(len(records), ratios)
    annotation_targets = largest_remainder_targets(
        sum(len(record.source_annotations) for record in records), ratios
    )
    patient_targets = largest_remainder_targets(len(groups), ratios)
    rng = random.Random(int(seed))
    best_assignment: dict[str, str] | None = None
    best_score: tuple[float, tuple[str, ...], tuple[str, ...]] | None = None

    for _ in range(max(int(search_trials), 1)):
        test_patients = _greedy_exact_subset(groups, image_targets["test"], rng)
        if test_patients is None:
            continue
        remaining = [item for item in groups if item[0] not in test_patients]
        val_patients = _greedy_exact_subset(remaining, image_targets["val"], rng)
        if val_patients is None:
            continue

        assignment = {
            patient_id: (
                "test" if patient_id in test_patients else "val" if patient_id in val_patients else "train"
            )
            for patient_id in grouped
        }
        annotation_counts = Counter()
        patient_counts = Counter(assignment.values())
        for record in records:
            annotation_counts[assignment[record.patient_id]] += len(record.source_annotations)
        annotation_error = sum(
            abs(annotation_counts[split] - annotation_targets[split])
            / max(float(annotation_targets[split]), 1.0)
            for split in SPLITS
        )
        patient_error = sum(
            abs(patient_counts[split] - patient_targets[split])
            / max(float(patient_targets[split]), 1.0)
            for split in SPLITS
        )
        tie_breaker = (tuple(sorted(test_patients)), tuple(sorted(val_patients)))
        score = (annotation_error + 0.25 * patient_error, *tie_breaker)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not find an exact patient-grouped NIH split")

    split_records = {split: [] for split in SPLITS}
    for record in records:
        split_records[best_assignment[record.patient_id]].append(record)
    for split in SPLITS:
        split_records[split].sort(key=lambda record: record.clean_name.lower())
        if len(split_records[split]) != image_targets[split]:
            raise RuntimeError(f"NIH split {split} missed target image count")

    patient_sets = {
        split: {record.patient_id for record in split_records[split]}
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if patient_sets[left] & patient_sets[right]:
                raise RuntimeError(f"NIH patient leakage detected between {left} and {right}")
    return split_records, image_targets, annotation_targets


def copy_base_split(base_root: Path, staging_root: Path, split: str) -> dict[str, Any]:
    source_split = base_root / split
    target_split = staging_root / split
    source_images = source_split / "images"
    if not source_images.is_dir():
        raise FileNotFoundError(source_images)
    shutil.copytree(source_images, target_split / "images")
    return load_json(source_split / "_annotations.keypoints.coco.json")


def build_split_payload(
    *,
    base_coco: dict[str, Any],
    split: str,
    records: list[NihRecord],
    target_split_dir: Path,
    order_by_category: dict[int, tuple[int, ...]],
    seed: int,
    nih_source_annotation_file: Path,
) -> dict[str, Any]:
    images = [dict(image) for image in base_coco.get("images", [])]
    annotations = [dict(annotation) for annotation in base_coco.get("annotations", [])]
    next_image_id = max((int(image["id"]) for image in images), default=0) + 1
    next_annotation_id = max((int(annotation["id"]) for annotation in annotations), default=0) + 1
    nih_image_dir = target_split_dir / "images" / NIH_SOURCE_KEY
    nih_image_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        image_id = next_image_id
        next_image_id += 1
        target_image_path = nih_image_dir / record.clean_name
        shutil.copy2(record.source_path, target_image_path)
        width = int(record.source_image["width"])
        height = int(record.source_image["height"])
        images.append(
            {
                "id": image_id,
                "file_name": f"images/{NIH_SOURCE_KEY}/{record.clean_name}",
                "width": width,
                "height": height,
                "source_dataset": NIH_SOURCE_NAME,
                "source_image_id": int(record.source_image["id"]),
                "source_file_name": str(record.source_image["file_name"]),
                "nih_patient_id": record.patient_id,
                "nih_study_id": record.study_id,
            }
        )

        sorted_annotations = sorted(
            record.source_annotations,
            key=lambda annotation: (
                float(normalize_keypoints(annotation, order_by_category)[:, 1].mean()),
                float(normalize_keypoints(annotation, order_by_category)[:, 0].mean()),
                int(annotation.get("id", -1)),
            ),
        )
        for vertebra_index, source_annotation in enumerate(sorted_annotations, start=1):
            annotations.append(
                normalized_annotation(
                    source_annotation,
                    annotation_id=next_annotation_id,
                    image_id=image_id,
                    vertebra_index=vertebra_index,
                    image_width=width,
                    image_height=height,
                    order_by_category=order_by_category,
                )
            )
            next_annotation_id += 1

    info = dict(base_coco.get("info") or {})
    info.update(
        {
            "description": "Merged spine vertebra keypoint COCO dataset with NIH ChestX-ray14",
            "version": "coco_nih",
            "split": split,
            "split_seed": int(seed),
            "split_ratios": dict(SPLIT_RATIOS),
            "base_dataset_root": "dataset/processed/coco",
            "nih_source_annotation_file": nih_source_annotation_file.as_posix(),
            "nih_split_policy": "patient_grouped_deterministic_80_10_10",
            "corner_order": list(CORNER_ORDER),
        }
    )
    return {
        "info": info,
        "licenses": list(base_coco.get("licenses") or []),
        "categories": [dict(CANONICAL_CATEGORY)],
        "images": images,
        "annotations": annotations,
    }


def validate_split(coco: dict[str, Any], split_dir: Path) -> None:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    image_ids = {int(image["id"]) for image in images}
    annotation_ids = {int(annotation["id"]) for annotation in annotations}
    if len(image_ids) != len(images):
        raise ValueError(f"Duplicate image IDs in {split_dir}")
    if len(annotation_ids) != len(annotations):
        raise ValueError(f"Duplicate annotation IDs in {split_dir}")
    if coco.get("categories") != [CANONICAL_CATEGORY]:
        raise ValueError(f"Unexpected category schema in {split_dir}")

    image_by_id = {int(image["id"]): image for image in images}
    for image in images:
        image_path = split_dir / str(image["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in image_ids:
            raise ValueError(f"Annotation {annotation.get('id')} references missing image {image_id}")
        if int(annotation.get("category_id", -1)) != 1:
            raise ValueError(f"Annotation {annotation.get('id')} has a noncanonical category")
        keypoints = np.asarray(annotation.get("keypoints", []), dtype=np.float64)
        if keypoints.size != 12 or not np.isfinite(keypoints).all():
            raise ValueError(f"Annotation {annotation.get('id')} has invalid keypoints")
        keypoints = keypoints.reshape(4, 3)
        image = image_by_id[image_id]
        if not np.all(keypoints[:, 2] > 0):
            raise ValueError(f"Annotation {annotation.get('id')} has absent keypoints")
        if not (
            np.all((keypoints[:, 0] >= 0.0) & (keypoints[:, 0] < float(image["width"])))
            and np.all((keypoints[:, 1] >= 0.0) & (keypoints[:, 1] < float(image["height"])))
        ):
            raise ValueError(f"Annotation {annotation.get('id')} has out-of-bounds keypoints")


def dataset_summary(
    *,
    staging_root: Path,
    split_records: dict[str, list[NihRecord]],
    excluded: list[dict[str, Any]],
    seed: int,
    base_root: Path,
    nih_root: Path,
    source_category_counts: Counter[int],
    image_targets: dict[str, int],
    annotation_targets: dict[str, int],
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    source_summaries: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for split in SPLITS:
        coco = load_json(staging_root / split / "_annotations.keypoints.coco.json")
        annotations_by_image = Counter(int(annotation["image_id"]) for annotation in coco["annotations"])
        images_by_source = Counter()
        annotations_by_source = Counter()
        for image in coco["images"]:
            source = canonical_source_key(image.get("source_dataset"))
            images_by_source[source] += 1
            annotations_by_source[source] += annotations_by_image[int(image["id"])]
        nih_patients = {record.patient_id for record in split_records[split]}
        split_summaries[split] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "images_by_source": dict(sorted(images_by_source.items())),
            "annotations_by_source": dict(sorted(annotations_by_source.items())),
            "nih_images": len(split_records[split]),
            "nih_annotations": sum(len(record.source_annotations) for record in split_records[split]),
            "nih_patients": len(nih_patients),
        }
        for source in sorted(images_by_source):
            source_summaries[source][split] = {
                "images": images_by_source[source],
                "annotations": annotations_by_source[source],
            }

    return {
        "seed": int(seed),
        "ratios": dict(SPLIT_RATIOS),
        "base_dataset_root": base_root.as_posix(),
        "nih_source_root": nih_root.as_posix(),
        "nih_source_annotation_file": (nih_root / "_annotations.coco.json").as_posix(),
        "corner_order": list(CORNER_ORDER),
        "nih_split_policy": "patient_grouped_deterministic_80_10_10",
        "nih_image_targets": image_targets,
        "nih_annotation_targets_for_balancing": annotation_targets,
        "nih_source_category_annotation_counts": {
            str(category_id): count for category_id, count in sorted(source_category_counts.items())
        },
        "nih_category_normalization": {
            "1": "source TL/TR/BR/BL reordered to TL/TR/BL/BR",
            "2": "source TL/TR/BL/BR retained",
            "output_category_id": 1,
        },
        "excluded_nih_images": excluded,
        "splits": split_summaries,
        "sources": dict(sorted(source_summaries.items())),
    }


def build_dataset(
    *,
    base_root: Path,
    nih_root: Path,
    output_root: Path,
    seed: int,
) -> dict[str, Any]:
    base_root = base_root.resolve()
    nih_root = nih_root.resolve()
    output_root = output_root.resolve()
    if not base_root.is_dir():
        raise FileNotFoundError(base_root)
    if not nih_root.is_dir():
        raise FileNotFoundError(nih_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    staging_root = output_root.with_name(output_root.name + ".building")
    if staging_root.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_root}")

    records, excluded, order_by_category = load_nih_records(nih_root)
    source_coco = load_json(nih_root / "_annotations.coco.json")
    source_category_counts = Counter(int(annotation["category_id"]) for annotation in source_coco["annotations"])
    split_records, image_targets, annotation_targets = split_nih_by_patient(records, seed=seed)

    staging_root.mkdir(parents=True)
    try:
        for split in SPLITS:
            base_coco = copy_base_split(base_root, staging_root, split)
            split_payload = build_split_payload(
                base_coco=base_coco,
                split=split,
                records=split_records[split],
                target_split_dir=staging_root / split,
                order_by_category=order_by_category,
                seed=seed,
                nih_source_annotation_file=nih_root / "_annotations.coco.json",
            )
            write_json_atomic(
                staging_root / split / "_annotations.keypoints.coco.json",
                split_payload,
            )
            validate_split(split_payload, staging_root / split)

        patient_sets = {
            split: {record.patient_id for record in split_records[split]}
            for split in SPLITS
        }
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                if patient_sets[left] & patient_sets[right]:
                    raise RuntimeError(f"NIH patient leakage between {left} and {right}")

        summary = dataset_summary(
            staging_root=staging_root,
            split_records=split_records,
            excluded=excluded,
            seed=seed,
            base_root=base_root,
            nih_root=nih_root,
            source_category_counts=source_category_counts,
            image_targets=image_targets,
            annotation_targets=annotation_targets,
        )
        write_json_atomic(staging_root / "split_summary.json", summary)
        if output_root.exists():
            output_root.rmdir()
        staging_root.replace(output_root)
        return summary
    except Exception:
        # Keep a non-empty staging directory for forensic inspection rather
        # than deleting a partially built dataset automatically.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the existing processed COCO dataset with patient-grouped NIH keypoints."
    )
    parser.add_argument("--base-root", type=Path, default=Path("dataset/processed/coco"))
    parser.add_argument("--nih-root", type=Path, default=Path("dataset/raw/NIH/version 0.5"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/processed/coco_nih"))
    parser.add_argument("--seed", type=int, default=20260627)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        base_root=args.base_root,
        nih_root=args.nih_root,
        output_root=args.output_root,
        seed=args.seed,
    )
    print(f"saved: {args.output_root}")
    print(f"excluded NIH images: {len(summary['excluded_nih_images'])}")
    for split in SPLITS:
        item = summary["splits"][split]
        print(
            f"{split}: {item['images']} images / {item['annotations']} annotations "
            f"(NIH {item['nih_images']} images / {item['nih_annotations']} annotations / "
            f"{item['nih_patients']} patients)"
        )


if __name__ == "__main__":
    main()
