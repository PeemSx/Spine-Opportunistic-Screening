from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


ROBOFLOW_NAME = re.compile(
    r"^(?P<nih_id>\d{8}_\d{3})_png\.rf\.[^.]+\.png$",
    flags=re.IGNORECASE,
)
CLEAN_NAME = re.compile(r"^NIH_(?P<nih_id>\d{8}_\d{3})\.png$", flags=re.IGNORECASE)
KEYPOINT_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
CORNER_FIELDS = (
    ("top_left_x", "top_left_y"),
    ("top_right_x", "top_right_y"),
    ("bottom_left_x", "bottom_left_y"),
    ("bottom_right_x", "bottom_right_y"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_nih_name(name: str) -> str | None:
    roboflow_match = ROBOFLOW_NAME.fullmatch(name)
    if roboflow_match:
        return f"NIH_{roboflow_match.group('nih_id')}.png"
    clean_match = CLEAN_NAME.fullmatch(name)
    if clean_match:
        return f"NIH_{clean_match.group('nih_id')}.png"
    return None


def rename_nih_images(images_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    images_dir = images_dir.resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)

    source_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
    )
    if not source_paths:
        raise FileNotFoundError(f"No top-level PNG images found in {images_dir}")

    plan: list[tuple[Path, Path]] = []
    target_names: set[str] = set()
    for source_path in source_paths:
        target_name = clean_nih_name(source_path.name)
        if target_name is None:
            raise ValueError(f"Unsupported NIH filename: {source_path.name}")
        target_key = target_name.lower()
        if target_key in target_names:
            raise FileExistsError(f"Multiple files would become {target_name}")
        target_names.add(target_key)
        target_path = images_dir / target_name
        if target_path.exists() and target_path.resolve() != source_path.resolve():
            raise FileExistsError(f"Rename target already exists: {target_path}")
        plan.append((source_path, target_path))

    # A repeated preparation run must not overwrite the only record of the
    # original Roboflow filenames after all files already have clean names.
    if manifest_path.is_file() and all(source.name == target.name for source, target in plan):
        with manifest_path.open("r", newline="", encoding="utf-8") as file:
            existing_records = list(csv.DictReader(file))
        mapped_names = {str(record.get("renamed_name", "")).lower() for record in existing_records}
        current_names = {path.name.lower() for path in source_paths}
        if mapped_names == current_names:
            return existing_records
        raise ValueError(
            f"Existing rename manifest does not match current NIH images: {manifest_path}"
        )

    records: list[dict[str, Any]] = []
    for source_path, target_path in plan:
        original_name = source_path.name
        checksum = sha256_file(source_path)
        if source_path.name != target_path.name:
            source_path.rename(target_path)
        records.append(
            {
                "original_name": original_name,
                "renamed_name": target_path.name,
                "sha256": checksum,
                "renamed": original_name != target_path.name,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("original_name", "renamed_name", "sha256", "renamed"),
        )
        writer.writeheader()
        writer.writerows(records)
    return records


def load_prediction_payload(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def points_from_prediction(row: dict[str, Any]) -> np.ndarray:
    points = np.asarray(
        [[float(row[x_field]), float(row[y_field])] for x_field, y_field in CORNER_FIELDS],
        dtype=np.float64,
    )
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"Invalid TL/TR/BL/BR points in prediction rank {row.get('rank')}")
    return points


def clipped_points(points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, bool]:
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, max(float(width - 1), 0.0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, max(float(height - 1), 0.0))
    was_clipped = not np.allclose(clipped, points, rtol=0.0, atol=1e-9)
    return clipped, was_clipped


def polygon_area(points_tl_tr_bl_br: np.ndarray) -> float:
    polygon = points_tl_tr_bl_br[[0, 1, 3, 2]]
    x_coord = polygon[:, 0]
    y_coord = polygon[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x_coord, np.roll(y_coord, -1))
            - np.dot(y_coord, np.roll(x_coord, -1))
        )
    )


def verify_image_dimensions(image_path: Path, expected_width: int, expected_height: int) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not decode image: {image_path}")
    actual_height, actual_width = image.shape[:2]
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise ValueError(
            f"Dimension mismatch for {image_path.name}: JSON "
            f"{expected_width}x{expected_height}, image {actual_width}x{actual_height}"
        )


def coco_from_predictions(
    prediction_payload: Iterable[dict[str, Any]],
    images_dir: Path,
    source_checkpoint: str,
    prediction_policy: str,
    source_dataset: str = "NIH ChestX-ray14",
) -> tuple[dict[str, Any], dict[str, int]]:
    images_dir = images_dir.resolve()
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    clipped_annotation_count = 0

    sorted_payload = sorted(
        prediction_payload,
        key=lambda item: Path(str(item.get("image", ""))).name.lower(),
    )
    for image_id, item in enumerate(sorted_payload, start=1):
        image_name = Path(str(item["image"])).name
        image_path = images_dir / image_name
        width = int(item["original_width"])
        height = int(item["original_height"])
        verify_image_dimensions(image_path, width, height)
        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
                "source_dataset": source_dataset,
                "source_file_name": image_name,
            }
        )

        rows = sorted(item.get("predictions", []), key=lambda row: int(row.get("rank", 0)))
        for row in rows:
            points, was_clipped = clipped_points(points_from_prediction(row), width, height)
            min_xy = points.min(axis=0)
            max_xy = points.max(axis=0)
            bbox_width, bbox_height = max_xy - min_xy
            if bbox_width <= 0.0 or bbox_height <= 0.0:
                continue
            if was_clipped:
                clipped_annotation_count += 1
            annotation_id = len(annotations) + 1
            keypoints = [
                coordinate
                for x_coord, y_coord in points
                for coordinate in (round(float(x_coord), 3), round(float(y_coord), 3), 2)
            ]
            segmentation_points = points[[0, 1, 3, 2]].reshape(-1)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        round(float(min_xy[0]), 3),
                        round(float(min_xy[1]), 3),
                        round(float(bbox_width), 3),
                        round(float(bbox_height), 3),
                    ],
                    "area": round(polygon_area(points), 3),
                    "iscrowd": 0,
                    "num_keypoints": 4,
                    "keypoints": keypoints,
                    "segmentation": [
                        [round(float(coordinate), 3) for coordinate in segmentation_points]
                    ],
                    "score": round(float(row.get("score", 0.0)), 6),
                    "pseudo_label": True,
                    "source_model": source_checkpoint,
                    "source_rank": int(row.get("rank", 0)),
                }
            )

    coco = {
        "info": {
            "description": f"{source_dataset} vertebra corner pseudo-annotations",
            "annotation_type": "COCO Keypoints",
            "corner_order": list(KEYPOINT_NAMES),
            "prediction_policy": prediction_policy,
            "source_checkpoint": source_checkpoint,
            "requires_human_review": True,
        },
        "licenses": [],
        "categories": [
            {
                "id": 1,
                "name": "vertebra",
                "supercategory": "spine",
                "keypoints": list(KEYPOINT_NAMES),
                "skeleton": [[1, 2], [3, 4], [1, 3], [2, 4]],
            }
        ],
        "images": images,
        "annotations": annotations,
    }
    stats = {
        "images": len(images),
        "annotations": len(annotations),
        "clipped_annotations": clipped_annotation_count,
    }
    return coco, stats


def validate_coco(coco: dict[str, Any], images_dir: Path) -> None:
    image_ids = {int(item["id"]) for item in coco["images"]}
    if len(image_ids) != len(coco["images"]):
        raise ValueError("Duplicate COCO image IDs")
    annotation_ids = {int(item["id"]) for item in coco["annotations"]}
    if len(annotation_ids) != len(coco["annotations"]):
        raise ValueError("Duplicate COCO annotation IDs")

    image_by_id = {int(item["id"]): item for item in coco["images"]}
    for image in coco["images"]:
        if not (images_dir / str(image["file_name"])).is_file():
            raise FileNotFoundError(images_dir / str(image["file_name"]))
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in image_ids:
            raise ValueError(f"Unknown image_id {image_id}")
        if len(annotation["keypoints"]) != 12 or int(annotation["num_keypoints"]) != 4:
            raise ValueError(f"Invalid keypoints for annotation {annotation['id']}")
        keypoints = np.asarray(annotation["keypoints"], dtype=float).reshape(4, 3)
        image = image_by_id[image_id]
        if not np.all(keypoints[:, 2] == 2):
            raise ValueError(f"Invalid visibility in annotation {annotation['id']}")
        if not (
            np.all((keypoints[:, 0] >= 0) & (keypoints[:, 0] < int(image["width"])))
            and np.all((keypoints[:, 1] >= 0) & (keypoints[:, 1] < int(image["height"])))
        ):
            raise ValueError(f"Out-of-bounds keypoint in annotation {annotation['id']}")


def export_coco(
    predictions_path: Path,
    images_dir: Path,
    output_path: Path,
    source_checkpoint: str,
    prediction_policy: str,
    source_dataset: str = "NIH ChestX-ray14",
) -> dict[str, int]:
    coco, stats = coco_from_predictions(
        load_prediction_payload(predictions_path),
        images_dir=images_dir,
        source_checkpoint=source_checkpoint,
        prediction_policy=prediction_policy,
        source_dataset=source_dataset,
    )
    validate_coco(coco, images_dir.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, indent=2, allow_nan=False)
    temporary_path.replace(output_path)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename NIH images and export CenterNet predictions as Roboflow-ready COCO keypoints."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rename_parser = subparsers.add_parser("rename", help="Remove Roboflow hashes from NIH filenames.")
    rename_parser.add_argument("--images-dir", type=Path, required=True)
    rename_parser.add_argument("--manifest", type=Path, required=True)

    export_parser = subparsers.add_parser("export", help="Convert predictor JSON to COCO keypoints.")
    export_parser.add_argument("--predictions", type=Path, required=True)
    export_parser.add_argument("--images-dir", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--source-checkpoint", required=True)
    export_parser.add_argument("--prediction-policy", required=True)
    export_parser.add_argument("--source-dataset", default="NIH ChestX-ray14")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "rename":
        records = rename_nih_images(args.images_dir, args.manifest)
        renamed_count = sum(bool(record["renamed"]) for record in records)
        print(f"images: {len(records)}")
        print(f"renamed: {renamed_count}")
        print(f"manifest: {args.manifest}")
        return

    stats = export_coco(
        predictions_path=args.predictions,
        images_dir=args.images_dir,
        output_path=args.output,
        source_checkpoint=args.source_checkpoint,
        prediction_policy=args.prediction_policy,
        source_dataset=args.source_dataset,
    )
    print(f"images: {stats['images']}")
    print(f"annotations: {stats['annotations']}")
    print(f"clipped annotations: {stats['clipped_annotations']}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
