from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import KEYPOINT_NAMES, bbox_from_points


SPLITS = ("train", "val", "test")
CLASS_ID = 0
YOLO_FLOAT_PRECISION = 6


@dataclass
class SplitConversionSummary:
    split: str
    images: int = 0
    annotations: int = 0
    labels: int = 0
    copied_images: int = 0
    skipped_annotations: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert project COCO keypoint annotations to Ultralytics YOLO pose format."
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("dataset/processed/coco"),
        help="Directory containing train/val/test COCO split folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/processed/yolo"),
        help="Output directory for YOLO pose images, labels, and YAML.",
    )
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy images into the YOLO dataset. Disable only if image folders already exist.",
    )
    parser.add_argument(
        "--preserve-source-folders",
        action="store_true",
        help="Keep source folders such as buu_ap/mendeley_pa under each split.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated YOLO images, labels, YAML, and summary before conversion.",
    )
    return parser.parse_args()


def load_coco(annotation_path: Path) -> dict[str, Any]:
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"{annotation_path} is missing required COCO key: {key}")
    return coco


def normalized(value: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("Normalization denominator must be positive")
    return min(max(float(value) / float(denominator), 0.0), 1.0)


def format_yolo_number(value: float) -> str:
    return f"{float(value):.{YOLO_FLOAT_PRECISION}f}".rstrip("0").rstrip(".")


def split_image_relative_path(file_name: str) -> Path:
    path = Path(file_name)
    parts = path.parts
    if parts and parts[0].lower() == "images":
        return Path(*parts[1:])
    return path


def flattened_image_name(file_name: str) -> str:
    image_relative_path = split_image_relative_path(file_name)
    stem_parts = [part for part in image_relative_path.with_suffix("").parts if part]
    return "_".join(stem_parts) + image_relative_path.suffix


def annotation_to_yolo_line(
    annotation: dict[str, Any],
    image_width: int,
    image_height: int,
) -> str | None:
    keypoints = np.asarray(annotation["keypoints"], dtype=float).reshape(-1, 3)
    if keypoints.shape != (len(KEYPOINT_NAMES), 3):
        raise ValueError(
            f"Expected {len(KEYPOINT_NAMES)} keypoints, got {keypoints.shape[0]}"
        )

    visibility = np.rint(keypoints[:, 2]).astype(int)
    visible_points = keypoints[visibility > 0, :2]
    if len(visible_points) == len(KEYPOINT_NAMES):
        x_min, y_min, box_width, box_height = bbox_from_points(visible_points)
    else:
        x_min, y_min, box_width, box_height = annotation["bbox"]

    x_max = min(float(x_min) + float(box_width), float(image_width))
    y_max = min(float(y_min) + float(box_height), float(image_height))
    x_min = max(float(x_min), 0.0)
    y_min = max(float(y_min), 0.0)
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0 or box_height <= 0:
        return None

    bbox_values = [
        normalized(x_min + box_width / 2.0, image_width),
        normalized(y_min + box_height / 2.0, image_height),
        normalized(box_width, image_width),
        normalized(box_height, image_height),
    ]

    keypoint_values: list[float | int] = []
    for x_coord, y_coord, visible in keypoints:
        visibility_value = int(round(float(visible)))
        if visibility_value <= 0:
            keypoint_values.extend([0.0, 0.0, 0])
            continue
        keypoint_values.extend(
            [
                normalized(x_coord, image_width),
                normalized(y_coord, image_height),
                visibility_value,
            ]
        )

    values: list[str] = [str(CLASS_ID)]
    values.extend(format_yolo_number(value) for value in bbox_values)
    values.extend(
        str(value) if isinstance(value, int) else format_yolo_number(value)
        for value in keypoint_values
    )
    return " ".join(values)


def write_dataset_yaml(output_root: Path) -> None:
    data = {
        "path": output_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [len(KEYPOINT_NAMES), 3],
        "flip_idx": [1, 0, 3, 2],
        "names": {CLASS_ID: "vertebra"},
        "kpt_names": {CLASS_ID: KEYPOINT_NAMES},
    }
    yaml_path = output_root / "spine_pose.yaml"
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def convert_split(
    split: str,
    coco_root: Path,
    output_root: Path,
    copy_images: bool,
    preserve_source_folders: bool,
) -> SplitConversionSummary:
    split_dir = coco_root / split
    coco = load_coco(split_dir / "_annotations.keypoints.coco.json")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    summary = SplitConversionSummary(
        split=split,
        images=len(coco["images"]),
        annotations=len(coco["annotations"]),
    )

    for image_info in coco["images"]:
        image_id = int(image_info["id"])
        image_file_name = image_info["file_name"]
        if preserve_source_folders:
            image_relative_path = split_image_relative_path(image_file_name)
        else:
            image_relative_path = Path(flattened_image_name(image_file_name))

        source_image_path = split_dir / image_file_name
        if not source_image_path.exists():
            raise FileNotFoundError(f"Missing image referenced by COCO: {source_image_path}")

        output_image_path = output_root / "images" / split / image_relative_path
        output_label_path = (
            output_root / "labels" / split / image_relative_path
        ).with_suffix(".txt")

        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        if copy_images:
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image_path, output_image_path)
            summary.copied_images += 1

        label_lines = []
        image_width = int(image_info["width"])
        image_height = int(image_info["height"])
        for annotation in annotations_by_image.get(image_id, []):
            line = annotation_to_yolo_line(
                annotation=annotation,
                image_width=image_width,
                image_height=image_height,
            )
            if line is None:
                summary.skipped_annotations += 1
                continue
            label_lines.append(line)

        with output_label_path.open("w", encoding="utf-8") as file:
            file.write("\n".join(label_lines))
            if label_lines:
                file.write("\n")
        summary.labels += 1

    return summary


def clean_generated_output(output_root: Path) -> None:
    project_root = PROJECT_ROOT.resolve()
    resolved_output_root = output_root.resolve()
    if project_root not in resolved_output_root.parents and resolved_output_root != project_root:
        raise ValueError(f"Refusing to clean path outside project root: {resolved_output_root}")

    for child in ("images", "labels"):
        target = resolved_output_root / child
        if target.exists():
            shutil.rmtree(target)

    for file_name in ("spine_pose.yaml", "conversion_summary.json"):
        target = resolved_output_root / file_name
        if target.exists():
            target.unlink()


def main() -> None:
    args = parse_args()
    coco_root = args.coco_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_generated_output(output_root)

    summaries = [
        convert_split(
            split=split,
            coco_root=coco_root,
            output_root=output_root,
            copy_images=args.copy_images,
            preserve_source_folders=args.preserve_source_folders,
        )
        for split in SPLITS
    ]
    write_dataset_yaml(output_root)

    summary_data = {
        "coco_root": coco_root.as_posix(),
        "output_root": output_root.as_posix(),
        "splits": [asdict(summary) for summary in summaries],
    }
    with (output_root / "conversion_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary_data, file, indent=2)

    total_images = sum(summary.images for summary in summaries)
    total_annotations = sum(summary.annotations for summary in summaries)
    total_skipped = sum(summary.skipped_annotations for summary in summaries)
    print(
        "Converted {images} images and {annotations} annotations to {output}. "
        "Skipped annotations: {skipped}".format(
            images=total_images,
            annotations=total_annotations,
            output=output_root,
            skipped=total_skipped,
        )
    )


if __name__ == "__main__":
    main()
