from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import KEYPOINT_NAMES, POLYGON_ORDER  # noqa: E402


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class YoloPoseAnnotation:
    bbox_xywh: np.ndarray
    keypoints: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save visual previews of YOLO pose labels for annotation rechecking."
    )
    parser.add_argument(
        "--yolo-root",
        type=Path,
        default=Path("dataset/processed/yolo"),
        help="YOLO pose dataset root containing images/, labels/, and spine_pose.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/yolo_anno_recheck"),
        help="Directory where preview PNGs are written.",
    )
    parser.add_argument(
        "--count-per-split",
        type=int,
        default=8,
        help="Number of annotated images to preview from each split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260627,
        help="Random seed used to sample preview images.",
    )
    parser.add_argument(
        "--show-keypoint-names",
        action="store_true",
        help="Draw top_left/top_right/bottom_left/bottom_right labels next to each point.",
    )
    return parser.parse_args()


def find_image_for_label(image_dir: Path, label_path: Path, label_dir: Path) -> Path:
    relative_stem = label_path.relative_to(label_dir).with_suffix("")
    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / relative_stem.with_suffix(extension)
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Could not find image matching label: {label_path}")


def load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def parse_yolo_label(label_path: Path, width: int, height: int) -> list[YoloPoseAnnotation]:
    annotations: list[YoloPoseAnnotation] = []
    expected_values = 5 + len(KEYPOINT_NAMES) * 3

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            values = stripped.split()
            if len(values) != expected_values:
                raise ValueError(
                    f"{label_path}:{line_number} has {len(values)} values, "
                    f"expected {expected_values}"
                )

            numeric = [float(value) for value in values]
            class_id = int(numeric[0])
            if class_id != 0:
                raise ValueError(f"{label_path}:{line_number} has unsupported class {class_id}")

            cx, cy, bbox_width, bbox_height = numeric[1:5]
            bbox_xywh = np.asarray(
                [
                    (cx - bbox_width / 2.0) * width,
                    (cy - bbox_height / 2.0) * height,
                    bbox_width * width,
                    bbox_height * height,
                ],
                dtype=float,
            )

            keypoints = np.asarray(numeric[5:], dtype=float).reshape(len(KEYPOINT_NAMES), 3)
            keypoints[:, 0] *= width
            keypoints[:, 1] *= height
            annotations.append(YoloPoseAnnotation(bbox_xywh=bbox_xywh, keypoints=keypoints))

    return annotations


def draw_yolo_annotations(
    axis: plt.Axes,
    image: np.ndarray,
    annotations: list[YoloPoseAnnotation],
    title: str,
    show_keypoint_names: bool,
) -> None:
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_title(title, fontsize=10)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(annotations), 1)))
    for index, annotation in enumerate(annotations, start=1):
        color = colors[(index - 1) % len(colors)]
        x_min, y_min, bbox_width, bbox_height = annotation.bbox_xywh
        rect = patches.Rectangle(
            (x_min, y_min),
            bbox_width,
            bbox_height,
            linewidth=1.1,
            edgecolor=color,
            facecolor="none",
            alpha=0.85,
        )
        axis.add_patch(rect)

        visible_points = annotation.keypoints[:, 2] > 0
        points = annotation.keypoints[:, :2]
        if visible_points.all():
            polygon = points[POLYGON_ORDER + [POLYGON_ORDER[0]]]
            axis.plot(polygon[:, 0], polygon[:, 1], "-", color=color, linewidth=1.8)

        axis.scatter(
            points[visible_points, 0],
            points[visible_points, 1],
            s=16,
            color=color,
            edgecolors="white",
            linewidths=0.6,
        )

        if show_keypoint_names:
            for point_name, (x_coord, y_coord, visible) in zip(KEYPOINT_NAMES, annotation.keypoints):
                if visible <= 0:
                    continue
                axis.text(
                    x_coord,
                    y_coord,
                    point_name.replace("_", "\n"),
                    color="white",
                    fontsize=5.5,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 0.6, "linewidth": 0},
                )

        center = points[visible_points].mean(axis=0) if visible_points.any() else points.mean(axis=0)
        axis.text(
            center[0],
            center[1],
            str(index),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1, "linewidth": 0},
        )


def save_preview(
    image: np.ndarray,
    annotations: list[YoloPoseAnnotation],
    output_path: Path,
    title: str,
    show_keypoint_names: bool,
) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(7, 7))
    draw_yolo_annotations(
        axis=axis,
        image=image,
        annotations=annotations,
        title=title,
        show_keypoint_names=show_keypoint_names,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def preview_split(
    yolo_root: Path,
    output_dir: Path,
    split: str,
    count: int,
    rng: random.Random,
    show_keypoint_names: bool,
) -> int:
    image_dir = yolo_root / "images" / split
    label_dir = yolo_root / "labels" / split
    label_paths = sorted(label_dir.rglob("*.txt"))
    rng.shuffle(label_paths)

    saved = 0
    for label_path in label_paths:
        if saved >= count:
            break

        image_path = find_image_for_label(image_dir, label_path, label_dir)
        image = load_rgb_image(image_path)
        height, width = image.shape[:2]
        annotations = parse_yolo_label(label_path, width=width, height=height)
        if not annotations:
            continue

        output_path = output_dir / f"{split}_{saved + 1:03d}_{image_path.stem}.png"
        save_preview(
            image=image,
            annotations=annotations,
            output_path=output_path,
            title=f"{split} | {image_path.name} | {len(annotations)} vertebrae",
            show_keypoint_names=show_keypoint_names,
        )
        saved += 1

    return saved


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    saved_by_split = {
        split: preview_split(
            yolo_root=args.yolo_root,
            output_dir=args.output_dir,
            split=split,
            count=args.count_per_split,
            rng=rng,
            show_keypoint_names=args.show_keypoint_names,
        )
        for split in SPLITS
    }
    total = sum(saved_by_split.values())
    print(f"Wrote {total} YOLO annotation previews to {args.output_dir}")
    for split, saved in saved_by_split.items():
        print(f"{split}: {saved}")


if __name__ == "__main__":
    main()
