from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import (  # noqa: E402
    AugmentationConfig,
    AugmentationError,
    build_train_transform,
    load_augmentation_config,
    seed_augmentation,
    transform_coco_sample,
)

POLYGON_ORDER = [0, 1, 3, 2, 0]

# COCO loading groups all vertebra annotations by image for sample-level transforms.
def load_coco_samples(annotation_path: Path) -> list[dict[str, Any]]:
    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    samples = []
    for image in coco["images"]:
        annotations = annotations_by_image.get(image["id"], [])
        if annotations:
            samples.append({"image": image, "annotations": annotations})
    return samples


def load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


# Drawing uses the vertebra corner order expected by this project.
def draw_annotations(
    axis: plt.Axes,
    image: np.ndarray,
    annotations: list[dict[str, Any]],
    title: str,
) -> None:
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_title(title, fontsize=10)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(annotations), 1)))
    for index, annotation in enumerate(annotations, start=1):
        points = np.asarray(annotation["keypoints"], dtype=float).reshape(4, 3)[:, :2]
        polygon = points[POLYGON_ORDER]
        color = colors[(index - 1) % len(colors)]

        axis.plot(polygon[:, 0], polygon[:, 1], "-", color=color, linewidth=1.8)
        axis.scatter(points[:, 0], points[:, 1], s=14, color=color, edgecolors="white")

        center = points.mean(axis=0)
        axis.text(
            center[0],
            center[1],
            str(index),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "linewidth": 0},
        )


# Preview export is intentionally separate from dataset generation.
def save_preview(
    original_image: np.ndarray,
    original_annotations: list[dict[str, Any]],
    augmented_image: np.ndarray,
    augmented_annotations: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_annotations(
        axes[0],
        original_image,
        original_annotations,
        title=f"Original | {len(original_annotations)} vertebrae",
    )
    draw_annotations(
        axes[1],
        augmented_image,
        augmented_annotations,
        title=f"Augmented | {len(augmented_annotations)} vertebrae",
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def preview_augmentations(
    dataset_dir: Path,
    output_dir: Path,
    config: AugmentationConfig,
    count: int,
) -> int:
    annotation_path = dataset_dir / "_annotations.keypoints.coco.json"
    samples = load_coco_samples(annotation_path)
    rng = random.Random(config.seed)
    rng.shuffle(samples)

    seed_augmentation(config.seed)
    transform = build_train_transform(config)

    saved = 0
    for sample in samples:
        if saved >= count:
            break

        image_info = sample["image"]
        image_path = dataset_dir / image_info["file_name"]
        original_image = load_rgb_image(image_path)

        try:
            augmented_image, augmented_annotations = transform_coco_sample(
                image=original_image,
                annotations=sample["annotations"],
                transform=transform,
            )
        except AugmentationError:
            continue

        file_parts = Path(image_info["file_name"]).parts
        source = file_parts[1] if len(file_parts) > 1 else "unknown"
        output_path = output_dir / f"{saved + 1:03d}_{source}_{image_path.stem}.png"
        save_preview(
            original_image=original_image,
            original_annotations=sample["annotations"],
            augmented_image=augmented_image,
            augmented_annotations=augmented_annotations,
            output_path=output_path,
            title=image_info["file_name"],
        )
        saved += 1

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save visual previews of safe train-time vertebra keypoint augmentation."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset/processed/train"),
        help="Train split directory containing images and COCO keypoint annotations.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Project config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/augmentation_preview"),
        help="Directory for preview PNG files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of previews to write. Defaults to augmentation.preview_count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_augmentation_config(args.config)
    count = args.count if args.count is not None else config.preview_count

    saved = preview_augmentations(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        config=config,
        count=count,
    )
    print(f"Wrote {saved} augmentation previews to {args.output_dir}")


if __name__ == "__main__":
    main()
