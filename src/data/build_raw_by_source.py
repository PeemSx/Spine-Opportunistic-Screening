from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from src.data.build_coco_nih import CORNER_ORDER, canonical_source_key


SPLITS = ("train", "val", "test")
ANNOTATION_FILE_NAME = "_annotations.keypoints.coco.json"
SUMMARY_FILE_NAME = "source_summary.json"
OUTPUT_VERSION = "source_separated_raw_v1"


@dataclass(frozen=True)
class CopyRecord:
    source_path: Path
    target_name: str


@dataclass
class SourceBundle:
    display_names: set[str] = field(default_factory=set)
    images: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    copies: list[CopyRecord] = field(default_factory=list)
    target_names: dict[str, Path] = field(default_factory=dict)
    split_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            split: {"images": 0, "annotations": 0} for split in SPLITS
        }
    )


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
        file.write("\n")
    temporary_path.replace(path)


def _require_integer_id(value: Any, *, field_name: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} has a non-integer {field_name}: {value!r}")
    return value


def _source_path_parts(file_name: Any, *, context: str) -> tuple[str, str]:
    value = str(file_name or "")
    relative_path = PurePosixPath(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{context} has an unsafe file_name: {value!r}")
    if len(relative_path.parts) != 3 or relative_path.parts[0] != "images":
        raise ValueError(
            f"{context} file_name must be images/<source>/<name>, got {value!r}"
        )
    source_key, basename = relative_path.parts[1:]
    if not source_key or not basename:
        raise ValueError(f"{context} has an invalid file_name: {value!r}")
    return source_key, basename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_and_verify(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    if target_path.is_symlink() or not target_path.is_file():
        raise RuntimeError(f"Copy did not create a regular file: {target_path}")
    if os.path.samefile(source_path, target_path):
        raise RuntimeError(f"Copy is not independent from its source: {target_path}")
    source_hash = _sha256(source_path)
    target_hash = _sha256(target_path)
    if source_hash != target_hash:
        raise RuntimeError(f"SHA-256 mismatch after copying {source_path} to {target_path}")


def _validate_categories(categories: Any, *, annotation_path: Path) -> list[dict[str, Any]]:
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"Missing COCO categories in {annotation_path}")
    category_ids: set[int] = set()
    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise ValueError(f"Category {index} in {annotation_path} is not an object")
        category_id = _require_integer_id(
            category.get("id"),
            field_name="id",
            context=f"Category {index} in {annotation_path}",
        )
        if category_id in category_ids:
            raise ValueError(f"Duplicate category id {category_id} in {annotation_path}")
        category_ids.add(category_id)
    return copy.deepcopy(categories)


def _collect_source_bundles(
    processed_root: Path,
) -> tuple[
    dict[str, SourceBundle],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str | None,
    int | None,
]:
    bundles: dict[str, SourceBundle] = {}
    reference_categories: list[dict[str, Any]] | None = None
    reference_licenses: list[dict[str, Any]] | None = None
    processed_versions: set[str] = set()
    split_seeds: set[int] = set()

    for split in SPLITS:
        annotation_path = processed_root / split / ANNOTATION_FILE_NAME
        payload = load_json(annotation_path)
        categories = _validate_categories(payload.get("categories"), annotation_path=annotation_path)
        licenses = payload.get("licenses") or []
        if not isinstance(licenses, list):
            raise ValueError(f"COCO licenses must be a list in {annotation_path}")
        if reference_categories is None:
            reference_categories = categories
            reference_licenses = copy.deepcopy(licenses)
        elif categories != reference_categories:
            raise ValueError(f"COCO categories differ between splits at {annotation_path}")
        elif licenses != reference_licenses:
            raise ValueError(f"COCO licenses differ between splits at {annotation_path}")

        info = payload.get("info") or {}
        if not isinstance(info, dict):
            raise ValueError(f"COCO info must be an object in {annotation_path}")
        if info.get("version") is not None:
            processed_versions.add(str(info["version"]))
        if info.get("split_seed") is not None:
            split_seeds.add(
                _require_integer_id(
                    info["split_seed"],
                    field_name="split_seed",
                    context=f"COCO info in {annotation_path}",
                )
            )

        images = payload.get("images")
        annotations = payload.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise ValueError(f"COCO images and annotations must be lists in {annotation_path}")

        images_by_id: dict[int, dict[str, Any]] = {}
        for image_index, image in enumerate(images):
            if not isinstance(image, dict):
                raise ValueError(f"Image {image_index} in {annotation_path} is not an object")
            image_id = _require_integer_id(
                image.get("id"),
                field_name="id",
                context=f"Image {image_index} in {annotation_path}",
            )
            if image_id in images_by_id:
                raise ValueError(f"Duplicate image id {image_id} in {annotation_path}")
            images_by_id[image_id] = image

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        split_annotation_ids: set[int] = set()
        category_ids = {int(category["id"]) for category in categories}
        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise ValueError(
                    f"Annotation {annotation_index} in {annotation_path} is not an object"
                )
            annotation_id = _require_integer_id(
                annotation.get("id"),
                field_name="id",
                context=f"Annotation {annotation_index} in {annotation_path}",
            )
            if annotation_id in split_annotation_ids:
                raise ValueError(f"Duplicate annotation id {annotation_id} in {annotation_path}")
            split_annotation_ids.add(annotation_id)
            image_id = _require_integer_id(
                annotation.get("image_id"),
                field_name="image_id",
                context=f"Annotation {annotation_id} in {annotation_path}",
            )
            category_id = _require_integer_id(
                annotation.get("category_id"),
                field_name="category_id",
                context=f"Annotation {annotation_id} in {annotation_path}",
            )
            if image_id not in images_by_id:
                raise ValueError(
                    f"Annotation {annotation_id} in {annotation_path} references missing image {image_id}"
                )
            if category_id not in category_ids:
                raise ValueError(
                    f"Annotation {annotation_id} in {annotation_path} references missing category {category_id}"
                )
            annotations_by_image[image_id].append(annotation)

        for image_id, image in sorted(
            images_by_id.items(),
            key=lambda item: (item[0], str(item[1].get("file_name") or "")),
        ):
            image_context = f"Image {image_id} in {annotation_path}"
            source_name = str(image.get("source_dataset") or "").strip()
            if not source_name:
                raise ValueError(f"{image_context} is missing source_dataset")
            metadata_source_key = canonical_source_key(source_name)
            path_source_key, basename = _source_path_parts(
                image.get("file_name"), context=image_context
            )
            if metadata_source_key != path_source_key:
                raise ValueError(
                    f"{image_context} source mismatch: metadata {metadata_source_key!r}, "
                    f"path {path_source_key!r}"
                )

            source_path = processed_root / split / "images" / path_source_key / basename
            if not source_path.is_file():
                raise FileNotFoundError(source_path)

            bundle = bundles.setdefault(path_source_key, SourceBundle())
            normalized_target_name = basename.casefold()
            if normalized_target_name in bundle.target_names:
                first_path = bundle.target_names[normalized_target_name]
                raise ValueError(
                    f"Filename collision while merging source {path_source_key}: "
                    f"{first_path} and {source_path}"
                )
            bundle.target_names[normalized_target_name] = source_path
            bundle.display_names.add(source_name)
            output_image_id = len(bundle.images) + 1
            output_image = copy.deepcopy(image)
            output_image["id"] = output_image_id
            output_image["file_name"] = f"images/{basename}"
            output_image["processed_split"] = split
            output_image["processed_image_id"] = image_id
            bundle.images.append(output_image)
            bundle.copies.append(CopyRecord(source_path=source_path, target_name=basename))
            bundle.split_counts[split]["images"] += 1

            for annotation in sorted(
                annotations_by_image.get(image_id, []), key=lambda item: int(item["id"])
            ):
                annotation_id = int(annotation["id"])
                annotation_source_key = canonical_source_key(annotation.get("source_dataset"))
                if annotation_source_key != path_source_key:
                    raise ValueError(
                        f"Annotation {annotation_id} in {annotation_path} source mismatch: "
                        f"{annotation_source_key!r} != {path_source_key!r}"
                    )
                output_annotation = copy.deepcopy(annotation)
                output_annotation["id"] = len(bundle.annotations) + 1
                output_annotation["image_id"] = output_image_id
                output_annotation["processed_split"] = split
                output_annotation["processed_annotation_id"] = annotation_id
                bundle.annotations.append(output_annotation)
                bundle.split_counts[split]["annotations"] += 1

    if reference_categories is None or reference_licenses is None:
        raise ValueError(f"No COCO split metadata found under {processed_root}")
    if len(processed_versions) > 1:
        raise ValueError(f"COCO dataset versions differ between splits: {sorted(processed_versions)}")
    if len(split_seeds) > 1:
        raise ValueError(f"COCO split seeds differ between splits: {sorted(split_seeds)}")
    for source_key, bundle in bundles.items():
        if len(bundle.display_names) != 1:
            raise ValueError(
                f"Source {source_key} has inconsistent names: {sorted(bundle.display_names)}"
            )

    processed_version = next(iter(processed_versions), None)
    split_seed = next(iter(split_seeds), None)
    return bundles, reference_categories, reference_licenses, processed_version, split_seed


def _source_payload(
    *,
    source_key: str,
    bundle: SourceBundle,
    categories: list[dict[str, Any]],
    licenses: list[dict[str, Any]],
    processed_root: Path,
    processed_version: str | None,
    split_seed: int | None,
) -> dict[str, Any]:
    source_name = next(iter(bundle.display_names))
    info: dict[str, Any] = {
        "description": "Unsplit source-separated spine vertebra keypoint COCO dataset",
        "version": OUTPUT_VERSION,
        "split": "all",
        "source_dataset": source_name,
        "source_key": source_key,
        "source_splits": list(SPLITS),
        "derived_from": processed_root.as_posix(),
        "corner_order": list(CORNER_ORDER),
        "category_policy": "preserved from processed dataset",
        "id_policy": (
            "dense per-source COCO ids; original processed ids retained in "
            "processed_image_id and processed_annotation_id"
        ),
    }
    if processed_version is not None:
        info["processed_dataset_version"] = processed_version
    if split_seed is not None:
        info["processed_split_seed"] = split_seed
    return {
        "info": info,
        "licenses": copy.deepcopy(licenses),
        "images": sorted(bundle.images, key=lambda image: (int(image["id"]), image["file_name"])),
        "annotations": sorted(
            bundle.annotations,
            key=lambda annotation: (int(annotation["id"]), int(annotation["image_id"])),
        ),
        "categories": copy.deepcopy(categories),
    }


def validate_source_payload(payload: dict[str, Any], source_root: Path) -> None:
    source_key = str(payload.get("info", {}).get("source_key") or "")
    image_ids: set[int] = set()
    file_names: set[str] = set()
    for image in payload["images"]:
        image_id = int(image["id"])
        if image_id in image_ids:
            raise ValueError(f"Duplicate image id {image_id} in source {source_key}")
        image_ids.add(image_id)
        file_name = str(image["file_name"])
        if file_name in file_names:
            raise ValueError(f"Duplicate file_name {file_name!r} in source {source_key}")
        file_names.add(file_name)
        image_path = source_root / file_name
        if image_path.is_symlink() or not image_path.is_file():
            raise FileNotFoundError(image_path)
        if canonical_source_key(image.get("source_dataset")) != source_key:
            raise ValueError(f"Image {image_id} has inconsistent source metadata")

    category_ids = {int(category["id"]) for category in payload["categories"]}
    annotation_ids: set[int] = set()
    for annotation in payload["annotations"]:
        annotation_id = int(annotation["id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"Duplicate annotation id {annotation_id} in source {source_key}")
        annotation_ids.add(annotation_id)
        if int(annotation["image_id"]) not in image_ids:
            raise ValueError(f"Annotation {annotation_id} references a missing image")
        if int(annotation["category_id"]) not in category_ids:
            raise ValueError(f"Annotation {annotation_id} references a missing category")
        if canonical_source_key(annotation.get("source_dataset")) != source_key:
            raise ValueError(f"Annotation {annotation_id} has inconsistent source metadata")


def build_dataset(*, processed_root: Path, output_root: Path) -> dict[str, Any]:
    processed_root = processed_root.resolve()
    output_root = output_root.resolve()
    if not processed_root.is_dir():
        raise FileNotFoundError(processed_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    if output_root == processed_root or output_root.is_relative_to(processed_root):
        raise ValueError("Output root must not be the processed root or one of its descendants")

    staging_root = output_root.with_name(output_root.name + ".building")
    if staging_root.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_root}")

    bundles, categories, licenses, processed_version, split_seed = _collect_source_bundles(
        processed_root
    )
    if not bundles:
        raise ValueError(f"No source datasets found under {processed_root}")

    staging_root.mkdir(parents=True)
    try:
        source_summaries: dict[str, dict[str, Any]] = {}
        total_images = 0
        total_annotations = 0
        for source_key in sorted(bundles):
            bundle = bundles[source_key]
            source_root = staging_root / source_key
            for copy_record in sorted(bundle.copies, key=lambda item: item.target_name.casefold()):
                _copy_and_verify(
                    copy_record.source_path,
                    source_root / "images" / copy_record.target_name,
                )

            payload = _source_payload(
                source_key=source_key,
                bundle=bundle,
                categories=categories,
                licenses=licenses,
                processed_root=processed_root,
                processed_version=processed_version,
                split_seed=split_seed,
            )
            validate_source_payload(payload, source_root)
            write_json_atomic(source_root / ANNOTATION_FILE_NAME, payload)

            image_count = len(payload["images"])
            annotation_count = len(payload["annotations"])
            total_images += image_count
            total_annotations += annotation_count
            source_summaries[source_key] = {
                "source_dataset": next(iter(bundle.display_names)),
                "images": image_count,
                "annotations": annotation_count,
                "former_splits": copy.deepcopy(bundle.split_counts),
            }

        summary: dict[str, Any] = {
            "version": OUTPUT_VERSION,
            "processed_root": processed_root.as_posix(),
            "source_splits": list(SPLITS),
            "copy_method": "independent shutil.copy2 copies",
            "copy_verification": "SHA-256",
            "corner_order": list(CORNER_ORDER),
            "id_policy": (
                "dense per-source COCO ids with original processed split and ids retained"
            ),
            "totals": {"images": total_images, "annotations": total_annotations},
            "sources": source_summaries,
        }
        if processed_version is not None:
            summary["processed_dataset_version"] = processed_version
        if split_seed is not None:
            summary["processed_split_seed"] = split_seed
        write_json_atomic(staging_root / SUMMARY_FILE_NAME, summary)

        if output_root.exists():
            output_root.rmdir()
        staging_root.replace(output_root)
        return summary
    except Exception:
        # Keep partial staging output for forensic inspection.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine processed train/val/test COCO data into one dataset per source."
    )
    parser.add_argument("--processed-root", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(
        processed_root=args.processed_root,
        output_root=args.output_root,
    )
    print(f"saved: {args.output_root}")
    for source_key, source_summary in summary["sources"].items():
        print(
            f"{source_key}: {source_summary['images']} images / "
            f"{source_summary['annotations']} annotations"
        )
    totals = summary["totals"]
    print(f"total: {totals['images']} images / {totals['annotations']} annotations")


if __name__ == "__main__":
    main()
