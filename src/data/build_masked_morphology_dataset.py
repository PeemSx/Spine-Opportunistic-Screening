from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.data.dataset_provenance import build_dataset_provenance


SPLITS = ("train", "val", "test")
CONTEXT_OFFSETS = (-2, -1, 1, 2)
DIMENSIONS = ("left_height", "right_height", "superior_width", "inferior_width")
MIN_AREA_FRACTION = 1e-5
MAX_EDGE_DIAGONAL_FRACTION = 0.25
SCHEMA_VERSION = 1

LANDMARK_COLUMNS = (
    "tl_x",
    "tl_y",
    "tr_x",
    "tr_y",
    "bl_x",
    "bl_y",
    "br_x",
    "br_y",
)

MORPHOLOGY_COLUMNS = (
    "center_x",
    "center_y",
    "superior_width",
    "inferior_width",
    "left_height",
    "right_height",
    "mean_width",
    "mean_height",
    "left_right_height_ratio",
    "height_asymmetry_fraction",
    "width_height_ratio",
    "superior_endplate_angle_deg",
    "inferior_endplate_angle_deg",
    "orientation_deg",
    "endplate_nonparallel_deg",
    "polygon_area_px2",
    "polygon_area_fraction",
    "max_edge_diagonal_fraction",
)

NEIGHBOR_FEATURES = (
    "left_height_norm",
    "right_height_norm",
    "superior_width_norm",
    "inferior_width_norm",
    "orientation_relative_deg",
    "center_dx_norm",
    "center_dy_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe relative-neighbor tables for masked frontal vertebral "
            "morphology modelling."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/processed/coco_nih_lumos"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/processed/masked_morphology_coco_nih_lumos"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace generated files when the output directory already exists.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(source: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", source.casefold()).strip("_")
    return value or "unknown"


def infer_view(image: Mapping[str, Any]) -> str:
    explicit = str(image.get("view") or image.get("view_position") or "").upper()
    if explicit in {"AP", "PA"}:
        return explicit
    text = " ".join(
        str(image.get(field) or "")
        for field in ("source_dataset", "file_name", "source_file_name")
    ).upper()
    if re.search(r"(^|[^A-Z])PA([^A-Z]|$)", text):
        return "PA"
    if re.search(r"(^|[^A-Z])AP([^A-Z]|$)", text):
        return "AP"
    return "unknown"


def patient_group(image: Mapping[str, Any]) -> tuple[str, str]:
    source = _source_key(str(image.get("source_dataset") or "unknown"))
    for field in ("patient_id", "nih_patient_id"):
        value = image.get(field)
        if value not in (None, ""):
            return f"{source}:patient:{value}", field
    lumos_case = image.get("lumos_case_id")
    if lumos_case not in (None, ""):
        return f"{source}:case:{lumos_case}", "lumos_case_id"
    file_stem = Path(str(image.get("file_name") or image.get("id"))).stem
    return f"{source}:image:{file_stem}", "image_only_no_patient_id"


def wrap_axial_deg(angle: float | np.ndarray) -> float | np.ndarray:
    result = (np.asarray(angle) + 90.0) % 180.0 - 90.0
    return float(result) if result.ndim == 0 else result


def axial_mean_deg(first: float, second: float) -> float:
    first2 = math.radians(2.0 * first)
    second2 = math.radians(2.0 * second)
    return float(
        wrap_axial_deg(
            0.5
            * math.degrees(
                math.atan2(
                    math.sin(first2) + math.sin(second2),
                    math.cos(first2) + math.cos(second2),
                )
            )
        )
    )


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(second) - np.asarray(first)))


def _line_angle(first: np.ndarray, second: np.ndarray) -> float:
    vector = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    return float(wrap_axial_deg(math.degrees(math.atan2(vector[1], vector[0]))))


def _polygon_area(points: np.ndarray) -> float:
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


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    first_vector = second - first
    second_vector = third - first
    return float(
        first_vector[0] * second_vector[1]
        - first_vector[1] * second_vector[0]
    )


def _segments_intersect(
    first_a: np.ndarray,
    first_b: np.ndarray,
    second_a: np.ndarray,
    second_b: np.ndarray,
) -> bool:
    o1 = _orientation(first_a, first_b, second_a)
    o2 = _orientation(first_a, first_b, second_b)
    o3 = _orientation(second_a, second_b, first_a)
    o4 = _orientation(second_a, second_b, first_b)
    epsilon = 1e-9
    return (o1 * o2 < -epsilon) and (o3 * o4 < -epsilon)


def _simple_quadrilateral(points: np.ndarray) -> bool:
    tl, tr, bl, br = points
    return not (
        _segments_intersect(tl, tr, br, bl)
        or _segments_intersect(tr, br, bl, tl)
    )


def _parse_keypoints(annotation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(annotation.get("keypoints", []), dtype=np.float64)
    if values.size != 12:
        raise ValueError("keypoints_must_contain_four_xyv_points")
    points = values.reshape(4, 3)
    return points[:, :2], points[:, 2]


def morphology_from_annotation(
    annotation: Mapping[str, Any],
    image: Mapping[str, Any],
) -> dict[str, Any]:
    width = int(image["width"])
    height = int(image["height"])
    reasons: list[str] = []
    try:
        points, visibility = _parse_keypoints(annotation)
    except (TypeError, ValueError) as error:
        points = np.full((4, 2), np.nan, dtype=np.float64)
        visibility = np.zeros(4, dtype=np.float64)
        reasons.append(str(error))

    if not np.isfinite(points).all():
        reasons.append("nonfinite_landmarks")
    if not np.all(visibility > 0):
        reasons.append("invisible_landmark")
    if np.isfinite(points).all():
        in_bounds = (
            np.all((points[:, 0] >= 0.0) & (points[:, 0] < float(width)))
            and np.all((points[:, 1] >= 0.0) & (points[:, 1] < float(height)))
        )
        if not in_bounds:
            reasons.append("out_of_bounds_landmark")

    tl, tr, bl, br = points
    superior_width = _distance(tl, tr)
    inferior_width = _distance(bl, br)
    left_height = _distance(tl, bl)
    right_height = _distance(tr, br)
    mean_width = 0.5 * (superior_width + inferior_width)
    mean_height = 0.5 * (left_height + right_height)
    polygon_area = _polygon_area(points)
    area_fraction = polygon_area / max(float(width * height), 1.0)
    image_diagonal = math.hypot(width, height)
    max_edge_diagonal_fraction = max(
        superior_width,
        inferior_width,
        left_height,
        right_height,
    ) / max(image_diagonal, 1.0)

    if min(superior_width, inferior_width, left_height, right_height) <= 0.0:
        reasons.append("nonpositive_dimension")
    if area_fraction < MIN_AREA_FRACTION:
        reasons.append("tiny_polygon")
    if max_edge_diagonal_fraction > MAX_EDGE_DIAGONAL_FRACTION:
        reasons.append("edge_too_large_for_image")
    if not _simple_quadrilateral(points):
        reasons.append("self_intersecting_quadrilateral")
    if float(np.mean([tl[1], tr[1]])) >= float(np.mean([bl[1], br[1]])):
        reasons.append("top_not_above_bottom")
    if float(np.mean([tl[0], bl[0]])) >= float(np.mean([tr[0], br[0]])):
        reasons.append("left_not_left_of_right")

    superior_angle = _line_angle(tl, tr)
    inferior_angle = _line_angle(bl, br)
    orientation = axial_mean_deg(superior_angle, inferior_angle)
    result: dict[str, Any] = {
        "geometry_valid": not reasons,
        "geometry_exclusion_reason": ";".join(dict.fromkeys(reasons)),
        "center_x": float(points[:, 0].mean()),
        "center_y": float(points[:, 1].mean()),
        "superior_width": superior_width,
        "inferior_width": inferior_width,
        "left_height": left_height,
        "right_height": right_height,
        "mean_width": mean_width,
        "mean_height": mean_height,
        "left_right_height_ratio": left_height / max(right_height, 1e-12),
        "height_asymmetry_fraction": abs(left_height - right_height)
        / max(mean_height, 1e-12),
        "width_height_ratio": mean_width / max(mean_height, 1e-12),
        "superior_endplate_angle_deg": superior_angle,
        "inferior_endplate_angle_deg": inferior_angle,
        "orientation_deg": orientation,
        "endplate_nonparallel_deg": abs(
            float(wrap_axial_deg(superior_angle - inferior_angle))
        ),
        "polygon_area_px2": polygon_area,
        "polygon_area_fraction": area_fraction,
        "max_edge_diagonal_fraction": max_edge_diagonal_fraction,
    }
    for name, value in zip(LANDMARK_COLUMNS, points.reshape(-1)):
        result[name] = float(value)
    return result


def build_vertebra_rows(
    coco: Mapping[str, Any],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    rows: list[dict[str, Any]] = []
    source_rank_mismatches = 0
    for image_id, image in images.items():
        source_dataset = str(image.get("source_dataset") or "unknown")
        source_key = _source_key(source_dataset)
        group_id, grouping_basis = patient_group(image)
        image_key = f"{source_key}:{Path(str(image['file_name'])).stem}"
        image_annotations = annotations_by_image.get(image_id, [])
        measured: list[dict[str, Any]] = []
        for annotation in image_annotations:
            morphology = morphology_from_annotation(annotation, image)
            measured.append(
                {
                    "split": split,
                    "group_id": group_id,
                    "grouping_basis": grouping_basis,
                    "image_key": image_key,
                    "coco_image_id": image_id,
                    "coco_annotation_id": int(annotation["id"]),
                    "source_dataset": source_dataset,
                    "source_dataset_key": source_key,
                    "view": infer_view(image),
                    "image_path": str(image["file_name"]),
                    "image_width": int(image["width"]),
                    "image_height": int(image["height"]),
                    "source_chain_rank": int(annotation.get("vertebra_index", 0)),
                    **morphology,
                }
            )

        center_ordered = sorted(
            measured,
            key=lambda row: (
                float(row["center_y"]),
                float(row["center_x"]),
                int(row["coco_annotation_id"]),
            )
        )
        for center_order_rank, row in enumerate(center_ordered, start=1):
            row["center_order_rank"] = center_order_rank
            if row["source_chain_rank"] not in (0, center_order_rank):
                source_rank_mismatches += 1

        source_ranks = [int(row["source_chain_rank"]) for row in measured]
        has_complete_source_order = sorted(source_ranks) == list(
            range(1, len(measured) + 1)
        )
        if has_complete_source_order:
            measured.sort(key=lambda row: int(row["source_chain_rank"]))
        else:
            measured = center_ordered

        for chain_rank, row in enumerate(measured, start=1):
            row["chain_rank"] = chain_rank
            row["chain_count"] = len(measured)
            rows.append(row)

    audit = {
        "images": len(images),
        "annotations": sum(len(value) for value in annotations_by_image.values()),
        "vertebra_rows": len(rows),
        "geometry_valid": sum(bool(row["geometry_valid"]) for row in rows),
        "geometry_invalid": sum(not bool(row["geometry_valid"]) for row in rows),
        "source_rank_mismatches": source_rank_mismatches,
        "geometry_exclusion_reasons": dict(
            Counter(
                reason
                for row in rows
                for reason in str(row["geometry_exclusion_reason"]).split(";")
                if reason
            )
        ),
    }
    return rows, audit


def _offset_name(offset: int) -> str:
    return f"m{abs(offset)}" if offset < 0 else f"p{offset}"


def numeric_feature_columns() -> list[str]:
    return [
        f"x_{_offset_name(offset)}_{feature}"
        for offset in CONTEXT_OFFSETS
        for feature in NEIGHBOR_FEATURES
    ]


def residual_target_columns() -> list[str]:
    return [f"y_{dimension}_residual" for dimension in DIMENSIONS] + [
        "y_orientation_residual_deg"
    ]


def build_masked_samples(vertebra_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows_by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in vertebra_rows:
        rows_by_image[str(row["image_key"])].append(row)

    samples: list[dict[str, Any]] = []
    for image_key, image_rows in rows_by_image.items():
        ordered = sorted(image_rows, key=lambda row: int(row["chain_rank"]))
        for target_position in range(2, len(ordered) - 2):
            window = ordered[target_position - 2 : target_position + 3]
            if len(window) != 5 or not all(bool(row["geometry_valid"]) for row in window):
                continue
            target = window[2]
            neighbors = {
                offset: ordered[target_position + offset]
                for offset in CONTEXT_OFFSETS
            }
            ref_height = float(np.median([row["mean_height"] for row in neighbors.values()]))
            ref_width = float(np.median([row["mean_width"] for row in neighbors.values()]))
            if not (np.isfinite(ref_height) and np.isfinite(ref_width)):
                continue
            if ref_height <= 0.0 or ref_width <= 0.0:
                continue

            nearest_superior = neighbors[-1]
            nearest_inferior = neighbors[1]
            baseline_orientation = axial_mean_deg(
                float(nearest_superior["orientation_deg"]),
                float(nearest_inferior["orientation_deg"]),
            )
            anchor_x = 0.5 * (
                float(nearest_superior["center_x"])
                + float(nearest_inferior["center_x"])
            )
            anchor_y = 0.5 * (
                float(nearest_superior["center_y"])
                + float(nearest_inferior["center_y"])
            )
            sample_id = f"{target['split']}:{target['coco_annotation_id']}"
            sample: dict[str, Any] = {
                "sample_id": sample_id,
                "split": target["split"],
                "group_id": target["group_id"],
                "grouping_basis": target["grouping_basis"],
                "image_key": image_key,
                "image_path": target["image_path"],
                "coco_image_id": target["coco_image_id"],
                "target_annotation_id": target["coco_annotation_id"],
                "source_dataset": target["source_dataset"],
                "source_dataset_key": target["source_dataset_key"],
                "view": target["view"],
                "target_chain_rank": target["chain_rank"],
                "chain_count": target["chain_count"],
                "sample_weight": 0.0,
                "reference_height_px": ref_height,
                "reference_width_px": ref_width,
                "baseline_center_x_px": anchor_x,
                "baseline_center_y_px": anchor_y,
                "baseline_orientation_deg": baseline_orientation,
            }

            for offset, neighbor in neighbors.items():
                prefix = f"x_{_offset_name(offset)}"
                sample[f"context_{_offset_name(offset)}_annotation_id"] = int(
                    neighbor["coco_annotation_id"]
                )
                sample[f"context_{_offset_name(offset)}_chain_rank"] = int(
                    neighbor["chain_rank"]
                )
                sample[f"{prefix}_left_height_norm"] = float(neighbor["left_height"]) / ref_height
                sample[f"{prefix}_right_height_norm"] = float(neighbor["right_height"]) / ref_height
                sample[f"{prefix}_superior_width_norm"] = float(neighbor["superior_width"]) / ref_width
                sample[f"{prefix}_inferior_width_norm"] = float(neighbor["inferior_width"]) / ref_width
                sample[f"{prefix}_orientation_relative_deg"] = float(
                    wrap_axial_deg(float(neighbor["orientation_deg"]) - baseline_orientation)
                )
                sample[f"{prefix}_center_dx_norm"] = (
                    float(neighbor["center_x"]) - anchor_x
                ) / ref_width
                sample[f"{prefix}_center_dy_norm"] = (
                    float(neighbor["center_y"]) - anchor_y
                ) / ref_height

            for dimension in DIMENSIONS:
                scale = ref_height if "height" in dimension else ref_width
                actual_px = float(target[dimension])
                baseline_px = 0.5 * (
                    float(nearest_superior[dimension])
                    + float(nearest_inferior[dimension])
                )
                actual_norm = actual_px / scale
                baseline_norm = baseline_px / scale
                sample[f"y_{dimension}_px"] = actual_px
                sample[f"y_{dimension}_norm"] = actual_norm
                sample[f"baseline_{dimension}_norm"] = baseline_norm
                sample[f"y_{dimension}_residual"] = actual_norm - baseline_norm

            sample["y_orientation_deg"] = float(target["orientation_deg"])
            sample["y_orientation_residual_deg"] = float(
                wrap_axial_deg(float(target["orientation_deg"]) - baseline_orientation)
            )
            samples.append(sample)

    counts_by_group = Counter(str(sample["group_id"]) for sample in samples)
    for sample in samples:
        sample["sample_weight"] = 1.0 / counts_by_group[str(sample["group_id"])]
    return samples


def _ordered_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        return []
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    return fieldnames


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    names = list(fieldnames or _ordered_fieldnames(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    temporary_path.replace(path)


def _review_rows(samples: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "group_id": sample["group_id"],
                "image_path": sample["image_path"],
                "source_dataset": sample["source_dataset"],
                "view": sample["view"],
                "target_annotation_id": sample["target_annotation_id"],
                "target_chain_rank": sample["target_chain_rank"],
                "chain_count": sample["chain_count"],
                "normality_status": "unreviewed",
                "visibility_status": "unreviewed",
                "include_as_expected_normal_target": 0,
                "reviewer": "",
                "review_notes": "",
            }
        )
    return rows


def _group_overlap(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[str]]:
    group_sets = {
        split: {str(row["group_id"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    return {
        "train_validation": sorted(group_sets["train"] & group_sets["val"]),
        "train_test": sorted(group_sets["train"] & group_sets["test"]),
        "validation_test": sorted(group_sets["val"] & group_sets["test"]),
    }


def _dataset_readme() -> str:
    return """# Masked vertebral morphology candidate dataset

This generated dataset uses frontal AP/PA four-corner annotations in canonical
TL, TR, BL, BR order. It preserves the source train/validation/test assignments.

Files per split:

- `vertebrae.csv`: one row per annotated vertebra, with landmarks and morphology.
- `masked_samples.csv`: one row per geometry-valid target with context at -2, -1,
  +1, and +2. `x_*` columns use visible neighbors only. `baseline_*` columns are
  nearest-neighbor interpolation. `y_*_residual` columns are optional learning
  targets for models that predict corrections to the baseline.

`target_review_queue.csv` is a template for adding verified anatomical levels,
visibility, and normality eligibility.

Important limitations:

- The source COCO files do not contain verified anatomical levels. `chain_rank`
  is superior-to-inferior order within the visible image, not T/L identity.
- Clinical normality is not labelled. All generated rows may train observed-shape
  reconstruction. Use `target_review_queue.csv` to select reviewed targets before
  claiming an expected-normal model.
- BUU, Mendeley, and MICCAI lack patient identifiers in the processed annotations;
  their grouping falls back to one radiograph per group. NIH is patient-grouped and
  Lumos is case-grouped.

Research screening-support data only. This dataset does not contain diagnoses.

Rebuild from the repository root:

```powershell
python -m src.data.build_masked_morphology_dataset `
  --dataset-root dataset/processed/coco_nih_lumos `
  --output-root dataset/processed/masked_morphology_coco_nih_lumos `
  --force
```

Minimal loading pattern:

```python
import json
import pandas as pd

root = "dataset/processed/masked_morphology_coco_nih_lumos"
schema = json.load(open(f"{root}/feature_schema.json"))
train = pd.read_csv(f"{root}/train/masked_samples.csv")
valid = pd.read_csv(f"{root}/val/masked_samples.csv")

features = schema["numeric_input_columns"]
target = "y_left_height_residual"

X_train = train[features]
X_valid = valid[features]
y_train = train[target]
y_valid = valid[target]
weights = train[schema["weight_column"]]
```

The final normalized dimension estimate is `baseline_*_norm + predicted residual`.
For orientation, add the predicted residual to `baseline_orientation_deg` and wrap
the result to the axial interval [-90, 90).

Models should use only `numeric_input_columns`. Image and source fields are kept
for auditing and group-aware evaluation, not as morphology inputs.
"""


def build_dataset(dataset_root: Path, output_root: Path, *, force: bool = False) -> dict[str, Any]:
    dataset_root = dataset_root.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root == dataset_root or dataset_root in output_root.parents:
        raise ValueError("Output root must not be the source dataset or a child of it")
    if output_root.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_root}. Pass --force to replace generated files."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    source_provenance = build_dataset_provenance(dataset_root)
    vertebrae_by_split: dict[str, list[dict[str, Any]]] = {}
    samples_by_split: dict[str, list[dict[str, Any]]] = {}
    split_audits: dict[str, Any] = {}
    for split in SPLITS:
        annotation_path = dataset_root / split / "_annotations.keypoints.coco.json"
        coco = _load_json(annotation_path)
        vertebrae, audit = build_vertebra_rows(coco, split=split)
        samples = build_masked_samples(vertebrae)
        vertebrae_by_split[split] = vertebrae
        samples_by_split[split] = samples
        split_audits[split] = {
            **audit,
            "masked_samples": len(samples),
            "groups": len({row["group_id"] for row in vertebrae}),
            "grouping_basis": dict(Counter(str(row["grouping_basis"]) for row in vertebrae)),
            "sources": dict(Counter(str(row["source_dataset_key"]) for row in vertebrae)),
            "views": dict(Counter(str(row["view"]) for row in vertebrae)),
        }

    overlap = _group_overlap(vertebrae_by_split)
    if any(overlap.values()):
        raise ValueError(f"Group leakage across source splits: {overlap}")

    vertebra_fields = _ordered_fieldnames(
        next((rows for rows in vertebrae_by_split.values() if rows), [])
    )
    sample_fields = _ordered_fieldnames(
        next((rows for rows in samples_by_split.values() if rows), [])
    )
    generated_paths: list[Path] = []
    for split in SPLITS:
        vertebra_path = output_root / split / "vertebrae.csv"
        sample_path = output_root / split / "masked_samples.csv"
        _write_csv(vertebra_path, vertebrae_by_split[split], fieldnames=vertebra_fields)
        _write_csv(sample_path, samples_by_split[split], fieldnames=sample_fields)
        generated_paths.extend([vertebra_path, sample_path])

    all_samples = [sample for split in SPLITS for sample in samples_by_split[split]]
    review_path = output_root / "target_review_queue.csv"
    _write_csv(review_path, _review_rows(all_samples))
    generated_paths.append(review_path)

    schema = {
        "schema_version": SCHEMA_VERSION,
        "corner_order": ["TL", "TR", "BL", "BR"],
        "point_order": ["x", "y"],
        "coordinate_space": "original_image_pixels",
        "context_offsets": list(CONTEXT_OFFSETS),
        "categorical_input_columns": [],
        "metadata_columns": [
            "sample_id",
            "group_id",
            "image_path",
            "source_dataset_key",
            "view",
            "target_chain_rank",
            "chain_count",
        ],
        "numeric_input_columns": numeric_feature_columns(),
        "baseline_columns": [
            f"baseline_{dimension}_norm" for dimension in DIMENSIONS
        ]
        + ["baseline_orientation_deg"],
        "residual_target_columns": residual_target_columns(),
        "direct_target_columns": [f"y_{dimension}_norm" for dimension in DIMENSIONS]
        + ["y_orientation_deg"],
        "group_column": "group_id",
        "weight_column": "sample_weight",
        "leakage_contract": (
            "Every x_* feature, reference scale, and baseline uses only offsets "
            "-2, -1, +1, and +2. Target morphology appears only in y_* columns."
        ),
    }
    schema_path = output_root / "feature_schema.json"
    _write_json(schema_path, schema)
    generated_paths.append(schema_path)

    readme_path = output_root / "README.md"
    readme_path.write_text(_dataset_readme(), encoding="utf-8")
    generated_paths.append(readme_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "masked_morphology_coco_nih_lumos",
        "source_dataset_root": str(dataset_root),
        "source_provenance": source_provenance,
        "split_policy": "preserved_from_source_coco",
        "group_overlap": overlap,
        "split_audits": split_audits,
        "total_vertebra_rows": sum(len(rows) for rows in vertebrae_by_split.values()),
        "total_masked_samples": len(all_samples),
        "normality_status": "unreviewed",
        "expected_normal_training_samples": 0,
        "reconstruction_training_samples": len(all_samples),
        "limitations": [
            "No verified anatomical vertebral levels are present in source COCO.",
            "No clinical normality labels are present in source COCO.",
            "BUU, Mendeley, and MICCAI grouping falls back to radiograph identity.",
        ],
        "files": {
            path.relative_to(output_root).as_posix(): {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in generated_paths
        },
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args.dataset_root, args.output_root, force=args.force)
    print(f"Masked morphology dataset: {args.output_root.resolve()}")
    print(f"Vertebra rows: {manifest['total_vertebra_rows']:,}")
    print(f"Masked samples: {manifest['total_masked_samples']:,}")
    print("Expected-normal samples: 0 (clinical review required)")
    for split in SPLITS:
        audit = manifest["split_audits"][split]
        print(
            f"{split}: {audit['vertebra_rows']:,} vertebrae, "
            f"{audit['masked_samples']:,} masked samples"
        )


if __name__ == "__main__":
    main()
