from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.inference.corner_refiner import load_corner_refiner, refine_centernet_prediction
from src.models.centernet import build_centernet_model
from src.predict_centernet import load_checkpoint, predict_image


MATCH_FIELDS = [
    "image",
    "source_dataset",
    "image_id",
    "annotation_id",
    "detector_index",
    "detector_score",
    "center_distance_px",
    "center_distance_fraction",
    "bbox_diagonal_px",
    "refiner_accepted",
    "refiner_reason",
    "identity_offset_fraction",
    "baseline_corner_error_px",
    "two_stage_corner_error_px",
    "baseline_nme",
    "two_stage_nme",
]


def canonical_source(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if "buu" in normalized:
        return "buu_ap"
    if "mendeley" in normalized:
        return "mendeley_pa"
    if "miccai" in normalized:
        return "miccai_2019"
    return normalized


def annotation_points(annotation: dict[str, Any]) -> np.ndarray | None:
    raw = np.asarray(annotation.get("keypoints", []), dtype=np.float32)
    if raw.size != 12:
        return None
    keypoints = raw.reshape(4, 3)
    if not np.isfinite(keypoints).all() or not np.all(keypoints[:, 2] > 0):
        return None
    return keypoints[:, :2]


def match_detector_to_gt(
    detector_centers: np.ndarray,
    gt_corners: np.ndarray,
    *,
    max_center_distance_fraction: float,
) -> list[tuple[int, int, float, float]]:
    if len(detector_centers) == 0 or len(gt_corners) == 0:
        return []
    gt_centers = gt_corners.mean(axis=1)
    extents = gt_corners.max(axis=1) - gt_corners.min(axis=1)
    diagonals = np.maximum(np.linalg.norm(extents, axis=1), 1e-6)
    distances = np.linalg.norm(detector_centers[:, None] - gt_centers[None], axis=-1)
    normalized = distances / diagonals[None]
    pairs = np.argwhere(normalized <= float(max_center_distance_fraction))
    ordered = sorted(pairs, key=lambda pair: float(normalized[pair[0], pair[1]]))
    used_detector: set[int] = set()
    used_gt: set[int] = set()
    matches = []
    for detector_index, gt_index in ordered:
        detector_index, gt_index = int(detector_index), int(gt_index)
        if detector_index in used_detector or gt_index in used_gt:
            continue
        used_detector.add(detector_index)
        used_gt.add(gt_index)
        matches.append(
            (
                detector_index,
                gt_index,
                float(distances[detector_index, gt_index]),
                float(normalized[detector_index, gt_index]),
            )
        )
    return matches


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "matched_instances": 0,
            "accepted_instances": 0,
            "acceptance_rate": 0.0,
            "baseline_mean_nme": None,
            "two_stage_mean_nme": None,
            "relative_improvement_percent": None,
        }
    baseline_nme = np.asarray([row["baseline_nme"] for row in rows], dtype=np.float64)
    two_stage_nme = np.asarray([row["two_stage_nme"] for row in rows], dtype=np.float64)
    accepted = np.asarray([row["refiner_accepted"] for row in rows], dtype=bool)
    baseline_corner_errors = np.asarray(
        [row["baseline_corner_errors"] for row in rows], dtype=np.float64
    )
    two_stage_corner_errors = np.asarray(
        [row["two_stage_corner_errors"] for row in rows], dtype=np.float64
    )
    diagonals = np.asarray([row["bbox_diagonal_px"] for row in rows], dtype=np.float64)
    baseline_corner_nme = baseline_corner_errors / diagonals[:, None]
    two_stage_corner_nme = two_stage_corner_errors / diagonals[:, None]
    baseline_mean = float(baseline_nme.mean())
    two_stage_mean = float(two_stage_nme.mean())
    improvement = 100.0 * (baseline_mean - two_stage_mean) / max(baseline_mean, 1e-12)
    accepted_baseline = baseline_nme[accepted]
    accepted_two_stage = two_stage_nme[accepted]
    return {
        "matched_instances": int(len(rows)),
        "accepted_instances": int(accepted.sum()),
        "acceptance_rate": float(accepted.mean()),
        "baseline_mean_nme": baseline_mean,
        "two_stage_mean_nme": two_stage_mean,
        "baseline_median_nme": float(np.median(baseline_nme)),
        "two_stage_median_nme": float(np.median(two_stage_nme)),
        "baseline_p95_nme": float(np.percentile(baseline_nme, 95)),
        "two_stage_p95_nme": float(np.percentile(two_stage_nme, 95)),
        "baseline_mean_corner_error_px": float(baseline_corner_errors.mean()),
        "two_stage_mean_corner_error_px": float(two_stage_corner_errors.mean()),
        "relative_improvement_percent": improvement,
        "accepted_baseline_mean_nme": (
            float(accepted_baseline.mean()) if len(accepted_baseline) else None
        ),
        "accepted_two_stage_mean_nme": (
            float(accepted_two_stage.mean()) if len(accepted_two_stage) else None
        ),
        "baseline_pck_0.02": float((baseline_corner_nme <= 0.02).mean()),
        "two_stage_pck_0.02": float((two_stage_corner_nme <= 0.02).mean()),
        "baseline_pck_0.05": float((baseline_corner_nme <= 0.05).mean()),
        "two_stage_pck_0.05": float((two_stage_corner_nme <= 0.05).mean()),
        "baseline_pck_0.10": float((baseline_corner_nme <= 0.10).mean()),
        "two_stage_pck_0.10": float((two_stage_corner_nme <= 0.10).mean()),
        "fallback_reasons": dict(
            Counter(row["refiner_reason"] for row in rows if not row["refiner_accepted"])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CenterNet and two-stage corners on identical matched GT instances."
    )
    parser.add_argument("--centernet-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/processed/coco"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/two_stage_evaluation"))
    parser.add_argument("--sources", nargs="+", default=["mendeley_pa", "miccai_2019"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--peak-thresh", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--max-center-distance-fraction", type=float, default=0.25)
    parser.add_argument("--refiner-batch-size", type=int, default=16)
    parser.add_argument("--refiner-crop-scale", type=float, default=None)
    parser.add_argument("--refiner-max-center-offset", type=float, default=0.20)
    parser.add_argument("--refiner-allow-boundary-peaks", action="store_true")
    parser.add_argument("--refiner-no-nearest-anchor", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--enforce-acceptance", action="store_true")
    parser.add_argument("--minimum-improvement-percent", type=float, default=20.0)
    parser.add_argument("--max-source-degradation-percent", type=float, default=5.0)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    center_checkpoint = load_checkpoint(args.centernet_checkpoint, device)
    center_args = center_checkpoint.get("args", {})
    backbone = center_args.get("backbone", "hrnet_w18")
    input_size = int(center_args.get("input_size", 1024))
    down_ratio = int(center_args.get("down_ratio", 4))
    peak_thresh = float(
        args.peak_thresh
        if args.peak_thresh is not None
        else center_args.get("peak_thresh", 0.05)
    )
    topk = int(args.topk or center_args.get("eval_topk", 100))
    center_model = build_centernet_model(backbone=backbone, pretrained=False)
    center_model.load_state_dict(center_checkpoint["model_state_dict"], strict=True)
    center_model.to(device).eval()

    refiner_model, _, checkpoint_runtime = load_corner_refiner(args.refiner_checkpoint, device)
    refiner_config = replace(
        checkpoint_runtime,
        crop_scale=float(
            args.refiner_crop_scale
            if args.refiner_crop_scale is not None
            else checkpoint_runtime.crop_scale
        ),
        batch_size=int(args.refiner_batch_size),
        max_center_offset_fraction=float(args.refiner_max_center_offset),
        reject_boundary_peaks=not args.refiner_allow_boundary_peaks,
        require_nearest_anchor=not args.refiner_no_nearest_anchor,
    )

    split_dir = args.dataset_root / args.split
    with (split_dir / "_annotations.keypoints.coco.json").open("r", encoding="utf-8") as file:
        coco = json.load(file)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    requested_sources = {canonical_source(source) for source in args.sources}
    images = [
        image
        for image in sorted(coco["images"], key=lambda item: str(item["file_name"]).lower())
        if canonical_source(image.get("source_dataset")) in requested_sources
    ]
    if args.limit is not None:
        images = images[: int(args.limit)]
    if not images:
        raise RuntimeError(f"No images selected for sources {sorted(requested_sources)}")

    rows: list[dict[str, Any]] = []
    for image_number, image_info in enumerate(images, start=1):
        image_path = split_dir / str(image_info["file_name"])
        image_rgb, _, baseline, _ = predict_image(
            center_model,
            image_path,
            device,
            input_size,
            down_ratio,
            peak_thresh,
            topk,
        )
        two_stage, diagnostics = refine_centernet_prediction(
            refiner_model,
            image_rgb,
            baseline,
            device,
            refiner_config,
            use_amp=not args.no_amp,
        )
        valid_annotations = []
        valid_points = []
        for annotation in annotations_by_image.get(int(image_info["id"]), []):
            points = annotation_points(annotation)
            if points is not None:
                valid_annotations.append(annotation)
                valid_points.append(points)
        gt_corners = np.asarray(valid_points, dtype=np.float32)
        matches = match_detector_to_gt(
            baseline["centers"],
            gt_corners,
            max_center_distance_fraction=args.max_center_distance_fraction,
        )
        source = canonical_source(image_info.get("source_dataset"))
        for detector_index, gt_index, center_distance, center_fraction in matches:
            target = gt_corners[gt_index]
            extent = target.max(axis=0) - target.min(axis=0)
            diagonal = max(float(np.linalg.norm(extent)), 1e-6)
            baseline_errors = np.linalg.norm(baseline["corners"][detector_index] - target, axis=1)
            two_stage_errors = np.linalg.norm(two_stage["corners"][detector_index] - target, axis=1)
            diagnostic = diagnostics[detector_index]
            rows.append(
                {
                    "image": str(image_info["file_name"]),
                    "source_dataset": source,
                    "image_id": int(image_info["id"]),
                    "annotation_id": int(valid_annotations[gt_index].get("id", -1)),
                    "detector_index": int(detector_index),
                    "detector_score": float(baseline["scores"][detector_index]),
                    "center_distance_px": center_distance,
                    "center_distance_fraction": center_fraction,
                    "bbox_diagonal_px": diagonal,
                    "refiner_accepted": bool(diagnostic["accepted"]),
                    "refiner_reason": str(diagnostic["reason"]),
                    "identity_offset_fraction": float(diagnostic["identity_offset_fraction"]),
                    "baseline_corner_error_px": float(baseline_errors.mean()),
                    "two_stage_corner_error_px": float(two_stage_errors.mean()),
                    "baseline_nme": float(baseline_errors.mean() / diagonal),
                    "two_stage_nme": float(two_stage_errors.mean() / diagonal),
                    "baseline_corner_errors": baseline_errors.astype(float).tolist(),
                    "two_stage_corner_errors": two_stage_errors.astype(float).tolist(),
                }
            )
        print(
            f"[{image_number}/{len(images)}] {image_info['file_name']} "
            f"detections={len(baseline['centers'])} matched={len(matches)} "
            f"accepted={sum(item['accepted'] for item in diagnostics)}",
            flush=True,
        )

    aggregate = summarize_rows(rows)
    per_source = {
        source: summarize_rows([row for row in rows if row["source_dataset"] == source])
        for source in sorted(requested_sources)
    }
    report = {
        "metadata": {
            "centernet_checkpoint": str(args.centernet_checkpoint),
            "refiner_checkpoint": str(args.refiner_checkpoint),
            "dataset_root": str(args.dataset_root),
            "split": args.split,
            "sources": sorted(requested_sources),
            "fixed_match_policy": {
                "basis": "CenterNet center to GT center",
                "max_center_distance_fraction_of_gt_diagonal": args.max_center_distance_fraction,
                "same_pairs_used_for_baseline_and_two_stage": True,
            },
        },
        "aggregate": aggregate,
        "per_source": per_source,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "two_stage_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
    with (args.output_dir / "matched_instances.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MATCH_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in MATCH_FIELDS} for row in rows)

    print(json.dumps({"aggregate": aggregate, "per_source": per_source}, indent=2))
    print(f"saved: {args.output_dir / 'two_stage_metrics.json'}")
    print(f"saved: {args.output_dir / 'matched_instances.csv'}")

    if args.enforce_acceptance:
        improvement = aggregate.get("relative_improvement_percent")
        failures = []
        if improvement is None or improvement < args.minimum_improvement_percent:
            failures.append(
                f"aggregate improvement {improvement} < {args.minimum_improvement_percent}%"
            )
        for source, metrics in per_source.items():
            source_improvement = metrics.get("relative_improvement_percent")
            if (
                source_improvement is None
                or source_improvement < -float(args.max_source_degradation_percent)
            ):
                failures.append(
                    f"{source} improvement {source_improvement} is materially degraded"
                )
        if failures:
            raise SystemExit("Two-stage acceptance failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
