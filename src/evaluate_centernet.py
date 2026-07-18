from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.centernet_dataset import CenterNetCocoDataset
from src.evaluation.center_metrics import CenterMetricAccumulator
from src.evaluation.centernet_decode import decode_centernet_outputs
from src.evaluation.cobb_angle import CobbResult, calculate_cobb_angles, cobb_smape
from src.models.centernet import build_centernet_model
from src.postprocessing.spine_chain import (
    candidates_to_prediction,
    prediction_to_candidates,
    select_spine_chain,
)
from src.training.centernet_loss import CenterNetLoss


SUMMARY_FIELDS = [
    "model",
    "split",
    "images",
    "predictions",
    "test_loss",
    "test_hm_loss",
    "test_reg_loss",
    "test_wh_loss",
    "count_mae",
    "center_precision_8px",
    "center_recall_8px",
    "center_f1_8px",
    "center_precision_12px",
    "center_recall_12px",
    "center_f1_12px",
    "center_precision_16px",
    "center_recall_16px",
    "center_f1_16px",
    "matched_12px",
    "center_mae_12px",
    "corner_mae_12px",
    "cobb_valid_images",
    "cobb_mae_deg",
    "cobb_smape_percent",
    "cobb_within_5deg_rate",
    "cobb_within_10deg_rate",
]
PER_IMAGE_FIELDS = [
    "model",
    "image",
    "image_id",
    "source_dataset",
    "gt_count",
    "pred_count",
    "count_error",
    "matched_12px",
    "center_mae_12px",
    "corner_mae_12px",
    "gt_cobb_1_deg",
    "gt_cobb_2_deg",
    "gt_cobb_3_deg",
    "pred_cobb_1_deg",
    "pred_cobb_2_deg",
    "pred_cobb_3_deg",
    "cobb_mae_deg",
    "cobb_smape_percent",
    "cobb_valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CenterNet raw and spine-chain predictions.")
    parser.add_argument("--checkpoint", type=Path, default=Path("src/weights/hrnet/best_center_f1.pt"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/processed/coco_no_buu"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--source-dataset",
        type=str,
        default=None,
        help="Evaluate only images whose COCO source_dataset matches this value (case-insensitive).",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hrnet_eval"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--down-ratio", type=int, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--peak-thresh", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chain-duplicate-iou", type=float, default=0.18)
    parser.add_argument("--chain-duplicate-center-scale", type=float, default=0.35)
    parser.add_argument("--chain-score-thresh", type=float, default=0.18)
    parser.add_argument("--chain-score-weight", type=float, default=3.0)
    parser.add_argument("--chain-min-len", type=int, default=3)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def average_stats(total: dict[str, float], batches: int) -> dict[str, float]:
    return {key: value / max(batches, 1) for key, value in total.items()}


def greedy_match_pairs(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    max_distance_px: float,
) -> list[tuple[int, int, float]]:
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return []

    deltas = pred_centers[:, None, :] - gt_centers[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    pairs = np.argwhere(distances <= float(max_distance_px))
    if len(pairs) == 0:
        return []

    pairs = sorted(pairs, key=lambda pair: float(distances[pair[0], pair[1]]))
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for pred_index, gt_index in pairs:
        pred_index = int(pred_index)
        gt_index = int(gt_index)
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append((pred_index, gt_index, float(distances[pred_index, gt_index])))
    return matches


def nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


@dataclass
class MatchMetricTotals:
    matched: int = 0
    center_error_sum: float = 0.0
    corner_error_sum: float = 0.0

    def update(
        self,
        prediction: dict[str, np.ndarray],
        gt_centers: np.ndarray,
        gt_corners: np.ndarray,
        max_distance_px: float = 12.0,
    ) -> dict[str, Any]:
        pairs = greedy_match_pairs(prediction["centers"], gt_centers, max_distance_px=max_distance_px)
        center_errors = []
        corner_errors = []
        for pred_index, gt_index, center_distance in pairs:
            center_errors.append(center_distance)
            pred_corners = prediction["corners"][pred_index]
            target_corners = gt_corners[gt_index]
            corner_errors.append(float(np.linalg.norm(pred_corners - target_corners, axis=1).mean()))

        matched = len(pairs)
        center_mae = float(np.mean(center_errors)) if center_errors else None
        corner_mae = float(np.mean(corner_errors)) if corner_errors else None
        self.matched += matched
        self.center_error_sum += float(np.sum(center_errors)) if center_errors else 0.0
        self.corner_error_sum += float(np.sum(corner_errors)) if corner_errors else 0.0
        return {
            "matched_12px": matched,
            "center_mae_12px": center_mae,
            "corner_mae_12px": corner_mae,
        }

    def compute(self) -> dict[str, Any]:
        if self.matched == 0:
            return {
                "matched_12px": 0,
                "center_mae_12px": None,
                "corner_mae_12px": None,
            }
        return {
            "matched_12px": self.matched,
            "center_mae_12px": self.center_error_sum / self.matched,
            "corner_mae_12px": self.corner_error_sum / self.matched,
        }


@dataclass
class CobbMetricTotals:
    valid_images: int = 0
    error_sum: float = 0.0
    smape_sum: float = 0.0
    within_5_count: int = 0
    within_10_count: int = 0

    def update(self, gt_result: CobbResult, pred_result: CobbResult) -> dict[str, Any]:
        gt_angle = gt_result.cobb_1_deg
        pred_angle = pred_result.cobb_1_deg
        valid = gt_result.valid and pred_result.valid and gt_angle is not None and pred_angle is not None
        if not valid:
            return {
                "gt_cobb_1_deg": gt_result.cobb_1_deg,
                "gt_cobb_2_deg": gt_result.cobb_2_deg,
                "gt_cobb_3_deg": gt_result.cobb_3_deg,
                "pred_cobb_1_deg": pred_result.cobb_1_deg,
                "pred_cobb_2_deg": pred_result.cobb_2_deg,
                "pred_cobb_3_deg": pred_result.cobb_3_deg,
                "cobb_mae_deg": None,
                "cobb_smape_percent": None,
                "cobb_valid": False,
            }

        error = abs(float(gt_angle) - float(pred_angle))
        smape = cobb_smape(float(gt_angle), float(pred_angle))
        self.valid_images += 1
        self.error_sum += error
        self.smape_sum += smape
        self.within_5_count += int(error <= 5.0)
        self.within_10_count += int(error <= 10.0)
        return {
            "gt_cobb_1_deg": gt_result.cobb_1_deg,
            "gt_cobb_2_deg": gt_result.cobb_2_deg,
            "gt_cobb_3_deg": gt_result.cobb_3_deg,
            "pred_cobb_1_deg": pred_result.cobb_1_deg,
            "pred_cobb_2_deg": pred_result.cobb_2_deg,
            "pred_cobb_3_deg": pred_result.cobb_3_deg,
            "cobb_mae_deg": error,
            "cobb_smape_percent": smape,
            "cobb_valid": True,
        }

    def compute(self) -> dict[str, Any]:
        if self.valid_images == 0:
            return {
                "cobb_valid_images": 0,
                "cobb_mae_deg": None,
                "cobb_smape_percent": None,
                "cobb_within_5deg_rate": None,
                "cobb_within_10deg_rate": None,
            }
        return {
            "cobb_valid_images": self.valid_images,
            "cobb_mae_deg": self.error_sum / self.valid_images,
            "cobb_smape_percent": self.smape_sum / self.valid_images,
            "cobb_within_5deg_rate": self.within_5_count / self.valid_images,
            "cobb_within_10deg_rate": self.within_10_count / self.valid_images,
        }


def make_chain_predictions(decoded: list[dict[str, np.ndarray]], args: argparse.Namespace) -> list[dict[str, np.ndarray]]:
    chain_predictions = []
    for prediction in decoded:
        candidates = prediction_to_candidates(prediction)
        _, chain_candidates, _ = select_spine_chain(
            candidates,
            duplicate_iou_threshold=args.chain_duplicate_iou,
            duplicate_center_scale=args.chain_duplicate_center_scale,
            score_threshold=args.chain_score_thresh,
            score_weight=args.chain_score_weight,
            min_chain_len=args.chain_min_len,
        )
        chain_predictions.append(candidates_to_prediction(chain_candidates))
    return chain_predictions


def build_model_and_settings(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = load_checkpoint(args.checkpoint, device)
    train_args = checkpoint.get("args", {})
    backbone = train_args.get("backbone", "hrnet_w18")
    input_size = int(args.input_size or train_args.get("input_size", 1024))
    down_ratio = int(args.down_ratio or train_args.get("down_ratio", 4))
    max_objects = int(args.max_objects or train_args.get("max_objects", 64))
    peak_thresh = float(args.peak_thresh if args.peak_thresh is not None else train_args.get("peak_thresh", 0.05))
    topk = int(args.topk or train_args.get("eval_topk", 100))

    model = build_centernet_model(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    settings = {
        "backbone": backbone,
        "input_size": input_size,
        "down_ratio": down_ratio,
        "max_objects": max_objects,
        "peak_thresh": peak_thresh,
        "topk": topk,
        "hm_weight": float(train_args.get("hm_weight", 1.0)),
        "reg_weight": float(train_args.get("reg_weight", 1.0)),
        "wh_weight": float(train_args.get("wh_weight", 0.1)),
    }
    return model, settings


def metric_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return round(value, 6)
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: metric_csv_value(row.get(field)) for field in fieldnames})


def format_summary_row(
    name: str,
    split: str,
    image_count: int,
    prediction_count: int,
    loss_stats: dict[str, float],
    center_metrics: dict[str, float],
    match_metrics: dict[str, Any],
    cobb_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": name,
        "split": split,
        "images": image_count,
        "predictions": prediction_count,
        "test_loss": loss_stats["loss"],
        "test_hm_loss": loss_stats["hm_loss"],
        "test_reg_loss": loss_stats["reg_loss"],
        "test_wh_loss": loss_stats["wh_loss"],
        **center_metrics,
        **match_metrics,
        **cobb_metrics,
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    device = resolve_device(args.device)
    model, settings = build_model_and_settings(args, device)
    dataset = CenterNetCocoDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        config_path=args.config,
        image_size=settings["input_size"],
        down_ratio=settings["down_ratio"],
        max_objects=settings["max_objects"],
        augment=False,
        limit=args.limit,
        source_dataset=args.source_dataset,
    )
    if args.source_dataset is not None:
        print(f"source dataset filter: {args.source_dataset} | matched images: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    criterion = CenterNetLoss(
        hm_weight=settings["hm_weight"],
        reg_weight=settings["reg_weight"],
        wh_weight=settings["wh_weight"],
    )

    raw_center_acc = CenterMetricAccumulator()
    chain_center_acc = CenterMetricAccumulator()
    raw_match_totals = MatchMetricTotals()
    chain_match_totals = MatchMetricTotals()
    raw_cobb_totals = CobbMetricTotals()
    chain_cobb_totals = CobbMetricTotals()
    loss_totals = {"loss": 0.0, "hm_loss": 0.0, "reg_loss": 0.0, "wh_loss": 0.0}
    raw_prediction_count = 0
    chain_prediction_count = 0
    image_count = 0
    batches = 0
    per_image_rows: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(loader, start=1):
        batch_on_device = move_batch_to_device(batch, device)
        outputs = model(batch_on_device["input"])
        loss_dict = criterion.loss_dict(outputs, batch_on_device)
        for key in loss_totals:
            loss_totals[key] += float(loss_dict[key].detach().cpu())
        batches += 1

        raw_predictions = decode_centernet_outputs(
            outputs,
            down_ratio=settings["down_ratio"],
            peak_thresh=settings["peak_thresh"],
            topk=settings["topk"],
        )
        chain_predictions = make_chain_predictions(raw_predictions, args)

        raw_center_acc.update(raw_predictions, batch["gt_centers"], batch["gt_count"])
        chain_center_acc.update(chain_predictions, batch["gt_centers"], batch["gt_count"])

        gt_centers_np = batch["gt_centers"].detach().cpu().numpy()
        gt_corners_np = batch["gt_corners"].detach().cpu().numpy()
        gt_count_np = batch["gt_count"].detach().cpu().numpy().astype(int)

        for sample_index, (raw_prediction, chain_prediction) in enumerate(zip(raw_predictions, chain_predictions)):
            count = int(gt_count_np[sample_index])
            gt_centers = gt_centers_np[sample_index, :count]
            gt_corners = gt_corners_np[sample_index, :count]
            file_name = str(batch["file_name"][sample_index])
            image_id = int(batch["image_id"][sample_index])
            source_dataset = str(batch["source_dataset"][sample_index])

            raw_stats = raw_match_totals.update(raw_prediction, gt_centers, gt_corners)
            chain_stats = chain_match_totals.update(chain_prediction, gt_centers, gt_corners)
            gt_cobb = calculate_cobb_angles(gt_corners)
            raw_cobb_stats = raw_cobb_totals.update(gt_cobb, calculate_cobb_angles(raw_prediction["corners"]))
            chain_cobb_stats = chain_cobb_totals.update(gt_cobb, calculate_cobb_angles(chain_prediction["corners"]))
            raw_pred_count = len(raw_prediction["centers"])
            chain_pred_count = len(chain_prediction["centers"])
            raw_prediction_count += raw_pred_count
            chain_prediction_count += chain_pred_count
            image_count += 1

            per_image_rows.append(
                {
                    "model": "raw",
                    "image": file_name,
                    "image_id": image_id,
                    "source_dataset": source_dataset,
                    "gt_count": count,
                    "pred_count": raw_pred_count,
                    "count_error": abs(raw_pred_count - count),
                    **raw_stats,
                    **raw_cobb_stats,
                }
            )
            per_image_rows.append(
                {
                    "model": "spine_chain",
                    "image": file_name,
                    "image_id": image_id,
                    "source_dataset": source_dataset,
                    "gt_count": count,
                    "pred_count": chain_pred_count,
                    "count_error": abs(chain_pred_count - count),
                    **chain_stats,
                    **chain_cobb_stats,
                }
            )

        print(f"[{batch_index}/{len(loader)}] evaluated {image_count}/{len(dataset)} images")

    loss_stats = average_stats(loss_totals, batches)
    summary_rows = [
        format_summary_row(
            name="raw",
            split=args.split,
            image_count=image_count,
            prediction_count=raw_prediction_count,
            loss_stats=loss_stats,
            center_metrics=raw_center_acc.compute(),
            match_metrics=raw_match_totals.compute(),
            cobb_metrics=raw_cobb_totals.compute(),
        ),
        format_summary_row(
            name="spine_chain",
            split=args.split,
            image_count=image_count,
            prediction_count=chain_prediction_count,
            loss_stats=loss_stats,
            center_metrics=chain_center_acc.compute(),
            match_metrics=chain_match_totals.compute(),
            cobb_metrics=chain_cobb_totals.compute(),
        ),
    ]
    metadata = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "source_dataset": args.source_dataset,
        "device": str(device),
        "settings": settings,
        "chain": {
            "duplicate_iou": args.chain_duplicate_iou,
            "duplicate_center_scale": args.chain_duplicate_center_scale,
            "score_thresh": args.chain_score_thresh,
            "score_weight": args.chain_score_weight,
            "min_len": args.chain_min_len,
        },
    }
    return summary_rows, per_image_rows, metadata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, per_image_rows, metadata = evaluate(args)

    metrics_csv = args.output_dir / "metrics.csv"
    metrics_json = args.output_dir / "metrics.json"
    per_image_csv = args.output_dir / "per_image_metrics.csv"
    write_csv(metrics_csv, SUMMARY_FIELDS, summary_rows)
    write_csv(per_image_csv, PER_IMAGE_FIELDS, per_image_rows)
    with metrics_json.open("w", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "metrics": summary_rows}, file, indent=2)

    print()
    print("model, f1@12, recall@12, count_mae, center_mae@12, corner_mae@12, cobb_mae")
    for row in summary_rows:
        print(
            "{model}, {center_f1_12px:.4f}, {center_recall_12px:.4f}, {count_mae:.3f}, {center_mae}, {corner_mae}, {cobb_mae}".format(
                model=row["model"],
                center_f1_12px=float(row["center_f1_12px"]),
                center_recall_12px=float(row["center_recall_12px"]),
                count_mae=float(row["count_mae"]),
                center_mae=metric_csv_value(row["center_mae_12px"]),
                corner_mae=metric_csv_value(row["corner_mae_12px"]),
                cobb_mae=metric_csv_value(row["cobb_mae_deg"]),
            )
        )
    print("saved:", metrics_csv)
    print("saved:", metrics_json)
    print("saved:", per_image_csv)


if __name__ == "__main__":
    main()
