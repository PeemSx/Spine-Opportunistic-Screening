from __future__ import annotations

import argparse
import csv
import hashlib
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
from src.data.inference_geometry import (
    TransformMeta,
    map_points_to_original,
    valid_center_mask,
)
from src.evaluation.center_metrics import CenterMetricAccumulator
from src.evaluation.centernet_decode import decode_centernet_outputs
from src.evaluation.cobb_angle import CobbResult, calculate_cobb_angles, cobb_smape
from src.evaluation.config import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_PEAK_THRESHOLD,
)
from src.evaluation.scorecard import (
    ImageEvaluation,
    bootstrap_confidence_intervals,
    evaluate_acceptance_gates,
    evaluate_prediction,
    postprocessing_comparison,
    source_summary_rows,
)
from src.models.centernet import build_centernet_model
from src.postprocessing.spine_chain import (
    candidates_to_prediction,
    prediction_to_candidates,
    select_spine_chain,
)
from src.training.centernet_loss import CenterNetLoss


SCHEMA_VERSION = 2
DEPLOYED_ARTIFACT_TYPE = "spine_centernet_inference"
DEPLOYED_PREPROCESSING_CONTRACT = {
    "color_space": "RGB",
    "resize": "longest_side_preserving_aspect_ratio",
    "padding": "centered_zero_padding",
    "normalization": "pixel_value / 255.0 - 0.5",
}
DEPLOYED_OUTPUT_CONTRACT = {
    "corner_order": ["TL", "TR", "BL", "BR"],
    "point_order": ["x", "y"],
    "coordinate_space": "original_image_pixels",
}
DEFAULT_DEPLOYED_CHECKPOINT = Path(
    "../Spine-Screening-API/app/weight/best_center_f1.pt"
)
DEFAULT_RESEARCH_CHECKPOINT = Path("src/weights/hrnet_nih/best_center_f1.pt")
DEFAULT_DATASET_ROOT = Path("dataset/processed/coco_nih")

LEGACY_SUMMARY_FIELDS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CenterNet raw and spine-chain predictions with schema-v2 metrics."
    )
    parser.add_argument(
        "--evaluation-profile",
        choices=["deployed", "research"],
        default="deployed",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--source-dataset", type=str, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation_v2"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--down-ratio", type=int, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--peak-thresh", type=float, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chain-duplicate-iou", type=float, default=None)
    parser.add_argument("--chain-duplicate-center-scale", type=float, default=None)
    parser.add_argument("--chain-score-thresh", type=float, default=None)
    parser.add_argument("--chain-score-weight", type=float, default=None)
    parser.add_argument("--chain-min-len", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--strict-gates", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary")
    return checkpoint


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


def _value_at(value: Any, index: int) -> Any:
    item = value[index]
    return item.item() if torch.is_tensor(item) else item


def transform_meta_from_batch(batch: dict[str, Any], index: int) -> TransformMeta:
    return TransformMeta(
        original_width=int(_value_at(batch["original_width"], index)),
        original_height=int(_value_at(batch["original_height"], index)),
        resized_width=int(_value_at(batch["resized_width"], index)),
        resized_height=int(_value_at(batch["resized_height"], index)),
        pad_left=int(_value_at(batch["pad_left"], index)),
        pad_top=int(_value_at(batch["pad_top"], index)),
        scale=float(_value_at(batch["scale"], index)),
        input_size=int(_value_at(batch["input_size"], index)),
    )


def prediction_to_original(
    prediction: dict[str, np.ndarray],
    meta: TransformMeta,
) -> dict[str, np.ndarray]:
    centers = map_points_to_original(prediction["centers"], meta)
    corners = map_points_to_original(prediction["corners"], meta)
    mask = valid_center_mask(centers, meta)
    return {
        "scores": np.asarray(prediction["scores"][mask], dtype=np.float32),
        "centers": np.asarray(centers[mask], dtype=np.float32),
        "corners": np.asarray(corners[mask], dtype=np.float32),
    }


def make_chain_prediction(
    prediction: dict[str, np.ndarray],
    chain: dict[str, Any],
) -> dict[str, np.ndarray]:
    _, candidates, _ = select_spine_chain(
        prediction_to_candidates(prediction),
        duplicate_iou_threshold=chain["duplicate_iou"],
        duplicate_center_scale=chain["duplicate_center_scale"],
        score_threshold=chain["score_thresh"],
        score_weight=chain["score_weight"],
        min_chain_len=chain["min_len"],
    )
    return candidates_to_prediction(candidates)


def _requested_deployed_overrides(args: argparse.Namespace) -> list[str]:
    names = [
        "input_size",
        "down_ratio",
        "max_objects",
        "peak_thresh",
        "topk",
        "chain_duplicate_iou",
        "chain_duplicate_center_scale",
        "chain_score_thresh",
        "chain_score_weight",
        "chain_min_len",
    ]
    return [name for name in names if getattr(args, name) is not None]


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.evaluation_profile == "deployed":
        return DEFAULT_DEPLOYED_CHECKPOINT
    return DEFAULT_RESEARCH_CHECKPOINT


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint field {name!r} must be a mapping")
    return value


def build_model_and_settings(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], Path]:
    checkpoint_path = resolve_checkpoint_path(args).resolve(strict=True)
    checkpoint = load_checkpoint(checkpoint_path, device)
    train_args = _mapping(checkpoint.get("args", {}), "args")
    embedded_peak = train_args.get("peak_thresh")
    artifact_type = checkpoint.get("artifact_type")

    if args.evaluation_profile == "deployed":
        overrides = _requested_deployed_overrides(args)
        if overrides:
            raise ValueError(
                "Deployed evaluation rejects metric-affecting overrides: "
                + ", ".join(overrides)
            )
        if artifact_type != DEPLOYED_ARTIFACT_TYPE:
            raise ValueError(
                "Deployed evaluation requires a compact spine_centernet_inference artifact"
            )
        preprocessing = _mapping(
            checkpoint.get("preprocessing"),
            "preprocessing",
        )
        output_contract = _mapping(
            checkpoint.get("output_contract"),
            "output_contract",
        )
        if preprocessing != DEPLOYED_PREPROCESSING_CONTRACT:
            raise ValueError(
                "Deployed artifact preprocessing contract does not match the "
                "evaluation implementation"
            )
        if output_contract != DEPLOYED_OUTPUT_CONTRACT:
            raise ValueError(
                "Deployed artifact output contract does not match the "
                "evaluation implementation"
            )
        effective_peak = float(train_args["peak_thresh"])
        if not np.isclose(effective_peak, DEFAULT_PEAK_THRESHOLD):
            raise ValueError(
                "Deployed artifact peak threshold must be "
                f"{DEFAULT_PEAK_THRESHOLD:.2f}, got {effective_peak:.6f}"
            )
        postprocessing = _mapping(checkpoint.get("postprocessing"), "postprocessing")
        if postprocessing.get("spine_chain_enabled") is not True:
            raise ValueError("Deployed artifact must enable spine-chain selection")
        chain = {
            "duplicate_iou": float(postprocessing["duplicate_iou_threshold"]),
            "duplicate_center_scale": float(postprocessing["duplicate_center_scale"]),
            "score_thresh": float(postprocessing["score_threshold"]),
            "score_weight": float(postprocessing["score_weight"]),
            "min_len": int(postprocessing["min_chain_len"]),
        }
    else:
        effective_peak = float(
            args.peak_thresh
            if args.peak_thresh is not None
            else DEFAULT_PEAK_THRESHOLD
        )
        chain = {
            "duplicate_iou": float(
                args.chain_duplicate_iou
                if args.chain_duplicate_iou is not None
                else 0.18
            ),
            "duplicate_center_scale": float(
                args.chain_duplicate_center_scale
                if args.chain_duplicate_center_scale is not None
                else 0.35
            ),
            "score_thresh": float(
                args.chain_score_thresh
                if args.chain_score_thresh is not None
                else 0.18
            ),
            "score_weight": float(
                args.chain_score_weight
                if args.chain_score_weight is not None
                else 3.0
            ),
            "min_len": int(
                args.chain_min_len
                if args.chain_min_len is not None
                else 3
            ),
        }

    backbone = str(train_args.get("backbone", "hrnet_w18"))
    input_size = int(
        args.input_size
        if args.input_size is not None
        else train_args.get("input_size", 1024)
    )
    down_ratio = int(
        args.down_ratio
        if args.down_ratio is not None
        else train_args.get("down_ratio", 4)
    )
    max_objects = int(args.max_objects if args.max_objects is not None else 64)
    topk = int(
        args.topk
        if args.topk is not None
        else train_args.get("eval_topk", 50)
    )

    model = build_centernet_model(backbone=backbone, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    settings = {
        "backbone": backbone,
        "input_size": input_size,
        "down_ratio": down_ratio,
        "max_objects": max_objects,
        "peak_thresh": effective_peak,
        "topk": topk,
        "hm_weight": float(train_args.get("hm_weight", 1.0)),
        "reg_weight": float(train_args.get("reg_weight", 1.0)),
        "wh_weight": float(train_args.get("wh_weight", 0.1)),
        "chain": chain,
    }
    checkpoint_metadata = {
        "artifact_type": artifact_type,
        "format_version": checkpoint.get("format_version"),
        "embedded_peak_thresh": (
            float(embedded_peak) if embedded_peak is not None else None
        ),
        "effective_peak_thresh": effective_peak,
        "nonstandard_peak_threshold": not np.isclose(
            effective_peak,
            DEFAULT_PEAK_THRESHOLD,
        ),
        "preprocessing": checkpoint.get("preprocessing"),
        "output_contract": checkpoint.get("output_contract"),
    }
    return model, settings, checkpoint_metadata, checkpoint_path


def greedy_match_pairs(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    max_distance_px: float,
) -> list[tuple[int, int, float]]:
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return []
    distances = np.linalg.norm(
        pred_centers[:, None, :] - gt_centers[None, :, :],
        axis=2,
    )
    pairs = sorted(
        np.argwhere(distances <= float(max_distance_px)),
        key=lambda pair: float(distances[pair[0], pair[1]]),
    )
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches = []
    for pred_index, gt_index in pairs:
        pred_index = int(pred_index)
        gt_index = int(gt_index)
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append(
            (pred_index, gt_index, float(distances[pred_index, gt_index]))
        )
    return matches


@dataclass
class LegacyMatchTotals:
    matched: int = 0
    center_error_sum: float = 0.0
    corner_error_sum: float = 0.0

    def update(
        self,
        prediction: dict[str, np.ndarray],
        gt_centers: np.ndarray,
        gt_corners: np.ndarray,
    ) -> None:
        pairs = greedy_match_pairs(prediction["centers"], gt_centers, 12.0)
        for pred_index, gt_index, center_distance in pairs:
            self.matched += 1
            self.center_error_sum += center_distance
            self.corner_error_sum += float(
                np.linalg.norm(
                    prediction["corners"][pred_index] - gt_corners[gt_index],
                    axis=1,
                ).mean()
            )

    def compute(self) -> dict[str, Any]:
        return {
            "matched_12px": self.matched,
            "center_mae_12px": (
                self.center_error_sum / self.matched if self.matched else None
            ),
            "corner_mae_12px": (
                self.corner_error_sum / self.matched if self.matched else None
            ),
        }


@dataclass
class LegacyCobbTotals:
    valid_images: int = 0
    error_sum: float = 0.0
    smape_sum: float = 0.0
    within_5: int = 0
    within_10: int = 0

    def update(self, gt_result: CobbResult, pred_result: CobbResult) -> None:
        if (
            not gt_result.valid
            or not pred_result.valid
            or gt_result.cobb_1_deg is None
            or pred_result.cobb_1_deg is None
        ):
            return
        error = abs(float(pred_result.cobb_1_deg) - float(gt_result.cobb_1_deg))
        self.valid_images += 1
        self.error_sum += error
        self.smape_sum += cobb_smape(
            float(gt_result.cobb_1_deg),
            float(pred_result.cobb_1_deg),
        )
        self.within_5 += int(error <= 5.0)
        self.within_10 += int(error <= 10.0)

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
            "cobb_within_5deg_rate": self.within_5 / self.valid_images,
            "cobb_within_10deg_rate": self.within_10 / self.valid_images,
        }


def format_legacy_summary(
    *,
    model: str,
    split: str,
    images: int,
    predictions: int,
    losses: dict[str, float],
    center: CenterMetricAccumulator,
    match: LegacyMatchTotals,
    cobb: LegacyCobbTotals,
) -> dict[str, Any]:
    return {
        "model": model,
        "split": split,
        "images": images,
        "predictions": predictions,
        "test_loss": losses["loss"],
        "test_hm_loss": losses["hm_loss"],
        "test_reg_loss": losses["reg_loss"],
        "test_wh_loss": losses["wh_loss"],
        **center.compute(),
        **match.compute(),
        **cobb.compute(),
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    model, settings, checkpoint_metadata, checkpoint_path = build_model_and_settings(
        args,
        device,
    )
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

    evaluations: dict[str, list[ImageEvaluation]] = {
        "raw": [],
        "spine_chain": [],
    }
    legacy_center = {
        "raw": CenterMetricAccumulator(),
        "spine_chain": CenterMetricAccumulator(),
    }
    legacy_match = {
        "raw": LegacyMatchTotals(),
        "spine_chain": LegacyMatchTotals(),
    }
    legacy_cobb = {
        "raw": LegacyCobbTotals(),
        "spine_chain": LegacyCobbTotals(),
    }
    legacy_prediction_count = {"raw": 0, "spine_chain": 0}
    loss_totals = {"loss": 0.0, "hm_loss": 0.0, "reg_loss": 0.0, "wh_loss": 0.0}
    batches = 0
    image_count = 0

    for batch_index, batch in enumerate(loader, start=1):
        batch_on_device = move_batch_to_device(batch, device)
        outputs = model(batch_on_device["input"])
        loss_dict = criterion.loss_dict(outputs, batch_on_device)
        for key in loss_totals:
            loss_totals[key] += float(loss_dict[key].detach().cpu())
        batches += 1

        raw_padded = decode_centernet_outputs(
            outputs,
            down_ratio=settings["down_ratio"],
            peak_thresh=settings["peak_thresh"],
            topk=settings["topk"],
        )
        legacy_chain_padded = [
            make_chain_prediction(prediction, settings["chain"])
            for prediction in raw_padded
        ]
        legacy_center["raw"].update(
            raw_padded,
            batch["gt_centers"],
            batch["gt_count"],
        )
        legacy_center["spine_chain"].update(
            legacy_chain_padded,
            batch["gt_centers"],
            batch["gt_count"],
        )

        gt_centers_padded = batch["gt_centers"].detach().cpu().numpy()
        gt_corners_padded = batch["gt_corners"].detach().cpu().numpy()
        gt_centers_original = batch["gt_centers_original"].detach().cpu().numpy()
        gt_corners_original = batch["gt_corners_original"].detach().cpu().numpy()
        gt_counts = batch["gt_count"].detach().cpu().numpy().astype(int)

        for sample_index, padded_prediction in enumerate(raw_padded):
            count = int(gt_counts[sample_index])
            meta = transform_meta_from_batch(batch, sample_index)
            raw_original = prediction_to_original(padded_prediction, meta)
            chain_original = make_chain_prediction(
                raw_original,
                settings["chain"],
            )
            file_name = str(batch["file_name"][sample_index])
            image_id = int(_value_at(batch["image_id"], sample_index))
            source_dataset = str(batch["source_dataset"][sample_index])
            cluster_id = str(batch["patient_cluster_id"][sample_index])
            gt_centers = gt_centers_original[sample_index, :count]
            gt_corners = gt_corners_original[sample_index, :count]

            for model_name, prediction in (
                ("raw", raw_original),
                ("spine_chain", chain_original),
            ):
                evaluations[model_name].append(
                    evaluate_prediction(
                        model=model_name,
                        prediction=prediction,
                        gt_centers=gt_centers,
                        gt_corners=gt_corners,
                        image=file_name,
                        image_id=image_id,
                        source_dataset=source_dataset,
                        cluster_id=cluster_id,
                        image_width=meta.original_width,
                        image_height=meta.original_height,
                    )
                )

            for model_name, prediction in (
                ("raw", padded_prediction),
                ("spine_chain", legacy_chain_padded[sample_index]),
            ):
                legacy_match[model_name].update(
                    prediction,
                    gt_centers_padded[sample_index, :count],
                    gt_corners_padded[sample_index, :count],
                )
                legacy_cobb[model_name].update(
                    calculate_cobb_angles(gt_corners_padded[sample_index, :count]),
                    calculate_cobb_angles(prediction["corners"]),
                )
                legacy_prediction_count[model_name] += len(prediction["centers"])
            image_count += 1

        print(
            f"[{batch_index}/{len(loader)}] evaluated "
            f"{image_count}/{len(dataset)} images"
        )

    losses = average_stats(loss_totals, batches)
    legacy_rows = [
        format_legacy_summary(
            model=model_name,
            split=args.split,
            images=image_count,
            predictions=legacy_prediction_count[model_name],
            losses=losses,
            center=legacy_center[model_name],
            match=legacy_match[model_name],
            cobb=legacy_cobb[model_name],
        )
        for model_name in ("raw", "spine_chain")
    ]
    legacy_by_model = {row["model"]: row for row in legacy_rows}

    summaries: dict[str, dict[str, Any]] = {}
    scorecard_rows: list[dict[str, Any]] = []
    source_rows_by_model: dict[str, list[dict[str, Any]]] = {}
    confidence_intervals: dict[str, Any] = {}
    for model_name in ("raw", "spine_chain"):
        overall, per_source, macro, worst = source_summary_rows(
            evaluations[model_name],
            model_name,
        )
        overall.update(
            {
                key: value
                for key, value in legacy_by_model[model_name].items()
                if key not in {"model", "images", "predictions", "count_mae", "cobb_mae_deg"}
            }
        )
        summaries[model_name] = {
            "overall": overall,
            "source_macro": macro,
            "worst_source": worst,
        }
        source_rows_by_model[model_name] = per_source
        scorecard_rows.extend([overall, macro, worst, *per_source])
        confidence_intervals[model_name] = bootstrap_confidence_intervals(
            evaluations[model_name],
            model=model_name,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )

    comparison = postprocessing_comparison(
        summaries["raw"]["overall"],
        summaries["spine_chain"]["overall"],
    )
    acceptance = evaluate_acceptance_gates(
        chain_overall=summaries["spine_chain"]["overall"],
        chain_source_macro=summaries["spine_chain"]["source_macro"],
        chain_worst_source=summaries["spine_chain"]["worst_source"],
        comparison=comparison,
    )
    per_image_rows = [
        item.to_row()
        for model_name in ("raw", "spine_chain")
        for item in evaluations[model_name]
    ]
    per_instance_rows = [
        row
        for model_name in ("raw", "spine_chain")
        for item in evaluations[model_name]
        for row in item.instance_rows
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_profile": args.evaluation_profile,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "source_dataset": args.source_dataset,
        "device": str(device),
        "coordinate_space": "original_image_pixels",
        "legacy_coordinate_space": "resized_padded_1024",
        "metric_protocol": {
            "matching": "one_to_one_hungarian_normalized_center_distance",
            "normalization": "gt_vertebra_bounding_box_diagonal",
            "primary_detection_gate": "0.20D",
            "detection_sensitivity_gates": ["0.10D", "0.25D"],
            "pck_thresholds": [0.05, 0.10, 0.20],
            "usable_vertebra": (
                "accepted center match, finite non-self-intersecting in-image "
                "quadrilateral, and NME <= 0.10"
            ),
            "padding_policy": (
                "predictions with centers outside the original image are "
                "removed before matching and spine-chain selection"
            ),
        },
        "settings": settings,
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "seed": int(args.bootstrap_seed),
            "cluster_unit": "patient_when_available_else_radiograph",
        },
        "medical_scope": (
            "Corner-derived geometric agreement only; outputs are not diagnostic "
            "accuracy or radiologist-reference measurements."
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "metrics": legacy_rows,
        "scorecard": {
            "rows": scorecard_rows,
            "confidence_intervals": confidence_intervals,
            "postprocessing_comparison": comparison,
            "acceptance": acceptance,
        },
        "per_image_rows": per_image_rows,
        "per_instance_rows": per_instance_rows,
    }


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return "" if not np.isfinite(numeric) else round(numeric, 6)
    return value


def fieldnames_for_rows(
    rows: list[dict[str, Any]],
    preferred: list[str],
) -> list[str]:
    available = {key for row in rows for key in row}
    return [key for key in preferred if key in available] + sorted(
        available - set(preferred)
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    preferred: list[str],
) -> None:
    fieldnames = fieldnames_for_rows(rows, preferred)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(args)
    scorecard_rows = result["scorecard"]["rows"]
    per_image_rows = result.pop("per_image_rows")
    per_instance_rows = result.pop("per_instance_rows")

    metrics_json = args.output_dir / "metrics.json"
    metrics_csv = args.output_dir / "metrics.csv"
    per_image_csv = args.output_dir / "per_image_metrics.csv"
    per_instance_csv = args.output_dir / "per_instance_metrics.csv"

    with metrics_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    write_csv(
        metrics_csv,
        scorecard_rows,
        ["model", "scope", "source_dataset", "images", "gt_vertebrae", "predictions"],
    )
    write_csv(
        per_image_csv,
        per_image_rows,
        ["model", "image", "image_id", "source_dataset", "patient_cluster_id"],
    )
    write_csv(
        per_instance_csv,
        per_instance_rows,
        [
            "model",
            "image",
            "image_id",
            "source_dataset",
            "status",
            "prediction_index",
            "gt_index",
        ],
    )

    chain = next(
        row
        for row in scorecard_rows
        if row["model"] == "spine_chain" and row["scope"] == "overall_micro"
    )
    print()
    print(
        "spine_chain | f1@0.20D={:.4f} | usable_recall={:.4f} | "
        "count_mae={:.3f} | cobb_mae={} | cobb@5={:.4f}".format(
            float(chain["center_f1_0.20d"]),
            float(chain["usable_vertebra_recall"]),
            float(chain["count_mae"]),
            csv_value(chain["cobb_mae_deg"]),
            float(chain["cobb_at_5deg"]),
        )
    )
    print("acceptance:", "PASS" if result["scorecard"]["acceptance"]["passed"] else "FAIL")
    for path in (metrics_json, metrics_csv, per_image_csv, per_instance_csv):
        print("saved:", path)
    if args.strict_gates and not result["scorecard"]["acceptance"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
