from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from src.data.dataset_provenance import build_dataset_provenance
from src.evaluate_centernet import load_checkpoint
from src.models.centernet import HRNET_PRETRAINED_BACKBONE_IDS


EXPERIMENT_NAME = (
    "centernet_hrnet_w18_imagenet_pretrained_existing_loss_"
    "coco_nih_lumos_seed20260627"
)
PRETRAINED_BACKBONE_ID = HRNET_PRETRAINED_BACKBONE_IDS["hrnet_w18"]
EXPECTED_DATASET_FINGERPRINT = (
    "d99fbd046ece0cd2d10598fa1fb1da0eed24cb8542270f54a732616c8cb9e8f3"
)
RESUME_ARGUMENT_CONTRACT: dict[str, Any] = {
    "experiment_name": EXPERIMENT_NAME,
    "backbone": "hrnet_w18",
    "pretrained": True,
    "input_size": 1024,
    "down_ratio": 4,
    "max_objects": 64,
    "batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "hm_weight": 1.0,
    "reg_weight": 1.0,
    "wh_weight": 0.5,
    "seed": 20260627,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled ImageNet-pretrained HRNet-W18 baseline with "
            "the existing CenterNet loss and the frozen 80-epoch protocol."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved training command without starting training.",
    )
    return parser.parse_args()


def build_training_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "src.train_centernet",
        "--config",
        str(args.config),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(args.output_dir),
        "--experiment-name",
        EXPERIMENT_NAME,
        "--backbone",
        "hrnet_w18",
        "--pretrained",
        "--input-size",
        "1024",
        "--down-ratio",
        "4",
        "--max-objects",
        "64",
        "--epochs",
        "80",
        "--batch-size",
        "8",
        "--num-workers",
        str(args.num_workers),
        "--lr",
        "1e-4",
        "--weight-decay",
        "1e-4",
        "--hm-weight",
        "1.0",
        "--reg-weight",
        "1.0",
        "--wh-weight",
        "0.5",
        "--peak-thresh",
        "0.10",
        "--eval-topk",
        "50",
        "--val-chain-duplicate-iou",
        "0.18",
        "--val-chain-duplicate-center-scale",
        "0.35",
        "--val-chain-score-thresh",
        "0.18",
        "--val-chain-score-weight",
        "3.0",
        "--val-chain-min-len",
        "3",
        "--seed",
        "20260627",
        "--amp",
        "--early-stop-patience",
        "15",
        "--early-stop-metric",
        "center_f1_12px",
        "--save-preview-every",
        "5",
        "--preview-images",
        "4",
        "--progress-every",
        "25",
    ]
    if args.backup_dir is not None:
        command.extend(["--backup-dir", str(args.backup_dir)])
    if args.resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(args.resume_checkpoint)])
    return command


def validate_dataset(dataset_root: Path) -> dict[str, Any]:
    provenance = build_dataset_provenance(dataset_root)
    observed = str(provenance.get("fingerprint", ""))
    if observed != EXPECTED_DATASET_FINGERPRINT:
        raise ValueError(
            "This experiment requires dataset fingerprint "
            f"{EXPECTED_DATASET_FINGERPRINT}, got {observed or '<missing>'}"
        )
    return provenance


def validate_resume_checkpoint(path: Path) -> None:
    checkpoint = load_checkpoint(path.resolve(strict=True), torch.device("cpu"))
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Resume checkpoint is missing its training argument contract")
    mismatches = {
        name: {"expected": expected, "observed": checkpoint_args.get(name)}
        for name, expected in RESUME_ARGUMENT_CONTRACT.items()
        if checkpoint_args.get(name) != expected
    }
    provenance = checkpoint.get("dataset_provenance")
    observed_fingerprint = (
        provenance.get("fingerprint") if isinstance(provenance, dict) else None
    )
    if observed_fingerprint != EXPECTED_DATASET_FINGERPRINT:
        mismatches["dataset_fingerprint"] = {
            "expected": EXPECTED_DATASET_FINGERPRINT,
            "observed": observed_fingerprint,
        }
    if mismatches:
        details = ", ".join(
            f"{name}={item['observed']!r} (expected {item['expected']!r})"
            for name, item in sorted(mismatches.items())
        )
        raise ValueError(f"Resume checkpoint does not match this experiment: {details}")


def main() -> None:
    args = parse_args()
    provenance = validate_dataset(args.dataset_root)
    if args.resume_checkpoint is not None:
        validate_resume_checkpoint(args.resume_checkpoint)
    command = build_training_command(args)
    print("Experiment:", EXPERIMENT_NAME, flush=True)
    print("Pretrained backbone:", PRETRAINED_BACKBONE_ID, flush=True)
    print("Dataset fingerprint:", provenance["fingerprint"], flush=True)
    print("Command:", " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
