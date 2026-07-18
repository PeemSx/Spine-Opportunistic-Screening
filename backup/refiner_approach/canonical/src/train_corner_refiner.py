from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):
        return iterable

from src.data.corner_refiner_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CornerRefinerCocoDataset,
    SourceBalancedImageBatchSampler,
)
from src.evaluation.corner_refiner_decode import decode_corner_heatmaps
from src.evaluation.corner_refiner_metrics import CornerMetricAccumulator, map_crop_predictions_to_original
from src.models.corner_refiner import CornerRefinerHRNetW18
from src.training.corner_refiner_loss import CornerRefinerLoss


CORNER_ORDER = ["TL", "TR", "BL", "BR"]
LOG_FIELDS = [
    "epoch", "train_loss", "train_coordinate_loss", "train_js_loss",
    "val_loss", "val_coordinate_loss", "val_js_loss", "mean_nme", "median_nme", "p95_nme",
    "mean_crop_coordinate_error_px", "mean_original_pixel_error", "pck_0.02", "pck_0.05", "pck_0.10",
    "identity_accuracy", "identity_switch_rate", "mean_identity_margin_px",
    "superior_endplate_angle_mae_deg", "inferior_endplate_angle_mae_deg",
    "left_height_mae_px", "right_height_mae_px", "height_ratio_mae", "crop_boundary_peak_rate",
    "backbone_lr", "head_lr",
]
ESSENTIAL_ARTIFACTS = [
    "last.pt", "best_nme.pt", "best_loss.pt", "train_log.csv", "training_config.json", "validation_metrics.json"
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_compatible(item) for item in value]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(json_compatible(payload), file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def backup_artifacts(run_dir: Path, backup_dir: Path | None) -> None:
    if backup_dir is None:
        return
    for name in ESSENTIAL_ARTIFACTS:
        source = run_dir / name
        if source.exists():
            atomic_copy(source, backup_dir / name)


def package_versions() -> dict[str, str]:
    packages = ["torch", "torchvision", "timm", "numpy", "opencv-python-headless"]
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def dataset_counts(dataset: CornerRefinerCocoDataset) -> dict[str, Any]:
    sources: dict[str, int] = {}
    for record in dataset.records:
        sources[record.source_dataset] = sources.get(record.source_dataset, 0) + 1
    return {
        "annotations": len(dataset),
        "images": len(dataset.image_record_indices),
        "annotations_by_source": dict(sorted(sources.items())),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def amp_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def average_totals(totals: dict[str, float], batches: int) -> dict[str, float]:
    if batches == 0:
        raise RuntimeError("Data loader produced no batches")
    return {key: value / batches for key, value in totals.items()}


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CornerRefinerLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
    gradient_clip: float,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "coordinate_loss": 0.0, "js_loss": 0.0}
    batches = 0
    for batch in tqdm(loader, desc="train", leave=False):
        if max_batches is not None and batches >= max_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(use_amp):
            logits = model(batch["input"])["heatmap_logits"]
            losses = criterion(logits, batch["target_heatmaps"], batch["target_points_crop"])
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        batches += 1
        for key in totals:
            totals[key] += float(losses[key].detach())
    return average_totals(totals, batches)


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CornerRefinerLoss,
    device: torch.device,
    *,
    use_amp: bool,
    crop_size: int,
    max_batches: int | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    totals = {"loss": 0.0, "coordinate_loss": 0.0, "js_loss": 0.0}
    crop_error_total = 0.0
    crop_corner_count = 0
    batches = 0
    accumulator = CornerMetricAccumulator()
    for batch in tqdm(loader, desc="validate", leave=False):
        if max_batches is not None and batches >= max_batches:
            break
        batch = move_batch(batch, device)
        with amp_context(use_amp):
            logits = model(batch["input"])["heatmap_logits"]
            losses = criterion(logits, batch["target_heatmaps"], batch["target_points_crop"])
        decoded = decode_corner_heatmaps(logits.float(), crop_size=crop_size)
        predicted_crop = decoded["points_crop"]
        predicted_original = map_crop_predictions_to_original(predicted_crop, batch["crop_to_original"].float())
        accumulator.update(
            predicted_original,
            batch["gt_points_original"],
            batch["gt_bbox_diagonal"],
            batch["source_dataset"],
            decoded["boundary_peaks"],
            batch["image_id"],
            batch["annotation_id"],
        )
        crop_error_total += float(
            torch.linalg.vector_norm(predicted_crop - batch["target_points_crop"], dim=-1).sum()
        )
        crop_corner_count += int(predicted_crop.shape[0] * predicted_crop.shape[1])
        batches += 1
        for key in totals:
            totals[key] += float(losses[key].detach())
    metrics = accumulator.compute()
    metrics["mean_crop_coordinate_error_px"] = crop_error_total / max(crop_corner_count, 1)
    return average_totals(totals, batches), metrics


def build_loaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[CornerRefinerCocoDataset, CornerRefinerCocoDataset, DataLoader, DataLoader, SourceBalancedImageBatchSampler | None]:
    overfit = args.overfit_samples > 0
    train_dataset = CornerRefinerCocoDataset(
        args.dataset_root,
        "train",
        crop_size=args.input_size,
        heatmap_size=args.heatmap_size,
        crop_scale=args.crop_scale,
        heatmap_sigma=args.heatmap_sigma,
        augment=not overfit,
        limit=args.overfit_samples if overfit else None,
    )
    val_dataset = train_dataset if overfit else CornerRefinerCocoDataset(
        args.dataset_root,
        "val",
        crop_size=args.input_size,
        heatmap_size=args.heatmap_size,
        crop_scale=args.crop_scale,
        heatmap_sigma=args.heatmap_sigma,
        augment=False,
    )
    common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    sampler: SourceBalancedImageBatchSampler | None = None
    if overfit:
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, **common
        )
    else:
        sampler = SourceBalancedImageBatchSampler(
            train_dataset, batch_size=args.batch_size, seed=args.seed, drop_last=False
        )
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, **common)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **common)
    return train_dataset, val_dataset, train_loader, val_loader, sampler


def build_optimizer(model: CornerRefinerHRNetW18, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_parameters = list(model.projections.parameters()) + list(model.refiner_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr, "name": "backbone"},
            {"params": head_parameters, "lr": args.head_lr, "name": "refiner_head"},
        ],
        weight_decay=args.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    def schedule_for(base_lr: float):
        minimum_multiplier = args.min_lr / base_lr

        def schedule(step: int) -> float:
            if step < args.warmup_epochs:
                return float(step + 1) / max(args.warmup_epochs, 1)
            decay_steps = max(args.epochs - args.warmup_epochs, 1)
            progress = min(max((step - args.warmup_epochs) / decay_steps, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_multiplier + (1.0 - minimum_multiplier) * cosine

        return schedule

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[schedule_for(args.backbone_lr), schedule_for(args.head_lr)],
    )


def checkpoint_payload(
    model: CornerRefinerHRNetW18,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    args: argparse.Namespace,
    val_loss: float,
    metrics: dict[str, Any],
    best_nme: float,
    best_loss: float,
    epochs_without_improvement: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": json_compatible(vars(args)),
        "model_args": model.model_arguments(),
        "val_loss": float(val_loss),
        "metrics": json_compatible(metrics),
        "best_nme": float(best_nme),
        "best_loss": float(best_loss),
        "epochs_without_improvement": int(epochs_without_improvement),
        "metadata": json_compatible(metadata),
    }


def append_log(path: Path, row: dict[str, Any]) -> None:
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as file:
            existing_rows = list(csv.DictReader(file))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    config_args, _ = config_parser.parse_known_args()
    config_defaults: dict[str, Any] = {}
    if config_args.config.is_file():
        with config_args.config.open("r", encoding="utf-8") as file:
            loaded_config = yaml.safe_load(file) or {}
        if not isinstance(loaded_config, dict):
            raise ValueError(f"Refiner config must be a mapping: {config_args.config}")
        config_defaults = loaded_config.get("corner_refiner", loaded_config)
        if not isinstance(config_defaults, dict):
            raise ValueError(
                f"'corner_refiner' must be a mapping in refiner config: {config_args.config}"
            )

    parser = argparse.ArgumentParser(description="Train the oracle-ROI four-corner vertebra refiner.")
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/processed/coco"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/refiner_runs"))
    parser.add_argument("--experiment-name", default="hrnet_w18_dsnt_full_coco")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--crop-scale", type=float, default=1.5)
    parser.add_argument("--heatmap-sigma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--allow-failed-overfit", action="store_true")
    valid_destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(config_defaults) - valid_destinations)
    if unknown:
        raise ValueError(f"Unknown refiner config keys: {unknown}")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    for field in ("config", "dataset_root", "output_dir", "backup_dir", "resume_checkpoint"):
        value = getattr(args, field)
        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_amp = bool(device.type == "cuda" and not args.no_amp)

    run_dir = args.output_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = args.backup_dir / args.experiment_name if args.backup_dir else None
    train_dataset, val_dataset, train_loader, val_loader, balanced_sampler = build_loaders(args, device)
    load_pretrained = not args.no_pretrained and args.resume_checkpoint is None
    model = CornerRefinerHRNetW18(
        pretrained=load_pretrained,
        output_size=args.heatmap_size,
    ).to(device)
    # Preserve the original training policy in metadata even though a resumed run
    # initializes directly from its checkpoint and does not need an ImageNet download.
    model.pretrained = not args.no_pretrained
    criterion = CornerRefinerLoss(crop_size=args.input_size)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metadata = {
        "model_arguments": model.model_arguments(),
        "corner_order": CORNER_ORDER,
        "crop_policy": {
            "base_scale": args.crop_scale,
            "center_jitter_fraction": 0.08,
            "scale_multiplier_range": [0.90, 1.10],
            "containment_retries": 10,
            "fallback": "deterministic_annotation_roi",
            "output_size": args.input_size,
        },
        "normalization": {
            "colorspace": "grayscale_repeated_to_three_channels",
            "mean": IMAGENET_MEAN[:, 0, 0].tolist(),
            "std": IMAGENET_STD[:, 0, 0].tolist(),
        },
        "package_versions": package_versions(),
        "dataset": {
            "root": str(args.dataset_root),
            "train": dataset_counts(train_dataset),
            "validation": dataset_counts(val_dataset),
        },
    }
    atomic_json(run_dir / "training_config.json", {"arguments": vars(args), "metadata": metadata})

    start_epoch, best_nme, best_loss, epochs_without_improvement = 1, float("inf"), float("inf"), 0
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_nme = float(checkpoint.get("best_nme", checkpoint.get("metrics", {}).get("mean_nme", float("inf"))))
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("val_loss", float("inf"))))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))

    print(
        f"device={device} amp={use_amp} train={len(train_dataset)} val={len(val_dataset)}",
        flush=True,
    )
    print(f"run_dir={run_dir}", flush=True)
    if args.resume_checkpoint:
        print(f"resumed={args.resume_checkpoint} start_epoch={start_epoch}", flush=True)

    final_metrics: dict[str, Any] | None = None
    for epoch in range(start_epoch, args.epochs + 1):
        if balanced_sampler is not None:
            balanced_sampler.set_epoch(epoch)
        train_stats = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_amp=use_amp, gradient_clip=args.gradient_clip, max_batches=args.max_train_batches,
        )
        val_stats, metrics = validate_epoch(
            model, val_loader, criterion, device,
            use_amp=use_amp, crop_size=args.input_size, max_batches=args.max_val_batches,
        )
        final_metrics = metrics
        scheduler.step()

        nme_improved = metrics["mean_nme"] < best_nme
        loss_improved = val_stats["loss"] < best_loss
        if nme_improved:
            best_nme = float(metrics["mean_nme"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if loss_improved:
            best_loss = float(val_stats["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_coordinate_loss": train_stats["coordinate_loss"],
            "train_js_loss": train_stats["js_loss"],
            "val_loss": val_stats["loss"],
            "val_coordinate_loss": val_stats["coordinate_loss"],
            "val_js_loss": val_stats["js_loss"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            **metrics,
        }
        append_log(run_dir / "train_log.csv", row)
        atomic_json(run_dir / "validation_metrics.json", {"epoch": epoch, "loss": val_stats, "metrics": metrics})
        payload = checkpoint_payload(
            model, optimizer, scheduler, scaler, epoch, args, val_stats["loss"], metrics,
            best_nme, best_loss, epochs_without_improvement, metadata,
        )
        atomic_torch_save(run_dir / "last.pt", payload)
        if nme_improved:
            atomic_torch_save(run_dir / "best_nme.pt", payload)
        if loss_improved:
            atomic_torch_save(run_dir / "best_loss.pt", payload)
        backup_artifacts(run_dir, backup_dir)

        print(
            f"epoch {epoch:02d}/{args.epochs} train={train_stats['loss']:.5f} "
            f"val={val_stats['loss']:.5f} nme={metrics['mean_nme']:.5f} "
            f"crop_px={metrics['mean_crop_coordinate_error_px']:.3f}",
            flush=True,
        )
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"early stopping: mean NME did not improve for {args.early_stop_patience} epochs")
            break

    if args.overfit_samples > 0 and final_metrics is not None:
        passed = (
            final_metrics["mean_crop_coordinate_error_px"] < 2.0
            and final_metrics["mean_nme"] < 0.01
        )
        print(f"overfit_gate={'PASS' if passed else 'FAIL'}")
        if not passed and not args.allow_failed_overfit:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
