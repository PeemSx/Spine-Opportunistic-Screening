from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal local venvs.
    class _TqdmFallback:
        def __init__(self, iterable, **_: Any):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **_: Any) -> None:
            return None

    def tqdm(iterable, **kwargs: Any):
        return _TqdmFallback(iterable, **kwargs)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.centernet_dataset import CenterNetCocoDataset
from src.evaluation.center_metrics import CenterMetricAccumulator
from src.evaluation.centernet_decode import decode_centernet_outputs
from src.models.centernet import SUPPORTED_BACKBONES, build_centernet_model
from src.training.centernet_loss import CenterNetLoss


LOG_FIELDS = [
    "epoch",
    "train_loss",
    "train_hm_loss",
    "train_reg_loss",
    "train_wh_loss",
    "val_loss",
    "val_hm_loss",
    "val_reg_loss",
    "val_wh_loss",
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
    "lr",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def average_stats(total: dict[str, float], batches: int) -> dict[str, float]:
    return {key: value / max(batches, 1) for key, value in total.items()}


def run_loss_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CenterNetLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
    phase: str | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_every: int = 25,
) -> dict[str, float]:
    is_train = optimizer is not None
    phase_name = phase or ("train" if is_train else "val")
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "hm_loss": 0.0,
        "reg_loss": 0.0,
        "wh_loss": 0.0,
    }
    batches = 0

    epoch_label = (
        f"epoch {epoch}/{total_epochs} {phase_name}"
        if epoch is not None and total_epochs is not None
        else phase_name
    )
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    progress = tqdm(loader, desc=epoch_label, leave=True, dynamic_ncols=True)
    for batch in progress:
        if max_batches is not None and batches >= max_batches:
            break
        batch = move_batch_to_device(batch, device)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch["input"])
                loss_dict = criterion.loss_dict(outputs, batch)
                loss = loss_dict["loss"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batches += 1
        for key in totals:
            totals[key] += float(loss_dict[key].detach().cpu())
        running = average_stats(totals, batches)
        progress.set_postfix(
            loss=f"{running['loss']:.4f}",
            hm=f"{running['hm_loss']:.4f}",
            reg=f"{running['reg_loss']:.4f}",
            corner=f"{running['wh_loss']:.4f}",
        )
        if progress_every > 0 and (batches % progress_every == 0 or batches == total_batches):
            print(
                "{} batch {}/{} | loss {:.4f} | hm {:.4f} | reg {:.4f} | corner {:.4f}".format(
                    epoch_label,
                    batches,
                    total_batches,
                    running["loss"],
                    running["hm_loss"],
                    running["reg_loss"],
                    running["wh_loss"],
                ),
                flush=True,
            )

    return average_stats(totals, batches)


@torch.no_grad()
def evaluate_centers(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    down_ratio: int,
    peak_thresh: float,
    topk: int,
    max_batches: int | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_every: int = 25,
) -> dict[str, float]:
    model.eval()
    accumulator = CenterMetricAccumulator()
    batches = 0
    epoch_label = (
        f"epoch {epoch}/{total_epochs} center-eval"
        if epoch is not None and total_epochs is not None
        else "center-eval"
    )
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    for batch in tqdm(loader, desc=epoch_label, leave=True, dynamic_ncols=True):
        if max_batches is not None and batches >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["input"])
        decoded = decode_centernet_outputs(
            outputs,
            down_ratio=down_ratio,
            peak_thresh=peak_thresh,
            topk=topk,
        )
        accumulator.update(decoded, batch["gt_centers"], batch["gt_count"])
        batches += 1
        if progress_every > 0 and (batches % progress_every == 0 or batches == total_batches):
            print(f"{epoch_label} batch {batches}/{total_batches}", flush=True)
    return accumulator.compute()


def tensor_to_image(input_tensor: torch.Tensor) -> np.ndarray:
    image = input_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return image


@torch.no_grad()
def save_prediction_previews(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    down_ratio: int,
    peak_thresh: float,
    topk: int,
    max_images: int = 4,
) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch in loader:
        batch_on_device = move_batch_to_device(batch, device)
        outputs = model(batch_on_device["input"])
        decoded = decode_centernet_outputs(
            outputs,
            down_ratio=down_ratio,
            peak_thresh=peak_thresh,
            topk=topk,
        )

        for batch_index, prediction in enumerate(decoded):
            if saved >= max_images:
                return
            image = tensor_to_image(batch["input"][batch_index])
            heat = outputs["hm"][batch_index, 0].detach().cpu().numpy()
            heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            gt_count = int(batch["gt_count"][batch_index])
            gt_centers = batch["gt_centers"][batch_index, :gt_count].detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(8, 10))
            ax.imshow(image)
            ax.imshow(heat, cmap="jet", alpha=0.42, vmin=0, vmax=1)
            if len(prediction["centers"]) > 0:
                ax.scatter(
                    prediction["centers"][:, 0],
                    prediction["centers"][:, 1],
                    s=16,
                    c="cyan",
                    edgecolors="black",
                    linewidths=0.3,
                    label="pred",
                )
            if len(gt_centers) > 0:
                ax.scatter(
                    gt_centers[:, 0],
                    gt_centers[:, 1],
                    s=16,
                    c="magenta",
                    edgecolors="black",
                    linewidths=0.3,
                    label="gt",
                )
            ax.set_title(
                f"epoch {epoch} | pred={len(prediction['centers'])} | gt={gt_count} | {batch['file_name'][batch_index]}",
                fontsize=9,
            )
            ax.axis("off")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"epoch_{epoch:03d}_{saved:02d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved += 1


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    args: argparse.Namespace,
    val_loss: float,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "args": vars(args),
            "val_loss": val_loss,
            "metrics": metrics,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    epoch = int(checkpoint.get("epoch", 0))
    return epoch + 1, checkpoint


def backup_run_files(run_dir: Path, backup_dir: Path | None) -> None:
    if backup_dir is None:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ["last.pt", "best_loss.pt", "best_center_f1.pt", "train_log.csv"]:
        src = run_dir / name
        if src.exists():
            destination = backup_dir / name
            temporary = destination.with_name(destination.name + ".tmp")
            shutil.copy2(src, temporary)
            temporary.replace(destination)


def append_log(log_path: Path, row: dict[str, float]) -> None:
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


def read_best_values_from_log(log_path: Path, early_stop_metric: str) -> tuple[float, float, float | None]:
    best_loss = float("inf")
    best_center_f1 = -1.0
    best_early_stop_metric: float | None = None
    if not log_path.exists():
        return best_loss, best_center_f1, best_early_stop_metric

    with log_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            try:
                val_loss = float(row.get("val_loss", ""))
                center_f1 = float(row.get("center_f1_12px", ""))
            except ValueError:
                continue
            best_loss = min(best_loss, val_loss)
            best_center_f1 = max(best_center_f1, center_f1)

            metric_text = row.get(early_stop_metric, "")
            if metric_text == "":
                continue
            try:
                metric_value = float(metric_text)
            except ValueError:
                continue
            if metric_improved(metric_value, best_early_stop_metric, early_stop_metric, min_delta=0.0):
                best_early_stop_metric = metric_value

    return best_loss, best_center_f1, best_early_stop_metric


def metric_improved(
    current: float,
    best: float | None,
    metric_name: str,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    lower_is_better = metric_name == "val_loss" or metric_name.endswith("_mae")
    if lower_is_better:
        return current < best - min_delta
    return current > best + min_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CenterNet-style vertebra center and corner model.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/processed/coco_no_buu"))
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/centernet_runs"))
    parser.add_argument("--experiment-name", type=str, default="centernet_hrnet_w18_no_buu")
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--down-ratio", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hm-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--wh-weight", type=float, default=0.1)
    parser.add_argument("--peak-thresh", type=float, default=0.05)
    parser.add_argument("--eval-topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--backbone", type=str, default="hrnet_w18", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-preview-every", type=int, default=5)
    parser.add_argument("--preview-images", type=int, default=4)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print running batch metrics every N batches; use 0 to disable text updates.",
    )
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--keep-epoch-checkpoints", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="center_f1_12px",
        choices=["center_f1_8px", "center_f1_12px", "center_f1_16px", "val_loss", "count_mae"],
    )
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_dir = args.output_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = run_dir / "previews"
    backup_dir = args.backup_dir / args.experiment_name if args.backup_dir is not None else None

    train_limit = args.overfit_samples if args.overfit_samples > 0 else None
    val_limit = args.overfit_samples if args.overfit_samples > 0 else None
    train_dataset = CenterNetCocoDataset(
        dataset_root=args.dataset_root,
        split="train",
        config_path=args.config,
        image_size=args.input_size,
        down_ratio=args.down_ratio,
        max_objects=args.max_objects,
        augment=args.overfit_samples <= 0,
        limit=train_limit,
    )
    val_dataset = CenterNetCocoDataset(
        dataset_root=args.dataset_root,
        split="val",
        config_path=args.config,
        image_size=args.input_size,
        down_ratio=args.down_ratio,
        max_objects=args.max_objects,
        augment=False,
        limit=val_limit,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_centernet_model(backbone=args.backbone, pretrained=args.pretrained).to(device)
    criterion = CenterNetLoss(
        hm_weight=args.hm_weight,
        reg_weight=args.reg_weight,
        wh_weight=args.wh_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = run_dir / "train_log.csv"
    best_loss, best_center_f1, best_early_stop_metric = read_best_values_from_log(
        log_path,
        args.early_stop_metric,
    )
    epochs_without_improvement = 0
    start_epoch = 1
    resume_checkpoint_data: dict[str, Any] | None = None
    if args.resume_checkpoint is not None:
        start_epoch, resume_checkpoint_data = load_training_checkpoint(
            path=args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        if best_loss == float("inf"):
            best_loss = float(resume_checkpoint_data.get("val_loss", float("inf")))
        metrics = resume_checkpoint_data.get("metrics", {})
        if best_center_f1 < 0.0:
            best_center_f1 = float(metrics.get("center_f1_12px", -1.0))
        if best_early_stop_metric is None:
            metric_value = resume_checkpoint_data.get("val_loss") if args.early_stop_metric == "val_loss" else metrics.get(args.early_stop_metric)
            if metric_value is not None:
                best_early_stop_metric = float(metric_value)

    print(f"device: {device}")
    print(f"backbone: {args.backbone}")
    print(f"train images: {len(train_dataset)} | val images: {len(val_dataset)}")
    print(f"run dir: {run_dir}")
    if backup_dir is not None:
        print(f"backup dir: {backup_dir}")
    if args.resume_checkpoint is not None:
        print(f"resumed checkpoint: {args.resume_checkpoint}")
        print(f"starting epoch: {start_epoch}")

    if start_epoch > args.epochs:
        print(
            f"checkpoint is already at or beyond requested epochs ({args.epochs}); nothing to train.",
            flush=True,
        )
        return

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===", flush=True)
        train_stats = run_loss_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=args.max_train_batches,
            phase="train",
            epoch=epoch,
            total_epochs=args.epochs,
            progress_every=args.progress_every,
        )
        print(
            "train complete | loss {:.4f} | hm {:.4f} | reg {:.4f} | corner {:.4f}".format(
                train_stats["loss"],
                train_stats["hm_loss"],
                train_stats["reg_loss"],
                train_stats["wh_loss"],
            ),
            flush=True,
        )
        val_stats = run_loss_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
            phase="val-loss",
            epoch=epoch,
            total_epochs=args.epochs,
            progress_every=args.progress_every,
        )
        print(
            "validation loss complete | loss {:.4f} | hm {:.4f} | reg {:.4f} | corner {:.4f}".format(
                val_stats["loss"],
                val_stats["hm_loss"],
                val_stats["reg_loss"],
                val_stats["wh_loss"],
            ),
            flush=True,
        )
        center_metrics = evaluate_centers(
            model=model,
            loader=val_loader,
            device=device,
            down_ratio=args.down_ratio,
            peak_thresh=args.peak_thresh,
            topk=args.eval_topk,
            max_batches=args.max_val_batches,
            epoch=epoch,
            total_epochs=args.epochs,
            progress_every=args.progress_every,
        )
        scheduler.step()

        val_loss = val_stats["loss"]
        center_f1 = center_metrics.get("center_f1_12px", 0.0)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_hm_loss": train_stats["hm_loss"],
            "train_reg_loss": train_stats["reg_loss"],
            "train_wh_loss": train_stats["wh_loss"],
            "val_loss": val_stats["loss"],
            "val_hm_loss": val_stats["hm_loss"],
            "val_reg_loss": val_stats["reg_loss"],
            "val_wh_loss": val_stats["wh_loss"],
            "lr": optimizer.param_groups[0]["lr"],
            **center_metrics,
        }
        append_log(log_path, row)

        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, args, val_loss, center_metrics)
        if args.keep_epoch_checkpoints:
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, scheduler, scaler, epoch, args, val_loss, center_metrics)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(run_dir / "best_loss.pt", model, optimizer, scheduler, scaler, epoch, args, val_loss, center_metrics)
        if center_f1 > best_center_f1:
            best_center_f1 = center_f1
            save_checkpoint(run_dir / "best_center_f1.pt", model, optimizer, scheduler, scaler, epoch, args, val_loss, center_metrics)

        if args.save_preview_every > 0 and (epoch == 1 or epoch % args.save_preview_every == 0):
            save_prediction_previews(
                model=model,
                loader=val_loader,
                device=device,
                output_dir=preview_dir,
                epoch=epoch,
                down_ratio=args.down_ratio,
                peak_thresh=args.peak_thresh,
                topk=args.eval_topk,
                max_images=args.preview_images,
            )

        backup_run_files(run_dir, backup_dir)

        early_stop_metric = val_loss if args.early_stop_metric == "val_loss" else row[args.early_stop_metric]
        if metric_improved(
            current=float(early_stop_metric),
            best=best_early_stop_metric,
            metric_name=args.early_stop_metric,
            min_delta=args.early_stop_min_delta,
        ):
            best_early_stop_metric = float(early_stop_metric)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            "epoch {}/{} complete | train {:.4f} | val {:.4f} | f1@12 {:.4f} | recall@12 {:.4f} | count_mae {:.3f} | early-stop {}/{} (best {:.6f})".format(
                epoch,
                args.epochs,
                train_stats["loss"],
                val_loss,
                center_metrics.get("center_f1_12px", 0.0),
                center_metrics.get("center_recall_12px", 0.0),
                center_metrics.get("count_mae", 0.0),
                epochs_without_improvement,
                args.early_stop_patience,
                best_early_stop_metric if best_early_stop_metric is not None else float("nan"),
            ),
            flush=True,
        )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                "early stopping at epoch {}: {} did not improve for {} epoch(s); best {:.6f}".format(
                    epoch,
                    args.early_stop_metric,
                    args.early_stop_patience,
                    best_early_stop_metric if best_early_stop_metric is not None else float("nan"),
                ),
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
