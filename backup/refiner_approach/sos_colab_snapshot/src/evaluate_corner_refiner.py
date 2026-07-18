from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.corner_refiner_dataset import CornerRefinerCocoDataset
from src.models.corner_refiner import CornerRefinerHRNetW18
from src.train_corner_refiner import atomic_json, atomic_torch_save, validate_epoch
from src.training.corner_refiner_loss import CornerRefinerLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a corner-refiner checkpoint in original coordinates.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/processed/coco"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--export-inference", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: dict, device: torch.device) -> CornerRefinerHRNetW18:
    model_args = checkpoint.get("model_args", checkpoint.get("metadata", {}).get("model_arguments", {}))
    model = CornerRefinerHRNetW18(
        pretrained=False,
        projection_channels=int(model_args.get("projection_channels", 32)),
        head_channels=int(model_args.get("head_channels", 128)),
        output_size=int(model_args.get("output_size", 64)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device)


def main() -> None:
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    crop_size = int(train_args.get("input_size", 256))
    heatmap_size = int(train_args.get("heatmap_size", 64))
    crop_scale = float(train_args.get("crop_scale", 1.5))
    heatmap_sigma = float(train_args.get("heatmap_sigma", 1.0))
    dataset = CornerRefinerCocoDataset(
        args.dataset_root,
        args.split,
        crop_size=crop_size,
        heatmap_size=heatmap_size,
        crop_scale=crop_scale,
        heatmap_sigma=heatmap_sigma,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_model(checkpoint, device)
    criterion = CornerRefinerLoss(crop_size=crop_size)
    losses, metrics = validate_epoch(
        model,
        loader,
        criterion,
        device,
        use_amp=device.type == "cuda" and not args.no_amp,
        crop_size=crop_size,
        max_batches=args.max_batches,
    )
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "loss": losses,
        "metrics": metrics,
    }
    atomic_json(output_dir / f"{args.split}_metrics.json", payload)
    print(f"split={args.split} samples={metrics['sample_count']}")
    print(
        f"loss={losses['loss']:.6f} mean_nme={metrics['mean_nme']:.6f} "
        f"median_nme={metrics['median_nme']:.6f} p95_nme={metrics['p95_nme']:.6f}"
    )
    for source, source_metrics in metrics["per_source"].items():
        print(f"{source}: n={source_metrics['sample_count']} mean_nme={source_metrics['mean_nme']:.6f}")

    if args.export_inference:
        inference_payload = {
            "format_version": 1,
            "model_state_dict": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "model_args": {
                **checkpoint.get("model_args", {}),
                "pretrained": False,
            },
            "metadata": checkpoint.get("metadata", {}),
            "validation": payload,
        }
        atomic_torch_save(output_dir / "inference_refiner.pt", inference_payload)
        print(f"exported={output_dir / 'inference_refiner.pt'}")


if __name__ == "__main__":
    main()
