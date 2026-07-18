from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.centernet_dataset import CenterNetCocoDataset
from src.models.centernet import SUPPORTED_BACKBONES, build_centernet_model
from src.training.centernet_loss import CenterNetLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check CenterNet COCO dataset tensors.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/processed/coco_no_buu"))
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--down-ratio", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--backbone", type=str, default="hrnet_w18", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--model-forward", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CenterNetCocoDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        config_path=args.config,
        image_size=args.image_size,
        down_ratio=args.down_ratio,
        max_objects=args.max_objects,
        augment=False,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batch = next(iter(loader))

    print("dataset length:", len(dataset))
    print("input:", tuple(batch["input"].shape), batch["input"].dtype)
    print("hm:", tuple(batch["hm"].shape), batch["hm"].dtype, "max:", float(batch["hm"].max()))
    print("reg:", tuple(batch["reg"].shape))
    print("wh:", tuple(batch["wh"].shape))
    print("ind:", tuple(batch["ind"].shape), batch["ind"].dtype)
    print("reg_mask:", tuple(batch["reg_mask"].shape), "sum:", batch["reg_mask"].sum(dim=1).tolist())
    print("gt_count:", batch["gt_count"].tolist())
    print("collisions:", batch["collisions"].tolist())
    print("truncated:", batch["truncated"].tolist())
    print("files:", list(batch["file_name"]))

    mask_sum = batch["reg_mask"].sum(dim=1).long()
    if not torch.equal(mask_sum, batch["gt_count"]):
        raise RuntimeError(f"reg_mask sum does not match gt_count: {mask_sum.tolist()} vs {batch['gt_count'].tolist()}")
    if float(batch["hm"].max()) < 0.99:
        raise RuntimeError("heatmap max is unexpectedly low")

    if args.model_forward:
        model = build_centernet_model(backbone=args.backbone, pretrained=False)
        criterion = CenterNetLoss()
        outputs = model(batch["input"])
        loss = criterion.loss_dict(outputs, batch)
        print("model outputs:", {key: tuple(value.shape) for key, value in outputs.items()})
        print("loss:", {key: float(value.detach()) for key, value in loss.items()})


if __name__ == "__main__":
    main()
