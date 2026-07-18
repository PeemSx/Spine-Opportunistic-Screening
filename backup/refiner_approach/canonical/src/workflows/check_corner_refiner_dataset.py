from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.corner_refiner_dataset import CornerRefinerCocoDataset
from src.data.corner_refiner_targets import build_gaussian_heatmaps, crop_points_to_heatmap
from src.data.roi_geometry import crop_from_roi, make_square_roi, roi_contains_points, transform_points
from src.evaluation.corner_refiner_decode import decode_corner_heatmaps
from src.models.corner_refiner import CornerRefinerHRNetW18
from src.training.corner_refiner_loss import CornerRefinerLoss


EXPECTED_FULL_COUNTS = {
    "train": (1403, 18230),
    "val": (175, 2495),
    "test": (175, 2579),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ROI geometry, targets, data, and optional model forward pass.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/processed/coco"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--verify-full-counts", action="store_true")
    parser.add_argument("--verify-all-files", action="store_true")
    parser.add_argument("--model-forward", action="store_true")
    return parser.parse_args()


def check_geometry() -> None:
    points = np.asarray([[2.0, 3.0], [10.0, 3.5], [2.5, 9.0], [10.5, 9.5]], dtype=np.float32)
    image = np.full((12, 12), 127, dtype=np.uint8)
    roi = make_square_roi(points, crop_scale=1.5)
    result = crop_from_roi(image, points, roi, output_size=256)
    if result["round_trip_error"] >= 0.1:
        raise RuntimeError(f"Round-trip error is too high: {result['round_trip_error']}")
    matrix = result["original_to_crop"]
    if not np.isclose(matrix[0, 0], matrix[1, 1], atol=1e-6):
        raise RuntimeError("Affine transform is not isotropic")
    if not result["inside_crop"]:
        raise RuntimeError("Deterministic crop lost a corner")

    padded_roi = np.asarray([-4.0, -4.0, 8.0, 8.0], dtype=np.float32)
    padded = crop_from_roi(image, points, padded_roi, output_size=24)
    if not padded["uses_padding"] or not np.all(padded["crop"][:6, :6] == 0):
        raise RuntimeError("Black boundary padding check failed")

    remapped = transform_points(result["crop_points"], result["crop_to_original"])
    if float(np.linalg.norm(remapped - points, axis=1).max()) >= 0.1:
        raise RuntimeError("Original/crop transform round trip failed")


def check_heatmaps() -> None:
    points = np.asarray([[40.0, 45.0], [205.0, 48.0], [42.0, 210.0], [207.0, 208.0]], dtype=np.float32)
    targets = build_gaussian_heatmaps(points, crop_size=256, heatmap_size=64, sigma=1.0)
    logits = torch.from_numpy(np.log(np.maximum(targets, 1e-30)))[None]
    decoded = decode_corner_heatmaps(logits)["points_heatmap"][0].numpy()
    expected = crop_points_to_heatmap(points)
    error = float(np.linalg.norm(decoded - expected, axis=-1).max())
    if error >= 0.25:
        raise RuntimeError(f"Heatmap encode/decode error is {error:.4f} cells")


def check_order(records) -> None:
    failures = []
    for record in records:
        points = record.points
        # Do not geometrically reorder authoritative annotations: a severely tilted or
        # anomalous source annotation can violate simple x/y tests. This check protects
        # the stored TL/TR/BL/BR array positions themselves.
        valid = points.shape == (4, 2) and np.isfinite(points).all()
        if not valid:
            failures.append(record.annotation_id)
    if failures:
        raise RuntimeError(f"TL/TR/BL/BR ordering failed for annotation IDs: {failures[:10]}")


def main() -> None:
    args = parse_args()
    check_geometry()
    check_heatmaps()
    dataset = CornerRefinerCocoDataset(
        args.dataset_root,
        args.split,
        augment=False,
        limit=args.limit if args.limit > 0 else None,
    )
    check_order(dataset.records)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batch = next(iter(loader))
    if tuple(batch["input"].shape[1:]) != (3, 256, 256):
        raise RuntimeError(f"Unexpected input shape: {tuple(batch['input'].shape)}")
    if tuple(batch["target_heatmaps"].shape[1:]) != (4, 64, 64):
        raise RuntimeError(f"Unexpected heatmap shape: {tuple(batch['target_heatmaps'].shape)}")
    if float(batch["round_trip_error"].max()) >= 0.1 or not bool(batch["target_heatmaps"].isfinite().all()):
        raise RuntimeError("Batch transform/target checks failed")
    expected_crop = []
    for index in range(len(batch["gt_points_original"])):
        expected_crop.append(
            transform_points(
                batch["gt_points_original"][index].numpy(),
                batch["original_to_crop"][index].numpy(),
            )
        )
    if not np.allclose(np.stack(expected_crop), batch["target_points_crop"].numpy(), atol=1e-4):
        raise RuntimeError("TL/TR/BL/BR positions changed during crop transformation")

    if args.verify_full_counts:
        full = CornerRefinerCocoDataset(args.dataset_root, args.split, augment=False)
        expected_images, expected_annotations = EXPECTED_FULL_COUNTS[args.split]
        actual = (len(full.image_record_indices), len(full))
        if actual != (expected_images, expected_annotations):
            raise RuntimeError(f"Count mismatch: expected {(expected_images, expected_annotations)}, got {actual}")
        sources = sorted(set(record.source_dataset for record in full.records))
        if sources != ["buu_ap", "mendeley_pa", "miccai_2019"]:
            raise RuntimeError(f"Unexpected sources: {sources}")
        check_order(full.records)
        containment_failures = [
            record.annotation_id
            for record in full.records
            if not roi_contains_points(make_square_roi(record.points, crop_scale=1.5), record.points)
        ]
        if containment_failures:
            raise RuntimeError(
                f"Deterministic ROI containment failed for annotation IDs: {containment_failures[:10]}"
            )
        if args.verify_all_files:
            missing = [
                name for name in {record.file_name for record in full.records}
                if not (args.dataset_root / args.split / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} images; first={missing[0]}")

    print("geometry=PASS heatmaps=PASS order=PASS")
    print(
        f"dataset={len(dataset)} input={tuple(batch['input'].shape)} "
        f"targets={tuple(batch['target_heatmaps'].shape)} files={list(batch['file_name'])}"
    )
    if args.model_forward:
        model = CornerRefinerHRNetW18(pretrained=False)
        outputs = model(batch["input"])
        logits = outputs["heatmap_logits"]
        if tuple(logits.shape[1:]) != (4, 64, 64):
            raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")
        losses = CornerRefinerLoss()(logits, batch["target_heatmaps"], batch["target_points_crop"])
        losses["loss"].backward()
        gradient_sum = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if not torch.isfinite(losses["loss"]) or gradient_sum <= 0.0:
            raise RuntimeError("Loss/gradient check failed")
        print(f"model_output={tuple(logits.shape)} loss={float(losses['loss'].detach()):.6f} gradients=PASS")


if __name__ == "__main__":
    main()
