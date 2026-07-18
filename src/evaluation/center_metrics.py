from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def greedy_center_match(
    pred_centers: np.ndarray,
    gt_centers: np.ndarray,
    max_distance_px: float,
) -> int:
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return 0

    deltas = pred_centers[:, None, :] - gt_centers[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    pairs = np.argwhere(distances <= float(max_distance_px))
    if len(pairs) == 0:
        return 0

    pairs = sorted(pairs, key=lambda pair: float(distances[pair[0], pair[1]]))
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches = 0
    for pred_index, gt_index in pairs:
        pred_index = int(pred_index)
        gt_index = int(gt_index)
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches += 1
    return matches


@dataclass
class CenterMetricAccumulator:
    thresholds: tuple[int, ...] = (8, 12, 16)
    tp: dict[int, int] = field(default_factory=dict)
    fp: dict[int, int] = field(default_factory=dict)
    fn: dict[int, int] = field(default_factory=dict)
    count_error_sum: float = 0.0
    image_count: int = 0

    def __post_init__(self) -> None:
        self.tp = {threshold: 0 for threshold in self.thresholds}
        self.fp = {threshold: 0 for threshold in self.thresholds}
        self.fn = {threshold: 0 for threshold in self.thresholds}

    def update(
        self,
        decoded: list[dict[str, np.ndarray]],
        gt_centers: torch.Tensor,
        gt_count: torch.Tensor,
    ) -> None:
        gt_centers_np = gt_centers.detach().cpu().numpy()
        gt_count_np = gt_count.detach().cpu().numpy().astype(int)
        for batch_index, prediction in enumerate(decoded):
            pred_centers = prediction["centers"]
            count = int(gt_count_np[batch_index])
            gt = gt_centers_np[batch_index, :count]
            self.count_error_sum += abs(len(pred_centers) - count)
            self.image_count += 1

            for threshold in self.thresholds:
                matches = greedy_center_match(pred_centers, gt, threshold)
                self.tp[threshold] += matches
                self.fp[threshold] += len(pred_centers) - matches
                self.fn[threshold] += len(gt) - matches

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "count_mae": self.count_error_sum / max(self.image_count, 1),
        }
        for threshold in self.thresholds:
            tp = self.tp[threshold]
            fp = self.fp[threshold]
            fn = self.fn[threshold]
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            metrics[f"center_precision_{threshold}px"] = precision
            metrics[f"center_recall_{threshold}px"] = recall
            metrics[f"center_f1_{threshold}px"] = f1
        return metrics
