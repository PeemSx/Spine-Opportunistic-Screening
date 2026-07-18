from __future__ import annotations

import torch
from torch import nn

from src.data.corner_refiner_targets import crop_points_to_normalized
from src.evaluation.corner_refiner_decode import decode_corner_heatmaps


class CornerRefinerLoss(nn.Module):
    def __init__(self, *, crop_size: int = 256, js_weight: float = 1.0) -> None:
        super().__init__()
        self.crop_size = int(crop_size)
        self.js_weight = float(js_weight)

    def forward(
        self,
        logits: torch.Tensor,
        target_heatmaps: torch.Tensor,
        target_points_crop: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        decoded = decode_corner_heatmaps(logits, crop_size=self.crop_size)
        prediction = decoded["points_normalized"]
        target_points = crop_points_to_normalized(target_points_crop, crop_size=self.crop_size)
        coordinate_loss = torch.linalg.vector_norm(prediction - target_points, dim=-1).mean()

        prediction_prob = decoded["probability_maps"].clamp_min(1e-12)
        target_prob = target_heatmaps / target_heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        target_prob = target_prob.clamp_min(1e-12)
        midpoint = 0.5 * (prediction_prob + target_prob)
        js_map = 0.5 * prediction_prob * (prediction_prob.log() - midpoint.log())
        js_map += 0.5 * target_prob * (target_prob.log() - midpoint.log())
        js_loss = js_map.sum(dim=(-2, -1)).mean()
        total = coordinate_loss + self.js_weight * js_loss
        return {
            "loss": total,
            "coordinate_loss": coordinate_loss,
            "js_loss": js_loss,
        }
