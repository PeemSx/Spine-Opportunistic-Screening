from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def nms_heatmap(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def _gather_feature_map(feature: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    feature = feature.permute(0, 2, 3, 1).contiguous()
    feature = feature.view(feature.size(0), -1, feature.size(3))
    dim = feature.size(2)
    indices = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), dim)
    return feature.gather(1, indices)


@torch.no_grad()
def decode_centernet_outputs(
    outputs: dict[str, torch.Tensor],
    down_ratio: int = 4,
    peak_thresh: float = 0.05,
    topk: int = 100,
) -> list[dict[str, np.ndarray]]:
    heat = nms_heatmap(outputs["hm"])
    batch, _, height, width = heat.shape
    topk = min(int(topk), height * width)

    scores, indices = torch.topk(heat.view(batch, -1), topk)
    ys = torch.div(indices, width, rounding_mode="floor").float()
    xs = (indices % width).float()

    reg = _gather_feature_map(outputs["reg"], indices)
    wh = _gather_feature_map(outputs["wh"], indices).view(batch, topk, 4, 2)
    centers_out = torch.stack([xs + reg[:, :, 0], ys + reg[:, :, 1]], dim=2)
    corners_out = centers_out[:, :, None, :] - wh

    centers_px = centers_out * float(down_ratio)
    corners_px = corners_out * float(down_ratio)

    decoded: list[dict[str, np.ndarray]] = []
    for batch_index in range(batch):
        keep = scores[batch_index] >= float(peak_thresh)
        decoded.append(
            {
                "scores": scores[batch_index, keep].detach().cpu().numpy().astype(np.float32),
                "centers": centers_px[batch_index, keep].detach().cpu().numpy().astype(np.float32),
                "corners": corners_px[batch_index, keep].detach().cpu().numpy().astype(np.float32),
            }
        )
    return decoded
