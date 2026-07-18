from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CornerRefinerHRNetW18(nn.Module):
    """HRNet-W18 multi-scale feature fusion for four ordered corner heatmaps."""

    corner_order = ("TL", "TR", "BL", "BR")

    def __init__(
        self,
        *,
        pretrained: bool = True,
        projection_channels: int = 32,
        head_channels: int = 128,
        output_size: int = 64,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:  # pragma: no cover - dependency error is actionable
            raise ImportError("CornerRefinerHRNetW18 requires timm") from error

        self.model_name = "hrnet_w18"
        self.pretrained = bool(pretrained)
        self.projection_channels = int(projection_channels)
        self.head_channels = int(head_channels)
        self.output_size = int(output_size)
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        feature_channels = list(self.backbone.feature_info.channels())
        feature_reductions = list(self.backbone.feature_info.reduction())
        if len(feature_channels) != 4 or feature_reductions != [4, 8, 16, 32]:
            raise RuntimeError(
                f"Unexpected HRNet features: channels={feature_channels}, reductions={feature_reductions}"
            )

        self.projections = nn.ModuleList(
            [ConvBnRelu(channels, self.projection_channels, 1) for channels in feature_channels]
        )
        fused_channels = self.projection_channels * len(feature_channels)
        self.refiner_head = nn.Sequential(
            ConvBnRelu(fused_channels, self.head_channels, 3),
            ConvBnRelu(self.head_channels, self.head_channels, 3),
            nn.Conv2d(self.head_channels, 4, kernel_size=1),
        )

    def model_arguments(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "projection_channels": self.projection_channels,
            "head_channels": self.head_channels,
            "output_size": self.output_size,
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(inputs)
        target_size = (self.output_size, self.output_size)
        projected = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            projected.append(feature)
        logits = self.refiner_head(torch.cat(projected, dim=1))
        return {"heatmap_logits": logits}
