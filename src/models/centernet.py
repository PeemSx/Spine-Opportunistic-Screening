from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34

try:
    import timm
except ImportError:  # pragma: no cover - exercised in environments without timm installed.
    timm = None

HRNET_BACKBONES = (
    "hrnet_w18",
    "hrnet_w30",
    "hrnet_w32",
    "hrnet_w40",
    "hrnet_w44",
    "hrnet_w48",
    "hrnet_w64",
)
SUPPORTED_BACKBONES = (*HRNET_BACKBONES, "resnet34")

# Use explicit timm weight identifiers so a pretrained experiment does not
# silently change when timm changes the default pretrained tag for an
# architecture alias such as ``hrnet_w18``.
HRNET_PRETRAINED_BACKBONE_IDS = {
    "hrnet_w18": "hrnet_w18.ms_aug_in1k",
    "hrnet_w30": "hrnet_w30.ms_in1k",
    "hrnet_w32": "hrnet_w32.ms_in1k",
    "hrnet_w40": "hrnet_w40.ms_in1k",
    "hrnet_w44": "hrnet_w44.ms_in1k",
    "hrnet_w48": "hrnet_w48.ms_in1k",
    "hrnet_w64": "hrnet_w64.ms_in1k",
}


def _conv_bn_relu(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _make_head(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    heatmap_head: bool = False,
) -> nn.Sequential:
    head = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
    )
    if heatmap_head:
        nn.init.constant_(head[-1].bias, -2.19)
    else:
        for module in head.modules():
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return head


class CenterNetResNetFPN(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        fpn_channels: int = 128,
        head_channels: int = 128,
    ) -> None:
        super().__init__()
        self.pretrained = bool(pretrained)
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        backbone = resnet34(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.lat2 = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(128, fpn_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(256, fpn_channels, kernel_size=1)
        self.lat5 = nn.Conv2d(512, fpn_channels, kernel_size=1)
        self.smooth2 = _conv_bn_relu(fpn_channels, fpn_channels)
        self.smooth3 = _conv_bn_relu(fpn_channels, fpn_channels)
        self.smooth4 = _conv_bn_relu(fpn_channels, fpn_channels)

        self.hm = _make_head(fpn_channels, head_channels, 1, heatmap_head=True)
        self.reg = _make_head(fpn_channels, head_channels, 2)
        self.wh = _make_head(fpn_channels, head_channels, 8)

    @staticmethod
    def _upsample_add(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        return F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False) + low

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lat5(c5)
        p4 = self.smooth4(self._upsample_add(p5, self.lat4(c4)))
        p3 = self.smooth3(self._upsample_add(p4, self.lat3(c3)))
        p2 = self.smooth2(self._upsample_add(p3, self.lat2(c2)))

        return {
            "hm": torch.sigmoid(self.hm(p2)),
            "reg": self.reg(p2),
            "wh": self.wh(p2),
        }


class CenterNetHRNet(nn.Module):
    """HRNet backbone with fused multi-resolution CenterNet heads."""

    def __init__(
        self,
        backbone_name: str = "hrnet_w18",
        pretrained: bool = False,
        branch_channels: int = 64,
        fusion_channels: int = 256,
        head_channels: int = 128,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError(
                f"backbone='{backbone_name}' requires the 'timm' package. "
                "Install project requirements or run: pip install timm"
            )

        self.backbone_name = backbone_name
        self.pretrained = bool(pretrained)
        self.backbone_model_id = (
            HRNET_PRETRAINED_BACKBONE_IDS[backbone_name]
            if pretrained
            else backbone_name
        )
        self.backbone = timm.create_model(
            self.backbone_model_id,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        encoder_channels = list(self.backbone.feature_info.channels())
        if len(encoder_channels) != 4:
            raise ValueError(f"Expected 4 HRNet feature maps, got {len(encoder_channels)}")

        self.projections = nn.ModuleList(
            [
                _conv_bn_relu(in_channels, branch_channels, kernel_size=1)
                for in_channels in encoder_channels
            ]
        )
        self.fuse = nn.Sequential(
            _conv_bn_relu(branch_channels * 4, fusion_channels, kernel_size=3),
            _conv_bn_relu(fusion_channels, fusion_channels, kernel_size=3),
        )
        self.hm = _make_head(fusion_channels, head_channels, 1, heatmap_head=True)
        self.reg = _make_head(fusion_channels, head_channels, 2)
        self.wh = _make_head(fusion_channels, head_channels, 8)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, target_size, mode="bilinear", align_corners=False)
            projected.append(feature)

        fused = self.fuse(torch.cat(projected, dim=1))
        return {
            "hm": torch.sigmoid(self.hm(fused)),
            "reg": self.reg(fused),
            "wh": self.wh(fused),
        }


def build_centernet_model(backbone: str = "hrnet_w18", pretrained: bool = False) -> nn.Module:
    if backbone in HRNET_BACKBONES:
        return CenterNetHRNet(backbone_name=backbone, pretrained=pretrained)
    if backbone == "resnet34":
        return CenterNetResNetFPN(pretrained=pretrained)
    raise ValueError(f"Unsupported CenterNet backbone: {backbone}")


def model_initialization_metadata(model: nn.Module) -> dict[str, object]:
    """Return JSON-compatible provenance for a CenterNet initialization."""

    if isinstance(model, CenterNetHRNet):
        pretrained_cfg = getattr(model.backbone, "pretrained_cfg", {}) or {}
        pretrained_source = None
        if model.pretrained:
            pretrained_source = (
                pretrained_cfg.get("hf_hub_id")
                or pretrained_cfg.get("url")
                or None
            )
        return {
            "backbone": model.backbone_name,
            "pretrained": model.pretrained,
            "pretrained_backbone_id": (
                model.backbone_model_id if model.pretrained else None
            ),
            "pretrained_source": pretrained_source,
        }
    if isinstance(model, CenterNetResNetFPN):
        return {
            "backbone": "resnet34",
            "pretrained": model.pretrained,
            "pretrained_backbone_id": (
                "ResNet34_Weights.DEFAULT" if model.pretrained else None
            ),
            "pretrained_source": "torchvision" if model.pretrained else None,
        }
    return {
        "backbone": type(model).__name__,
        "pretrained": False,
        "pretrained_backbone_id": None,
        "pretrained_source": None,
    }
