from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.models.centernet import (
    HRNET_PRETRAINED_BACKBONE_IDS,
    build_centernet_model,
    model_initialization_metadata,
)


class _FeatureInfo:
    @staticmethod
    def channels() -> list[int]:
        return [18, 36, 72, 144]


class _FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_info = _FeatureInfo()
        self.pretrained_cfg = {
            "hf_hub_id": "timm/hrnet_w18.ms_aug_in1k",
        }


class CenterNetPretrainedInitializationTests(unittest.TestCase):
    def test_pretrained_w18_uses_explicit_timm_weight_identifier(self) -> None:
        with patch(
            "src.models.centernet.timm.create_model",
            return_value=_FakeBackbone(),
        ) as create_model:
            model = build_centernet_model("hrnet_w18", pretrained=True)

        create_model.assert_called_once_with(
            "hrnet_w18.ms_aug_in1k",
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        self.assertEqual(
            model_initialization_metadata(model),
            {
                "backbone": "hrnet_w18",
                "pretrained": True,
                "pretrained_backbone_id": "hrnet_w18.ms_aug_in1k",
                "pretrained_source": "timm/hrnet_w18.ms_aug_in1k",
            },
        )

    def test_random_w18_keeps_architecture_alias_and_no_weight_source(self) -> None:
        with patch(
            "src.models.centernet.timm.create_model",
            return_value=_FakeBackbone(),
        ) as create_model:
            model = build_centernet_model("hrnet_w18", pretrained=False)

        create_model.assert_called_once_with(
            "hrnet_w18",
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        metadata = model_initialization_metadata(model)
        self.assertFalse(metadata["pretrained"])
        self.assertIsNone(metadata["pretrained_backbone_id"])
        self.assertIsNone(metadata["pretrained_source"])

    def test_every_supported_hrnet_has_an_explicit_pretrained_identifier(self) -> None:
        self.assertEqual(
            set(HRNET_PRETRAINED_BACKBONE_IDS),
            {
                "hrnet_w18",
                "hrnet_w30",
                "hrnet_w32",
                "hrnet_w40",
                "hrnet_w44",
                "hrnet_w48",
                "hrnet_w64",
            },
        )


if __name__ == "__main__":
    unittest.main()
