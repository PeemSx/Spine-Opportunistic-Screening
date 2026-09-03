from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from src.evaluate_centernet import (
    DEPLOYED_ARTIFACT_TYPE,
    DEPLOYED_OUTPUT_CONTRACT,
    DEPLOYED_PREPROCESSING_CONTRACT,
)
from src.export_centernet_deployment import (
    DEPLOYED_FORMAT_VERSION,
    DEPLOYED_POSTPROCESSING_CONTRACT,
    build_deployment_artifact,
    export_checkpoint,
)


def source_checkpoint(*, peak_thresh: float = 0.10) -> dict:
    return {
        "epoch": 12,
        "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer_state_dict": {"large_training_state": True},
        "args": {
            "backbone": "hrnet_w18",
            "input_size": 1024,
            "down_ratio": 4,
            "max_objects": 64,
            "peak_thresh": peak_thresh,
            "eval_topk": 50,
            "hm_weight": 1.0,
            "reg_weight": 1.0,
            "wh_weight": 0.5,
            "experiment_name": "unit-test-experiment",
        },
        "dataset_provenance": {
            "dataset_version": "unit-test-data",
            "fingerprint": "abc123",
        },
    }


class ExportCenterNetDeploymentTests(unittest.TestCase):
    def test_builds_compact_contract_and_preserves_evaluation_weights(self) -> None:
        artifact = build_deployment_artifact(
            source_checkpoint(),
            source_checkpoint_path=Path("best_center_f1.pt"),
            source_checkpoint_sha256="deadbeef",
        )

        self.assertEqual(artifact["artifact_type"], DEPLOYED_ARTIFACT_TYPE)
        self.assertEqual(artifact["format_version"], DEPLOYED_FORMAT_VERSION)
        self.assertEqual(artifact["preprocessing"], DEPLOYED_PREPROCESSING_CONTRACT)
        self.assertEqual(artifact["output_contract"], DEPLOYED_OUTPUT_CONTRACT)
        self.assertEqual(artifact["postprocessing"], DEPLOYED_POSTPROCESSING_CONTRACT)
        self.assertEqual(artifact["args"]["wh_weight"], 0.5)
        self.assertEqual(artifact["provenance"]["source_epoch"], 12)
        self.assertEqual(artifact["provenance"]["dataset_fingerprint"], "abc123")
        self.assertIn("model_state_dict", artifact)
        self.assertNotIn("optimizer_state_dict", artifact)

    def test_rejects_nonstandard_deployed_peak_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "require peak_thresh 0.10"):
            build_deployment_artifact(
                source_checkpoint(peak_thresh=0.05),
                source_checkpoint_path=Path("best_center_f1.pt"),
                source_checkpoint_sha256="deadbeef",
            )

    def test_export_round_trip_strips_training_state_and_loads_strictly(self) -> None:
        class WeightOnly(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(2))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "full.pt"
            output_path = root / "compact.pt"
            torch.save(source_checkpoint(), source_path)
            with patch(
                "src.export_centernet_deployment.build_centernet_model",
                return_value=WeightOnly(),
            ):
                export_checkpoint(source_path, output_path, force=False)

            compact = torch.load(output_path, map_location="cpu", weights_only=False)

        self.assertEqual(compact["model_state_dict"]["weight"].tolist(), [1.0, 2.0])
        self.assertNotIn("optimizer_state_dict", compact)
        self.assertNotIn("scheduler_state_dict", compact)
        self.assertNotIn("scaler_state_dict", compact)


if __name__ == "__main__":
    unittest.main()
