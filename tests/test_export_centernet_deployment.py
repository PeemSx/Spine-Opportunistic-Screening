from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
