from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from src.workflows.train_pretrained_w18_existing_loss import (
    EXPECTED_DATASET_FINGERPRINT,
    EXPERIMENT_NAME,
    RESUME_ARGUMENT_CONTRACT,
    build_training_command,
    validate_resume_checkpoint,
)
import torch


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


class PretrainedW18ExperimentTests(unittest.TestCase):
    def test_command_changes_initialization_but_preserves_baseline_protocol(self) -> None:
        args = argparse.Namespace(
            dataset_root=Path("dataset"),
            output_dir=Path("runs"),
            backup_dir=Path("drive-runs"),
            config=Path("configs/config.yaml"),
            resume_checkpoint=None,
            num_workers=2,
        )
        command = build_training_command(args)

        self.assertIn("--pretrained", command)
        self.assertEqual(_option_value(command, "--experiment-name"), EXPERIMENT_NAME)
        self.assertEqual(_option_value(command, "--backbone"), "hrnet_w18")
        self.assertEqual(_option_value(command, "--epochs"), "80")
        self.assertEqual(_option_value(command, "--batch-size"), "8")
        self.assertEqual(_option_value(command, "--seed"), "20260627")
        self.assertEqual(_option_value(command, "--hm-weight"), "1.0")
        self.assertEqual(_option_value(command, "--reg-weight"), "1.0")
        self.assertEqual(_option_value(command, "--wh-weight"), "0.5")
        self.assertEqual(
            _option_value(command, "--early-stop-metric"),
            "center_f1_12px",
        )
        self.assertNotIn("scale-normalized", " ".join(command))

    def test_resume_checkpoint_is_forwarded_without_changing_experiment(self) -> None:
        args = argparse.Namespace(
            dataset_root=Path("dataset"),
            output_dir=Path("runs"),
            backup_dir=None,
            config=Path("configs/config.yaml"),
            resume_checkpoint=Path("runs/last.pt"),
            num_workers=0,
        )
        command = build_training_command(args)

        self.assertEqual(
            _option_value(command, "--resume-checkpoint"),
            "runs/last.pt",
        )
        self.assertEqual(_option_value(command, "--experiment-name"), EXPERIMENT_NAME)

    def test_rejects_resume_from_a_scratch_initialization(self) -> None:
        checkpoint = {
            "args": {**RESUME_ARGUMENT_CONTRACT, "pretrained": False},
            "dataset_provenance": {
                "fingerprint": EXPECTED_DATASET_FINGERPRINT,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "pretrained=False"):
                validate_resume_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
