from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.compact_centernet_checkpoints import (
    compact_checkpoint_directory,
    model_state_sha256,
)
from src.evaluate_centernet import file_sha256, load_checkpoint
from src.export_centernet_deployment import build_deployment_artifact


def checkpoint(weight: list[float], *, input_size: int = 1024, epoch: int = 1) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": {
            "weight": torch.tensor(weight, dtype=torch.float32),
            "counter": torch.tensor(epoch, dtype=torch.int64),
        },
        "optimizer_state_dict": {"training_only": True},
        "scheduler_state_dict": {"training_only": True},
        "scaler_state_dict": {"training_only": True},
        "metrics": {"training_only": True},
        "val_loss": 1.0,
        "args": {
            "backbone": "hrnet_w18",
            "input_size": input_size,
            "down_ratio": 4,
            "max_objects": 64,
            "peak_thresh": 0.10,
            "eval_topk": 50,
            "hm_weight": 1.0,
            "reg_weight": 1.0,
            "wh_weight": 0.1,
        },
        "dataset_provenance": {
            "dataset_version": "fixture",
            "fingerprint": "fixture-fingerprint",
        },
    }


def fixture_export(source_path: Path, output_path: Path, *, force: bool) -> Path:
    if output_path.exists() and not force:
        raise FileExistsError(output_path)
    source = load_checkpoint(source_path, torch.device("cpu"))
    artifact = build_deployment_artifact(
        source,
        source_checkpoint_path=source_path,
        source_checkpoint_sha256=file_sha256(source_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return output_path


class CompactCheckpointDirectoryTests(unittest.TestCase):
    def _write_sources(self, root: Path) -> None:
        torch.save(checkpoint([1.0, 2.0], epoch=12), root / "a.pt")
        torch.save(checkpoint([1.0, 2.0], epoch=12), root / "b.pt")
        torch.save(checkpoint([3.0, 4.0], epoch=15), root / "c.pt")

    def test_compacts_unique_states_and_records_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            output_root = root / "compact"
            source_root.mkdir()
            self._write_sources(source_root)

            with patch(
                "src.compact_centernet_checkpoints.export_checkpoint",
                side_effect=fixture_export,
            ):
                manifest = compact_checkpoint_directory(
                    source_root,
                    output_root,
                    representative_names=["a.pt", "c.pt"],
                )

            aliases = json.loads((output_root / "checkpoint_aliases.json").read_text())
            compact_a = load_checkpoint(output_root / "a.pt", torch.device("cpu"))
            source_a = load_checkpoint(source_root / "a.pt", torch.device("cpu"))

            self.assertEqual(manifest["source_checkpoint_count"], 3)
            self.assertEqual(manifest["unique_state_count"], 2)
            self.assertEqual(
                sorted(path.name for path in output_root.iterdir()),
                ["a.pt", "c.pt", "checkpoint_aliases.json", "compaction_manifest.json"],
            )
            self.assertEqual(aliases["checkpoint_to_representative"]["b.pt"], "a.pt")
            self.assertNotIn("optimizer_state_dict", compact_a)
            self.assertNotIn("metrics", compact_a)
            self.assertEqual(
                model_state_sha256(compact_a["model_state_dict"]),
                model_state_sha256(source_a["model_state_dict"]),
            )
            self.assertFalse((output_root / "a.pt").is_symlink())

    def test_rejects_nonempty_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            output_root = root / "compact"
            source_root.mkdir()
            output_root.mkdir()
            (output_root / "keep.txt").write_text("user data")
            self._write_sources(source_root)

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                compact_checkpoint_directory(
                    source_root,
                    output_root,
                    representative_names=["a.pt", "c.pt"],
                )

            self.assertEqual((output_root / "keep.txt").read_text(), "user data")

    def test_rejects_two_representatives_for_one_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            self._write_sources(source_root)

            with self.assertRaisesRegex(ValueError, "share one model state"):
                compact_checkpoint_directory(
                    source_root,
                    root / "compact",
                    representative_names=["a.pt", "b.pt", "c.pt"],
                )

    def test_rejects_identical_state_with_conflicting_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            first = checkpoint([1.0, 2.0], input_size=1024, epoch=12)
            second = checkpoint([1.0, 2.0], input_size=512, epoch=12)
            torch.save(first, source_root / "a.pt")
            torch.save(second, source_root / "b.pt")

            with self.assertRaisesRegex(ValueError, "conflicting inference contracts"):
                compact_checkpoint_directory(
                    source_root,
                    root / "compact",
                    representative_names=["a.pt"],
                )


if __name__ == "__main__":
    unittest.main()
