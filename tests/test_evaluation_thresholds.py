from __future__ import annotations

import argparse
import inspect
import pathlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.evaluate_centernet import (
    DEPLOYED_OUTPUT_CONTRACT,
    DEPLOYED_PREPROCESSING_CONTRACT,
    build_model_and_settings,
    load_checkpoint,
)
from src.evaluation.centernet_decode import decode_centernet_outputs
from src.evaluation.config import DEFAULT_PEAK_THRESHOLD


def arguments(
    *,
    profile: str,
    checkpoint: Path,
    peak_thresh: float | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        evaluation_profile=profile,
        checkpoint=checkpoint,
        input_size=None,
        down_ratio=None,
        max_objects=None,
        peak_thresh=peak_thresh,
        topk=None,
        chain_duplicate_iou=None,
        chain_duplicate_center_scale=None,
        chain_score_thresh=None,
        chain_score_weight=None,
        chain_min_len=None,
    )


def checkpoint_payload(
    peak_thresh: float,
    *,
    deployed: bool,
) -> dict:
    payload = {
        "model_state_dict": {},
        "args": {
            "backbone": "hrnet_w18",
            "input_size": 1024,
            "down_ratio": 4,
            "peak_thresh": peak_thresh,
            "eval_topk": 50,
        },
    }
    if deployed:
        payload.update(
            {
                "artifact_type": "spine_centernet_inference",
                "format_version": 1,
                "preprocessing": DEPLOYED_PREPROCESSING_CONTRACT,
                "output_contract": DEPLOYED_OUTPUT_CONTRACT,
                "postprocessing": {
                    "spine_chain_enabled": True,
                    "duplicate_iou_threshold": 0.18,
                    "duplicate_center_scale": 0.35,
                    "score_threshold": 0.18,
                    "score_weight": 3.0,
                    "min_chain_len": 3,
                },
            }
        )
    return payload


class EvaluationThresholdTests(unittest.TestCase):
    def test_loads_checkpoint_serialized_with_newer_pathlib_module(self) -> None:
        module = types.ModuleType("pathlib._local")
        newer_posix_path = type(
            "PosixPath",
            (pathlib.PosixPath,),
            {"__module__": "pathlib._local"},
        )
        module.PosixPath = newer_posix_path
        sys.modules["pathlib._local"] = module
        setattr(pathlib, "_local", module)
        try:
            with tempfile.TemporaryDirectory() as directory:
                checkpoint_path = Path(directory) / "newer-python.pt"
                torch.save({"saved_path": newer_posix_path("weights.pt")}, checkpoint_path)
                del sys.modules["pathlib._local"]
                delattr(pathlib, "_local")

                checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))

            self.assertEqual(checkpoint["saved_path"], Path("weights.pt"))
            self.assertIs(type(checkpoint["saved_path"]), pathlib.PosixPath)
        finally:
            sys.modules.pop("pathlib._local", None)
            if hasattr(pathlib, "_local"):
                delattr(pathlib, "_local")

    def test_shared_decoder_default_is_point_ten(self) -> None:
        self.assertEqual(DEFAULT_PEAK_THRESHOLD, 0.10)
        default = inspect.signature(decode_centernet_outputs).parameters[
            "peak_thresh"
        ].default
        self.assertEqual(default, DEFAULT_PEAK_THRESHOLD)

    def test_research_overrides_legacy_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.pt"
            checkpoint_path.touch()
            with (
                patch(
                    "src.evaluate_centernet.load_checkpoint",
                    return_value=checkpoint_payload(0.05, deployed=False),
                ),
                patch(
                    "src.evaluate_centernet.build_centernet_model",
                    return_value=torch.nn.Identity(),
                ),
            ):
                _, settings, metadata, _ = build_model_and_settings(
                    arguments(
                        profile="research",
                        checkpoint=checkpoint_path,
                    ),
                    torch.device("cpu"),
                )
            self.assertEqual(settings["peak_thresh"], 0.10)
            self.assertEqual(metadata["embedded_peak_thresh"], 0.05)
            self.assertEqual(metadata["effective_peak_thresh"], 0.10)
            self.assertFalse(metadata["nonstandard_peak_threshold"])

    def test_explicit_research_override_is_labelled_nonstandard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.pt"
            checkpoint_path.touch()
            with (
                patch(
                    "src.evaluate_centernet.load_checkpoint",
                    return_value=checkpoint_payload(0.05, deployed=False),
                ),
                patch(
                    "src.evaluate_centernet.build_centernet_model",
                    return_value=torch.nn.Identity(),
                ),
            ):
                _, settings, metadata, _ = build_model_and_settings(
                    arguments(
                        profile="research",
                        checkpoint=checkpoint_path,
                        peak_thresh=0.12,
                    ),
                    torch.device("cpu"),
                )
            self.assertEqual(settings["peak_thresh"], 0.12)
            self.assertTrue(metadata["nonstandard_peak_threshold"])

    def test_research_reuses_checkpoint_validation_chain_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "phase1.pt"
            checkpoint_path.touch()
            payload = checkpoint_payload(0.10, deployed=False)
            payload["args"].update(
                {
                    "val_chain_duplicate_iou": 0.22,
                    "val_chain_duplicate_center_scale": 0.40,
                    "val_chain_score_thresh": 0.25,
                    "val_chain_score_weight": 2.5,
                    "val_chain_min_len": 4,
                }
            )
            with (
                patch(
                    "src.evaluate_centernet.load_checkpoint",
                    return_value=payload,
                ),
                patch(
                    "src.evaluate_centernet.build_centernet_model",
                    return_value=torch.nn.Identity(),
                ),
            ):
                _, settings, _, _ = build_model_and_settings(
                    arguments(
                        profile="research",
                        checkpoint=checkpoint_path,
                    ),
                    torch.device("cpu"),
                )
            self.assertEqual(
                settings["chain"],
                {
                    "duplicate_iou": 0.22,
                    "duplicate_center_scale": 0.40,
                    "score_thresh": 0.25,
                    "score_weight": 2.5,
                    "min_len": 4,
                },
            )

    def test_deployed_artifact_rejects_nonstandard_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "deployed.pt"
            checkpoint_path.touch()
            with patch(
                "src.evaluate_centernet.load_checkpoint",
                return_value=checkpoint_payload(0.05, deployed=True),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "must be 0.10",
                ):
                    build_model_and_settings(
                        arguments(
                            profile="deployed",
                            checkpoint=checkpoint_path,
                        ),
                        torch.device("cpu"),
                    )

    def test_deployed_profile_rejects_metric_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "deployed.pt"
            checkpoint_path.touch()
            with patch(
                "src.evaluate_centernet.load_checkpoint",
                return_value=checkpoint_payload(0.10, deployed=True),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "rejects metric-affecting overrides",
                ):
                    build_model_and_settings(
                        arguments(
                            profile="deployed",
                            checkpoint=checkpoint_path,
                            peak_thresh=0.10,
                        ),
                        torch.device("cpu"),
                    )


if __name__ == "__main__":
    unittest.main()
