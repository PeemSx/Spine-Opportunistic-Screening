from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.evaluation.training_validation import evaluate_validation_landmarks
from src.train_centernet import (
    LANDMARK_LOG_FIELDS,
    landmark_checkpoint_guardrails,
    merge_best_landmark_values,
    read_best_landmark_values,
    snapshot_validation_artifacts,
    write_validation_artifacts,
)


def square(center_x: float, center_y: float, size: float = 10.0) -> np.ndarray:
    half = size / 2.0
    return np.asarray(
        [
            [center_x - half, center_y - half],
            [center_x + half, center_y - half],
            [center_x - half, center_y + half],
            [center_x + half, center_y + half],
        ],
        dtype=np.float32,
    )


class _SingleBatchLoader:
    def __init__(self, batch: dict[str, object]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        yield self.batch


class _DummyModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"unused": inputs[:, :1]}


class TrainingLandmarkValidationTests(unittest.TestCase):
    def test_validation_reports_raw_chain_source_scale_and_signed_residuals(self) -> None:
        gt_corners = square(50.0, 50.0)
        predicted_corners = gt_corners + np.asarray([2.0, -1.0], dtype=np.float32)
        decoded = [
            {
                "scores": np.asarray([0.95], dtype=np.float32),
                "centers": np.asarray([[52.0, 49.0]], dtype=np.float32),
                "corners": np.asarray([predicted_corners], dtype=np.float32),
            }
        ]
        batch: dict[str, object] = {
            "input": torch.zeros((1, 3, 100, 100), dtype=torch.float32),
            "gt_centers": torch.tensor([[[50.0, 50.0]]], dtype=torch.float32),
            "gt_corners": torch.from_numpy(gt_corners[None, None].copy()),
            "gt_centers_original": torch.tensor(
                [[[50.0, 50.0]]], dtype=torch.float32
            ),
            "gt_corners_original": torch.from_numpy(
                gt_corners[None, None].copy()
            ),
            "gt_count": torch.tensor([1], dtype=torch.int64),
            "file_name": ["sample.png"],
            "image_id": torch.tensor([7], dtype=torch.int64),
            "source_dataset": ["source-a"],
            "patient_cluster_id": ["patient-a"],
            "original_width": torch.tensor([100]),
            "original_height": torch.tensor([100]),
            "resized_width": torch.tensor([100]),
            "resized_height": torch.tensor([100]),
            "pad_left": torch.tensor([0]),
            "pad_top": torch.tensor([0]),
            "scale": torch.tensor([1.0]),
            "input_size": torch.tensor([100]),
        }
        chain_settings = {
            "duplicate_iou": 0.18,
            "duplicate_center_scale": 0.35,
            "score_thresh": 0.18,
            "score_weight": 3.0,
            "min_len": 1,
        }

        with patch(
            "src.evaluation.training_validation.decode_centernet_outputs",
            return_value=decoded,
        ):
            result = evaluate_validation_landmarks(
                model=_DummyModel(),
                loader=_SingleBatchLoader(batch),
                device=torch.device("cpu"),
                down_ratio=4,
                peak_thresh=0.10,
                topk=50,
                chain_settings=chain_settings,
                progress_every=0,
            )

        self.assertAlmostEqual(result.flat_metrics["spine_chain_center_f1_0.20d"], 1.0)
        self.assertAlmostEqual(result.flat_metrics["spine_chain_pck_0.10"], 0.0)
        self.assertAlmostEqual(
            result.flat_metrics["spine_chain_usable_vertebra_recall"],
            0.0,
        )
        report = result.report["models"]["spine_chain"]
        self.assertEqual(report["by_source"][0]["source_dataset"], "source-a")
        self.assertEqual(report["by_scale"][0]["scale_bin"], "large")
        self.assertAlmostEqual(report["by_scale"][0]["tl_dx_px_bias"], 2.0)
        self.assertAlmostEqual(
            report["by_source_scale"][0]["tl_dy_px_bias"],
            -1.0,
        )
        matched = next(row for row in result.instance_rows if row["model"] == "spine_chain")
        self.assertAlmostEqual(matched["tl_dx_px"], 2.0)
        self.assertAlmostEqual(matched["tl_dy_px"], -1.0)

    def test_checkpoint_guardrails_require_all_detection_checks(self) -> None:
        args = argparse.Namespace(
            landmark_checkpoint_min_f1_020d=0.85,
            landmark_checkpoint_min_worst_source_recall_020d=0.75,
            landmark_checkpoint_max_fp_per_image=1.0,
            landmark_checkpoint_max_count_mae=1.5,
        )
        metrics = {
            "spine_chain_source_macro_center_f1_0.20d": 0.90,
            "spine_chain_worst_source_center_recall_0.20d": 0.80,
            "spine_chain_false_positives_per_image": 0.50,
            "spine_chain_count_mae": 1.0,
        }
        self.assertTrue(landmark_checkpoint_guardrails(metrics, args)["passed"])
        metrics["spine_chain_false_positives_per_image"] = 1.01
        self.assertFalse(landmark_checkpoint_guardrails(metrics, args)["passed"])

    def test_resume_metrics_restore_guarded_landmark_best_values(self) -> None:
        args = argparse.Namespace(
            landmark_checkpoint_min_f1_020d=0.85,
            landmark_checkpoint_min_worst_source_recall_020d=0.75,
            landmark_checkpoint_max_fp_per_image=1.0,
            landmark_checkpoint_max_count_mae=1.5,
        )
        metrics = {
            "spine_chain_source_macro_center_f1_0.20d": 0.90,
            "spine_chain_worst_source_center_recall_0.20d": 0.80,
            "spine_chain_false_positives_per_image": 0.50,
            "spine_chain_count_mae": 1.0,
            "spine_chain_corner_nme_mean": 0.07,
            "spine_chain_usable_vertebra_recall": 0.80,
        }
        self.assertEqual(
            merge_best_landmark_values(
                best_corner_nme=float("inf"),
                best_usable_recall=-1.0,
                metrics=metrics,
                args=args,
            ),
            (0.07, 0.80),
        )

        metrics["spine_chain_source_macro_center_f1_0.20d"] = 0.84
        self.assertEqual(
            merge_best_landmark_values(
                best_corner_nme=0.08,
                best_usable_recall=0.70,
                metrics=metrics,
                args=args,
            ),
            (0.08, 0.70),
        )

    def test_best_landmark_values_recompute_current_guardrails(self) -> None:
        args = argparse.Namespace(
            landmark_checkpoint_min_f1_020d=0.85,
            landmark_checkpoint_min_worst_source_recall_020d=0.75,
            landmark_checkpoint_max_fp_per_image=1.0,
            landmark_checkpoint_max_count_mae=1.5,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "validation_landmark_log.csv"
            with log_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=LANDMARK_LOG_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": 1,
                        "guardrails_passed": True,
                        "spine_chain_source_macro_center_f1_0.20d": 0.90,
                        "spine_chain_worst_source_center_recall_0.20d": 0.80,
                        "spine_chain_false_positives_per_image": 1.10,
                        "spine_chain_count_mae": 1.0,
                        "spine_chain_corner_nme_mean": 0.01,
                        "spine_chain_usable_vertebra_recall": 0.99,
                    }
                )
                writer.writerow(
                    {
                        "epoch": 2,
                        "guardrails_passed": False,
                        "spine_chain_source_macro_center_f1_0.20d": 0.90,
                        "spine_chain_worst_source_center_recall_0.20d": 0.80,
                        "spine_chain_false_positives_per_image": 0.50,
                        "spine_chain_count_mae": 1.0,
                        "spine_chain_corner_nme_mean": 0.07,
                        "spine_chain_usable_vertebra_recall": 0.80,
                    }
                )
            self.assertEqual(
                read_best_landmark_values(log_path, args),
                (0.07, 0.80),
            )

    def test_validation_artifacts_are_snapshotted_with_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            report = {"schema_version": 1, "images": 1}
            rows = [{"model": "spine_chain", "status": "matched"}]
            write_validation_artifacts(run_dir, report, rows)
            snapshot_validation_artifacts(run_dir, "best_corner_nme")

            with (run_dir / "best_corner_nme_metrics.json").open(
                "r", encoding="utf-8"
            ) as file:
                self.assertEqual(json.load(file), report)
            with (run_dir / "best_corner_nme_instances.csv").open(
                "r", newline="", encoding="utf-8"
            ) as file:
                self.assertEqual(list(csv.DictReader(file)), rows)


if __name__ == "__main__":
    unittest.main()
