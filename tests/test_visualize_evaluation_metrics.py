import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.visualization.visualize_evaluation_metrics import (
    FRACTURE_READINESS_NOTICE,
    checkpoint_label,
    load_fracture_readiness_inputs,
    parse_args,
    save_page,
    short_source,
)


class VisualizeEvaluationMetricsTests(unittest.TestCase):
    def test_general_focus_remains_the_cli_default(self) -> None:
        with patch(
            "sys.argv",
            ["visualize", "--metrics-csv", "metrics.csv", "--output-pdf", "out.pdf"],
        ):
            args = parse_args()
        self.assertEqual(args.focus, "general")

    def test_fracture_readiness_focus_is_selectable(self) -> None:
        with patch(
            "sys.argv",
            [
                "visualize",
                "--metrics-csv",
                "metrics.csv",
                "--output-pdf",
                "out.pdf",
                "--focus",
                "fracture-readiness",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.focus, "fracture-readiness")

    def test_one_page_summary_focus_is_selectable(self) -> None:
        with patch(
            "sys.argv",
            [
                "visualize",
                "--metrics-csv",
                "metrics.csv",
                "--output-pdf",
                "out.pdf",
                "--focus",
                "one-page-summary",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.focus, "one-page-summary")

    def test_fracture_readiness_inputs_are_inferred_from_metrics_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "evaluation"
            checkpoint_root = run_root / "best_usable_recall"
            checkpoint_root.mkdir(parents=True)
            metrics_csv = checkpoint_root / "metrics.csv"
            metrics_csv.write_text(
                "model,scope,source_dataset\nspine_chain,overall_micro,\n",
                encoding="utf-8",
            )
            (checkpoint_root / "metrics.json").write_text(
                json.dumps({"metadata": {"checkpoint": "best_usable_recall.pt"}}),
                encoding="utf-8",
            )
            for filename in ("per_image_metrics.csv", "per_instance_metrics.csv"):
                with (checkpoint_root / filename).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["model"])
                    writer.writeheader()
                    writer.writerow({"model": "spine_chain"})
            (run_root / "evaluation_manifest.json").write_text(
                json.dumps(
                    {
                        "recommended_checkpoint": "best_usable_recall.pt",
                        "checkpoints": {"best_usable_recall.pt": {"epoch": 49}},
                    }
                ),
                encoding="utf-8",
            )

            metrics, manifest, per_image, per_instance = (
                load_fracture_readiness_inputs(metrics_csv)
            )

            self.assertEqual(
                metrics["metadata"]["checkpoint"], "best_usable_recall.pt"
            )
            self.assertEqual(manifest["recommended_checkpoint"], "best_usable_recall.pt")
            self.assertEqual(per_image, [{"model": "spine_chain"}])
            self.assertEqual(per_instance, [{"model": "spine_chain"}])

    def test_missing_supporting_artifacts_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics_csv = Path(directory) / "checkpoint" / "metrics.csv"
            metrics_csv.parent.mkdir()
            metrics_csv.write_text("model,scope,source_dataset\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "metrics JSON"):
                load_fracture_readiness_inputs(metrics_csv)

    def test_checkpoint_label_uses_manifest_epoch(self) -> None:
        metrics = {
            "metadata": {
                "checkpoint": "/models/80 epochs/best_usable_recall.pt",
            }
        }
        manifest = {"checkpoints": {"best_usable_recall.pt": {"epoch": 49}}}
        self.assertEqual(
            checkpoint_label(metrics, manifest),
            "best_usable_recall.pt (epoch 49)",
        )

    def test_five_source_short_labels_include_lumos(self) -> None:
        names = [
            "BUU AP",
            "Lumos AP",
            "MICCAI-2019",
            "Mendeley PA",
            "NIH ChestX-ray14",
        ]
        self.assertEqual(
            [short_source(name) for name in names],
            ["BUU", "Lumos", "MICCAI", "Mendeley", "NIH"],
        )

    def test_notice_is_non_diagnostic(self) -> None:
        self.assertIn("No fracture labels", FRACTURE_READINESS_NOTICE)
        self.assertIn("no validated fracture classifier", FRACTURE_READINESS_NOTICE)
        self.assertIn("clinical review is required", FRACTURE_READINESS_NOTICE)

    def test_save_page_can_write_a_named_standalone_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_pdf = Path(directory) / "summary.pdf"
            output_png = Path(directory) / "summary.png"
            with PdfPages(output_pdf) as pdf:
                figure = plt.figure(figsize=(2, 1))
                save_page(
                    pdf,
                    figure,
                    1,
                    None,
                    output_png=output_png,
                )

            self.assertTrue(output_pdf.is_file())
            self.assertTrue(output_png.is_file())


if __name__ == "__main__":
    unittest.main()
