from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.scorecard import (
    aggregate_evaluations,
    bootstrap_confidence_intervals,
    evaluate_acceptance_gates,
    evaluate_prediction,
    hungarian_center_match,
    postprocessing_comparison,
    scale_summary_rows,
    source_summary_rows,
    valid_quadrilateral,
    vertebral_diagonals,
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


def prediction(
    centers: list[list[float]],
    corners: list[np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "scores": np.full((len(centers),), 0.9, dtype=np.float32),
        "centers": np.asarray(centers, dtype=np.float32).reshape(-1, 2),
        "corners": np.asarray(corners, dtype=np.float32).reshape(-1, 4, 2),
    }


class MatchingTests(unittest.TestCase):
    def test_empty_inputs(self) -> None:
        corners = np.zeros((0, 4, 2), dtype=np.float32)
        matches = hungarian_center_match(
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            corners,
            0.20,
        )
        self.assertEqual(matches, [])

    def test_hungarian_matching_is_one_to_one(self) -> None:
        gt_centers = np.asarray([[10.0, 10.0], [30.0, 10.0]], dtype=np.float32)
        gt_corners = np.asarray([square(10.0, 10.0), square(30.0, 10.0)])
        pred_centers = np.asarray(
            [[10.1, 10.0], [10.2, 10.0], [29.9, 10.0]],
            dtype=np.float32,
        )
        matches = hungarian_center_match(
            pred_centers,
            gt_centers,
            gt_corners,
            0.20,
        )
        self.assertEqual(len(matches), 2)
        self.assertEqual(len({match.pred_index for match in matches}), 2)
        self.assertEqual(len({match.gt_index for match in matches}), 2)

    def test_threshold_boundary_and_scale_invariance(self) -> None:
        gt_corners = np.asarray([square(10.0, 10.0)])
        diagonal = float(vertebral_diagonals(gt_corners)[0])
        pred = np.asarray([[10.0 + 0.20 * diagonal, 10.0]], dtype=np.float32)
        matches = hungarian_center_match(
            pred,
            np.asarray([[10.0, 10.0]], dtype=np.float32),
            gt_corners,
            0.20,
        )
        self.assertEqual(len(matches), 1)

        scale = 7.0
        scaled = hungarian_center_match(
            pred * scale,
            np.asarray([[10.0, 10.0]], dtype=np.float32) * scale,
            gt_corners * scale,
            0.20,
        )
        self.assertEqual(len(scaled), 1)
        self.assertAlmostEqual(
            matches[0].normalized_distance,
            scaled[0].normalized_distance,
            places=6,
        )


class LandmarkAndCountingTests(unittest.TestCase):
    def test_geometry_validation(self) -> None:
        self.assertTrue(valid_quadrilateral(square(10.0, 10.0)))
        bow_tie = np.asarray(
            [[5.0, 5.0], [15.0, 15.0], [5.0, 15.0], [15.0, 5.0]],
            dtype=np.float32,
        )
        self.assertFalse(valid_quadrilateral(bow_tie))
        non_finite = square(10.0, 10.0)
        non_finite[0, 0] = np.nan
        self.assertFalse(valid_quadrilateral(non_finite))

    def test_misses_penalize_end_to_end_pck_and_usable_recall(self) -> None:
        gt_centers = np.asarray([[10.0, 10.0], [30.0, 30.0]], dtype=np.float32)
        gt_corners = np.asarray([square(10.0, 10.0), square(30.0, 30.0)])
        invalid_fp = np.asarray(
            [[70.0, 70.0], [80.0, 80.0], [70.0, 80.0], [80.0, 70.0]],
            dtype=np.float32,
        )
        item = evaluate_prediction(
            model="raw",
            prediction=prediction(
                [[10.0, 10.0], [75.0, 75.0]],
                [square(10.0, 10.0), invalid_fp],
            ),
            gt_centers=gt_centers,
            gt_corners=gt_corners,
            image="sample.png",
            image_id=1,
            source_dataset="source-a",
            cluster_id="patient-a",
            image_width=100,
            image_height=100,
        )
        self.assertEqual(item.detection["0.20"], {"tp": 1, "fp": 1, "fn": 1})
        self.assertEqual(item.usable_count, 1)
        self.assertEqual(item.invalid_geometry_count, 1)

        summary = aggregate_evaluations([item], model="raw", scope="overall")
        self.assertAlmostEqual(summary["pck_0.10"], 1.0)
        self.assertAlmostEqual(summary["end_to_end_pck_0.10"], 0.5)
        self.assertAlmostEqual(summary["usable_vertebra_recall"], 0.5)
        self.assertAlmostEqual(summary["count_mae"], 0.0)
        self.assertAlmostEqual(summary["count_bias"], 0.0)
        self.assertEqual(summary["count_p90_absolute_error"], 0.0)

        rows = {row["status"]: row for row in item.instance_rows}
        self.assertAlmostEqual(rows["matched"]["pck_0.10"], 1.0)
        self.assertEqual(rows["missed_ground_truth"]["pck_0.10"], 0.0)

    def test_nme_boundary_is_usable(self) -> None:
        gt = square(50.0, 50.0)
        diagonal = float(vertebral_diagonals(np.asarray([gt]))[0])
        shifted = gt + np.asarray([0.10 * diagonal, 0.0], dtype=np.float32)
        item = evaluate_prediction(
            model="raw",
            prediction=prediction(
                [[50.0 + 0.10 * diagonal, 50.0]],
                [shifted],
            ),
            gt_centers=np.asarray([[50.0, 50.0]], dtype=np.float32),
            gt_corners=np.asarray([gt]),
            image="boundary.png",
            image_id=2,
            source_dataset="source-a",
            cluster_id="image:2",
            image_width=100,
            image_height=100,
        )
        self.assertEqual(item.usable_count, 1)
        self.assertAlmostEqual(item.nmes[0], 0.10, places=6)

    def test_signed_corner_residuals_use_prediction_minus_ground_truth(self) -> None:
        gt = square(50.0, 50.0, size=20.0)
        shift = np.asarray([2.0, -3.0], dtype=np.float32)
        item = evaluate_prediction(
            model="spine_chain",
            prediction=prediction([[52.0, 47.0]], [gt + shift]),
            gt_centers=np.asarray([[50.0, 50.0]], dtype=np.float32),
            gt_corners=np.asarray([gt]),
            image="residual.png",
            image_id=3,
            source_dataset="source-a",
            cluster_id="image:3",
            image_width=100,
            image_height=100,
        )
        matched = item.instance_rows[0]
        diagonal = float(vertebral_diagonals(np.asarray([gt]))[0])
        self.assertAlmostEqual(matched["tl_dx_px"], 2.0)
        self.assertAlmostEqual(matched["tl_dy_px"], -3.0)
        self.assertAlmostEqual(matched["tl_dx_normalized"], 2.0 / diagonal)
        self.assertAlmostEqual(matched["tl_dy_normalized"], -3.0 / diagonal)

        summary = aggregate_evaluations(
            [item],
            model="spine_chain",
            scope="overall",
        )
        self.assertAlmostEqual(
            summary["tl_dx_normalized_bias"],
            2.0 / diagonal,
        )
        self.assertAlmostEqual(
            summary["tl_dy_normalized_bias"],
            -3.0 / diagonal,
        )

    def test_scale_breakdown_uses_diagonal_fraction_and_preserves_source(self) -> None:
        small_gt = square(20.0, 20.0, size=4.0)
        large_gt = square(60.0, 60.0, size=20.0)
        items = [
            evaluate_prediction(
                model="spine_chain",
                prediction=prediction([[20.0, 20.0]], [small_gt]),
                gt_centers=np.asarray([[20.0, 20.0]], dtype=np.float32),
                gt_corners=np.asarray([small_gt]),
                image="small.png",
                image_id=4,
                source_dataset="source-a",
                cluster_id="image:4",
                image_width=100,
                image_height=100,
            ),
            evaluate_prediction(
                model="spine_chain",
                prediction=prediction([[60.0, 60.0]], [large_gt]),
                gt_centers=np.asarray([[60.0, 60.0]], dtype=np.float32),
                gt_corners=np.asarray([large_gt]),
                image="large.png",
                image_id=5,
                source_dataset="source-b",
                cluster_id="image:5",
                image_width=100,
                image_height=100,
            ),
        ]
        by_scale, by_source_scale = scale_summary_rows(items, "spine_chain")
        scale_counts = {row["scale_bin"]: row["gt_vertebrae"] for row in by_scale}
        self.assertEqual(scale_counts, {"small": 1, "large": 1})
        self.assertEqual(
            {
                (row["source_dataset"], row["scale_bin"])
                for row in by_source_scale
            },
            {("source-a", "small"), ("source-b", "large")},
        )

    def test_worst_source_corner_nme_selects_highest_error(self) -> None:
        gt = square(50.0, 50.0, size=20.0)
        items = [
            evaluate_prediction(
                model="spine_chain",
                prediction=prediction([[50.0, 50.0]], [gt]),
                gt_centers=np.asarray([[50.0, 50.0]], dtype=np.float32),
                gt_corners=np.asarray([gt]),
                image="source-a.png",
                image_id=6,
                source_dataset="source-a",
                cluster_id="image:6",
                image_width=100,
                image_height=100,
            ),
            evaluate_prediction(
                model="spine_chain",
                prediction=prediction([[50.0, 50.0]], [gt + [2.0, 0.0]]),
                gt_centers=np.asarray([[50.0, 50.0]], dtype=np.float32),
                gt_corners=np.asarray([gt]),
                image="source-b.png",
                image_id=7,
                source_dataset="source-b",
                cluster_id="image:7",
                image_width=100,
                image_height=100,
            ),
        ]
        _, per_source, _, worst = source_summary_rows(items, "spine_chain")
        expected = max(float(row["corner_nme_mean"]) for row in per_source)
        self.assertAlmostEqual(worst["corner_nme_mean"], expected)


class CobbPostprocessingAndGateTests(unittest.TestCase):
    def _item(
        self,
        *,
        model: str,
        pred_centers: list[list[float]],
        pred_corners: list[np.ndarray],
        image_id: int = 1,
    ):
        gt_corners = np.asarray([square(20.0, 20.0), square(20.0, 50.0)])
        return evaluate_prediction(
            model=model,
            prediction=prediction(pred_centers, pred_corners),
            gt_centers=np.asarray([[20.0, 20.0], [20.0, 50.0]], dtype=np.float32),
            gt_corners=gt_corners,
            image=f"{image_id}.png",
            image_id=image_id,
            source_dataset="source-a",
            cluster_id=f"patient-{image_id // 2}",
            image_width=100,
            image_height=100,
        )

    def test_cobb_invalid_prediction_reduces_coverage(self) -> None:
        item = self._item(
            model="raw",
            pred_centers=[[20.0, 20.0]],
            pred_corners=[square(20.0, 20.0)],
        )
        summary = aggregate_evaluations([item], model="raw", scope="overall")
        self.assertEqual(summary["cobb_eligible_images"], 1)
        self.assertEqual(summary["cobb_valid_predictions"], 0)
        self.assertEqual(summary["cobb_coverage"], 0.0)
        self.assertIsNone(summary["cobb_mae_deg"])
        self.assertIsNotNone(summary["zero_cobb_baseline_mae_deg"])
        self.assertEqual(summary["cobb_lt_5deg_images"], 1)
        self.assertEqual(summary["cobb_lt_5deg_coverage"], 0.0)
        self.assertEqual(summary["cobb_lt_5deg_at_10deg"], 0.0)

    def test_raw_chain_deltas_and_acceptance_gates(self) -> None:
        raw_item = self._item(
            model="raw",
            pred_centers=[[20.0, 20.0], [20.0, 50.0], [80.0, 80.0]],
            pred_corners=[
                square(20.0, 20.0),
                square(20.0, 50.0),
                square(80.0, 80.0),
            ],
        )
        chain_item = self._item(
            model="spine_chain",
            pred_centers=[[20.0, 20.0], [20.0, 50.0]],
            pred_corners=[square(20.0, 20.0), square(20.0, 50.0)],
        )
        raw = aggregate_evaluations([raw_item], model="raw", scope="overall")
        chain = aggregate_evaluations(
            [chain_item],
            model="spine_chain",
            scope="overall",
        )
        comparison = postprocessing_comparison(raw, chain)
        self.assertEqual(chain["center_f1_0.15d"], 1.0)
        self.assertEqual(comparison["false_positive_reduction"], 1.0)
        self.assertEqual(comparison["recall_loss_0.20d"], 0.0)
        self.assertLess(comparison["count_mae_change"], 0.0)
        self.assertEqual(chain["cobb_endpoint_exact_rate"], 1.0)
        self.assertEqual(chain["cobb_endpoint_within_one_rate"], 1.0)
        self.assertEqual(chain["cobb_lt_5deg_coverage"], 1.0)
        self.assertEqual(chain["cobb_lt_5deg_at_10deg"], 1.0)

        result = evaluate_acceptance_gates(
            chain_overall=chain,
            chain_source_macro=chain,
            chain_worst_source=chain,
            comparison=comparison,
        )
        self.assertEqual(len(result["checks"]), 13)
        self.assertNotIn("invalid_geometry_rate", result["checks"])
        self.assertNotIn("absolute_cobb_bias_deg", result["checks"])

    def test_bootstrap_is_clustered_and_deterministic(self) -> None:
        evaluations = [
            self._item(
                model="spine_chain",
                pred_centers=[[20.0, 20.0], [20.0, 50.0]],
                pred_corners=[square(20.0, 20.0), square(20.0, 50.0)],
                image_id=index,
            )
            for index in range(4)
        ]
        first = bootstrap_confidence_intervals(
            evaluations,
            model="spine_chain",
            samples=40,
            seed=20260627,
        )
        second = bootstrap_confidence_intervals(
            evaluations,
            model="spine_chain",
            samples=40,
            seed=20260627,
        )
        self.assertEqual(first, second)
        self.assertIn("center_f1_0.20d", first)


if __name__ == "__main__":
    unittest.main()
