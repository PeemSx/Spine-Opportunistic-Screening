from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.data.inference_geometry import (
    map_points_to_model,
    map_points_to_original,
    resize_pad_image,
    valid_center_mask,
)
from src.postprocessing.spine_chain import (
    prediction_to_candidates,
    select_spine_chain,
)


class InferenceGeometryTests(unittest.TestCase):
    def test_round_trip_and_padding_filter(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, meta = resize_pad_image(image, input_size=1024)
        self.assertEqual(padded.shape, (1024, 1024, 3))
        self.assertEqual(meta.pad_left, 0)
        self.assertEqual(meta.pad_top, 256)

        original = np.asarray(
            [[0.0, 0.0], [199.0, 99.0], [100.0, 50.0]],
            dtype=np.float32,
        )
        model = map_points_to_model(original, meta)
        restored = map_points_to_original(model, meta)
        np.testing.assert_allclose(restored, original, atol=1e-5)

        original_centers = np.asarray(
            [[0.0, 0.0], [199.999, 99.999], [-0.01, 50.0], [200.0, 50.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            valid_center_mask(original_centers, meta),
            np.asarray([True, True, False, False]),
        )

    def test_synthetic_parity_with_deployed_api(self) -> None:
        api_root = Path(__file__).resolve().parents[2] / "Spine-Screening-API"
        self.assertTrue(api_root.exists(), "deployed API checkout is required")
        sys.path.insert(0, str(api_root))
        try:
            from app.inference.preprocessing import (
                map_points_to_original as api_map_points_to_original,
            )
            from app.inference.preprocessing import (
                resize_pad_image as api_resize_pad_image,
            )
            from app.inference.preprocessing import (
                valid_center_mask as api_valid_center_mask,
            )
            from app.inference.types import RawPrediction
            from app.postprocessing.spine_chain import (
                select_spine_chain as api_select_spine_chain,
            )

            rng = np.random.default_rng(20260627)
            image = rng.integers(0, 256, size=(173, 311, 3), dtype=np.uint8)
            local_image, local_meta = resize_pad_image(image, input_size=1024)
            api_image, api_meta = api_resize_pad_image(image, input_size=1024)
            np.testing.assert_array_equal(local_image, api_image)
            self.assertEqual(local_meta.to_dict(), asdict(api_meta))

            points = np.asarray(
                [[[0.0, 0.0], [512.5, 200.25], [1023.0, 1023.0]]],
                dtype=np.float32,
            )
            local_points = map_points_to_original(points, local_meta)
            api_points = api_map_points_to_original(points, api_meta)
            np.testing.assert_allclose(local_points, api_points, atol=1e-4)
            np.testing.assert_array_equal(
                valid_center_mask(local_points[0], local_meta),
                api_valid_center_mask(api_points[0], api_meta),
            )

            centers = np.asarray(
                [
                    [50.0, 10.0],
                    [50.5, 10.5],
                    [51.0, 30.0],
                    [53.0, 50.0],
                    [90.0, 80.0],
                ],
                dtype=np.float32,
            )
            corners = np.asarray(
                [
                    [
                        [center[0] - 5.0, center[1] - 4.0],
                        [center[0] + 5.0, center[1] - 4.0],
                        [center[0] - 5.0, center[1] + 4.0],
                        [center[0] + 5.0, center[1] + 4.0],
                    ]
                    for center in centers
                ],
                dtype=np.float32,
            )
            scores = np.asarray([0.95, 0.70, 0.91, 0.89, 0.20], dtype=np.float32)
            local_prediction = {
                "scores": scores,
                "centers": centers,
                "corners": corners,
            }
            _, local_selected, local_debug = select_spine_chain(
                prediction_to_candidates(local_prediction),
            )
            api_selection = api_select_spine_chain(
                RawPrediction(
                    scores=scores,
                    centers=centers,
                    corners=corners,
                )
            )
            self.assertEqual(
                [candidate.index for candidate in local_selected],
                [candidate.candidate_id for candidate in api_selection.selected],
            )
            np.testing.assert_allclose(
                np.asarray([candidate.center for candidate in local_selected]),
                np.asarray(
                    [candidate.center for candidate in api_selection.selected]
                ),
                atol=1e-4,
            )
            for key in (
                "score",
                "target_dy",
                "min_dy",
                "max_dy",
                "node_score_threshold",
            ):
                self.assertAlmostEqual(
                    float(local_debug[key]),
                    float(api_selection.debug[key]),
                    places=4,
                )
        finally:
            sys.path.remove(str(api_root))


if __name__ == "__main__":
    unittest.main()
