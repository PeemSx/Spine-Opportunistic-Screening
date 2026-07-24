from __future__ import annotations

import json
import unittest

import numpy as np

from src.analysis.vertebral_morphology import (
    MORPHOLOGY_DISCLAIMER,
    extract_chain_morphology,
    morphology_json_payload,
)
from src.postprocessing.spine_chain import SpineCandidate


def candidate(index: int, center_y: float, height: float) -> SpineCandidate:
    center = np.asarray([50.0, center_y], dtype=np.float32)
    corners = np.asarray(
        [
            [40.0, center_y - height / 2],
            [60.0, center_y - height / 2],
            [40.0, center_y + height / 2],
            [60.0, center_y + height / 2],
        ],
        dtype=np.float32,
    )
    return SpineCandidate(
        index=index,
        score=0.9,
        center=center,
        corners=corners,
        box=np.asarray([40.0, center_y - height / 2, 60.0, center_y + height / 2]),
    )


class VertebralMorphologyTests(unittest.TestCase):
    def test_neighbor_features_and_centernet_provenance(self) -> None:
        candidates = [candidate(0, 10, 10), candidate(1, 30, 8), candidate(2, 50, 10)]
        rows = extract_chain_morphology("image.jpg", candidates)
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3])
        self.assertEqual(
            [row["landmark_source"] for row in rows],
            ["centernet", "centernet", "centernet"],
        )
        self.assertEqual(rows[1]["fallback_reason"], "")
        self.assertAlmostEqual(rows[1]["height_ratio_to_neighbors"], 0.8)
        self.assertAlmostEqual(rows[1]["relative_height_deviation"], 0.2)
        self.assertAlmostEqual(rows[1]["endplate_nonparallel_deg"], 0.0)

    def test_payload_is_strict_json_and_cautious(self) -> None:
        rows = extract_chain_morphology("single.jpg", [candidate(0, 10, 10)])
        payload = morphology_json_payload("single.jpg", rows)
        encoded = json.dumps(payload, allow_nan=False)
        self.assertIn("not a diagnosis", MORPHOLOGY_DISCLAIMER)
        self.assertIn("original_image_pixels", encoded)


if __name__ == "__main__":
    unittest.main()
