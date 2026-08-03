from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.prepare_nih_annotations import coco_from_predictions


class CocoSourceDatasetTest(unittest.TestCase):
    def test_custom_source_dataset_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            images_dir = Path(temporary_directory)
            image_name = "lumos_AP_001.png"
            image = np.zeros((20, 30), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(images_dir / image_name), image))

            coco, stats = coco_from_predictions(
                [
                    {
                        "image": image_name,
                        "original_width": 30,
                        "original_height": 20,
                        "predictions": [],
                    }
                ],
                images_dir=images_dir,
                source_checkpoint="checkpoint.pt",
                prediction_policy="test policy",
                source_dataset="LUMOS AP",
            )

        self.assertEqual(stats["images"], 1)
        self.assertEqual(coco["images"][0]["source_dataset"], "LUMOS AP")
        self.assertEqual(
            coco["info"]["description"],
            "LUMOS AP vertebra corner pseudo-annotations",
        )


if __name__ == "__main__":
    unittest.main()
