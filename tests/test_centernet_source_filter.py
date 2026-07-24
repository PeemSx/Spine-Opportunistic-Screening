from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.centernet_dataset import CenterNetCocoDataset


class CenterNetSourceFilterTests(unittest.TestCase):
    def _dataset_root(self, temporary_root: str) -> Path:
        root = Path(temporary_root)
        split_dir = root / "test"
        split_dir.mkdir(parents=True)
        images = [
            {"id": 1, "file_name": "images/a.png", "source_dataset": "Legacy"},
            {"id": 2, "file_name": "images/b.png", "source_dataset": "NIH ChestX-ray14"},
            {"id": 3, "file_name": "images/c.png", "source_dataset": "NIH ChestX-ray14"},
        ]
        annotations = [
            {"id": index, "image_id": index, "category_id": 1}
            for index in range(1, 4)
        ]
        (split_dir / "_annotations.keypoints.coco.json").write_text(
            json.dumps({"images": images, "annotations": annotations, "categories": []}),
            encoding="utf-8",
        )
        return root

    def test_source_filter_is_case_insensitive_and_precedes_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            dataset = CenterNetCocoDataset(
                self._dataset_root(temporary_root),
                "test",
                config_path=None,
                augment=False,
                source_dataset="nih chestx-ray14",
                limit=1,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.samples[0]["image"]["id"], 2)

    def test_unknown_source_reports_available_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            with self.assertRaisesRegex(ValueError, "Available sources"):
                CenterNetCocoDataset(
                    self._dataset_root(temporary_root),
                    "test",
                    config_path=None,
                    augment=False,
                    source_dataset="missing",
                )


if __name__ == "__main__":
    unittest.main()
