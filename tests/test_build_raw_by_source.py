from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.data.build_raw_by_source import (
    ANNOTATION_FILE_NAME,
    SUMMARY_FILE_NAME,
    build_dataset,
    load_json,
)


CATEGORY = {
    "id": 1,
    "name": "vertebra",
    "supercategory": "spine",
    "keypoints": ["top_left", "top_right", "bottom_left", "bottom_right"],
    "skeleton": [[1, 2], [3, 4], [1, 3], [2, 4]],
}


class BuildRawBySourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)
        self.processed_root = self.workspace / "processed"
        self.output_root = self.workspace / "raw"
        self.records_by_split = {
            "train": [(10, "buu_ap", "BUU AP", "buu_train.jpg", b"train-buu")],
            "val": [(20, "lumos_ap", "Lumos AP", "lumos_val.png", b"val-lumos")],
            "test": [(30, "buu_ap", "BUU AP", "buu_test.jpg", b"test-buu")],
        }
        self._write_fixture()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _annotation(image_id: int, source_name: str) -> dict[str, Any]:
        return {
            "id": image_id * 10,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [1.0, 1.0, 8.0, 8.0],
            "area": 64.0,
            "iscrowd": 0,
            "num_keypoints": 4,
            "keypoints": [1, 1, 2, 9, 1, 2, 1, 9, 2, 9, 9, 2],
            "segmentation": [[1, 1, 9, 1, 9, 9, 1, 9]],
            "source_dataset": source_name,
            "source_annotation_id": image_id * 10,
        }

    def _write_fixture(self) -> None:
        for split, records in self.records_by_split.items():
            images = []
            annotations = []
            for image_id, source_key, source_name, basename, content in records:
                image_path = self.processed_root / split / "images" / source_key / basename
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(content)
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"images/{source_key}/{basename}",
                        "width": 10,
                        "height": 10,
                        "source_dataset": source_name,
                        "source_image_id": image_id,
                    }
                )
                annotations.append(self._annotation(image_id, source_name))
            annotation_path = self.processed_root / split / ANNOTATION_FILE_NAME
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(
                json.dumps(
                    {
                        "info": {
                            "version": "fixture_v1",
                            "split": split,
                            "split_seed": 123,
                        },
                        "licenses": [],
                        "images": images,
                        "annotations": annotations,
                        "categories": [CATEGORY],
                    }
                ),
                encoding="utf-8",
            )

    def _processed_snapshot(self) -> dict[str, bytes]:
        return {
            path.relative_to(self.processed_root).as_posix(): path.read_bytes()
            for path in sorted(self.processed_root.rglob("*"))
            if path.is_file()
        }

    def test_build_combines_splits_into_independent_source_coco_datasets(self) -> None:
        processed_before = self._processed_snapshot()

        summary = build_dataset(
            processed_root=self.processed_root,
            output_root=self.output_root,
        )

        self.assertEqual(summary["totals"], {"images": 3, "annotations": 3})
        self.assertEqual(set(summary["sources"]), {"buu_ap", "lumos_ap"})
        self.assertEqual(summary["sources"]["buu_ap"]["images"], 2)
        self.assertEqual(summary["sources"]["buu_ap"]["annotations"], 2)
        self.assertEqual(
            summary["sources"]["buu_ap"]["former_splits"],
            {
                "train": {"images": 1, "annotations": 1},
                "val": {"images": 0, "annotations": 0},
                "test": {"images": 1, "annotations": 1},
            },
        )

        buu_payload = load_json(self.output_root / "buu_ap" / ANNOTATION_FILE_NAME)
        self.assertEqual(buu_payload["info"]["split"], "all")
        self.assertEqual(buu_payload["info"]["source_key"], "buu_ap")
        self.assertEqual([image["id"] for image in buu_payload["images"]], [1, 2])
        self.assertEqual(
            [image["processed_image_id"] for image in buu_payload["images"]],
            [10, 30],
        )
        self.assertEqual(
            [image["processed_split"] for image in buu_payload["images"]],
            ["train", "test"],
        )
        self.assertEqual(
            [image["file_name"] for image in buu_payload["images"]],
            ["images/buu_train.jpg", "images/buu_test.jpg"],
        )
        self.assertEqual(
            [annotation["id"] for annotation in buu_payload["annotations"]],
            [1, 2],
        )
        self.assertEqual(
            [annotation["image_id"] for annotation in buu_payload["annotations"]],
            [1, 2],
        )
        self.assertEqual(
            [annotation["processed_annotation_id"] for annotation in buu_payload["annotations"]],
            [100, 300],
        )
        self.assertEqual(buu_payload["categories"], [CATEGORY])

        copied_path = self.output_root / "buu_ap" / "images" / "buu_train.jpg"
        source_path = self.processed_root / "train" / "images" / "buu_ap" / "buu_train.jpg"
        self.assertEqual(copied_path.read_bytes(), source_path.read_bytes())
        self.assertFalse(copied_path.is_symlink())
        self.assertFalse(os.path.samefile(copied_path, source_path))
        self.assertTrue((self.output_root / SUMMARY_FILE_NAME).is_file())
        self.assertFalse(any((self.output_root / split).exists() for split in self.records_by_split))
        self.assertEqual(self._processed_snapshot(), processed_before)

    def test_non_empty_output_is_rejected_without_changes(self) -> None:
        self.output_root.mkdir()
        marker = self.output_root / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "Output directory is not empty"):
            build_dataset(processed_root=self.processed_root, output_root=self.output_root)

        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse(self.output_root.with_name("raw.building").exists())

    def test_duplicate_source_filename_across_splits_is_rejected(self) -> None:
        self.records_by_split["val"].append(
            (40, "buu_ap", "BUU AP", "buu_train.jpg", b"different-buu")
        )
        self._write_fixture()

        with self.assertRaisesRegex(ValueError, "Filename collision"):
            build_dataset(processed_root=self.processed_root, output_root=self.output_root)

        self.assertFalse(self.output_root.exists())
        self.assertFalse(self.output_root.with_name("raw.building").exists())

    def test_reused_processed_ids_are_remapped_with_provenance(self) -> None:
        self.records_by_split["test"] = [
            (10, "buu_ap", "BUU AP", "buu_test.jpg", b"test-buu")
        ]
        self._write_fixture()

        build_dataset(processed_root=self.processed_root, output_root=self.output_root)

        payload = load_json(self.output_root / "buu_ap" / ANNOTATION_FILE_NAME)
        self.assertEqual([image["id"] for image in payload["images"]], [1, 2])
        self.assertEqual([image["processed_image_id"] for image in payload["images"]], [10, 10])
        self.assertEqual(
            [image["processed_split"] for image in payload["images"]],
            ["train", "test"],
        )
        self.assertEqual([annotation["id"] for annotation in payload["annotations"]], [1, 2])
        self.assertEqual(
            [annotation["processed_annotation_id"] for annotation in payload["annotations"]],
            [100, 100],
        )

    def test_missing_processed_image_is_rejected(self) -> None:
        missing_path = (
            self.processed_root / "test" / "images" / "buu_ap" / "buu_test.jpg"
        )
        missing_path.unlink()

        with self.assertRaises(FileNotFoundError):
            build_dataset(processed_root=self.processed_root, output_root=self.output_root)

        self.assertFalse(self.output_root.exists())
        self.assertFalse(self.output_root.with_name("raw.building").exists())


if __name__ == "__main__":
    unittest.main()
