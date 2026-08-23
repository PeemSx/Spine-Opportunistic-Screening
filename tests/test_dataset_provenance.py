from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.dataset_provenance import (
    build_dataset_provenance,
    validate_resume_dataset,
)


class DatasetProvenanceTests(unittest.TestCase):
    def _dataset_root(self, temporary_root: str) -> Path:
        root = Path(temporary_root)
        (root / "split_summary.json").write_text(
            json.dumps({"version": "test_dataset"}), encoding="utf-8"
        )
        for split in ("train", "val", "test"):
            split_dir = root / split
            split_dir.mkdir()
            (split_dir / "_annotations.keypoints.coco.json").write_text(
                json.dumps(
                    {
                        "categories": [{"id": 1, "name": "vertebra"}],
                        "images": [{"id": 1, "file_name": f"{split}.png"}],
                        "annotations": [{"id": 1, "image_id": 1}],
                    }
                ),
                encoding="utf-8",
            )
        return root

    def test_fingerprint_is_deterministic_and_records_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            root = self._dataset_root(temporary_root)
            first = build_dataset_provenance(root)
            second = build_dataset_provenance(root)

        self.assertEqual(first, second)
        self.assertEqual(first["dataset_version"], "test_dataset")
        self.assertEqual(first["split_counts"]["train"], {"images": 1, "annotations": 1})
        self.assertEqual(len(first["fingerprint"]), 64)

    def test_resume_rejects_changed_or_legacy_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_root:
            root = self._dataset_root(temporary_root)
            original = build_dataset_provenance(root)
            train_path = root / "train" / "_annotations.keypoints.coco.json"
            payload = json.loads(train_path.read_text(encoding="utf-8"))
            payload["images"].append({"id": 2, "file_name": "changed.png"})
            train_path.write_text(json.dumps(payload), encoding="utf-8")
            changed = build_dataset_provenance(root)

        with self.assertRaisesRegex(RuntimeError, "dataset mismatch"):
            validate_resume_dataset({"dataset_provenance": original}, changed)
        with self.assertRaisesRegex(RuntimeError, "no dataset provenance"):
            validate_resume_dataset({}, changed)
        validate_resume_dataset({}, changed, allow_unsafe_resume=True)


if __name__ == "__main__":
    unittest.main()
