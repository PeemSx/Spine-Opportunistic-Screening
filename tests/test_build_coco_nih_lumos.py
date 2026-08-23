from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.data.build_coco_nih import CANONICAL_CATEGORY, category_corner_indices
from src.data.build_coco_nih_lumos import (
    IncrementRecord,
    assign_source_records,
    image_identity,
    normalize_annotation,
)


class BuildCocoNihLumosTests(unittest.TestCase):
    @staticmethod
    def _annotation(annotation_id: int = 1) -> dict[str, object]:
        return {
            "id": annotation_id,
            "image_id": annotation_id,
            "category_id": 2,
            "num_keypoints": 4,
            "keypoints": [1, 1, 2, 9, 1, 2, 1, 9, 2, 9, 9, 2],
        }

    def test_two_source_label_schemas_collapse_to_one_vertebra_category(self) -> None:
        categories = [
            {"id": 1, "keypoints": ["tl", "tr", "br", "bl"]},
            {
                "id": 2,
                "keypoints": ["top_left", "top_right", "bottom_left", "bottom_right"],
            },
        ]
        orders = category_corner_indices(categories)
        source_annotation = {
            "id": 7,
            "image_id": 3,
            "category_id": 1,
            # Category 1 is TL/TR/BR/BL, and these semantic labels were
            # horizontally reversed in the source export.
            "keypoints": [9, 1, 2, 1, 1, 2, 9, 9, 2, 1, 9, 2],
        }
        result = normalize_annotation(
            source_annotation,
            order_by_category=orders,
            annotation_id=70,
            image_id=30,
            image_width=20,
            image_height=20,
        )

        self.assertIsNotNone(result.annotation)
        self.assertTrue(result.corner_order_repaired)
        assert result.annotation is not None
        self.assertEqual(result.annotation["category_id"], 1)
        self.assertEqual(result.annotation["source_category_id"], 1)
        points = np.asarray(result.annotation["keypoints"], dtype=float).reshape(4, 3)
        np.testing.assert_allclose(
            points[:, :2],
            np.asarray([[1, 1], [9, 1], [1, 9], [9, 9]], dtype=float),
        )
        self.assertEqual(CANONICAL_CATEGORY["name"], "vertebra")
        self.assertEqual(
            CANONICAL_CATEGORY["keypoints"],
            ["top_left", "top_right", "bottom_left", "bottom_right"],
        )

    def test_tiny_annotation_is_excluded(self) -> None:
        categories = [
            {
                "id": 2,
                "keypoints": ["top_left", "top_right", "bottom_left", "bottom_right"],
            }
        ]
        result = normalize_annotation(
            {
                "id": 9,
                "image_id": 1,
                "category_id": 2,
                "keypoints": [10, 10, 2, 11, 10, 2, 10, 11, 2, 11, 11, 2],
            },
            order_by_category=category_corner_indices(categories),
            annotation_id=9,
            image_id=1,
            image_width=1000,
            image_height=1000,
        )

        self.assertIsNone(result.annotation)
        self.assertEqual(result.exclusion_reason, "tiny_annotation_below_1e-5_image_area")

    def test_filename_identity_normalization(self) -> None:
        self.assertEqual(
            image_identity("NIH_00008295_006_png.rf.hash.png"),
            ("nih_chestxray14", "00008295", "006"),
        )
        self.assertEqual(
            image_identity("lumos_AP_106_png.rf.hash.png"),
            ("lumos_ap", "106"),
        )

    def test_grouped_assignment_preserves_existing_patient_split(self) -> None:
        records = []
        group_ids = ["existing_train", "existing_test"] + [f"new_{index}" for index in range(8)]
        for index, group_id in enumerate(group_ids, start=1):
            annotation = self._annotation(index)
            records.append(
                IncrementRecord(
                    identity=("nih_chestxray14", group_id, f"{index:03d}"),
                    source_key="nih_chestxray14",
                    source_name="NIH ChestX-ray14",
                    group_id=group_id,
                    clean_name=f"NIH_{group_id}_{index:03d}.png",
                    source_path=Path(f"unused_{index}.png"),
                    source_image={"id": index, "width": 20, "height": 20},
                    source_annotations=(annotation,),
                )
            )
        orders = category_corner_indices(
            [
                {
                    "id": 2,
                    "keypoints": [
                        "top_left",
                        "top_right",
                        "bottom_left",
                        "bottom_right",
                    ],
                }
            ]
        )
        kwargs = {
            "existing_image_counts": {"train": 8, "val": 1, "test": 1},
            "existing_annotation_counts": {"train": 8, "val": 1, "test": 1},
            "existing_group_splits": {
                "existing_train": "train",
                "existing_test": "test",
            },
            "order_by_category": orders,
            "seed": 20260810,
            "search_trials": 100,
        }
        first, targets, _ = assign_source_records(records, **kwargs)
        second, _, _ = assign_source_records(records, **kwargs)

        assignments = {
            record.group_id: split
            for split, split_records in first.items()
            for record in split_records
        }
        self.assertEqual(assignments["existing_train"], "train")
        self.assertEqual(assignments["existing_test"], "test")
        self.assertEqual(
            {split: len(first[split]) + kwargs["existing_image_counts"][split] for split in first},
            targets,
        )
        self.assertEqual(
            {split: [record.identity for record in first[split]] for split in first},
            {split: [record.identity for record in second[split]] for split in second},
        )


if __name__ == "__main__":
    unittest.main()
