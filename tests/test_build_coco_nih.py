from __future__ import annotations

import unittest

import numpy as np

from src.data.build_coco_nih import (
    NihRecord,
    canonical_source_key,
    category_corner_indices,
    largest_remainder_targets,
    normalize_keypoints,
    split_nih_by_patient,
)


class BuildCocoNihTests(unittest.TestCase):
    def test_category_specific_corner_order_is_normalized(self) -> None:
        categories = [
            {"id": 1, "keypoints": ["tl", "tr", "br", "bl"]},
            {
                "id": 2,
                "keypoints": ["top_left", "top_right", "bottom_left", "bottom_right"],
            },
        ]
        orders = category_corner_indices(categories)
        category_one = {
            "id": 10,
            "category_id": 1,
            "keypoints": [1, 2, 2, 9, 2, 2, 8, 8, 2, 2, 8, 2],
        }
        category_two = {
            "id": 11,
            "category_id": 2,
            "keypoints": [1, 2, 2, 9, 2, 2, 2, 8, 2, 8, 8, 2],
        }
        expected = np.asarray([[1, 2], [9, 2], [2, 8], [8, 8]], dtype=float)
        np.testing.assert_allclose(normalize_keypoints(category_one, orders)[:, :2], expected)
        np.testing.assert_allclose(normalize_keypoints(category_two, orders)[:, :2], expected)

    def test_patient_grouped_split_is_deterministic_and_disjoint(self) -> None:
        records = []
        annotation = ({"id": 1, "category_id": 2, "keypoints": [0, 0, 2] * 4},)
        patient_sizes = {"00000001": 3, "00000002": 2}
        patient_sizes.update({f"{index:08d}": 1 for index in range(3, 23)})
        for patient_id, count in patient_sizes.items():
            for study in range(count):
                records.append(
                    NihRecord(
                        patient_id=patient_id,
                        study_id=f"{study:03d}",
                        clean_name=f"NIH_{patient_id}_{study:03d}.png",
                        source_path=None,  # type: ignore[arg-type]
                        source_image={},
                        source_annotations=annotation,
                    )
                )
        first, targets, _ = split_nih_by_patient(records, seed=20260627, search_trials=200)
        second, _, _ = split_nih_by_patient(records, seed=20260627, search_trials=200)
        self.assertEqual(
            {split: [record.clean_name for record in first[split]] for split in first},
            {split: [record.clean_name for record in second[split]] for split in second},
        )
        patient_sets = {
            split: {record.patient_id for record in split_records}
            for split, split_records in first.items()
        }
        self.assertFalse(patient_sets["train"] & patient_sets["val"])
        self.assertFalse(patient_sets["train"] & patient_sets["test"])
        self.assertFalse(patient_sets["val"] & patient_sets["test"])
        self.assertEqual({split: len(first[split]) for split in first}, targets)

    def test_largest_remainder_targets_sum_to_total(self) -> None:
        targets = largest_remainder_targets(184, {"train": 0.8, "val": 0.1, "test": 0.1})
        self.assertEqual(targets, {"train": 147, "val": 19, "test": 18})
        self.assertEqual(sum(targets.values()), 184)

    def test_nih_source_name_is_canonical(self) -> None:
        self.assertEqual(canonical_source_key("NIH ChestX-ray14"), "nih_chestxray14")


if __name__ == "__main__":
    unittest.main()
