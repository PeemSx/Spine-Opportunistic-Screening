from __future__ import annotations

import copy
import unittest

from src.data.build_masked_morphology_dataset import (
    build_masked_samples,
    build_vertebra_rows,
    numeric_feature_columns,
)


def rectangle_annotation(annotation_id: int, image_id: int, rank: int) -> dict:
    center_y = 100.0 * rank
    half_width = 20.0 + rank
    half_height = 10.0 + rank
    points = [
        (100.0 - half_width, center_y - half_height),
        (100.0 + half_width, center_y - half_height),
        (100.0 - half_width, center_y + half_height),
        (100.0 + half_width, center_y + half_height),
    ]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "vertebra_index": rank,
        "keypoints": [value for x, y in points for value in (x, y, 2)],
    }


def synthetic_coco() -> dict:
    return {
        "images": [
            {
                "id": 7,
                "file_name": "images/nih_chestxray14/NIH_00000001_001.png",
                "width": 400,
                "height": 800,
                "source_dataset": "NIH ChestX-ray14",
                "nih_patient_id": "00000001",
            }
        ],
        "annotations": [rectangle_annotation(rank, 7, rank) for rank in range(1, 6)],
    }


class MaskedMorphologyDatasetTests(unittest.TestCase):
    def test_constructs_one_five_vertebra_masked_sample(self) -> None:
        rows, audit = build_vertebra_rows(synthetic_coco(), split="train")
        samples = build_masked_samples(rows)

        self.assertEqual(audit["geometry_valid"], 5)
        self.assertEqual(audit["source_rank_mismatches"], 0)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["target_chain_rank"], 3)
        self.assertEqual(sample["group_id"], "nih_chestx_ray14:patient:00000001")
        for removed in (
            "target_anatomical_level",
            "anatomical_level_status",
            "normality_review_status",
            "visibility_review_status",
            "normality_review_required",
            "reconstruction_training_eligible",
            "expected_normal_training_eligible",
        ):
            self.assertNotIn(removed, sample)
        self.assertAlmostEqual(sample["sample_weight"], 1.0)
        self.assertEqual(len(numeric_feature_columns()), 28)
        self.assertNotIn("x_m1_width_height_ratio", sample)
        self.assertNotIn("x_m1_endplate_nonparallel_deg", sample)

    def test_target_landmarks_do_not_change_input_features_or_baselines(self) -> None:
        original = synthetic_coco()
        changed = copy.deepcopy(original)
        target = changed["annotations"][2]
        target["keypoints"] = [
            value
            for x, y in ((40, 260), (160, 260), (40, 340), (160, 340))
            for value in (x, y, 2)
        ]

        original_rows, _ = build_vertebra_rows(original, split="train")
        changed_rows, _ = build_vertebra_rows(changed, split="train")
        original_sample = build_masked_samples(original_rows)[0]
        changed_sample = build_masked_samples(changed_rows)[0]

        protected = numeric_feature_columns() + [
            "reference_height_px",
            "reference_width_px",
            "baseline_center_x_px",
            "baseline_center_y_px",
            "baseline_orientation_deg",
            "baseline_left_height_norm",
            "baseline_right_height_norm",
            "baseline_superior_width_norm",
            "baseline_inferior_width_norm",
        ]
        for name in protected:
            self.assertAlmostEqual(original_sample[name], changed_sample[name], places=9)
        self.assertNotEqual(
            original_sample["y_left_height_norm"],
            changed_sample["y_left_height_norm"],
        )


if __name__ == "__main__":
    unittest.main()
