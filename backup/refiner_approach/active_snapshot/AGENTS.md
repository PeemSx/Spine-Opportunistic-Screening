# Project Context for Coding Agents

## Mission and medical scope

Spine Opportunistic Screening is a research system for extracting vertebral landmarks and screening-oriented measurements from routine AP/PA chest and abdominal radiographs. Its purpose is to help investigate findings that may otherwise be overlooked.

This is not a diagnostic system. Never claim that an image proves scoliosis, osteoporosis, vertebral compression fracture, or another disease. Use language such as suspicious morphology, possible abnormality signal, screening support, or requires clinical review.

The current code measures vertebral geometry and spinal curvature. It does not yet implement a validated suspicious-morphology classifier. Do not present measurement-only morphology features as a disease prediction.

## Active pipeline

    radiograph
      -> deterministic resize/pad preprocessing
      -> CenterNet center and four-corner prediction
      -> optional single-ROI HRNet-W18 corner refinement
      -> spine-chain duplicate removal and candidate selection
      -> Cobb-angle and morphology measurements
      -> overlays and optional CSV/JSON reports

CenterNet is the primary model. Supported backbones are HRNet-W18/W30/W32/W40/W44/W48/W64 and ResNet34. The current cross-domain checkpoint is src/weights/hrnet_nih/best_center_f1.pt, trained on the COCO+NIH dataset.

Spine-chain selection is enabled by default during prediction. Always distinguish raw detector metrics from spine-chain metrics.

The corner refiner is optional and experimental. Its current matched validation improvement is about 10.3 percent NME, below the 20 percent project acceptance target. The active two-stage path always replaces CenterNet corners with refiner corners. Refiner QC rejection, multi-crop consensus, and CenterNet fallback are inactive and archived under backup/refiner_qc_consensus.

## Non-negotiable data and geometry contracts

- The canonical corner order is TL, TR, BL, BR.
- Each point is stored as x, y in image-pixel coordinates.
- Never silently swap bottom-left and bottom-right. Some source NIH annotations used TL, TR, BR, BL and must be normalized explicitly.
- A CenterNet center is derived from the four ordered corners. Keep center, offset, and eight-value corner regression targets consistent.
- Preserve mappings between original images, resized/padded model inputs, ROI crops, heatmaps, and original-image outputs.
- Transform floating-point coordinates without premature rounding or clamping.
- Refiner ROIs use square isotropic scaling and black padding outside image boundaries.
- Horizontal flip is disabled in the current configuration. If enabled later, corner semantics must be reordered deliberately.
- Do not change COCO annotation fields, category/keypoint order, output stride, or measurement definitions without migration notes and tests.

## Datasets and split policy

dataset/processed/coco is the authoritative base dataset:

- BUU AP: 400 images and 2,000 annotations.
- Mendeley PA: 872 images and 13,128 annotations.
- MICCAI 2019: 481 images and 8,176 annotations.
- Split totals: 1,403 train, 175 validation, 175 test images.

dataset/processed/coco_nih extends the base data with NIH ChestX-ray14:

- 147 NIH train images, 19 validation images, and 18 test images.
- Combined totals: 1,550 train, 194 validation, 193 test images.
- NIH split assignment is deterministic and patient-grouped with seed 20260627.
- No NIH patient may appear in multiple splits.
- Two zero-annotation NIH images are deliberately excluded.

One COCO image is one radiograph. Keep all vertebra annotations from a radiograph in the same split. Never create annotation-level leakage.

Use split_summary.json as the provenance and count authority. Do not edit processed annotation JSON manually when the change can be reproduced through src/data/build_coco_nih.py or another preparation script.

## Evaluation conventions

- Evaluate from a COCO dataset root, not directly from an image subdirectory, when ground-truth metrics are required.
- Use --source-dataset "NIH ChestX-ray14" for NIH-only CenterNet evaluation. The filter is case-insensitive and runs before --limit.
- Checkpoint metadata supplies architecture and preprocessing defaults unless the CLI overrides them.
- Fair model comparisons require identical dataset split, source filter, peak threshold, top-k, input size, output stride, and spine-chain settings.
- Report raw and spine-chain metrics separately.
- Center and corner MAE are conditional on matched detections; a model with higher recall may include harder matches and show slightly higher conditional MAE.
- Tune architecture, thresholds, and chain policy on validation. Run test evaluation only after choices are frozen.
- The NIH test set currently has only 18 images, so avoid overgeneralizing domain-performance conclusions.

CenterNet evaluation writes metrics.json, metrics.csv, and per_image_metrics.csv. Corner-refiner evaluation maps predictions back to original coordinates before computing NME, PCK, pixel errors, angle errors, and morphology-related errors.

## Prediction and output behavior

- src/predict_centernet.py accepts files or directories and recursively discovers supported radiographs.
- Spine-chain and Cobb overlays default to enabled.
- The default overlay background is the heatmap blend; --overlay-background original draws directly on the radiograph.
- Tables default to disabled. --save-tables writes prediction, chain, Cobb, and morphology outputs.
- --show-refiner-boxes draws final refined quadrilaterals only for candidates selected downstream.
- With a refiner checkpoint, refiner_diagnostics.json is diagnostic only and does not reject or replace candidates conditionally.
- morphology_features.csv and morphology_features.json contain measurements, ratios, endplate angles, neighbor-relative values, and landmark provenance. They are not diagnoses.

Visual debugging outputs are part of correctness. Preserve clear overlays for centers, ordered corners, selected vertebrae, Cobb lines, and transformed ROIs.

## Training and Colab workflow

The main repository is canonical. The adjacent SOS Colab folder is a deployable private-Drive package, not the source of truth.

- Main configuration: configs/config.yaml.
- Current CenterNet notebook: notebooks/colab/train_centernet_hrnet_w18_nih.ipynb.
- Current refiner notebook: notebooks/colab/train_corner_refiner.ipynb.

CenterNet training prints live batch and epoch progress, supports AMP, resume checkpoints, early stopping, previews, and atomic Drive backup. Use a distinct experiment name when changing backbone or data so an incompatible last.pt cannot be resumed.

Before a full run, validate dataset tensors and a model forward pass. The refiner additionally requires ROI/target contract checks and the fixed-crop overfit gate.

No repository sync script is currently authoritative. When code used by Colab changes, deliberately synchronize the corresponding source, configuration, notebook, and annotation files into the adjacent SOS Colab package, then verify split counts and file resolution. Do not duplicate unchanged base images.

## Code ownership map

- src/data/augmentation.py: shared resize/pad and radiograph augmentation.
- src/data/centernet_dataset.py and centernet_targets.py: CenterNet COCO samples and targets.
- src/data/build_coco_nih.py: reproducible COCO+NIH merge and patient grouping.
- src/data/roi_geometry.py: shared square ROI affine geometry.
- src/data/corner_refiner_dataset.py and corner_refiner_targets.py: one-annotation-per-sample refiner data.
- src/models/centernet.py: supported CenterNet backbones and heads.
- src/models/corner_refiner.py: HRNet-W18 four-heatmap refiner.
- src/training: CenterNet and corner-refiner losses.
- src/evaluation: decoders, metrics, and Cobb calculations.
- src/postprocessing/spine_chain.py: duplicate handling and chain optimization.
- src/analysis/vertebral_morphology.py: measurement-only morphology features.
- src/inference/corner_refiner.py: refiner checkpoint loading and ROI inference.
- src/predict_centernet.py: end-to-end prediction, overlays, and tables.

## Implementation rules

- Preserve user changes in a dirty worktree and avoid unrelated rewrites.
- Keep existing one-stage behavior working when changing optional refiner code.
- Prefer reproducible scripts over manual dataset edits.
- Keep paths configurable and serialize checkpoint metadata with plain JSON-compatible values and string paths.
- Do not add large datasets, model checkpoints, generated outputs, or virtual environments to Git.
- Keep small model cards, manifests, configuration, tests, and reproducibility scripts trackable.
- Add or update tests for ordering, geometry, filtering, checkpoint compatibility, overlays, and metric logic when those areas change.
- Use Python unittest for the existing test suite.

## Required validation

Run the relevant subset during development and the complete suite before handoff when practical:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

For model/data changes, also run the appropriate workflow checker:

    python -m src.workflows.check_centernet_dataset
    python -m src.workflows.check_corner_refiner_dataset

Full training is intended for Colab. Local validation should be limited to deterministic data checks, small CPU/GPU smoke passes, checkpoint load/forward checks, and focused evaluation.
