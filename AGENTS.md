# Project Context for Coding Agents

## Mission and medical scope

Spine Opportunistic Screening is a research system for extracting vertebral landmarks and screening-oriented measurements from routine AP/PA chest and abdominal radiographs.

This is not a diagnostic system. Never claim that an image proves scoliosis, osteoporosis, vertebral compression fracture, or another disease. Use language such as suspicious morphology, possible abnormality signal, screening support, or requires clinical review.

The code currently measures vertebral geometry and spinal curvature. It does not implement a validated suspicious-morphology classifier. Measurement-only features must not be presented as disease predictions.

## Active pipeline

    radiograph
      -> deterministic resize/pad preprocessing
      -> CenterNet center and four-corner prediction
      -> spine-chain duplicate removal and candidate selection
      -> Cobb-angle and morphology measurements
      -> overlays and optional CSV/JSON reports

CenterNet is the only active inference model. Supported backbones include HRNet-W18/W30/W32/W40/W44/W48/W64 and ResNet34. The current cross-domain checkpoint is src/weights/hrnet_nih/best_center_f1.pt, trained on the combined COCO+NIH dataset.

Spine-chain selection is enabled by default during prediction. Always distinguish raw detector metrics from spine-chain metrics.

## Retired corner-refiner experiment

The HRNet-W18 corner-refiner/two-stage approach is retired. On 2,282 matched Mendeley/MICCAI test instances it reduced mean NME from 0.073432 to 0.065879, a 10.285 percent relative improvement, below the 20 percent acceptance gate. The additional checkpoint, ROI pipeline, QC/consensus experiments, and operational complexity were not justified.

Its source, tests, checkpoints, notebooks, active-file snapshots, results, and restoration instructions are self-contained under backup/refiner_approach. Nothing in that directory is imported by active code. Do not reintroduce or modify the archived approach unless the user explicitly asks to restore it.

For backward-compatible output schemas, morphology rows may still contain landmark_source and fallback_reason. In the active pipeline these indicate CenterNet provenance only; they do not imply that a refiner ran.

## Non-negotiable data and geometry contracts

- The canonical corner order is TL, TR, BL, BR.
- Each point is stored as x, y in image-pixel coordinates.
- Never silently swap bottom-left and bottom-right. Source annotations using TL, TR, BR, BL must be normalized explicitly.
- A CenterNet center is derived from the four ordered corners. Keep center, offset, and eight-value corner regression targets consistent.
- Preserve the mapping between original radiographs, resized/padded inputs, heatmaps, and original-image outputs.
- Transform floating-point coordinates without premature rounding or clamping.
- Horizontal flip is disabled. If enabled later, corner semantics must be reordered deliberately.
- Do not change COCO fields, category/keypoint order, output stride, or measurement definitions without migration notes and tests.

## Datasets and split policy

dataset/processed/coco is the authoritative base dataset:

- BUU AP: 400 images and 2,000 annotations.
- Mendeley PA: 872 images and 13,128 annotations.
- MICCAI 2019: 481 images and 8,176 annotations.
- Split totals: 1,403 train, 175 validation, and 175 test images.

dataset/processed/coco_nih extends it with NIH ChestX-ray14:

- NIH: 147 train, 19 validation, and 18 test images.
- Combined totals: 1,550 train, 194 validation, and 193 test images.
- NIH assignment is deterministic and patient-grouped with seed 20260627.
- No NIH patient may appear in multiple splits.
- Two zero-annotation NIH images are deliberately excluded.

One COCO image is one radiograph. Keep all vertebrae from a radiograph in the same split. Never create annotation-level leakage.

Use split_summary.json as the provenance/count authority. Prefer reproducible builders such as src/data/build_coco_nih.py over manual processed-JSON edits.

## Evaluation conventions

- Use a COCO dataset root, not an image directory, for ground-truth evaluation.
- Use --source-dataset "NIH ChestX-ray14" for NIH-only evaluation. Filtering is case-insensitive and occurs before --limit.
- Checkpoint metadata supplies architecture and preprocessing defaults unless explicitly overridden.
- Fair comparisons require the same split, source filter, peak threshold, top-k, input size, output stride, and chain policy.
- Report raw and spine-chain metrics separately.
- Center and corner MAE are conditional on matched detections; interpret them with precision, recall, and F1.
- Tune architecture, thresholds, and chain policy on validation. Evaluate test only after choices are frozen.
- The NIH test subset has 18 images, so avoid broad domain claims from it alone.

CenterNet evaluation writes metrics.json, metrics.csv, and per_image_metrics.csv.

## Prediction and output behavior

- src/predict_centernet.py accepts a file or directory and recursively discovers supported radiographs.
- Spine-chain and Cobb overlays are enabled by default.
- The default overlay background is a heatmap blend; --overlay-background original uses the radiograph.
- Tables are disabled by default. --save-tables writes prediction, chain, Cobb, and morphology outputs.
- morphology_features.csv/json contain measurements, ratios, endplate angles, neighbor-relative values, and CenterNet provenance. They are not diagnoses.

Visual debugging outputs are part of correctness. Preserve clear overlays for centers, ordered corners, selected vertebrae, and Cobb lines.

## Training and Colab workflow

The main repository is canonical. The adjacent SOS Colab folder is a deployable private-Drive package, not the source of truth.

- Active configuration: configs/config.yaml.
- Active notebook: notebooks/colab/train_centernet_hrnet_w18_nih.ipynb.

CenterNet training prints live batch and epoch progress, supports AMP, resume checkpoints, early stopping, previews, and atomic Drive backup. Use a distinct experiment name after changing data or architecture so an incompatible last.pt cannot be resumed.

Before a full run, validate dataset tensors and a model forward pass. When active code used by Colab changes, deliberately synchronize its source, config, notebook, and annotations into SOS Colab, then verify split counts and file resolution.

## Code ownership map

- src/data/augmentation.py: shared resize/pad and radiograph augmentation.
- src/data/centernet_dataset.py and centernet_targets.py: CenterNet COCO samples and targets.
- src/data/build_coco_nih.py: reproducible COCO+NIH merge and patient grouping.
- src/models/centernet.py: supported CenterNet backbones and heads.
- src/training/centernet_loss.py: CenterNet objectives.
- src/evaluation: active decoding, metrics, and Cobb calculations.
- src/postprocessing/spine_chain.py: duplicate handling and chain optimization.
- src/analysis/vertebral_morphology.py: measurement-only morphology features.
- src/predict_centernet.py: one-stage prediction, overlays, and tables.
- backup/refiner_approach: frozen retired experiment; not active source.

## Implementation rules

- Preserve user changes in a dirty worktree and avoid unrelated rewrites.
- Keep one-stage CenterNet prediction and evaluation behavior working.
- Do not import from backup or silently revive refiner flags, dependencies, or checkpoints.
- Prefer reproducible scripts over manual dataset edits.
- Keep paths configurable and checkpoint metadata JSON-compatible.
- Do not add large datasets, checkpoints, generated outputs, or virtual environments to Git.
- Add/update tests for ordering, filtering, checkpoint compatibility, overlays, and metrics when those areas change.
- Use Python unittest for the existing suite.

## Required validation

Run the relevant subset during development and the full suite before handoff when practical:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

For model/data changes, also run:

    python -m src.workflows.check_centernet_dataset

Full training belongs on Colab. Local validation should use deterministic data checks, small forward/prediction smoke passes, checkpoint loading, and focused evaluation.

