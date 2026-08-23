# Spine Opportunistic Screening

Deep learning-based vertebral landmark detection for morphology and spinal-curvature assessment in routine AP/PA radiographs.

This is a research and screening-support project. It does not diagnose scoliosis, osteoporosis, vertebral compression fracture, or any other condition. Measurements and overlays require clinical review.

## Current system

The active pipeline is a single CenterNet-style model. It predicts a center and four ordered corners for each visible vertebra. Spine-chain postprocessing removes duplicate and anatomically inconsistent candidates before Cobb-angle and morphology measurements are calculated.

    AP/PA radiograph
      -> resize/pad preprocessing
      -> CenterNet center and corner prediction
      -> spine-chain candidate selection
      -> Cobb-angle and measurement-only morphology features
      -> overlays and optional CSV/JSON tables

Suspicious-morphology classification is not implemented yet. Current morphology output contains geometric measurements and neighbor-relative features, not a diagnosis.

## Installation

From the repository root on Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirement.txt
~~~

The active configuration file is configs/config.yaml.

## Checkpoints

| Path | Purpose |
|---|---|
| src/weights/hrnet_nih/best_center_f1.pt | Current HRNet-W18 model trained with the combined COCO+NIH data |
| src/weights/hrnet/best_center_f1.pt | Earlier HRNet-W18 CenterNet checkpoint |
| src/weights/hrnet_32/best_center_f1.pt | HRNet-W32 CenterNet experiment |

The archived refiner checkpoint is not part of active prediction. See backup/refiner_approach/README.md if the experiment must be restored.

## Datasets

dataset/processed/coco contains BUU, Mendeley, and MICCAI 2019 data: 1,403 train, 175 validation, and 175 test radiographs.

dataset/processed/coco_nih adds NIH ChestX-ray14 annotations: 1,550 train, 194 validation, and 193 test radiographs. The NIH contribution is 147 train, 19 validation, and 18 test images, grouped by patient to prevent leakage. The canonical corner order is TL, TR, BL, BR.

dataset/processed/coco_nih_lumos extends that checkpoint with the reviewed August 10, 2026 Roboflow export: 1,788 train, 223 validation, and 223 test radiographs. It contains 257/32/32 NIH images and 128/16/16 Lumos AP images. Existing assignments are preserved, additional NIH studies are grouped with their patient, and the output uses one canonical `vertebra` category with TL, TR, BL, BR keypoints.

Rebuild the combined dataset with:

~~~powershell
python -m src.data.build_coco_nih `
  --base-root dataset/processed/coco `
  --nih-root "dataset/raw/NIH/version 0.5" `
  --output-root dataset/processed/coco_nih
~~~

Extend that checkpoint with the reviewed Roboflow export using:

~~~powershell
python -m src.data.build_coco_nih_lumos `
  --base-root dataset/processed/coco_nih `
  --increment-root "dataset/raw/Roboflow/10 AUG 2026" `
  --output-root dataset/processed/coco_nih_lumos `
  --seed 20260810
~~~

The Colab training notebook for this dataset is:

    notebooks/colab/train_centernet_hrnet_w18_nih_lumos.ipynb

It uses the distinct experiment `centernet_hrnet_w18_coco_nih_lumos`. Checkpoints include a dataset fingerprint and cannot be resumed against different annotations without an explicit unsafe override.

### Landmark validation during training

Every validation epoch now runs the same normalized landmark protocol used by
the standalone scorecard, for both raw detections and the selected spine chain.
The run directory receives:

- `validation_landmark_log.csv`: epoch-level center F1/recall, corner NME,
  PCK@0.10, usable-vertebra recall, false positives, count MAE, and source
  robustness summaries.
- `validation_metrics_latest.json`: complete overall, per-source, vertebral-scale,
  and source-by-scale summaries.
- `validation_instances_latest.csv`: matched/missed instance rows with TL, TR,
  BL, and BR signed residuals. Residuals are prediction minus ground truth, so
  positive x is right and positive y is down.
- `best_corner_nme.pt` and `best_usable_recall.pt`, plus matching frozen JSON/CSV
  artifacts. These supplement `best_loss.pt` and `best_center_f1.pt`.

Scale groups use the GT vertebral diagonal divided by the larger original-image
dimension: small below 1/16, medium from 1/16 to below 3/32, and large from
3/32 upward. The two landmark checkpoints are only saved when spine-chain
detection guardrails pass (by default source-macro F1@0.20D at least 0.85,
worst-source recall at least 0.75, no more than 1 false positive per image, and
count MAE no more than 1.5). The thresholds are configurable through the
corresponding `--landmark-checkpoint-*` training arguments. Research evaluation
automatically reuses the validation spine-chain settings embedded in these
training checkpoints unless explicit chain overrides are supplied.

## Prediction

Recommended prediction with the NIH-adapted HRNet-W18 checkpoint:

~~~powershell
python -m src.predict_centernet `
  --checkpoint src/weights/hrnet_nih/best_center_f1.pt `
  --source path/to/radiographs `
  --output-dir outputs/predictions/hrnet18_nih `
  --overlay-background original `
  --show-corners box `
  --save-tables
~~~

Spine-chain selection and Cobb overlays are enabled by default. Checkpoint metadata supplies the backbone, input size, output stride, threshold, and top-k settings unless explicitly overridden.

| Option | Effect |
|---|---|
| --raw | Bypass spine-chain selection and display raw candidates |
| --overlay-background original | Draw on the original radiograph instead of a heatmap blend |
| --show-corners no | Hide corner marks |
| --show-corners point | Draw corner points only |
| --show-corners cross | Draw diagonal corner crosses |
| --show-corners box | Draw vertebral quadrilaterals |
| --confidence-labels low | Label only centers below the low-confidence threshold |
| --confidence-labels none | Hide center confidence labels |
| --no-show-cobb | Hide Cobb lines and labels |
| --save-tables | Save prediction, chain, Cobb, and morphology CSV/JSON outputs |
| --save-morphology | Save morphology tables even when other tables are disabled |

## Evaluation

Evaluate the full combined test split with the deployed compact artifact. Its
frozen peak threshold is `0.10`; deployed mode rejects metric-affecting
overrides:

~~~powershell
python -m src.evaluate_centernet `
  --evaluation-profile deployed `
  --dataset-root dataset/processed/coco_nih `
  --split test `
  --output-dir outputs/evaluation_v2/deployed_full_test `
  --batch-size 2 `
  --num-workers 0
~~~

Run a source-specific research evaluation with the standard operating point:

~~~powershell
python -m src.evaluate_centernet `
  --evaluation-profile research `
  --checkpoint src/weights/hrnet_nih/best_center_f1.pt `
  --dataset-root dataset/processed/coco_nih `
  --split test `
  --source-dataset "NIH ChestX-ray14" `
  --output-dir outputs/evaluation_v2/research_nih_test `
  --batch-size 2 `
  --num-workers 0 `
  --peak-thresh 0.10 `
  --topk 50
~~~

Evaluation writes `metrics.json`, `metrics.csv`, `per_image_metrics.csv`, and
`per_instance_metrics.csv` with separate raw and spine-chain results. Add
`--strict-gates` for a nonzero exit when any technical acceptance gate fails.
The source filter is case-insensitive and is applied before `--limit`.

The primary detection gate is normalized center distance `0.20D`, where `D` is
the matched GT vertebra's bounding-box diagonal. Landmark NME and PCK use the
same scale. Usable-vertebra recall requires a center match, a finite,
non-self-intersecting in-image quadrilateral, and NME no greater than `0.10`.
For fair research comparisons, keep the dataset, split, source filter, peak
threshold, top-k, input size, and chain settings fixed.

## Repository map

- src/models/centernet.py: CenterNet backbones and prediction heads.
- src/data/centernet_dataset.py and src/data/centernet_targets.py: COCO loading and CenterNet targets.
- src/postprocessing/spine_chain.py: duplicate removal and spine-chain selection.
- src/analysis/vertebral_morphology.py: measurement-only morphology features.
- src/evaluation: CenterNet decoding, metrics, and Cobb calculations.
- src/predict_centernet.py: end-to-end one-stage prediction and output generation.
- src/evaluate_centernet.py: full/source-filtered CenterNet evaluation.
- backup/refiner_approach: retired two-stage refiner experiment and restoration notes.
- backup/yolo_pose_experiment: retired COCO-to-YOLO-pose utilities and prototype.
