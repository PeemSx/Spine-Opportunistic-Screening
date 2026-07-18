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

Rebuild the combined dataset with:

~~~powershell
python -m src.data.build_coco_nih `
  --base-root dataset/processed/coco `
  --nih-root "dataset/raw/NIH/version 0.5" `
  --output-root dataset/processed/coco_nih
~~~

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

Evaluate the full combined test split:

~~~powershell
python -m src.evaluate_centernet `
  --checkpoint src/weights/hrnet_nih/best_center_f1.pt `
  --dataset-root dataset/processed/coco_nih `
  --split test `
  --output-dir outputs/evaluation/hrnet18_nih_full_test `
  --batch-size 2 `
  --num-workers 0 `
  --peak-thresh 0.05 `
  --topk 50
~~~

Evaluate only the NIH source:

~~~powershell
python -m src.evaluate_centernet `
  --checkpoint src/weights/hrnet_nih/best_center_f1.pt `
  --dataset-root dataset/processed/coco_nih `
  --split test `
  --source-dataset "NIH ChestX-ray14" `
  --output-dir outputs/evaluation/hrnet18_nih_test `
  --batch-size 2 `
  --num-workers 0 `
  --peak-thresh 0.05 `
  --topk 50
~~~

Evaluation writes metrics.json, metrics.csv, and per_image_metrics.csv with separate raw and spine-chain rows. The source filter is case-insensitive and is applied before --limit.

For a fair model comparison, use the same dataset, split, source filter, peak threshold, top-k, input size, and chain settings. Center/corner errors are conditional on matched detections, so interpret them alongside precision, recall, and F1.

## Repository map

- src/models/centernet.py: CenterNet backbones and prediction heads.
- src/data/centernet_dataset.py and src/data/centernet_targets.py: COCO loading and CenterNet targets.
- src/postprocessing/spine_chain.py: duplicate removal and spine-chain selection.
- src/analysis/vertebral_morphology.py: measurement-only morphology features.
- src/evaluation: CenterNet decoding, metrics, and Cobb calculations.
- src/predict_centernet.py: end-to-end one-stage prediction and output generation.
- src/evaluate_centernet.py: full/source-filtered CenterNet evaluation.
- backup/refiner_approach: retired two-stage refiner experiment and restoration notes.
