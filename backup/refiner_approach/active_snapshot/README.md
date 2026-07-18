# Spine Opportunistic Screening

Deep learning-based vertebral landmark detection for morphology and spinal-curvature assessment in routine AP/PA radiographs.

This is a research and screening-support project. It does not diagnose scoliosis, osteoporosis, vertebral compression fracture, or any other condition. Measurements and overlays require clinical review.

## Current system

The active one-stage pipeline uses a CenterNet-style model to predict a vertebral center, center offset, and four ordered corners for every visible vertebra. A spine-chain algorithm removes duplicate and anatomically inconsistent candidates before Cobb-angle and morphology measurements are calculated.

An optional HRNet-W18 corner refiner is maintained as an experimental second stage. It replaces CenterNet corners using one annotation-derived or detector-derived ROI per candidate. The current matched evaluation improved corner NME by about 10.3 percent, below the project acceptance target of 20 percent, so the refiner is optional rather than the recommended default.

Current processing flow:

    AP/PA radiograph
      -> resize/pad preprocessing
      -> CenterNet vertebra and corner prediction
      -> optional corner refiner
      -> spine-chain candidate selection
      -> Cobb-angle and measurement-only morphology features
      -> overlays and optional CSV/JSON tables

Suspicious-morphology classification is not implemented yet. The current morphology output contains measurements and neighbor-relative features only.

## Installation

From the repository root on Windows PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirement.txt
~~~

The main configuration file is configs/config.yaml. It contains image augmentation settings and the corner-refiner training defaults.

## Prediction

Recommended one-stage prediction with the NIH-adapted HRNet-W18 checkpoint:

~~~powershell
python -m src.predict_centernet --checkpoint src/weights/hrnet_nih/best_center_f1.pt --source path/to/radiographs --output-dir outputs/predictions/hrnet18_nih --overlay-background original --show-corners box --save-tables
~~~

Spine-chain selection and Cobb overlays are enabled by default. Checkpoint metadata supplies the backbone, input size, output stride, threshold, and top-k settings unless explicitly overridden.

Useful switches:

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

To enable the experimental corner refiner:

~~~powershell
python -m src.predict_centernet --checkpoint src/weights/hrnet_nih/best_center_f1.pt --refiner-checkpoint src/weights/hrnet_lanmarking_refiner/inference_refiner.pt --source path/to/radiographs --output-dir outputs/predictions/two_stage --overlay-background original --show-refiner-boxes --save-tables
~~~

## CenterNet evaluation

Evaluate the complete test split:

~~~powershell
python -m src.evaluate_centernet --checkpoint src/weights/hrnet_nih/best_center_f1.pt --dataset-root dataset/processed/coco_nih --split test --output-dir outputs/evaluation/hrnet18_nih_full_test --batch-size 2 --num-workers 0 --peak-thresh 0.05 --topk 50
~~~

Evaluate only the NIH source:

~~~powershell
python -m src.evaluate_centernet --checkpoint src/weights/hrnet_nih/best_center_f1.pt --dataset-root dataset/processed/coco_nih --split test --source-dataset "NIH ChestX-ray14" --output-dir outputs/evaluation/hrnet18_nih_test --batch-size 2 --num-workers 0 --peak-thresh 0.05 --topk 50
~~~

The source filter is case-insensitive and is applied before --limit. Evaluation writes metrics.json, metrics.csv, and per_image_metrics.csv with separate raw and spine-chain rows.

For fair checkpoint comparisons, explicitly use the same dataset, source filter, peak threshold, top-k, input size, and chain settings. Do not compare results that used different thresholds without re-evaluating them.

## CenterNet training and Colab

Validate dataset tensors and one model forward pass:

~~~powershell
python -m src.workflows.check_centernet_dataset --dataset-root dataset/processed/coco_nih --split train --image-size 1024 --backbone hrnet_w18 --batch-size 1 --num-workers 0 --limit 2 --model-forward
~~~

The current Colab notebook is:

    notebooks/colab/train_centernet_hrnet_w18_nih.ipynb

It trains on local Colab storage, prints live train/validation/evaluation progress, resumes from last.pt, and atomically backs up essential artifacts to Google Drive.
