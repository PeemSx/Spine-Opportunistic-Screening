# SOS Colab: CenterNet HRNet-W18 COCO + NIH

This folder is ready to upload to private Google Drive and train a new CenterNet HRNet-W18 model on BUU, Mendeley, MICCAI 2019, and the newly annotated NIH ChestX-ray14 images.

Use this notebook:

    notebooks/colab/train_centernet_hrnet_w18_nih.ipynb

The dataset remains a single copy:

    dataset/
      split_summary.json
      train/
      val/
      test/

Merged counts:

| Split | Images | Annotations | NIH images |
|---|---:|---:|---:|
| train | 1,550 | 19,822 | 147 |
| val | 194 | 2,695 | 19 |
| test | 193 | 2,778 | 18 |

NIH patients are kept in only one split. The corner order is TL, TR, BL, BR.

The default experiment is:

    centernet_hrnet_w18_coco_nih

Training writes to fast local Colab storage and synchronizes essential checkpoints to:

    MyDrive/spine_centernet_runs_nih/centernet_hrnet_w18_coco_nih

The training cell shows live batch progress for training, validation loss, and center evaluation. It also prints a complete epoch summary and the current early-stopping counter.

Upload this complete folder to MyDrive/SOS Colab, open the notebook in Colab, select a GPU runtime, and run the cells from top to bottom. The older BUU and refiner notebooks are retained for reproducibility.
