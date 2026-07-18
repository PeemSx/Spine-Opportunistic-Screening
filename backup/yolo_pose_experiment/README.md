# Retired YOLO-pose utilities

Status: archived on 2026-07-18. These files are not used by the active CenterNet pipeline.

## Why this was archived

The project now trains and evaluates CenterNet models directly from COCO keypoint annotations. No active source file, test, training notebook, or documented command uses the YOLO-pose conversion path.

## Contents

- src/data/convert_coco_to_yolo_pose.py: converts project COCO annotations into Ultralytics YOLO-pose format.
- src/visualization/preview_yolo_annotations.py: previews converted YOLO-pose labels.
- notebooks/yolo/spine_chain_prototype.ipynb: historical YOLO/spine-chain prototype.
- sos_colab_snapshot/: matching utilities removed from the active SOS Colab package.

## Restoration

Copy a file back to the same repository-relative path shown inside this archive. Restore the converter and preview utility together if YOLO-pose experimentation resumes.

