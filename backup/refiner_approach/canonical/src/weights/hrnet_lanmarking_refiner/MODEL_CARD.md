# HRNet-W18 vertebra corner refiner

Production checkpoint: `inference_refiner.pt`

- Architecture: HRNet-W18 with multi-scale corner-refinement head
- Input: grayscale vertebra ROI repeated to three channels, `256 x 256`
- Output: four `64 x 64` heatmaps in `TL, TR, BL, BR` order
- Decoder: spatial softmax with DSNT expected coordinates
- Selected checkpoint: epoch 7 (`best_nme.pt`)
- Validation samples: 2,495 vertebrae
- Validation mean NME: `0.0499754100`
- Validation median NME: `0.0421568691`
- Validation p95 NME: `0.0957091508`
- Mean original-coordinate corner error: `9.2932 px`
- Crop boundary peak rate on oracle validation crops: `0.0`

The production file contains model weights, architecture/crop metadata, and validation metrics. It
does not contain optimizer, learning-rate scheduler, AMP scaler, or resumable training state.

`training_artifacts/last.pt` is retained separately only for resuming or investigating training.
It should not be used for normal prediction.

Example:

```powershell
python -m src.predict_centernet `
  --checkpoint src/weights/hrnet/best_center_f1.pt `
  --refiner-checkpoint src/weights/hrnet_lanmarking_refiner/inference_refiner.pt `
  --source path/to/images `
  --output-dir outputs/two_stage `
  --show-refiner-rois `
  --save-tables
```

This model provides research screening-support landmarks and is not a diagnostic system.
