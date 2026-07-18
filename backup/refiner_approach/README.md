# Retired corner-refiner approach

Status: archived on 2026-07-18. Nothing in this directory is imported by the active pipeline.

## Decision

The second-stage HRNet-W18 corner refiner was retired because its matched evaluation improved aggregate corner NME by 10.2855 percent, below the predefined 20 percent acceptance target, while adding a second checkpoint, ROI generation, extra inference cost, and additional failure modes.

The archived evaluation used 2,282 fixed CenterNet-to-ground-truth matches from the Mendeley and MICCAI 2019 test sources:

- Baseline mean NME: 0.0734322.
- Two-stage mean NME: 0.0658793.
- Baseline mean corner error: 14.5866 px.
- Two-stage mean corner error: 12.8158 px.
- Relative mean-NME improvement: 10.2855 percent.

The project now uses CenterNet corners directly, followed by spine-chain selection, Cobb-angle calculation, and measurement-only morphology extraction.

## Archive contents

- canonical/: standalone refiner data, ROI, model, loss, decoder, evaluator, workflow checker, notebooks, tests, and checkpoint artifacts moved from the main repository.
- active_snapshot/: exact active configuration, predictor, morphology module, tests, README, and AGENTS context before refiner removal.
- sos_colab_snapshot/: the refiner-specific Colab package files plus its pre-removal config, predictor, morphology module, and README.
- evaluation/: two_stage_metrics.json and matched_instances.csv supporting the retirement decision.
- qc_consensus/: the older multi-crop consensus, QC rejection, and fallback experiment.
- compiled_artifacts/: stale refiner bytecode removed from the active main and Colab source trees.

The historical checkpoint directory retains its original misspelling, hrnet_lanmarking_refiner, for artifact compatibility.

Model binaries remain ignored by the repository-wide Git rules even inside this archive. Keep them in local or private storage if the archive is moved between machines.

## Restoration

Restoration is intentionally manual so that retired behavior cannot silently return.

1. Copy the required modules from canonical/ back to their original repository-relative paths.
2. Restore src/predict_centernet.py, src/analysis/vertebral_morphology.py, and configs/config.yaml from active_snapshot/.
3. Restore the archived refiner tests and run the complete unittest suite.
4. Restore the checkpoint directory if two-stage inference is required.
5. Copy sos_colab_snapshot/ files back only if Colab refiner training is also required.
6. Re-evaluate on fixed matched instances before making the refiner active.

Do not restore only the CLI flags. The predictor, ROI geometry, model loader, morphology provenance, diagnostics, tests, and checkpoint metadata form one compatibility unit.
