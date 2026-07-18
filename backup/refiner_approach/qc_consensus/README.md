# Archived refiner QC and consensus

This folder preserves the inference safeguards removed from the active
two-stage pipeline. Nothing under `backup/` is imported at runtime.

Archived behavior:

- seven crops: base, 10% smaller/larger, and 5% left/right/up/down;
- coordinate-wise median corner consensus;
- unstable consensus when p95 disagreement exceeds 4% of base ROI side;
- rejection for center displacement above 20% of ROI side;
- rejection when the refined center is nearer another detector anchor;
- TL/TR/BL/BR ordering and 0.5%-of-ROI minimum polygon area checks;
- rejection when a base-crop heatmap argmax touches its outermost cell;
- complete CenterNet landmark fallback when any rule fails.

The exact reusable geometry, consensus, and rejection helpers are in
`qc_consensus.py`. To restore the feature, import these helpers into
`src/inference/corner_refiner.py`, construct seven ROIs before batching, use
`compute_corner_consensus` as the refined output, and call
`candidate_qc_reasons` before replacing CenterNet corners. Re-add the archived
CLI controls listed in `cli_options.txt` to prediction and evaluation.

This QC/consensus variant and the later single-crop refiner pipeline are now
both retired. See ../README.md for the archive decision and complete
restoration sequence.
