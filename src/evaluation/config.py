from __future__ import annotations


DEFAULT_PEAK_THRESHOLD = 0.10
MATCH_THRESHOLDS = (0.10, 0.20, 0.25)
PRIMARY_MATCH_THRESHOLD = 0.20
PCK_THRESHOLDS = (0.05, 0.10, 0.20)
USABLE_NME_THRESHOLD = 0.10
NUMERICAL_TOLERANCE = 1e-6
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260627

DEFAULT_ACCEPTANCE_GATES = {
    "source_macro_f1_0.20d_min": 0.85,
    "worst_source_recall_0.20d_min": 0.75,
    "mean_nme_max": 0.08,
    "pck_0.10_min": 0.78,
    "usable_vertebra_recall_min": 0.65,
    "count_mae_max": 1.5,
    "absolute_count_bias_max": 0.5,
    "chain_fp_reduction_min": 0.30,
    "chain_recall_loss_max": 0.05,
    "cobb_coverage_min": 0.98,
    "cobb_mae_deg_max": 5.0,
    "cobb_at_5deg_min": 0.75,
    "cobb_at_10deg_min": 0.90,
}
