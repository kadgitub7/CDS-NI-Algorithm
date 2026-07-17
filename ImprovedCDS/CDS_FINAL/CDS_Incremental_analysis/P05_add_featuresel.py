"""P05 — Progressive Layer 5: + Feature selection.
Correlation filter (0.8) + features_per_class (18) to reduce noise features.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    fisher=True,
    scoring='ratio',
    ratio_eps=0.1,
    threshold_mode='healthy_bar',
    healthy_weight=1.05,
    suspicion_hcut=2.0,
    suspicion_offset=0.3,
    healthy_bar_cap=5.0,
    corr_threshold=0.8,
    features_per_class=18,
)

if __name__ == "__main__":
    run_variant(cfg, label="P05: + Feature Selection")
