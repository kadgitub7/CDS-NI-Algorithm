"""99 — Final CDS-OVR with ALL changes combined.

This is the complete cds_ovr.py configuration — all improvements applied:
- Supervised chi-squared binning (MAX_BINS=6)
- Correlation filter (0.8) with FEATURES_PER_CLASS=18
- Dual AF (for/against)
- Fisher weighting
- Ratio scoring (eps=0.1)
- Healthy bar + threshold + suspicion
- Rare class parameters {4,5,9}
- Remove classes {7,8,11,12,13}
- Laplace smoothing (alpha=1.0)
- Per-class thresholds with class 10 at 3.0
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    binning='supervised',
    max_bins=6,
    corr_threshold=0.8,
    features_per_class=18,
    laplace_alpha=1.0,
    min_support=3,
    conf_support=10,
    against_scale=0.8,
    rare_classes={4, 5, 9},
    rare_min_support=2,
    rare_conf_support=5,
    rare_against_scale=0.5,
    af_mode='dual',
    fisher=True,
    scoring='ratio',
    ratio_eps=0.1,
    threshold_mode='healthy_bar',
    healthy_weight=1.05,
    suspicion_hcut=2.0,
    suspicion_offset=0.3,
    healthy_bar_cap=5.0,
    class_thresholds={2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.0},
    remove_classes={7, 8, 11, 12, 13},
)

if __name__ == "__main__":
    run_variant(cfg, label="99: Final CDS-OVR (All Changes)")
