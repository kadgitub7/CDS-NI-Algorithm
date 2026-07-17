"""P10 — Progressive Layer 10: + Laplace smoothing = FULL SYSTEM.
Laplace smoothing prevents zero-probability estimates in supervised bins.
This is identical to 99_final — the complete CDS-OVR system.
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
    remove_classes={7, 8, 11, 12, 13},
    rare_classes={4, 5, 9},
    rare_min_support=2,
    rare_conf_support=5,
    rare_against_scale=0.5,
    class_thresholds={2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.0},
    binning='supervised',
    max_bins=6,
    laplace_alpha=1.0,
)

if __name__ == "__main__":
    run_variant(cfg, label="P10: + Laplace = FULL SYSTEM")
