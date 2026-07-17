"""P07 — Progressive Layer 7: + Rare class parameters.
Special support thresholds for remaining rare classes {4,5,9}.
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
)

if __name__ == "__main__":
    run_variant(cfg, label="P07: + Rare Class Params")
