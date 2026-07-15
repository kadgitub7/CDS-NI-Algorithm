"""07 param sweep — HEALTHY_WEIGHT = 1.0."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    class_thresholds={2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.5},
    healthy_bar_cap=5.0,
    healthy_weight=1.0,
    ratio_eps=0.1,
    scoring='ratio',
    suspicion_hcut=2.0,
    suspicion_offset=0.3,
    threshold_mode='healthy_bar',
)

if __name__ == "__main__":
    run_variant(cfg, label="07: Healthy Weight=1.0")
