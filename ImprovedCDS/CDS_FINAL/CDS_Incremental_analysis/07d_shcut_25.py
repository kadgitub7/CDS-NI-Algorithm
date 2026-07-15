"""07 param sweep — SUSPICION_HCUT = 2.5."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    class_thresholds={2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.5},
    healthy_bar_cap=5.0,
    healthy_weight=1.05,
    ratio_eps=0.1,
    scoring='ratio',
    suspicion_hcut=2.5,
    suspicion_offset=0.3,
    threshold_mode='healthy_bar',
)

if __name__ == "__main__":
    run_variant(cfg, label="07: Suspicion HCut=2.5")
