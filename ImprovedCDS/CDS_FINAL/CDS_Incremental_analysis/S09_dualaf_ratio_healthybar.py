"""S09 — Dual AF + Ratio Scoring + Healthy Bar (the full scoring pipeline).
Ratio scoring needs dual AF for the for/against ratio, and healthy bar for proper thresholding.
This tests the complete scoring/decision pipeline without feature engineering changes.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    scoring='ratio',
    ratio_eps=0.1,
    threshold_mode='healthy_bar',
    healthy_weight=1.05,
    suspicion_hcut=2.0,
    suspicion_offset=0.3,
    healthy_bar_cap=5.0,
    class_thresholds={2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.0},
)

if __name__ == "__main__":
    run_variant(cfg, label="S09: Dual AF + Ratio + Healthy Bar (full scoring pipeline)")
