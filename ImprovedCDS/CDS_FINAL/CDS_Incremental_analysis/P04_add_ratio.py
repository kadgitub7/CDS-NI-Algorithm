"""P04 — Progressive Layer 4: + Ratio scoring.
Ratio scoring uses AF_for/AF_against ratio instead of simple difference.
Requires dual AF for the for/against split and healthy bar for proper thresholding.
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
)

if __name__ == "__main__":
    run_variant(cfg, label="P04: + Ratio Scoring")
