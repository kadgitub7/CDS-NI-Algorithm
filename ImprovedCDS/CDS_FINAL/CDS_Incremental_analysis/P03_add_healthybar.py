"""P03 — Progressive Layer 3: + Healthy Bar thresholding.
Adaptive thresholds based on healthy-class evidence replace fixed threshold.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    fisher=True,
    threshold_mode='healthy_bar',
    healthy_weight=1.05,
    suspicion_hcut=2.0,
    suspicion_offset=0.3,
    healthy_bar_cap=5.0,
)

if __name__ == "__main__":
    run_variant(cfg, label="P03: + Healthy Bar")
