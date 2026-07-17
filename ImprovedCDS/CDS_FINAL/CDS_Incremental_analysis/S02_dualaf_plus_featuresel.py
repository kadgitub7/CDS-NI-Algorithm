"""S02 — Dual AF + Feature Selection.
Dual AF is the most impactful single change but noisy features dilute evidence quality.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    corr_threshold=0.8,
    features_per_class=18,
)

if __name__ == "__main__":
    run_variant(cfg, label="S02: Dual AF + Feature Selection")
