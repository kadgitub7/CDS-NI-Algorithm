"""S01 — Supervised Binning + Correlation Feature Selection.
Binning alone overfits on 279 features. Feature selection limits to 18, letting supervised bins shine.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    binning='supervised',
    max_bins=6,
    corr_threshold=0.8,
    features_per_class=18,
)

if __name__ == "__main__":
    run_variant(cfg, label="S01: Supervised Binning + Feature Selection")
