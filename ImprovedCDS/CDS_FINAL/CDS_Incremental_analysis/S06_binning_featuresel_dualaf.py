"""S06 — Supervised Binning + Feature Selection + Dual AF.
Foundation trio: discriminative bins + curated features + for/against evidence.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    binning='supervised',
    max_bins=6,
    corr_threshold=0.8,
    features_per_class=18,
    af_mode='dual',
)

if __name__ == "__main__":
    run_variant(cfg, label="S06: Binning + Feature Selection + Dual AF")
