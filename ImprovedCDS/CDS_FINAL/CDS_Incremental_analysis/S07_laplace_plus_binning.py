"""S07 — Laplace Smoothing + Supervised Binning.
Supervised binning can create bins with zero counts for some classes. Laplace smoothing
prevents zero-probability estimates that would block evidence accumulation.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    binning='supervised',
    max_bins=6,
    laplace_alpha=1.0,
)

if __name__ == "__main__":
    run_variant(cfg, label="S07: Laplace Smoothing + Supervised Binning")
