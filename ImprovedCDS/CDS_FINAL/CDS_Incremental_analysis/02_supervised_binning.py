"""02 — OVR + Supervised chi-squared binning (MAX_BINS=6).

Replaces Sturges' uniform binning with chi-squared supervised binning
that finds optimal split points based on class separation."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    binning='supervised',
    max_bins=6,
)

if __name__ == "__main__":
    run_variant(cfg, label="02: +Supervised Binning (MAX_BINS=6)")
