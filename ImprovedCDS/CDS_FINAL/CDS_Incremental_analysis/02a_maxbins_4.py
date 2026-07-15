"""02 param sweep — Supervised binning with MAX_BINS=4."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    binning='supervised',
    max_bins=4,
)

if __name__ == "__main__":
    run_variant(cfg, label="02: +Supervised Binning (MAX_BINS=4)")
