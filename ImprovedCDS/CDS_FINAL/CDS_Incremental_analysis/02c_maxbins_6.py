"""02 param sweep — Supervised binning with MAX_BINS=6."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    binning='supervised',
    max_bins=6,
)

if __name__ == "__main__":
    run_variant(cfg, label="02: +Supervised Binning (MAX_BINS=6)")
