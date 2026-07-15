"""03 param sweep — Correlation threshold = 0.7, FPC=18."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    corr_threshold=0.7,
    features_per_class=18,
)

if __name__ == "__main__":
    run_variant(cfg, label="03: +Corr Filter (threshold=0.7)")
