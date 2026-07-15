"""03 param sweep — Features per class = 14, CORR=0.8."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    corr_threshold=0.8,
    features_per_class=14,
)

if __name__ == "__main__":
    run_variant(cfg, label="03: +Corr Filter (FPC=14)")
