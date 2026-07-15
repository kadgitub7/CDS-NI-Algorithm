"""03 — OVR + Correlation-based feature filter.

Limits features per class to 18, discards features with
correlation > 0.8 to remove redundancy."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    corr_threshold=0.8,
    features_per_class=18,
)

if __name__ == "__main__":
    run_variant(cfg, label="03: +Correlation Filter (0.8, FPC=18)")
