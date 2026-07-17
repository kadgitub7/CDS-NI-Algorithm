"""S08 — Class Removal + Rare Class Parameters.
Both address class distribution: remove unlearnable classes, then give remaining rare classes
special treatment with lower support thresholds.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    remove_classes={7, 8, 11, 12, 13},
    rare_classes={4, 5, 9},
    rare_min_support=2,
    rare_conf_support=5,
    rare_against_scale=0.5,
)

if __name__ == "__main__":
    run_variant(cfg, label="S08: Class Removal + Rare Class Parameters")
