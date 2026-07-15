"""08 param sweep — Rare class MIN_SUPPORT = 3."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    rare_against_scale=0.5,
    rare_classes={4, 5, 9},
    rare_conf_support=5,
    rare_min_support=3,
)

if __name__ == "__main__":
    run_variant(cfg, label="08: Rare Min Support=3")
