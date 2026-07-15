"""08 param sweep — Rare class CONF_SUPPORT = 5."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    rare_against_scale=0.5,
    rare_classes={4, 5, 9},
    rare_conf_support=5,
    rare_min_support=2,
)

if __name__ == "__main__":
    run_variant(cfg, label="08: Rare Conf Support=5")
