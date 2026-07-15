"""06 param sweep — Ratio scoring with epsilon = 0.1."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    ratio_eps=0.1,
    scoring='ratio',
)

if __name__ == "__main__":
    run_variant(cfg, label="06: +Ratio Scoring (eps=0.1)")
