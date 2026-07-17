"""S03 — Dual AF + Fisher Weighting.
Both improve evidence quality — dual AF tracks for/against, Fisher weights by discriminative power.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    fisher=True,
)

if __name__ == "__main__":
    run_variant(cfg, label="S03: Dual AF + Fisher Weighting")
