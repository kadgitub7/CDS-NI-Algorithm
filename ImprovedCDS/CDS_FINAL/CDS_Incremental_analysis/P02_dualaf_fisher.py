"""P02 — Progressive Layer 2: OVR + Dual AF + Fisher weighting.
Fisher weights features by discriminative power, amplifying dual AF's evidence quality.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    fisher=True,
)

if __name__ == "__main__":
    run_variant(cfg, label="P02: + Fisher Weighting")
