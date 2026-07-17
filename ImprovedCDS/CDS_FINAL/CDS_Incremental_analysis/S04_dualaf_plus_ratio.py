"""S04 — Dual AF + Ratio Scoring (no healthy bar).
Ratio scoring was designed for dual AF's for/against structure but drops accuracy
without the healthy bar threshold system — this tests whether that pairing helps alone.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    scoring='ratio',
    ratio_eps=0.1,
)

if __name__ == "__main__":
    run_variant(cfg, label="S04: Dual AF + Ratio Scoring")
