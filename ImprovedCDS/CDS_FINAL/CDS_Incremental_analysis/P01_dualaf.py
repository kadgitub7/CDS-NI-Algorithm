"""P01 — Progressive Layer 1: OVR + Dual AF.
Starting from the OVR baseline, add dual AF evidence tracking.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
)

if __name__ == "__main__":
    run_variant(cfg, label="P01: OVR + Dual AF")
