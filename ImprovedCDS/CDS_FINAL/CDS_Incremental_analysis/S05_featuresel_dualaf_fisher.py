"""S05 — Feature Selection + Dual AF + Fisher.
The core evidence-improvement trio: select good features, track both for/against, weight by discriminative power.
"""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    corr_threshold=0.8,
    features_per_class=18,
    af_mode='dual',
    fisher=True,
)

if __name__ == "__main__":
    run_variant(cfg, label="S05: Feature Selection + Dual AF + Fisher")
