"""05 — OVR + Fisher discriminant weighting.

Weights feature contributions by their Fisher discriminant ratio,
normalizing to [0.1, 1.0]. Features that better separate target
from rest get higher weight."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    fisher=True,
)

if __name__ == "__main__":
    run_variant(cfg, label="05: +Fisher Weighting")
