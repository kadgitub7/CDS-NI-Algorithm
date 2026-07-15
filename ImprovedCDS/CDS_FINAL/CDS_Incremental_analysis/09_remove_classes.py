"""09 — OVR + Remove sparse classes from dataset.

Classes {7,8,11,12,13} have very few samples and add noise.
Removing them focuses the model on classes with enough data."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    remove_classes={7, 8, 11, 12, 13},
)

if __name__ == "__main__":
    run_variant(cfg, label="09: +Remove Classes {7,8,11,12,13}")
