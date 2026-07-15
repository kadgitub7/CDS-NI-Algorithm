"""04 — OVR + Dual AF (for/against evidence tracking).

Splits the AF score into positive evidence (FOR the class)
and negative evidence (AGAINST), allowing the model to weigh
contradictory signals separately."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
)

if __name__ == "__main__":
    run_variant(cfg, label="04: +Dual AF")
