"""10 — OVR + Laplace smoothing on bin probabilities.

P(class|bin) = (target_count + alpha*prior) / (bin_count + alpha)
Prevents zero probabilities in sparse bins."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    laplace_alpha=1.0,
)

if __name__ == "__main__":
    run_variant(cfg, label="10: +Laplace Smoothing (alpha=1.0)")
