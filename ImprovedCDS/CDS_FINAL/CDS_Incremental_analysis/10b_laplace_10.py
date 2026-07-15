"""10 param sweep — LAPLACE_ALPHA = 1.0."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    laplace_alpha=1.0,
)

if __name__ == "__main__":
    run_variant(cfg, label="10: Laplace alpha=1.0")
