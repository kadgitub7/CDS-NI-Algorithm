"""06 — OVR + Ratio scoring.

Changes final score from simple sum to:
  Score = (AF_for + eps) / (AF_against + eps)
Automatically uses dual AF for the computation.
Score=1 means balanced evidence; higher = more evidence for class."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    af_mode='dual',
    ratio_eps=0.1,
    scoring='ratio',
)

if __name__ == "__main__":
    run_variant(cfg, label="06: +Ratio Scoring (eps=0.1)")
