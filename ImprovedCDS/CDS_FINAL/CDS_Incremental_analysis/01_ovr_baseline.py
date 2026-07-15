"""01 — OVR Baseline: Sturges binning, score-based refinement, simple AF, fixed threshold.

This is the minimal OVR conversion of the original algorithm.
All subsequent changes are measured against this baseline."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    # All defaults
)

if __name__ == "__main__":
    run_variant(cfg, label="01: OVR Baseline")
