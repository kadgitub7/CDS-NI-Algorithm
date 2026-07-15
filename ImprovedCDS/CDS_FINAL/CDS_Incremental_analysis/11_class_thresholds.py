"""11 — OVR + Per-class decision thresholds (fixed mode).

Different classes need different evidence thresholds.
Uses the same thresholds as the final model but without
healthy bar or suspicion."""
from _ovr_engine import CDSConfig, run_variant

cfg = CDSConfig(
    class_thresholds={2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.5},
)

if __name__ == "__main__":
    run_variant(cfg, label="11: +Per-Class Thresholds")
