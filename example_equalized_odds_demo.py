"""
================================================================================
example_equalized_odds_demo.py – Demonstration of Equalized Odds Integration
================================================================================

PURPOSE
-------
Shows how to:
  1. Run Algorithm 4 to get initial predictions
  2. Apply fairness post-processing
  3. Compare baseline vs. fair predictions
  4. Save results for analysis

USAGE
-----
    python example_equalized_odds_demo.py

This script assumes you have:
  - The arrhythmia dataset (arrhythmia.data)
  - All four algorithms implemented (Algorithm1-4)
  - Fairness modules created

================================================================================
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from CDS_Paper_Algorithms import (
    load_dataset,
    build_decision_tree,
    classify_features,
    DIAGNOSTIC_THRESHOLD,
)
from Algorithm2 import run_algorithm2
from Algorithm3 import run_algorithm3
from Algorithm4 import run_loocv, _build_logger as build_logger_alg4
from Algorithm4_FairnessIntegration import (
    apply_fairness_post_processing,
    print_fairness_summary,
)

log = build_logger_alg4("DEMO")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Dataset path (adjust to your environment)
DATA_PATH = r"C:\Users\Javad\Documents\GitHub\CDS-NI-Algorithm\arrhythmia.data"

# Number of users for quick demo (set to None for full LOOCV)
MAX_USERS = 50  # Set to None to run full LOOCV (slower)

# Random seed for reproducibility
RNG_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main demo pipeline."""
    log.info("=" * 80)
    log.info("EQUALIZED ODDS POST-PROCESSING DEMO")
    log.info("=" * 80)

    # Step 1: Load data
    log.info("\n[STEP 1] Loading dataset...")
    data, labels = load_dataset(DATA_PATH)
    log.info(f"  Loaded {len(data)} users × 279 features")
    log.info(f"  Class distribution: {dict((c, int((labels == c).sum())) for c in sorted(set(labels)))}")

    # Step 2: Build decision tree (Algorithm 1)
    log.info("\n[STEP 2] Building decision tree (Algorithm 1)...")
    feature_kinds = classify_features(data)
    tree = build_decision_tree(data, labels, threshold=DIAGNOSTIC_THRESHOLD, max_m=2)
    log.info(f"  Built tree with {len(tree.nodes)} nodes")

    # Step 3: Run Algorithm 2 (Perceptor & Executive Training)
    log.info("\n[STEP 3] Running Algorithm 2 (Perceptor & Executive)...")
    alg2_output = run_algorithm2(tree, data, labels, n_bins=10)
    log.info(f"  Generated {len(alg2_output.perceptor_library)} perceptor entries")
    log.info(f"  Generated {len(alg2_output.executive_library)} executive actions")

    # Step 4: Run Algorithm 3 (Executive Action Refinement)
    log.info("\n[STEP 4] Running Algorithm 3 (Action Refinement)...")
    alg3_output = run_algorithm3(alg2_output, tree, data, labels, reset_per_h=False)
    log.info(f"  Refined {len(alg3_output.refined_actions)} actions")
    log.info(f"  Removed {len(alg3_output.removed_actions)} actions")

    # Step 5: Run Algorithm 4 (Prediction / Inference)
    log.info("\n[STEP 5] Running Algorithm 4 (Prediction with LOOCV)...")
    alg4_output = run_loocv(
        data,
        labels,
        max_users=MAX_USERS,
        rng_seed=RNG_SEED,
        verbose=False,
        n_bins=10,
    )
    log.info(f"  Completed predictions for {len(alg4_output.records)} users")
    log.info(f"  Healthy: {alg4_output.n_healthy_correct}/{alg4_output.n_healthy_total}")
    log.info(f"  Diseased: {alg4_output.n_diseased_correct}/{alg4_output.n_diseased_total}")

    # Step 6: Apply Fairness Post-Processing
    log.info("\n[STEP 6] Applying Equalized Odds Post-Processing...")
    fairness_output = apply_fairness_post_processing(
        alg4_output,
        data,
        labels,
        baseline_threshold=DIAGNOSTIC_THRESHOLD,
        verbose=True,
    )

    # Step 7: Print results
    log.info("\n[STEP 7] Results Summary...")
    print_fairness_summary(fairness_output)

    # Step 8: Save results (optional)
    log.info("\n[STEP 8] Saving results...")
    _save_results(fairness_output, alg4_output)

    log.info("\n" + "=" * 80)
    log.info("DEMO COMPLETE")
    log.info("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(fairness_output, alg4_output) -> None:
    """Save fairness results to CSV."""
    import csv

    output_file = project_root / "fairness_results.csv"

    try:
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "user_idx",
                "true_label",
                "rw_score",
                "prediction_original",
                "prediction_fair",
                "gender",
            ])

            # Data
            for i, record in enumerate(alg4_output.records):
                gender = "M" if fairness_output.thresholds_optimized.diagnostics else "F"
                writer.writerow([
                    record.user_global_idx,
                    record.true_label,
                    record.af_trace[-1].rw_real if record.af_trace else "N/A",
                    fairness_output.predictions_original[i],
                    fairness_output.predictions_fair[i],
                    gender,
                ])

        log.info(f"  Saved results to {output_file}")
    except Exception as e:
        log.warning(f"  Failed to save results: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        log.error(f"Dataset not found: {e}")
        log.error(f"Please ensure arrhythmia.data exists at: {DATA_PATH}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
