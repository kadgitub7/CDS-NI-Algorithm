"""
================================================================================
Algorithm4_FairnessIntegration.py – Integration of Equalized Odds with Algorithm 4
================================================================================

PURPOSE
-------
Bridges Algorithm 4 (prediction) with Fairness_EqualizedOdds post-processing.

This module:
  1. Extracts prediction scores (rw values) from Algorithm4Output
  2. Calls fairness post-processing to compute gender-specific thresholds
  3. Re-classifies predictions using fair thresholds
  4. Validates fairness metrics (equal TPR across genders)
  5. Reports utility loss and performance trade-offs

INTEGRATION POINT
-----------------
After running Algorithm 4 (LOOCV or batch prediction):
  1. Collect Algorithm4Output
  2. Call apply_fairness_post_processing()
  3. Use returned thresholds for deployment

USAGE EXAMPLE
-----------
    from Algorithm4 import run_loocv
    from Algorithm4_FairnessIntegration import apply_fairness_post_processing

    # Run Algorithm 4 to get predictions
    alg4_output = run_loocv(data, labels, ...)

    # Apply equalized odds post-processing
    fairness_output = apply_fairness_post_processing(
        alg4_output,
        data,
        labels,
        baseline_threshold=0.025
    )

    # Use fairness_output.thresholds for future predictions
    threshold_male = fairness_output.thresholds_optimized.threshold_male
    threshold_female = fairness_output.thresholds_optimized.threshold_female
================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Import fairness module ─────────────────────────────────────────────────────
from Fairness_EqualizedOdds import (
    compute_bayes_optimal_predictor,
    apply_equalized_odds_thresholds,
    EqualizedOddsResult,
    SEX_FEATURE_IDX,
    MALE_VALUE,
    FEMALE_VALUE,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger(name: str = "CDS.Alg4.Fairness") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)-7s | %(name)s | %(message)s")
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(fmt)
    log.addHandler(h)
    log.propagate = False
    return log

log = _build_logger()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FairnessMetrics:
    """Fairness and performance metrics."""
    
    # Equalized odds metrics
    tpr_male: float                   # True positive rate for males
    tpr_female: float                 # True positive rate for females
    tpr_gap: float                    # |TPR_male - TPR_female|
    tpr_ratio: float                  # min(TPR_male, TPR_female) / max(TPR_male, TPR_female)
    
    # False positive rates
    fpr_male: float
    fpr_female: float
    fpr_gap: float
    
    # Accuracy metrics
    accuracy_overall: float            # (TP + TN) / N
    accuracy_male: float               # (TP + TN) / N_male
    accuracy_female: float             # (TP + TN) / N_female
    accuracy_drop: float               # accuracy_baseline - accuracy_fair
    
    # Utility loss
    utility_loss: float                # from fairness optimization
    
    # Counts
    n_male: int
    n_female: int
    n_total: int


@dataclass
class FairnessPostProcessingOutput:
    """
    Complete output of fairness post-processing integration.

    Attributes
    ----------
    thresholds_baseline : baseline single threshold (usually 0.025).
    thresholds_optimized: EqualizedOddsResult with gender-specific thresholds.
    fairness_metrics    : FairnessMetrics before and after post-processing.
    predictions_original: original predictions from Algorithm 4.
    predictions_fair    : re-classified predictions using fair thresholds.
    """
    thresholds_baseline: float
    thresholds_optimized: EqualizedOddsResult
    fairness_metrics_original: FairnessMetrics
    fairness_metrics_fair: FairnessMetrics
    predictions_original: np.ndarray
    predictions_fair: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – HELPER FUNCTIONS: EXTRACT SCORES & LABELS FROM ALGORITHM 4 OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def extract_rw_scores_from_algorithm4(
    alg4_output,  # Algorithm4Output
) -> np.ndarray:
    """
    Extract rw (remaining work) scores from Algorithm 4 output.

    [PAPER] rw = 1 - AF (assurance factor).
    In Algorithm 4, the DIAGNOSTIC_THRESHOLD is compared against rw.
    Lower rw = higher confidence in healthy.

    Parameters
    ----------
    alg4_output : Algorithm4Output from run_algorithm4() or run_loocv().

    Returns
    -------
    scores : array of shape (n,) with rw values for each user.
    """
    n_users = len(alg4_output.records)
    scores = np.zeros(n_users, dtype=float)

    for i, record in enumerate(alg4_output.records):
        # Extract final rw from AF trace
        if record.af_trace:
            final_rw = record.af_trace[-1].rw_real
        else:
            # Fallback: compute from decision
            # HEALTHY → high confidence (low rw)
            # UNHEALTHY → alarm triggered (rw may vary)
            # SCREENING → insufficient confidence (high rw)
            final_rw = 1.0  # default

        scores[i] = final_rw

    log.info(f"Extracted {n_users} rw scores from Algorithm 4 output")
    log.info(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")

    return scores


def extract_true_labels_from_algorithm4(
    alg4_output,  # Algorithm4Output
) -> np.ndarray:
    """
    Extract ground-truth labels from Algorithm 4 output.

    Parameters
    ----------
    alg4_output : Algorithm4Output.

    Returns
    -------
    labels : array of shape (n,) with true class labels (1=healthy, ≥2=diseased).
    """
    n_users = len(alg4_output.records)
    labels = np.zeros(n_users, dtype=int)

    for i, record in enumerate(alg4_output.records):
        labels[i] = record.true_label

    return labels


def extract_decisions_from_algorithm4(
    alg4_output,  # Algorithm4Output
) -> np.ndarray:
    """
    Extract binary predictions from Algorithm 4 output.

    Returns
    -------
    predictions : array of shape (n,) with 0=healthy, 1=unhealthy.
    """
    from Algorithm4 import HealthDecision

    n_users = len(alg4_output.records)
    predictions = np.zeros(n_users, dtype=int)

    for i, record in enumerate(alg4_output.records):
        # Convert decision to binary prediction
        # HEALTHY → 0 (negative class)
        # UNHEALTHY → 1 (positive class)
        # SCREENING → undefined; treat as 0 for now
        predictions[i] = 1 if record.decision == HealthDecision.UNHEALTHY else 0

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – COMPUTE FAIRNESS METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_fairness_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    data: np.ndarray,
) -> FairnessMetrics:
    """
    Compute fairness and performance metrics for predictions.

    Parameters
    ----------
    predictions : binary predictions (0=healthy, 1=unhealthy). Shape (n,).
    labels      : ground-truth labels (1=healthy, ≥2=diseased). Shape (n,).
    data        : feature matrix (used to extract gender). Shape (n, 279).

    Returns
    -------
    FairnessMetrics object.
    """
    # Convert labels: 1=negative (healthy), ≥2=positive (diseased)
    true_positive = (labels != 1).astype(int)

    # Extract gender
    gender_col = data[:, SEX_FEATURE_IDX]
    male_mask = (gender_col == MALE_VALUE)
    female_mask = (gender_col == FEMALE_VALUE)

    # Compute metrics by gender
    def compute_group_metrics(mask, group_name):
        pred_g = predictions[mask]
        true_g = true_positive[mask]

        # Confusion matrix
        tp = np.sum((pred_g == 1) & (true_g == 1))
        fp = np.sum((pred_g == 1) & (true_g == 0))
        tn = np.sum((pred_g == 0) & (true_g == 0))
        fn = np.sum((pred_g == 0) & (true_g == 1))

        # Metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        return {
            "tpr": tpr,
            "fpr": fpr,
            "accuracy": accuracy,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    male_metrics = compute_group_metrics(male_mask, "Male")
    female_metrics = compute_group_metrics(female_mask, "Female")

    # Overall metrics
    tp_total = male_metrics["tp"] + female_metrics["tp"]
    tn_total = male_metrics["tn"] + female_metrics["tn"]
    n_total = len(predictions)
    accuracy_overall = (tp_total + tn_total) / n_total if n_total > 0 else 0.0

    # Fairness gaps
    tpr_gap = abs(male_metrics["tpr"] - female_metrics["tpr"])
    tpr_ratio = min(male_metrics["tpr"], female_metrics["tpr"]) / max(
        male_metrics["tpr"], female_metrics["tpr"]
    ) if max(male_metrics["tpr"], female_metrics["tpr"]) > 0 else 0.0
    fpr_gap = abs(male_metrics["fpr"] - female_metrics["fpr"])

    return FairnessMetrics(
        tpr_male=male_metrics["tpr"],
        tpr_female=female_metrics["tpr"],
        tpr_gap=tpr_gap,
        tpr_ratio=tpr_ratio,
        fpr_male=male_metrics["fpr"],
        fpr_female=female_metrics["fpr"],
        fpr_gap=fpr_gap,
        accuracy_overall=accuracy_overall,
        accuracy_male=male_metrics["accuracy"],
        accuracy_female=female_metrics["accuracy"],
        accuracy_drop=0.0,  # computed later
        utility_loss=0.0,   # computed later
        n_male=int(np.sum(male_mask)),
        n_female=int(np.sum(female_mask)),
        n_total=n_total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – MAIN INTEGRATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def apply_fairness_post_processing(
    alg4_output,  # Algorithm4Output
    data: np.ndarray,
    labels: np.ndarray,
    baseline_threshold: float = 0.025,
    verbose: bool = True,
) -> FairnessPostProcessingOutput:
    """
    Apply equalized odds post-processing to Algorithm 4 predictions.

    This is the main integration function.

    [HARDT] Section 4: "Algorithm 1: Computing the Bayes-optimal fair classifier"

    Parameters
    ----------
    alg4_output        : Algorithm4Output from run_loocv() or run_algorithm4().
    data               : feature matrix. Shape (n, 279).
    labels             : ground-truth labels. Shape (n,).
    baseline_threshold : original threshold used in Algorithm 4 (usually 0.025).
    verbose            : whether to log detailed information.

    Returns
    -------
    FairnessPostProcessingOutput with original and fair predictions.
    """
    log.info("=" * 80)
    log.info("ALGORITHM 4 – FAIRNESS POST-PROCESSING INTEGRATION")
    log.info("=" * 80)

    # Step 1: Extract scores and labels
    log.info("\nStep 1: Extracting prediction scores from Algorithm 4...")
    rw_scores = extract_rw_scores_from_algorithm4(alg4_output)
    true_labels = extract_true_labels_from_algorithm4(alg4_output)
    original_predictions = extract_decisions_from_algorithm4(alg4_output)

    log.info(f"  Extracted {len(rw_scores)} predictions")

    # Step 2: Compute fairness metrics on original predictions
    log.info("\nStep 2: Computing fairness metrics on original predictions...")
    metrics_original = compute_fairness_metrics(original_predictions, true_labels, data)

    log.info(f"\nOriginal predictions (baseline threshold={baseline_threshold}):")
    _print_fairness_metrics(metrics_original, "ORIGINAL")

    # Step 3: Apply fairness post-processing
    log.info("\nStep 3: Applying equalized odds post-processing...")
    fairness_result = compute_bayes_optimal_predictor(
        rw_scores,
        true_labels,
        data,
        baseline_threshold=baseline_threshold,
        verbose=verbose,
    )

    # Step 4: Apply fair thresholds
    log.info("\nStep 4: Re-classifying predictions with fair thresholds...")
    fair_predictions = apply_equalized_odds_thresholds(rw_scores, data, fairness_result)

    log.info(f"  Re-classified {np.sum(fair_predictions != original_predictions)} predictions")

    # Step 5: Compute fairness metrics on fair predictions
    log.info("\nStep 5: Computing fairness metrics on fair predictions...")
    metrics_fair = compute_fairness_metrics(fair_predictions, true_labels, data)

    log.info(f"\nFair predictions (equalized odds thresholds):")
    _print_fairness_metrics(metrics_fair, "FAIR")

    # Step 6: Compare
    log.info("\nStep 6: Comparing original vs. fair predictions...")
    _print_comparison(metrics_original, metrics_fair)

    output = FairnessPostProcessingOutput(
        thresholds_baseline=baseline_threshold,
        thresholds_optimized=fairness_result,
        fairness_metrics_original=metrics_original,
        fairness_metrics_fair=metrics_fair,
        predictions_original=original_predictions,
        predictions_fair=fair_predictions,
    )

    log.info("=" * 80)
    log.info("POST-PROCESSING COMPLETE")
    log.info("=" * 80)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _print_fairness_metrics(metrics: FairnessMetrics, label: str) -> None:
    """Print fairness metrics."""
    log.info(f"\n{label} METRICS")
    log.info("-" * 80)
    log.info(f"  True Positive Rate (Sensitivity / Recall):")
    log.info(f"    Male:                  {metrics.tpr_male:.4f}")
    log.info(f"    Female:                {metrics.tpr_female:.4f}")
    log.info(f"    Gap (|difference|):    {metrics.tpr_gap:.4f}")
    log.info(f"    Ratio (min/max):       {metrics.tpr_ratio:.4f}")
    log.info(f"  False Positive Rate (1 - Specificity):")
    log.info(f"    Male:                  {metrics.fpr_male:.4f}")
    log.info(f"    Female:                {metrics.fpr_female:.4f}")
    log.info(f"    Gap:                   {metrics.fpr_gap:.4f}")
    log.info(f"  Accuracy:")
    log.info(f"    Overall:               {metrics.accuracy_overall:.4f}")
    log.info(f"    Male:                  {metrics.accuracy_male:.4f}")
    log.info(f"    Female:                {metrics.accuracy_female:.4f}")
    log.info(f"  Dataset: {metrics.n_male} male, {metrics.n_female} female")
    log.info("-" * 80)


def _print_comparison(
    metrics_original: FairnessMetrics,
    metrics_fair: FairnessMetrics,
) -> None:
    """Print comparison between original and fair predictions."""
    log.info("\nCOMPARISON: ORIGINAL vs. FAIR")
    log.info("-" * 80)

    log.info("TPR Gap (lower is fairer):")
    log.info(
        f"  Original: {metrics_original.tpr_gap:.4f} → "
        f"Fair: {metrics_fair.tpr_gap:.4f} "
        f"(Δ = {metrics_fair.tpr_gap - metrics_original.tpr_gap:+.4f})"
    )

    log.info("TPR Ratio (higher is fairer):")
    log.info(
        f"  Original: {metrics_original.tpr_ratio:.4f} → "
        f"Fair: {metrics_fair.tpr_ratio:.4f} "
        f"(Δ = {metrics_fair.tpr_ratio - metrics_original.tpr_ratio:+.4f})"
    )

    log.info("Overall Accuracy:")
    log.info(
        f"  Original: {metrics_original.accuracy_overall:.4f} → "
        f"Fair: {metrics_fair.accuracy_overall:.4f} "
        f"(Δ = {metrics_fair.accuracy_overall - metrics_original.accuracy_overall:+.4f})"
    )

    log.info("FPR Gap:")
    log.info(
        f"  Original: {metrics_original.fpr_gap:.4f} → "
        f"Fair: {metrics_fair.fpr_gap:.4f} "
        f"(Δ = {metrics_fair.fpr_gap - metrics_original.fpr_gap:+.4f})"
    )

    log.info("-" * 80)


def print_fairness_summary(
    fairness_output: FairnessPostProcessingOutput,
) -> None:
    """Print complete summary of fairness post-processing."""
    log.info("\n" + "=" * 80)
    log.info("FAIRNESS POST-PROCESSING SUMMARY")
    log.info("=" * 80)

    log.info("\nBASELINE (Single threshold for all users):")
    _print_fairness_metrics(fairness_output.fairness_metrics_original, "ORIGINAL")

    log.info("\nFAIR (Gender-specific thresholds from equalized odds):")
    _print_fairness_metrics(fairness_output.fairness_metrics_fair, "FAIR")

    log.info("\nOPTIMAL THRESHOLDS:")
    log.info(
        f"  Male:                    {fairness_output.thresholds_optimized.threshold_male:.6f}"
    )
    log.info(
        f"  Female:                  {fairness_output.thresholds_optimized.threshold_female:.6f}"
    )
    log.info(
        f"  Equalized TPR:           {fairness_output.thresholds_optimized.tpr_equalized:.4f}"
    )
    log.info(
        f"  Utility Loss:            {fairness_output.thresholds_optimized.utility_loss:.6f}"
    )

    log.info("\n" + "=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Module initialization."""
    log.info("Algorithm4_FairnessIntegration module loaded successfully.")
    log.info("Use apply_fairness_post_processing() as main entry point.")


if __name__ == "__main__":
    main()
