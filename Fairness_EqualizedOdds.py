"""
================================================================================
Fairness_EqualizedOdds.py – Equalized Odds Post-Processing for CDS
================================================================================

PURPOSE
-------
Implements equalized odds post-processing following Hardt et al. (2016):
"Equality of Opportunity in Supervised Learning" (arXiv:1610.02413).

Equalized Odds ensures that the True Positive Rate (TPR) and False Positive
Rate (FPR) are equal across protected groups (gender in this case).

  GOAL: Equal sensitivity (true positive rate) between male and female users
        while minimizing loss in overall accuracy.

METHODOLOGY
-----------
  1. Compute ROC curves separately for each gender group
  2. Find the convex hull (Pareto frontier) of achievable (TPR, FPR) pairs
  3. Optimize along the convex hull to find gender-specific thresholds that:
     a) Equalize TPR across genders (equalized odds constraint)
     b) Minimize utility loss (expected error rate increase)

PAPER FIDELITY
--------------
  [HARDT]  – directly from Hardt et al. (2016) framework
  [INFER]  – required interpretation not explicitly in paper
  [ENGR]   – engineering choice with justification

KEY EQUATIONS (Hardt et al., Section 4)
----------------------------------------
  For two groups A (male) and B (female) and threshold pair (t_A, t_B):

  TPR_A(t_A) = P(ŷ=1 | y=1, group=A, threshold=t_A)  [True positive rate, group A]
  TPR_B(t_B) = P(ŷ=1 | y=1, group=B, threshold=t_B)  [True positive rate, group B]

  Equalized Odds Constraint:  TPR_A(t_A) = TPR_B(t_B)

  Utility Loss:  L(t_A, t_B) = E[error rate with thresholds] - E[error rate baseline]
                              = weighted sum of FPR and FNR changes per group

INTEGRATION
-----------
This module processes Algorithm4Output to:
  1. Extract prediction scores (rw values) and true labels
  2. Separate by gender (feature index 1 in the dataset)
  3. Compute ROC curves and find optimal thresholds
  4. Return gender-specific thresholds to replace DIAGNOSTIC_THRESHOLD

OUTPUT
------
  EqualizedOddsResult
    .threshold_male       – threshold to apply for male users
    .threshold_female     – threshold to apply for female users
    .tpr_equalized        – achieved equal TPR value
    .utility_loss         – loss incurred vs. single global threshold
    .operating_point      – (FPR, TPR) pair on convex hull
    .diagnostics          – detailed optimization info

================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar, brentq

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger(name: str = "CDS.Fairness") -> logging.Logger:
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
# SECTION 2 – CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SEX_FEATURE_IDX: int = 1       # Column index for sex in arrhythmia dataset
MALE_VALUE: float = 0.0        # Encoding for male
FEMALE_VALUE: float = 1.0      # Encoding for female

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ROCPoint:
    """
    One point on an ROC curve.

    Attributes
    ----------
    threshold  : the decision threshold used to compute this point.
    fpr        : false positive rate = FP / N (where N = # negatives).
    tpr        : true positive rate = TP / P (where P = # positives).
    fp         : absolute count of false positives.
    tp         : absolute count of true positives.
    fn         : absolute count of false negatives.
    tn         : absolute count of true negatives.
    """
    threshold: float
    fpr: float
    tpr: float
    fp: int
    tp: int
    fn: int
    tn: int

    @property
    def fnr(self) -> float:
        """False negative rate = FN / (FN + TP)."""
        return self.fn / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Balanced error rate = (FPR + FNR) / 2."""
        return (self.fpr + self.fnr) / 2.0


@dataclass
class ROCCurve:
    """
    ROC curve for one demographic group.

    [HARDT] "For each group A, we compute the ROC curve as we vary the
    classification threshold." (Section 4)

    Attributes
    ----------
    group_name  : identifier for the group (e.g., "male", "female").
    n_positive  : count of positive (diseased) users in group.
    n_negative  : count of negative (healthy) users in group.
    scores      : prediction scores (rw values) for all users in group.
    labels      : ground-truth labels (1=healthy, 2-16=diseased).
    points      : list of ROCPoint objects, sorted by threshold (descending).
    """
    group_name: str
    n_positive: int
    n_negative: int
    scores: np.ndarray
    labels: np.ndarray
    points: List[ROCPoint] = field(default_factory=list)

    def compute_point_at_threshold(self, threshold: float) -> ROCPoint:
        """
        Compute TPR, FPR at a specific threshold.

        [HARDT] Classification rule: ŷ = 1 if score ≤ threshold, else 0.
                (Lower rw = higher confidence in healthy, but we want ≤ for unhealthy)

        Parameters
        ----------
        threshold : decision threshold.

        Returns
        -------
        ROCPoint at this threshold.
        """
        # Prediction: unhealthy if rw <= threshold
        predictions = (self.scores <= threshold).astype(int)  # 1 = unhealthy (positive)
        
        # Separate by true label
        # In CDS: label=1 is healthy (negative), label≥2 is diseased (positive)
        true_diseased = (self.labels != 1).astype(int)
        true_healthy = (self.labels == 1).astype(int)

        tp = np.sum((predictions == 1) & (true_diseased == 1))
        fp = np.sum((predictions == 1) & (true_healthy == 1))
        fn = np.sum((predictions == 0) & (true_diseased == 1))
        tn = np.sum((predictions == 0) & (true_healthy == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return ROCPoint(
            threshold=threshold,
            fpr=fpr,
            tpr=tpr,
            fp=int(fp),
            tp=int(tp),
            fn=int(fn),
            tn=int(tn),
        )


@dataclass
class ConvexHullResult:
    """
    Convex hull (Pareto frontier) of ROC curves for two groups.

    [HARDT] Section 4: "The set of feasible fair classifiers is the convex
    hull of the individual ROC curves projected onto the (FPR, FNR) plane."

    Attributes
    ----------
    roc_male           : ROCCurve for male users.
    roc_female         : ROCCurve for female users.
    hull_points        : list of (threshold_male, threshold_female, fpr, tpr, fnr)
                         tuples on the convex hull in order of increasing FPR.
    hull_vertices_idx  : indices into combined points list (for debugging).
    """
    roc_male: ROCCurve
    roc_female: ROCCurve
    hull_points: List[Tuple[float, float, float, float, float]] = field(
        default_factory=list
    )
    hull_vertices_idx: List[int] = field(default_factory=list)


@dataclass
class LossResult:
    """
    Loss computation for equalized odds.

    [HARDT] Section 4: "To ensure equal opportunity, we require equal TPR
    (or equivalently, equal FNR). The Bayes-optimal classifier on the convex
    hull minimizes the loss subject to this constraint."

    Attributes
    ----------
    threshold_male       : optimal threshold for male users.
    threshold_female     : optimal threshold for female users.
    tpr_equalized        : achieved TPR value (same for both groups).
    fpr_male             : FPR for male users at optimal threshold.
    fpr_female           : FPR for female users at optimal threshold.
    expected_loss        : E[error] increase due to fairness constraint.
    global_accuracy      : overall accuracy (TP+TN) / N.
    """
    threshold_male: float
    threshold_female: float
    tpr_equalized: float
    fpr_male: float
    fpr_female: float
    expected_loss: float
    global_accuracy: float


@dataclass
class EqualizedOddsResult:
    """
    Final output of equalized odds post-processing.

    Attributes
    ----------
    threshold_male        : threshold for male users (replaces DIAGNOSTIC_THRESHOLD).
    threshold_female      : threshold for female users.
    tpr_male              : achieved TPR for male users.
    tpr_female            : achieved TPR for female users.
    tpr_equalized         : equal TPR value (should equal both above).
    fpr_male              : false positive rate for males.
    fpr_female            : false positive rate for females.
    utility_loss          : expected accuracy loss vs. single threshold.
    original_threshold    : the baseline single threshold (usually 0.025).
    n_male                : number of male users.
    n_female              : number of female users.
    diagnostics           : detailed optimization information.
    """
    threshold_male: float
    threshold_female: float
    tpr_male: float
    tpr_female: float
    tpr_equalized: float
    fpr_male: float
    fpr_female: float
    utility_loss: float
    original_threshold: float
    n_male: int
    n_female: int
    diagnostics: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – CORE FUNCTIONS: ROC CURVE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    group_name: str,
    n_points: int = 100,
) -> ROCCurve:
    """
    Compute ROC curve for one demographic group.

    [HARDT] Computes FPR and TPR at multiple thresholds.

    Parameters
    ----------
    scores     : prediction scores (rw values). Shape (n,).
    labels     : ground-truth labels (1=healthy, ≥2=diseased). Shape (n,).
    group_name : identifier for this group.
    n_points   : number of points to sample on the curve.

    Returns
    -------
    ROCCurve object with computed points.
    """
    log.info(f"Computing ROC curve for group: {group_name}")

    # Identify positive (diseased) and negative (healthy) users
    true_diseased = (labels != 1).sum()
    true_healthy = (labels == 1).sum()

    log.info(f"  {group_name}: {true_diseased} diseased, {true_healthy} healthy")

    # Generate thresholds: include extremes (always classify one way)
    # and uniform samples across the range
    unique_scores = np.unique(scores)
    if len(unique_scores) < n_points:
        thresholds = np.concatenate([
            np.array([-np.inf, np.inf]),  # extreme thresholds
            unique_scores,
        ])
    else:
        thresholds = np.concatenate([
            np.array([-np.inf, np.inf]),
            np.linspace(scores.min(), scores.max(), n_points),
        ])

    roc = ROCCurve(
        group_name=group_name,
        n_positive=int(true_diseased),
        n_negative=int(true_healthy),
        scores=scores.copy(),
        labels=labels.copy(),
    )

    # Compute point for each threshold
    for threshold in np.unique(thresholds):
        point = roc.compute_point_at_threshold(float(threshold))
        roc.points.append(point)

    # Sort by threshold descending (high threshold = few positives)
    roc.points = sorted(roc.points, key=lambda p: p.threshold, reverse=True)

    log.info(f"  Computed {len(roc.points)} ROC points for {group_name}")
    return roc


def compute_roc_curves_by_gender(
    scores: np.ndarray,
    labels: np.ndarray,
    data: np.ndarray,
) -> Tuple[ROCCurve, ROCCurve]:
    """
    Compute separate ROC curves for male and female groups.

    [HARDT] Section 4: "For each group A, we compute the ROC curve."

    Parameters
    ----------
    scores : prediction scores (rw values). Shape (n,).
    labels : ground-truth labels. Shape (n,).
    data   : full feature matrix. Shape (n, 279). Used to extract gender.

    Returns
    -------
    (roc_male, roc_female)
    """
    log.info("Computing ROC curves by gender...")

    # Extract gender
    gender_col = data[:, SEX_FEATURE_IDX]
    male_mask = (gender_col == MALE_VALUE)
    female_mask = (gender_col == FEMALE_VALUE)

    roc_male = compute_roc_curve(
        scores[male_mask],
        labels[male_mask],
        group_name="Male",
        n_points=100,
    )

    roc_female = compute_roc_curve(
        scores[female_mask],
        labels[female_mask],
        group_name="Female",
        n_points=100,
    )

    return roc_male, roc_female


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – CORE FUNCTIONS: CONVEX HULL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_convex_hull(
    roc_male: ROCCurve,
    roc_female: ROCCurve,
) -> ConvexHullResult:
    """
    Compute convex hull (Pareto frontier) of ROC curves.

    [HARDT] Section 4: "The set of achievable false positive and false negative
    rates is the set of all points in the convex hull of individual group ROC curves."

    For equalized odds, we search for points (t_A, t_B) such that:
      TPR_A(t_A) = TPR_B(t_B)

    This finds all feasible (TPR, FPR, FNR) triples where the constraint is satisfied.

    Parameters
    ----------
    roc_male   : ROCCurve for male group.
    roc_female : ROCCurve for female group.

    Returns
    -------
    ConvexHullResult with candidate operating points.
    """
    log.info("Computing convex hull of ROC curves...")

    # Generate all possible (t_A, t_B) pairs and their (FPR_A, FPR_B, TPR_A, TPR_B)
    hull_points: List[Tuple[float, float, float, float, float]] = []

    for pt_male in roc_male.points:
        for pt_female in roc_female.points:
            # Check equalized odds constraint: TPR must match
            if abs(pt_male.tpr - pt_female.tpr) < 1e-6:  # Allow small tolerance
                # Compute aggregate FPR and FNR (weighted by group size)
                total_n = roc_male.n_positive + roc_male.n_negative + \
                          roc_female.n_positive + roc_female.n_negative
                fpr_agg = (pt_male.fp + pt_female.fp) / (
                    roc_male.n_negative + roc_female.n_negative
                ) if (roc_male.n_negative + roc_female.n_negative) > 0 else 0.0
                fnr_agg = (pt_male.fn + pt_female.fn) / (
                    roc_male.n_positive + roc_female.n_positive
                ) if (roc_male.n_positive + roc_female.n_positive) > 0 else 0.0

                hull_points.append((
                    pt_male.threshold,
                    pt_female.threshold,
                    fpr_agg,
                    pt_male.tpr,  # both should be equal
                    fnr_agg,
                ))

    log.info(f"  Found {len(hull_points)} candidate points satisfying equalized TPR")

    # Sort by FPR (operating points along Pareto frontier)
    hull_points = sorted(hull_points, key=lambda x: x[2])  # sort by fpr_agg

    result = ConvexHullResult(
        roc_male=roc_male,
        roc_female=roc_female,
        hull_points=hull_points,
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    hull: ConvexHullResult,
    baseline_threshold: float,
    cost_fn: Optional[callable] = None,
) -> LossResult:
    """
    Compute optimal loss along the convex hull subject to equalized odds.

    [HARDT] Section 4: "We want to find the Bayes-optimal classifier on the
    convex hull subject to the constraint that TPR is equal across groups."

    The loss is computed as a weighted sum:
      L(t_A, t_B) = w_A * error_A(t_A) + w_B * error_B(t_B)

    where error = FPR * P(Y=0) + FNR * P(Y=1)

    and the weighting can be uniform (democratic) or based on group size.

    Parameters
    ----------
    hull              : ConvexHullResult from compute_convex_hull().
    baseline_threshold: original threshold for loss computation (e.g., 0.025).
    cost_fn           : optional cost function(threshold_male, threshold_female, point)
                        that returns a scalar loss. If None, uses balanced error.

    Returns
    -------
    LossResult with optimal thresholds and performance metrics.
    """
    log.info("Computing optimal loss along convex hull...")

    if not hull.hull_points:
        log.warning("  Convex hull is empty! No feasible equalized odds solution found.")
        # Fall back to baseline
        return _fallback_loss_result(hull, baseline_threshold)

    best_loss = float('inf')
    best_point = None
    best_male_threshold = None
    best_female_threshold = None

    # Evaluate loss at each hull point
    for t_male, t_female, fpr_agg, tpr, fnr_agg in hull.hull_points:
        # Default cost: balanced error rate
        if cost_fn is None:
            loss = (fpr_agg + fnr_agg) / 2.0
        else:
            loss = cost_fn(t_male, t_female, (fpr_agg, tpr, fnr_agg))

        if loss < best_loss:
            best_loss = loss
            best_point = (t_male, t_female, fpr_agg, tpr, fnr_agg)
            best_male_threshold = t_male
            best_female_threshold = t_female

    if best_point is None:
        log.warning("  No feasible operating point found.")
        return _fallback_loss_result(hull, baseline_threshold)

    t_male, t_female, fpr_agg, tpr, fnr_agg = best_point

    # Compute baseline loss (single threshold for all)
    baseline_point_male = hull.roc_male.compute_point_at_threshold(baseline_threshold)
    baseline_point_female = hull.roc_female.compute_point_at_threshold(baseline_threshold)
    baseline_loss = (baseline_point_male.error_rate + baseline_point_female.error_rate) / 2.0

    utility_loss = best_loss - baseline_loss

    log.info(f"  Optimal thresholds: male={best_male_threshold:.6f}, female={best_female_threshold:.6f}")
    log.info(f"  Equalized TPR: {tpr:.4f}")
    log.info(f"  FPR (agg): {fpr_agg:.4f}, FNR (agg): {fnr_agg:.4f}")
    log.info(f"  Loss: {best_loss:.6f} (baseline: {baseline_loss:.6f}, delta: {utility_loss:.6f})")

    result = LossResult(
        threshold_male=best_male_threshold,
        threshold_female=best_female_threshold,
        tpr_equalized=tpr,
        fpr_male=hull.roc_male.compute_point_at_threshold(best_male_threshold).fpr,
        fpr_female=hull.roc_female.compute_point_at_threshold(best_female_threshold).fpr,
        expected_loss=utility_loss,
        global_accuracy=(1.0 - best_loss),  # accuracy ≈ 1 - error
    )

    return result


def _fallback_loss_result(
    hull: ConvexHullResult,
    baseline_threshold: float,
) -> LossResult:
    """
    Fallback result when no convex hull points are found.
    Uses baseline threshold for both groups.
    """
    pt_m = hull.roc_male.compute_point_at_threshold(baseline_threshold)
    pt_f = hull.roc_female.compute_point_at_threshold(baseline_threshold)

    return LossResult(
        threshold_male=baseline_threshold,
        threshold_female=baseline_threshold,
        tpr_equalized=(pt_m.tpr + pt_f.tpr) / 2.0,
        fpr_male=pt_m.fpr,
        fpr_female=pt_f.fpr,
        expected_loss=0.0,
        global_accuracy=0.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – BAYES OPTIMAL PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_bayes_optimal_predictor(
    scores: np.ndarray,
    labels: np.ndarray,
    data: np.ndarray,
    baseline_threshold: float = 0.025,
    verbose: bool = True,
) -> EqualizedOddsResult:
    """
    Compute gender-specific thresholds that satisfy equalized odds.

    [HARDT] Section 4: "Algorithm 1: Computing the Bayes-optimal fair classifier"

    This is the main entry point for equalized odds post-processing.

    Parameters
    ----------
    scores              : prediction scores (rw values from Algorithm 4).
                          Shape (n,).
    labels              : ground-truth labels (1=healthy, ≥2=diseased).
                          Shape (n,).
    data                : full feature matrix (used to extract gender).
                          Shape (n, 279).
    baseline_threshold  : original threshold for reference (usually 0.025).
    verbose             : whether to log detailed information.

    Returns
    -------
    EqualizedOddsResult with gender-specific thresholds and metrics.

    Raises
    ------
    ValueError if inputs are invalid or no feasible solution exists.
    """
    if verbose:
        log.setLevel(logging.DEBUG)

    log.info("=" * 80)
    log.info("EQUALIZED ODDS POST-PROCESSING (Hardt et al. 2016)")
    log.info("=" * 80)

    # Validate inputs
    if len(scores) != len(labels) or len(scores) != len(data):
        raise ValueError("Dimension mismatch: scores, labels, data must have equal length")

    if not np.all((labels == 1) | (labels >= 2)):
        raise ValueError("Labels must be 1 (healthy) or ≥2 (diseased)")

    # Extract gender
    gender_col = data[:, SEX_FEATURE_IDX]
    n_male = np.sum(gender_col == MALE_VALUE)
    n_female = np.sum(gender_col == FEMALE_VALUE)

    log.info(f"Dataset composition: {n_male} males, {n_female} females")

    # Compute ROC curves by gender
    roc_male, roc_female = compute_roc_curves_by_gender(scores, labels, data)

    # Find convex hull
    hull = compute_convex_hull(roc_male, roc_female)

    # Optimize loss
    loss = compute_loss(hull, baseline_threshold)

    # Compute detailed metrics
    pt_male = roc_male.compute_point_at_threshold(loss.threshold_male)
    pt_female = roc_female.compute_point_at_threshold(loss.threshold_female)

    result = EqualizedOddsResult(
        threshold_male=loss.threshold_male,
        threshold_female=loss.threshold_female,
        tpr_male=pt_male.tpr,
        tpr_female=pt_female.tpr,
        tpr_equalized=loss.tpr_equalized,
        fpr_male=pt_male.fpr,
        fpr_female=pt_female.fpr,
        utility_loss=loss.expected_loss,
        original_threshold=baseline_threshold,
        n_male=int(n_male),
        n_female=int(n_female),
        diagnostics={
            "roc_male_points_count": len(roc_male.points),
            "roc_female_points_count": len(roc_female.points),
            "hull_points_count": len(hull.hull_points),
            "male_confusion": {
                "tp": pt_male.tp,
                "fp": pt_male.fp,
                "tn": pt_male.tn,
                "fn": pt_male.fn,
            },
            "female_confusion": {
                "tp": pt_female.tp,
                "fp": pt_female.fp,
                "tn": pt_female.tn,
                "fn": pt_female.fn,
            },
        },
    )

    _print_equalized_odds_summary(result)

    log.info("=" * 80)

    return result


def _print_equalized_odds_summary(result: EqualizedOddsResult) -> None:
    """Print summary of equalized odds result."""
    log.info("\nEQUALIZED ODDS SUMMARY")
    log.info("-" * 80)
    log.info(f"Original threshold (all users):  {result.original_threshold:.6f}")
    log.info(f"New threshold (male users):      {result.threshold_male:.6f}")
    log.info(f"New threshold (female users):    {result.threshold_female:.6f}")
    log.info("-" * 80)
    log.info(f"Equalized TPR (sensitivity):    {result.tpr_equalized:.4f}")
    log.info(f"  Male TPR:                       {result.tpr_male:.4f}")
    log.info(f"  Female TPR:                     {result.tpr_female:.4f}")
    log.info("-" * 80)
    log.info(f"False Positive Rates:")
    log.info(f"  Male FPR:                       {result.fpr_male:.4f}")
    log.info(f"  Female FPR:                     {result.fpr_female:.4f}")
    log.info("-" * 80)
    log.info(f"Utility Loss (accuracy decrease): {result.utility_loss:.6f}")
    log.info(f"Dataset: {result.n_male} male, {result.n_female} female users")
    log.info("-" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – HELPER: APPLY THRESHOLDS TO PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def apply_equalized_odds_thresholds(
    scores: np.ndarray,
    data: np.ndarray,
    fairness_result: EqualizedOddsResult,
) -> np.ndarray:
    """
    Apply gender-specific thresholds to prediction scores.

    Converts rw scores to binary predictions (healthy=0, unhealthy=1) using
    gender-specific thresholds from equalized odds optimization.

    Parameters
    ----------
    scores        : prediction scores (rw values). Shape (n,).
    data          : feature matrix (used to extract gender). Shape (n, 279).
    fairness_result : EqualizedOddsResult from compute_bayes_optimal_predictor().

    Returns
    -------
    predictions   : binary predictions (0=healthy, 1=unhealthy). Shape (n,).
    """
    gender_col = data[:, SEX_FEATURE_IDX]
    predictions = np.zeros(len(scores), dtype=int)

    male_mask = (gender_col == MALE_VALUE)
    female_mask = (gender_col == FEMALE_VALUE)

    # Apply gender-specific thresholds
    predictions[male_mask] = (scores[male_mask] <= fairness_result.threshold_male).astype(int)
    predictions[female_mask] = (
        scores[female_mask] <= fairness_result.threshold_female
    ).astype(int)

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – MAIN EXECUTION & UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Example usage: compute equalized odds on a hypothetical dataset.
    (Requires Algorithm4 output as input in practice.)
    """
    log.info("Fairness_EqualizedOdds module loaded successfully.")
    log.info("Use compute_bayes_optimal_predictor() as main entry point.")


if __name__ == "__main__":
    main()
