"""
================================================================================
Fairness_EqualizedOdds.py – Equalized Odds Post-Processing for CDS
================================================================================

PURPOSE
-------
Implements equalized odds post-processing following Hardt et al. (2016):
"Equality of Opportunity in Supervised Learning" (arXiv:1610.02413).

Equalized Odds (Definition 2.1) requires that BOTH the True Positive Rate (TPR)
AND the False Positive Rate (FPR) are equal across protected groups:

    Pr{Ŷ=1 | A=a, Y=y} is the same for all groups a, for BOTH y=0 and y=1.

This means:
    TPR_male = TPR_female   (equal true positive rates)
    FPR_male = FPR_female   (equal false positive rates)

METHODOLOGY (Section 4.2 of Hardt et al.)
------------------------------------------
  1. Compute per-group ROC curves C_a(t) in (FPR, TPR) space.
  2. Compute the convex hull D_a = convhull{C_a(t)} for each group.
  3. Find the intersection region ∩_a D_a where both groups can achieve
     the SAME (FPR, TPR) point.
  4. Optimize over the upper-left boundary of this intersection to find
     the operating point that minimizes expected loss.
  5. For each group, realize the chosen (FPR, TPR) point using a RANDOMIZED
     classifier — a mixture of two threshold predictors. This is necessary
     because the chosen point may not lie exactly on either group's ROC curve.

RANDOMIZED CLASSIFIERS (Section 4.2)
--------------------------------------
  For each group a, the derived predictor is:
      Ŷ = I{R > T_a}
  where T_a is a random threshold:
      T_a = t_a   with probability p_a
      T_a = t̄_a  with probability 1 - p_a
  Equivalently: if R < t_a → predict 0; if R > t̄_a → predict 1;
  if t_a < R < t̄_a → predict 1 with probability p_a.

PAPER FIDELITY TAGS
-------------------
  [HARDT]  – directly from Hardt et al. (2016) framework
  [INFER]  – required interpretation not explicitly in paper
  [ENGR]   – engineering choice with justification

================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

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

    [HARDT] Section 4.2: C_a(t) = (Pr{R>t|A=a,Y=0}, Pr{R>t|A=a,Y=1})
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
        return self.fn / (self.fn + self.tp) if (self.fn + self.tp) > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return (self.fpr + self.fnr) / 2.0


@dataclass
class ROCCurve:
    """
    ROC curve for one demographic group.

    [HARDT] Section 4.2: "For each group A, we compute the ROC curve
    as we vary the classification threshold."
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

        [HARDT] Classification rule for CDS:
            rw > threshold  =>  UNHEALTHY (positive prediction)
            rw <= threshold =>  HEALTHY   (negative prediction)
        """
        predictions = (self.scores > threshold).astype(int)

        true_diseased = (self.labels != 1).astype(int)
        true_healthy = (self.labels == 1).astype(int)

        tp = np.sum((predictions == 1) & (true_diseased == 1))
        fp = np.sum((predictions == 1) & (true_healthy == 1))
        fn = np.sum((predictions == 0) & (true_diseased == 1))
        tn = np.sum((predictions == 0) & (true_healthy == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return ROCPoint(
            threshold=threshold, fpr=fpr, tpr=tpr,
            fp=int(fp), tp=int(tp), fn=int(fn), tn=int(tn),
        )


@dataclass
class RandomizedThreshold:
    """
    [HARDT] Section 4.2: A randomized threshold predictor for one group.

    The predictor uses threshold t_lo with probability p and t_hi with
    probability (1-p). Equivalently:
        R < t_lo  → predict 0
        R > t_hi  → predict 1
        t_lo <= R <= t_hi → predict 1 with probability p
    """
    t_lo: float
    t_hi: float
    p: float

    @property
    def is_deterministic(self) -> bool:
        return abs(self.t_lo - self.t_hi) < 1e-12 or self.p < 1e-12 or self.p > 1 - 1e-12


@dataclass
class EqualizedOddsResult:
    """
    Final output of equalized odds post-processing.

    [HARDT] The result specifies per-group randomized thresholds that achieve
    equal (FPR, TPR) across groups.
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
    randomized_threshold_male: Optional[RandomizedThreshold] = None
    randomized_threshold_female: Optional[RandomizedThreshold] = None
    diagnostics: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – ROC CURVE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    group_name: str,
    n_points: int = 100,
) -> ROCCurve:
    """
    [HARDT] Compute ROC curve C_a(t) for one group by varying threshold.
    """
    log.info(f"Computing ROC curve for group: {group_name}")

    true_diseased = (labels != 1).sum()
    true_healthy = (labels == 1).sum()

    log.info(f"  {group_name}: {true_diseased} diseased, {true_healthy} healthy")

    unique_scores = np.unique(scores)
    if len(unique_scores) < n_points:
        thresholds = np.concatenate([
            np.array([-np.inf, np.inf]),
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

    for threshold in np.unique(thresholds):
        point = roc.compute_point_at_threshold(float(threshold))
        roc.points.append(point)

    roc.points = sorted(roc.points, key=lambda p: p.threshold, reverse=True)

    log.info(f"  Computed {len(roc.points)} ROC points for {group_name}")
    return roc


def compute_roc_curves_by_gender(
    scores: np.ndarray,
    labels: np.ndarray,
    data: np.ndarray,
    n_points: int = 100,
) -> Tuple[ROCCurve, ROCCurve]:
    """
    [HARDT] Section 4.2: Compute separate ROC curves for each group.
    """
    log.info("Computing ROC curves by gender...")

    gender_col = data[:, SEX_FEATURE_IDX]
    male_mask = (gender_col == MALE_VALUE)
    female_mask = (gender_col == FEMALE_VALUE)

    roc_male = compute_roc_curve(
        scores[male_mask], labels[male_mask],
        group_name="Male", n_points=n_points,
    )
    roc_female = compute_roc_curve(
        scores[female_mask], labels[female_mask],
        group_name="Female", n_points=n_points,
    )

    return roc_male, roc_female


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – CONVEX HULL OF ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────

def _roc_to_convex_hull_points(roc: ROCCurve) -> np.ndarray:
    """
    [HARDT] Equation 4.4: D_a = convhull{C_a(t): t in [0,1]}

    Extract the (FPR, TPR) points from the ROC curve and compute the upper
    boundary of their convex hull (points above the main diagonal).

    Returns array of shape (K, 2) with columns [fpr, tpr].
    """
    pts = np.array([[p.fpr, p.tpr] for p in roc.points])

    pts = np.vstack([pts, [0.0, 0.0], [1.0, 1.0]])

    pts = np.unique(pts, axis=0)

    above_diag = pts[:, 1] >= pts[:, 0] - 1e-9
    pts = pts[above_diag]

    return pts


def _upper_boundary(hull_pts: np.ndarray) -> np.ndarray:
    """
    Extract the upper-left boundary of a convex hull in (FPR, TPR) space.

    [HARDT] The optimal equalized odds predictor lies on the upper-left
    boundary of the intersection region (Figure 2, middle panel).

    Returns points sorted by FPR ascending, forming a concave curve from
    (0,0) toward (1,1).
    """
    if len(hull_pts) < 3:
        return hull_pts[hull_pts[:, 1].argsort()[::-1]]

    try:
        ch = ConvexHull(hull_pts)
        vertices = hull_pts[ch.vertices]
    except Exception:
        return hull_pts[hull_pts[:, 0].argsort()]

    sorted_v = vertices[vertices[:, 0].argsort()]

    upper = [sorted_v[0]]
    for pt in sorted_v[1:]:
        if pt[1] >= upper[-1][1] - 1e-9 or pt[0] >= upper[-1][0]:
            upper.append(pt)

    upper = np.array(upper)

    boundary = []
    best_tpr = -1.0
    for pt in upper[upper[:, 0].argsort()]:
        if pt[1] > best_tpr - 1e-9:
            boundary.append(pt)
            best_tpr = max(best_tpr, pt[1])

    return np.array(boundary)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – INTERPOLATION ON ROC CONVEX HULL
# ─────────────────────────────────────────────────────────────────────────────

def _interpolate_on_hull(hull_pts: np.ndarray, target_fpr: float) -> Optional[float]:
    """
    Interpolate the upper boundary of a convex hull to find TPR at a given FPR.
    """
    sorted_pts = hull_pts[hull_pts[:, 0].argsort()]

    if target_fpr < sorted_pts[0, 0] - 1e-9 or target_fpr > sorted_pts[-1, 0] + 1e-9:
        return None

    for pt in sorted_pts:
        if abs(pt[0] - target_fpr) < 1e-9:
            return float(pt[1])

    for i in range(len(sorted_pts) - 1):
        if sorted_pts[i, 0] <= target_fpr <= sorted_pts[i + 1, 0]:
            alpha = (target_fpr - sorted_pts[i, 0]) / (sorted_pts[i + 1, 0] - sorted_pts[i, 0])
            return float(sorted_pts[i, 1] + alpha * (sorted_pts[i + 1, 1] - sorted_pts[i, 1]))

    return None


def _find_feasible_region(
    hull_male: np.ndarray,
    hull_female: np.ndarray,
) -> np.ndarray:
    """
    [HARDT] Section 4.2: Find the intersection ∩_a D_a of the convex hulls.

    The feasible set of (FPR, TPR) for equalized odds predictors is the
    intersection of the areas under the A-conditional ROC curves (and above
    the main diagonal).

    Returns array of feasible (fpr, tpr) points on the upper-left boundary
    of the intersection.
    """
    fpr_min = max(hull_male[:, 0].min(), hull_female[:, 0].min())
    fpr_max = min(hull_male[:, 0].max(), hull_female[:, 0].max())

    if fpr_min > fpr_max + 1e-9:
        return np.array([])

    n_sample = 500
    fpr_candidates = np.linspace(fpr_min, fpr_max, n_sample)

    all_fpr_vals = np.concatenate([
        hull_male[:, 0], hull_female[:, 0], fpr_candidates
    ])
    all_fpr_vals = np.unique(all_fpr_vals)
    all_fpr_vals = all_fpr_vals[(all_fpr_vals >= fpr_min - 1e-9) & (all_fpr_vals <= fpr_max + 1e-9)]

    feasible = []
    for fpr in all_fpr_vals:
        tpr_m = _interpolate_on_hull(hull_male, fpr)
        tpr_f = _interpolate_on_hull(hull_female, fpr)
        if tpr_m is not None and tpr_f is not None:
            tpr = min(tpr_m, tpr_f)
            if tpr >= fpr - 1e-9:
                feasible.append([fpr, tpr])

    if not feasible:
        return np.array([])

    return np.array(feasible)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – REALIZE OPERATING POINT VIA RANDOMIZED THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

def _realize_operating_point(
    roc: ROCCurve,
    target_fpr: float,
    target_tpr: float,
) -> RandomizedThreshold:
    """
    [HARDT] Section 4.2: Realize a target (FPR, TPR) point on one group's
    convex hull using a randomized mixture of two threshold predictors.

    Any point in D_a = convhull{C_a(t)} can be achieved as a convex
    combination of two points on the ROC curve. We find the two ROC points
    that bracket the target and compute the mixing probability.
    """
    pts = sorted(roc.points, key=lambda p: (p.fpr, p.tpr))

    unique_pts = []
    seen = set()
    for p in pts:
        key = (round(p.fpr, 10), round(p.tpr, 10))
        if key not in seen:
            seen.add(key)
            unique_pts.append(p)

    best_dist = float('inf')
    best_threshold = RandomizedThreshold(t_lo=unique_pts[0].threshold, t_hi=unique_pts[0].threshold, p=1.0)

    for p in unique_pts:
        dist = math.sqrt((p.fpr - target_fpr) ** 2 + (p.tpr - target_tpr) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_threshold = RandomizedThreshold(t_lo=p.threshold, t_hi=p.threshold, p=1.0)

    if best_dist < 1e-6:
        return best_threshold

    for i in range(len(unique_pts)):
        for j in range(i + 1, len(unique_pts)):
            p1 = unique_pts[i]
            p2 = unique_pts[j]

            dfpr = p2.fpr - p1.fpr
            dtpr = p2.tpr - p1.tpr

            if abs(dfpr) < 1e-12 and abs(dtpr) < 1e-12:
                continue

            if abs(dfpr) > abs(dtpr):
                alpha = (target_fpr - p1.fpr) / dfpr
            else:
                alpha = (target_tpr - p1.tpr) / dtpr

            if alpha < -1e-9 or alpha > 1 + 1e-9:
                continue

            alpha = np.clip(alpha, 0, 1)
            achieved_fpr = p1.fpr + alpha * dfpr
            achieved_tpr = p1.tpr + alpha * dtpr
            dist = math.sqrt((achieved_fpr - target_fpr) ** 2 + (achieved_tpr - target_tpr) ** 2)

            if dist < best_dist:
                best_dist = dist
                if p1.threshold > p2.threshold:
                    best_threshold = RandomizedThreshold(
                        t_lo=p2.threshold, t_hi=p1.threshold, p=1.0 - alpha,
                    )
                else:
                    best_threshold = RandomizedThreshold(
                        t_lo=p1.threshold, t_hi=p2.threshold, p=alpha,
                    )

    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – LOSS OPTIMIZATION OVER FEASIBLE REGION
# ─────────────────────────────────────────────────────────────────────────────

def _optimize_over_feasible_region(
    feasible: np.ndarray,
    roc_male: ROCCurve,
    roc_female: ROCCurve,
    baseline_threshold: float,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> Tuple[float, float, float]:
    """
    [HARDT] Equation 4.5: Find the point on the upper-left boundary of the
    feasible region that minimizes expected loss.

        min  gamma_0 * l(1,0) + (1 - gamma_1) * l(0,1)
        s.t. gamma in ∩_a D_a

    where gamma = (FPR, TPR) and l(1,0), l(0,1) are the costs of false
    positives and false negatives respectively.

    Returns (best_fpr, best_tpr, best_loss).
    """
    best_loss = float('inf')
    best_fpr = 0.0
    best_tpr = 0.0

    for fpr, tpr in feasible:
        fnr = 1.0 - tpr
        loss = cost_fp * fpr + cost_fn * fnr
        if loss < best_loss:
            best_loss = loss
            best_fpr = fpr
            best_tpr = tpr

    return best_fpr, best_tpr, best_loss


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – LINEAR PROGRAM FORMULATION (BINARY PREDICTOR CASE)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_equalized_odds_lp(
    roc_male: ROCCurve,
    roc_female: ROCCurve,
    baseline_threshold: float,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> Optional[Tuple[float, float]]:
    """
    [HARDT] Proposition 4.4: Solve the equalized odds optimization as a
    linear program when the convex hull intersection approach does not yield
    a clean solution.

    This is a fallback that searches for the best (FPR, TPR) operating point
    by directly scanning feasible points where both groups' convex hulls overlap.

    Returns (optimal_fpr, optimal_tpr) or None if infeasible.
    """
    hull_m = _roc_to_convex_hull_points(roc_male)
    hull_f = _roc_to_convex_hull_points(roc_female)

    feasible = _find_feasible_region(hull_m, hull_f)
    if len(feasible) == 0:
        return None

    best_fpr, best_tpr, _ = _optimize_over_feasible_region(
        feasible, roc_male, roc_female, baseline_threshold, cost_fp, cost_fn
    )
    return (best_fpr, best_tpr)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def compute_bayes_optimal_predictor(
    scores: np.ndarray,
    labels: np.ndarray,
    data: np.ndarray,
    baseline_threshold: float = 0.025,
    verbose: bool = True,
    n_roc_points: int = 100,
) -> EqualizedOddsResult:
    """
    Compute gender-specific (randomized) thresholds satisfying equalized odds.

    [HARDT] Section 4.2: Derives an equalized odds predictor from a score
    function R by finding the optimal operating point in the intersection
    of the per-group ROC convex hulls, then realizing that point via
    randomized threshold classifiers for each group.

    Parameters
    ----------
    scores              : prediction scores (rw values from Algorithm 4).
    labels              : ground-truth labels (1=healthy, >=2=diseased).
    data                : full feature matrix (used to extract gender).
    baseline_threshold  : original threshold for reference.
    verbose             : whether to log detailed information.
    n_roc_points        : number of points to sample on each ROC curve.

    Returns
    -------
    EqualizedOddsResult with gender-specific thresholds and metrics.
    """
    if verbose:
        log.setLevel(logging.DEBUG)

    log.info("=" * 80)
    log.info("EQUALIZED ODDS POST-PROCESSING (Hardt et al. 2016)")
    log.info("  Constraint: BOTH TPR and FPR equalized across groups")
    log.info("  Method: Convex hull intersection + randomized thresholds")
    log.info("=" * 80)

    if len(scores) != len(labels) or len(scores) != len(data):
        raise ValueError("Dimension mismatch: scores, labels, data must have equal length")

    if not np.all((labels == 1) | (labels >= 2)):
        raise ValueError("Labels must be 1 (healthy) or >=2 (diseased)")

    gender_col = data[:, SEX_FEATURE_IDX]
    n_male = int(np.sum(gender_col == MALE_VALUE))
    n_female = int(np.sum(gender_col == FEMALE_VALUE))

    log.info(f"Dataset composition: {n_male} males, {n_female} females")

    # Step 1: Compute per-group ROC curves
    roc_male, roc_female = compute_roc_curves_by_gender(
        scores, labels, data, n_points=n_roc_points
    )

    # Step 2: Compute convex hulls D_a for each group
    hull_m = _roc_to_convex_hull_points(roc_male)
    hull_f = _roc_to_convex_hull_points(roc_female)

    log.info(f"  Male convex hull: {len(hull_m)} points")
    log.info(f"  Female convex hull: {len(hull_f)} points")

    # Step 3: Find intersection of convex hulls
    feasible = _find_feasible_region(hull_m, hull_f)

    if len(feasible) == 0:
        log.warning("No feasible equalized odds solution found. Falling back to baseline.")
        return _fallback_result(roc_male, roc_female, baseline_threshold, n_male, n_female)

    log.info(f"  Feasible region: {len(feasible)} points")

    # Step 4: Optimize over feasible region
    best_fpr, best_tpr, best_loss = _optimize_over_feasible_region(
        feasible, roc_male, roc_female, baseline_threshold
    )

    log.info(f"  Optimal operating point: FPR={best_fpr:.4f}, TPR={best_tpr:.4f}")

    # Step 5: Realize the operating point via randomized thresholds for each group
    rand_thresh_male = _realize_operating_point(roc_male, best_fpr, best_tpr)
    rand_thresh_female = _realize_operating_point(roc_female, best_fpr, best_tpr)

    log.info(f"  Male randomized threshold: t_lo={rand_thresh_male.t_lo:.6f}, "
             f"t_hi={rand_thresh_male.t_hi:.6f}, p={rand_thresh_male.p:.4f}"
             f" (deterministic={rand_thresh_male.is_deterministic})")
    log.info(f"  Female randomized threshold: t_lo={rand_thresh_female.t_lo:.6f}, "
             f"t_hi={rand_thresh_female.t_hi:.6f}, p={rand_thresh_female.p:.4f}"
             f" (deterministic={rand_thresh_female.is_deterministic})")

    # Use the midpoint of each group's randomized threshold range as
    # the representative deterministic threshold (for backward compatibility)
    if rand_thresh_male.is_deterministic:
        threshold_male = rand_thresh_male.t_lo
    else:
        threshold_male = rand_thresh_male.t_lo * rand_thresh_male.p + \
                         rand_thresh_male.t_hi * (1.0 - rand_thresh_male.p)

    if rand_thresh_female.is_deterministic:
        threshold_female = rand_thresh_female.t_lo
    else:
        threshold_female = rand_thresh_female.t_lo * rand_thresh_female.p + \
                           rand_thresh_female.t_hi * (1.0 - rand_thresh_female.p)

    # Compute baseline loss for comparison
    baseline_pt_m = roc_male.compute_point_at_threshold(baseline_threshold)
    baseline_pt_f = roc_female.compute_point_at_threshold(baseline_threshold)
    baseline_loss = (baseline_pt_m.fpr + baseline_pt_m.fnr +
                     baseline_pt_f.fpr + baseline_pt_f.fnr) / 4.0
    utility_loss = best_loss / 2.0 - baseline_loss

    result = EqualizedOddsResult(
        threshold_male=threshold_male,
        threshold_female=threshold_female,
        tpr_male=best_tpr,
        tpr_female=best_tpr,
        tpr_equalized=best_tpr,
        fpr_male=best_fpr,
        fpr_female=best_fpr,
        utility_loss=utility_loss,
        original_threshold=baseline_threshold,
        n_male=n_male,
        n_female=n_female,
        randomized_threshold_male=rand_thresh_male,
        randomized_threshold_female=rand_thresh_female,
        diagnostics={
            "roc_male_points_count": len(roc_male.points),
            "roc_female_points_count": len(roc_female.points),
            "hull_male_points_count": len(hull_m),
            "hull_female_points_count": len(hull_f),
            "feasible_region_points_count": len(feasible),
            "optimal_fpr": best_fpr,
            "optimal_tpr": best_tpr,
            "randomized_male": not rand_thresh_male.is_deterministic,
            "randomized_female": not rand_thresh_female.is_deterministic,
        },
    )

    _print_equalized_odds_summary(result)

    log.info("=" * 80)

    return result


def _fallback_result(
    roc_male: ROCCurve,
    roc_female: ROCCurve,
    baseline_threshold: float,
    n_male: int,
    n_female: int,
) -> EqualizedOddsResult:
    """Fallback when no feasible equalized odds solution exists."""
    pt_m = roc_male.compute_point_at_threshold(baseline_threshold)
    pt_f = roc_female.compute_point_at_threshold(baseline_threshold)

    return EqualizedOddsResult(
        threshold_male=baseline_threshold,
        threshold_female=baseline_threshold,
        tpr_male=pt_m.tpr,
        tpr_female=pt_f.tpr,
        tpr_equalized=(pt_m.tpr + pt_f.tpr) / 2.0,
        fpr_male=pt_m.fpr,
        fpr_female=pt_f.fpr,
        utility_loss=0.0,
        original_threshold=baseline_threshold,
        n_male=n_male,
        n_female=n_female,
        diagnostics={"fallback": True},
    )


def _print_equalized_odds_summary(result: EqualizedOddsResult) -> None:
    """Print summary of equalized odds result."""
    log.info("\nEQUALIZED ODDS SUMMARY")
    log.info("-" * 80)
    log.info(f"Original threshold (all users):  {result.original_threshold:.6f}")
    log.info(f"New threshold (male users):      {result.threshold_male:.6f}")
    log.info(f"New threshold (female users):    {result.threshold_female:.6f}")
    log.info("-" * 80)
    log.info(f"Equalized TPR (sensitivity):     {result.tpr_equalized:.4f}")
    log.info(f"  Male TPR:                      {result.tpr_male:.4f}")
    log.info(f"  Female TPR:                    {result.tpr_female:.4f}")
    log.info(f"Equalized FPR:                   {result.fpr_male:.4f}")
    log.info(f"  Male FPR:                      {result.fpr_male:.4f}")
    log.info(f"  Female FPR:                    {result.fpr_female:.4f}")
    log.info("-" * 80)
    if result.randomized_threshold_male and not result.randomized_threshold_male.is_deterministic:
        rt = result.randomized_threshold_male
        log.info(f"Male: RANDOMIZED classifier (t_lo={rt.t_lo:.4f}, t_hi={rt.t_hi:.4f}, p={rt.p:.4f})")
    else:
        log.info(f"Male: deterministic threshold = {result.threshold_male:.6f}")
    if result.randomized_threshold_female and not result.randomized_threshold_female.is_deterministic:
        rt = result.randomized_threshold_female
        log.info(f"Female: RANDOMIZED classifier (t_lo={rt.t_lo:.4f}, t_hi={rt.t_hi:.4f}, p={rt.p:.4f})")
    else:
        log.info(f"Female: deterministic threshold = {result.threshold_female:.6f}")
    log.info("-" * 80)
    log.info(f"Utility Loss (accuracy change):  {result.utility_loss:.6f}")
    log.info(f"Dataset: {result.n_male} male, {result.n_female} female users")
    log.info("-" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – APPLY THRESHOLDS TO PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_randomized_threshold(
    scores: np.ndarray,
    rt: RandomizedThreshold,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    [HARDT] Section 4.2: Apply a randomized threshold predictor.

    For each score R:
        R > t_hi  → predict 1 (always)
        R < t_lo  → predict 0 (always)
        t_lo <= R <= t_hi → predict 1 with probability p
    """
    predictions = np.zeros(len(scores), dtype=int)

    above = scores > rt.t_hi
    predictions[above] = 1

    if abs(rt.t_lo - rt.t_hi) > 1e-12:
        between = (scores >= rt.t_lo) & (scores <= rt.t_hi)
        n_between = np.sum(between)
        if n_between > 0:
            coin_flips = rng.random(n_between) < rt.p
            predictions[between] = coin_flips.astype(int)
    else:
        at_threshold = np.abs(scores - rt.t_lo) < 1e-12
        n_at = np.sum(at_threshold)
        if n_at > 0:
            coin_flips = rng.random(n_at) < rt.p
            predictions[at_threshold] = coin_flips.astype(int)

    return predictions


def apply_equalized_odds_thresholds(
    scores: np.ndarray,
    data: np.ndarray,
    fairness_result: EqualizedOddsResult,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply equalized odds thresholds to prediction scores.

    [HARDT] Section 4.2: Uses randomized classifiers when the optimal
    operating point requires mixing between two threshold predictors.
    Falls back to deterministic thresholds when randomization is not needed.

    Parameters
    ----------
    scores          : prediction scores (rw values).
    data            : feature matrix (used to extract gender).
    fairness_result : EqualizedOddsResult from compute_bayes_optimal_predictor().
    seed            : optional random seed for reproducibility of randomized
                      classifiers. If None, uses fresh randomness.

    Returns
    -------
    predictions : binary predictions (0=healthy, 1=unhealthy).
    """
    rng = np.random.default_rng(seed)

    gender_col = data[:, SEX_FEATURE_IDX]
    predictions = np.zeros(len(scores), dtype=int)

    male_mask = (gender_col == MALE_VALUE)
    female_mask = (gender_col == FEMALE_VALUE)

    rt_male = fairness_result.randomized_threshold_male
    rt_female = fairness_result.randomized_threshold_female

    if rt_male is not None:
        predictions[male_mask] = _apply_randomized_threshold(
            scores[male_mask], rt_male, rng,
        )
    else:
        predictions[male_mask] = (
            scores[male_mask] > fairness_result.threshold_male
        ).astype(int)

    if rt_female is not None:
        predictions[female_mask] = _apply_randomized_threshold(
            scores[female_mask], rt_female, rng,
        )
    else:
        predictions[female_mask] = (
            scores[female_mask] > fairness_result.threshold_female
        ).astype(int)

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Fairness_EqualizedOdds module loaded successfully.")
    log.info("Use compute_bayes_optimal_predictor() as main entry point.")
    log.info("Implements full equalized odds (Hardt et al. 2016):")
    log.info("  - Constrains BOTH TPR and FPR to be equal across groups")
    log.info("  - Supports randomized classifiers (mixture of two thresholds)")


if __name__ == "__main__":
    main()
