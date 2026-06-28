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

    Pr{Y_hat=1 | A=a, Y=y} is the same for all groups a, for BOTH y=0 and y=1.

This means:
    TPR_male = TPR_female   (equal true positive rates)
    FPR_male = FPR_female   (equal false positive rates)

METHODOLOGY (Section 4.2 of Hardt et al.)
------------------------------------------
  1. Compute per-group ROC curves C_a(t) in (FPR, TPR) space.
  2. Compute the upper concave envelope of each ROC curve — this is the
     boundary of D_a = convhull{C_a(t)}, the set of achievable (FPR, TPR)
     points via randomized classifiers.
  3. Find the intersection region where both groups can achieve the SAME
     (FPR, TPR) point: the upper boundary is min(envelope_a, envelope_b).
  4. Optimize over this boundary using prevalence-weighted expected loss:
         L = P(Y=0) * FPR + P(Y=1) * (1 - TPR)
  5. For each group, realize the chosen (FPR, TPR) point using a RANDOMIZED
     classifier — a mixture of two threshold predictors.

RANDOMIZED CLASSIFIERS (Section 4.2)
--------------------------------------
  For each group a, the derived predictor is:
      Y_hat = I{R > T_a}
  where T_a is a random threshold:
      T_a = t_lo with probability p, t_hi with probability (1-p)
  Equivalently: if R < t_lo -> predict 0; if R > t_hi -> predict 1;
  if t_lo <= R <= t_hi -> predict 1 with probability p.

PAPER FIDELITY TAGS
-------------------
  [HARDT]  - directly from Hardt et al. (2016) framework
  [INFER]  - required interpretation not explicitly in paper
  [ENGR]   - engineering choice with justification

================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

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
        R > t_hi  -> predict 1
        R < t_lo  -> predict 0
        t_lo <= R <= t_hi -> predict 1 with probability p
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
            # Add midpoints between consecutive unique scores for finer resolution
            (unique_scores[:-1] + unique_scores[1:]) / 2.0 if len(unique_scores) > 1 else np.array([]),
        ])
    else:
        thresholds = np.concatenate([
            np.array([-np.inf, np.inf]),
            np.linspace(scores.min(), scores.max(), n_points),
            unique_scores,  # always include exact score values
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
# SECTION 5 – UPPER CONCAVE ENVELOPE OF ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────

def _upper_concave_envelope(roc: ROCCurve) -> np.ndarray:
    """
    [HARDT] Compute the upper concave envelope of the ROC curve.

    D_a = convhull{C_a(t): t in [0,1]}  (Equation 4.4)

    The upper boundary of D_a is the concave envelope of the ROC curve.
    Any (FPR, TPR) point below this envelope and above the diagonal can be
    achieved by a randomized classifier mixing two thresholds.

    Returns array of shape (K, 2) with [fpr, tpr] sorted by FPR ascending,
    forming a concave curve from (0,0) to (1,1).
    """
    # Collect all ROC (fpr, tpr) points
    pts = [[p.fpr, p.tpr] for p in roc.points]
    pts.append([0.0, 0.0])  # always include origin
    pts.append([1.0, 1.0])  # always include (1,1)

    # Remove duplicates and sort by FPR
    pts = np.array(pts)
    pts = np.unique(pts, axis=0)
    pts = pts[pts[:, 0].argsort()]

    # Build upper concave hull using Andrew's monotone chain (upper hull).
    # The upper hull of ROC points is the concave envelope: for every FPR
    # value, it gives the maximum achievable TPR.
    #
    # We walk left-to-right and keep only points that maintain concavity
    # (i.e., the slope is non-increasing).
    upper = []
    for pt in pts:
        # Remove points that create a convex (upward) turn
        while len(upper) >= 2:
            # Cross product of vectors (upper[-1] - upper[-2]) x (pt - upper[-2])
            # If positive, the three points make a left (convex) turn -> remove middle
            o = upper[-2]
            a = upper[-1]
            cross = (a[0] - o[0]) * (pt[1] - o[1]) - (a[1] - o[1]) * (pt[0] - o[0])
            if cross >= -1e-12:  # left turn or collinear -> remove
                upper.pop()
            else:
                break
        upper.append(pt.tolist())

    envelope = np.array(upper)

    # Ensure all points are above or on the diagonal (TPR >= FPR).
    # Points below the diagonal are worse than random and never optimal.
    above = envelope[:, 1] >= envelope[:, 0] - 1e-9
    if not np.all(above):
        # Keep all points but clip TPR to diagonal
        envelope[:, 1] = np.maximum(envelope[:, 1], envelope[:, 0])

    return envelope


def _interpolate_envelope(envelope: np.ndarray, target_fpr: float) -> Optional[float]:
    """
    Linearly interpolate the concave envelope to find TPR at a given FPR.

    Returns TPR or None if target_fpr is outside the envelope's range.
    """
    if len(envelope) == 0:
        return None

    if target_fpr < envelope[0, 0] - 1e-9 or target_fpr > envelope[-1, 0] + 1e-9:
        return None

    # Exact match
    for i in range(len(envelope)):
        if abs(envelope[i, 0] - target_fpr) < 1e-9:
            return float(envelope[i, 1])

    # Linear interpolation between bracketing points
    for i in range(len(envelope) - 1):
        if envelope[i, 0] <= target_fpr <= envelope[i + 1, 0]:
            span = envelope[i + 1, 0] - envelope[i, 0]
            if span < 1e-12:
                return float(max(envelope[i, 1], envelope[i + 1, 1]))
            alpha = (target_fpr - envelope[i, 0]) / span
            return float(envelope[i, 1] + alpha * (envelope[i + 1, 1] - envelope[i, 1]))

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – FEASIBLE REGION (INTERSECTION OF CONVEX HULLS)
# ─────────────────────────────────────────────────────────────────────────────

def _find_feasible_region(
    envelope_male: np.ndarray,
    envelope_female: np.ndarray,
    n_sample: int = 1000,
) -> np.ndarray:
    """
    [HARDT] Section 4.2: Find the intersection of achievable (FPR, TPR) regions.

    The feasible set for equalized odds is the set of (FPR, TPR) points
    achievable by BOTH groups simultaneously. Its upper boundary is:

        TPR_feasible(fpr) = min(envelope_male(fpr), envelope_female(fpr))

    We sample this boundary densely and return only points that lie ABOVE
    the main diagonal (TPR > FPR), since points on or below the diagonal
    are never optimal for any reasonable loss function.

    Returns array of shape (K, 2) with columns [fpr, tpr].
    """
    # Determine FPR range where both envelopes are defined
    fpr_min = max(envelope_male[0, 0], envelope_female[0, 0])
    fpr_max = min(envelope_male[-1, 0], envelope_female[-1, 0])

    if fpr_min > fpr_max + 1e-9:
        return np.array([])

    # Collect candidate FPR values: envelope vertices + uniform samples
    candidate_fprs = set()
    for pt in envelope_male:
        if fpr_min - 1e-9 <= pt[0] <= fpr_max + 1e-9:
            candidate_fprs.add(float(pt[0]))
    for pt in envelope_female:
        if fpr_min - 1e-9 <= pt[0] <= fpr_max + 1e-9:
            candidate_fprs.add(float(pt[0]))
    for f in np.linspace(fpr_min, fpr_max, n_sample):
        candidate_fprs.add(float(f))

    feasible = []
    for fpr in sorted(candidate_fprs):
        tpr_m = _interpolate_envelope(envelope_male, fpr)
        tpr_f = _interpolate_envelope(envelope_female, fpr)
        if tpr_m is not None and tpr_f is not None:
            # Intersection upper boundary: min of both envelopes
            tpr = min(tpr_m, tpr_f)
            # Only keep points strictly above the diagonal
            if tpr > fpr + 1e-9:
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
    [HARDT] Section 4.2: Realize a target (FPR, TPR) on one group's convex
    hull using a randomized mixture of two threshold predictors.

    Searches all pairs of ROC curve points for the convex combination
    alpha * pt1 + (1-alpha) * pt2 that best matches the target.

    The resulting randomized predictor:
        - Uses threshold t_lo with probability p
        - Uses threshold t_hi with probability (1-p)
    """
    # Collect unique (fpr, tpr, threshold) triples, sorted by FPR
    seen = set()
    unique_pts = []
    for p in sorted(roc.points, key=lambda p: (p.fpr, p.tpr)):
        key = (round(p.fpr, 10), round(p.tpr, 10))
        if key not in seen:
            seen.add(key)
            unique_pts.append(p)

    # Filter to finite thresholds only — avoid -inf/+inf as actual thresholds.
    # The ROC points at -inf and +inf are the trivial classifiers (all positive
    # or all negative). We only use them if absolutely necessary.
    finite_pts = [p for p in unique_pts if np.isfinite(p.threshold)]
    if not finite_pts:
        finite_pts = unique_pts  # fallback if all are non-finite

    # Try to find an exact match first (deterministic threshold)
    best_dist = float('inf')
    best_result = RandomizedThreshold(
        t_lo=finite_pts[0].threshold, t_hi=finite_pts[0].threshold, p=1.0
    )

    for p in finite_pts:
        dist = math.sqrt((p.fpr - target_fpr) ** 2 + (p.tpr - target_tpr) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_result = RandomizedThreshold(t_lo=p.threshold, t_hi=p.threshold, p=1.0)

    if best_dist < 1e-6:
        return best_result

    # Search all pairs for the best convex combination
    # [HARDT] Any point in D_a can be expressed as alpha*C_a(t1) + (1-alpha)*C_a(t2)
    all_pts = finite_pts if len(finite_pts) >= 2 else unique_pts

    for i in range(len(all_pts)):
        for j in range(i + 1, len(all_pts)):
            p1 = all_pts[i]
            p2 = all_pts[j]

            dfpr = p2.fpr - p1.fpr
            dtpr = p2.tpr - p1.tpr

            if abs(dfpr) < 1e-12 and abs(dtpr) < 1e-12:
                continue

            # Solve for alpha: target = alpha * p2 + (1-alpha) * p1
            # target_fpr = p1.fpr + alpha * dfpr
            # target_tpr = p1.tpr + alpha * dtpr
            # Use the axis with larger range for numerical stability
            if abs(dfpr) >= abs(dtpr):
                alpha = (target_fpr - p1.fpr) / dfpr
            else:
                alpha = (target_tpr - p1.tpr) / dtpr

            if alpha < -0.01 or alpha > 1.01:
                continue

            alpha = float(np.clip(alpha, 0.0, 1.0))
            achieved_fpr = p1.fpr + alpha * dfpr
            achieved_tpr = p1.tpr + alpha * dtpr
            dist = math.sqrt((achieved_fpr - target_fpr) ** 2 +
                             (achieved_tpr - target_tpr) ** 2)

            if dist < best_dist:
                best_dist = dist
                # Convention: t_lo <= t_hi.
                # Mixing: use t_lo with prob p, t_hi with prob (1-p).
                # p1 corresponds to (1-alpha), p2 corresponds to alpha.
                if p1.threshold <= p2.threshold:
                    # p1 is the lower threshold, p2 is the higher threshold
                    # p1 gives MORE positives (lower threshold -> more R > t)
                    # alpha controls weight on p2 (fewer positives)
                    best_result = RandomizedThreshold(
                        t_lo=p1.threshold, t_hi=p2.threshold, p=1.0 - alpha,
                    )
                else:
                    best_result = RandomizedThreshold(
                        t_lo=p2.threshold, t_hi=p1.threshold, p=alpha,
                    )

    return best_result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – LOSS OPTIMIZATION OVER FEASIBLE REGION
# ─────────────────────────────────────────────────────────────────────────────

def _optimize_over_feasible_region(
    feasible: np.ndarray,
    prevalence_neg: float,
    prevalence_pos: float,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
) -> Tuple[float, float, float]:
    """
    [HARDT] Find the point on the upper-left boundary of the feasible region
    that minimizes the prevalence-weighted expected loss:

        L = P(Y=0) * cost_fp * FPR  +  P(Y=1) * cost_fn * (1 - TPR)

    This is the standard expected error rate when cost_fp = cost_fn = 1:
        L = P(Y=0) * FPR + P(Y=1) * FNR

    The prevalence weighting ensures the optimizer doesn't select degenerate
    operating points (e.g., FPR=1.0) when negatives far outnumber positives.

    Returns (best_fpr, best_tpr, best_loss).
    """
    best_loss = float('inf')
    best_fpr = 0.0
    best_tpr = 0.0

    for fpr, tpr in feasible:
        fnr = 1.0 - tpr
        loss = prevalence_neg * cost_fp * fpr + prevalence_pos * cost_fn * fnr
        # Tiebreak: prefer lower FPR (clinically, false alarms are costly)
        if loss < best_loss - 1e-12 or (abs(loss - best_loss) < 1e-12 and fpr < best_fpr):
            best_loss = loss
            best_fpr = fpr
            best_tpr = tpr

    return best_fpr, best_tpr, best_loss


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – MAIN ENTRY POINT
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
    log.info("  Method: Concave envelope intersection + randomized thresholds")
    log.info("  Loss: prevalence-weighted expected error rate")
    log.info("=" * 80)

    if len(scores) != len(labels) or len(scores) != len(data):
        raise ValueError("Dimension mismatch: scores, labels, data must have equal length")

    if not np.all((labels == 1) | (labels >= 2)):
        raise ValueError("Labels must be 1 (healthy) or >=2 (diseased)")

    gender_col = data[:, SEX_FEATURE_IDX]
    n_male = int(np.sum(gender_col == MALE_VALUE))
    n_female = int(np.sum(gender_col == FEMALE_VALUE))

    # Compute overall prevalence for loss weighting
    n_total = len(labels)
    n_pos = int(np.sum(labels != 1))  # diseased
    n_neg = int(np.sum(labels == 1))  # healthy
    prevalence_pos = n_pos / n_total if n_total > 0 else 0.5
    prevalence_neg = n_neg / n_total if n_total > 0 else 0.5

    log.info(f"Dataset: {n_male} males, {n_female} females, "
             f"{n_neg} healthy ({prevalence_neg:.1%}), "
             f"{n_pos} diseased ({prevalence_pos:.1%})")

    # Step 1: Compute per-group ROC curves
    roc_male, roc_female = compute_roc_curves_by_gender(
        scores, labels, data, n_points=n_roc_points
    )

    # Step 2: Compute upper concave envelopes (boundary of D_a)
    envelope_m = _upper_concave_envelope(roc_male)
    envelope_f = _upper_concave_envelope(roc_female)

    log.info(f"  Male envelope: {len(envelope_m)} vertices")
    log.info(f"  Female envelope: {len(envelope_f)} vertices")

    # Step 3: Find intersection of achievable regions
    feasible = _find_feasible_region(envelope_m, envelope_f)

    if len(feasible) == 0:
        log.warning("No feasible equalized odds solution found. Falling back to baseline.")
        return _fallback_result(roc_male, roc_female, baseline_threshold, n_male, n_female)

    log.info(f"  Feasible region: {len(feasible)} points above diagonal")

    # Step 4: Optimize using prevalence-weighted loss
    best_fpr, best_tpr, best_loss = _optimize_over_feasible_region(
        feasible, prevalence_neg, prevalence_pos
    )

    log.info(f"  Optimal operating point: FPR={best_fpr:.4f}, TPR={best_tpr:.4f}")
    log.info(f"  Prevalence-weighted loss: {best_loss:.6f}")

    # Step 5: Realize via randomized thresholds for each group
    rand_thresh_male = _realize_operating_point(roc_male, best_fpr, best_tpr)
    rand_thresh_female = _realize_operating_point(roc_female, best_fpr, best_tpr)

    log.info(f"  Male threshold: t_lo={rand_thresh_male.t_lo:.6f}, "
             f"t_hi={rand_thresh_male.t_hi:.6f}, p={rand_thresh_male.p:.4f}"
             f" (deterministic={rand_thresh_male.is_deterministic})")
    log.info(f"  Female threshold: t_lo={rand_thresh_female.t_lo:.6f}, "
             f"t_hi={rand_thresh_female.t_hi:.6f}, p={rand_thresh_female.p:.4f}"
             f" (deterministic={rand_thresh_female.is_deterministic})")

    # Compute representative deterministic thresholds for backward compatibility
    threshold_male = _representative_threshold(rand_thresh_male)
    threshold_female = _representative_threshold(rand_thresh_female)

    # Compute baseline loss for utility comparison
    baseline_pt_m = roc_male.compute_point_at_threshold(baseline_threshold)
    baseline_pt_f = roc_female.compute_point_at_threshold(baseline_threshold)
    # Weighted average baseline FPR/FNR across both groups
    baseline_fpr = (baseline_pt_m.fpr * roc_male.n_negative +
                    baseline_pt_f.fpr * roc_female.n_negative) / n_neg if n_neg > 0 else 0
    baseline_fnr = (baseline_pt_m.fnr * roc_male.n_positive +
                    baseline_pt_f.fnr * roc_female.n_positive) / n_pos if n_pos > 0 else 0
    baseline_loss = prevalence_neg * baseline_fpr + prevalence_pos * baseline_fnr
    utility_loss = best_loss - baseline_loss

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
            "envelope_male_vertices": len(envelope_m),
            "envelope_female_vertices": len(envelope_f),
            "feasible_region_points_count": len(feasible),
            "optimal_fpr": best_fpr,
            "optimal_tpr": best_tpr,
            "prevalence_pos": prevalence_pos,
            "prevalence_neg": prevalence_neg,
            "prevalence_weighted_loss": best_loss,
            "baseline_loss": baseline_loss,
            "randomized_male": not rand_thresh_male.is_deterministic,
            "randomized_female": not rand_thresh_female.is_deterministic,
        },
    )

    _print_equalized_odds_summary(result)

    log.info("=" * 80)

    return result


def _representative_threshold(rt: RandomizedThreshold) -> float:
    """Compute a single representative threshold for backward compatibility."""
    if rt.is_deterministic:
        return rt.t_lo
    # Weighted average of the two thresholds
    return rt.t_lo * rt.p + rt.t_hi * (1.0 - rt.p)


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
    log.info(f"Utility Loss (vs baseline):      {result.utility_loss:+.6f}")
    log.info(f"Dataset: {result.n_male} male, {result.n_female} female users")
    log.info("-" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – APPLY THRESHOLDS TO PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_randomized_threshold(
    scores: np.ndarray,
    rt: RandomizedThreshold,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    [HARDT] Section 4.2: Apply a randomized threshold predictor.

    For each score R:
        R > t_hi  -> predict 1 (always positive)
        R < t_lo  -> predict 0 (always negative)
        t_lo <= R <= t_hi -> predict 1 with probability p
    """
    predictions = np.zeros(len(scores), dtype=int)

    above = scores > rt.t_hi
    predictions[above] = 1

    if abs(rt.t_lo - rt.t_hi) > 1e-12:
        between = (scores >= rt.t_lo) & (scores <= rt.t_hi)
        n_between = int(np.sum(between))
        if n_between > 0:
            coin_flips = rng.random(n_between) < rt.p
            predictions[between] = coin_flips.astype(int)
    else:
        # Deterministic with possible coin flip at the exact threshold
        at_threshold = np.abs(scores - rt.t_lo) < 1e-12
        n_at = int(np.sum(at_threshold))
        if n_at > 0 and 0 < rt.p < 1:
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
# SECTION 11 – MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Fairness_EqualizedOdds module loaded successfully.")
    log.info("Use compute_bayes_optimal_predictor() as main entry point.")
    log.info("Implements full equalized odds (Hardt et al. 2016):")
    log.info("  - Constrains BOTH TPR and FPR to be equal across groups")
    log.info("  - Supports randomized classifiers (mixture of two thresholds)")
    log.info("  - Uses prevalence-weighted expected loss for optimization")


if __name__ == "__main__":
    main()
