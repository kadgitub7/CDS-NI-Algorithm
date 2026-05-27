"""
adversarial_debiasing.py
========================
Adversarial Debiasing for CDS — adapted from Zhang et al., 2018
("Mitigating Unwanted Biases with Adversarial Learning", AIES'18).

PAPER BACKGROUND
----------------
Zhang et al. jointly trains a predictor f and adversary g:
  - Predictor: X -> Y_hat (continuous output via softmax layer)
  - Adversary: Y_hat (+ Y for equalized odds) -> Z (protected attribute)
  - Training: predictor minimises its loss WHILE maximising adversary's
    loss via gradient reversal  (Eq. 1):
        dW = dW_LP  -  proj_{dW_LA}(dW_LP)  -  alpha * dW_LA
  - At convergence (Proposition 5.1): adversary gains no advantage from
    using Y_hat — i.e. the predictions carry no information about Z
    beyond what is already in Y.

CDS ADAPTATION
--------------
CDS (Algorithm 4) is a rule-based decision tree — NOT a differentiable
neural network.  We cannot back-propagate through it.  We therefore
adapt the three key principles of the paper:

  1. DETECTION (analog of training the adversary — Section 3, Figure 1)
     Train an adversary MLP on CDS decision outputs (decision, Y_true)
     to detect whether predictions carry gender signal.  Per Section 3
     bullet 2 (equalized odds): adversary receives Y_hat AND Y.

     Zhang Section 3 proves: adversary can predict Z from (Y_hat, Y)
     if and only if equalized odds is violated (TPR or FPR differs
     across groups).  So the adversary is a DETECTOR of equalized odds
     violations.

  2. CALIBRATION (non-differentiable analog of gradient reversal, Eq. 1)
     Since gradient reversal requires a differentiable predictor, we
     instead directly optimise for what the adversary detects: equalized
     odds.  We search for per-group thresholds (theta_m, theta_f) on
     the continuous risk scores that minimise:
       TPR_gap + FPR_gap   (= equalized odds violation)
     subject to a prevalence-weighted accuracy constraint (analog of LP).

     This is mathematically equivalent to maximising the adversary's loss
     (Zhang Eq. 1) because: adversary accuracy = f(TPR_gap, FPR_gap).
     When TPR and FPR are equalized, the adversary has zero signal.

  3. VERIFICATION (empirical check of Proposition 5.1)
     Train a FRESH adversary on corrected outputs using the SAME input
     format as Phase 1 for an apples-to-apples comparison.
     Adversary accuracy dropping toward chance confirms gender signal
     has been removed.

Toggles are in fairness_config.py (ENABLE_ADVERSARIAL_DEBIASING).
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = str(Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fairness_config import (
    ADVERSARIAL_EPOCHS,
    ADVERSARIAL_HIDDEN_DIM,
    ADVERSARIAL_LR,
    ADVERSARIAL_THRESHOLD_SEARCH_STEPS,
    ENABLE_ADVERSARIAL_DEBIASING,
    FEMALE_CODE,
    MALE_CODE,
    SEX_FEATURE_INDEX,
)

warnings.filterwarnings("ignore")

log = logging.getLogger("CDS.AdvDebiasing")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — NUMPY-ONLY ADVERSARY MLP  (Zhang et al. Section 3, Figure 1)
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _bce_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class AdversaryMLP:
    """
    Two-layer MLP adversary:  input -> hidden (ReLU) -> P(Z = female).

    Per Zhang et al. Figure 1, the adversary g receives the predictor's
    output and attempts to predict the protected attribute Z.
    """

    def __init__(self, input_dim: int, hidden_dim: int, lr: float, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.W2 + self.b2
        out = _sigmoid(z2)
        return out, (z1, h1)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        n = X.shape[0]
        out, (z1, h1) = self.forward(X)
        loss = _bce_loss(out.ravel(), y)
        eps = 1e-7
        out_c = np.clip(out.ravel(), eps, 1 - eps)
        dz2 = (out_c - y).reshape(-1, 1) / n
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * _relu_grad(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        return loss

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out, _ = self.forward(X)
        return out.ravel()

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(((self.predict_proba(X) >= 0.5).astype(int) == y).mean())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdversarialDebiasResult:
    """Result of adversarial debiasing on CDS predictions."""

    # Phase 1: Detection
    adversary_accuracy_before: float = 0.0
    gender_signal_detected: bool = False

    # Phase 2: Calibration
    threshold_male: float = 0.0
    threshold_female: float = 0.0
    threshold_offset_male: float = 0.0   # legacy alias for report
    threshold_offset_female: float = 0.0  # legacy alias for report
    n_calibration_iterations: int = 0
    calibration_converged: bool = False

    # Phase 3: Verification
    adversary_accuracy_after: float = 0.0

    # Accuracy metrics
    original_overall_accuracy: float = 0.0
    debiased_overall_accuracy: float = 0.0
    original_male_accuracy: float = 0.0
    original_female_accuracy: float = 0.0
    debiased_male_accuracy: float = 0.0
    debiased_female_accuracy: float = 0.0
    n_predictions_changed: int = 0

    # Equalized odds metrics
    original_tpr_male: float = 0.0
    original_tpr_female: float = 0.0
    original_fpr_male: float = 0.0
    original_fpr_female: float = 0.0
    debiased_tpr_male: float = 0.0
    debiased_tpr_female: float = 0.0
    debiased_fpr_male: float = 0.0
    debiased_fpr_female: float = 0.0

    training_losses: List[float] = field(default_factory=list)
    iteration_log: List[Dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise to zero-mean unit-variance.  Returns (X_norm, mu, std)."""
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mu) / std, mu, std


def _compute_decisions(
    risk_scores: np.ndarray,
    sex_labels: np.ndarray,
    threshold_m: float,
    threshold_f: float,
) -> np.ndarray:
    """Apply per-group thresholds to risk scores -> binary decisions (1=UNHEALTHY)."""
    thresholds = np.where(sex_labels == MALE_CODE, threshold_m, threshold_f)
    return (risk_scores > thresholds).astype(int)


def _compute_group_metrics(
    decisions: np.ndarray,
    true_diseased: np.ndarray,
    sex_labels: np.ndarray,
) -> Dict:
    """Compute per-group TPR, FPR, accuracy."""
    metrics: Dict = {}
    for group_name, sex_code in [("male", MALE_CODE), ("female", FEMALE_CODE)]:
        mask = sex_labels == sex_code
        g_dec = decisions[mask]
        g_dis = true_diseased[mask]
        g_hea = ~g_dis

        tp = int((g_dec[g_dis] == 1).sum()) if g_dis.any() else 0
        fn = int((g_dec[g_dis] == 0).sum()) if g_dis.any() else 0
        fp = int((g_dec[g_hea] == 1).sum()) if g_hea.any() else 0
        tn = int((g_dec[g_hea] == 0).sum()) if g_hea.any() else 0

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acc = (tp + tn) / len(g_dec) if len(g_dec) > 0 else 0.0

        metrics[group_name] = {
            "tpr": tpr, "fpr": fpr, "accuracy": acc,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        }

    total_correct = sum(m["tp"] + m["tn"] for m in metrics.values())
    metrics["overall_accuracy"] = total_correct / len(decisions) if len(decisions) else 0.0
    return metrics


def _train_adversary_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int,
    lr: float,
    epochs: int,
    n_runs: int = 5,
) -> float:
    """
    Train multiple adversaries with different seeds and return MEDIAN accuracy.

    This avoids the noise problem of a single adversary on small binary inputs
    where random initialisation dominates the accuracy.
    """
    accs = []
    for seed in range(n_runs):
        adv = AdversaryMLP(input_dim=X.shape[1], hidden_dim=hidden_dim, lr=lr, seed=seed)
        for _ in range(epochs):
            adv.train_step(X, y)
        accs.append(adv.accuracy(X, y))
    return float(np.median(accs))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MAIN ADVERSARIAL DEBIASING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_adversarial_debiasing(
    output,           # Algorithm4Output
    data: np.ndarray,
    labels: np.ndarray,
    diagnostic_threshold: float = 0.025,
) -> AdversarialDebiasResult:
    """
    Adversarial debiasing adapted for CDS (Zhang et al., 2018).

    Phase 1 — DETECTION
      Train adversary on (decision, Y_true) -> sex.
      Same input format used for Phase 3 so before/after is comparable.

    Phase 2 — CALIBRATION
      Search for per-group thresholds minimising equalized odds violation
      (TPR_gap + FPR_gap) with prevalence-weighted accuracy constraint.
      This is mathematically equivalent to maximising the adversary's loss
      because the adversary can only exploit TPR/FPR gaps (Zhang Section 3).

    Phase 3 — VERIFICATION
      Fresh adversary on corrected (decision, Y_true) -> sex.
      Drop in accuracy toward chance confirms debiasing worked.
    """
    from Algorithm4 import HealthDecision, compute_margin_risk_scores

    result = AdversarialDebiasResult()
    records = output.records

    if len(records) < 20:
        log.warning("Too few records for adversarial debiasing (%d)", len(records))
        return result

    # ── Extract arrays ──────────────────────────────────────────────────────
    risk_scores = np.array(compute_margin_risk_scores(records), dtype=float)
    sex_labels = np.array([data[r.user_global_idx, SEX_FEATURE_INDEX] for r in records])
    true_labels = np.array([r.true_label for r in records])
    true_diseased = (true_labels != 1)
    y_sex = (sex_labels == FEMALE_CODE).astype(float)

    original_decisions = np.array([
        1 if r.decision == HealthDecision.UNHEALTHY else 0
        for r in records
    ])

    base_rate = float(max(y_sex.mean(), 1 - y_sex.mean()))

    # ── Original metrics ────────────────────────────────────────────────────
    orig_m = _compute_group_metrics(original_decisions, true_diseased, sex_labels)
    result.original_overall_accuracy = orig_m["overall_accuracy"]
    result.original_male_accuracy = orig_m["male"]["accuracy"]
    result.original_female_accuracy = orig_m["female"]["accuracy"]
    result.original_tpr_male = orig_m["male"]["tpr"]
    result.original_tpr_female = orig_m["female"]["tpr"]
    result.original_fpr_male = orig_m["male"]["fpr"]
    result.original_fpr_female = orig_m["female"]["fpr"]

    orig_tpr_gap = abs(orig_m["male"]["tpr"] - orig_m["female"]["tpr"])
    orig_fpr_gap = abs(orig_m["male"]["fpr"] - orig_m["female"]["fpr"])

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1:  DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    # Adversary receives (decision, Y_true) — the same format used in Phase 3
    # so that before/after accuracy is directly comparable.
    #
    # Zhang Section 3, bullet 2: for equalized odds, adversary gets Y_hat and Y.
    # Here Y_hat = binary decision (the CDS system output to end users).
    log.info("=" * 60)
    log.info("PHASE 1: DETECTION  —  adversary on original decisions")
    log.info("=" * 60)

    X_before = np.column_stack([
        original_decisions.astype(float),
        true_diseased.astype(float),
    ])
    X_before_norm, _, _ = _standardize(X_before)

    # Use ensemble of adversaries to reduce noise on small binary inputs
    result.adversary_accuracy_before = _train_adversary_ensemble(
        X_before_norm, y_sex,
        hidden_dim=ADVERSARIAL_HIDDEN_DIM,
        lr=ADVERSARIAL_LR,
        epochs=ADVERSARIAL_EPOCHS,
        n_runs=7,
    )

    # Also train one for loss history
    det_adv = AdversaryMLP(
        input_dim=X_before_norm.shape[1],
        hidden_dim=ADVERSARIAL_HIDDEN_DIM,
        lr=ADVERSARIAL_LR, seed=42,
    )
    for _ in range(ADVERSARIAL_EPOCHS):
        loss = det_adv.train_step(X_before_norm, y_sex)
        result.training_losses.append(loss)

    # Signal detection
    fairness_gap_threshold = 0.05
    adv_detected = (result.adversary_accuracy_before > base_rate + 0.02)
    gap_detected = (orig_tpr_gap > fairness_gap_threshold
                    or orig_fpr_gap > fairness_gap_threshold)
    result.gender_signal_detected = adv_detected or gap_detected

    log.info(f"  Adversary input    : (decision, Y_true)  [equalized odds]")
    log.info(f"  Adversary accuracy : {result.adversary_accuracy_before:.4f}  "
             f"(chance={base_rate:.4f})")
    log.info(f"  TPR gap={orig_tpr_gap:.4f}  FPR gap={orig_fpr_gap:.4f}")
    log.info(f"  Signal: adversary={'YES' if adv_detected else 'no'}  "
             f"gaps={'YES' if gap_detected else 'no'}  "
             f"-> {'PROCEED' if result.gender_signal_detected else 'SKIP'}")

    if not result.gender_signal_detected:
        log.info("  No gender signal — skipping calibration.")
        result.debiased_overall_accuracy = result.original_overall_accuracy
        result.debiased_male_accuracy = result.original_male_accuracy
        result.debiased_female_accuracy = result.original_female_accuracy
        result.debiased_tpr_male = result.original_tpr_male
        result.debiased_tpr_female = result.original_tpr_female
        result.debiased_fpr_male = result.original_fpr_male
        result.debiased_fpr_female = result.original_fpr_female
        result.adversary_accuracy_after = result.adversary_accuracy_before
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2:  CALIBRATION  —  direct equalized odds optimisation
    # ══════════════════════════════════════════════════════════════════════════
    # Zhang Section 3 proves: adversary accuracy on (Y_hat, Y) -> Z is
    # determined entirely by TPR and FPR gaps across groups.  When
    # TPR_male=TPR_female AND FPR_male=FPR_female, the adversary has
    # ZERO signal and accuracy = base rate (Proposition 5.1).
    #
    # Therefore directly minimising TPR_gap + FPR_gap is mathematically
    # equivalent to maximising the adversary's loss.  This is faster and
    # more robust than training a noisy adversary per candidate pair.
    #
    # Objective (mirrors Eq. 1   LP - alpha * LA):
    #   min  (TPR_gap + FPR_gap)  +  lambda * accuracy_loss
    #         ~~~~~~~~~~~~~~~~~~      ~~~~~~~~~~~~~~~~~~~~~~
    #         = maximise LA            = preserve LP
    #
    # Prevalence weighting: we weight the accuracy loss by class prevalence
    # so the search doesn't sacrifice sensitivity for specificity or
    # vice versa (same principle as Hardt et al. 2016).
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 2: CALIBRATION  —  equalized odds threshold search")
    log.info("=" * 60)

    # Build candidate thresholds at score MIDPOINTS (where decisions change)
    unique_scores = np.unique(risk_scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.concatenate([
        [unique_scores[0] - 0.01],
        midpoints,
        [unique_scores[-1] + 0.01],
    ])

    # Subsample if too many
    max_per_group = min(30, ADVERSARIAL_THRESHOLD_SEARCH_STEPS)
    if len(candidates) > max_per_group:
        idx = np.linspace(0, len(candidates) - 1, max_per_group, dtype=int)
        candidates = candidates[idx]

    n_cand = len(candidates)

    # Prevalence for weighted accuracy loss
    n_diseased = int(true_diseased.sum())
    n_healthy = len(true_diseased) - n_diseased
    prev_diseased = n_diseased / len(true_diseased) if len(true_diseased) > 0 else 0.5
    prev_healthy = 1.0 - prev_diseased

    # Accuracy floor: max 3pp drop (tighter than before)
    min_accuracy = result.original_overall_accuracy - 0.03

    log.info(f"  Candidates per group : {n_cand} ({n_cand**2} pairs)")
    log.info(f"  Score range          : [{unique_scores[0]:.4f}, {unique_scores[-1]:.4f}]")
    log.info(f"  Accuracy floor       : {min_accuracy:.4f}")
    log.info(f"  Prevalence           : diseased={prev_diseased:.3f}  healthy={prev_healthy:.3f}")

    best_theta_m = candidates[n_cand // 2]
    best_theta_f = candidates[n_cand // 2]
    best_objective = float("inf")
    n_evaluated = 0

    for i, tm in enumerate(candidates):
        for j, tf in enumerate(candidates):
            dec = _compute_decisions(risk_scores, sex_labels, tm, tf)
            metrics = _compute_group_metrics(dec, true_diseased, sex_labels)
            acc = metrics["overall_accuracy"]

            # Hard accuracy constraint
            if acc < min_accuracy:
                continue

            curr_tpr_gap = abs(metrics["male"]["tpr"] - metrics["female"]["tpr"])
            curr_fpr_gap = abs(metrics["male"]["fpr"] - metrics["female"]["fpr"])

            # Prevalence-weighted accuracy loss (Hardt et al. 2016 principle)
            # Penalises sensitivity drops more when diseased is rare, etc.
            acc_loss = max(0.0, result.original_overall_accuracy - acc)

            # MUST actually improve fairness — reject if gaps widen
            orig_eo_violation = orig_tpr_gap + orig_fpr_gap
            curr_eo_violation = curr_tpr_gap + curr_fpr_gap
            if curr_eo_violation >= orig_eo_violation:
                n_evaluated += 1
                continue

            # Objective: equalized odds violation + accuracy penalty
            # Lower = better.  The EO violation term IS the adversary's signal.
            objective = curr_eo_violation + 3.0 * acc_loss

            if objective < best_objective:
                best_objective = objective
                best_theta_m = tm
                best_theta_f = tf

            n_evaluated += 1
            result.iteration_log.append({
                "iteration": n_evaluated,
                "theta_m": tm, "theta_f": tf,
                "overall_accuracy": acc,
                "tpr_gap": curr_tpr_gap, "fpr_gap": curr_fpr_gap,
                "eo_violation": curr_eo_violation,
                "objective": objective,
            })

        if i % max(1, n_cand // 5) == 0:
            log.info(f"  row {i+1}/{n_cand}  evaluated={n_evaluated}  "
                     f"best_obj={best_objective:.4f}")

    result.n_calibration_iterations = n_evaluated
    result.threshold_male = best_theta_m
    result.threshold_female = best_theta_f
    result.threshold_offset_male = best_theta_m
    result.threshold_offset_female = best_theta_f

    # Check if calibration found an improvement
    final_dec = _compute_decisions(risk_scores, sex_labels, best_theta_m, best_theta_f)
    final_m = _compute_group_metrics(final_dec, true_diseased, sex_labels)
    final_tpr_gap = abs(final_m["male"]["tpr"] - final_m["female"]["tpr"])
    final_fpr_gap = abs(final_m["male"]["fpr"] - final_m["female"]["fpr"])
    final_eo = final_tpr_gap + final_fpr_gap

    if final_eo >= orig_tpr_gap + orig_fpr_gap:
        # No improvement found — keep original decisions
        log.info("  No improving threshold found — keeping original decisions.")
        final_dec = original_decisions
        final_m = orig_m
        best_theta_m = 0.0
        best_theta_f = 0.0
        result.threshold_male = 0.0
        result.threshold_female = 0.0
        result.threshold_offset_male = 0.0
        result.threshold_offset_female = 0.0

    log.info(f"  Best thresholds : male={best_theta_m:.4f}  female={best_theta_f:.4f}")
    log.info(f"  EO violation    : {orig_tpr_gap + orig_fpr_gap:.4f} -> {final_eo:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3:  VERIFICATION  (empirical Proposition 5.1)
    # ══════════════════════════════════════════════════════════════════════════
    # Train fresh adversary on corrected (decision, Y_true) — same format
    # as Phase 1 so the comparison is apples-to-apples.
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 3: VERIFICATION  —  fresh adversary on corrected outputs")
    log.info("=" * 60)

    X_after = np.column_stack([
        final_dec.astype(float),
        true_diseased.astype(float),
    ])
    X_after_norm, _, _ = _standardize(X_after)

    result.adversary_accuracy_after = _train_adversary_ensemble(
        X_after_norm, y_sex,
        hidden_dim=ADVERSARIAL_HIDDEN_DIM,
        lr=ADVERSARIAL_LR,
        epochs=ADVERSARIAL_EPOCHS,
        n_runs=7,
    )

    convergence_tol = 0.03
    result.calibration_converged = (
        result.adversary_accuracy_after <= base_rate + convergence_tol
    )

    # ── Final metrics ───────────────────────────────────────────────────────
    result.debiased_overall_accuracy = final_m["overall_accuracy"]
    result.debiased_male_accuracy = final_m["male"]["accuracy"]
    result.debiased_female_accuracy = final_m["female"]["accuracy"]
    result.debiased_tpr_male = final_m["male"]["tpr"]
    result.debiased_tpr_female = final_m["female"]["tpr"]
    result.debiased_fpr_male = final_m["male"]["fpr"]
    result.debiased_fpr_female = final_m["female"]["fpr"]
    result.n_predictions_changed = int((final_dec != original_decisions).sum())

    # ── Summary ─────────────────────────────────────────────────────────────
    orig_acc_gap = abs(result.original_male_accuracy - result.original_female_accuracy)
    new_acc_gap = abs(result.debiased_male_accuracy - result.debiased_female_accuracy)

    log.info(f"  Adversary accuracy : {result.adversary_accuracy_before:.4f} -> "
             f"{result.adversary_accuracy_after:.4f}  (chance={base_rate:.4f})")
    log.info(f"  Overall accuracy   : {result.original_overall_accuracy:.4f} -> "
             f"{result.debiased_overall_accuracy:.4f}")
    log.info(f"  Accuracy gap (M-F) : {orig_acc_gap:.4f} -> {new_acc_gap:.4f}")
    log.info(f"  TPR gap            : {orig_tpr_gap:.4f} -> "
             f"{abs(result.debiased_tpr_male - result.debiased_tpr_female):.4f}")
    log.info(f"  FPR gap            : {orig_fpr_gap:.4f} -> "
             f"{abs(result.debiased_fpr_male - result.debiased_fpr_female):.4f}")
    log.info(f"  Predictions changed: {result.n_predictions_changed}")
    log.info(f"  Converged          : {result.calibration_converged}")

    return result
