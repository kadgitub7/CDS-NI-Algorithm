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
    using Y_hat — i.e.  the predictions carry no information about Z
    beyond what is already in Y.

CDS ADAPTATION
--------------
CDS (Algorithm 4) is a rule-based decision tree — NOT a differentiable
neural network.  We cannot back-propagate through it.  We therefore
adapt the three key principles of the paper:

  1. DETECTION (analog of training the adversary — Section 3, Figure 1)
     Train an adversary MLP on CDS continuous outputs to detect whether
     they carry gender signal.
       Input:  Y_hat (continuous margin risk score)  +  Y (true label)
     This matches Section 3 bullet 2 (equalized odds: adversary receives
     Y_hat AND Y).  The paper notes Y_hat should be the continuous output
     layer, not the discrete prediction, for gradient flow.  We use the
     margin risk score — the continuous analog of CDS's binary alarm.

  2. ADVERSARY-GUIDED CALIBRATION (analog of gradient reversal, Eq. 1)
     Non-differentiable analog of the weight-update rule.  Instead of
     modifying predictor weights W via the adversary gradient:
         W  <-  W  - eta * (dW_LP - alpha * dW_LA)       [paper Eq. 1]
     we optimise per-group decision thresholds (theta_m, theta_f) to
     maximise the adversary's loss:
         theta  <-  theta  +  eta * dLA/dtheta            [our analog]
     where dLA/dtheta is estimated via finite differences (since the
     threshold -> decision -> adversary pipeline is non-differentiable).

     An accuracy constraint (analog of LP in Eq. 1) prevents the
     thresholds from drifting to degenerate operating points.

     The adversary for this phase receives (decision, Y_true) — the
     discrete outputs after thresholding.  Per the paper's Section 7:
     "in the common case of discrete output and protected variables,
     a very simple adversary suffices regardless of predictor complexity."

  3. VERIFICATION (empirical check of Proposition 5.1)
     Train a FRESH adversary from scratch on corrected outputs.
     If accuracy ~= 50% (chance), gender signal has been removed.

CONNECTION TO EQUALIZED ODDS
----------------------------
Section 3 states: for equalized odds, the adversary gets Y_hat and Y.
The adversary exploits TPR and FPR differences across groups.  When
per-group thresholds equalise TPR and FPR, the adversary loses all
signal -> accuracy drops to base rate.  Thus our calibration naturally
converges to equalized odds thresholds, validated by the adversary.

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
# Pure-numpy forward/backward — no PyTorch/TensorFlow dependency.
# The adversary g receives the predictor's output and attempts to predict Z.

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _bce_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Binary cross-entropy loss (adversary's LA)."""
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class AdversaryMLP:
    """
    Two-layer MLP adversary:  input -> hidden (ReLU) -> P(Z = female).

    Per Zhang et al. Figure 1, this is the adversary network g that
    receives the predictor's output layer and attempts to predict Z.

    For equalized odds (Section 3, bullet 2): input = (Y_hat, Y).
    """

    def __init__(self, input_dim: int, hidden_dim: int, lr: float, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1) * scale2
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.W2 + self.b2
        out = _sigmoid(z2)
        return out, (z1, h1)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """One gradient step minimising BCE(g(X), y).  Returns loss."""
        n = X.shape[0]
        out, (z1, h1) = self.forward(X)
        loss = _bce_loss(out.ravel(), y)

        # Backward pass
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

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return _bce_loss(self.predict_proba(X), y)


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
    # Legacy aliases used by reproducibility report
    threshold_offset_male: float = 0.0
    threshold_offset_female: float = 0.0
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

    # Equalized odds metrics (TPR / FPR per group)
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
    """Standardise features to zero-mean unit-variance.  Returns (X_norm, mu, std)."""
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
    """
    Compute per-group TPR, FPR, accuracy.

    Returns dict with keys: "male", "female" (each a sub-dict with
    tpr, fpr, accuracy, tp, fn, fp, tn), and "overall_accuracy".
    """
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


def _train_fresh_adversary(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int,
    lr: float,
    epochs: int,
    seed: int = 42,
) -> Tuple[AdversaryMLP, List[float]]:
    """Train a fresh adversary MLP from scratch.  Returns (adversary, losses)."""
    adv = AdversaryMLP(input_dim=X.shape[1], hidden_dim=hidden_dim, lr=lr, seed=seed)
    losses = []
    for _ in range(epochs):
        loss = adv.train_step(X, y)
        losses.append(loss)
    return adv, losses


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

    Three phases mirroring the paper's approach:

    Phase 1 — DETECTION  (train the adversary — Section 3, Figure 1)
      Train adversary on continuous CDS outputs:
        Input:  (risk_score, Y_true)    [equalized odds variant, Section 3]
        Target: sex (Z)
      The risk score is the continuous analog of Y_hat (softmax output).
      If adversary accuracy >> chance, CDS outputs carry gender signal.

    Phase 2 — CALIBRATION  (non-differentiable gradient reversal, Eq. 1)
      Iteratively adjust per-group thresholds (theta_m, theta_f) using
      finite-difference gradients of the adversary loss:
        dLA/dtheta  ~=  [LA(theta+eps) - LA(theta-eps)] / (2*eps)
      Move thresholds in direction that increases adversary loss (= reduces
      its ability to predict sex), subject to accuracy constraint.

      This is the non-differentiable analog of Eq. 1:
        Paper:   W  <- W - eta*(dW_LP - alpha*dW_LA)  [modify weights]
        Ours: theta <- theta + eta * dLA/dtheta        [modify thresholds]

    Phase 3 — VERIFICATION  (empirical Proposition 5.1)
      Train fresh adversary on corrected (decision, Y_true) outputs.
      Adversary accuracy ~= 50% confirms gender signal removed.

    Parameters
    ----------
    output : Algorithm4Output with completed prediction records.
    data   : full dataset array (N x features).
    labels : full labels array (N,).
    diagnostic_threshold : base CDS threshold (0.025, for reporting).

    Returns
    -------
    AdversarialDebiasResult with before/after metrics and iteration log.
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
    true_diseased = (true_labels != 1)            # True  = diseased (label >= 2)
    y_sex = (sex_labels == FEMALE_CODE).astype(float)  # target for adversary

    original_decisions = np.array([
        1 if r.decision == HealthDecision.UNHEALTHY else 0
        for r in records
    ])

    base_rate = float(max(y_sex.mean(), 1 - y_sex.mean()))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1:  DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    # Per Zhang Section 3, bullet 2 (equalized odds): adversary receives
    # Y_hat AND Y.  We use the continuous risk score as Y_hat (the paper
    # specifies the continuous output layer, not the discrete prediction).
    log.info("=" * 60)
    log.info("PHASE 1: DETECTION  —  training adversary on CDS outputs")
    log.info("=" * 60)

    X_detect = np.column_stack([risk_scores, true_diseased.astype(float)])
    X_detect_norm, _, _ = _standardize(X_detect)

    detection_adv, det_losses = _train_fresh_adversary(
        X_detect_norm, y_sex,
        hidden_dim=ADVERSARIAL_HIDDEN_DIM,
        lr=ADVERSARIAL_LR,
        epochs=ADVERSARIAL_EPOCHS,
    )
    result.adversary_accuracy_before = detection_adv.accuracy(X_detect_norm, y_sex)
    result.training_losses = det_losses

    # ── Original metrics (computed early for the fairness-gap fallback) ─────
    orig_m = _compute_group_metrics(original_decisions, true_diseased, sex_labels)
    result.original_overall_accuracy = orig_m["overall_accuracy"]
    result.original_male_accuracy = orig_m["male"]["accuracy"]
    result.original_female_accuracy = orig_m["female"]["accuracy"]
    result.original_tpr_male = orig_m["male"]["tpr"]
    result.original_tpr_female = orig_m["female"]["tpr"]
    result.original_fpr_male = orig_m["male"]["fpr"]
    result.original_fpr_female = orig_m["female"]["fpr"]

    tpr_gap = abs(orig_m["male"]["tpr"] - orig_m["female"]["tpr"])
    fpr_gap = abs(orig_m["male"]["fpr"] - orig_m["female"]["fpr"])

    # Signal detection: adversary must exceed base rate + margin,
    # OR the decision-level TPR/FPR gaps are large enough to warrant
    # calibration.  The paper (Zhang et al. Section 3) always runs the
    # adversary — we only skip if BOTH the adversary AND the actual
    # fairness gaps agree that no signal exists.
    signal_margin = 0.03
    fairness_gap_threshold = 0.05  # 5% TPR or FPR gap triggers calibration
    adv_detected = (result.adversary_accuracy_before > base_rate + signal_margin)
    gap_detected = (tpr_gap > fairness_gap_threshold or fpr_gap > fairness_gap_threshold)
    result.gender_signal_detected = adv_detected or gap_detected

    log.info(f"  Adversary input      : (risk_score, Y_true)  [equalized odds]")
    log.info(f"  Adversary accuracy   : {result.adversary_accuracy_before:.4f}")
    log.info(f"  Chance baseline      : {base_rate:.4f}")
    log.info(f"  Adversary signal     : {'DETECTED' if adv_detected else 'not detected'}")
    log.info(f"  TPR gap={tpr_gap:.4f}  FPR gap={fpr_gap:.4f}  "
             f"-> fairness gap {'DETECTED' if gap_detected else 'within tolerance'}")

    if not result.gender_signal_detected:
        log.info("  No gender signal (adversary + fairness gaps) — skipping calibration.")
        result.debiased_overall_accuracy = result.original_overall_accuracy
        result.debiased_male_accuracy = result.original_male_accuracy
        result.debiased_female_accuracy = result.original_female_accuracy
        result.debiased_tpr_male = result.original_tpr_male
        result.debiased_tpr_female = result.original_tpr_female
        result.debiased_fpr_male = result.original_fpr_male
        result.debiased_fpr_female = result.original_fpr_female
        result.adversary_accuracy_after = result.adversary_accuracy_before
        return result

    log.info(f"  Original TPR  male={orig_m['male']['tpr']:.4f}  "
             f"female={orig_m['female']['tpr']:.4f}  gap={tpr_gap:.4f}")
    log.info(f"  Original FPR  male={orig_m['male']['fpr']:.4f}  "
             f"female={orig_m['female']['fpr']:.4f}  gap={fpr_gap:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2:  ADVERSARY-GUIDED CALIBRATION
    # ══════════════════════════════════════════════════════════════════════════
    # Non-differentiable analog of Zhang Eq. 1 gradient reversal.
    #
    # CDS risk scores are BIMODAL: non-alarm users cluster near 0, alarm
    # users cluster near 0.5+, with an empty gap between ~0.04 and ~0.5.
    # A pure finite-difference gradient approach fails because perturbations
    # in the gap change zero decisions (gradient = 0).
    #
    # We therefore place candidate thresholds at SCORE QUANTILES — points
    # where the risk scores actually exist — so that each threshold shift
    # meaningfully changes some users' classifications.  For each candidate
    # (theta_m, theta_f) pair, we train a calibration adversary on
    # (decision, Y_true) -> sex  and use the adversary's accuracy as the
    # PRIMARY optimisation objective (adversary-guided).
    #
    # Objective (mirrors Eq. 1  LP - alpha * LA):
    #   min  |adv_acc - 0.5|  +  lambda_acc * accuracy_loss
    #         ~~~~~~~~~~~~~~      ~~~~~~~~~~~~~~~~~~~~~~~~~
    #         maximise LA          preserve LP (predictor quality)
    #
    # The calibration adversary is per Section 7: "a very simple adversary
    # suffices regardless of the complexity of the underlying model" for
    # discrete output and protected variables.
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 2: CALIBRATION  —  adversary-guided threshold search")
    log.info("=" * 60)

    # ── Build candidate thresholds from score quantiles ──────────────────
    # Place thresholds at unique score midpoints so each candidate
    # actually changes the classification of at least one user.
    unique_scores = np.unique(risk_scores)
    # Midpoints between consecutive unique scores
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    # Also include extremes (classify-all and classify-none)
    candidates = np.concatenate([
        [unique_scores[0] - 0.01],   # below all scores: everyone UNHEALTHY
        midpoints,
        [unique_scores[-1] + 0.01],  # above all scores: everyone HEALTHY
    ])

    # Subsample if too many (cap at ~25 per group for 625 pairs max)
    max_per_group = min(25, ADVERSARIAL_THRESHOLD_SEARCH_STEPS)
    if len(candidates) > max_per_group:
        idx = np.linspace(0, len(candidates) - 1, max_per_group, dtype=int)
        candidates = candidates[idx]

    n_cand = len(candidates)
    min_accuracy = result.original_overall_accuracy - 0.05  # max 5pp accuracy drop
    cal_epochs = max(40, ADVERSARIAL_EPOCHS // 2)

    best_theta_m = candidates[n_cand // 2]
    best_theta_f = candidates[n_cand // 2]
    best_objective = float("inf")

    log.info(f"  Threshold candidates : {n_cand} per group ({n_cand**2} pairs)")
    log.info(f"  Score range          : [{unique_scores[0]:.4f}, {unique_scores[-1]:.4f}]")
    log.info(f"  Accuracy floor       : {min_accuracy:.4f}")

    n_evaluated = 0
    for i, tm in enumerate(candidates):
        for j, tf in enumerate(candidates):
            # Compute decisions with candidate thresholds
            dec = _compute_decisions(risk_scores, sex_labels, tm, tf)

            # Accuracy constraint (analog of LP in Eq. 1)
            metrics = _compute_group_metrics(dec, true_diseased, sex_labels)
            acc = metrics["overall_accuracy"]
            if acc < min_accuracy:
                continue

            # Build adversary input: (decision, Y_true) per Zhang Section 3
            X_cal = np.column_stack([dec.astype(float), true_diseased.astype(float)])
            X_cal_norm, _, _ = _standardize(X_cal)

            # Train calibration adversary
            cal_adv, _ = _train_fresh_adversary(
                X_cal_norm, y_sex,
                hidden_dim=max(8, ADVERSARIAL_HIDDEN_DIM // 2),
                lr=ADVERSARIAL_LR,
                epochs=cal_epochs,
                seed=42 + i * n_cand + j,
            )
            adv_acc = cal_adv.accuracy(X_cal_norm, y_sex)

            curr_tpr_gap = abs(metrics["male"]["tpr"] - metrics["female"]["tpr"])
            curr_fpr_gap = abs(metrics["male"]["fpr"] - metrics["female"]["fpr"])
            acc_loss = max(0.0, result.original_overall_accuracy - acc)

            # Objective: adversary close to chance + preserve accuracy
            # Mirrors Eq. 1: balance LP (accuracy) against LA (adversary)
            objective = (abs(adv_acc - 0.5)
                         + 2.0 * acc_loss
                         + 0.5 * curr_tpr_gap
                         + 0.5 * curr_fpr_gap)

            if objective < best_objective:
                best_objective = objective
                best_theta_m = tm
                best_theta_f = tf

            n_evaluated += 1
            result.iteration_log.append({
                "iteration": n_evaluated,
                "theta_m": tm, "theta_f": tf,
                "adv_accuracy": adv_acc,
                "overall_accuracy": acc,
                "tpr_gap": curr_tpr_gap, "fpr_gap": curr_fpr_gap,
            })

        # Progress logging every few rows
        if i % max(1, n_cand // 5) == 0:
            log.info(f"  Search progress: row {i+1}/{n_cand}  "
                     f"evaluated={n_evaluated}  best_obj={best_objective:.4f}")

    result.n_calibration_iterations = n_evaluated
    result.threshold_male = best_theta_m
    result.threshold_female = best_theta_f
    result.threshold_offset_male = best_theta_m     # legacy alias for report
    result.threshold_offset_female = best_theta_f   # legacy alias for report

    # Check convergence: verify best point's adversary is near chance
    best_dec = _compute_decisions(risk_scores, sex_labels, best_theta_m, best_theta_f)
    X_best = np.column_stack([best_dec.astype(float), true_diseased.astype(float)])
    X_best_norm, _, _ = _standardize(X_best)
    best_adv, _ = _train_fresh_adversary(
        X_best_norm, y_sex,
        hidden_dim=max(8, ADVERSARIAL_HIDDEN_DIM // 2),
        lr=ADVERSARIAL_LR, epochs=cal_epochs, seed=777,
    )
    best_adv_acc = best_adv.accuracy(X_best_norm, y_sex)
    convergence_tol = 0.03
    result.calibration_converged = (best_adv_acc <= base_rate + convergence_tol)

    log.info(f"  Best thresholds: male={best_theta_m:.4f}  female={best_theta_f:.4f}")
    log.info(f"  Best adv accuracy: {best_adv_acc:.4f}  "
             f"(converged={result.calibration_converged})")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3:  VERIFICATION  (empirical Proposition 5.1)
    # ══════════════════════════════════════════════════════════════════════════
    # Train a FRESH adversary from scratch on the corrected outputs.
    # If it cannot predict sex => Proposition 5.1 holds empirically:
    # "the adversary gains no advantage from using Y_hat."
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 3: VERIFICATION  —  fresh adversary on corrected outputs")
    log.info("=" * 60)

    final_decisions = _compute_decisions(risk_scores, sex_labels, best_theta_m, best_theta_f)

    # Verification adversary: (decision, Y_true) -> sex  [equalized odds]
    X_verify = np.column_stack([
        final_decisions.astype(float),
        true_diseased.astype(float),
    ])
    X_verify_norm, _, _ = _standardize(X_verify)

    verify_adv, _ = _train_fresh_adversary(
        X_verify_norm, y_sex,
        hidden_dim=ADVERSARIAL_HIDDEN_DIM,
        lr=ADVERSARIAL_LR,
        epochs=ADVERSARIAL_EPOCHS,
        seed=999,   # distinct seed from all calibration adversaries
    )
    result.adversary_accuracy_after = verify_adv.accuracy(X_verify_norm, y_sex)
    if not result.calibration_converged:
        result.calibration_converged = (result.adversary_accuracy_after <= base_rate + convergence_tol)

    # ── Final metrics ───────────────────────────────────────────────────────
    final_m = _compute_group_metrics(final_decisions, true_diseased, sex_labels)
    result.debiased_overall_accuracy = final_m["overall_accuracy"]
    result.debiased_male_accuracy = final_m["male"]["accuracy"]
    result.debiased_female_accuracy = final_m["female"]["accuracy"]
    result.debiased_tpr_male = final_m["male"]["tpr"]
    result.debiased_tpr_female = final_m["female"]["tpr"]
    result.debiased_fpr_male = final_m["male"]["fpr"]
    result.debiased_fpr_female = final_m["female"]["fpr"]
    result.n_predictions_changed = int((final_decisions != original_decisions).sum())

    # ── Summary ─────────────────────────────────────────────────────────────
    log.info(f"  Adversary accuracy  : {result.adversary_accuracy_before:.4f} -> "
             f"{result.adversary_accuracy_after:.4f}  (chance={base_rate:.4f})")
    log.info(f"  Overall accuracy    : {result.original_overall_accuracy:.4f} -> "
             f"{result.debiased_overall_accuracy:.4f}")
    log.info(f"  TPR gap             : "
             f"{abs(result.original_tpr_male - result.original_tpr_female):.4f} -> "
             f"{abs(result.debiased_tpr_male - result.debiased_tpr_female):.4f}")
    log.info(f"  FPR gap             : "
             f"{abs(result.original_fpr_male - result.original_fpr_female):.4f} -> "
             f"{abs(result.debiased_fpr_male - result.debiased_fpr_female):.4f}")
    log.info(f"  Thresholds          : male={best_theta_m:.4f}  female={best_theta_f:.4f}")
    log.info(f"  Predictions changed : {result.n_predictions_changed}")
    log.info(f"  Converged           : {result.calibration_converged}")

    return result
