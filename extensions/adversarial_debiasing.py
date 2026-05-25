"""
adversarial_debiasing.py
========================
Adversarial Debiasing for CDS (Zhang et al., 2018 adapted).

The standard adversarial debiasing trains a classifier jointly with an
adversary that predicts the protected attribute from the classifier's
outputs.  Gradient reversal minimises the adversary's ability to infer
sex from predictions.

CDS is not a differentiable neural network, so we adapt the idea as a
POST-PROCESSING calibration step:

  1. Run Algorithm 4 on ALL users (LOOCV), collecting per-user features:
       (AF_real, rw_real, n_actions_applied, max_focus_reached, decision)
  2. Train a small adversary MLP:  features -> P(sex=female)
  3. If the adversary can predict sex above chance, the CDS outputs carry
     gender signal.  We then search for per-group diagnostic threshold
     offsets that minimise the adversary's accuracy while preserving
     overall accuracy.
  4. Return adjusted thresholds and re-scored predictions.

This is a TRUE adversarial debiasing implementation — the adversary is an
actual neural network, not just the RL penalty from Algorithm4.py.

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
# SECTION 1 — NUMPY-ONLY ADVERSARY MLP
# ─────────────────────────────────────────────────────────────────────────────
# No PyTorch/TensorFlow dependency — pure numpy forward/backward.

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
    Two-layer MLP: input -> hidden (ReLU) -> output (sigmoid).

    Predicts P(sex=female) from CDS prediction features.
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
        n = X.shape[0]
        out, (z1, h1) = self.forward(X)
        loss = _bce_loss(out.ravel(), y)

        # Backward
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
        preds = (self.predict_proba(X) >= 0.5).astype(int)
        return (preds == y).mean()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE EXTRACTION FROM ALGORITHM 4 OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdversarialDebiasResult:
    """Result of running adversarial debiasing on CDS predictions."""
    adversary_accuracy_before: float = 0.0
    adversary_accuracy_after: float = 0.0
    threshold_offset_male: float = 0.0
    threshold_offset_female: float = 0.0
    n_predictions_changed: int = 0
    original_overall_accuracy: float = 0.0
    debiased_overall_accuracy: float = 0.0
    original_male_accuracy: float = 0.0
    original_female_accuracy: float = 0.0
    debiased_male_accuracy: float = 0.0
    debiased_female_accuracy: float = 0.0
    training_losses: List[float] = field(default_factory=list)


def extract_adversary_features(records, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature vectors for the adversary from Algorithm4 prediction records.

    Features per user:
      0: margin_risk_score (continuous risk score from compute_margin_risk_scores)
      1: total_actions_applied (number of sensor activations)
      2: max_focus_reached
      3: decision_encoded (0=HEALTHY, 1=UNHEALTHY, 2=SCREENING)
      4: has_alarm (1 if alarm triggered, 0 otherwise)

    NOTE: Replaces AF_real/rw_real (both non-discriminating — rw is always 0.0
    for non-alarm users and >0.19 for alarm users) with the continuous
    margin-based risk score which captures how close each user was to
    triggering an alarm.

    Target: sex (0=male, 1=female)
    """
    from Algorithm4 import HealthDecision, compute_margin_risk_scores

    decision_map = {
        HealthDecision.HEALTHY: 0,
        HealthDecision.UNHEALTHY: 1,
        HealthDecision.SCREENING: 2,
        HealthDecision.UNKNOWN: 2,
    }

    risk_scores = compute_margin_risk_scores(records)

    n = len(records)
    X = np.zeros((n, 5), dtype=float)
    y = np.zeros(n, dtype=float)

    for i, r in enumerate(records):
        X[i, 0] = risk_scores[i]
        X[i, 1] = r.total_actions_applied
        X[i, 2] = r.max_focus_reached
        X[i, 3] = decision_map.get(r.decision, 2)
        X[i, 4] = 1.0 if r.alarm_class is not None else 0.0
        y[i] = data[r.user_global_idx, SEX_FEATURE_INDEX]

    # Standardize features
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X = (X - mu) / std

    return X, y, mu, std, risk_scores


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ADVERSARIAL DEBIASING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_adversarial_debiasing(
    output,  # Algorithm4Output
    data: np.ndarray,
    labels: np.ndarray,
    diagnostic_threshold: float = 0.025,
) -> AdversarialDebiasResult:
    """
    Run adversarial debiasing on completed Algorithm 4 predictions.

    Uses MARGIN-BASED RISK SCORES instead of rw values.  The original rw
    values are bimodal (0.0 for all non-alarm users, >0.19 for alarm users)
    with no discrimination power.  Margin-based scores measure how close
    each user's feature values are to the healthy-range boundary, giving a
    continuous distribution that allows meaningful threshold adjustment.

    Steps:
      1. Extract features (including margin risk scores) from records.
      2. Train adversary to predict sex from CDS outputs.
      3. Search for per-group margin thresholds that reduce the adversary's
         ability to predict sex while preserving accuracy.
      4. Re-score ALL predictions (including alarm-triggered ones) using
         the margin-based threshold.

    Parameters
    ----------
    output : Algorithm4Output with completed prediction records.
    data   : full dataset array.
    labels : full labels array.
    diagnostic_threshold : base threshold (0.025 from paper, used for reporting).

    Returns
    -------
    AdversarialDebiasResult with before/after metrics.
    """
    from Algorithm4 import HealthDecision

    result = AdversarialDebiasResult()
    records = output.records

    if len(records) < 10:
        log.warning("Too few records for adversarial debiasing")
        return result

    X, y_sex, feat_mu, feat_std, risk_scores = extract_adversary_features(records, data)

    # Step 1: Train adversary
    adversary = AdversaryMLP(
        input_dim=X.shape[1],
        hidden_dim=ADVERSARIAL_HIDDEN_DIM,
        lr=ADVERSARIAL_LR,
    )

    for epoch in range(ADVERSARIAL_EPOCHS):
        loss = adversary.train_step(X, y_sex)
        result.training_losses.append(loss)

    result.adversary_accuracy_before = adversary.accuracy(X, y_sex)
    log.info(f"Adversary accuracy (before debiasing): {result.adversary_accuracy_before:.4f}")

    # Step 2: Compute original per-group accuracy
    male_correct = 0
    male_total = 0
    female_correct = 0
    female_total = 0
    for r in records:
        sex = data[r.user_global_idx, SEX_FEATURE_INDEX]
        if sex == MALE_CODE:
            male_total += 1
            if r.is_correct:
                male_correct += 1
        else:
            female_total += 1
            if r.is_correct:
                female_correct += 1

    result.original_overall_accuracy = sum(1 for r in records if r.is_correct) / len(records)
    result.original_male_accuracy = male_correct / male_total if male_total > 0 else 0
    result.original_female_accuracy = female_correct / female_total if female_total > 0 else 0

    # Step 3: Search for per-group MARGIN thresholds
    # Piecewise risk scores: non-alarm users score 0–0.04, alarm users score 0.5+.
    # The alarm boundary is at ~0.25 (midpoint of the gap).
    # Search a range around the boundary to find gender-specific thresholds.
    BASELINE_MARGIN_THRESHOLD = 0.25
    best_threshold_m = BASELINE_MARGIN_THRESHOLD
    best_threshold_f = BASELINE_MARGIN_THRESHOLD
    best_score = float("inf")

    n_steps = ADVERSARIAL_THRESHOLD_SEARCH_STEPS
    # Search range: 0.0 to 0.7 — covers non-alarm zone through alarm zone
    # Lower threshold → more users classified UNHEALTHY (aggressive)
    # Higher threshold → fewer users classified UNHEALTHY (lenient)
    threshold_candidates = np.linspace(0.0, 0.7, n_steps)

    decision_map = {
        HealthDecision.HEALTHY: 0,
        HealthDecision.UNHEALTHY: 1,
        HealthDecision.SCREENING: 2,
    }

    for threshold_m in threshold_candidates:
        for threshold_f in threshold_candidates:

            # Re-score predictions using MARGIN-BASED thresholds
            # This overrides even alarm decisions: a user whose margin score
            # is above the threshold is UNHEALTHY, below is HEALTHY.
            new_decisions = []
            for idx, r in enumerate(records):
                sex = data[r.user_global_idx, SEX_FEATURE_INDEX]
                thresh = threshold_m if sex == MALE_CODE else threshold_f
                score = risk_scores[idx]

                if score > thresh:
                    new_decisions.append(HealthDecision.UNHEALTHY)
                else:
                    new_decisions.append(HealthDecision.HEALTHY)

            # Rebuild adversary features with new decisions
            X_new = X.copy()
            for i, d in enumerate(new_decisions):
                X_new[i, 3] = (decision_map.get(d, 2) - feat_mu[3]) / feat_std[3]

            # Train a FRESH adversary on the modified features
            fresh_adversary = AdversaryMLP(
                input_dim=X_new.shape[1],
                hidden_dim=ADVERSARIAL_HIDDEN_DIM,
                lr=ADVERSARIAL_LR,
            )
            for _ in range(ADVERSARIAL_EPOCHS):
                fresh_adversary.train_step(X_new, y_sex)
            adv_acc = fresh_adversary.accuracy(X_new, y_sex)

            # Compute per-group accuracy to measure fairness
            m_correct = m_total = f_correct = f_total = 0
            for i, r in enumerate(records):
                true_label = r.true_label
                d = new_decisions[i]
                sex = data[r.user_global_idx, SEX_FEATURE_INDEX]
                if true_label == 1:
                    is_correct = (d != HealthDecision.UNHEALTHY)
                else:
                    is_correct = (d == HealthDecision.UNHEALTHY)
                if sex == MALE_CODE:
                    m_total += 1
                    if is_correct:
                        m_correct += 1
                else:
                    f_total += 1
                    if is_correct:
                        f_correct += 1

            acc_new = (m_correct + f_correct) / len(records)
            acc_loss = max(0, result.original_overall_accuracy - acc_new)
            acc_m = m_correct / m_total if m_total > 0 else 0
            acc_f = f_correct / f_total if f_total > 0 else 0
            gender_gap = abs(acc_m - acc_f)

            # Objective: minimize adversary accuracy + accuracy loss + gender gap
            # The gender_gap term creates incentive to find thresholds that
            # equalise accuracy across groups, even if the adversary can still
            # predict sex from non-decision features.
            score_val = (abs(adv_acc - 0.5)
                         + 2.0 * acc_loss
                         + 1.5 * gender_gap)

            if score_val < best_score:
                best_score = score_val
                best_threshold_m = threshold_m
                best_threshold_f = threshold_f

    # Report as offsets from the baseline alarm boundary (0.0)
    result.threshold_offset_male = best_threshold_m
    result.threshold_offset_female = best_threshold_f

    # Step 4: Apply best thresholds and compute final metrics
    n_changed = 0
    male_correct_new = 0
    male_total_new = 0
    female_correct_new = 0
    female_total_new = 0

    final_decisions = []
    for idx, r in enumerate(records):
        sex = data[r.user_global_idx, SEX_FEATURE_INDEX]
        thresh = best_threshold_m if sex == MALE_CODE else best_threshold_f
        score = risk_scores[idx]

        # Apply margin-based threshold to ALL users (including alarm users)
        if score > thresh:
            new_d = HealthDecision.UNHEALTHY
        else:
            new_d = HealthDecision.HEALTHY

        if new_d != r.decision:
            n_changed += 1
        final_decisions.append(new_d)

        true_label = r.true_label
        if true_label == 1:
            is_correct_new = (new_d != HealthDecision.UNHEALTHY)
        else:
            is_correct_new = (new_d == HealthDecision.UNHEALTHY)

        if sex == MALE_CODE:
            male_total_new += 1
            if is_correct_new:
                male_correct_new += 1
        else:
            female_total_new += 1
            if is_correct_new:
                female_correct_new += 1

    result.n_predictions_changed = n_changed
    result.debiased_overall_accuracy = (male_correct_new + female_correct_new) / len(records)
    result.debiased_male_accuracy = male_correct_new / male_total_new if male_total_new > 0 else 0
    result.debiased_female_accuracy = female_correct_new / female_total_new if female_total_new > 0 else 0

    # Re-evaluate: train a FRESH adversary on debiased outputs
    X_debiased = X.copy()
    for i, d in enumerate(final_decisions):
        X_debiased[i, 3] = (decision_map.get(d, 2) - feat_mu[3]) / feat_std[3]
    fresh_eval = AdversaryMLP(
        input_dim=X_debiased.shape[1],
        hidden_dim=ADVERSARIAL_HIDDEN_DIM,
        lr=ADVERSARIAL_LR,
    )
    for _ in range(ADVERSARIAL_EPOCHS):
        fresh_eval.train_step(X_debiased, y_sex)
    result.adversary_accuracy_after = fresh_eval.accuracy(X_debiased, y_sex)

    log.info(f"Adversary accuracy (after debiasing): {result.adversary_accuracy_after:.4f}")
    log.info(f"Margin thresholds: male={best_threshold_m:+.4f}, female={best_threshold_f:+.4f}")
    log.info(f"Predictions changed: {n_changed}")
    log.info(f"Accuracy: {result.original_overall_accuracy:.4f} -> {result.debiased_overall_accuracy:.4f}")

    return result
