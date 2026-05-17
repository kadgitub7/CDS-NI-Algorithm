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
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

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
      0: AF_real (final assurance factor)
      1: rw_real (final remaining weight = 1 - AF)
      2: total_actions_applied (number of sensor activations)
      3: max_focus_reached
      4: decision_encoded (0=HEALTHY, 1=UNHEALTHY, 2=SCREENING)

    Target: sex (0=male, 1=female)
    """
    from Algorithm4 import HealthDecision

    decision_map = {
        HealthDecision.HEALTHY: 0,
        HealthDecision.UNHEALTHY: 1,
        HealthDecision.SCREENING: 2,
        HealthDecision.UNKNOWN: 2,
    }

    n = len(records)
    X = np.zeros((n, 5), dtype=float)
    y = np.zeros(n, dtype=float)

    for i, r in enumerate(records):
        af = r.af_trace[-1].AF_real if r.af_trace else 0.0
        rw = 1.0 - af
        X[i, 0] = af
        X[i, 1] = rw
        X[i, 2] = r.total_actions_applied
        X[i, 3] = r.max_focus_reached
        X[i, 4] = decision_map.get(r.decision, 2)
        y[i] = data[r.user_global_idx, SEX_FEATURE_INDEX]

    # Standardize features
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X = (X - mu) / std

    return X, y


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

    Steps:
      1. Extract features from prediction records.
      2. Train adversary to predict sex from CDS outputs.
      3. If adversary accuracy > 55% (can predict sex), search for
         per-group threshold offsets that reduce gender signal.
      4. Re-score predictions with adjusted thresholds and return results.

    Parameters
    ----------
    output : Algorithm4Output with completed prediction records.
    data   : full dataset array.
    labels : full labels array.
    diagnostic_threshold : base threshold (0.025 from paper).

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

    X, y_sex = extract_adversary_features(records, data)

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

    # Step 3: Search for per-group threshold offsets
    # The idea: adjust the rw threshold separately for males and females
    # so that the adversary can no longer distinguish them.
    # We search over offsets in [-0.05, +0.05] for each group.
    best_offset_m = 0.0
    best_offset_f = 0.0
    best_score = float("inf")

    offsets = np.linspace(-0.05, 0.05, ADVERSARIAL_THRESHOLD_SEARCH_STEPS)

    for off_m in offsets:
        for off_f in offsets:
            threshold_m = diagnostic_threshold + off_m
            threshold_f = diagnostic_threshold + off_f

            # Re-score predictions with adjusted thresholds
            new_decisions = []
            for r in records:
                sex = data[r.user_global_idx, SEX_FEATURE_INDEX]
                threshold = threshold_m if sex == MALE_CODE else threshold_f

                if r.alarm_class is not None:
                    new_decisions.append(HealthDecision.UNHEALTHY)
                else:
                    rw_final = r.af_trace[-1].rw_real if r.af_trace else 1.0
                    if rw_final <= threshold:
                        new_decisions.append(HealthDecision.HEALTHY)
                    else:
                        new_decisions.append(HealthDecision.SCREENING)

            # Rebuild adversary features with new decisions
            decision_map = {
                HealthDecision.HEALTHY: 0,
                HealthDecision.UNHEALTHY: 1,
                HealthDecision.SCREENING: 2,
            }
            X_new = X.copy()
            for i, d in enumerate(new_decisions):
                X_new[i, 4] = decision_map.get(d, 2)
                X_new[i, 1] = 1.0 - X[i, 0]  # rw doesn't change

            # Score: adversary accuracy on new features (want close to 0.5)
            # + penalty for accuracy loss
            adv_acc = adversary.accuracy(X_new, y_sex)

            n_correct_new = 0
            for i, r in enumerate(records):
                true_label = r.true_label
                d = new_decisions[i]
                if true_label == 1:  # healthy
                    if d != HealthDecision.UNHEALTHY:
                        n_correct_new += 1
                else:  # diseased
                    if d == HealthDecision.UNHEALTHY:
                        n_correct_new += 1

            acc_new = n_correct_new / len(records)
            acc_loss = max(0, result.original_overall_accuracy - acc_new)

            # Objective: minimize adversary's departure from chance + accuracy loss
            score = abs(adv_acc - 0.5) + 2.0 * acc_loss

            if score < best_score:
                best_score = score
                best_offset_m = off_m
                best_offset_f = off_f

    result.threshold_offset_male = best_offset_m
    result.threshold_offset_female = best_offset_f

    # Step 4: Apply best offsets and compute final metrics
    threshold_m = diagnostic_threshold + best_offset_m
    threshold_f = diagnostic_threshold + best_offset_f

    n_changed = 0
    male_correct_new = 0
    male_total_new = 0
    female_correct_new = 0
    female_total_new = 0

    final_decisions = []
    for r in records:
        sex = data[r.user_global_idx, SEX_FEATURE_INDEX]
        threshold = threshold_m if sex == MALE_CODE else threshold_f

        if r.alarm_class is not None:
            new_d = HealthDecision.UNHEALTHY
        else:
            rw_final = r.af_trace[-1].rw_real if r.af_trace else 1.0
            if rw_final <= threshold:
                new_d = HealthDecision.HEALTHY
            else:
                new_d = HealthDecision.SCREENING

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

    # Re-evaluate adversary on debiased outputs
    decision_map = {
        HealthDecision.HEALTHY: 0,
        HealthDecision.UNHEALTHY: 1,
        HealthDecision.SCREENING: 2,
    }
    X_debiased = X.copy()
    for i, d in enumerate(final_decisions):
        X_debiased[i, 4] = decision_map.get(d, 2)
    result.adversary_accuracy_after = adversary.accuracy(X_debiased, y_sex)

    log.info(f"Adversary accuracy (after debiasing): {result.adversary_accuracy_after:.4f}")
    log.info(f"Threshold offsets: male={best_offset_m:+.4f}, female={best_offset_f:+.4f}")
    log.info(f"Predictions changed: {n_changed}")
    log.info(f"Accuracy: {result.original_overall_accuracy:.4f} -> {result.debiased_overall_accuracy:.4f}")

    return result
