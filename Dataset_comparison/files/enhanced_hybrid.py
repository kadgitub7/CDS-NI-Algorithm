"""
================================================================================
Enhanced Hybrid CDS-ANN-RNN Model
================================================================================

Adaptive classifier with 5 hybrid modes for combining CDS (Cognitive Dynamic
System), ANN (MLP Neural Network), and RNN (Echo State Network / Reservoir
Computing) to maximize arrhythmia classification on the UCI dataset.

MODELS
------
  CDS  : Cognitive Dynamic System (Algorithms 1-4) - Bayesian range-based
  ANN  : Multi-Layer Perceptron (sklearn) - feedforward neural network
  RNN  : Echo State Network (reservoir computing) - recurrent network
         Inspired by optical reservoir computing (Peng et al., 2024)
         State update:  x(n) = (1-a)*x(n-1) + a*tanh(W_in*u(n) + W*x(n-1))
         Output:        y(n) = W_out * x(n)
         Training:      Ridge regression (no backpropagation needed)

HYBRID MODES (--hybrid-mode)
-----------------------------
  1. CASCADE     - CDS first, then RNN/ANN for uncertain cases only
  2. VOTE        - CDS + ANN + RNN vote, majority wins
  3. CONFIDENCE  - Weighted combination based on each model's confidence
  4. STACKED     - CDS features (AF, rw, alarm) fed as inputs to ANN/RNN
  5. SELECTOR    - Meta-learner picks best model per patient profile

USAGE
-----
  python enhanced_hybrid.py --mode all --max-users 20
  python enhanced_hybrid.py --mode hybrid --hybrid-mode cascade --max-users 50
  python enhanced_hybrid.py --mode hybrid --hybrid-mode vote --max-users 50
  python enhanced_hybrid.py --mode rnn-only --max-users 50
  python enhanced_hybrid.py --mode compare-hybrid --max-users 20

================================================================================
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))

from reservoir_network import EchoStateNetwork

log = logging.getLogger("Enhanced.Hybrid")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    log.addHandler(h)
    log.propagate = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Prediction:
    """Single user prediction result."""
    user_idx: int
    true_label: int
    predicted_label: int
    source: str
    is_correct: bool = False
    cds_decision: str = ""
    cds_af: float = 0.0
    cds_rw: float = 1.0
    ann_prediction: int = -1
    ann_confidence: float = 0.0
    rnn_prediction: int = -1
    rnn_confidence: float = 0.0
    elapsed_ms: float = 0.0


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    mode: str
    predictions: List[Prediction] = field(default_factory=list)
    accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    source_counts: Dict[str, int] = field(default_factory=dict)
    total_time_s: float = 0.0


# ============================================================================
# ANN COMPONENT (MLP)
# ============================================================================

class ANNClassifier:
    """MLP classifier with preprocessing pipeline."""

    def __init__(self, hidden_layers: Tuple[int, ...] = (64, 32),
                 max_iter: int = 500, random_state: int = 42):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate="adaptive",
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ANNClassifier":
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean = self.imputer.transform(X)
        return self.clf.predict(self.scaler.transform(X_clean))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean = self.imputer.transform(X)
        return self.clf.predict_proba(self.scaler.transform(X_clean))

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[int, float]:
        proba = self.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred_class = self.clf.classes_[pred_idx]
        return int(pred_class), float(proba[pred_idx])


# ============================================================================
# CDS COMPONENT
# ============================================================================

def _suppress_cds_loggers():
    import logging as _logging
    for name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3", "CDS.Alg4",
                 "CDS.Alg1.ForcedSex", "CDS"):
        lgr = _logging.getLogger(name)
        lgr.setLevel(_logging.CRITICAL)
        for h in lgr.handlers:
            h.setLevel(_logging.CRITICAL)


def _run_cds_pipeline(train_data, train_labels, n_bins=20):
    """Build full CDS pipeline (Algorithms 1-3) on training data."""
    from CDS_Paper_Algorithms import build_decision_tree
    from Algorithm2 import run_algorithm2
    from Algorithm3 import run_algorithm3

    tree = build_decision_tree(train_data, train_labels)
    nodes_filter = ["root"]
    if 2 in tree.nodes_by_level:
        for node in tree.nodes_by_level[2]:
            nodes_filter.append(node.node_id)
            if len(nodes_filter) >= 3:
                break
    alg2_out = run_algorithm2(tree=tree, data=train_data, labels=train_labels,
                              n_bins=n_bins, nodes_filter=nodes_filter)
    alg3_out = run_algorithm3(alg2_output=alg2_out, tree=tree,
                              data=train_data, labels=train_labels,
                              nodes_filter=nodes_filter, reset_per_h=False, verbose=False)
    return tree, alg2_out, alg3_out


def _run_cds_predict(user_idx, data, labels, train_data, train_labels,
                     tree, alg2_out, alg3_out, rng_seed=42):
    """Run CDS Algorithm 4 on one user. Returns (decision, af, rw, record)."""
    from Algorithm4 import run_algorithm4, HealthDecision
    try:
        record = run_algorithm4(
            user_global_idx=user_idx, data=data, labels=labels,
            tree=tree, alg2_output=alg2_out, alg3_output=alg3_out,
            rng_seed=rng_seed, train_data=train_data, train_labels=train_labels,
        )
        af = record.af_trace[-1].AF_real if record.af_trace else 0.0
        return record.decision, af, 1.0 - af, record
    except Exception:
        return HealthDecision.SCREENING, 0.0, 1.0, None


def _cds_decision_to_label(decision, record, labels, healthy_class):
    from Algorithm4 import HealthDecision
    if decision == HealthDecision.UNHEALTHY:
        alarm_class = record.alarm_class if record else None
        if alarm_class is not None:
            return alarm_class
        return max(set(labels) - {healthy_class},
                   key=lambda c: int((labels == c).sum()),
                   default=healthy_class + 1)
    return healthy_class


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_uci_data() -> Tuple[np.ndarray, np.ndarray]:
    from CDS_Paper_Algorithms import load_dataset
    data_path = str(Path(__file__).parent / "data" / "arrhythmia.data")
    if not Path(data_path).exists():
        data_path = str(Path(__file__).parent.parent / "data" / "arrhythmia.data")
    return load_dataset(data_path)


# ============================================================================
# INDIVIDUAL MODEL LOOCV
# ============================================================================

def run_single_model_loocv(data, labels, model_type="ann", max_users=None,
                           healthy_class=1, **model_kwargs) -> EvalResult:
    """Run LOOCV for a single model type: 'ann', 'rnn', or 'cds'."""
    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    result = EvalResult(mode=f"{model_type.upper()}-only")
    t0 = time.time()

    if model_type == "cds":
        _suppress_cds_loggers()
        from Algorithm4 import HealthDecision
        _suppress_cds_loggers()

    log.info("=" * 70)
    log.info(f"{model_type.upper()}-ONLY LOOCV  ({n_total} users)")
    log.info("=" * 70)

    for i in range(n_total):
        t_start = time.perf_counter()
        train_mask = np.ones(data.shape[0], dtype=bool)
        train_mask[i] = False
        train_data, train_labels = data[train_mask], labels[train_mask]
        true_label = int(labels[i])

        pred_label = healthy_class
        source = model_type.upper()
        confidence = 0.5
        pred = Prediction(user_idx=i, true_label=true_label, predicted_label=0, source=source)

        if model_type == "ann":
            ann = ANNClassifier(**{k: v for k, v in model_kwargs.items()
                                  if k in ("hidden_layers", "random_state")})
            ann.fit(train_data, train_labels)
            pred_label, confidence = ann.predict_with_confidence(data[i:i+1])
            pred.ann_prediction = pred_label
            pred.ann_confidence = confidence

        elif model_type == "rnn":
            rnn = EchoStateNetwork(**{k: v for k, v in model_kwargs.items()
                                     if k in ("n_reservoir", "spectral_radius",
                                              "leak_rate", "ridge_alpha")})
            rnn.fit(train_data, train_labels)
            pred_label, confidence = rnn.predict_with_confidence(data[i:i+1])
            pred.rnn_prediction = pred_label
            pred.rnn_confidence = confidence

        elif model_type == "cds":
            try:
                tree, a2, a3 = _run_cds_pipeline(train_data, train_labels)
                decision, af, rw, record = _run_cds_predict(
                    i, data, labels, train_data, train_labels, tree, a2, a3, 42 + i)
                pred_label = _cds_decision_to_label(decision, record, labels, healthy_class)
                pred.cds_decision = decision.value
                pred.cds_af = af
                pred.cds_rw = rw
                if decision == HealthDecision.UNHEALTHY:
                    source = "CDS-ALARM"
                elif decision == HealthDecision.HEALTHY:
                    source = "CDS-HEALTHY"
                else:
                    source = "CDS-SCREENING"
            except Exception:
                source = "CDS-ERROR"

        pred.predicted_label = pred_label
        pred.source = source

        if true_label == healthy_class:
            pred.is_correct = (pred_label == healthy_class)
        else:
            pred.is_correct = (pred_label != healthy_class)

        pred.elapsed_ms = (time.perf_counter() - t_start) * 1000
        result.predictions.append(pred)
        result.source_counts[source] = result.source_counts.get(source, 0) + 1

        if (i + 1) % max(1, n_total // 5) == 0 or i == n_total - 1:
            acc = sum(1 for p in result.predictions if p.is_correct) / len(result.predictions)
            log.info(f"  [{i+1}/{n_total}] acc={acc*100:.1f}%")

    _compute_metrics(result, healthy_class)
    result.total_time_s = time.time() - t0
    return result


# ============================================================================
# HYBRID MODE 1: CASCADE
# ============================================================================

def run_hybrid_cascade(data, labels, max_users=None, healthy_class=1,
                       ml_model="rnn", **kwargs) -> EvalResult:
    """
    MODE 1: CASCADE — CDS runs first; ANN or RNN handles uncertain cases.

    Flow:
      CDS ALARM  -> trust CDS (disease detected)
      CDS HEALTHY -> trust CDS (confident healthy)
      CDS SCREENING -> pass to ANN or RNN for final decision
    """
    _suppress_cds_loggers()
    from Algorithm4 import HealthDecision
    _suppress_cds_loggers()

    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    result = EvalResult(mode=f"Cascade({ml_model.upper()})")
    t0 = time.time()

    log.info("=" * 70)
    log.info(f"HYBRID MODE 1: CASCADE (CDS -> {ml_model.upper()})")
    log.info(f"Users: {n_total}")
    log.info("=" * 70)

    for i in range(n_total):
        t_start = time.perf_counter()
        train_mask = np.ones(data.shape[0], dtype=bool)
        train_mask[i] = False
        train_data, train_labels = data[train_mask], labels[train_mask]
        true_label = int(labels[i])

        tree, a2, a3 = _run_cds_pipeline(train_data, train_labels)
        decision, af, rw, record = _run_cds_predict(
            i, data, labels, train_data, train_labels, tree, a2, a3, 42 + i)

        pred = Prediction(user_idx=i, true_label=true_label, predicted_label=0,
                          source="", cds_af=af, cds_rw=rw,
                          cds_decision=decision.value if hasattr(decision, 'value') else str(decision))

        if decision == HealthDecision.UNHEALTHY:
            pred.predicted_label = _cds_decision_to_label(decision, record, labels, healthy_class)
            pred.source = "CDS-ALARM"
        elif decision == HealthDecision.HEALTHY:
            pred.predicted_label = healthy_class
            pred.source = "CDS-HEALTHY"
        else:
            if ml_model == "rnn":
                model = EchoStateNetwork()
                model.fit(train_data, train_labels)
                ml_pred, ml_conf = model.predict_with_confidence(data[i:i+1])
                pred.rnn_prediction = ml_pred
                pred.rnn_confidence = ml_conf
            else:
                model = ANNClassifier()
                model.fit(train_data, train_labels)
                ml_pred, ml_conf = model.predict_with_confidence(data[i:i+1])
                pred.ann_prediction = ml_pred
                pred.ann_confidence = ml_conf
            pred.predicted_label = ml_pred
            pred.source = f"{ml_model.upper()}"

        _set_correctness(pred, healthy_class)
        pred.elapsed_ms = (time.perf_counter() - t_start) * 1000
        result.predictions.append(pred)
        result.source_counts[pred.source] = result.source_counts.get(pred.source, 0) + 1

        if (i + 1) % max(1, n_total // 5) == 0 or i == n_total - 1:
            acc = sum(1 for p in result.predictions if p.is_correct) / len(result.predictions)
            log.info(f"  [{i+1}/{n_total}] acc={acc*100:.1f}%  sources={result.source_counts}")

    _compute_metrics(result, healthy_class)
    result.total_time_s = time.time() - t0
    return result


# ============================================================================
# HYBRID MODE 2: MAJORITY VOTE
# ============================================================================

def run_hybrid_vote(data, labels, max_users=None, healthy_class=1,
                    ml_model="rnn", **kwargs) -> EvalResult:
    """
    MODE 2: VOTE — CDS and one ML model each vote; tie broken by CDS.

    Both CDS and the selected ML model (ANN or RNN) cast a binary vote
    (healthy vs unhealthy). If they agree, that's the answer. If they
    disagree, CDS wins (medical interpretability as tiebreaker).
    """
    _suppress_cds_loggers()
    from Algorithm4 import HealthDecision
    _suppress_cds_loggers()

    ml_tag = ml_model.upper()
    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    result = EvalResult(mode=f"Vote(CDS+{ml_tag})")
    t0 = time.time()

    log.info("=" * 70)
    log.info(f"HYBRID MODE 2: VOTE (CDS + {ml_tag})")
    log.info(f"Users: {n_total}")
    log.info("=" * 70)

    for i in range(n_total):
        t_start = time.perf_counter()
        train_mask = np.ones(data.shape[0], dtype=bool)
        train_mask[i] = False
        train_data, train_labels = data[train_mask], labels[train_mask]
        true_label = int(labels[i])

        # CDS vote
        tree, a2, a3 = _run_cds_pipeline(train_data, train_labels)
        decision, af, rw, record = _run_cds_predict(
            i, data, labels, train_data, train_labels, tree, a2, a3, 42 + i)
        cds_label = _cds_decision_to_label(decision, record, labels, healthy_class)
        cds_vote = 0 if cds_label == healthy_class else 1

        # ML vote
        if ml_model == "rnn":
            model = EchoStateNetwork()
        else:
            model = ANNClassifier()
        model.fit(train_data, train_labels)
        ml_pred, ml_conf = model.predict_with_confidence(data[i:i+1])
        ml_vote = 0 if ml_pred == healthy_class else 1

        # Decision: agree -> use that; disagree -> CDS wins (tiebreaker)
        if cds_vote == ml_vote:
            if cds_vote == 1:
                final_label = ml_pred if ml_vote == 1 else cds_label
                source = f"VOTE-AGREE-UNHEALTHY"
            else:
                final_label = healthy_class
                source = f"VOTE-AGREE-HEALTHY"
        else:
            final_label = cds_label
            source = f"VOTE-TIE-CDS"

        pred = Prediction(
            user_idx=i, true_label=true_label, predicted_label=final_label,
            source=source, cds_af=af, cds_rw=rw,
            cds_decision=decision.value if hasattr(decision, 'value') else str(decision),
        )
        if ml_model == "rnn":
            pred.rnn_prediction = ml_pred
            pred.rnn_confidence = ml_conf
        else:
            pred.ann_prediction = ml_pred
            pred.ann_confidence = ml_conf

        _set_correctness(pred, healthy_class)
        pred.elapsed_ms = (time.perf_counter() - t_start) * 1000
        result.predictions.append(pred)
        result.source_counts[source] = result.source_counts.get(source, 0) + 1

        if (i + 1) % max(1, n_total // 5) == 0 or i == n_total - 1:
            acc = sum(1 for p in result.predictions if p.is_correct) / len(result.predictions)
            log.info(f"  [{i+1}/{n_total}] acc={acc*100:.1f}%  CDS:{cds_vote} {ml_tag}:{ml_vote}")

    _compute_metrics(result, healthy_class)
    result.total_time_s = time.time() - t0
    return result


# ============================================================================
# HYBRID MODE 3: CONFIDENCE-WEIGHTED
# ============================================================================

def run_hybrid_confidence(data, labels, max_users=None, healthy_class=1,
                          ml_model="rnn", **kwargs) -> EvalResult:
    """
    MODE 3: CONFIDENCE — Weight CDS and one ML model by confidence scores.

    CDS confidence = assurance factor (boosted to 0.8 on ALARM)
    ML confidence  = max class probability from ANN or RNN

    Weighted P(unhealthy) = w_cds * cds_p + w_ml * ml_p
    If weighted > 0.4, predict unhealthy using the higher-confidence source.
    """
    _suppress_cds_loggers()
    from Algorithm4 import HealthDecision
    _suppress_cds_loggers()

    ml_tag = ml_model.upper()
    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    result = EvalResult(mode=f"Confidence(CDS+{ml_tag})")
    t0 = time.time()

    log.info("=" * 70)
    log.info(f"HYBRID MODE 3: CONFIDENCE-WEIGHTED (CDS + {ml_tag})")
    log.info(f"Users: {n_total}")
    log.info("=" * 70)

    for i in range(n_total):
        t_start = time.perf_counter()
        train_mask = np.ones(data.shape[0], dtype=bool)
        train_mask[i] = False
        train_data, train_labels = data[train_mask], labels[train_mask]
        true_label = int(labels[i])

        # CDS
        tree, a2, a3 = _run_cds_pipeline(train_data, train_labels)
        decision, af, rw, record = _run_cds_predict(
            i, data, labels, train_data, train_labels, tree, a2, a3, 42 + i)
        cds_label = _cds_decision_to_label(decision, record, labels, healthy_class)
        cds_conf = af
        if decision == HealthDecision.UNHEALTHY:
            cds_conf = max(cds_conf, 0.8)
        cds_p_unhealthy = 0.0 if cds_label == healthy_class else cds_conf

        # ML model
        if ml_model == "rnn":
            model = EchoStateNetwork()
        else:
            model = ANNClassifier()
        model.fit(train_data, train_labels)
        ml_pred, ml_conf = model.predict_with_confidence(data[i:i+1])
        ml_p_unhealthy = 0.0 if ml_pred == healthy_class else ml_conf

        # Weighted combination (CDS gets higher weight — domain expertise)
        w_cds, w_ml = 0.55, 0.45
        weighted_p_unhealthy = w_cds * cds_p_unhealthy + w_ml * ml_p_unhealthy

        if weighted_p_unhealthy > 0.4:
            candidates = []
            if cds_label != healthy_class:
                candidates.append((cds_conf, cds_label, "CDS"))
            if ml_pred != healthy_class:
                candidates.append((ml_conf, ml_pred, ml_tag))
            if candidates:
                best = max(candidates, key=lambda x: x[0])
                final_label = best[1]
                source = f"CONF-{best[2]}"
            else:
                final_label = healthy_class
                source = "CONF-HEALTHY"
        else:
            final_label = healthy_class
            source = "CONF-HEALTHY"

        pred = Prediction(
            user_idx=i, true_label=true_label, predicted_label=final_label,
            source=source, cds_af=af, cds_rw=rw,
            cds_decision=decision.value if hasattr(decision, 'value') else str(decision),
        )
        if ml_model == "rnn":
            pred.rnn_prediction = ml_pred
            pred.rnn_confidence = ml_conf
        else:
            pred.ann_prediction = ml_pred
            pred.ann_confidence = ml_conf

        _set_correctness(pred, healthy_class)
        pred.elapsed_ms = (time.perf_counter() - t_start) * 1000
        result.predictions.append(pred)
        result.source_counts[source] = result.source_counts.get(source, 0) + 1

        if (i + 1) % max(1, n_total // 5) == 0 or i == n_total - 1:
            acc = sum(1 for p in result.predictions if p.is_correct) / len(result.predictions)
            log.info(f"  [{i+1}/{n_total}] acc={acc*100:.1f}%  wP={weighted_p_unhealthy:.2f}")

    _compute_metrics(result, healthy_class)
    result.total_time_s = time.time() - t0
    return result


# ============================================================================
# HYBRID MODE 4: STACKED (CDS features -> ML)
# ============================================================================

def run_hybrid_stacked(data, labels, max_users=None, healthy_class=1,
                       ml_model="ann", **kwargs) -> EvalResult:
    """
    MODE 4: STACKED — CDS outputs become additional features for ANN/RNN.

    CDS produces per-user signals: AF, rw, alarm_flag, decision_code.
    These are appended to the original feature vector and fed to ANN or RNN.
    The ML model learns when CDS is reliable and when to override it.

    Since we can't run CDS on the test user to generate stacked features
    without leaking, we compute CDS features on training users during training,
    and run CDS on the test user during prediction to get its CDS features.
    """
    _suppress_cds_loggers()
    from Algorithm4 import HealthDecision
    _suppress_cds_loggers()

    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    result = EvalResult(mode=f"Stacked(CDS->{ml_model.upper()})")
    t0 = time.time()

    log.info("=" * 70)
    log.info(f"HYBRID MODE 4: STACKED (CDS features -> {ml_model.upper()})")
    log.info(f"Users: {n_total}")
    log.info("=" * 70)

    for i in range(n_total):
        t_start = time.perf_counter()
        train_mask = np.ones(data.shape[0], dtype=bool)
        train_mask[i] = False
        train_data, train_labels = data[train_mask], labels[train_mask]
        true_label = int(labels[i])

        # Build CDS pipeline
        tree, a2, a3 = _run_cds_pipeline(train_data, train_labels)

        # Get CDS features for test user
        decision, af, rw, record = _run_cds_predict(
            i, data, labels, train_data, train_labels, tree, a2, a3, 42 + i)
        alarm_flag = 1.0 if decision == HealthDecision.UNHEALTHY else 0.0
        healthy_flag = 1.0 if decision == HealthDecision.HEALTHY else 0.0
        screening_flag = 1.0 if decision == HealthDecision.SCREENING else 0.0
        cds_features_test = np.array([[af, rw, alarm_flag, healthy_flag, screening_flag]])

        # Augment test features
        test_augmented = np.hstack([data[i:i+1], cds_features_test])

        # For training: approximate CDS features using simple heuristics
        # (running full CDS on each training user would be too expensive)
        n_train = train_data.shape[0]
        train_cds_features = np.zeros((n_train, 5))
        # Simple proxy: use distance from class centroid as AF proxy
        healthy_mask = train_labels == healthy_class
        if healthy_mask.sum() > 0:
            healthy_centroid = np.nanmean(train_data[healthy_mask], axis=0)
            for j in range(n_train):
                diff = train_data[j] - healthy_centroid
                dist = np.nanmean(np.abs(diff))
                proxy_af = min(1.0, dist / (np.nanstd(diff) + 1e-10) * 0.1)
                is_healthy = train_labels[j] == healthy_class
                train_cds_features[j, 0] = proxy_af
                train_cds_features[j, 1] = 1.0 - proxy_af
                train_cds_features[j, 2] = 0.0 if is_healthy else float(proxy_af > 0.5)
                train_cds_features[j, 3] = 1.0 if is_healthy else 0.0
                train_cds_features[j, 4] = 0.0

        train_augmented = np.hstack([train_data, train_cds_features])

        # Train ML model on augmented features
        if ml_model == "rnn":
            model = EchoStateNetwork(n_reservoir=200)
            model.fit(train_augmented, train_labels)
            pred_label, conf = model.predict_with_confidence(test_augmented)
        else:
            model = ANNClassifier()
            model.fit(train_augmented, train_labels)
            pred_label, conf = model.predict_with_confidence(test_augmented)

        pred = Prediction(
            user_idx=i, true_label=true_label, predicted_label=pred_label,
            source=f"STACKED-{ml_model.upper()}", cds_af=af, cds_rw=rw,
            cds_decision=decision.value if hasattr(decision, 'value') else str(decision),
        )
        _set_correctness(pred, healthy_class)
        pred.elapsed_ms = (time.perf_counter() - t_start) * 1000
        result.predictions.append(pred)
        result.source_counts[pred.source] = result.source_counts.get(pred.source, 0) + 1

        if (i + 1) % max(1, n_total // 5) == 0 or i == n_total - 1:
            acc = sum(1 for p in result.predictions if p.is_correct) / len(result.predictions)
            log.info(f"  [{i+1}/{n_total}] acc={acc*100:.1f}%")

    _compute_metrics(result, healthy_class)
    result.total_time_s = time.time() - t0
    return result


# ============================================================================
# HYBRID MODE 5: SELECTOR (Meta-learner picks best model)
# ============================================================================

def run_hybrid_selector(data, labels, max_users=None, healthy_class=1,
                        ml_model="rnn", **kwargs) -> EvalResult:
    """
    MODE 5: SELECTOR — Adaptive meta-learner picks CDS or ML per user.

    Runs CDS and one ML model (ANN or RNN). Selects which to trust:
      - CDS ALARM -> always trust CDS
      - CDS HEALTHY with high AF -> trust CDS
      - CDS SCREENING -> use ML, weighted by running accuracy
      - Tracks per-model accuracy over time and adjusts trust dynamically
    """
    _suppress_cds_loggers()
    from Algorithm4 import HealthDecision
    _suppress_cds_loggers()

    ml_tag = ml_model.upper()
    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    result = EvalResult(mode=f"Selector(CDS+{ml_tag})")
    t0 = time.time()

    log.info("=" * 70)
    log.info(f"HYBRID MODE 5: SELECTOR (CDS + {ml_tag})")
    log.info(f"Users: {n_total}")
    log.info("=" * 70)

    model_correct = {"CDS": 0, ml_tag: 0}
    model_total = {"CDS": 0, ml_tag: 0}

    for i in range(n_total):
        t_start = time.perf_counter()
        train_mask = np.ones(data.shape[0], dtype=bool)
        train_mask[i] = False
        train_data, train_labels = data[train_mask], labels[train_mask]
        true_label = int(labels[i])

        # CDS
        tree, a2, a3 = _run_cds_pipeline(train_data, train_labels)
        decision, af, rw, record = _run_cds_predict(
            i, data, labels, train_data, train_labels, tree, a2, a3, 42 + i)
        cds_label = _cds_decision_to_label(decision, record, labels, healthy_class)

        # ML
        if ml_model == "rnn":
            model = EchoStateNetwork()
        else:
            model = ANNClassifier()
        model.fit(train_data, train_labels)
        ml_pred, ml_conf = model.predict_with_confidence(data[i:i+1])

        model_total["CDS"] += 1
        model_total[ml_tag] += 1

        def _acc(name):
            if model_total[name] <= 1:
                return 0.5
            return model_correct[name] / (model_total[name] - 1)

        cds_acc = _acc("CDS")
        ml_acc = _acc(ml_tag)

        # Selection logic
        if decision == HealthDecision.UNHEALTHY:
            final_label = cds_label
            source = "SEL-CDS-ALARM"
        elif decision == HealthDecision.HEALTHY and af > 0.5:
            final_label = healthy_class
            source = "SEL-CDS-HEALTHY"
        else:
            ml_score = ml_conf * (0.5 + 0.5 * ml_acc)
            cds_score = af * (0.5 + 0.5 * cds_acc)

            if ml_score >= cds_score:
                final_label = ml_pred
                source = f"SEL-{ml_tag}"
            else:
                final_label = cds_label
                source = "SEL-CDS"

        pred = Prediction(
            user_idx=i, true_label=true_label, predicted_label=final_label,
            source=source, cds_af=af, cds_rw=rw,
            cds_decision=decision.value if hasattr(decision, 'value') else str(decision),
        )
        if ml_model == "rnn":
            pred.rnn_prediction = ml_pred
            pred.rnn_confidence = ml_conf
        else:
            pred.ann_prediction = ml_pred
            pred.ann_confidence = ml_conf

        _set_correctness(pred, healthy_class)
        pred.elapsed_ms = (time.perf_counter() - t_start) * 1000
        result.predictions.append(pred)
        result.source_counts[source] = result.source_counts.get(source, 0) + 1

        # Update running accuracy AFTER prediction
        if true_label == healthy_class:
            cds_ok = (cds_label == healthy_class)
            ml_ok = (ml_pred == healthy_class)
        else:
            cds_ok = (cds_label != healthy_class)
            ml_ok = (ml_pred != healthy_class)
        if cds_ok:
            model_correct["CDS"] += 1
        if ml_ok:
            model_correct[ml_tag] += 1

        if (i + 1) % max(1, n_total // 5) == 0 or i == n_total - 1:
            acc = sum(1 for p in result.predictions if p.is_correct) / len(result.predictions)
            log.info(f"  [{i+1}/{n_total}] acc={acc*100:.1f}%  "
                     f"running: CDS={_acc('CDS'):.2f} {ml_tag}={_acc(ml_tag):.2f}")

    _compute_metrics(result, healthy_class)
    result.total_time_s = time.time() - t0
    return result


# ============================================================================
# HELPERS
# ============================================================================

def _set_correctness(pred: Prediction, healthy_class: int):
    if pred.true_label == healthy_class:
        pred.is_correct = (pred.predicted_label == healthy_class)
    else:
        pred.is_correct = (pred.predicted_label != healthy_class)


def _compute_metrics(result: EvalResult, healthy_class: int = 1):
    if not result.predictions:
        return
    n = len(result.predictions)
    n_correct = sum(1 for p in result.predictions if p.is_correct)
    result.accuracy = n_correct / n

    n_healthy = sum(1 for p in result.predictions if p.true_label == healthy_class)
    n_diseased = n - n_healthy
    healthy_correct = sum(1 for p in result.predictions
                         if p.true_label == healthy_class and p.is_correct)
    diseased_correct = sum(1 for p in result.predictions
                          if p.true_label != healthy_class and p.is_correct)
    result.specificity = healthy_correct / n_healthy if n_healthy > 0 else 0.0
    result.sensitivity = diseased_correct / n_diseased if n_diseased > 0 else 0.0


# ============================================================================
# RESULTS SAVING
# ============================================================================

RESULTS_DIR = Path(__file__).parent / "results"


def _safe_mode_filename(mode: str) -> str:
    """Convert mode name to a safe filename slug."""
    return mode.replace("(", "_").replace(")", "").replace("+", "_").replace(" ", "_").replace("->", "_to_")


def save_result(result: EvalResult, healthy_class: int = 1):
    """
    Save detailed results for one mode to Enhanced_model/results/<mode>.txt

    Contents:
      - Run metadata (timestamp, mode, n_users)
      - Summary metrics (accuracy, sensitivity, specificity, PPV, NPV, F1)
      - Decision source breakdown with per-source accuracy
      - Per-class breakdown
      - Full misclassification trace (every wrong prediction with details)
      - Confusion matrix
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    filename = _safe_mode_filename(result.mode) + ".txt"
    filepath = RESULTS_DIR / filename

    n = len(result.predictions)
    if n == 0:
        return filepath

    n_correct = sum(1 for p in result.predictions if p.is_correct)
    n_wrong = n - n_correct

    # Compute extra metrics
    n_healthy = sum(1 for p in result.predictions if p.true_label == healthy_class)
    n_diseased = n - n_healthy

    tp = sum(1 for p in result.predictions if p.true_label != healthy_class and p.predicted_label != healthy_class)
    tn = sum(1 for p in result.predictions if p.true_label == healthy_class and p.predicted_label == healthy_class)
    fp = sum(1 for p in result.predictions if p.true_label == healthy_class and p.predicted_label != healthy_class)
    fn = sum(1 for p in result.predictions if p.true_label != healthy_class and p.predicted_label == healthy_class)

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    lines = []
    lines.append("=" * 78)
    lines.append(f"  RESULTS: {result.mode}")
    lines.append("=" * 78)
    lines.append(f"  Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Mode:            {result.mode}")
    lines.append(f"  Users evaluated: {n}")
    lines.append(f"  Total time:      {result.total_time_s:.1f}s  ({result.total_time_s/n*1000:.0f}ms/user)")
    lines.append("")

    # ── Summary metrics ──────────────────────────────────────────────────
    lines.append("-" * 78)
    lines.append("  SUMMARY METRICS")
    lines.append("-" * 78)
    lines.append(f"  Accuracy:              {result.accuracy*100:.2f}%  ({n_correct}/{n})")
    lines.append(f"  Sensitivity (Recall):  {result.sensitivity*100:.2f}%  (diseased detected: {tp}/{tp+fn})")
    lines.append(f"  Specificity:           {result.specificity*100:.2f}%  (healthy correct:   {tn}/{tn+fp})")
    lines.append(f"  Precision (PPV):       {ppv*100:.2f}%  (positive predictions correct: {tp}/{tp+fp})")
    lines.append(f"  Neg Pred Value (NPV):  {npv*100:.2f}%  (negative predictions correct: {tn}/{tn+fn})")
    lines.append(f"  F1 Score:              {f1*100:.2f}%")
    lines.append(f"  Misclassifications:    {n_wrong}")
    lines.append("")

    # ── Confusion matrix ─────────────────────────────────────────────────
    lines.append("-" * 78)
    lines.append("  CONFUSION MATRIX  (binary: healthy vs unhealthy)")
    lines.append("-" * 78)
    lines.append("                          Predicted")
    lines.append("                    Healthy    Unhealthy")
    lines.append(f"  Actual Healthy    {tn:>7d}    {fp:>9d}")
    lines.append(f"  Actual Unhealthy  {fn:>7d}    {tp:>9d}")
    lines.append("")

    # ── Decision source breakdown ────────────────────────────────────────
    if result.source_counts:
        lines.append("-" * 78)
        lines.append("  DECISION SOURCE BREAKDOWN")
        lines.append("-" * 78)
        lines.append(f"  {'Source':<30s}  {'Count':>6s}  {'%':>7s}  {'Accuracy':>9s}")
        lines.append(f"  {'─'*30}  {'─'*6}  {'─'*7}  {'─'*9}")
        for source, count in sorted(result.source_counts.items()):
            src_preds = [p for p in result.predictions if p.source == source]
            src_correct = sum(1 for p in src_preds if p.is_correct)
            src_acc = src_correct / len(src_preds) if src_preds else 0
            lines.append(f"  {source:<30s}  {count:>6d}  {count/n*100:>6.1f}%  {src_acc*100:>8.1f}%")
        lines.append("")

    # ── Per-class breakdown ──────────────────────────────────────────────
    class_labels = sorted(set(p.true_label for p in result.predictions))
    lines.append("-" * 78)
    lines.append("  PER-CLASS BREAKDOWN")
    lines.append("-" * 78)
    lines.append(f"  {'Class':>7s}  {'Correct':>8s}  {'Total':>6s}  {'Accuracy':>9s}  {'Label':<10s}")
    lines.append(f"  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*10}")
    for c in class_labels:
        c_preds = [p for p in result.predictions if p.true_label == c]
        c_correct = sum(1 for p in c_preds if p.is_correct)
        tag = "Healthy" if c == healthy_class else f"Disease-{c}"
        lines.append(f"  {c:>7d}  {c_correct:>8d}  {len(c_preds):>6d}  "
                      f"{c_correct/len(c_preds)*100:>8.1f}%  {tag:<10s}")
    lines.append("")

    # ── Misclassification trace ──────────────────────────────────────────
    wrong_preds = [p for p in result.predictions if not p.is_correct]
    lines.append("-" * 78)
    lines.append(f"  MISCLASSIFICATION TRACE  ({len(wrong_preds)} errors)")
    lines.append("-" * 78)
    if wrong_preds:
        lines.append(f"  {'User':>6s}  {'True':>6s}  {'Pred':>6s}  {'Source':<28s}  "
                      f"{'CDS_AF':>7s}  {'ANN_Conf':>9s}  {'RNN_Conf':>9s}  {'CDS_Dec':<12s}  {'Time_ms':>8s}")
        lines.append(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*28}  "
                      f"{'─'*7}  {'─'*9}  {'─'*9}  {'─'*12}  {'─'*8}")
        for p in wrong_preds:
            true_tag = "H" if p.true_label == healthy_class else f"D{p.true_label}"
            pred_tag = "H" if p.predicted_label == healthy_class else f"D{p.predicted_label}"
            ann_c = f"{p.ann_confidence:.3f}" if p.ann_confidence > 0 else "   -   "
            rnn_c = f"{p.rnn_confidence:.3f}" if p.rnn_confidence > 0 else "   -   "
            cds_d = p.cds_decision if p.cds_decision else "-"
            lines.append(f"  {p.user_idx:>6d}  {true_tag:>6s}  {pred_tag:>6s}  {p.source:<28s}  "
                          f"{p.cds_af:>7.3f}  {ann_c:>9s}  {rnn_c:>9s}  {cds_d:<12s}  {p.elapsed_ms:>8.0f}")
    else:
        lines.append("  (none — all predictions correct)")
    lines.append("")
    lines.append("=" * 78)

    filepath.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Results saved to: {filepath}")
    return filepath


def save_comparison(results: List[EvalResult]):
    """Save a comparison summary across all modes to results/comparison.txt"""
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath = RESULTS_DIR / "comparison.txt"

    lines = []
    lines.append("=" * 90)
    lines.append(f"  COMPARISON SUMMARY")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)
    lines.append("")

    lines.append(f"  {'Mode':<30s}  {'Accuracy':>9s}  {'Sensitivity':>12s}  "
                  f"{'Specificity':>12s}  {'F1':>7s}  {'Time':>8s}")
    lines.append(f"  {'─'*30}  {'─'*9}  {'─'*12}  {'─'*12}  {'─'*7}  {'─'*8}")

    for r in results:
        n = len(r.predictions)
        if n == 0:
            continue
        tp = sum(1 for p in r.predictions if p.true_label != 1 and p.predicted_label != 1)
        fp = sum(1 for p in r.predictions if p.true_label == 1 and p.predicted_label != 1)
        fn = sum(1 for p in r.predictions if p.true_label != 1 and p.predicted_label == 1)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        lines.append(f"  {r.mode:<30s}  {r.accuracy*100:>8.1f}%  {r.sensitivity*100:>11.1f}%  "
                      f"{r.specificity*100:>11.1f}%  {f1*100:>6.1f}%  {r.total_time_s:>7.1f}s")

    lines.append("")

    best = max(results, key=lambda r: r.accuracy)
    worst = min(results, key=lambda r: r.accuracy)
    lines.append(f"  Best:   {best.mode}  ({best.accuracy*100:.1f}%)")
    lines.append(f"  Worst:  {worst.mode}  ({worst.accuracy*100:.1f}%)")
    lines.append(f"  Spread: {(best.accuracy - worst.accuracy)*100:.1f}%")
    lines.append("")

    # Error overlap analysis
    lines.append("-" * 90)
    lines.append("  ERROR OVERLAP ANALYSIS")
    lines.append("-" * 90)
    for r in results:
        wrong_users = set(p.user_idx for p in r.predictions if not p.is_correct)
        lines.append(f"  {r.mode:<30s}  errors on users: {sorted(wrong_users) if wrong_users else '(none)'}")
    lines.append("")

    if len(results) >= 2:
        all_wrong_sets = [set(p.user_idx for p in r.predictions if not p.is_correct) for r in results]
        common_wrong = all_wrong_sets[0]
        for ws in all_wrong_sets[1:]:
            common_wrong = common_wrong & ws
        lines.append(f"  Users wrong in ALL modes: {sorted(common_wrong) if common_wrong else '(none)'}")
        any_wrong = set()
        for ws in all_wrong_sets:
            any_wrong |= ws
        lines.append(f"  Users wrong in ANY mode:  {sorted(any_wrong) if any_wrong else '(none)'}")

    lines.append("")
    lines.append("=" * 90)

    filepath.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"  Comparison saved to: {filepath}")
    return filepath


# ============================================================================
# REPORTING
# ============================================================================

def print_result(result: EvalResult):
    n = len(result.predictions)
    log.info("\n" + "=" * 70)
    log.info(f"{result.mode} RESULTS")
    log.info("=" * 70)
    log.info(f"Overall accuracy:     {result.accuracy*100:.1f}%  "
             f"({sum(1 for p in result.predictions if p.is_correct)}/{n})")
    log.info(f"Sensitivity:          {result.sensitivity*100:.1f}%")
    log.info(f"Specificity:          {result.specificity*100:.1f}%")

    if result.source_counts:
        log.info(f"\nDecision source breakdown:")
        for source, count in sorted(result.source_counts.items()):
            src_preds = [p for p in result.predictions if p.source == source]
            src_acc = sum(1 for p in src_preds if p.is_correct) / len(src_preds) if src_preds else 0
            log.info(f"  {source:25s}: {count:4d} ({count/n*100:5.1f}%)  acc={src_acc*100:.1f}%")

    class_labels = sorted(set(p.true_label for p in result.predictions))
    log.info(f"\nPer-class breakdown:")
    for c in class_labels:
        c_preds = [p for p in result.predictions if p.true_label == c]
        if c_preds:
            c_correct = sum(1 for p in c_preds if p.is_correct)
            log.info(f"  Class {c:2d}: {c_correct}/{len(c_preds)} ({c_correct/len(c_preds)*100:.1f}%)")

    log.info(f"\nTotal time: {result.total_time_s:.1f}s ({result.total_time_s/n*1000:.0f}ms/user)")


def print_comparison(results: List[EvalResult]):
    log.info("\n" + "=" * 78)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 78)
    log.info(f"{'Mode':28s} | {'Accuracy':>10s} | {'Sensitivity':>12s} | {'Specificity':>12s} | {'Time':>8s}")
    log.info("-" * 78)
    for r in results:
        log.info(f"{r.mode:28s} | {r.accuracy*100:>9.1f}% | {r.sensitivity*100:>11.1f}% | "
                 f"{r.specificity*100:>11.1f}% | {r.total_time_s:>7.1f}s")
    log.info("=" * 78)
    best = max(results, key=lambda r: r.accuracy)
    log.info(f"\nBest: {best.mode} at {best.accuracy*100:.1f}%")


# ============================================================================
# MAIN
# ============================================================================

HYBRID_MODES = {
    "cascade":    run_hybrid_cascade,
    "vote":       run_hybrid_vote,
    "confidence": run_hybrid_confidence,
    "stacked":    run_hybrid_stacked,
    "selector":   run_hybrid_selector,
}

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Hybrid CDS-ANN-RNN Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Hybrid modes:
  1. cascade     CDS first -> ANN/RNN for uncertain cases
  2. vote        CDS + ANN + RNN majority vote
  3. confidence  Weighted by each model's confidence
  4. stacked     CDS features fed as inputs to ANN/RNN
  5. selector    Meta-learner picks best model per patient
        """,
    )
    parser.add_argument("--max-users", type=int, default=None,
                        help="Limit LOOCV to N users (default: all 452)")
    parser.add_argument("--mode",
                        choices=["ann-only", "rnn-only", "cds-only",
                                 "hybrid", "compare-hybrid", "all"],
                        default="all")
    parser.add_argument("--hybrid-mode",
                        choices=list(HYBRID_MODES.keys()),
                        default="cascade",
                        help="Which hybrid strategy (used with --mode hybrid)")
    parser.add_argument("--ml-model", choices=["ann", "rnn"], default="rnn",
                        help="ML model for cascade/stacked modes")
    args = parser.parse_args()

    _suppress_cds_loggers()
    data, labels = load_uci_data()
    _suppress_cds_loggers()

    log.info(f"\nDataset: {data.shape[0]} users x {data.shape[1]} features")
    label_dist = {int(c): int((labels == c).sum()) for c in sorted(set(labels))}
    log.info(f"Label distribution: {label_dist}")

    results = []
    mu = args.max_users

    if args.mode == "ann-only":
        r = run_single_model_loocv(data, labels, "ann", mu)
        print_result(r)
        save_result(r)
        results.append(r)

    elif args.mode == "rnn-only":
        r = run_single_model_loocv(data, labels, "rnn", mu)
        print_result(r)
        save_result(r)
        results.append(r)

    elif args.mode == "cds-only":
        r = run_single_model_loocv(data, labels, "cds", mu)
        print_result(r)
        save_result(r)
        results.append(r)

    elif args.mode == "hybrid":
        fn = HYBRID_MODES[args.hybrid_mode]
        r = fn(data, labels, max_users=mu, ml_model=args.ml_model)
        print_result(r)
        save_result(r)
        results.append(r)

    elif args.mode == "compare-hybrid":
        log.info("\nRunning all 10 hybrid combinations (5 modes x 2 ML models)...\n")
        for ml in ("ann", "rnn"):
            for name, fn in HYBRID_MODES.items():
                combo = f"{name.upper()} + {ml.upper()}"
                log.info(f"\n{'#' * 70}")
                log.info(f"HYBRID: {combo}")
                log.info(f"{'#' * 70}")
                r = fn(data, labels, max_users=mu, ml_model=ml)
                print_result(r)
                save_result(r)
                results.append(r)
        print_comparison(results)
        save_comparison(results)

    elif args.mode == "all":
        for mt in ("ann", "rnn", "cds"):
            r = run_single_model_loocv(data, labels, mt, mu)
            print_result(r)
            save_result(r)
            results.append(r)
        r = run_hybrid_cascade(data, labels, max_users=mu)
        print_result(r)
        save_result(r)
        results.append(r)
        print_comparison(results)
        save_comparison(results)

    return results


if __name__ == "__main__":
    main()
