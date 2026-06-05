"""
================================================================================
Resource Analysis: Time, Memory & Computational Cost
================================================================================

Runs all 13 model combinations (3 standalone + 10 hybrid) with instrumentation
to measure:
  - Wall-clock time (total, per-user, train vs predict)
  - Peak memory usage (RSS delta during each fold)
  - Computational cost (multiplication count for ANN, RNN, CDS)
  - Classification metrics (accuracy, sensitivity, specificity, PPV, NPV, F1)

Results are saved to resource_analysis/results/ with one file per mode plus
a master comparison CSV and summary text file.

USAGE
-----
  python resource_analysis.py --max-users 50       # quick test
  python resource_analysis.py                      # full 452-user LOOCV
  python resource_analysis.py --mode ann-only      # single mode
  python resource_analysis.py --mode compare-all   # all 13 + comparison

================================================================================
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import csv
import logging
import math
import os
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent))

from reservoir_network import EchoStateNetwork

log = logging.getLogger("Resource")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    log.addHandler(h)
    log.propagate = False

RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ResourceRecord:
    """Per-user resource measurements."""
    user_idx: int
    true_label: int
    predicted_label: int
    is_correct: bool
    source: str
    # Time (milliseconds)
    time_total_ms: float = 0.0
    time_cds_train_ms: float = 0.0
    time_cds_predict_ms: float = 0.0
    time_ml_train_ms: float = 0.0
    time_ml_predict_ms: float = 0.0
    # Memory (bytes)
    peak_memory_bytes: int = 0
    # Computational cost (multiplications)
    cds_multiplications: int = 0
    ml_train_multiplications: int = 0
    ml_inference_multiplications: int = 0


@dataclass
class ResourceResult:
    """Aggregated results for one mode."""
    mode: str
    records: List[ResourceRecord] = field(default_factory=list)
    # Classification metrics
    accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    ppv: float = 0.0
    npv: float = 0.0
    f1: float = 0.0
    # Aggregated resource metrics
    total_time_s: float = 0.0
    avg_time_per_user_ms: float = 0.0
    avg_cds_train_ms: float = 0.0
    avg_cds_predict_ms: float = 0.0
    avg_ml_train_ms: float = 0.0
    avg_ml_predict_ms: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    total_multiplications: int = 0
    avg_multiplications_per_user: float = 0.0
    total_cds_mults: int = 0
    total_ml_train_mults: int = 0
    total_ml_infer_mults: int = 0


# ============================================================================
# ANN with multiplication counting
# ============================================================================

class ANNClassifier:
    """MLP with multiplication instrumentation."""

    def __init__(self, hidden_layers=(64, 32), max_iter=500, random_state=42):
        self.hidden_layers = hidden_layers
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers, activation="relu", solver="adam",
            max_iter=max_iter, random_state=random_state, early_stopping=True,
            validation_fraction=0.1, learning_rate="adaptive",
        )
        self.is_fitted = False
        self.train_multiplications = 0
        self.inference_multiplications = 0

    def _count_forward_pass_mults(self, n_input, n_output):
        """Count multiplications for one forward pass through the MLP."""
        sizes = [n_input] + list(self.hidden_layers) + [n_output]
        total = 0
        for i in range(len(sizes) - 1):
            total += sizes[i] * sizes[i + 1]  # weight multiplications
        return total

    def fit(self, X, y):
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        n_input = X.shape[1]
        n_output = len(self.clf.classes_)
        fwd_mults = self._count_forward_pass_mults(n_input, n_output)
        # Training: n_iterations * n_samples * forward_pass + backward_pass (~2x forward)
        n_iter = self.clf.n_iter_
        n_samples = X.shape[0]
        self.train_multiplications = n_iter * n_samples * fwd_mults * 3
        return self

    def predict(self, X):
        X_clean = self.imputer.transform(X)
        return self.clf.predict(self.scaler.transform(X_clean))

    def predict_proba(self, X):
        X_clean = self.imputer.transform(X)
        return self.clf.predict_proba(self.scaler.transform(X_clean))

    def predict_with_confidence(self, X):
        proba = self.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        n_input = X.shape[1]
        n_output = len(self.clf.classes_)
        self.inference_multiplications = self._count_forward_pass_mults(n_input, n_output)
        return int(self.clf.classes_[pred_idx]), float(proba[pred_idx])


# ============================================================================
# CDS COMPONENT
# ============================================================================

def _suppress_cds_loggers():
    import logging as _l
    for n in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3", "CDS.Alg4", "CDS.Alg1.ForcedSex", "CDS"):
        lgr = _l.getLogger(n)
        lgr.setLevel(_l.CRITICAL)
        for h in lgr.handlers:
            h.setLevel(_l.CRITICAL)


def _run_cds_pipeline(train_data, train_labels, n_bins=20):
    from CDS_Paper_Algorithms import build_decision_tree
    from Algorithm2 import run_algorithm2
    from Algorithm3 import run_algorithm3
    tree = build_decision_tree(train_data, train_labels)
    nf = ["root"]
    if 2 in tree.nodes_by_level:
        for node in tree.nodes_by_level[2]:
            nf.append(node.node_id)
            if len(nf) >= 3:
                break
    a2 = run_algorithm2(tree=tree, data=train_data, labels=train_labels, n_bins=n_bins, nodes_filter=nf)
    a3 = run_algorithm3(alg2_output=a2, tree=tree, data=train_data, labels=train_labels,
                        nodes_filter=nf, reset_per_h=False, verbose=False)
    return tree, a2, a3


def _run_cds_predict(user_idx, data, labels, train_data, train_labels, tree, a2, a3, rng_seed=42):
    from Algorithm4 import run_algorithm4, HealthDecision
    try:
        record = run_algorithm4(user_global_idx=user_idx, data=data, labels=labels,
                                tree=tree, alg2_output=a2, alg3_output=a3,
                                rng_seed=rng_seed, train_data=train_data, train_labels=train_labels)
        af = record.af_trace[-1].AF_real if record.af_trace else 0.0
        n_actions = len(record.af_trace) if record.af_trace else 0
        return record.decision, af, 1.0 - af, record, n_actions
    except Exception:
        return HealthDecision.SCREENING, 0.0, 1.0, None, 0


def _cds_decision_to_label(decision, record, labels, healthy_class):
    from Algorithm4 import HealthDecision
    if decision == HealthDecision.UNHEALTHY:
        ac = record.alarm_class if record else None
        if ac is not None:
            return ac
        return max(set(labels) - {healthy_class}, key=lambda c: int((labels == c).sum()), default=healthy_class + 1)
    return healthy_class


def _cds_multiplications(n_features, n_actions):
    """Estimate CDS multiplications: comparisons + AF updates per action."""
    # Each action: 1 range comparison (2 mults) + AF update (3 mults) + RL lookahead (~10 mults)
    per_action = 15
    return n_actions * per_action + n_features  # initial feature scan


def load_uci_data():
    from CDS_Paper_Algorithms import load_dataset
    p = str(Path(__file__).parent / "data" / "arrhythmia.data")
    if not Path(p).exists():
        p = str(Path(__file__).parent.parent / "data" / "arrhythmia.data")
    return load_dataset(p)


# ============================================================================
# UNIVERSAL INSTRUMENTED RUNNER
# ============================================================================

def _run_fold(i, data, labels, healthy_class, mode_name, ml_model,
              hybrid_mode) -> ResourceRecord:
    """
    Run one LOOCV fold with full resource instrumentation.
    Returns a ResourceRecord with all measurements.
    """
    from Algorithm4 import HealthDecision

    train_mask = np.ones(data.shape[0], dtype=bool)
    train_mask[i] = False
    train_data, train_labels = data[train_mask], labels[train_mask]
    true_label = int(labels[i])
    n_features = data.shape[1]

    rec = ResourceRecord(user_idx=i, true_label=true_label, predicted_label=healthy_class,
                         is_correct=False, source="")

    tracemalloc.start()
    t_total_start = time.perf_counter()

    # ── CDS phase ────────────────────────────────────────────────────────
    cds_decision = None
    cds_af = 0.0
    cds_label = healthy_class
    cds_record = None
    n_cds_actions = 0

    needs_cds = (hybrid_mode != "none")  # standalone ML modes don't need CDS
    if mode_name == "CDS-only":
        needs_cds = True
    if mode_name in ("ANN-only", "RNN-only"):
        needs_cds = False

    if needs_cds:
        t_cds_train = time.perf_counter()
        tree, a2, a3 = _run_cds_pipeline(train_data, train_labels)
        rec.time_cds_train_ms = (time.perf_counter() - t_cds_train) * 1000

        t_cds_pred = time.perf_counter()
        cds_decision, cds_af, cds_rw, cds_record, n_cds_actions = _run_cds_predict(
            i, data, labels, train_data, train_labels, tree, a2, a3, 42 + i)
        rec.time_cds_predict_ms = (time.perf_counter() - t_cds_pred) * 1000
        cds_label = _cds_decision_to_label(cds_decision, cds_record, labels, healthy_class)
        rec.cds_multiplications = _cds_multiplications(n_features, n_cds_actions)

    # ── ML phase ─────────────────────────────────────────────────────────
    ml_pred = healthy_class
    ml_conf = 0.5
    ml_model_obj = None

    needs_ml = mode_name in ("ANN-only", "RNN-only") or hybrid_mode in (
        "cascade", "vote", "confidence", "stacked", "selector",
        "alarm-refine", "af-gated", "disagree-rules", "alarm-specialist", "triple-cascade")
    # For cascade: only train ML if CDS is uncertain
    if hybrid_mode == "cascade" and cds_decision in (HealthDecision.UNHEALTHY, HealthDecision.HEALTHY):
        needs_ml = False
    # For new modes: need ML for ALARM (D2 check) and SCREENING, but not HEALTHY
    if hybrid_mode in ("alarm-refine", "af-gated", "disagree-rules",
                       "alarm-specialist", "triple-cascade"):
        if cds_decision == HealthDecision.HEALTHY:
            needs_ml = False

    if needs_ml:
        use_rnn = (ml_model == "rnn") or mode_name == "RNN-only"
        use_ann = (ml_model == "ann") or mode_name == "ANN-only"

        train_input = train_data
        test_input = data[i:i+1]

        # Stacked mode: augment features with CDS outputs
        if hybrid_mode == "stacked" and needs_cds:
            alarm_f = 1.0 if cds_decision == HealthDecision.UNHEALTHY else 0.0
            healthy_f = 1.0 if cds_decision == HealthDecision.HEALTHY else 0.0
            screen_f = 1.0 if cds_decision == HealthDecision.SCREENING else 0.0
            test_cds = np.array([[cds_af, 1.0 - cds_af, alarm_f, healthy_f, screen_f]])
            test_input = np.hstack([data[i:i+1], test_cds])
            # Proxy CDS features for training
            n_train = train_data.shape[0]
            train_cds = np.zeros((n_train, 5))
            h_mask = train_labels == healthy_class
            if h_mask.sum() > 0:
                centroid = np.nanmean(train_data[h_mask], axis=0)
                for j in range(n_train):
                    diff = train_data[j] - centroid
                    dist = np.nanmean(np.abs(diff))
                    proxy_af = min(1.0, dist / (np.nanstd(diff) + 1e-10) * 0.1)
                    ih = train_labels[j] == healthy_class
                    train_cds[j] = [proxy_af, 1.0 - proxy_af, 0.0 if ih else float(proxy_af > 0.5), float(ih), 0.0]
            train_input = np.hstack([train_data, train_cds])

        t_ml_train = time.perf_counter()
        if use_rnn:
            ml_model_obj = EchoStateNetwork()
            ml_model_obj.fit(train_input, train_labels)
        else:
            ml_model_obj = ANNClassifier()
            ml_model_obj.fit(train_input, train_labels)
        rec.time_ml_train_ms = (time.perf_counter() - t_ml_train) * 1000

        t_ml_pred = time.perf_counter()
        ml_pred, ml_conf = ml_model_obj.predict_with_confidence(test_input)
        rec.time_ml_predict_ms = (time.perf_counter() - t_ml_pred) * 1000

        rec.ml_train_multiplications = ml_model_obj.train_multiplications
        rec.ml_inference_multiplications = ml_model_obj.inference_multiplications

    # ── Decision logic ───────────────────────────────────────────────────
    if mode_name == "ANN-only" or mode_name == "RNN-only":
        rec.predicted_label = ml_pred
        rec.source = mode_name

    elif mode_name == "CDS-only":
        rec.predicted_label = cds_label
        rec.source = "CDS-ALARM" if cds_decision == HealthDecision.UNHEALTHY else (
            "CDS-HEALTHY" if cds_decision == HealthDecision.HEALTHY else "CDS-SCREENING")

    elif hybrid_mode == "cascade":
        if cds_decision == HealthDecision.UNHEALTHY:
            rec.predicted_label = cds_label
            rec.source = "CDS-ALARM"
        elif cds_decision == HealthDecision.HEALTHY:
            rec.predicted_label = healthy_class
            rec.source = "CDS-HEALTHY"
        else:
            rec.predicted_label = ml_pred
            rec.source = ml_model.upper()

    elif hybrid_mode == "vote":
        cds_vote = 0 if cds_label == healthy_class else 1
        ml_vote = 0 if ml_pred == healthy_class else 1
        if cds_vote == ml_vote:
            rec.predicted_label = ml_pred if ml_vote == 1 else healthy_class
            rec.source = "VOTE-AGREE"
        else:
            rec.predicted_label = cds_label
            rec.source = "VOTE-TIE-CDS"

    elif hybrid_mode == "confidence":
        cds_conf = max(cds_af, 0.8) if cds_decision == HealthDecision.UNHEALTHY else cds_af
        cds_p = 0.0 if cds_label == healthy_class else cds_conf
        ml_p = 0.0 if ml_pred == healthy_class else ml_conf
        weighted = 0.55 * cds_p + 0.45 * ml_p
        if weighted > 0.4:
            if cds_p >= ml_p and cds_label != healthy_class:
                rec.predicted_label = cds_label
                rec.source = "CONF-CDS"
            elif ml_pred != healthy_class:
                rec.predicted_label = ml_pred
                rec.source = f"CONF-{ml_model.upper()}"
            else:
                rec.predicted_label = healthy_class
                rec.source = "CONF-HEALTHY"
        else:
            rec.predicted_label = healthy_class
            rec.source = "CONF-HEALTHY"

    elif hybrid_mode == "stacked":
        rec.predicted_label = ml_pred
        rec.source = f"STACKED-{ml_model.upper()}"

    elif hybrid_mode == "selector":
        if cds_decision == HealthDecision.UNHEALTHY:
            rec.predicted_label = cds_label
            rec.source = "SEL-CDS-ALARM"
        elif cds_decision == HealthDecision.HEALTHY and cds_af > 0.5:
            rec.predicted_label = healthy_class
            rec.source = "SEL-CDS-HEALTHY"
        else:
            rec.predicted_label = ml_pred
            rec.source = f"SEL-{ml_model.upper()}"

    elif hybrid_mode == "alarm-refine":
        # Multi-signal ANN veto: ANN conf + ANN margin + CDS AF
        if cds_decision == HealthDecision.HEALTHY:
            rec.predicted_label = healthy_class
            rec.source = "AREF-CDS-HEALTHY"
        elif cds_decision == HealthDecision.UNHEALTHY:
            ml_proba = ml_model_obj.predict_proba(data[i:i+1])[0] if ml_model_obj else np.array([0.5, 0.5])
            ml_margin = float(np.sort(ml_proba)[-1] - np.sort(ml_proba)[-2]) if len(ml_proba) >= 2 else 0.0
            override = (ml_pred == healthy_class and ml_conf >= 0.70
                        and ml_margin >= 0.30 and cds_af < 0.70)
            if override:
                rec.predicted_label = healthy_class
                rec.source = "AREF-ANN-OVERRIDE"
            else:
                rec.predicted_label = cds_label
                rec.source = "AREF-CDS-ALARM"
        else:
            rec.predicted_label = ml_pred
            rec.source = "AREF-ANN-SCREEN"

    elif hybrid_mode == "af-gated":
        # Route by CDS evidence depth: few actions checked → consult ANN
        n_actions_cds = cds_record.total_actions_applied if cds_record and hasattr(cds_record, 'total_actions_applied') else 0
        median_actions = n_features * 0.15
        if cds_decision == HealthDecision.HEALTHY:
            rec.predicted_label = healthy_class
            rec.source = "AFG-CDS-HEALTHY"
        elif cds_decision == HealthDecision.UNHEALTHY:
            if n_actions_cds <= median_actions and cds_af < 0.60:
                if ml_pred == healthy_class and ml_conf >= 0.70:
                    rec.predicted_label = healthy_class
                    rec.source = "AFG-ANN-OVERRIDE"
                else:
                    rec.predicted_label = cds_label
                    rec.source = "AFG-CDS-ALARM"
            else:
                rec.predicted_label = cds_label
                rec.source = "AFG-CDS-ALARM"
        else:
            rec.predicted_label = ml_pred
            rec.source = "AFG-ANN-SCREEN"

    elif hybrid_mode == "disagree-rules":
        # Override ALARM only when ANN is decisive AND CDS evidence is weak
        ann_healthy = (ml_pred == healthy_class)
        ml_proba = ml_model_obj.predict_proba(data[i:i+1])[0] if ml_model_obj else np.array([0.5, 0.5])
        ml_margin = float(np.sort(ml_proba)[-1] - np.sort(ml_proba)[-2]) if len(ml_proba) >= 2 else 0.0
        ann_decisive = (ml_margin >= 0.30)
        if cds_decision == HealthDecision.HEALTHY:
            rec.predicted_label = healthy_class
            rec.source = "DRULE-CDS-HEALTHY"
        elif cds_decision == HealthDecision.UNHEALTHY:
            if not ann_healthy:
                rec.predicted_label = cds_label
                rec.source = "DRULE-BOTH-SICK"
            elif ann_healthy and ann_decisive and cds_af < 0.60:
                rec.predicted_label = healthy_class
                rec.source = "DRULE-ANN-OVERRIDE"
            else:
                rec.predicted_label = cds_label
                rec.source = "DRULE-CDS-ALARM"
        else:
            rec.predicted_label = ml_pred
            rec.source = "DRULE-ANN-SCREEN"

    elif hybrid_mode == "alarm-specialist":
        # Binary specialist only on weak-evidence ALARM cases
        n_actions_cds = cds_record.total_actions_applied if cds_record and hasattr(cds_record, 'total_actions_applied') else 0
        weak_evidence = (cds_af < 0.65 and n_actions_cds < n_features * 0.20)
        if cds_decision == HealthDecision.HEALTHY:
            rec.predicted_label = healthy_class
            rec.source = "ASPEC-CDS-HEALTHY"
        elif cds_decision == HealthDecision.UNHEALTHY and weak_evidence:
            binary_labels = np.where(train_labels == healthy_class, 0, 1)
            specialist = ANNClassifier(hidden_layers=(16,), max_iter=300, random_state=42)
            t_spec_train = time.perf_counter()
            specialist.fit(train_data, binary_labels)
            rec.time_ml_train_ms = (time.perf_counter() - t_spec_train) * 1000
            t_spec_pred = time.perf_counter()
            spec_proba = specialist.predict_proba(data[i:i+1])[0]
            rec.time_ml_predict_ms = (time.perf_counter() - t_spec_pred) * 1000
            rec.ml_train_multiplications = specialist.train_multiplications
            rec.ml_inference_multiplications = specialist.inference_multiplications
            healthy_idx = np.where(specialist.clf.classes_ == 0)[0]
            spec_healthy_conf = float(spec_proba[healthy_idx[0]]) if len(healthy_idx) > 0 else 0.0
            spec_margin = abs(float(spec_proba[0]) - float(spec_proba[1])) if len(spec_proba) >= 2 else 0.0
            if spec_healthy_conf >= 0.65 and spec_margin >= 0.25:
                rec.predicted_label = healthy_class
                rec.source = "ASPEC-OVERRIDE"
            else:
                rec.predicted_label = cds_label
                rec.source = "ASPEC-CDS-ALARM"
        elif cds_decision == HealthDecision.UNHEALTHY:
            rec.predicted_label = cds_label
            rec.source = "ASPEC-CDS-ALARM"
        else:
            rec.predicted_label = ml_pred
            rec.source = "ASPEC-ANN-SCREEN"

    elif hybrid_mode == "triple-cascade":
        # Multi-signal veto on ALARM + triple majority on SCREENING
        if cds_decision == HealthDecision.HEALTHY:
            rec.predicted_label = healthy_class
            rec.source = "TRI-CDS-HEALTHY"
        elif cds_decision == HealthDecision.UNHEALTHY:
            ml_proba = ml_model_obj.predict_proba(data[i:i+1])[0] if ml_model_obj else np.array([0.5, 0.5])
            ml_margin = float(np.sort(ml_proba)[-1] - np.sort(ml_proba)[-2]) if len(ml_proba) >= 2 else 0.0
            override = (ml_pred == healthy_class and ml_conf >= 0.70
                        and ml_margin >= 0.30 and cds_af < 0.70)
            if override:
                rec.predicted_label = healthy_class
                rec.source = "TRI-ANN-VETO"
            else:
                rec.predicted_label = cds_label
                rec.source = "TRI-CDS-ALARM"
        else:
            rnn_model = EchoStateNetwork()
            t_rnn_train = time.perf_counter()
            rnn_model.fit(train_data, train_labels)
            rnn_train_ms = (time.perf_counter() - t_rnn_train) * 1000
            t_rnn_pred = time.perf_counter()
            rnn_pred, rnn_conf = rnn_model.predict_with_confidence(data[i:i+1])
            rnn_pred_ms = (time.perf_counter() - t_rnn_pred) * 1000
            rec.time_ml_train_ms += rnn_train_ms
            rec.time_ml_predict_ms += rnn_pred_ms
            rec.ml_train_multiplications += rnn_model.train_multiplications
            rec.ml_inference_multiplications += rnn_model.inference_multiplications
            ann_vote = 0 if ml_pred == healthy_class else 1
            rnn_vote = 0 if rnn_pred == healthy_class else 1
            cds_vote = 0
            sick_votes = cds_vote + ann_vote + rnn_vote
            if sick_votes >= 2:
                candidates = []
                if ann_vote == 1:
                    candidates.append((ml_conf, ml_pred))
                if rnn_vote == 1:
                    candidates.append((rnn_conf, rnn_pred))
                rec.predicted_label = max(candidates, key=lambda x: x[0])[1] if candidates else healthy_class
                rec.source = "TRI-MAJORITY-SICK"
            else:
                rec.predicted_label = healthy_class
                rec.source = "TRI-MAJORITY-HEALTHY"

    # Correctness
    if true_label == healthy_class:
        rec.is_correct = (rec.predicted_label == healthy_class)
    else:
        rec.is_correct = (rec.predicted_label != healthy_class)

    rec.time_total_ms = (time.perf_counter() - t_total_start) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rec.peak_memory_bytes = peak

    return rec


# ============================================================================
# MODE RUNNER
# ============================================================================

MODE_CONFIGS = {
    # (mode_name, ml_model, hybrid_mode)
    "ann-only":          ("ANN-only",             "ann", "none"),
    "rnn-only":          ("RNN-only",             "rnn", "none"),
    "cds-only":          ("CDS-only",             "cds", "none"),
    "cascade-ann":       ("Cascade(ANN)",         "ann", "cascade"),
    "cascade-rnn":       ("Cascade(RNN)",         "rnn", "cascade"),
    "vote-ann":          ("Vote(CDS+ANN)",        "ann", "vote"),
    "vote-rnn":          ("Vote(CDS+RNN)",        "rnn", "vote"),
    "confidence-ann":    ("Confidence(CDS+ANN)",  "ann", "confidence"),
    "confidence-rnn":    ("Confidence(CDS+RNN)",  "rnn", "confidence"),
    "stacked-ann":       ("Stacked(CDS->ANN)",    "ann", "stacked"),
    "stacked-rnn":       ("Stacked(CDS->RNN)",    "rnn", "stacked"),
    "selector-ann":      ("Selector(CDS+ANN)",    "ann", "selector"),
    "selector-rnn":      ("Selector(CDS+RNN)",    "rnn", "selector"),
    "alarm-refine":      ("AlarmRefine(ANN)",     "ann", "alarm-refine"),
    "af-gated":          ("AFGated(ANN)",         "ann", "af-gated"),
    "disagree-rules":    ("DisagreeRules(ANN)",   "ann", "disagree-rules"),
    "alarm-specialist":  ("AlarmSpecialist(ANN)", "ann", "alarm-specialist"),
    "triple-cascade":    ("TripleCascade",        "ann", "triple-cascade"),
}


def run_mode(data, labels, mode_key, max_users=None, healthy_class=1) -> ResourceResult:
    """Run one mode with full resource instrumentation."""
    mode_name, ml_model, hybrid_mode = MODE_CONFIGS[mode_key]

    if hybrid_mode != "none" or mode_key == "cds-only":
        _suppress_cds_loggers()
        from Algorithm4 import HealthDecision
        _suppress_cds_loggers()

    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    result = ResourceResult(mode=mode_name)
    t0 = time.time()

    log.info("=" * 70)
    log.info(f"RESOURCE ANALYSIS: {mode_name}  ({n_total} users)")
    log.info("=" * 70)

    for i in range(n_total):
        rec = _run_fold(i, data, labels, healthy_class, mode_name, ml_model, hybrid_mode)
        result.records.append(rec)

        if (i + 1) % max(1, n_total // 10) == 0 or i == n_total - 1:
            acc = sum(1 for r in result.records if r.is_correct) / len(result.records)
            avg_ms = np.mean([r.time_total_ms for r in result.records])
            avg_mem = np.mean([r.peak_memory_bytes for r in result.records]) / 1e6
            log.info(f"  [{i+1}/{n_total}] acc={acc*100:.1f}%  "
                     f"avg_time={avg_ms:.0f}ms  avg_mem={avg_mem:.1f}MB")

    result.total_time_s = time.time() - t0
    _compute_result_metrics(result, healthy_class)
    return result


def _compute_result_metrics(result: ResourceResult, healthy_class=1):
    """Compute all aggregated metrics."""
    recs = result.records
    n = len(recs)
    if n == 0:
        return

    # Classification
    n_correct = sum(1 for r in recs if r.is_correct)
    result.accuracy = n_correct / n

    tp = sum(1 for r in recs if r.true_label != healthy_class and r.predicted_label != healthy_class)
    tn = sum(1 for r in recs if r.true_label == healthy_class and r.predicted_label == healthy_class)
    fp = sum(1 for r in recs if r.true_label == healthy_class and r.predicted_label != healthy_class)
    fn = sum(1 for r in recs if r.true_label != healthy_class and r.predicted_label == healthy_class)

    result.sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    result.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    result.ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    result.npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    result.f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Time
    result.avg_time_per_user_ms = np.mean([r.time_total_ms for r in recs])
    result.avg_cds_train_ms = np.mean([r.time_cds_train_ms for r in recs])
    result.avg_cds_predict_ms = np.mean([r.time_cds_predict_ms for r in recs])
    result.avg_ml_train_ms = np.mean([r.time_ml_train_ms for r in recs])
    result.avg_ml_predict_ms = np.mean([r.time_ml_predict_ms for r in recs])

    # Memory
    result.peak_memory_mb = max(r.peak_memory_bytes for r in recs) / 1e6
    result.avg_memory_mb = np.mean([r.peak_memory_bytes for r in recs]) / 1e6

    # Computational cost
    result.total_cds_mults = sum(r.cds_multiplications for r in recs)
    result.total_ml_train_mults = sum(r.ml_train_multiplications for r in recs)
    result.total_ml_infer_mults = sum(r.ml_inference_multiplications for r in recs)
    result.total_multiplications = result.total_cds_mults + result.total_ml_train_mults + result.total_ml_infer_mults
    result.avg_multiplications_per_user = result.total_multiplications / n


# ============================================================================
# RESULTS SAVING
# ============================================================================

def _safe_name(mode: str) -> str:
    return mode.replace("(", "_").replace(")", "").replace("+", "_").replace(" ", "_").replace("->", "_to_")


def save_result(result: ResourceResult):
    """Save detailed results for one mode."""
    RESULTS_DIR.mkdir(exist_ok=True)
    fname = _safe_name(result.mode) + ".txt"
    fp = RESULTS_DIR / fname
    recs = result.records
    n = len(recs)
    if n == 0:
        return fp

    L = []
    L.append("=" * 85)
    L.append(f"  RESOURCE ANALYSIS: {result.mode}")
    L.append("=" * 85)
    L.append(f"  Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"  Users evaluated:      {n}")
    L.append("")

    L.append("-" * 85)
    L.append("  CLASSIFICATION METRICS")
    L.append("-" * 85)
    L.append(f"  Accuracy:             {result.accuracy*100:.2f}%")
    L.append(f"  Sensitivity:          {result.sensitivity*100:.2f}%")
    L.append(f"  Specificity:          {result.specificity*100:.2f}%")
    L.append(f"  Precision (PPV):      {result.ppv*100:.2f}%")
    L.append(f"  Neg Pred Value:       {result.npv*100:.2f}%")
    L.append(f"  F1 Score:             {result.f1*100:.2f}%")
    L.append("")

    L.append("-" * 85)
    L.append("  TIME METRICS")
    L.append("-" * 85)
    L.append(f"  Total wall-clock:     {result.total_time_s:.1f}s")
    L.append(f"  Avg per user:         {result.avg_time_per_user_ms:.1f}ms")
    L.append(f"  Avg CDS train:        {result.avg_cds_train_ms:.1f}ms")
    L.append(f"  Avg CDS predict:      {result.avg_cds_predict_ms:.1f}ms")
    L.append(f"  Avg ML train:         {result.avg_ml_train_ms:.1f}ms")
    L.append(f"  Avg ML predict:       {result.avg_ml_predict_ms:.1f}ms")
    L.append("")

    L.append("-" * 85)
    L.append("  MEMORY METRICS")
    L.append("-" * 85)
    L.append(f"  Peak memory:          {result.peak_memory_mb:.2f} MB")
    L.append(f"  Avg memory per fold:  {result.avg_memory_mb:.2f} MB")
    L.append("")

    L.append("-" * 85)
    L.append("  COMPUTATIONAL COST (multiplications)")
    L.append("-" * 85)
    L.append(f"  Total multiplications:     {result.total_multiplications:>15,}")
    L.append(f"  Avg per user:              {result.avg_multiplications_per_user:>15,.0f}")
    L.append(f"  CDS total:                 {result.total_cds_mults:>15,}")
    L.append(f"  ML training total:         {result.total_ml_train_mults:>15,}")
    L.append(f"  ML inference total:        {result.total_ml_infer_mults:>15,}")
    L.append("")

    # Per-class
    classes = sorted(set(r.true_label for r in recs))
    L.append("-" * 85)
    L.append("  PER-CLASS BREAKDOWN")
    L.append("-" * 85)
    for c in classes:
        cr = [r for r in recs if r.true_label == c]
        cc = sum(1 for r in cr if r.is_correct)
        tag = "Healthy" if c == 1 else f"Disease-{c}"
        L.append(f"  Class {c:>3d} ({tag:<12s}): {cc}/{len(cr)} ({cc/len(cr)*100:.1f}%)")
    L.append("")

    # Misclassification trace
    wrong = [r for r in recs if not r.is_correct]
    L.append("-" * 85)
    L.append(f"  MISCLASSIFICATION TRACE ({len(wrong)} errors)")
    L.append("-" * 85)
    if wrong:
        L.append(f"  {'User':>6s}  {'True':>6s}  {'Pred':>6s}  {'Source':<22s}  "
                 f"{'Time_ms':>8s}  {'Mem_MB':>7s}  {'Mults':>12s}")
        L.append(f"  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*22}  {'─'*8}  {'─'*7}  {'─'*12}")
        for r in wrong:
            tl = "H" if r.true_label == 1 else f"D{r.true_label}"
            pl = "H" if r.predicted_label == 1 else f"D{r.predicted_label}"
            mults = r.cds_multiplications + r.ml_train_multiplications + r.ml_inference_multiplications
            L.append(f"  {r.user_idx:>6d}  {tl:>6s}  {pl:>6s}  {r.source:<22s}  "
                     f"{r.time_total_ms:>8.0f}  {r.peak_memory_bytes/1e6:>7.2f}  {mults:>12,}")
    else:
        L.append("  (none)")
    L.append("")

    # Per-user time breakdown (top 10 slowest)
    L.append("-" * 85)
    L.append("  SLOWEST USERS (top 10)")
    L.append("-" * 85)
    slowest = sorted(recs, key=lambda r: r.time_total_ms, reverse=True)[:10]
    L.append(f"  {'User':>6s}  {'Total_ms':>9s}  {'CDS_train':>10s}  {'CDS_pred':>9s}  "
             f"{'ML_train':>9s}  {'ML_pred':>8s}  {'Correct':>8s}")
    L.append(f"  {'─'*6}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}")
    for r in slowest:
        ok = "Y" if r.is_correct else "N"
        L.append(f"  {r.user_idx:>6d}  {r.time_total_ms:>9.0f}  {r.time_cds_train_ms:>10.0f}  "
                 f"{r.time_cds_predict_ms:>9.0f}  {r.time_ml_train_ms:>9.0f}  "
                 f"{r.time_ml_predict_ms:>8.0f}  {ok:>8s}")
    L.append("")
    L.append("=" * 85)

    fp.write_text("\n".join(L), encoding="utf-8")
    log.info(f"  Saved: {fp}")
    return fp


def save_comparison_csv(results: List[ResourceResult]):
    """Save a master CSV with all metrics across all modes."""
    RESULTS_DIR.mkdir(exist_ok=True)
    fp = RESULTS_DIR / "comparison.csv"
    fields = [
        "Mode", "Accuracy%", "Sensitivity%", "Specificity%", "PPV%", "NPV%", "F1%",
        "Total_Time_s", "Avg_Time_Per_User_ms",
        "Avg_CDS_Train_ms", "Avg_CDS_Predict_ms", "Avg_ML_Train_ms", "Avg_ML_Predict_ms",
        "Peak_Memory_MB", "Avg_Memory_MB",
        "Total_Multiplications", "Avg_Mults_Per_User",
        "CDS_Mults", "ML_Train_Mults", "ML_Infer_Mults",
        "N_Users",
    ]
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in results:
            w.writerow([
                r.mode,
                f"{r.accuracy*100:.2f}", f"{r.sensitivity*100:.2f}", f"{r.specificity*100:.2f}",
                f"{r.ppv*100:.2f}", f"{r.npv*100:.2f}", f"{r.f1*100:.2f}",
                f"{r.total_time_s:.1f}", f"{r.avg_time_per_user_ms:.1f}",
                f"{r.avg_cds_train_ms:.1f}", f"{r.avg_cds_predict_ms:.1f}",
                f"{r.avg_ml_train_ms:.1f}", f"{r.avg_ml_predict_ms:.1f}",
                f"{r.peak_memory_mb:.2f}", f"{r.avg_memory_mb:.2f}",
                r.total_multiplications, f"{r.avg_multiplications_per_user:.0f}",
                r.total_cds_mults, r.total_ml_train_mults, r.total_ml_infer_mults,
                len(r.records),
            ])
    log.info(f"  CSV saved: {fp}")
    return fp


def save_comparison_txt(results: List[ResourceResult]):
    """Save a human-readable comparison summary."""
    RESULTS_DIR.mkdir(exist_ok=True)
    fp = RESULTS_DIR / "comparison.txt"

    L = []
    L.append("=" * 120)
    L.append(f"  RESOURCE ANALYSIS — FULL COMPARISON")
    L.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 120)
    L.append("")

    # Table header
    L.append(f"  {'Mode':<26s}  {'Acc%':>6s}  {'Sens%':>6s}  {'Spec%':>6s}  {'F1%':>5s}  "
             f"{'Time/user':>10s}  {'Peak MB':>8s}  {'Avg Mults/user':>15s}  {'Total s':>8s}")
    L.append(f"  {'─'*26}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  "
             f"{'─'*10}  {'─'*8}  {'─'*15}  {'─'*8}")

    for r in results:
        L.append(f"  {r.mode:<26s}  {r.accuracy*100:>5.1f}%  {r.sensitivity*100:>5.1f}%  "
                 f"{r.specificity*100:>5.1f}%  {r.f1*100:>4.1f}%  "
                 f"{r.avg_time_per_user_ms:>9.0f}ms  {r.peak_memory_mb:>7.1f}M  "
                 f"{r.avg_multiplications_per_user:>15,.0f}  {r.total_time_s:>7.1f}s")
    L.append("")

    # Rankings
    L.append("-" * 120)
    L.append("  RANKINGS")
    L.append("-" * 120)
    by_acc = sorted(results, key=lambda r: r.accuracy, reverse=True)
    by_speed = sorted(results, key=lambda r: r.avg_time_per_user_ms)
    by_mem = sorted(results, key=lambda r: r.peak_memory_mb)
    by_cost = sorted(results, key=lambda r: r.avg_multiplications_per_user)

    L.append(f"  Best accuracy:    {by_acc[0].mode} ({by_acc[0].accuracy*100:.1f}%)")
    L.append(f"  Fastest:          {by_speed[0].mode} ({by_speed[0].avg_time_per_user_ms:.0f}ms/user)")
    L.append(f"  Lowest memory:    {by_mem[0].mode} ({by_mem[0].peak_memory_mb:.1f}MB)")
    L.append(f"  Cheapest compute: {by_cost[0].mode} ({by_cost[0].avg_multiplications_per_user:,.0f} mults/user)")
    L.append("")

    # Time breakdown
    L.append("-" * 120)
    L.append("  TIME BREAKDOWN (avg ms per user)")
    L.append("-" * 120)
    L.append(f"  {'Mode':<26s}  {'CDS_train':>10s}  {'CDS_pred':>9s}  {'ML_train':>9s}  {'ML_pred':>8s}  {'Total':>8s}")
    L.append(f"  {'─'*26}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*8}")
    for r in results:
        L.append(f"  {r.mode:<26s}  {r.avg_cds_train_ms:>10.0f}  {r.avg_cds_predict_ms:>9.0f}  "
                 f"{r.avg_ml_train_ms:>9.0f}  {r.avg_ml_predict_ms:>8.0f}  {r.avg_time_per_user_ms:>8.0f}")
    L.append("")

    # Error overlap
    L.append("-" * 120)
    L.append("  ERROR OVERLAP")
    L.append("-" * 120)
    for r in results:
        wrong = sorted(set(rec.user_idx for rec in r.records if not rec.is_correct))
        L.append(f"  {r.mode:<26s}  errors: {wrong if wrong else '(none)'}")
    L.append("")
    L.append("=" * 120)

    fp.write_text("\n".join(L), encoding="utf-8")
    log.info(f"  Summary saved: {fp}")
    return fp


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Resource Analysis for CDS-ANN-RNN")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--mode", choices=list(MODE_CONFIGS.keys()) + ["compare-all"],
                        default="compare-all")
    args = parser.parse_args()

    _suppress_cds_loggers()
    data, labels = load_uci_data()
    _suppress_cds_loggers()

    log.info(f"\nDataset: {data.shape[0]} users x {data.shape[1]} features")
    log.info(f"Label distribution: { {int(c): int((labels==c).sum()) for c in sorted(set(labels))} }")

    results = []

    if args.mode == "compare-all":
        log.info(f"\nRunning all 13 combinations...\n")
        for key in MODE_CONFIGS:
            r = run_mode(data, labels, key, args.max_users)
            save_result(r)
            results.append(r)
            log.info(f"  >> {r.mode}: acc={r.accuracy*100:.1f}%  "
                     f"time={r.avg_time_per_user_ms:.0f}ms/user  "
                     f"mem={r.peak_memory_mb:.1f}MB  "
                     f"mults={r.avg_multiplications_per_user:,.0f}/user\n")
        save_comparison_csv(results)
        save_comparison_txt(results)
    else:
        r = run_mode(data, labels, args.mode, args.max_users)
        save_result(r)
        results.append(r)

    return results


if __name__ == "__main__":
    main()
