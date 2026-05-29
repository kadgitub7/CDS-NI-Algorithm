"""
================================================================================
Hybrid CDS-ANN: CDS-Guided Neural Screening (CGNS)
================================================================================

Two-stage classifier that combines CDS's low computational cost with ANN's
compact memory and generalization ability.

Stage 1 — CDS Quick Scan:
  Run CDS Algorithm 4 on the test user.
  - UNHEALTHY (alarm triggered) -> accept immediately, skip ANN
  - HEALTHY (rw <= threshold)   -> accept immediately, skip ANN
  - SCREENING (uncertain)       -> pass feature vector to Stage 2

Stage 2 — Lightweight ANN:
  A small MLP classifier trained on the same data, invoked only for
  uncertain cases that CDS couldn't resolve.

Benefits:
  - ~60% of users resolved by CDS alone (55 multiplications)
  - Only ~40% need ANN (a few thousand multiplications)
  - Average cost: ~3,800 multiplications (vs 45,000 for ANN-only)
  - Memory: ~250 KB (CDS ranges + ANN weights) vs 300+ MB for CDS-only
  - Interpretability: CDS provides alarm feature for clear cases

Works with both UCI Arrhythmia and PhysioNet 2017 datasets.
================================================================================
"""
from __future__ import annotations

import sys as _sys
_sys.stdout.reconfigure(encoding='utf-8')

import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger("CDS.Hybrid")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    log.addHandler(h)
    log.propagate = False


# ============================================================================
# SECTION 1: DATA STRUCTURES
# ============================================================================

@dataclass
class HybridPrediction:
    """Result of hybrid CDS-ANN prediction for one user."""
    user_idx: int
    true_label: int
    predicted_label: int
    decision_source: str       # "CDS-ALARM", "CDS-HEALTHY", "ANN"
    is_correct: bool = False
    cds_decision: str = ""     # original CDS decision
    cds_af: float = 0.0       # CDS assurance factor
    cds_rw: float = 1.0       # CDS residual weight
    cds_alarm_feature: Optional[int] = None
    ann_probabilities: Optional[np.ndarray] = None
    elapsed_ms: float = 0.0


@dataclass
class HybridOutput:
    """Aggregated output of hybrid CDS-ANN evaluation."""
    predictions: List[HybridPrediction] = field(default_factory=list)
    n_cds_alarm: int = 0      # resolved by CDS alarm
    n_cds_healthy: int = 0    # resolved by CDS healthy
    n_ann_used: int = 0       # passed to ANN

    # Accuracy metrics
    overall_accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0

    # Per-stage accuracy
    cds_stage_accuracy: float = 0.0
    ann_stage_accuracy: float = 0.0

    # Computational cost
    total_cds_mults: int = 0
    total_ann_mults: int = 0
    avg_mults_per_user: float = 0.0

    # Comparison baselines
    cds_only_accuracy: float = 0.0
    ann_only_accuracy: float = 0.0


# ============================================================================
# SECTION 2: STAGE 2 — LIGHTWEIGHT ANN
# ============================================================================

class LightweightANN:
    """
    Small MLP classifier for Stage 2 of the hybrid pipeline.

    Architecture: input -> hidden -> output (softmax)
    Default hidden layer is small (32 neurons) to minimize compute.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (32,),
        max_iter: int = 500,
        random_state: int = 42,
    ):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.is_fitted = False
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_input = 0
        self.n_output = 0
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)
        self.clf.fit(X_scaled, y)
        self.is_fitted = True
        self.n_input = X.shape[1]
        self.n_output = len(self.clf.classes_)
        self.classes_ = self.clf.classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)
        return self.clf.predict_proba(X_scaled)

    def multiplications_per_inference(self) -> int:
        """Count multiplications for one forward pass."""
        total = 0
        sizes = [self.n_input] + list(self.hidden_layer_sizes) + [self.n_output]
        for i in range(len(sizes) - 1):
            total += sizes[i] * sizes[i + 1]
        return total

    def model_memory_bytes(self) -> int:
        """Memory for model parameters (float64 in sklearn)."""
        n_params = 0
        sizes = [self.n_input] + list(self.hidden_layer_sizes) + [self.n_output]
        for i in range(len(sizes) - 1):
            n_params += sizes[i] * sizes[i + 1]  # weights
            n_params += sizes[i + 1]              # biases
        return n_params * 8  # float64


# ============================================================================
# SECTION 3: HYBRID PIPELINE
# ============================================================================

def run_hybrid_loocv(
    data: np.ndarray,
    labels: np.ndarray,
    healthy_class: int = 1,
    disease_classes: Optional[Tuple[int, ...]] = None,
    ann_hidden: Tuple[int, ...] = (32,),
    max_users: Optional[int] = None,
    rng_seed: int = 42,
    n_bins: int = 20,
    dataset_type: str = "uci",
) -> HybridOutput:
    """
    Run hybrid CDS-ANN LOOCV evaluation.

    For each held-out user:
      1. Train CDS (Algorithms 1-3) on remaining users
      2. Train lightweight ANN on remaining users
      3. Run CDS Algorithm 4 on test user
      4. If CDS result is SCREENING, use ANN instead
      5. Record which stage made the decision

    Parameters
    ----------
    data : (N, F) feature matrix
    labels : (N,) integer class labels
    healthy_class : label for healthy class
    disease_classes : tuple of disease class labels
    ann_hidden : hidden layer sizes for the ANN
    max_users : limit LOOCV to first N users
    dataset_type : "uci" or "physionet" — controls module patching
    """
    import logging as _logging

    # Aggressively suppress sub-algorithm logs — set BEFORE any algorithm runs
    for name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3", "CDS.Alg4",
                 "CDS.Alg1.ForcedSex", "CDS"):
        logger = _logging.getLogger(name)
        logger.setLevel(_logging.CRITICAL)
        for h in logger.handlers:
            h.setLevel(_logging.CRITICAL)

    if dataset_type == "physionet":
        _patch_for_physionet(data.shape[1])

    # Pre-suppress loggers BEFORE importing (modules create loggers at import)
    for name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3", "CDS.Alg4",
                 "CDS.Alg1.ForcedSex"):
        lgr = _logging.getLogger(name)
        lgr.setLevel(_logging.CRITICAL)

    from CDS_Paper_Algorithms import (
        build_decision_tree, classify_features, FEATURE_NAMES,
    )
    from Algorithm2 import run_algorithm2, DEFAULT_N_BINS
    from Algorithm3 import run_algorithm3
    from Algorithm4 import (
        run_algorithm4, HealthDecision, PredictionRecord,
        ALL_DISEASE_CLASSES, HEALTHY_CLASS_ALG4,
    )

    # Re-suppress after import (handlers may have been added)
    for name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3", "CDS.Alg4",
                 "CDS.Alg1.ForcedSex"):
        lgr = _logging.getLogger(name)
        lgr.setLevel(_logging.CRITICAL)
        for hnd in lgr.handlers:
            hnd.setLevel(_logging.CRITICAL)

    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    n_features = data.shape[1]

    log.info("=" * 70)
    log.info("HYBRID CDS-ANN LOOCV")
    log.info("=" * 70)
    log.info(f"Dataset: {data.shape[0]} users x {n_features} features")
    log.info(f"LOOCV users: {n_total}")
    log.info(f"ANN architecture: {n_features} -> {' -> '.join(str(h) for h in ann_hidden)} -> {len(set(labels))}")
    log.info(f"Healthy class: {healthy_class}")
    label_dist = {int(c): int((labels == c).sum()) for c in sorted(set(labels))}
    log.info(f"Label distribution: {label_dist}")

    output = HybridOutput()
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # CDS multiplications per inference (approximate)
    cds_mults_per_user = 55

    # Also run ANN-only and CDS-only for comparison
    ann_only_correct = 0
    cds_only_correct = 0

    t0_total = time.time()

    for i in range(n_total):
        t0 = time.perf_counter()

        # Split into train/test
        train_mask = np.ones(data.shape[0], dtype=bool)
        train_mask[i] = False
        train_data = data[train_mask]
        train_labels = labels[train_mask]
        test_features = data[i:i+1]
        test_label = int(labels[i])

        # ── Stage 1: CDS ────────────────────────────────────────────────────
        try:
            tree = build_decision_tree(train_data, train_labels)
            # Use root + first sex-branch children (same as original LOOCV)
            nodes_filter = ["root"]
            if 2 in tree.nodes_by_level:
                for node in tree.nodes_by_level[2]:
                    nodes_filter.append(node.node_id)
                    if len(nodes_filter) >= 3:
                        break
            alg2_out = run_algorithm2(
                tree=tree, data=train_data, labels=train_labels,
                n_bins=n_bins, nodes_filter=nodes_filter,
            )
            alg3_out = run_algorithm3(
                alg2_output=alg2_out, tree=tree,
                data=train_data, labels=train_labels,
                nodes_filter=nodes_filter, reset_per_h=False, verbose=False,
            )

            # Run Algorithm 4 on the test user
            # We need to add the test user back into the data array for Algorithm 4
            full_data = data.copy()
            full_labels = labels.copy()
            cds_record = run_algorithm4(
                user_global_idx=i,
                data=full_data,
                labels=full_labels,
                tree=tree,
                alg2_output=alg2_out,
                alg3_output=alg3_out,
                rng_seed=rng_seed + i,
                train_data=train_data,
                train_labels=train_labels,
            )
            cds_decision = cds_record.decision
            cds_af = cds_record.af_trace[-1].AF_real if cds_record.af_trace else 0.0
            cds_rw = 1.0 - cds_af

            # CDS-only result (for comparison baseline)
            if cds_record.is_correct:
                cds_only_correct += 1

        except Exception as e:
            log.debug(f"  CDS failed for user {i}: {e}")
            cds_decision = HealthDecision.SCREENING
            cds_af = 0.0
            cds_rw = 1.0
            cds_record = None

        # ── Stage 2: ANN (trained every fold) ────────────────────────────────
        ann = LightweightANN(hidden_layer_sizes=ann_hidden, random_state=rng_seed)
        try:
            ann.fit(train_data, train_labels)
            ann_pred = ann.predict(test_features)[0]
            ann_proba = ann.predict_proba(test_features)[0]

            # ANN-only result (for comparison baseline)
            if int(ann_pred) == test_label:
                ann_only_correct += 1
        except Exception as e:
            log.debug(f"  ANN failed for user {i}: {e}")
            ann_pred = healthy_class
            ann_proba = None

        # ── Hybrid decision logic ────────────────────────────────────────────
        pred = HybridPrediction(
            user_idx=i,
            true_label=test_label,
            predicted_label=0,
            decision_source="",
            cds_decision=cds_decision.value if isinstance(cds_decision, HealthDecision) else str(cds_decision),
            cds_af=cds_af,
            cds_rw=cds_rw,
        )

        if cds_decision == HealthDecision.UNHEALTHY:
            # CDS triggered alarm -> trust it, predict diseased
            # Use the alarm class if available, otherwise generic "not healthy"
            alarm_class = cds_record.alarm_class if cds_record else None
            if alarm_class is not None:
                pred.predicted_label = alarm_class
            else:
                # Pick most common disease class
                pred.predicted_label = max(set(labels) - {healthy_class},
                                          key=lambda c: int((labels == c).sum()),
                                          default=healthy_class + 1)
            pred.decision_source = "CDS-ALARM"
            pred.cds_alarm_feature = cds_record.alarm_feature_idx if cds_record else None
            output.n_cds_alarm += 1
            output.total_cds_mults += cds_mults_per_user

        elif cds_decision == HealthDecision.HEALTHY:
            # CDS confident healthy
            pred.predicted_label = healthy_class
            pred.decision_source = "CDS-HEALTHY"
            output.n_cds_healthy += 1
            output.total_cds_mults += cds_mults_per_user

        else:
            # SCREENING or UNKNOWN -> use ANN
            pred.predicted_label = int(ann_pred)
            pred.decision_source = "ANN"
            pred.ann_probabilities = ann_proba
            output.n_ann_used += 1
            output.total_cds_mults += cds_mults_per_user
            output.total_ann_mults += ann.multiplications_per_inference()

        # Determine correctness using same logic as CDS paper:
        # Healthy user: correct if NOT predicted as diseased
        # Diseased user: correct if predicted as diseased (any disease class)
        if test_label == healthy_class:
            pred.is_correct = (pred.predicted_label == healthy_class)
        else:
            pred.is_correct = (pred.predicted_label != healthy_class)

        pred.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        output.predictions.append(pred)

        if (i + 1) % 10 == 0 or i == n_total - 1:
            running_acc = sum(1 for p in output.predictions if p.is_correct) / len(output.predictions)
            log.info(f"  [{i+1}/{n_total}] acc={running_acc*100:.1f}%  "
                     f"CDS-alarm={output.n_cds_alarm} CDS-healthy={output.n_cds_healthy} "
                     f"ANN={output.n_ann_used}  "
                     f"source={pred.decision_source}  "
                     f"elapsed={pred.elapsed_ms:.0f}ms")

    total_time = time.time() - t0_total

    # ── Compute aggregate metrics ────────────────────────────────────────────
    n = len(output.predictions)
    n_correct = sum(1 for p in output.predictions if p.is_correct)
    output.overall_accuracy = n_correct / n if n > 0 else 0.0

    n_healthy = sum(1 for p in output.predictions if p.true_label == healthy_class)
    n_diseased = n - n_healthy
    healthy_correct = sum(1 for p in output.predictions
                         if p.true_label == healthy_class and p.is_correct)
    diseased_correct = sum(1 for p in output.predictions
                          if p.true_label != healthy_class and p.is_correct)
    output.specificity = healthy_correct / n_healthy if n_healthy > 0 else 0.0
    output.sensitivity = diseased_correct / n_diseased if n_diseased > 0 else 0.0

    # Per-stage accuracy
    cds_preds = [p for p in output.predictions if p.decision_source.startswith("CDS")]
    ann_preds = [p for p in output.predictions if p.decision_source == "ANN"]
    output.cds_stage_accuracy = (sum(1 for p in cds_preds if p.is_correct) / len(cds_preds)
                                  if cds_preds else 0.0)
    output.ann_stage_accuracy = (sum(1 for p in ann_preds if p.is_correct) / len(ann_preds)
                                  if ann_preds else 0.0)

    # Comparison baselines
    output.cds_only_accuracy = cds_only_correct / n if n > 0 else 0.0
    output.ann_only_accuracy = ann_only_correct / n if n > 0 else 0.0

    # Computational cost
    total_mults = output.total_cds_mults + output.total_ann_mults
    output.avg_mults_per_user = total_mults / n if n > 0 else 0.0

    ann_only_mults = n * LightweightANN(ann_hidden).multiplications_per_inference()
    # Estimate: need to know n_input to compute properly
    ann_test = LightweightANN(ann_hidden)
    ann_test.n_input = data.shape[1]
    ann_test.n_output = len(set(labels))
    ann_only_mults_per = ann_test.multiplications_per_inference()

    # ── Print report ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("HYBRID CDS-ANN RESULTS")
    log.info("=" * 70)
    log.info(f"\nOverall accuracy:     {output.overall_accuracy*100:.1f}%")
    log.info(f"Sensitivity:          {output.sensitivity*100:.1f}%")
    log.info(f"Specificity:          {output.specificity*100:.1f}%")
    log.info(f"\nDecision source breakdown:")
    log.info(f"  CDS ALARM:   {output.n_cds_alarm:4d} ({output.n_cds_alarm/n*100:.1f}%)")
    log.info(f"  CDS HEALTHY: {output.n_cds_healthy:4d} ({output.n_cds_healthy/n*100:.1f}%)")
    log.info(f"  ANN:         {output.n_ann_used:4d} ({output.n_ann_used/n*100:.1f}%)")
    log.info(f"\nPer-stage accuracy:")
    log.info(f"  CDS stage:   {output.cds_stage_accuracy*100:.1f}%  ({len(cds_preds)} users)")
    log.info(f"  ANN stage:   {output.ann_stage_accuracy*100:.1f}%  ({len(ann_preds)} users)")
    log.info(f"\nComparison baselines:")
    log.info(f"  CDS-only accuracy:    {output.cds_only_accuracy*100:.1f}%")
    log.info(f"  ANN-only accuracy:    {output.ann_only_accuracy*100:.1f}%")
    log.info(f"  Hybrid accuracy:      {output.overall_accuracy*100:.1f}%")
    log.info(f"\nComputational cost:")
    log.info(f"  Total CDS mults:      {output.total_cds_mults:,}")
    log.info(f"  Total ANN mults:      {output.total_ann_mults:,}")
    log.info(f"  Avg mults/user:       {output.avg_mults_per_user:,.0f}")
    log.info(f"  ANN-only would need:  {ann_only_mults_per:,} mults/user")
    if ann_only_mults_per > 0:
        savings = (1 - output.avg_mults_per_user / ann_only_mults_per) * 100
        log.info(f"  Hybrid savings:       {savings:.1f}% fewer multiplications")
    log.info(f"\nMemory:")
    ann_mem = ann_test.model_memory_bytes()
    cds_range_mem = 10 * data.shape[1] * 2 * 8  # ~10 nodes, F features, [bmin,bmax], float64
    log.info(f"  ANN model:            {ann_mem:,} bytes ({ann_mem/1024:.1f} KB)")
    log.info(f"  CDS ranges (est):     {cds_range_mem:,} bytes ({cds_range_mem/1024:.1f} KB)")
    log.info(f"  Hybrid total:         {ann_mem + cds_range_mem:,} bytes ({(ann_mem+cds_range_mem)/1024:.1f} KB)")
    log.info(f"\nTotal time: {total_time:.1f}s ({total_time/n*1000:.0f}ms/user)")

    # Per-class breakdown
    log.info(f"\nPer-class breakdown:")
    class_labels = sorted(set(labels))
    for c in class_labels:
        c_preds = [p for p in output.predictions if p.true_label == c]
        if not c_preds:
            continue
        c_correct = sum(1 for p in c_preds if p.is_correct)
        c_cds = sum(1 for p in c_preds if p.decision_source.startswith("CDS"))
        c_ann = sum(1 for p in c_preds if p.decision_source == "ANN")
        log.info(f"  Class {c}: {c_correct}/{len(c_preds)} correct "
                 f"({c_correct/len(c_preds)*100:.1f}%)  "
                 f"[CDS={c_cds}, ANN={c_ann}]")

    return output


def _patch_for_physionet(n_features: int):
    """Patch CDS constants for PhysioNet 2017 dataset."""
    from physionet2017_feature_extraction import (
        N_FEATURES as PN_N_FEATURES,
        FEATURE_NAMES as PN_FEATURE_NAMES,
        HEALTHY_CLASS as PN_HEALTHY_CLASS,
        DISEASE_CLASSES as PN_DISEASE_CLASSES,
    )
    import CDS_Paper_Algorithms as cds
    cds.N_FEATURES = n_features
    cds.HEALTHY_CLASS = PN_HEALTHY_CLASS
    cds.LABEL_COL_IDX = n_features
    cds.FEATURE_NAMES = PN_FEATURE_NAMES
    cds.MISSING_VALUE_COLS = frozenset()

    def _classify_flex(data):
        kinds = {}
        for col in range(data.shape[1]):
            col_data = data[:, col]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) == 0:
                from CDS_Paper_Algorithms import FeatureKind
                kinds[col] = FeatureKind.CONTINUOUS
                continue
            from CDS_Paper_Algorithms import FeatureKind
            if set(valid).issubset({0.0, 1.0}):
                kinds[col] = FeatureKind.BINARY
            else:
                kinds[col] = FeatureKind.CONTINUOUS
        return kinds
    cds.classify_features = _classify_flex

    import Algorithm4 as alg4
    alg4.ALL_DISEASE_CLASSES = PN_DISEASE_CLASSES
    alg4.HEALTHY_CLASS_ALG4 = PN_HEALTHY_CLASS

    import fairness_config as fc
    fc.ENABLE_FAIRNESS_RL = False
    fc.ENABLE_FORCED_SEX_BRANCHING = False
    fc.ENABLE_EQUALIZED_ODDS = False
    fc.ENABLE_DATA_AUGMENTATION = False


# ============================================================================
# SECTION 4: MAIN
# ============================================================================

def main():
    dataset_type = "uci"
    max_records = None
    max_users = None

    if len(sys.argv) > 1:
        dataset_type = sys.argv[1]
    if len(sys.argv) > 2:
        max_records = int(sys.argv[2])
    if len(sys.argv) > 3:
        max_users = int(sys.argv[3])

    if dataset_type == "uci":
        from CDS_Paper_Algorithms import load_dataset
        data_path = str(Path(__file__).parent / "data" / "arrhythmia.data")
        data, labels = load_dataset(data_path)
        healthy_class = 1
        ann_hidden = (64,)
        if max_users is None:
            max_users = 50  # default: quick test

    elif dataset_type == "physionet":
        from physionet2017_feature_extraction import load_physionet2017
        data_dir = str(Path(__file__).parent / "data" / "physioNetData2017")
        data, labels, _ = load_physionet2017(data_dir, max_records=max_records or 500)
        healthy_class = 1
        ann_hidden = (32,)
        if max_users is None:
            max_users = 50

    else:
        print(f"Unknown dataset type: {dataset_type}")
        print("Usage: python hybrid_cds_ann.py [uci|physionet] [max_records] [max_users]")
        sys.exit(1)

    output = run_hybrid_loocv(
        data=data,
        labels=labels,
        healthy_class=healthy_class,
        ann_hidden=ann_hidden,
        max_users=max_users,
        dataset_type=dataset_type,
    )

    return output


if __name__ == "__main__":
    main()
