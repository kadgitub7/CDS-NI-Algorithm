"""
================================================================================
Algorithm 4: CDS User-Health Prediction
(Planner, Reinforcement Learning, Policy in Executive
 and Running Diagnostic Test in the Perceptor)
REVISED FOR:
  • UCI Arrhythmia tabular dataset
  • WFDB-based datasets via tabularized feature extraction
PART 1 OF 4
================================================================================

PURPOSE
-------
Algorithm 4 is the prediction / inference phase of the CDS. Given a single
test user/record whose health state is unknown, it applies a sequence of
cognitive actions (sensor activations), accumulates an Assurance Factor (AF),
and produces one of three decisions:

  UNHEALTHY  – at least one feature value fell outside the healthy range;
               the user/record is referred for disease diagnosis.
  HEALTHY    – rw = (1 - AF) ≤ Threshold; the user/record is referred for
               healthy-living recommendations.
  SCREENING  – all disease classes checked but AF still insufficient;
               the user/record is sent for screening.

PAPER FIDELITY NOTATION
-----------------------
  [PAPER]  – directly stated in the paper / pseudocode.
  [INFER]  – logically required but unspecified in the paper.
  [ENGR]   – engineering choice with justification.
  [AMBIG]  – paper is ambiguous; chosen interpretation is documented.

REVISION GOALS
--------------
This revision preserves the original Algorithm 4 structure while adding:
  1. Dataset-schema awareness via Algorithm 1 / 2 abstractions
  2. Removal of hard-coded Arrhythmia-only assumptions:
       • 279 feature count
       • 452 user count
       • healthy class fixed to 1 only
       • sex feature fixed at column 1 with male=0 / female=1 only
       • disease class set fixed to the UCI Arrhythmia labels only
  3. Compatibility with WFDB-derived tabular matrices produced by Algorithm 1
  4. Optional compatibility helpers for dataset introspection / smoke tests
  5. Backward compatibility with original Arrhythmia behavior

IMPORTANT WFDB NOTE
-------------------
Algorithm 4 consumes:
  • DecisionTree (Algorithm 1)
  • Algorithm2Output – perceptor model library
  • Algorithm3Output – refined executive action library
  • A tabular feature matrix (N × F)

WFDB waveform databases are NOT consumed directly here. They must first be
converted into one fixed-length feature vector per record by Algorithm 1's
generalized loading / tabularization layer.

================================================================================
"""

from __future__ import annotations

import copy
import logging
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd

# ── Import from Algorithms 1–3 ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from Algorithm1 import (
    DecisionTree,
    TreeNode,
    BranchDef,
    FeatureKind,
    load_dataset,
    load_dataset_bundle,
    build_decision_tree,
    FEATURE_NAMES,
    HEALTHY_CLASS,
    DIAGNOSTIC_THRESHOLD,
    U_MIN,
    N_FEATURES,
    DatasetSchema,
    DatasetBundle,
)
from Algorithm2 import (
    Algorithm2Output,
    ExecutiveActionEntry,
    PerceptorModelEntry,
    run_algorithm2,
    DEFAULT_N_BINS,
)
from Algorithm3 import (
    Algorithm3Output,
    run_algorithm3,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger(name: str = "CDS.Alg4") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)-7s | %(name)s | %(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)   # default: INFO; set to DEBUG for full traces
    h.setFormatter(fmt)
    log.addHandler(h)
    log.propagate = False
    return log

log = _build_logger()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – CONSTANTS (must match Algorithms 1–3)
# ─────────────────────────────────────────────────────────────────────────────

# [PAPER] Eq. 3 / Section VI.A: acceptable error probability
DIAGNOSTIC_THRESHOLD_ALG4: float = DIAGNOSTIC_THRESHOLD

# [PAPER] Eq. 3 / Section VII.A: minimum users for reliable model extraction
U_MIN_ALG4: int = U_MIN

# [LEGACY BACKWARD-COMPATIBLE DEFAULT]
HEALTHY_CLASS_ALG4: int = HEALTHY_CLASS

# [LEGACY DEFAULT] retained only for backward compatibility / paper reproduction.
ALL_DISEASE_CLASSES: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16)

# AF accumulates across the disease-class loop.
AF_NORMALISE_TO_CAPACITY: bool = False
AF_DISEASE_SCALE: float = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS: IN-PROCESSING RL REWARD MODIFICATION
# ─────────────────────────────────────────────────────────────────────────────
import fairness_config as _fairness_cfg

SEX_FEATURE_INDEX_ALG4: int = _fairness_cfg.SEX_FEATURE_INDEX
FAIRNESS_LAMBDA: float = _fairness_cfg.FAIRNESS_LAMBDA


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2B – SCHEMA / DATASET GENERALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _schema_from_tree(tree: DecisionTree) -> Optional[DatasetSchema]:
    """Return schema from tree if present."""
    return getattr(tree, "schema", None)


def _healthy_class(schema: Optional[DatasetSchema]) -> int:
    """Healthy class from schema or fallback constant."""
    if schema is not None:
        return schema.healthy_class
    return HEALTHY_CLASS_ALG4


def _feature_name(feature_idx: int, schema: Optional[DatasetSchema]) -> str:
    """Generalized feature-name lookup."""
    if schema is not None:
        return schema.feature_name(feature_idx)
    return FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")


def _sex_feature_index(schema: Optional[DatasetSchema]) -> Optional[int]:
    """Sex feature index from schema when available."""
    if schema is not None:
        return schema.sex_feature_index
    return SEX_FEATURE_INDEX_ALG4


def _male_code(schema: Optional[DatasetSchema]) -> Optional[float]:
    """Male code from schema when available."""
    if schema is not None:
        return schema.male_code
    return 0.0


def _female_code(schema: Optional[DatasetSchema]) -> Optional[float]:
    """Female code from schema when available."""
    if schema is not None:
        return schema.female_code
    return 1.0


def _all_classes_from_labels(labels: np.ndarray) -> List[int]:
    """Sorted unique classes present in labels."""
    return sorted(int(x) for x in np.unique(labels).tolist())


def _all_disease_classes_from_labels(labels: np.ndarray,
                                     schema: Optional[DatasetSchema]) -> List[int]:
    """All non-healthy classes present in labels."""
    hcls = _healthy_class(schema)
    return sorted(int(x) for x in np.unique(labels).tolist() if int(x) != hcls)


def _all_disease_classes_from_node(node: TreeNode,
                                   schema: Optional[DatasetSchema]) -> List[int]:
    """All non-healthy classes present in this tree node."""
    hcls = _healthy_class(schema)
    return sorted(int(h) for h in node.health_dist.keys() if int(h) != hcls and node.health_dist[h] > 0)


def _dataset_description(schema: Optional[DatasetSchema],
                         data: Optional[np.ndarray] = None,
                         labels: Optional[np.ndarray] = None) -> str:
    """Compact dataset descriptor for logs."""
    if schema is not None:
        base = f"{schema.dataset_name} ({schema.source_format})"
        if data is not None and labels is not None:
            return f"{base}: N={data.shape[0]}, F={data.shape[1]}, classes={sorted(np.unique(labels).tolist())}"
        return base
    if data is not None and labels is not None:
        return f"LegacyDataset: N={data.shape[0]}, F={data.shape[1]}, classes={sorted(np.unique(labels).tolist())}"
    return "unknown-dataset"


def _wfdb_or_tabular_guard(data: np.ndarray, labels: np.ndarray) -> None:
    """
    Enforce that Algorithm 4 receives a 2D tabular matrix plus 1D label vector.

    WFDB support is achieved through prior tabularization, not direct waveform
    handling inside Algorithm 4.
    """
    if data.ndim != 2:
        raise ValueError(f"Algorithm 4 expects a 2D tabular matrix. Got shape={data.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Algorithm 4 expects a 1D labels vector. Got shape={labels.shape}")
    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Row mismatch: data.shape[0]={data.shape[0]} != labels.shape[0]={labels.shape[0]}"
        )


def _sex_value_to_group(value: float,
                        schema: Optional[DatasetSchema]) -> Optional[str]:
    """
    Map a sex-coded scalar to 'male' / 'female' / None using schema metadata.
    """
    if np.isnan(value):
        return None
    m = _male_code(schema)
    f = _female_code(schema)
    if m is not None and value == m:
        return "male"
    if f is not None and value == f:
        return "female"
    return None


def _user_group_label(user_global_idx: int,
                      data: np.ndarray,
                      schema: Optional[DatasetSchema]) -> str:
    """
    Human-readable user sex/group label where available.
    """
    sidx = _sex_feature_index(schema)
    if sidx is None or sidx < 0 or sidx >= data.shape[1]:
        return "unknown"
    val = data[user_global_idx, sidx]
    grp = _sex_value_to_group(val, schema)
    return grp if grp is not None else "unknown"


def _is_valid_feature_idx(feature_idx: int, data: np.ndarray) -> bool:
    """Validate feature index against matrix dimensionality."""
    return 0 <= int(feature_idx) < data.shape[1]


def _feature_dimensionality(data: np.ndarray) -> int:
    """Return number of features in the current tabular dataset."""
    return int(data.shape[1])


def _user_count(data: np.ndarray) -> int:
    """Return number of users/records in the current tabular dataset."""
    return int(data.shape[0])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class HealthDecision(Enum):
    """
    [PAPER] Section VI.B: three possible outcomes of Algorithm 4.
    """
    HEALTHY   = "Healthy"
    UNHEALTHY = "Unhealthy"
    SCREENING = "Screening"
    UNKNOWN   = "Unknown"


@dataclass
class AFTraceEntry:
    """
    One step in the Assurance Factor trace.
    """
    pac_number:     int
    node_id:        str
    focus_level:    int
    disease_h:      int
    feature_idx:    int
    feature_name:   str
    raw_value:      float
    b_min:          float
    b_max:          float
    r_j_h:          float
    p_h_f:          float
    p_h_gt1_f:      float
    delta_AF:       float
    AF_real:        float
    rw_real:        float
    triggered_alarm: bool
    is_nan:         bool


@dataclass
class RLLookaheadEntry:
    """
    RL lookahead record for one candidate action during the RL selection step.
    """
    action_feature_idx:  int
    action_feature_name: str
    action_weight:       float
    AF_sim_increment:    float
    rw_sim:              float
    is_selected:         bool


@dataclass
class DiseaseCheckRecord:
    """
    Record of checking one disease class h for one user/record in one node.
    """
    node_id:           str
    focus_level:       int
    disease_h:         int
    n_actions_in_buf:  int
    n_actions_applied: int
    alarm_triggered:   bool
    alarm_feature_idx: Optional[int]
    alarm_raw_value:   Optional[float]
    alarm_b_min:       Optional[float]
    alarm_b_max:       Optional[float]
    AF_real_at_end:    float
    rw_real_at_end:    float
    rl_selections:     List[RLLookaheadEntry] = field(default_factory=list)


@dataclass
class NodePredictionRecord:
    """
    Record of all predictions performed at one tree node for one user/record.
    """
    node_id:              str
    focus_level:          int
    disease_checks:       List[DiseaseCheckRecord] = field(default_factory=list)
    final_AF_real:        float = 0.0
    final_rw_real:        float = 1.0
    met_threshold:        bool  = False
    focus_increased:      bool  = False
    sent_to_screening:    bool  = False


@dataclass
class PredictionRecord:
    """
    Complete prediction record for one user/record.
    """
    user_global_idx:      int
    true_label:           int
    decision:             HealthDecision = HealthDecision.UNKNOWN
    is_correct:           bool           = False
    alarm_class:          Optional[int]  = None
    alarm_feature_idx:    Optional[int]  = None
    max_focus_reached:    int            = 1
    total_pac_count:      int            = 0
    total_actions_applied:int            = 0
    elapsed_ms:           float          = 0.0
    af_trace:             List[AFTraceEntry]          = field(default_factory=list)
    node_records:         List[NodePredictionRecord]  = field(default_factory=list)
    initial_action_feat:  Optional[int]               = None
    dataset_name:         str                         = ""
    source_format:        str                         = ""

    @property
    def true_is_healthy(self) -> bool:
        return False

    @property
    def true_is_diseased(self) -> bool:
        return not self.true_is_healthy


@dataclass
class Algorithm4Output:
    """
    Aggregated output of running Algorithm 4 over multiple users/records.
    """
    records:           List[PredictionRecord] = field(default_factory=list)
    data:              Optional[np.ndarray] = None
    schema:            Optional[DatasetSchema] = None

    n_healthy_correct: int   = 0
    n_healthy_total:   int   = 0
    n_diseased_correct:int   = 0
    n_diseased_total:  int   = 0
    n_screening:       int   = 0
    overall_accuracy:  float = 0.0
    sensitivity:       float = 0.0
    specificity:       float = 0.0
    false_alarm_rate:  float = 0.0
    total_elapsed_ms:  float = 0.0

    total_abnormal_males:       int = 0
    total_abnormal_females:     int = 0
    incorrect_abnormal_males:   int = 0
    incorrect_abnormal_females: int = 0
    error_rate_abnormal_males:  float = 0.0
    error_rate_abnormal_females:float = 0.0

    healthy_correct_males: int = 0
    healthy_total_males: int = 0
    diseased_correct_males: int = 0
    diseased_total_males: int = 0
    healthy_correct_females: int = 0
    healthy_total_females: int = 0
    diseased_correct_females: int = 0
    diseased_total_females: int = 0

    misclassified_females_by_class: Dict[int, int] = field(default_factory=dict)
    misclassified_males_by_class:   Dict[int, int] = field(default_factory=dict)

    total_men: int = 0
    total_women: int = 0
    abnormal_men: int = 0
    abnormal_women: int = 0
    unhealthy_abnormal_men: int = 0
    unhealthy_abnormal_women: int = 0
    normal_men: int = 0
    normal_women: int = 0
    unhealthy_normal_men: int = 0
    unhealthy_normal_women: int = 0

    fairness_spd: float = 0.0
    fairness_di: float = 0.0
    fairness_eo_diff: float = 0.0
    fairness_eo_tpr_diff: float = 0.0
    fairness_eo_fpr_diff: float = 0.0
    fairness_tpr_male: float = 0.0
    fairness_tpr_female: float = 0.0
    fairness_fpr_male: float = 0.0
    fairness_fpr_female: float = 0.0

    def _recompute_stats(self) -> None:
        """
        Recompute aggregate statistics from records, generalized for schema-aware
        healthy class and optional sex encoding.
        """
        schema = self.schema
        healthy_cls = _healthy_class(schema)

        self.n_healthy_correct = sum(
            1 for r in self.records
            if r.true_label == healthy_cls and r.decision != HealthDecision.UNHEALTHY
        )
        self.n_healthy_total = sum(1 for r in self.records if r.true_label == healthy_cls)

        self.n_diseased_correct = sum(
            1 for r in self.records
            if r.true_label != healthy_cls and r.decision == HealthDecision.UNHEALTHY
        )
        self.n_diseased_total = sum(1 for r in self.records if r.true_label != healthy_cls)

        self.n_screening = sum(1 for r in self.records if r.decision == HealthDecision.SCREENING)

        n_total = len(self.records)
        n_correct_total = self.n_healthy_correct + self.n_diseased_correct
        self.overall_accuracy = n_correct_total / n_total if n_total else 0.0
        self.sensitivity = (self.n_diseased_correct / self.n_diseased_total
                            if self.n_diseased_total else 0.0)
        self.specificity = (self.n_healthy_correct / self.n_healthy_total
                            if self.n_healthy_total else 0.0)

        n_fa = sum(
            1 for r in self.records
            if r.true_label == healthy_cls and r.decision == HealthDecision.UNHEALTHY
        )
        self.false_alarm_rate = n_fa / self.n_healthy_total if self.n_healthy_total else 0.0

        # ── Schema-aware sex/fairness analysis ────────────────────────────────
        if self.data is not None:
            sidx = _sex_feature_index(schema)
            male_code = _male_code(schema)
            female_code = _female_code(schema)

            if sidx is not None and 0 <= sidx < self.data.shape[1] and male_code is not None and female_code is not None:
                def is_male(rec: PredictionRecord) -> bool:
                    return self.data[rec.user_global_idx, sidx] == male_code

                def is_female(rec: PredictionRecord) -> bool:
                    return self.data[rec.user_global_idx, sidx] == female_code

                self.total_abnormal_males = sum(
                    1 for r in self.records if r.true_label != healthy_cls and is_male(r)
                )
                self.total_abnormal_females = sum(
                    1 for r in self.records if r.true_label != healthy_cls and is_female(r)
                )
                self.incorrect_abnormal_males = sum(
                    1 for r in self.records if r.true_label != healthy_cls and not r.is_correct and is_male(r)
                )
                self.incorrect_abnormal_females = sum(
                    1 for r in self.records if r.true_label != healthy_cls and not r.is_correct and is_female(r)
                )

                self.error_rate_abnormal_males = (
                    self.incorrect_abnormal_males / self.total_abnormal_males
                    if self.total_abnormal_males else 0.0
                )
                self.error_rate_abnormal_females = (
                    self.incorrect_abnormal_females / self.total_abnormal_females
                    if self.total_abnormal_females else 0.0
                )

                self.healthy_correct_males = sum(
                    1 for r in self.records
                    if r.true_label == healthy_cls and r.decision != HealthDecision.UNHEALTHY and is_male(r)
                )
                self.healthy_total_males = sum(
                    1 for r in self.records if r.true_label == healthy_cls and is_male(r)
                )
                self.diseased_correct_males = sum(
                    1 for r in self.records
                    if r.true_label != healthy_cls and r.decision == HealthDecision.UNHEALTHY and is_male(r)
                )
                self.diseased_total_males = sum(
                    1 for r in self.records if r.true_label != healthy_cls and is_male(r)
                )

                self.healthy_correct_females = sum(
                    1 for r in self.records
                    if r.true_label == healthy_cls and r.decision != HealthDecision.UNHEALTHY and is_female(r)
                )
                self.healthy_total_females = sum(
                    1 for r in self.records if r.true_label == healthy_cls and is_female(r)
                )
                self.diseased_correct_females = sum(
                    1 for r in self.records
                    if r.true_label != healthy_cls and r.decision == HealthDecision.UNHEALTHY and is_female(r)
                )
                self.diseased_total_females = sum(
                    1 for r in self.records if r.true_label != healthy_cls and is_female(r)
                )

                self.misclassified_females_by_class = {}
                self.misclassified_males_by_class = {}
                for r in self.records:
                    if r.true_label != healthy_cls and not r.is_correct:
                        cls = int(r.true_label)
                        if is_female(r):
                            self.misclassified_females_by_class[cls] = self.misclassified_females_by_class.get(cls, 0) + 1
                        elif is_male(r):
                            self.misclassified_males_by_class[cls] = self.misclassified_males_by_class.get(cls, 0) + 1

                self.total_men = sum(1 for r in self.records if is_male(r))
                self.total_women = sum(1 for r in self.records if is_female(r))
                self.abnormal_men = sum(1 for r in self.records if r.true_label != healthy_cls and is_male(r))
                self.abnormal_women = sum(1 for r in self.records if r.true_label != healthy_cls and is_female(r))
                self.unhealthy_abnormal_men = sum(
                    1 for r in self.records if r.true_label != healthy_cls and r.decision == HealthDecision.UNHEALTHY and is_male(r)
                )
                self.unhealthy_abnormal_women = sum(
                    1 for r in self.records if r.true_label != healthy_cls and r.decision == HealthDecision.UNHEALTHY and is_female(r)
                )
                self.normal_men = sum(1 for r in self.records if r.true_label == healthy_cls and is_male(r))
                self.normal_women = sum(1 for r in self.records if r.true_label == healthy_cls and is_female(r))
                self.unhealthy_normal_men = sum(
                    1 for r in self.records if r.true_label == healthy_cls and r.decision == HealthDecision.UNHEALTHY and is_male(r)
                )
                self.unhealthy_normal_women = sum(
                    1 for r in self.records if r.true_label == healthy_cls and r.decision == HealthDecision.UNHEALTHY and is_female(r)
                )

                p_unhealthy_male = ((self.unhealthy_abnormal_men + self.unhealthy_normal_men) / self.total_men
                                    if self.total_men > 0 else 0.0)
                p_unhealthy_female = ((self.unhealthy_abnormal_women + self.unhealthy_normal_women) / self.total_women
                                      if self.total_women > 0 else 0.0)
                self.fairness_spd = p_unhealthy_male - p_unhealthy_female
                self.fairness_di = p_unhealthy_female / p_unhealthy_male if p_unhealthy_male > 0 else 0.0

                tpr_male = self.unhealthy_abnormal_men / self.abnormal_men if self.abnormal_men > 0 else 0.0
                fpr_male = self.unhealthy_normal_men / self.normal_men if self.normal_men > 0 else 0.0
                tpr_female = self.unhealthy_abnormal_women / self.abnormal_women if self.abnormal_women > 0 else 0.0
                fpr_female = self.unhealthy_normal_women / self.normal_women if self.normal_women > 0 else 0.0

                self.fairness_tpr_male = tpr_male
                self.fairness_tpr_female = tpr_female
                self.fairness_fpr_male = fpr_male
                self.fairness_fpr_female = fpr_female
                self.fairness_eo_tpr_diff = tpr_male - tpr_female
                self.fairness_eo_fpr_diff = fpr_male - fpr_female
                self.fairness_eo_diff = max(abs(tpr_male - tpr_female), abs(fpr_male - fpr_female))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3B – MARGIN-BASED RISK SCORE (for fairness post-processing)
# ─────────────────────────────────────────────────────────────────────────────

def compute_margin_risk_scores(records: List['PredictionRecord']) -> np.ndarray:
    """
    Compute continuous risk scores from CDS prediction records.
    """
    MIN_RANGE = 2.0
    CUTOFF = 0.15
    ALARM_BASE = 0.5

    scores = np.zeros(len(records))
    for i, r in enumerate(records):
        proximity_sum = 0.0
        n_features = 0
        max_violation = 0.0
        has_alarm = (r.alarm_class is not None)

        for t in r.af_trace:
            if t.is_nan:
                continue
            b_range = t.b_max - t.b_min

            if t.triggered_alarm and b_range > 1e-12:
                violation = abs(min(t.raw_value - t.b_min,
                                    t.b_max - t.raw_value)) / b_range
                max_violation = max(max_violation, violation)

            if b_range < MIN_RANGE:
                continue

            n_features += 1
            margin_lo = t.raw_value - t.b_min
            margin_hi = t.b_max - t.raw_value
            margin = min(margin_lo, margin_hi) / b_range

            if margin >= 0 and margin < CUTOFF:
                proximity_sum += (CUTOFF - margin)

        if n_features == 0:
            if has_alarm:
                scores[i] = ALARM_BASE + max_violation
            else:
                scores[i] = 0.0
        elif has_alarm:
            scores[i] = ALARM_BASE + max_violation
        else:
            scores[i] = proximity_sum / n_features

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_raw_value(global_user_idx: int, feature_idx: int,
                   data: np.ndarray) -> float:
    """
    Read the raw feature value for a user/record from the tabular data matrix.
    """
    val = data[global_user_idx, feature_idx]
    return float(val)


def _is_outside_healthy_range(value: float, b_min: float, b_max: float) -> bool:
    """
    Healthy-range check used by Algorithm 4.
    """
    if np.isnan(value):
        return False
    if np.isnan(b_min) or np.isnan(b_max) or b_min > b_max:
        return False
    return (value < b_min) or (value > b_max)


def _compute_p_h_f(node: TreeNode, disease_h: int) -> float:
    """
    P(h, f^k_m) = count(users with class h in this node) / n_users_node
    """
    n_total = node.n_users
    if n_total == 0:
        return 0.0
    n_class_h = node.health_dist.get(disease_h, 0)
    return n_class_h / n_total


def _compute_p_h_gt1_f(node: TreeNode, healthy_class: int) -> float:
    """
    P(h>1, f^k_m) generalized to:
      P(h != healthy_class, f^k_m)
    """
    n_total = node.n_users
    if n_total == 0:
        return 1e-9
    n_diseased = sum(cnt for cls, cnt in node.health_dist.items() if int(cls) != healthy_class)
    if n_diseased == 0:
        return 1e-9
    return n_diseased / n_total


def _compute_AF_increment(p_h_f: float, r_j_h: float, p_h_gt1_f: float) -> float:
    """
    AF increment from Eq. 7.
    """
    if p_h_gt1_f < 1e-12:
        return 0.0
    delta = (p_h_f * r_j_h) / p_h_gt1_f
    return float(max(0.0, delta * AF_DISEASE_SCALE))


def _update_AF(AF_real: float, delta_AF: float) -> float:
    """
    Apply Eq. 7 with fuzzy-logic cap at 1.0.
    """
    return float(min(1.0, max(0.0, AF_real + delta_AF)))


def _find_applicable_node(
    user_global_idx: int,
    focus_level: int,
    tree: DecisionTree,
    data: np.ndarray,
    valid_node_ids: Optional[set] = None,
) -> Optional[TreeNode]:
    """
    Find the tree node at `focus_level` that the test user/record belongs to.
    """
    if focus_level == 1:
        return tree.root

    nodes_at_level = tree.nodes_by_level.get(focus_level, [])
    for node in nodes_at_level:
        if valid_node_ids is not None and node.node_id not in valid_node_ids:
            continue
        if node.branch_def is not None:
            feat_idx = node.branch_def.feature_idx
            if not _is_valid_feature_idx(feat_idx, data):
                continue
            user_val = data[user_global_idx, feat_idx]
            if node.branch_def.contains(user_val):
                log.debug(
                    f"  User {user_global_idx}: m={focus_level} -> node {node.node_id!r} "
                    f"(feat={feat_idx} val={user_val} {node.branch_def.label})"
                )
                return node

    log.debug(f"  User {user_global_idx}: not matched at m={focus_level}; "
              f"signalling caller to halt focus escalation")
    return None


def _get_sorted_disease_actions(
    node_id: str,
    disease_h: int,
    alg3_output: Algorithm3Output,
    consumed: Optional[Set[Tuple[int, int]]] = None,
) -> List[ExecutiveActionEntry]:
    """
    Return refined actions for (node_id, disease_h) sorted by weight DESC.
    """
    acts = alg3_output.retained_for_node_disease(node_id, disease_h)
    out = [a for a in acts if a.action_weight > 0.0]
    if consumed:
        out = [a for a in out if (a.feature_idx, disease_h) not in consumed]
    return out


def _rl_select_best_action(
    candidate_actions: List[ExecutiveActionEntry],
    node: TreeNode,
    disease_h: int,
    AF_real_current: float,
    alg2_output: Algorithm2Output,
    user_sex: Optional[float] = None,
    data: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    train_data: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    schema: Optional[DatasetSchema] = None,
) -> Tuple[Optional[ExecutiveActionEntry], List[RLLookaheadEntry]]:
    """
    RL lookahead: select action J that minimises rw_sim.
    """
    if not candidate_actions:
        return None, []

    healthy_cls = _healthy_class(schema)
    p_h_f = _compute_p_h_f(node, disease_h)
    p_h_gt1_f = _compute_p_h_gt1_f(node, healthy_class=healthy_cls)

    fairness_penalties: Dict[int, float] = {}
    if _fairness_cfg.ENABLE_FAIRNESS_RL and data is not None and labels is not None:
        fair_data = train_data if train_data is not None else data
        fair_labels = train_labels if train_labels is not None else labels
        sidx = _sex_feature_index(schema)

        if sidx is not None and 0 <= sidx < fair_data.shape[1]:
            mcode = _male_code(schema)
            fcode = _female_code(schema)
            if mcode is not None and fcode is not None:
                node_users = node.user_indices
                node_sex = fair_data[node_users, sidx]
                node_labels = fair_labels[node_users]
                male_mask = (node_sex == mcode)
                female_mask = (node_sex == fcode)

                n_male_node = int(male_mask.sum())
                n_female_node = int(female_mask.sum())

                male_labels = node_labels[male_mask]
                female_labels = node_labels[female_mask]
                p_h_f_male = (male_labels == disease_h).sum() / n_male_node if n_male_node > 0 else 0.0
                p_h_f_female = (female_labels == disease_h).sum() / n_female_node if n_female_node > 0 else 0.0
                p_h_gt1_f_male = ((male_labels != healthy_cls).sum() / n_male_node) if n_male_node > 0 else 0.0
                p_h_gt1_f_female = ((female_labels != healthy_cls).sum() / n_female_node) if n_female_node > 0 else 0.0

                for action in candidate_actions:
                    j = action.feature_idx
                    vals = fair_data[node_users, j]
                    valid_mask = ~np.isnan(vals)

                    model = alg2_output.get_model(node.node_id, j)
                    if model is None:
                        fairness_penalties[j] = 0.0
                        continue
                    b_min_h = model.healthy_range.b_min_healthy
                    b_max_h = model.healthy_range.b_max_healthy

                    disease_mask = (node_labels == disease_h)
                    male_disease = disease_mask & male_mask & valid_mask
                    female_disease = disease_mask & female_mask & valid_mask

                    n_male_d = int(male_disease.sum())
                    n_female_d = int(female_disease.sum())

                    if n_male_d > 0:
                        male_vals = vals[male_disease]
                        r_male = float(((male_vals < b_min_h) | (male_vals > b_max_h)).sum()) / n_male_d
                    else:
                        r_male = 0.0

                    if n_female_d > 0:
                        female_vals = vals[female_disease]
                        r_female = float(((female_vals < b_min_h) | (female_vals > b_max_h)).sum()) / n_female_d
                    else:
                        r_female = 0.0

                    AF_male = p_h_f_male * r_male / p_h_gt1_f_male if p_h_gt1_f_male > 0 else 0.0
                    AF_female = p_h_f_female * r_female / p_h_gt1_f_female if p_h_gt1_f_female > 0 else 0.0
                    fairness_penalties[j] = abs(AF_male - AF_female)

    rl_entries: List[RLLookaheadEntry] = []
    best_action: Optional[ExecutiveActionEntry] = None
    best_rw_sim: float = float("inf")

    for action in candidate_actions:
        j = action.feature_idx
        r_j_h = action.action_weight

        AF_sim_cj = _compute_AF_increment(p_h_f, r_j_h, p_h_gt1_f)
        rw_sim_cj = 1.0 - (AF_sim_cj + AF_real_current)

        if _fairness_cfg.ENABLE_FAIRNESS_RL and j in fairness_penalties:
            rw_sim_cj += _fairness_cfg.FAIRNESS_LAMBDA * fairness_penalties[j]

        is_sel = False
        if rw_sim_cj < best_rw_sim:
            best_rw_sim = rw_sim_cj
            best_action = action
            is_sel = True
            for e in rl_entries:
                e.is_selected = False

        rl_entries.append(RLLookaheadEntry(
            action_feature_idx=j,
            action_feature_name=action.feature_name,
            action_weight=r_j_h,
            AF_sim_increment=AF_sim_cj,
            rw_sim=rw_sim_cj,
            is_selected=is_sel,
        ))

    log.debug(
        f"  RL selected: feat={best_action.feature_idx if best_action else None}  "
        f"rw_sim={best_rw_sim:.4f}  AF_real={AF_real_current:.4f}"
    )
    return best_action, rl_entries


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – CORE PREDICTION ENGINE (SINGLE USER)
# ─────────────────────────────────────────────────────────────────────────────

def _predict_at_node(
    user_global_idx:  int,
    node:             TreeNode,
    disease_classes:  List[int],
    data:             np.ndarray,
    labels:           np.ndarray,
    alg2_output:      Algorithm2Output,
    alg3_output:      Algorithm3Output,
    AF_real_init:     float,
    pac_counter:      List[int],
    record:           PredictionRecord,
    consumed_actions: Optional[Set[Tuple[int, int]]] = None,
    initial_action_h: Optional[int] = None,
    train_data: Optional[np.ndarray] = None,
    train_labels: Optional[np.ndarray] = None,
    schema: Optional[DatasetSchema] = None,
) -> Tuple[HealthDecision, float, Optional[int]]:
    """
    Run the inner prediction loop of Algorithm 4 at one tree node.
    """
    nid = node.node_id
    m = node.focus_level
    healthy_cls = _healthy_class(schema)

    AF_real: float = AF_real_init
    alarm_class: Optional[int] = None
    node_rec = NodePredictionRecord(node_id=nid, focus_level=m)

    log.debug(f"\n  {'='*55}")
    log.debug(f"  Node {nid!r}  m={m}  n_users={node.n_users}")
    log.debug(f"  disease_classes={disease_classes}  AF_init={AF_real_init:.4f}")

    p_h_gt1_f_node = _compute_p_h_gt1_f(node, healthy_class=healthy_cls)

    for h in disease_classes:
        p_h_f = _compute_p_h_f(node, h)
        p_h_gt1_f = p_h_gt1_f_node

        sorted_actions_h = _get_sorted_disease_actions(
            nid, h, alg3_output,
            consumed=consumed_actions,
        )

        if (initial_action_h is not None
                and h == initial_action_h
                and record.initial_action_feat is not None):
            sorted_actions_h = [
                a for a in sorted_actions_h
                if a.feature_idx != record.initial_action_feat
            ]

        if not sorted_actions_h:
            log.debug(f"    h={h}: no refined actions at node {nid!r} -> skip")
            dc_rec = DiseaseCheckRecord(
                node_id=nid, focus_level=m, disease_h=h,
                n_actions_in_buf=0, n_actions_applied=0,
                alarm_triggered=False, alarm_feature_idx=None,
                alarm_raw_value=None, alarm_b_min=None, alarm_b_max=None,
                AF_real_at_end=AF_real, rw_real_at_end=1.0 - AF_real,
            )
            node_rec.disease_checks.append(dc_rec)
            continue

        O_mkf: List[int] = [a.feature_idx for a in sorted_actions_h]
        C_buf: List[ExecutiveActionEntry] = list(sorted_actions_h)
        n_actions_total = len(C_buf)

        disease_check_rec = DiseaseCheckRecord(
            node_id=nid, focus_level=m, disease_h=h,
            n_actions_in_buf=n_actions_total, n_actions_applied=0,
            alarm_triggered=False, alarm_feature_idx=None,
            alarm_raw_value=None, alarm_b_min=None, alarm_b_max=None,
            AF_real_at_end=AF_real, rw_real_at_end=1.0 - AF_real,
        )

        while C_buf:
            A_tmkf = list(C_buf)

            sidx = _sex_feature_index(schema)
            user_sex_val: Optional[float] = None
            if sidx is not None and 0 <= sidx < data.shape[1]:
                user_sex_val = float(data[user_global_idx, sidx])

            selected_action, rl_entries = _rl_select_best_action(
                candidate_actions=A_tmkf,
                node=node,
                disease_h=h,
                AF_real_current=AF_real,
                alg2_output=alg2_output,
                user_sex=user_sex_val,
                data=data,
                labels=labels,
                train_data=train_data,
                train_labels=train_labels,
                schema=schema,
            )
            disease_check_rec.rl_selections.extend(rl_entries)

            if selected_action is None:
                break

            J = selected_action
            C_buf = [a for a in C_buf if a.feature_idx != J.feature_idx]

            O_buf: List[int] = [J.feature_idx]
            VO_buf: Dict[int, float] = {}
            for j in O_buf:
                if not _is_valid_feature_idx(j, data):
                    continue
                VO_buf[j] = _get_raw_value(user_global_idx, j, data)

            features_to_test = [j for j in O_buf if j in O_mkf and _is_valid_feature_idx(j, data)]

            for j in features_to_test:
                V_j = VO_buf[j]

                if np.isnan(V_j):
                    log.debug(f"    [SKIP NaN] j={j} for h={h}: sensor failure, no PAC update")
                    continue

                pac_counter[0] += 1
                t_mkf = pac_counter[0]

                disease_check_rec.n_actions_applied += 1

                if consumed_actions is not None:
                    consumed_actions.add((j, h))

                model_entry = alg2_output.get_model(nid, j)
                if model_entry is None:
                    log.debug(f"    h={h} j={j}: no model entry at node {nid!r} -> skip")
                    continue

                b_min = model_entry.healthy_range.b_min_healthy
                b_max = model_entry.healthy_range.b_max_healthy

                exec_entry = alg2_output.get_action(nid, j, h)
                r_j_h = exec_entry.action_weight if exec_entry else J.action_weight

                delta_AF = _compute_AF_increment(p_h_f, r_j_h, p_h_gt1_f)
                AF_real = _update_AF(AF_real, delta_AF)
                rw_real = 1.0 - AF_real

                alarm = _is_outside_healthy_range(V_j, b_min, b_max)

                trace_entry = AFTraceEntry(
                    pac_number=t_mkf,
                    node_id=nid,
                    focus_level=m,
                    disease_h=h,
                    feature_idx=j,
                    feature_name=_feature_name(j, schema),
                    raw_value=V_j,
                    b_min=b_min,
                    b_max=b_max,
                    r_j_h=r_j_h,
                    p_h_f=p_h_f,
                    p_h_gt1_f=p_h_gt1_f,
                    delta_AF=delta_AF,
                    AF_real=AF_real,
                    rw_real=rw_real,
                    triggered_alarm=alarm,
                    is_nan=False,
                )
                record.af_trace.append(trace_entry)
                record.total_actions_applied += 1

                log.debug(
                    f"    PAC={t_mkf:3d}  h={h:2d}  j={j:3d}({_feature_name(j, schema)[:15]:15s})"
                    f"  V={V_j:8.3f}  range=[{b_min:.3f},{b_max:.3f}]"
                    f"  r={r_j_h:.4f}  dAF={delta_AF:.4f}"
                    f"  AF={AF_real:.4f}  rw={rw_real:.4f}"
                    + ("  *** ALARM ***" if alarm else "")
                )

                if alarm:
                    disease_check_rec.alarm_triggered = True
                    disease_check_rec.alarm_feature_idx = j
                    disease_check_rec.alarm_raw_value = V_j
                    disease_check_rec.alarm_b_min = b_min
                    disease_check_rec.alarm_b_max = b_max
                    disease_check_rec.AF_real_at_end = AF_real
                    disease_check_rec.rw_real_at_end = rw_real
                    node_rec.disease_checks.append(disease_check_rec)
                    node_rec.final_AF_real = AF_real
                    node_rec.final_rw_real = rw_real
                    record.node_records.append(node_rec)
                    alarm_class = h
                    log.info(
                        f"  [ALARM] user={user_global_idx}  h={h}  "
                        f"feat={j}({_feature_name(j, schema)})  "
                        f"V={V_j:.3f} outside [{b_min:.3f},{b_max:.3f}]"
                    )
                    return HealthDecision.UNHEALTHY, AF_real, alarm_class

            disease_check_rec.AF_real_at_end = AF_real
            disease_check_rec.rw_real_at_end = 1.0 - AF_real
            node_rec.disease_checks.append(disease_check_rec)

    if AF_NORMALISE_TO_CAPACITY and record.total_actions_applied > 0:
        max_af_capacity = 0.0
        for h in disease_classes:
            p_h_f_cap = _compute_p_h_f(node, h)
            p_h_gt1_f_cap = _compute_p_h_gt1_f(node, healthy_class=healthy_cls)
            all_acts = _get_sorted_disease_actions(nid, h, alg3_output, consumed=None)
            for a in all_acts:
                max_af_capacity += _compute_AF_increment(p_h_f_cap, a.action_weight, p_h_gt1_f_cap)
        if max_af_capacity > 1e-9:
            AF_real = min(1.0, AF_real / max_af_capacity)

    rw_final = 1.0 - AF_real
    node_rec.final_AF_real = AF_real
    node_rec.final_rw_real = rw_final
    record.node_records.append(node_rec)

    if rw_final <= DIAGNOSTIC_THRESHOLD_ALG4:
        node_rec.met_threshold = True
        log.debug(f"  Node {nid!r}: rw={rw_final:.4f} <= Threshold="
                  f"{DIAGNOSTIC_THRESHOLD_ALG4} -> HEALTHY")
        return HealthDecision.HEALTHY, AF_real, None

    can_increase_focus = False
    next_m = m + 1
    next_nodes = [
        child for child in node.all_children
        if child.focus_level == next_m and child.n_users >= U_MIN_ALG4
    ]
    if next_nodes:
        can_increase_focus = True

    if can_increase_focus:
        node_rec.focus_increased = True
        log.debug(f"  Node {nid!r}: rw={rw_final:.4f} > threshold, "
                  f"focus can increase to m={next_m}")
        return HealthDecision.UNKNOWN, AF_real, None

    node_rec.sent_to_screening = True
    log.debug(f"  Node {nid!r}: rw={rw_final:.4f} > threshold, "
              f"focus cannot increase -> SCREENING")
    return HealthDecision.SCREENING, AF_real, None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – MAIN PREDICTION FUNCTION (ALGORITHM 4)
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm4(
    user_global_idx:  int,
    data:             np.ndarray,
    labels:           np.ndarray,
    tree:             DecisionTree,
    alg2_output:      Algorithm2Output,
    alg3_output:      Algorithm3Output,
    rng_seed:         Optional[int] = None,
    verbose:          bool = False,
    train_data:       Optional[np.ndarray] = None,
    train_labels:     Optional[np.ndarray] = None,
) -> PredictionRecord:
    """
    Run Algorithm 4 for a single test user/record.

    This is a faithful line-by-line implementation of the Algorithm 4
    pseudocode from the paper, generalized to schema-aware tabular datasets
    including WFDB-derived feature matrices.
    """
    if verbose:
        log.setLevel(logging.DEBUG)
        for h in log.handlers:
            h.setLevel(logging.DEBUG)

    _wfdb_or_tabular_guard(data, labels)
    schema = _schema_from_tree(tree)

    if rng_seed is not None:
        random.seed(rng_seed)
        np.random.seed(rng_seed)

    t0 = time.perf_counter()
    true_label = int(labels[user_global_idx])

    record = PredictionRecord(
        user_global_idx=user_global_idx,
        true_label=true_label,
        dataset_name=schema.dataset_name if schema is not None else "LegacyDataset",
        source_format=schema.source_format if schema is not None else "unknown",
    )

    log.debug(f"\n{'='*65}")
    log.debug(f"  Algorithm 4 | user_idx={user_global_idx}  true_label={true_label}")

    # Schema-aware property semantics while preserving the original class
    PredictionRecord.true_is_healthy = property(lambda self: self.true_label == _healthy_class(schema))
    PredictionRecord.true_is_diseased = property(lambda self: self.true_label != _healthy_class(schema))

    # ── Initialization: select initial action randomly from root actions ──
    root_node = tree.root
    AF_real: float = 0.0
    pac_counter = [0]

    consumed_pairs: Set[Tuple[int, int]] = set()

    root_all_actions = alg3_output.retained_for_node(root_node.node_id)
    h_init = -1
    if root_all_actions:
        valid_candidates = [
            a for a in root_all_actions
            if _is_valid_feature_idx(a.feature_idx, data)
            and not np.isnan(_get_raw_value(user_global_idx, a.feature_idx, data))
        ]

        chosen = False
        j_init = -1
        h_init = -1
        initial_action = None

        if valid_candidates:
            initial_action = random.choice(valid_candidates)
            j_init  = initial_action.feature_idx
            h_init  = initial_action.disease_class
            V_init  = _get_raw_value(user_global_idx, j_init, data)
            chosen  = True

        if not chosen:
            record.initial_action_feat = None
            AF_real = 0.0
            log.debug("  All initial-action candidates returned NaN — skipping init")
        else:
            record.initial_action_feat = j_init

            log.debug(f"  Initial action: feat={j_init}"
                      f"({_feature_name(j_init, schema)})  "
                      f"h={h_init}  w={initial_action.action_weight:.4f}")

            model_entry = alg2_output.get_model(root_node.node_id, j_init)
            if model_entry is not None:
                b_min = model_entry.healthy_range.b_min_healthy
                b_max = model_entry.healthy_range.b_max_healthy

                p_h_f = _compute_p_h_f(root_node, h_init)
                p_h_gt1_f = _compute_p_h_gt1_f(root_node, healthy_class=_healthy_class(schema))
                r_j_h = initial_action.action_weight

                delta_AF_init = _compute_AF_increment(p_h_f, r_j_h, p_h_gt1_f)
                AF_real = _update_AF(AF_real, delta_AF_init)

                pac_counter[0] += 1
                record.total_pac_count += 1
                record.total_actions_applied += 1

                alarm_init = _is_outside_healthy_range(V_init, b_min, b_max)

                trace_entry = AFTraceEntry(
                    pac_number=pac_counter[0],
                    node_id=root_node.node_id,
                    focus_level=1,
                    disease_h=h_init,
                    feature_idx=j_init,
                    feature_name=_feature_name(j_init, schema),
                    raw_value=V_init,
                    b_min=b_min,
                    b_max=b_max,
                    r_j_h=r_j_h,
                    p_h_f=p_h_f,
                    p_h_gt1_f=p_h_gt1_f,
                    delta_AF=delta_AF_init,
                    AF_real=AF_real,
                    rw_real=1.0 - AF_real,
                    triggered_alarm=alarm_init,
                    is_nan=False,
                )
                record.af_trace.append(trace_entry)
                consumed_pairs.add((j_init, h_init))

                log.debug(
                    f"    INIT: PAC={pac_counter[0]:3d}  j={j_init:3d}"
                    f"({_feature_name(j_init, schema)[:15]:15s})  "
                    f"V={V_init:8.3f}  range=[{b_min:.3f},{b_max:.3f}]  "
                    f"r={r_j_h:.4f}  dAF={delta_AF_init:.4f}  "
                    f"AF={AF_real:.4f}  rw={1.0-AF_real:.4f}"
                    + ("  *** INIT ALARM ***" if alarm_init else "")
                )

                if alarm_init:
                    record.decision = HealthDecision.UNHEALTHY
                    record.is_correct = (true_label != _healthy_class(schema))
                    record.alarm_class = h_init
                    record.alarm_feature_idx = j_init
                    record.max_focus_reached = 1
                    record.elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    log.info(
                        f"  [INIT ALARM] user={user_global_idx}  "
                        f"feat={j_init}({_feature_name(j_init, schema)})  "
                        f"h={h_init}  V={V_init:.3f} outside "
                        f"[{b_min:.3f},{b_max:.3f}]  -> UNHEALTHY"
                    )
                    return record
    else:
        record.initial_action_feat = None
        log.debug("  No root actions available for initial selection")

    all_disease_classes: List[int] = _all_disease_classes_from_labels(labels, schema)
    decision = HealthDecision.UNKNOWN

    max_focus_level = tree.depth()

    valid_node_ids = {e.node_id for e in alg2_output.perceptor_library}

    for current_focus in range(1, max_focus_level + 1):
        active_node = _find_applicable_node(
            user_global_idx=user_global_idx,
            focus_level=current_focus,
            tree=tree,
            data=data,
            valid_node_ids=valid_node_ids,
        )

        if active_node is None:
            log.debug(f"  m={current_focus}: no applicable node -> stop")
            decision = HealthDecision.SCREENING
            break

        record.max_focus_reached = current_focus

        log.info(
            f"  Focus m={current_focus}  node={active_node.node_id!r}  "
            f"n_users={active_node.n_users}  AF_init={AF_real:.4f}"
        )

        node_disease_classes = sorted([
            h for h in all_disease_classes
            if active_node.health_dist.get(h, 0) > 0
        ])
        if not node_disease_classes:
            log.debug(f"  Node {active_node.node_id!r}: no disease classes -> skip")
            decision = HealthDecision.HEALTHY
            break

        pac_counter = [0]

        decision, AF_real, alarm_class = _predict_at_node(
            user_global_idx=user_global_idx,
            node=active_node,
            disease_classes=node_disease_classes,
            data=data,
            labels=labels,
            alg2_output=alg2_output,
            alg3_output=alg3_output,
            AF_real_init=AF_real,
            pac_counter=pac_counter,
            record=record,
            consumed_actions=consumed_pairs,
            initial_action_h=h_init,
            train_data=train_data,
            train_labels=train_labels,
            schema=schema,
        )
        record.total_pac_count += pac_counter[0]

        if decision == HealthDecision.UNHEALTHY:
            record.alarm_class = alarm_class
            for nr in reversed(record.node_records):
                for dc in reversed(nr.disease_checks):
                    if dc.alarm_triggered and dc.alarm_feature_idx is not None:
                        record.alarm_feature_idx = dc.alarm_feature_idx
                        break
                if record.alarm_feature_idx is not None:
                    break
            break

        if decision == HealthDecision.HEALTHY:
            break

        if decision == HealthDecision.SCREENING:
            break

        log.info(f"  m={current_focus}: focus increasing to m={current_focus + 1}")
        continue

    if decision == HealthDecision.UNKNOWN:
        decision = HealthDecision.SCREENING
        log.debug("  Decision remained UNKNOWN after all focus levels -> SCREENING")

    if true_label == _healthy_class(schema):
        is_correct = (decision != HealthDecision.UNHEALTHY)
    else:
        is_correct = (decision == HealthDecision.UNHEALTHY)

    record.decision = decision
    record.is_correct = is_correct
    record.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    log.info(
        f"  DECISION: user={user_global_idx}  true={true_label}  "
        f"-> {decision.value}  correct={is_correct}  "
        f"PAC={record.total_pac_count}  AF={AF_real:.4f}  "
        f"elapsed={record.elapsed_ms:.1f}ms"
    )

    return record


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – LOOCV PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_loocv(
    data:         np.ndarray,
    labels:       np.ndarray,
    max_users:    Optional[int] = None,
    rng_seed:     int = 42,
    verbose:      bool = False,
    n_bins:       int = DEFAULT_N_BINS,
    nodes_filter: Optional[List[str]] = None,
    schema:       Optional[DatasetSchema] = None,
) -> Algorithm4Output:
    """
    Leave-One-Out Cross-Validation wrapper for Algorithm 4.
    Generalized for schema-aware tabular datasets including WFDB-derived
    matrices.
    """
    import fairness_config as _fc
    _wfdb_or_tabular_guard(data, labels)

    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    log.info(f"\n{'='*65}")
    log.info(f"LOOCV (per-fold retraining) | n_users={n_total}  "
             f"threshold={DIAGNOSTIC_THRESHOLD_ALG4}")
    log.info(f"Dataset: {_dataset_description(schema, data, labels)}")
    log.info(f"Config: {_fc.summary()}")
    log.info(f"{'='*65}")

    if _fc.ENABLE_FORCED_SEX_BRANCHING:
        from Algorithm1_forcedBranch import build_forced_sex_forest, build_sex_specific_tree, route_user

    import logging as _logging
    for name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3", "CDS.Alg1.ForcedSex"):
        _logging.getLogger(name).setLevel(_logging.WARNING)

    if _fc.ENABLE_DATA_AUGMENTATION:
        _ext_dir = str(Path(__file__).parent / "extensions")
        if _ext_dir not in sys.path:
            sys.path.insert(0, _ext_dir)
        from extensions.augmentation_strategies import apply_augmentation

    output = Algorithm4Output(data=data, schema=schema)
    random.seed(rng_seed)

    n_users_total = data.shape[0]

    aug_data = None
    aug_labels = None
    n_real = n_users_total
    if _fc.ENABLE_DATA_AUGMENTATION and _fc.AUGMENTATION_STRATEGY != "none":
        aug_data, aug_labels = apply_augmentation(
            strategy_name=_fc.AUGMENTATION_STRATEGY,
            X_train=data,
            y_train=labels,
            rng_seed=rng_seed,
        )
        n_synthetic = aug_data.shape[0] - n_real
        log.info(f"Data augmentation: generated {n_synthetic} synthetic users "
                 f"(total training pool: {aug_data.shape[0]})")

    treeCounter = 0
    for i in range(n_total):
        if aug_data is not None:
            train_mask = np.ones(aug_data.shape[0], dtype=bool)
            train_mask[i] = False
            train_data = aug_data[train_mask]
            train_labels = aug_labels[train_mask]
        else:
            train_mask = np.ones(n_users_total, dtype=bool)
            train_mask[i] = False
            train_data = data[train_mask]
            train_labels = labels[train_mask]

        if _fc.ENABLE_FORCED_SEX_BRANCHING:
            sidx = _sex_feature_index(schema)
            mcode = _male_code(schema)
            fcode = _female_code(schema)
            if sidx is None or mcode is None or fcode is None:
                tree_i = build_decision_tree(train_data, train_labels, schema=schema)
            else:
                test_user_sex = data[i, sidx]
                if test_user_sex == mcode:
                    sex_indices = np.where(train_data[:, sidx] == mcode)[0]
                    tree_i = build_sex_specific_tree(
                        data=train_data, labels=train_labels,
                        sex_user_indices=sex_indices, sex_label="male",
                    )
                else:
                    sex_indices = np.where(train_data[:, sidx] == fcode)[0]
                    tree_i = build_sex_specific_tree(
                        data=train_data, labels=train_labels,
                        sex_user_indices=sex_indices, sex_label="female",
                    )
        else:
            tree_i = build_decision_tree(train_data, train_labels, schema=schema)

        root_id = tree_i.root.node_id
        nodes_filter_i = [root_id]

        if _fc.ENABLE_FORCED_SEX_BRANCHING:
            pass
        else:
            sex_k = _sex_feature_index(schema)
            if sex_k is not None:
                sex_children = [
                    n for n in tree_i.nodes_by_level.get(2, [])
                    if n.branching_feat_k == sex_k and not n.is_leaf
                ]
                if len(sex_children) >= 2:
                    for child in sex_children:
                        nodes_filter_i.append(child.node_id)

        treeCounter += 1
        print(f"Tree built: {treeCounter}")

        alg2_i = run_algorithm2(
            tree=tree_i,
            data=train_data,
            labels=train_labels,
            n_bins=n_bins,
            nodes_filter=nodes_filter_i,
        )

        alg3_i = run_algorithm3(
            alg2_output=alg2_i,
            tree=tree_i,
            data=train_data,
            labels=train_labels,
            nodes_filter=nodes_filter_i,
            reset_per_h=False,
            verbose=False,
        )

        pred = run_algorithm4(
            user_global_idx=i,
            data=data,
            labels=labels,
            tree=tree_i,
            alg2_output=alg2_i,
            alg3_output=alg3_i,
            rng_seed=rng_seed,
            verbose=verbose,
            train_data=train_data,
            train_labels=train_labels,
        )
        output.records.append(pred)
        output.total_elapsed_ms += pred.elapsed_ms

        if (i + 1) % 50 == 0 or i == n_total - 1:
            n_done = i + 1
            n_correct = sum(1 for r in output.records if r.is_correct)
            log.info(
                f"  Progress: {n_done}/{n_total}  "
                f"running_acc={n_correct/n_done*100:.1f}%  "
                f"total_ms={output.total_elapsed_ms:.0f}"
            )

    output._recompute_stats()

    return output

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_AF_monotonicity(record: PredictionRecord) -> List[str]:
    """
    Verify that AF_real is monotonically non-decreasing across the entire
    prediction trajectory.
    """
    issues: List[str] = []
    prev_AF = 0.0
    for entry in record.af_trace:
        if entry.AF_real < prev_AF - 1e-9:
            issues.append(
                f"PAC={entry.pac_number} m={entry.focus_level}: AF decreased from "
                f"{prev_AF:.6f} to {entry.AF_real:.6f} "
                f"(feat={entry.feature_idx}, h={entry.disease_h})"
            )
        prev_AF = entry.AF_real
    return issues


def validate_rw_monotone_decrease(record: PredictionRecord) -> List[str]:
    """
    Verify that rw = 1 - AF is monotonically non-increasing.
    """
    issues: List[str] = []
    prev_rw = 1.0
    for entry in record.af_trace:
        if entry.rw_real > prev_rw + 1e-9:
            issues.append(
                f"PAC={entry.pac_number} m={entry.focus_level}: rw increased from "
                f"{prev_rw:.6f} to {entry.rw_real:.6f} "
                f"(feat={entry.feature_idx}, h={entry.disease_h})"
            )
        prev_rw = entry.rw_real
    return issues


def validate_unhealthy_immediate_return(record: PredictionRecord) -> List[str]:
    """
    Verify that once an alarm fires, the decision is UNHEALTHY and the
    trace ends immediately.
    """
    issues: List[str] = []
    if record.decision != HealthDecision.UNHEALTHY:
        return issues
    alarm_indices = [i for i, e in enumerate(record.af_trace) if e.triggered_alarm]
    if not alarm_indices:
        issues.append(
            f"Decision=UNHEALTHY but no alarm entry found in trace "
            f"(user={record.user_global_idx})"
        )
        return issues
    alarm_idx = alarm_indices[-1]
    post_alarm = record.af_trace[alarm_idx + 1:]
    if post_alarm:
        first = post_alarm[0]
        issues.append(
            f"user={record.user_global_idx}: {len(post_alarm)} PACs after alarm "
            f"at trace_idx={alarm_idx} (first post: feat={first.feature_idx})"
        )
    return issues


def validate_no_healthy_class_actions(record: PredictionRecord,
                                      schema: Optional[DatasetSchema] = None) -> List[str]:
    """
    Verify that no actions were applied for the healthy class.
    """
    issues: List[str] = []
    healthy_cls = _healthy_class(schema)
    for entry in record.af_trace:
        if entry.disease_h == healthy_cls:
            issues.append(
                f"user={record.user_global_idx}: action applied for healthy class "
                f"h={healthy_cls} (feat={entry.feature_idx}, PAC={entry.pac_number})"
            )
    return issues


def validate_healthy_rw_threshold(record: PredictionRecord) -> List[str]:
    """
    Verify that HEALTHY decisions have rw ≤ threshold at the end.
    """
    issues: List[str] = []
    if record.decision != HealthDecision.HEALTHY:
        return issues
    if not record.af_trace:
        return issues
    final_rw = record.af_trace[-1].rw_real
    if final_rw > DIAGNOSTIC_THRESHOLD_ALG4 + 1e-9:
        issues.append(
            f"user={record.user_global_idx}: HEALTHY decision but final "
            f"rw={final_rw:.6f} > Threshold={DIAGNOSTIC_THRESHOLD_ALG4}"
        )
    return issues


def validate_AF_in_unit_interval(record: PredictionRecord) -> List[str]:
    """
    Verify that AF stays in [0, 1].
    """
    issues: List[str] = []
    for entry in record.af_trace:
        if entry.AF_real < -1e-9 or entry.AF_real > 1.0 + 1e-9:
            issues.append(
                f"user={record.user_global_idx}: AF={entry.AF_real:.6f} "
                f"out of [0,1] at PAC={entry.pac_number} "
                f"(feat={entry.feature_idx}, h={entry.disease_h})"
            )
    return issues


def validate_focus_level_progression(record: PredictionRecord) -> List[str]:
    """
    Verify that HEALTHY is only declared after a threshold-met state.
    """
    issues: List[str] = []
    if record.decision != HealthDecision.HEALTHY:
        return issues
    if not record.af_trace:
        issues.append(
            f"user={record.user_global_idx}: HEALTHY decision with EMPTY af_trace "
            f"(no PAC was ever executed — premature decision)"
        )
        return issues
    last = record.af_trace[-1]
    if last.rw_real > DIAGNOSTIC_THRESHOLD_ALG4 + 1e-9:
        issues.append(
            f"user={record.user_global_idx}: HEALTHY but final rw={last.rw_real:.6f} "
            f"exceeds threshold={DIAGNOSTIC_THRESHOLD_ALG4}"
        )
    return issues


def validate_specificity_invariant(output: Algorithm4Output) -> List[str]:
    """
    Verify that no truly-healthy user/record is classified UNHEALTHY.
    """
    issues: List[str] = []
    healthy_cls = _healthy_class(output.schema)
    for r in output.records:
        if r.true_label == healthy_cls and r.decision == HealthDecision.UNHEALTHY:
            feat = r.alarm_feature_idx
            issues.append(
                f"user={r.user_global_idx}: TRUE HEALTHY classified UNHEALTHY "
                f"(alarm feat={feat}, h={r.alarm_class})"
            )
    return issues


def validate_diseased_not_healthy(output: Algorithm4Output) -> List[str]:
    """
    Verify that no truly-diseased user/record is classified HEALTHY.
    """
    issues: List[str] = []
    healthy_cls = _healthy_class(output.schema)
    for r in output.records:
        if r.true_label != healthy_cls and r.decision == HealthDecision.HEALTHY:
            issues.append(
                f"user={r.user_global_idx}: TRUE DISEASED (class={r.true_label}) "
                f"classified HEALTHY (missed diagnosis)"
            )
    return issues


def validate_feature_indices_in_bounds_output(output: Algorithm4Output,
                                              n_features: int) -> List[str]:
    """
    Ensure recorded feature indices are compatible with current dataset
    dimensionality, including WFDB-tabularized datasets.
    """
    issues: List[str] = []
    for rec in output.records:
        if rec.initial_action_feat is not None and not (0 <= rec.initial_action_feat < n_features):
            issues.append(
                f"user={rec.user_global_idx}: initial_action_feat={rec.initial_action_feat} out of bounds"
            )
        for e in rec.af_trace:
            if not (0 <= e.feature_idx < n_features):
                issues.append(
                    f"user={rec.user_global_idx}: trace feature_idx={e.feature_idx} out of bounds"
                )
        if rec.alarm_feature_idx is not None and not (0 <= rec.alarm_feature_idx < n_features):
            issues.append(
                f"user={rec.user_global_idx}: alarm_feature_idx={rec.alarm_feature_idx} out of bounds"
            )
    return issues


def run_all_validations_single(record: PredictionRecord,
                               verbose: bool = True,
                               schema: Optional[DatasetSchema] = None) -> bool:
    """
    Run all single-record validations.
    """
    all_issues: List[str] = []

    checks = [
        ("AF monotonicity",          validate_AF_monotonicity(record)),
        ("AF in [0,1]",              validate_AF_in_unit_interval(record)),
        ("rw monotone decrease",     validate_rw_monotone_decrease(record)),
        ("unhealthy immediate return", validate_unhealthy_immediate_return(record)),
        ("no healthy-class actions", validate_no_healthy_class_actions(record, schema=schema)),
        ("healthy rw threshold",     validate_healthy_rw_threshold(record)),
        ("focus-level progression",  validate_focus_level_progression(record)),
    ]

    all_ok = True
    for name, issues in checks:
        if issues:
            all_ok = False
            all_issues.extend(issues)
            if verbose:
                for iss in issues:
                    print(f"  [FAIL] {name}: {iss}")
        else:
            if verbose:
                print(f"  [PASS] {name}")

    return all_ok


def run_all_validations_output(output: Algorithm4Output, verbose: bool = True) -> bool:
    """
    Run all validations across the full LOOCV output.
    """
    sep = "─" * 65
    if verbose:
        print(f"\n{sep}")
        print("ALGORITHM 4 VALIDATION SUITE")
        print(sep)

    all_ok = True
    total_issues = 0
    n_records = len(output.records)

    n_af_unit_violations = 0
    n_premature_healthy = 0
    for rec in output.records:
        issues_mono_af = validate_AF_monotonicity(rec)
        issues_af_unit = validate_AF_in_unit_interval(rec)
        issues_mono_rw = validate_rw_monotone_decrease(rec)
        issues_immediate = validate_unhealthy_immediate_return(rec)
        issues_no_h1 = validate_no_healthy_class_actions(rec, schema=output.schema)
        issues_threshold = validate_healthy_rw_threshold(rec)
        issues_focus = validate_focus_level_progression(rec)

        if issues_af_unit:
            n_af_unit_violations += len(issues_af_unit)
        if issues_focus:
            n_premature_healthy += len(issues_focus)

        all_rec_issues = (issues_mono_af + issues_af_unit + issues_mono_rw +
                          issues_immediate + issues_no_h1 + issues_threshold +
                          issues_focus)
        if all_rec_issues:
            all_ok = False
            total_issues += len(all_rec_issues)
            if verbose and len(all_rec_issues) <= 3:
                for iss in all_rec_issues:
                    print(f"  [FAIL] user={rec.user_global_idx}: {iss}")

    if verbose:
        if total_issues == 0:
            print(f"  [PASS] Per-record validations: all {n_records} records passed")
        else:
            print(f"  [FAIL] Per-record validations: {total_issues} issues across {n_records} records")
        if n_af_unit_violations == 0:
            print(f"  [PASS] AF in [0,1] invariant: all PACs across all users/records")
        else:
            print(f"  [FAIL] AF in [0,1] invariant: {n_af_unit_violations} PACs out of bounds")
        if n_premature_healthy == 0:
            print(f"  [PASS] Premature HEALTHY: no record declared HEALTHY before threshold met")
        else:
            print(f"  [FAIL] Premature HEALTHY: {n_premature_healthy} records flagged")

    spec_violations = validate_specificity_invariant(output)
    if verbose:
        if not spec_violations:
            print(f"  [PASS] Specificity = 100% (no true-healthy user/record classified UNHEALTHY)")
        else:
            print(f"  [WARN] Specificity < 100%: {len(spec_violations)} healthy users/records classified UNHEALTHY")

    fnr_violations = validate_diseased_not_healthy(output)
    if verbose:
        if not fnr_violations:
            print(f"  [PASS] Sensitivity = 100% (no true-diseased user/record classified HEALTHY)")
        else:
            fnr_pct = len(fnr_violations) / max(1, output.n_diseased_total) * 100
            print(f"  [INFO] Missed diagnoses: {len(fnr_violations)} true-diseased users/records "
                  f"classified HEALTHY (FNR={fnr_pct:.1f}%)")

    unknowns = [r for r in output.records if r.decision == HealthDecision.UNKNOWN]
    if unknowns:
        all_ok = False
        if verbose:
            print(f"  [FAIL] {len(unknowns)} UNKNOWN decisions (should not occur)")
    elif verbose:
        print(f"  [PASS] No UNKNOWN decisions")

    if output.data is not None:
        dim_issues = validate_feature_indices_in_bounds_output(output, output.data.shape[1])
        if dim_issues:
            all_ok = False
            if verbose:
                for iss in dim_issues[:10]:
                    print(f"  [FAIL] Dimensionality: {iss}")
                if len(dim_issues) > 10:
                    print(f"  ... and {len(dim_issues)-10} more dimensionality issues")
        elif verbose:
            print(f"  [PASS] Feature-index dimensionality invariant")

    if verbose:
        n_missed_diag = sum(
            1 for r in output.records
            if r.true_label != _healthy_class(output.schema) and r.decision != HealthDecision.UNHEALTHY
        )
        miss_rate = (n_missed_diag / output.n_diseased_total
                     if output.n_diseased_total else 0.0)
        print(f"\n  Accuracy Summary:")
        print(f"    Overall accuracy  : {output.overall_accuracy*100:.1f}%  "
              f"({sum(1 for r in output.records if r.is_correct)}/{n_records})")
        print(f"    Sensitivity       : {output.sensitivity*100:.1f}%")
        print(f"    Specificity       : {output.specificity*100:.1f}%")
        print(f"    False-alarm rate  : {output.false_alarm_rate*100:.1f}%")
        print(f"    Missed-diagnosis rate (FNR): {miss_rate*100:.1f}%")
        print(f"    Screening count   : {output.n_screening}")
        print(f"    Total time        : {output.total_elapsed_ms:.0f}ms  "
              f"({output.total_elapsed_ms/n_records:.1f}ms/user)")
        print(sep)

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – REPORTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_prediction_detail(record: PredictionRecord,
                            show_full_trace: bool = True,
                            schema: Optional[DatasetSchema] = None) -> None:
    """
    Print detailed information for one prediction record.
    """
    sep = "─" * 65
    healthy_cls = _healthy_class(schema)
    print(f"\n{sep}")
    print(f"PREDICTION DETAIL | user={record.user_global_idx}  "
          f"true_label={record.true_label}  "
          f"({'HEALTHY' if record.true_label == healthy_cls else 'DISEASED'})")
    print(f"  Dataset        : {record.dataset_name} ({record.source_format})")
    print(f"  Decision       : {record.decision.value}  "
          f"({'CORRECT' if record.is_correct else 'WRONG'})")
    print(f"  Focus reached  : m={record.max_focus_reached}")
    print(f"  PAC count      : {record.total_pac_count}")
    print(f"  Actions applied: {record.total_actions_applied}")
    print(f"  Elapsed        : {record.elapsed_ms:.1f}ms")
    if record.alarm_class is not None:
        print(f"  ALARM class    : h={record.alarm_class}  "
              f"feat={record.alarm_feature_idx}"
              f"({_feature_name(record.alarm_feature_idx, schema) if record.alarm_feature_idx is not None else '?'})")
    if record.initial_action_feat is not None:
        fname = _feature_name(record.initial_action_feat, schema)
        print(f"  Initial action : feat={record.initial_action_feat} ({fname})")

    print(f"\n  Node Traversal:")
    for nr in record.node_records:
        status = ""
        if nr.met_threshold:
            status = "[THRESHOLD MET]"
        elif nr.focus_increased:
            status = "[FOCUS UP]"
        elif nr.sent_to_screening:
            status = "[SCREENING]"
        print(f"    m={nr.focus_level}  node={nr.node_id!r}  "
              f"AF_final={nr.final_AF_real:.4f}  rw={nr.final_rw_real:.4f}  {status}")
        for dc in nr.disease_checks:
            alarm_str = ""
            if dc.alarm_triggered:
                alarm_str = (f" *** ALARM feat={dc.alarm_feature_idx} "
                             f"V={dc.alarm_raw_value:.3f} "
                             f"outside [{dc.alarm_b_min:.3f},{dc.alarm_b_max:.3f}]")
            print(f"      h={dc.disease_h:2d}: "
                  f"actions_in_buf={dc.n_actions_in_buf:3d}  "
                  f"applied={dc.n_actions_applied:3d}  "
                  f"AF={dc.AF_real_at_end:.4f}{alarm_str}")

    if show_full_trace and record.af_trace:
        print(f"\n  AF/rw Trace (first 20 of {len(record.af_trace)} PACs):")
        print(f"  {'PAC':4} {'node':20} {'h':3} {'feat':4} {'feature_name':18} "
              f"{'V':8} {'b_min':7} {'b_max':7} {'r':6} {'ΔAF':6} {'AF':6} {'rw':6}")
        print("  " + "─" * 95)
        for entry in record.af_trace[:20]:
            v_str = "NaN" if entry.is_nan else f"{entry.raw_value:8.3f}"
            alarm_flag = " ***" if entry.triggered_alarm else ""
            print(
                f"  {entry.pac_number:4d} {entry.node_id:20s} "
                f"{entry.disease_h:3d} {entry.feature_idx:4d} "
                f"{_feature_name(entry.feature_idx, schema)[:18]:18s} "
                f"{v_str:8s} {entry.b_min:7.3f} {entry.b_max:7.3f} "
                f"{entry.r_j_h:6.4f} {entry.delta_AF:6.4f} "
                f"{entry.AF_real:6.4f} {entry.rw_real:6.4f}"
                + alarm_flag
            )
        if len(record.af_trace) > 20:
            print(f"  … ({len(record.af_trace) - 20} more PACs) …")


def dump_wrong_user_data(output: Algorithm4Output,
                         filepath: str = "wrongUserData.txt") -> None:
    """
    Dump detailed traces for all misclassified users/records.
    """
    wrong_records = [r for r in output.records if not r.is_correct]
    schema = output.schema

    if not wrong_records:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("All predictions were correct. No misclassified users/records.\n")
        return

    healthy_cls = _healthy_class(schema)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("MISCLASSIFIED USER/RECORD DIAGNOSTIC TRACE\n")
        f.write(f"Total misclassified: {len(wrong_records)} / {len(output.records)}\n")
        f.write(f"Dataset: {output.schema.dataset_name if output.schema else 'unknown'} "
                f"({output.schema.source_format if output.schema else 'unknown'})\n")
        f.write(f"Overall accuracy: {output.overall_accuracy*100:.1f}%\n")
        f.write(f"Sensitivity: {output.sensitivity*100:.1f}%  Specificity: {output.specificity*100:.1f}%\n")
        f.write("=" * 120 + "\n\n")

        missed = [r for r in wrong_records if r.true_label != healthy_cls]
        false_alarms = [r for r in wrong_records if r.true_label == healthy_cls]
        f.write(f"Missed diagnoses (diseased -> HEALTHY/SCREENING): {len(missed)}\n")
        f.write(f"False alarms (healthy -> UNHEALTHY): {len(false_alarms)}\n\n")

        for idx, rec in enumerate(wrong_records):
            f.write("#" * 120 + "\n")
            f.write(f"WRONG USER/RECORD #{idx+1}: user_idx={rec.user_global_idx}\n")
            f.write("#" * 120 + "\n")
            true_type = "HEALTHY" if rec.true_label == healthy_cls else f"DISEASED (class {rec.true_label})"
            f.write(f"  Dataset        : {rec.dataset_name} ({rec.source_format})\n")
            f.write(f"  True label     : {rec.true_label} ({true_type})\n")
            f.write(f"  Prediction     : {rec.decision.value}\n")
            if rec.true_label != healthy_cls:
                f.write(f"  Error type     : MISSED DIAGNOSIS\n")
            else:
                f.write(f"  Error type     : FALSE ALARM\n")
            f.write(f"  Focus reached  : m={rec.max_focus_reached}\n")
            f.write(f"  Total PAC count: {rec.total_pac_count}\n")
            f.write(f"  Total actions  : {rec.total_actions_applied}\n")
            f.write(f"  Elapsed        : {rec.elapsed_ms:.1f}ms\n")
            if rec.initial_action_feat is not None:
                fname = _feature_name(rec.initial_action_feat, schema)
                f.write(f"  Initial action : feat={rec.initial_action_feat} ({fname})\n")
            if rec.alarm_class is not None:
                f.write(f"  Alarm class    : h={rec.alarm_class}\n")
                f.write(f"  Alarm feature  : {rec.alarm_feature_idx} "
                        f"({_feature_name(rec.alarm_feature_idx, schema) if rec.alarm_feature_idx is not None else '?'})\n")
            f.write("\n")

            f.write("  --- NODE TRAVERSAL ---\n")
            for nr in rec.node_records:
                status = ""
                if nr.met_threshold:
                    status = "[THRESHOLD MET -> HEALTHY]"
                elif nr.focus_increased:
                    status = "[FOCUS INCREASED]"
                elif nr.sent_to_screening:
                    status = "[SCREENING]"
                f.write(f"  Node: {nr.node_id!r}  m={nr.focus_level}  "
                        f"AF_final={nr.final_AF_real:.6f}  rw_final={nr.final_rw_real:.6f}  {status}\n")

                for dc in nr.disease_checks:
                    alarm_str = ""
                    if dc.alarm_triggered:
                        alarm_str = (f" *** ALARM feat={dc.alarm_feature_idx}"
                                     f" V={dc.alarm_raw_value:.6f}"
                                     f" outside [{dc.alarm_b_min:.6f},{dc.alarm_b_max:.6f}]")
                    f.write(f"    h={dc.disease_h:2d}: buf_size={dc.n_actions_in_buf:3d}  "
                            f"applied={dc.n_actions_applied:3d}  "
                            f"AF_end={dc.AF_real_at_end:.6f}  "
                            f"rw_end={dc.rw_real_at_end:.6f}{alarm_str}\n")
            f.write("\n")

            f.write("  --- FULL AF/rw TRACE (every action) ---\n")
            f.write(f"  {'PAC':>4} | {'node':<20} | {'m':>2} | {'h':>3} | {'feat':>4} | "
                    f"{'feature_name':<22} | {'raw_value':>10} | {'b_min':>10} | {'b_max':>10} | "
                    f"{'r_j_h':>8} | {'p_h_f':>10} | {'p_h_gt1_f':>10} | "
                    f"{'delta_AF':>10} | {'AF_cum':>10} | {'rw':>10} | {'alarm':>5} | {'calc_detail'}\n")
            f.write("  " + "-" * 200 + "\n")

            for entry in rec.af_trace:
                alarm_flag = "YES" if entry.triggered_alarm else "no"
                nan_flag = " [NaN]" if entry.is_nan else ""
                calc = f"dAF = {entry.p_h_f:.6f} * {entry.r_j_h:.6f} / {entry.p_h_gt1_f:.6f} = {entry.delta_AF:.6f}"
                v_str = "NaN" if entry.is_nan else f"{entry.raw_value:10.4f}"
                f.write(
                    f"  {entry.pac_number:4d} | {entry.node_id:<20} | {entry.focus_level:2d} | "
                    f"{entry.disease_h:3d} | {entry.feature_idx:4d} | "
                    f"{_feature_name(entry.feature_idx, schema)[:22]:<22} | "
                    f"{v_str:>10} | {entry.b_min:10.4f} | {entry.b_max:10.4f} | "
                    f"{entry.r_j_h:8.6f} | {entry.p_h_f:10.6f} | {entry.p_h_gt1_f:10.6f} | "
                    f"{entry.delta_AF:10.6f} | {entry.AF_real:10.6f} | {entry.rw_real:10.6f} | "
                    f"{alarm_flag:>5} | {calc}{nan_flag}\n"
                )

            f.write("\n")

            if rec.af_trace:
                unique_features = set(e.feature_idx for e in rec.af_trace if not e.is_nan)
                unique_h = set(e.disease_h for e in rec.af_trace)
                total_delta = sum(e.delta_AF for e in rec.af_trace)
                f.write(f"  --- SUMMARY FOR USER/RECORD {rec.user_global_idx} ---\n")
                f.write(f"  Unique features tested: {len(unique_features)} {sorted(unique_features)}\n")
                f.write(f"  Disease classes processed: {sorted(unique_h)}\n")
                f.write(f"  Total AF accumulated: {total_delta:.6f}\n")
                f.write(f"  Final AF: {rec.af_trace[-1].AF_real:.6f}  Final rw: {rec.af_trace[-1].rw_real:.6f}\n")
                f.write(f"  Threshold: {DIAGNOSTIC_THRESHOLD_ALG4}  (rw <= threshold -> HEALTHY)\n")
                if rec.true_label != healthy_cls and rec.decision != HealthDecision.UNHEALTHY:
                    f.write(f"  WHY MISSED: rw={rec.af_trace[-1].rw_real:.6f}, "
                            f"needed alarm (out-of-range value) but none found across "
                            f"{len(rec.af_trace)} actions on {len(unique_features)} unique features\n")
                    f.write(f"  RANGE CHECK DETAIL (all features tested):\n")
                    for entry in rec.af_trace:
                        if entry.is_nan:
                            continue
                        in_range = "IN RANGE" if not entry.triggered_alarm else "OUT OF RANGE"
                        margin_low = entry.raw_value - entry.b_min
                        margin_high = entry.b_max - entry.raw_value
                        f.write(f"    feat={entry.feature_idx:4d} ({_feature_name(entry.feature_idx, schema)[:20]:<20}): "
                                f"V={entry.raw_value:10.4f}  range=[{entry.b_min:10.4f}, {entry.b_max:10.4f}]  "
                                f"{in_range}  margin_low={margin_low:+10.4f}  margin_high={margin_high:+10.4f}\n")
            f.write("\n" + "=" * 120 + "\n\n")

    log.info(f"Wrote detailed diagnostic trace for {len(wrong_records)} misclassified users/records to {filepath}")


def print_algorithm4_summary(output: Algorithm4Output) -> None:
    """
    Print the overall performance summary.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("ALGORITHM 4 PERFORMANCE SUMMARY")
    print(sep)
    n = len(output.records)
    schema = output.schema
    print(f"  Dataset           : {schema.dataset_name if schema else 'unknown'} "
          f"({schema.source_format if schema else 'unknown'})")
    print(f"  Users/records eval: {n}")
    print(f"  Overall accuracy  : {output.overall_accuracy*100:.1f}%")
    print(f"  Sensitivity       : {output.sensitivity*100:.1f}%")
    print(f"  Specificity       : {output.specificity*100:.1f}%")
    print(f"  False-alarm rate  : {output.false_alarm_rate*100:.1f}%")
    print(f"  HEALTHY decisions : {sum(1 for r in output.records if r.decision==HealthDecision.HEALTHY)}")
    print(f"  UNHEALTHY decisions: {sum(1 for r in output.records if r.decision==HealthDecision.UNHEALTHY)}")
    print(f"  SCREENING decisions: {output.n_screening}")
    print(f"  Total elapsed     : {output.total_elapsed_ms:.0f}ms  "
          f"({output.total_elapsed_ms/n:.1f}ms/user)" if n else "  Total elapsed     : 0ms")

    print(f"\n  Sex-based error analysis (abnormal users only):")
    print(f"    Abnormal Males   : {output.total_abnormal_males} total, {output.incorrect_abnormal_males} incorrect "
          f"({output.error_rate_abnormal_males*100:.1f}% error rate)")
    print(f"    Abnormal Females : {output.total_abnormal_females} total, {output.incorrect_abnormal_females} incorrect "
          f"({output.error_rate_abnormal_females*100:.1f}% error rate)")
    error_diff = output.error_rate_abnormal_males - output.error_rate_abnormal_females
    print(f"    Difference: Males error rate - Females error rate = {error_diff*100:.1f}%")

    print(f"\n  Per-gender confusion matrices:")
    print(f"    Males (Healthy): {output.healthy_correct_males}/{output.healthy_total_males} correct "
          f"({output.healthy_correct_males/output.healthy_total_males*100:.1f}% accuracy)" if output.healthy_total_males else "    Males (Healthy): No healthy males")
    print(f"    Males (Diseased): {output.diseased_correct_males}/{output.diseased_total_males} correct "
          f"({output.diseased_correct_males/output.diseased_total_males*100:.1f}% sensitivity)" if output.diseased_total_males else "    Males (Diseased): No diseased males")
    print(f"    Females (Healthy): {output.healthy_correct_females}/{output.healthy_total_females} correct "
          f"({output.healthy_correct_females/output.healthy_total_females*100:.1f}% accuracy)" if output.healthy_total_females else "    Females (Healthy): No healthy females")
    print(f"    Females (Diseased): {output.diseased_correct_females}/{output.diseased_total_females} correct "
          f"({output.diseased_correct_females/output.diseased_total_females*100:.1f}% sensitivity)" if output.diseased_total_females else "    Females (Diseased): No diseased females")

    all_classes = sorted(
        set(output.misclassified_females_by_class) | set(output.misclassified_males_by_class)
    )
    if all_classes:
        print(f"\n  Misclassified diseased users/records by class and gender:")
        print(f"    {'Class':<8}  {'Female':>8}  {'Male':>8}")
        print(f"    {'-'*28}")
        for cls in all_classes:
            f_count = output.misclassified_females_by_class.get(cls, 0)
            m_count = output.misclassified_males_by_class.get(cls, 0)
            print(f"    {cls:<8}  {f_count:>8}  {m_count:>8}")
    else:
        print(f"\n  No misclassified diseased users/records.")

    diseased_recs = [r for r in output.records if r.true_label != _healthy_class(schema)]
    if diseased_recs:
        print(f"\n  Per-class breakdown (diseased users/records only):")
        by_class: Dict[int, List[PredictionRecord]] = defaultdict(list)
        for r in diseased_recs:
            by_class[r.true_label].append(r)
        print(f"  {'class':6} {'n_users':8} {'n_UNHEALTHY':11} {'n_HEALTHY':10} "
              f"{'n_SCREEN':9} {'detect%':8}")
        print("  " + "─" * 58)
        for cls in sorted(by_class.keys()):
            recs = by_class[cls]
            n_cls = len(recs)
            n_detected = sum(1 for r in recs if r.decision == HealthDecision.UNHEALTHY)
            n_miss = sum(1 for r in recs if r.decision == HealthDecision.HEALTHY)
            n_screen = sum(1 for r in recs if r.decision == HealthDecision.SCREENING)
            pct = n_detected / n_cls * 100 if n_cls else 0
            print(f"  {cls:6d} {n_cls:8d} {n_detected:11d} {n_miss:10d} "
                  f"{n_screen:9d} {pct:7.1f}%")

    print(f"\n  Fairness Metrics:")
    print(f"    Statistical Parity Difference (SPD): {output.fairness_spd:.4f}")
    print(f"    Disparate Impact (DI): {output.fairness_di:.4f}")
    print(f"    Equalized Odds Difference (EO): {output.fairness_eo_diff:.4f}")
    print(sep)


def print_action_ranking_summary(output: Algorithm4Output,
                                 top_k: int = 10,
                                 schema: Optional[DatasetSchema] = None) -> None:
    """
    Print the most frequently selected actions across predictions.
    """
    print(f"\n{'─'*65}")
    print(f"TOP {top_k} MOST FREQUENTLY SELECTED ACTIONS")
    counter: Dict[int, int] = defaultdict(int)
    for rec in output.records:
        for entry in rec.af_trace:
            counter[entry.feature_idx] += 1
    top = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"  {'rank':4} {'feat':4} {'feature_name':30} {'count':8}")
    print("  " + "─" * 50)
    for rank, (feat, cnt) in enumerate(top, 1):
        print(f"  {rank:4d} {feat:4d} {_feature_name(feat, schema):30s} {cnt:8d}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – PAPER FIDELITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

def assess_paper_fidelity() -> None:
    """
    Print a line-by-line mapping of the Algorithm 4 pseudocode to the
    implementation, with [PAPER]/[INFER]/[ENGR]/[AMBIG] annotations.

    [ENGR] This function serves as a living documentation of all design
    decisions and deviations from the pseudocode.
    """
    mapping = """
================================================================================
ALGORITHM 4 PAPER FIDELITY ASSESSMENT
================================================================================

INITIALIZATION BLOCK
---------------------
[PAPER] "c0 <- an action randomly selected from C_{m=1|h>1}^{Om=1}(FA=0)"
  -> Implementation: random.choice(alg3_output.retained_for_node("root"))
  -> initial_action_feat recorded in PredictionRecord.initial_action_feat

[PAPER] "m=1, k_m=∅, f_m=∅, t=∅, Decision=1, Threshold=Acceptable error"
  -> Implementation: current_focus=1, AF_real=0.0, pac_counter=[0]

[PAPER] "Extract features o^{0,0}_{m=1} ... choose with maximum r_{o_m=1|h}"
  -> [INFER] Initial random action already applied; the PAC framework
    implicitly handles feature extraction through the RL loop.
  -> [AMBIG] No separate initial measurement before the main loops.
    Initial action is the first entry into the while-loop for m=1.

[PAPER] "Calculate AF_{t=1,0,0} = P(h,m=1)*r_{m=1|h} / P(h>1,m=1)"
  -> Implementation: AF_real = 0.0 at start; first PAC adds the increment.
  -> Initial AF=0 means no assurance before any actions.              [PAPER]

LINE-BY-LINE MAPPING
--------------------
Line 1  : for k = set of k_m
  -> for current_focus in range(1, max_focus_level+1)                 [INFER]
  -> k corresponds to the branching feature of the current level's node.

Line 3  : for f = set of f_mk
  -> active_node = _find_applicable_node(user_idx, focus_level, ...)  [INFER]
  -> "f" is the branch the user belongs to (e.g., male or female).

Line 4  : t_mkf = 0
  -> pac_counter = [record.total_pac_count]  (reset per focus level)  [PAPER]

Line 5  : for h = 2 to H do
  -> for h in node_disease_classes (sorted ascending)                  [PAPER]

Line 6  : r_buf = r_{O_m^{kf}|h}
  -> sorted_actions_h = _get_sorted_disease_actions(nid, h, alg3)     [PAPER]

Line 7  : Sort r_buf descending, remove 0 value elements
  -> Already done in retained_for_node_disease() (sorted by weight)   [PAPER]
  -> Zero-weight actions excluded by action_weight > 0 filter          [PAPER]

Line 8  : O_m^kf = update features based on elements on r_buf
  -> O_mkf = [a.feature_idx for a in sorted_actions_h]                [PAPER]

Line 9  : C_buf <- C^mkf_{h|O^mkf}(FA=0)
  -> C_buf = list(sorted_actions_h)                                    [PAPER]

Line 10 : A_tmkf <- C_buf
  -> A_tmkf = list(C_buf)  (inside while-loop)                        [PAPER]

Lines 11-16 : RL lookahead (for all c ∈ A_tmkf)
  -> _rl_select_best_action(candidate_actions, node, h, AF_real, ...)
  -> For each action c: AF_sim_cj = P(h,f)*r_j|h / P(h>1,f)         [PAPER]
  -> rw_sim_cj = 1 - (AF_sim_cj + AF_real_current)                   [PAPER]
  -> [AMBIG] AF_sim starts from 0 for each RL call (not from AF_real).
    The paper's "+AF^cj" on the RHS means cumulation over past RL
    calls FOR THE SAME CANDIDATE; we reset to 0 per RL invocation.
    Rationale: AF_real already captures the running total.

Line 17 : J <- argmin_{J∈A_tmkf} Σ_j rw^{A_tmkf_j}_{t+j}
  -> J = action with minimum rw_sim_cj                                [PAPER]
  -> [INFER] With one feature per action, argmin rw_sim = argmax AF_sim.

Line 18 : remove J from C_buf
  -> C_buf = [a for a in C_buf if a.feature_idx != J.feature_idx]    [PAPER]

Line 19 : Apply action (sensor activation) J on User
  -> VO_buf[j] = _get_raw_value(user_global_idx, j, data)            [PAPER]

Line 20 : O_buf <- features of J
  -> O_buf = [J.feature_idx]  (one feature per action)               [PAPER]

Line 21 : VO_buf <- save values related to O_buf
  -> VO_buf dict populated from data matrix                           [PAPER]

Line 22 : t_mkf = t_mkf + 1  (inside "for j ∈ O_buf AND O_m^kf")
  -> pac_counter[0] += 1                                              [PAPER]

Line 23 : AF_{t_mkf} = P(h,f)*r_j|h / P(h>1,f) + AF_{t-1}
  -> AF_real += _compute_AF_increment(p_h_f, r_j_h, p_h_gt1_f)      [PAPER]
  -> Eq. 7 implemented exactly.

Line 24 : rw_{t_mkf} = 1 - AF_{t_mkf}
  -> rw_real = max(0.0, 1.0 - AF_real)                               [PAPER]
  -> [ENGR] clamped to avoid negative rw from numerical overflow.

Line 25 : if V_j < b_min OR V_j > b_max
  -> _is_outside_healthy_range(V_j, b_min, b_max)                    [PAPER]
  -> NaN -> False (missing measurement ≠ abnormality)                  [ENGR]

Lines 26-28 : Alarm -> Decision=Unhealthy -> Return
  -> return HealthDecision.UNHEALTHY                                  [PAPER]
  -> record.alarm_class = h                                           [INFER]

Line 29 : End if (alarm)
  -> end of if alarm block in inner loop                              [PAPER]

Line 30 : for (j ∈ set of O_buf) AND (j ∈ set of O_m^kf)
  -> features_to_test = [j for j in O_buf if j in O_mkf]            [PAPER]

Line 31-33 : if C_buf ≠ ∅ -> Go to line 18  (else continue)
  -> while C_buf: ... (while-loop continues to next action)          [PAPER]
  -> [AMBIG] "Go to line 18" is the canonical while-loop structure.

Line 34 : End for (h)
  -> end of "for h in disease_classes" loop                          [PAPER]

Line 35 : if rw_{t_mkf} <= Threshold
  -> if rw_final <= DIAGNOSTIC_THRESHOLD_ALG4                        [PAPER]

Lines 36-38 : Healthy life recommendations -> Decision=Healthy -> Return
  -> return HealthDecision.HEALTHY                                    [PAPER]

Line 39 : elseif Users in focus level m+1 >= u_min
  -> next_nodes = [c for c in node.all_children if c.n_users >= U_MIN_ALG4] [PAPER]
  -> [INFER] "Users in focus level m+1" = children with enough users.

Line 40 : increase focus level
  -> continue outer loop (current_focus += 1)                        [PAPER]

Lines 41-42 : else -> Run Screening process
  -> return HealthDecision.SCREENING                                  [PAPER]

Lines 44-45 : End for (f) / End for (k)
  -> end of focus-level loop                                          [PAPER]

KEY AMBIGUITIES RESOLVED
------------------------
[AMBIG-1] "AF^cj" in line 13 uses the same variable on both sides.
  -> Resolved: AF_sim per candidate starts at 0 each RL call.
  -> The "+AF^cj_{prev}" accumulates only within a multi-step RL sequence
    (not implemented here; single-step lookahead is sufficient for
     the paper's one-feature-per-action assumption).

[AMBIG-2] P(h, f^k_m) interpretation.
  -> Resolved: count(class-h users in node) / total users in node.   [INFER]
  -> Conditioned on current node, not the full dataset.

[AMBIG-3] What happens if the test user has NaN for the branching feature?
  -> Resolved: user stays at the root (focus level 1 node).          [ENGR]
  -> Rationale: Cannot route user to sex-specific branch if sex is unknown.

[AMBIG-4] "Increase focus level" – does it repeat ALL disease checks?
  -> Resolved: Yes – at the new node, all h=2..H are re-processed.   [INFER]
  -> AF_real is CARRIED FORWARD to the new focus level.              [ENGR]
  -> Rationale: evidence accumulated at level 1 should not be discarded.

================================================================================
"""
    print(mapping)

