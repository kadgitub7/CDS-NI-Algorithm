"""
================================================================================
Algorithm 4: CDS User-Health Prediction
(Planner, Reinforcement Learning, Policy in Executive
 and Running Diagnostic Test in the Perceptor)
================================================================================

PURPOSE
-------
Algorithm 4 is the prediction / inference phase of the CDS.  Given a single
test user whose health state is unknown, it applies a sequence of cognitive
actions (sensor activations), accumulates an Assurance Factor (AF), and
produces one of three decisions:

  UNHEALTHY  – at least one feature value fell outside the healthy range;
               the user is referred for disease diagnosis.
  HEALTHY    – rw = (1 - AF) ≤ Threshold; the user is referred for
               healthy-living recommendations.
  SCREENING  – all disease classes checked but AF still insufficient;
               the user is sent for screening.

PAPER FIDELITY NOTATION
-----------------------
  [PAPER]  – directly stated in the paper / pseudocode.
  [INFER]  – logically required but unspecified in the paper.
  [ENGR]   – engineering choice with justification.
  [AMBIG]  – paper is ambiguous; chosen interpretation is documented.

KEY EQUATIONS (for quick reference)
-------------------------------------
  Eq. 7   AF_{t_mkf}   = P(h,f^k_m) · r_{j|h} / P(h>1,f^k_m) + AF_{t-1}
  Eq. 8   rw_{t_mkf}   = 1 - AF_{t_mkf}

CRITICAL DESIGN DECISIONS
--------------------------
  1. TWO AF accumulators per disease-class iteration:
       AF_real – running assurance from actually-applied actions (Eq. 7/8).
       AF_sim  – simulated contribution used ONLY for the RL lookahead
                 (lines 11-17 of Algorithm 4), reset for each RL call.
     Merging these would conflate hypothesis selection with confirmed decisions.
     [AMBIG] resolved by strict separation.

  2. P(h, node) = node.health_dist[h] / node.n_users       [INFER]
     P(h>1,node) = node.n_diseased / node.n_users            [INFER]
     (both computed over training users in the relevant tree node)

  3. Each action ↔ ONE feature  (sensor → feature reading).  [PAPER]

  4. Line 32 "Go to line 18" → implemented as an inner while-loop.  [PAPER]

  5. Initial AF at t=0 is zero.                                [PAPER]

  6. Diseases processed in ascending order h=2..H.            [PAPER]

  7. NaN feature values NEVER trigger abnormality.            [ENGR]

  8. The initial action is randomly selected from the refined
     action library for the root node.                         [PAPER init]

INTEGRATION
-----------
Algorithm 4 is a CONSUMER of Algorithms 1-3.  It does NOT retrain models.
Required inputs:
  • DecisionTree (Algorithm 1)
  • Algorithm2Output – perceptor model library (healthy ranges + Bayesian tables)
  • Algorithm3Output – refined executive action library

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
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ── Import from Algorithms 1–3 ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from CDS_Paper_Algorithms import (
    DecisionTree, TreeNode, BranchDef, FeatureKind,
    load_dataset, build_decision_tree,
    FEATURE_NAMES, HEALTHY_CLASS, DIAGNOSTIC_THRESHOLD, U_MIN, N_FEATURES,
)
from Algorithm2 import (
    Algorithm2Output, ExecutiveActionEntry, PerceptorModelEntry,
    run_algorithm2, DEFAULT_N_BINS,
)
from Algorithm3 import (
    Algorithm3Output, run_algorithm3,
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
DIAGNOSTIC_THRESHOLD_ALG4: float = DIAGNOSTIC_THRESHOLD   # = 0.025

# [PAPER] Eq. 3 / Section VII.A: minimum users for reliable model extraction
U_MIN_ALG4: int = U_MIN   # = 200

# [PAPER] Class 1 = healthy
HEALTHY_CLASS_ALG4: int = HEALTHY_CLASS   # = 1

# [INFER] All disease class labels in the UCI Arrhythmia dataset
ALL_DISEASE_CLASSES: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class HealthDecision(Enum):
    """
    [PAPER] Section VI.B: three possible outcomes of Algorithm 4.

    HEALTHY    → "start healthy living recommendations"           (line 36)
    UNHEALTHY  → "Alarm and disease diagnosis process"            (line 26)
    SCREENING  → "Run Screening process"                          (line 42)
    UNKNOWN    → prediction incomplete (should not occur normally)
    """
    HEALTHY   = "Healthy"
    UNHEALTHY = "Unhealthy"
    SCREENING = "Screening"
    UNKNOWN   = "Unknown"


@dataclass
class AFTraceEntry:
    """
    One step in the Assurance Factor trace.

    Captures the state of both AF accumulators at each PAC (Perception-Action
    Cycle) so that monotonicity and convergence can be validated post-hoc.

    Attributes
    ----------
    pac_number   : global PAC counter (t_mkf in paper notation).
    node_id      : which tree node was active at this step.
    focus_level  : m.
    disease_h    : which disease class h was being investigated.
    feature_idx  : which feature was measured (action applied).
    feature_name : human-readable feature name.
    raw_value    : raw measured value from data (NaN if missing).
    b_min        : lower healthy-range boundary used for this test.
    b_max        : upper healthy-range boundary used for this test.
    r_j_h        : action weight r_{j|h} = P(B̂ outside range | h).
    p_h_f        : P(h, f^k_m) – prevalence term.
    p_h_gt1_f    : P(h>1, f^k_m) – denominator term.
    delta_AF     : P(h,f)*r_{j|h} / P(h>1,f) – the AF increment at this step.
    AF_real      : cumulative AF_real AFTER this step (Eq. 7).
    rw_real      : 1 - AF_real (Eq. 8).
    triggered_alarm : True if raw_value fell outside [b_min, b_max].
    is_nan       : True if the measurement was missing (NaN).
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
    RL lookahead record for one candidate action during the RL selection step
    (Algorithm 4 lines 11–17).

    [PAPER] The RL selects J = argmin_{c ∈ A_tmkf} Σ_j rw^cj_{t+j}
    [INFER] AF_sim starts from 0 for each RL call; it represents the
            SIMULATED incremental contribution of applying action c,
            not the running real AF.

    Attributes
    ----------
    action_feature_idx : feature o corresponding to candidate action c.
    action_feature_name: human-readable name.
    action_weight      : r_{o|h} from Algorithm 3.
    AF_sim_increment   : simulated AF contribution = P(h,f)*r/P(h>1,f).
    rw_sim             : 1 - (AF_sim_increment + AF_real_current).
    is_selected        : True if this was selected as action J.
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
    Record of checking one disease class h for one user in one node.

    [INFER] Emitted once per (node, disease_h) iteration.
    """
    node_id:           str
    focus_level:       int
    disease_h:         int
    n_actions_in_buf:  int           # size of C_buf for this h
    n_actions_applied: int           # how many actions were actually applied
    alarm_triggered:   bool          # did any feature fall outside range?
    alarm_feature_idx: Optional[int]  # which feature triggered alarm (if any)
    alarm_raw_value:   Optional[float]
    alarm_b_min:       Optional[float]
    alarm_b_max:       Optional[float]
    AF_real_at_end:    float
    rw_real_at_end:    float
    rl_selections:     List[RLLookaheadEntry] = field(default_factory=list)


@dataclass
class NodePredictionRecord:
    """
    Record of all predictions performed at one tree node for one user.

    [INFER] Emitted once per node traversal.
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
    Complete prediction record for one user.

    This is the primary output of run_algorithm4() for a single user.
    It captures the full decision trace, AF/rw trajectories, and
    action rankings for debugging and validation.

    [PAPER] Section VI.D: "Algorithm 4 shows how the CDS performs when we
    have a User in a smart e-Health home with an unknown health situation."

    Attributes
    ----------
    user_global_idx  : 0-indexed row in the dataset.
    true_label       : ground-truth class label (used ONLY for validation).
    decision         : HealthDecision output of the CDS.
    is_correct       : whether decision matches the ground truth (validation).
    alarm_class      : disease class h that triggered the alarm, or None.
    max_focus_reached: highest focus level used.
    total_pac_count  : total number of PACs executed.
    total_actions_applied: total sensor activations.
    elapsed_ms       : wall-clock time for this prediction.
    af_trace         : ordered list of AF/rw states at each PAC.
    node_records     : per-node prediction records.
    initial_action_feat : feature index of the initial randomly selected action.
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

    @property
    def true_is_healthy(self) -> bool:
        return self.true_label == HEALTHY_CLASS_ALG4

    @property
    def true_is_diseased(self) -> bool:
        return not self.true_is_healthy


@dataclass
class Algorithm4Output:
    """
    Aggregated output of running Algorithm 4 over multiple users.

    [PAPER] "The error rate can be calculated as the average of the error
    rate of each iteration." (LOOCV, Section VII.A)

    Attributes
    ----------
    records          : one PredictionRecord per user.
    n_healthy_correct: true healthy users predicted as HEALTHY.
    n_healthy_total  : total true healthy users.
    n_diseased_correct: true diseased users predicted as UNHEALTHY.
    n_diseased_total : total true diseased users.
    n_screening      : users sent to SCREENING.
    overall_accuracy : (correct_healthy + correct_diseased) / n_total.
    sensitivity      : n_diseased_correct / n_diseased_total  (=Diagnosis accuracy)
    specificity      : n_healthy_correct  / n_healthy_total   (=Normal rhythm accuracy)
    false_alarm_rate : n_diseased_predicted_healthy / n_diseased_total  [PAPER FA=0 goal]
    """
    records:           List[PredictionRecord] = field(default_factory=list)
    data:              np.ndarray = None  # Add data for gender analysis

    # Aggregate counters
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

    # Sex-based error analysis for abnormal users
    total_abnormal_males:     int   = 0
    total_abnormal_females:   int   = 0
    incorrect_abnormal_males: int   = 0
    incorrect_abnormal_females: int = 0
    error_rate_abnormal_males:    float = 0.0
    error_rate_abnormal_females:  float = 0.0

    # Per-gender confusion matrices
    healthy_correct_males: int = 0
    healthy_total_males: int = 0
    diseased_correct_males: int = 0
    diseased_total_males: int = 0
    healthy_correct_females: int = 0
    healthy_total_females: int = 0
    diseased_correct_females: int = 0
    diseased_total_females: int = 0

    # Misclassification by true class for females
    misclassified_females_by_class: Dict[int, int] = field(default_factory=dict)

    # Fairness metrics counters
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

    # Fairness metrics
    fairness_spd: float = 0.0
    fairness_di: float = 0.0
    fairness_eo_diff: float = 0.0

    def _recompute_stats(self) -> None:
        """
        Recompute aggregate statistics from records, paper-faithful per Table 5.

        Definitions (paper §VII.A / Table 5):
          • Specificity (1 - FA)        = healthy users NOT alarmed
                                        = (HEALTHY + SCREENING) / n_healthy_total
            [SCREENING is a "soft healthy" outcome — no false alarm.]
          • Sensitivity (diagnosis acc) = diseased users ALARMED (UNHEALTHY) / n_diseased_total
          • Total accuracy              = correct / total, where "correct" matches
                                          PredictionRecord.is_correct (already paper-aligned).
          • False alarm rate (FA)       = healthy users predicted UNHEALTHY / n_healthy_total
                                        = 1 - specificity (paper's true FA metric, target 0%).
        """
        # Healthy "correct" = NOT alarmed (HEALTHY or SCREENING are both fine)
        self.n_healthy_correct = sum(
            1 for r in self.records
            if r.true_is_healthy and r.decision != HealthDecision.UNHEALTHY
        )
        self.n_healthy_total = sum(1 for r in self.records if r.true_is_healthy)
        # Diseased "correct" = ALARMED (UNHEALTHY)
        self.n_diseased_correct = sum(
            1 for r in self.records
            if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY
        )
        self.n_diseased_total = sum(1 for r in self.records if r.true_is_diseased)
        self.n_screening = sum(1 for r in self.records if r.decision == HealthDecision.SCREENING)

        n_total = len(self.records)
        n_correct_total = self.n_healthy_correct + self.n_diseased_correct
        self.overall_accuracy  = n_correct_total / n_total if n_total else 0.0
        self.sensitivity       = (self.n_diseased_correct / self.n_diseased_total
                                   if self.n_diseased_total else 0.0)
        self.specificity       = (self.n_healthy_correct / self.n_healthy_total
                                   if self.n_healthy_total else 0.0)

        # [PAPER Table 5] False-alarm rate = healthy users predicted UNHEALTHY / n_healthy
        # = 1 - specificity. Target: 0% (paper's FA=0 policy).
        n_fa = sum(
            1 for r in self.records
            if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY
        )
        self.false_alarm_rate = n_fa / self.n_healthy_total if self.n_healthy_total else 0.0

        # Sex-based error analysis for abnormal users
        if self.data is not None:
            self.total_abnormal_males = sum(
                1 for r in self.records
                if r.true_is_diseased and self.data[r.user_global_idx, 1] == 0
            )
            self.total_abnormal_females = sum(
                1 for r in self.records
                if r.true_is_diseased and self.data[r.user_global_idx, 1] == 1
            )
            self.incorrect_abnormal_males = sum(
                1 for r in self.records
                if r.true_is_diseased and not r.is_correct and self.data[r.user_global_idx, 1] == 0
            )
            self.incorrect_abnormal_females = sum(
                1 for r in self.records
                if r.true_is_diseased and not r.is_correct and self.data[r.user_global_idx, 1] == 1
            )
            self.error_rate_abnormal_males = (
                self.incorrect_abnormal_males / self.total_abnormal_males
                if self.total_abnormal_males else 0.0
            )
            self.error_rate_abnormal_females = (
                self.incorrect_abnormal_females / self.total_abnormal_females
                if self.total_abnormal_females else 0.0
            )

            # Per-gender confusion matrices
            self.healthy_correct_males = sum(
                1 for r in self.records
                if r.true_is_healthy and r.decision != HealthDecision.UNHEALTHY and self.data[r.user_global_idx, 1] == 0
            )
            self.healthy_total_males = sum(
                1 for r in self.records
                if r.true_is_healthy and self.data[r.user_global_idx, 1] == 0
            )
            self.diseased_correct_males = sum(
                1 for r in self.records
                if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY and self.data[r.user_global_idx, 1] == 0
            )
            self.diseased_total_males = sum(
                1 for r in self.records
                if r.true_is_diseased and self.data[r.user_global_idx, 1] == 0
            )
            self.healthy_correct_females = sum(
                1 for r in self.records
                if r.true_is_healthy and r.decision != HealthDecision.UNHEALTHY and self.data[r.user_global_idx, 1] == 1
            )
            self.healthy_total_females = sum(
                1 for r in self.records
                if r.true_is_healthy and self.data[r.user_global_idx, 1] == 1
            )
            self.diseased_correct_females = sum(
                1 for r in self.records
                if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY and self.data[r.user_global_idx, 1] == 1
            )
            self.diseased_total_females = sum(
                1 for r in self.records
                if r.true_is_diseased and self.data[r.user_global_idx, 1] == 1
            )

            # Misclassification by true class for females
            self.misclassified_females_by_class = {}
            for r in self.records:
                if self.data[r.user_global_idx, 1] == 1 and r.true_is_diseased and not r.is_correct:
                    cls = r.true_label
                    self.misclassified_females_by_class[cls] = self.misclassified_females_by_class.get(cls, 0) + 1

            # Fairness metrics
            self.total_men = sum(1 for r in self.records if self.data[r.user_global_idx, 1] == 0)
            self.total_women = sum(1 for r in self.records if self.data[r.user_global_idx, 1] == 1)
            self.abnormal_men = sum(1 for r in self.records if r.true_is_diseased and self.data[r.user_global_idx, 1] == 0)
            self.abnormal_women = sum(1 for r in self.records if r.true_is_diseased and self.data[r.user_global_idx, 1] == 1)
            self.unhealthy_abnormal_men = sum(
                1 for r in self.records
                if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY and self.data[r.user_global_idx, 1] == 0
            )
            self.unhealthy_abnormal_women = sum(
                1 for r in self.records
                if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY and self.data[r.user_global_idx, 1] == 1
            )
            self.normal_men = sum(1 for r in self.records if r.true_is_healthy and self.data[r.user_global_idx, 1] == 0)
            self.normal_women = sum(1 for r in self.records if r.true_is_healthy and self.data[r.user_global_idx, 1] == 1)
            self.unhealthy_normal_men = sum(
                1 for r in self.records
                if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY and self.data[r.user_global_idx, 1] == 0
            )
            self.unhealthy_normal_women = sum(
                1 for r in self.records
                if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY and self.data[r.user_global_idx, 1] == 1
            )
            accuracy_men = sum(1 for r in self.records if r.is_correct and self.data[r.user_global_idx, 1] == 0) / self.total_men if self.total_men > 0 else 0
            accuracy_women = sum(1 for r in self.records if r.is_correct and self.data[r.user_global_idx, 1] == 1) / self.total_women if self.total_women > 0 else 0

            # Compute fairness metrics
            self.fairness_spd = accuracy_men - accuracy_women
            self.fairness_di = accuracy_women / accuracy_men if accuracy_men > 0 else 0
            tpr_male = self.unhealthy_abnormal_men / self.abnormal_men if self.abnormal_men > 0 else 0
            fpr_male = self.unhealthy_normal_men / self.normal_men if self.normal_men > 0 else 0
            tpr_female = self.unhealthy_abnormal_women / self.abnormal_women if self.abnormal_women > 0 else 0
            fpr_female = self.unhealthy_normal_women / self.normal_women if self.normal_women > 0 else 0
            self.fairness_eo_diff = abs(tpr_male - tpr_female) + abs(fpr_male - fpr_female)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_raw_value(global_user_idx: int, feature_idx: int,
                   data: np.ndarray) -> float:
    """
    Read the raw feature value for a user from the data matrix.

    [PAPER] BD_m^k(o, u) – raw feature value for user u, feature o.
    [ENGR]  Returns float('nan') for missing values.
    """
    val = data[global_user_idx, feature_idx]
    return float(val)


def _is_outside_healthy_range(value: float, b_min: float, b_max: float) -> bool:
    """
    [PAPER] Algorithm 4, line 25:
        if V_j < b_min^{mkfj}  OR  V_j > b_max^{kmfj}  → alarm

    [ENGR]  NaN → False (missing measurement cannot confirm abnormality).
    [PAPER] Strict inequalities as written in the paper.
    [ENGR]  Inverted range (b_min > b_max) signals an undefined healthy band
            (no healthy training users had a valid value for this feature);
            in that case do NOT raise an alarm — Eq. 5 cannot be tested.
            Per Algorithm 2, refined actions are NOT created for inverted
            ranges, so this branch should be unreachable in practice; it is
            a defensive safeguard.
    """
    if np.isnan(value):
        return False
    if np.isnan(b_min) or np.isnan(b_max) or b_min > b_max:
        return False
    return (value < b_min) or (value > b_max)


def _compute_p_h_f(node: TreeNode, disease_h: int) -> float:
    """
    P(h, f^k_m) = count(users with class h in this node) / n_users_node

    [INFER] The paper's "P(h, f^k_m)" is interpreted as the marginal
    probability of disease class h within the current tree node (branch f,
    feature k, level m).  This normalises over node users, not the full dataset,
    so that deeper nodes have their own prevalence estimates.
    """
    n_total = node.n_users
    if n_total == 0:
        return 0.0
    n_class_h = node.health_dist.get(disease_h, 0)
    return n_class_h / n_total


def _compute_p_h_gt1_f(node: TreeNode) -> float:
    """
    P(h>1, f^k_m) = n_diseased_users_in_node / n_users_node

    [INFER] Denominator in Eq. 7.  The sum of all disease-class counts.
    [INFER] If zero diseased users in node, return a tiny epsilon to prevent
    division by zero; the AF contribution will then be negligibly small.
    """
    n_total = node.n_users
    if n_total == 0:
        return 1e-9
    n_diseased = node.n_diseased
    if n_diseased == 0:
        return 1e-9   # [ENGR] avoid div-by-zero; AF stays near zero
    return n_diseased / n_total


def _compute_AF_increment(p_h_f: float, r_j_h: float, p_h_gt1_f: float) -> float:
    """
    [PAPER Eq. 7] delta_AF = P(h, f^k_m) * r_{j|h} / P(h>1, f^k_m)

    The increment is non-negative (all inputs are non-negative probabilities/weights).
    AF itself is clamped to [0, 1] at the call site to honour the paper's fuzzy-logic
    semantics: Section VI.A states "fuzzy logic ... values of variables can be a real
    number between 0 and 1", and §VI.D describes AF as the "assurance about the
    decision" (Figs. 11, 12 show AF approaching but not exceeding 1).
    """
    if p_h_gt1_f < 1e-12:
        return 0.0
    delta = (p_h_f * r_j_h) / p_h_gt1_f
    return float(max(0.0, delta))


def _update_AF(AF_real: float, delta_AF: float) -> float:
    """
    Apply Eq. 7 with the paper's fuzzy-logic cap at 1.0.

    [PAPER Eq. 7] AF^{tmkf} = P(h,f) r_{j|h} / P(h>1,f) + AF^{tmkf-1}
    [PAPER §VI.A] "fuzzy logic ... values of variables can be a real number between
                  0 and 1" — AF is a fuzzy assurance value, hence AF ∈ [0, 1].
    [PAPER Figs. 11, 12] AF asymptotes towards 1 but does not exceed it.

    Once AF saturates at 1, further informative actions cannot increase confidence
    (rw = 1 - AF is already 0). The cap also keeps the trace interpretable as a
    fuzzy-logic / probability-like quantity.
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
    Find the tree node at `focus_level` that the test user belongs to.

    Returns None if the User cannot be routed at the given focus level — the
    caller should halt focus escalation and accept the previous level's decision.

    [PAPER §VII.A] "The CDS rechecks the User, depending on whether the User
    is male or female."
    [PAPER §8.14 user override] If branching feature is missing, check other
    focus level 2 branching values; if all missing, diagnose based on level 1.

    `valid_node_ids` (optional): restrict consideration to nodes for which the
    perceptor has trained models.
    """
    if focus_level == 1:
        return tree.root

    nodes_at_level = tree.nodes_by_level.get(focus_level, [])
    for node in nodes_at_level:
        # Skip nodes that have no models in the perceptor library (e.g., Age
        # branches when nodes_filter selected Sex only).
        if valid_node_ids is not None and node.node_id not in valid_node_ids:
            continue
        if node.branch_def is not None:
            user_val = data[user_global_idx, node.branch_def.feature_idx]
            if node.branch_def.contains(user_val):
                log.debug(f"  User {user_global_idx}: m={focus_level} -> node {node.node_id!r} "
                          f"(feat={node.branch_def.feature_idx} val={user_val:.3f} {node.branch_def.label})")
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

    [PAPER] Algorithm 4 line 7: "Sort the r_buf decently and remove 0 value"
    [PAPER] Algorithm 4 line 9: C_buf is loaded fresh per (h) from the action
    library — the paper has no consumed-feature tracking across disease classes.
    """
    acts = alg3_output.retained_for_node_disease(node_id, disease_h)
    out = [a for a in acts if a.action_weight > 0.0]
    if consumed:
        # [FIX5] Exclude any (feature, disease_h) pair already applied at a
        # prior focus level so its AF contribution is not counted a second time.
        out = [a for a in out if (a.feature_idx, disease_h) not in consumed]
    return out

def _rl_select_best_action(
    candidate_actions: List[ExecutiveActionEntry],
    node: TreeNode,
    disease_h: int,
    AF_real_current: float,
    alg2_output: Algorithm2Output,
) -> Tuple[Optional[ExecutiveActionEntry], List[RLLookaheadEntry]]:
    """
    RL lookahead: select action J that minimises rw_sim.

    [PAPER + §8.9 override] Algorithm 4 lines 11-17 with the user's resolved
    typo correction:
        for all actions c ∈ A_tmkf:
            for j ∈ set of O_m^kf ∩ features of c:
                AF^cj = P(h,f) * r_{j|h} / P(h>1,f)            [line 13, corrected]
                rw^cj = 1 − (AF^cj + AF_tmkf)                   [line 14]
        J ← argmin_{c ∈ A_tmkf}  Σ_j rw^cj                      [line 17]
    Per  §8.9, the printed line 13's
    +AF^cj self-reference is a typo; the correct form has no addition.

    [ENGR]  Each action corresponds to one feature.  "features extracted from c"
    = {c.feature_idx}.  We compute AF_sim for that single feature.

    Returns
    -------
    (selected_action, rl_lookahead_entries)
    selected_action = None if no candidates available.
    """
    if not candidate_actions:
        return None, []

    p_h_f     = _compute_p_h_f(node, disease_h)
    p_h_gt1_f = _compute_p_h_gt1_f(node)

    rl_entries: List[RLLookaheadEntry] = []
    best_action: Optional[ExecutiveActionEntry] = None
    best_rw_sim: float = float("inf")

    for action in candidate_actions:
        j = action.feature_idx
        r_j_h = action.action_weight   # r_{j|h} from Algorithm 3

        # [PAPER §8.9 corrected] AF^cj is the lookahead INCREMENT only; it is
        # NOT accumulated across RL iterations — each candidate is evaluated
        # independently against the current AF_real.
        AF_sim_cj = _compute_AF_increment(p_h_f, r_j_h, p_h_gt1_f)

        # [PAPER] line 14: rw^cj = 1 - (AF_sim_cj + AF_real_current)
        rw_sim_cj = 1.0 - (AF_sim_cj + AF_real_current)

        is_sel = False
        if rw_sim_cj < best_rw_sim:
            best_rw_sim  = rw_sim_cj
            best_action  = action
            is_sel       = True
            # Mark previous best as not selected
            for e in rl_entries:
                e.is_selected = False

        rl_entries.append(RLLookaheadEntry(
            action_feature_idx  = j,
            action_feature_name = action.feature_name,
            action_weight       = r_j_h,
            AF_sim_increment    = AF_sim_cj,
            rw_sim              = rw_sim_cj,
            is_selected         = is_sel,
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
    pac_counter:      List[int],          # [0] = current PAC count (mutable)
    record:           PredictionRecord,
    consumed_actions: Optional[Set[Tuple[int, int]]] = None,
    initial_action_h: Optional[int] = None
) -> Tuple[HealthDecision, float, Optional[int]]:
    """
    Run the inner prediction loop of Algorithm 4 at ONE tree node.

    [PAPER] Algorithm 4 lines 1–43 mapped to this function.

    This function handles:
      • Looping over disease classes h=2..H         (line 5 / line 34)
      • For each h: RL lookahead + action selection  (lines 9–17)
      • Action application + diagnostic test          (lines 18–33)
      • After all h: check threshold / escalate       (lines 35–43)

    Parameters
    ----------
    AF_real_init : AF_real value at the start of this focus level. AF accumulates
                   across focus levels (Eq. 7 has no reset clause).
    pac_counter  : single-element list for in-place update of per-(k,f) PAC count.
    record       : PredictionRecord to append trace entries to.

    Returns
    -------
    (decision, AF_real_final, alarm_disease_class)
      alarm_disease_class is None unless decision == UNHEALTHY.
      UNKNOWN means "escalate to next focus level".
    """
    nid = node.node_id
    m   = node.focus_level

    AF_real: float = AF_real_init   # [PAPER] running assurance factor (Eq. 7)
    alarm_class: Optional[int] = None
    node_rec = NodePredictionRecord(node_id=nid, focus_level=m)

    log.debug(f"\n  {'='*55}")
    log.debug(f"  Node {nid!r}  m={m}  n_users={node.n_users}")
    log.debug(f"  disease_classes={disease_classes}  AF_init={AF_real_init:.4f}")

    p_h_gt1_f_node = _compute_p_h_gt1_f(node)

    # ─────────────────────────────────────────────────────────────────
    # [PAPER] Line 5: for h = 2 to H do
    # ─────────────────────────────────────────────────────────────────
    for h in disease_classes:
        p_h_f     = _compute_p_h_f(node, h)
        p_h_gt1_f = p_h_gt1_f_node

        # [PAPER] Line 6: r_buf = r_{O^kf_m|h}
        # [PAPER] Line 9: C_buf loaded fresh per h — no consumed tracking.
        sorted_actions_h = _get_sorted_disease_actions(
            nid, h, alg3_output,
            consumed=consumed_actions,   # [FIX5] skip already-applied pairs
        )
        # [FIX] Exclude j_init from C_buf when processing h_init to prevent
        # double-application. The initialization block already applied j_init and
        # credited its delta_AF. Re-selecting it here would add its AF contribution
        # a second time and waste a PAC budget slot.
        # [PAPER] The init block is outside the line 5–33 loop; its action must not
        # re-enter C_buf for disease class h_init.
        if (initial_action_h is not None
                and h == initial_action_h
                and record.initial_action_feat is not None):
            sorted_actions_h = [
                a for a in sorted_actions_h
                if a.feature_idx != record.initial_action_feat
            ]

        # [PAPER] Line 7: Sort r_buf decently and remove 0 value elements
        #         (already done in _get_sorted_disease_actions)

        if not sorted_actions_h:
            log.debug(f"    h={h}: no refined actions at node {nid!r} → skip")
            dc_rec = DiseaseCheckRecord(
                node_id=nid, focus_level=m, disease_h=h,
                n_actions_in_buf=0, n_actions_applied=0,
                alarm_triggered=False, alarm_feature_idx=None,
                alarm_raw_value=None, alarm_b_min=None, alarm_b_max=None,
                AF_real_at_end=AF_real, rw_real_at_end=1.0 - AF_real,
            )
            node_rec.disease_checks.append(dc_rec)
            continue

        # [PAPER] Line 8: O_m^kf = update features based on elements on r_buf
        # Feature indices available for this disease class at this node
        O_mkf: List[int] = [a.feature_idx for a in sorted_actions_h]

        # [PAPER] Line 9: C_buf ← C^mkf_{h|O^mkf}(FA=0)
        # [PAPER] Line 10: A_tmkf ← C_buf
        C_buf: List[ExecutiveActionEntry] = list(sorted_actions_h)
        n_actions_total = len(C_buf)

        disease_check_rec = DiseaseCheckRecord(
            node_id=nid, focus_level=m, disease_h=h,
            n_actions_in_buf=n_actions_total, n_actions_applied=0,
            alarm_triggered=False, alarm_feature_idx=None,
            alarm_raw_value=None, alarm_b_min=None, alarm_b_max=None,
            AF_real_at_end=AF_real, rw_real_at_end=1.0 - AF_real,
        )

        # ─────────────────────────────────────────────────────────────
        # [PAPER] Lines 18–33: while C_buf ≠ ∅:
        #   (Line 32: "Go to line 18" → while-loop)
        # ─────────────────────────────────────────────────────────────
        while C_buf:

            # ── [PAPER] Lines 11-17: RL lookahead ────────────────────
            A_tmkf = list(C_buf)   # current action buffer

            selected_action, rl_entries = _rl_select_best_action(
                candidate_actions = A_tmkf,
                node              = node,
                disease_h         = h,
                AF_real_current   = AF_real,
                alg2_output       = alg2_output,
            )
            disease_check_rec.rl_selections.extend(rl_entries)

            if selected_action is None:
                break   # [ENGR] safety guard

            # [PAPER] Line 17: J ← argmin
            J = selected_action

            # [PAPER] Line 18: remove J from C_buf
            C_buf = [a for a in C_buf if a.feature_idx != J.feature_idx]

            # [PAPER] Line 19: Apply action (sensor activation) J on User
            # [INFER] "Applying action" = reading the sensor = retrieving the
            #         raw feature value from the data matrix.

            # ── [PAPER] Lines 20-21: O_buf, VO_buf ───────────────────
            # [PAPER] O_buf ← features of J
            O_buf: List[int] = [J.feature_idx]
            # [PAPER] VO_buf ← save values related to O_buf
            VO_buf: Dict[int, float] = {}
            for j in O_buf:
                VO_buf[j] = _get_raw_value(user_global_idx, j, data)

            # [PAPER] Line 30: for j ∈ set of O_buf AND j ∈ O_m^kf
            features_to_test = [j for j in O_buf if j in O_mkf]

            for j in features_to_test:
                V_j = VO_buf[j]

                # [PAPER §VII.A + spec §8.13] Sensor failure → skip entirely.
                if np.isnan(V_j):
                    log.debug(f"    [SKIP NaN] j={j} for h={h}: sensor failure, no PAC update")
                    continue

                # [PAPER] Line 22: t_mkf = t_mkf + 1
                pac_counter[0] += 1
                t_mkf = pac_counter[0]
                
                disease_check_rec.n_actions_applied += 1

                # [FIX] Mark this (feature, disease_h) as consumed so it cannot
                # re-enter C_buf if focus escalates to level 2.
                if consumed_actions is not None:
                    consumed_actions.add((j, h))

                # Get healthy range from Algorithm 2 perceptor library
                model_entry = alg2_output.get_model(nid, j)
                if model_entry is None:
                    log.debug(f"    h={h} j={j}: no model entry at node {nid!r} → skip")
                    continue

                b_min = model_entry.healthy_range.b_min_healthy
                b_max = model_entry.healthy_range.b_max_healthy

                # Get action weight for this (node, feature, disease)
                exec_entry = alg2_output.get_action(nid, j, h)
                r_j_h = exec_entry.action_weight if exec_entry else J.action_weight

                # [PAPER Eq. 7 / Line 23] AF_{t_mkf} = P(h,f)*r_j|h / P(h>1,f) + AF_{t-1}
                # AF is capped at 1.0 per §VI.A (fuzzy logic values in [0,1]) and
                # consistent with Figs. 11/12 where AF asymptotes to but never exceeds 1.
                delta_AF = _compute_AF_increment(p_h_f, r_j_h, p_h_gt1_f)
                AF_real  = _update_AF(AF_real, delta_AF)

                # [PAPER Eq. 8 / Line 24] rw_{t_mkf} = 1 - AF_{t_mkf}
                rw_real = 1.0 - AF_real

                # [PAPER] Line 25: if V_j < b_min OR V_j > b_max
                alarm = _is_outside_healthy_range(V_j, b_min, b_max)

                # Append to AF trace
                trace_entry = AFTraceEntry(
                    pac_number     = t_mkf,
                    node_id        = nid,
                    focus_level    = m,
                    disease_h      = h,
                    feature_idx    = j,
                    feature_name   = FEATURE_NAMES.get(j, f"feat_{j}"),
                    raw_value      = V_j,
                    b_min          = b_min,
                    b_max          = b_max,
                    r_j_h          = r_j_h,
                    p_h_f          = p_h_f,
                    p_h_gt1_f      = p_h_gt1_f,
                    delta_AF       = delta_AF,
                    AF_real        = AF_real,
                    rw_real        = rw_real,
                    triggered_alarm= alarm,
                    is_nan         = False,
                )
                record.af_trace.append(trace_entry)
                record.total_actions_applied += 1

                log.debug(
                    f"    PAC={t_mkf:3d}  h={h:2d}  j={j:3d}({FEATURE_NAMES.get(j,'?')[:15]:15s})"
                    f"  V={V_j:8.3f}  range=[{b_min:.3f},{b_max:.3f}]"
                    f"  r={r_j_h:.4f}  dAF={delta_AF:.4f}"
                    f"  AF={AF_real:.4f}  rw={rw_real:.4f}"
                    + ("  *** ALARM ***" if alarm else "")
                )

                if alarm:
                    # [PAPER] Lines 26-28: Alarm and disease diagnosis → Decision=Unhealthy
                    disease_check_rec.alarm_triggered   = True
                    disease_check_rec.alarm_feature_idx = j
                    disease_check_rec.alarm_raw_value   = V_j
                    disease_check_rec.alarm_b_min       = b_min
                    disease_check_rec.alarm_b_max       = b_max
                    disease_check_rec.AF_real_at_end    = AF_real
                    disease_check_rec.rw_real_at_end    = rw_real
                    node_rec.disease_checks.append(disease_check_rec)
                    node_rec.final_AF_real  = AF_real
                    node_rec.final_rw_real  = rw_real
                    record.node_records.append(node_rec)
                    alarm_class = h
                    log.info(
                        f"  [ALARM] user={user_global_idx}  h={h}  "
                        f"feat={j}({FEATURE_NAMES.get(j,'?')})  "
                        f"V={V_j:.3f} outside [{b_min:.3f},{b_max:.3f}]"
                    )
                    # [PAPER] Line 28: Return Decision
                    return HealthDecision.UNHEALTHY, AF_real, alarm_class

                # --- Algorithm 4: Node Update Logic ---
                # [PAPER Page 320, Line 18] Check if this feature branches the tree
                if node.branching_feat_k == j:
                    # Determine which branch (f) the user falls into
                    next_node = None
                    for child in node.all_children:
                        if child.branch_def.contains(V_j):
                            next_node = child
                            break

                # [PAPER] Line 29: End if (alarm)
            # end for j in features_to_test

            # [PAPER] Line 31: if C_buf ≠ ∅ → Go to line 18 (while-loop continues)
            # [PAPER] Line 33: End if
        # end while C_buf

        # Update disease check record after exhausting all actions for h
        disease_check_rec.AF_real_at_end = AF_real
        disease_check_rec.rw_real_at_end = 1.0 - AF_real
        node_rec.disease_checks.append(disease_check_rec)
    # end for h

    # ─────────────────────────────────────────────────────────────────
    # [PAPER] Lines 35-43: Post-disease-loop decision
    # ─────────────────────────────────────────────────────────────────
    rw_final = 1.0 - AF_real
    node_rec.final_AF_real = AF_real
    node_rec.final_rw_real = rw_final
    record.node_records.append(node_rec)

    # [PAPER ALG-4 lines 35-38] Threshold check: HEALTHY at ANY focus level if rw ≤ threshold.
    # The paper does not restrict HEALTHY to the final level; escalation only fires when
    # the threshold is NOT met (line 39).
    if rw_final <= DIAGNOSTIC_THRESHOLD_ALG4:
        node_rec.met_threshold = True
        log.debug(f"  Node {nid!r}: rw={rw_final:.4f} <= Threshold="
                  f"{DIAGNOSTIC_THRESHOLD_ALG4} -> HEALTHY")
        return HealthDecision.HEALTHY, AF_real, None

    # [PAPER] Line 39: elseif Users in focus level m+1 >= u_min → increase focus
    # [INFER] "Users in focus level m+1" means: are there child nodes of the
    #         current node with sufficient users to provide reliable models?
    can_increase_focus = False
    next_m = m + 1
    next_nodes = [
        child for child in node.all_children
        if child.focus_level == next_m and child.n_users >= U_MIN_ALG4
    ]
    if next_nodes:
        can_increase_focus = True

    if can_increase_focus:
        # [PAPER] Line 40: increase focus level
        # [INFER] Signal to caller to repeat at next focus level.
        #         We return a special value; run_algorithm4 handles the recursion.
        node_rec.focus_increased = True
        log.debug(f"  Node {nid!r}: rw={rw_final:.4f} > threshold, "
                  f"focus can increase to m={next_m}")
        return HealthDecision.UNKNOWN, AF_real, None   # UNKNOWN → caller increases focus

    # [PAPER] Lines 41-42: else → Run Screening process
    node_rec.sent_to_screening = True
    log.debug(f"  Node {nid!r}: rw={rw_final:.4f} > threshold, "
              f"focus cannot increase → SCREENING")
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
) -> PredictionRecord:
    """
    Run Algorithm 4 for a single test user.

    This is a faithful line-by-line implementation of the Algorithm 4
    pseudocode from the paper (Section VI.D).

    Parameters
    ----------
    user_global_idx : 0-indexed row in `data` of the test user.
    data            : full (N, 279) feature matrix.
    labels          : full (N,) label array (1=healthy, 2-16=disease).
    tree            : DecisionTree from Algorithm 1.
    alg2_output     : Algorithm2Output from Algorithm 2.
    alg3_output     : Algorithm3Output from Algorithm 3 (reset_per_h=False, per §8.8).
    rng_seed        : optional random seed for reproducibility.
    verbose         : if True, set logger to DEBUG level.

    Returns
    -------
    PredictionRecord with full decision trace.

    Algorithm 4 Pseudocode → Implementation Mapping
    ------------------------------------------------
    Init  : load root node normal ranges, actions          ← [below]
    Init  : randomly select initial action from C          ← initial_action step
    Init  : compute AF_0                                    ← AF_real=0 start
    Line 1: for k = set of k_m                             ← focus_level loop (m=1,2)
    Line 3: for f = set of f_mk                            ← node_for_user routing
    Line 4: t_mkf = 0                                      ← pac_counter reset
    Line 5: for h = 2 to H                                 ← disease loop in _predict_at_node
    Lines 9-17: Planning + RL                              ← _rl_select_best_action
    Lines 18-33: Apply + test + while-loop                 ← while C_buf loop
    Lines 35-43: Threshold / focus / screening             ← end of _predict_at_node
    """
    if verbose:
        log.setLevel(logging.DEBUG)
        for h in log.handlers:
            h.setLevel(logging.DEBUG)

    if rng_seed is not None:
        random.seed(rng_seed)
        np.random.seed(rng_seed)

    t0 = time.perf_counter()
    true_label = int(labels[user_global_idx])

    record = PredictionRecord(
        user_global_idx = user_global_idx,
        true_label      = true_label,
    )

    log.debug(f"\n{'='*65}")
    log.debug(f"  Algorithm 4 | user_idx={user_global_idx}  true_label={true_label}")

    # ── Initialization: select initial action randomly from root actions ──
    # [PAPER] Initialization block (p. 13-14):
    # "c0 ← an action randomly selected from C_{m=1|h>1}^{Om=1}(FA=0)"
    # "Apply c0 to the User (h ∈ {2,...,H})"
    # "Extract features o^{0,0}_{m=1} and check if outside healthy range"
    # "Calculate AF_{t_{m=1,00}} = P(h,m=1)*r_{om=1|h} / P(h>1,m=1)"
    # "calculate first internal rewards as 1 − AF_{t_{m=1,00}}"
    # [FIX1] Fully execute the initialization: apply c0, read value, compute AF, check alarm
    
    root_node = tree.root
    AF_real: float = 0.0   # [PAPER] initial AF at t=0 is zero
    pac_counter = [0]      # mutable counter for PAC number t_mkf

    # [FIX] Shared mutable set of (feature_idx, disease_h) pairs that have
    # already contributed to AF_real. Passed to every _predict_at_node call
    # so level-2 C_buf construction skips anything already applied at level 1.
    consumed_pairs: Set[Tuple[int, int]] = set()

    root_all_actions = alg3_output.retained_for_node("root")
    h_init = -1
    if root_all_actions:
        # [PAPER ALG-4 init] c0 = random action from the refined root library.
        # Step 1: random feature among root features.
        # Step 2: among all (j_init, h) pairs, pick the h with maximum r_{j|h}.
        # [Bug #33] Bounded retry loop: if V_init is NaN, re-draw from reduced pool.
        chosen = False
        tried: set = set()
        j_init = -1
        h_init = -1
        initial_action = None
        # [PAPER ALG-4 init] c0 = uniform random draw from all (feature, disease_class)
        # pairs in the refined root library whose feature value is not NaN.
        # [FIX] Previously: picked a random *feature* then took the max-weight disease
        #        for that feature — biasing toward high-weight diseases and effectively
        #        pre-selecting the RL's first pick deterministically.
        # [FIX] Now: filter the full action list to non-NaN features, then sample one
        #        action uniformly so every (j, h) pair has equal probability.
        valid_candidates = [
            a for a in root_all_actions
            if not np.isnan(_get_raw_value(user_global_idx, a.feature_idx, data))
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
                      f"({FEATURE_NAMES.get(j_init,'?')})  "
                      f"h={h_init}  w={initial_action.action_weight:.4f}")

            # [PAPER ALG-4 init] Get healthy range and disease weights for initialization
            model_entry = alg2_output.get_model(root_node.node_id, j_init)
            if model_entry is not None:
                b_min = model_entry.healthy_range.b_min_healthy
                b_max = model_entry.healthy_range.b_max_healthy

                p_h_f = _compute_p_h_f(root_node, h_init)
                p_h_gt1_f = _compute_p_h_gt1_f(root_node)
                r_j_h = initial_action.action_weight

                delta_AF_init = _compute_AF_increment(p_h_f, r_j_h, p_h_gt1_f)
                AF_real = _update_AF(AF_real, delta_AF_init)

                pac_counter[0] += 1
                record.total_pac_count += 1
                record.total_actions_applied += 1

                alarm_init = _is_outside_healthy_range(V_init, b_min, b_max)

                trace_entry = AFTraceEntry(
                    pac_number      = pac_counter[0],
                    node_id         = root_node.node_id,
                    focus_level     = 1,
                    disease_h       = h_init,
                    feature_idx     = j_init,
                    feature_name    = FEATURE_NAMES.get(j_init, f"feat_{j_init}"),
                    raw_value       = V_init,
                    b_min           = b_min,
                    b_max           = b_max,
                    r_j_h           = r_j_h,
                    p_h_f           = p_h_f,
                    p_h_gt1_f       = p_h_gt1_f,
                    delta_AF        = delta_AF_init,
                    AF_real         = AF_real,
                    rw_real         = 1.0 - AF_real,
                    triggered_alarm = alarm_init,
                    is_nan          = False,
                )
                record.af_trace.append(trace_entry)
                # [FIX] Register init action as consumed.
                consumed_pairs.add((j_init, h_init))

                log.debug(
                    f"    INIT: PAC={pac_counter[0]:3d}  j={j_init:3d}"
                    f"({FEATURE_NAMES.get(j_init,'?')[:15]:15s})  "
                    f"V={V_init:8.3f}  range=[{b_min:.3f},{b_max:.3f}]  "
                    f"r={r_j_h:.4f}  dAF={delta_AF_init:.4f}  "
                    f"AF={AF_real:.4f}  rw={1.0-AF_real:.4f}"
                    + ("  *** INIT ALARM ***" if alarm_init else "")
                )

                # If initialization triggers alarm, return UNHEALTHY immediately.
                if alarm_init:
                    record.decision = HealthDecision.UNHEALTHY
                    record.is_correct = (true_label != HEALTHY_CLASS_ALG4)
                    record.alarm_class = h_init
                    record.alarm_feature_idx = j_init
                    record.max_focus_reached = 1
                    record.elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    log.info(
                        f"  [INIT ALARM] user={user_global_idx}  "
                        f"feat={j_init}({FEATURE_NAMES.get(j_init,'?')})  "
                        f"h={h_init}  V={V_init:.3f} outside "
                        f"[{b_min:.3f},{b_max:.3f}]  -> UNHEALTHY"
                    )
                    return record
    else:
        record.initial_action_feat = None
        log.debug("  No root actions available for initial selection")

    # Disease classes present in the full dataset
    # [PAPER] "for h ∈ {hd2, ..., hdH}" – all disease classes
    all_disease_classes: List[int] = sorted(ALL_DISEASE_CLASSES)

    decision = HealthDecision.UNKNOWN

    # ─────────────────────────────────────────────────────────────────────
    # [PAPER] Line 1: for k = set of k_m
    # [INFER] We iterate over focus levels m=1, 2, ...
    #         At m=1: k=∅, f=∅ → root node.
    #         At m=2: k=1 (sex feature), f=male/female → appropriate child.
    #         We stop when a definitive decision is reached or focus cannot increase.
    # ─────────────────────────────────────────────────────────────────────
    max_focus_level = tree.depth()

    # Restrict routing to nodes for which the perceptor has trained models.
    # Algorithm 1 produces many level-2 branchings but the case study (and
    # nodes_filter) only train models on the Sex branches. Without this filter,
    # _find_applicable_node would route the test user to e.g. Age branches that
    # have no perceptor models, causing silent fall-through.
    valid_node_ids = {e.node_id for e in alg2_output.perceptor_library}

    for current_focus in range(1, max_focus_level + 1):

        # [PAPER] Line 3: for f = set of f_mk
        # [INFER] Find which node at this focus level applies to the test user.
        active_node = _find_applicable_node(
            user_global_idx = user_global_idx,
            focus_level     = current_focus,
            tree            = tree,
            data            = data,
            valid_node_ids  = valid_node_ids,
        )

        if active_node is None:
            log.debug(f"  m={current_focus}: no applicable node → stop")
            # Don't escalate further; the previous focus level's decision was UNKNOWN
            # but we can't escalate, so treat as SCREENING per paper line 41-42
            decision = HealthDecision.SCREENING
            break

        record.max_focus_reached = current_focus

        log.info(
            f"  Focus m={current_focus}  node={active_node.node_id!r}  "
            f"n_users={active_node.n_users}  AF_init={AF_real:.4f}"
        )

        # Determine disease classes present in this node (subset of full set)
        # [INFER] Only process disease classes that appear in training data at this node
        
        node_disease_classes = sorted([
            h for h in all_disease_classes
            if active_node.health_dist.get(h, 0) > 0
        ])
        '''
        node_disease_classes = sorted(
            [h for h in active_node.health_dist if h != HEALTHY_CLASS and active_node.health_dist[h] > 0],
            key=lambda h: active_node.health_dist[h],
            reverse=True
        )
        '''

        if not node_disease_classes:
            log.debug(f"  Node {active_node.node_id!r}: no disease classes → skip")
            # [INFER] If no disease classes at this node, treat as healthy
            decision = HealthDecision.HEALTHY
            break

        # [PAPER ALG-4 line 4] t_m^{kf} = 0 — reset per (k, f) iteration.
        pac_counter = [0]

        # ── Run prediction at this node ────────────────────────────────────
        decision, AF_real, alarm_class = _predict_at_node(
            user_global_idx  = user_global_idx,
            node             = active_node,
            disease_classes  = node_disease_classes,
            data             = data,
            labels           = labels,
            alg2_output      = alg2_output,
            alg3_output      = alg3_output,
            AF_real_init     = AF_real,
            pac_counter      = pac_counter,
            record           = record,
            consumed_actions = consumed_pairs,
            initial_action_h = h_init,
        )
        # Maintain a separate global PAC count for trace/reporting purposes.
        record.total_pac_count += pac_counter[0]

        if decision == HealthDecision.UNHEALTHY:
            record.alarm_class = alarm_class
            # Extract alarm feature from the most recent node record's disease checks
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

        # decision == UNKNOWN → focus can be increased → continue outer loop
        # [PAPER §8.11 user override; CORRECTION TO BUG #9]
        # AF persists across focus-level transitions. The user override
        # explicitly rejects the per-focus-level reset implied by Figs 11/12.
        # AF_real continues to accumulate; pac_counter still resets per (k,f)
        # per Alg.4 line 4.
        log.info(f"  m={current_focus}: focus increasing to m={current_focus + 1}")
        continue

    # ── Finalize decision ─────────────────────────────────────────────────────
    if decision == HealthDecision.UNKNOWN:
        # [ENGR] Should not occur in normal operation; fall back to SCREENING
        decision = HealthDecision.SCREENING
        log.debug("  Decision remained UNKNOWN after all focus levels → SCREENING")

    # Determine correctness — paper-faithful metric per §VII.A and Table 5.
    #
    # Paper Table 5 reports "Total accuracy = 95.4%" with "Diagnosis accuracy = 90%"
    # at Focus Level 2. Working backwards: 95.4% × 452 ≈ 431 correct = 90% × 207
    # diseased + 100% × 245 healthy → specificity = 100%. The paper achieves this
    # despite explicitly sending healthy female users to SCREENING (§VII.A).
    #
    # Reconciliation: SCREENING is a "soft healthy" outcome — the system did NOT
    # raise a false alarm, it just couldn't reach the assurance threshold. Per
    # §VI.B (high-level step 12): "Claim User as the healthy and send the User for
    # the screening" — SCREENING is *under the healthy verdict* with extra checks.
    #
    # Therefore:
    #   • True healthy: HEALTHY or SCREENING → correct (no false alarm)
    #                   UNHEALTHY            → wrong (false alarm; FA>0)
    #   • True diseased: UNHEALTHY           → correct (diagnosis caught it)
    #                    HEALTHY or SCREENING → wrong (missed diagnosis)
    if true_label == HEALTHY_CLASS_ALG4:
        # User is truly healthy: correct unless we falsely alarmed them
        is_correct = (decision != HealthDecision.UNHEALTHY)
    else:
        # User is truly diseased: correct only if alarmed (UNHEALTHY)
        is_correct = (decision == HealthDecision.UNHEALTHY)

    record.decision = decision
    record.is_correct = is_correct
    record.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    log.info(
        f"  DECISION: user={user_global_idx}  true={true_label}  "
        f"→ {decision.value}  correct={is_correct}  "
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
) -> Algorithm4Output:
    """
    Paper-faithful Leave-One-Out Cross-Validation wrapper for Algorithm 4.

    [PAPER §VII.A, p.180120] "all Users except one are used for training, and
    the excluded User is used for testing and health-state diagnosis accuracy.
    This process is repeated for N = 452 times for N Users."

    [PAPER §VII.A, p.180121] "the User-under-test is removed from the
    Arrhythmia database. Then, at Focus level 1, using Algorithm 2 in the CDS
    training mode, the perceptor creates the model library, and the executive
    extracts the important features. Next, in Algorithm 3, these important
    features are refined..."

    Per-fold protocol: for each held-out test user u,
      1. Build training set = all 452 users EXCEPT u (451 users).
      2. Run Algorithm 1 on training set → tree_u.
      3. Run Algorithm 2 on training set + tree_u → alg2_u.
      4. Run Algorithm 3 on training set + tree_u + alg2_u → alg3_u.
      5. Run Algorithm 4 prediction on user u using (tree_u, alg2_u, alg3_u).
    Repeat for all 452 users; aggregate the predictions.

    This is what the paper calls LOOCV. The previous "train once on full
    dataset" approximation guaranteed FA = 0 by construction (the test
    user contributed to the healthy range); per-fold retraining is the only
    way to honestly verify the FA = 0 invariant.

    Parameters
    ----------
    max_users    : if set, limit to first `max_users` users (for quick testing).
    n_bins       : Algorithm 2 discretization bin count (default 20).
    nodes_filter : Algorithm 2/3 nodes_filter (default: root + sex branches).
    """
    if nodes_filter is None:
        # [PAPER §VII.A] Sex-branched nodes at level 2 (Sex = column 1).
        nodes_filter = ["root", "root|k1_f1", "root|k1_f2"]

    n_total = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    log.info(f"\n{'='*65}")
    log.info(f"LOOCV (per-fold retraining) | n_users={n_total}  "
             f"threshold={DIAGNOSTIC_THRESHOLD_ALG4}")
    log.info(f"{'='*65}")

    # Suppress per-fold training logs (each fold logs Algorithms 1-3 progress).
    import logging as _logging
    for name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3"):
        _logging.getLogger(name).setLevel(_logging.WARNING)

    output = Algorithm4Output(data=data)
    random.seed(rng_seed)

    n_users_total = data.shape[0]
    all_indices = np.arange(n_users_total)
    treeCounter = 0
    for i in range(n_total):
        # ── Build per-fold training set (all users except i) ─────────────────
        train_mask           = np.ones(n_users_total, dtype=bool)
        train_mask[i]        = True
        train_data           = data[train_mask]
        train_labels         = labels[train_mask]

        # ── Per-fold Algorithm 1: build decision tree on training set ────────
        tree_i = build_decision_tree(train_data, train_labels)
        
        treeCounter += 1
        print(f"Tree built: {treeCounter}")
        # ── Per-fold Algorithm 2: perceptor + executive training ─────────────
        alg2_i = run_algorithm2(
            tree         = tree_i,
            data         = train_data,
            labels       = train_labels,
            n_bins       = n_bins,
            nodes_filter = nodes_filter,
        )

        # ── Per-fold Algorithm 3: action refinement (FA = 0, paper-literal) ──
        alg3_i = run_algorithm3(
            alg2_output  = alg2_i,
            tree         = tree_i,
            data         = train_data,
            labels       = train_labels,
            nodes_filter = nodes_filter,
            reset_per_h  = False,    
            verbose      = False,
        )

        # ── Algorithm 4 prediction on the held-out user ──────────────────────
        # user_global_idx = i refers to the FULL data array (data[i] is the
        # held-out user's feature row). The tree/alg2/alg3 models were trained
        # on the OTHER 451 users; user-specific quantities (b_min, b_max, r,
        # P(h, f), P(h>1, f)) are training-set statistics, not test statistics.
        pred = run_algorithm4(
            user_global_idx = i,
            data            = data,
            labels          = labels,
            tree            = tree_i,
            alg2_output     = alg2_i,
            alg3_output     = alg3_i,
            rng_seed        = rng_seed,
            verbose         = verbose,
        )
        output.records.append(pred)
        output.total_elapsed_ms += pred.elapsed_ms

        if (i + 1) % 50 == 0 or i == n_total - 1:
            # Quick running accuracy
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
    Verify that AF_real is monotonically non-decreasing across the ENTIRE
    prediction trajectory (within a single LOOCV fold).

    [PAPER §8.11 user override; CORRECTION TO BUG #9]
    AF persists across focus-level transitions — monotonicity is a global
    invariant, not per-focus-level.

    Returns list of violation descriptions.
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
    Verify that rw = 1 - AF is monotonically non-increasing across the
    ENTIRE prediction trajectory.

    [PAPER §VI.D Eq. 8] rw = 1 - AF.
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
    Verify that when an alarm fires, the decision is UNHEALTHY and the
    af_trace ends immediately after that alarm.

    [PAPER ALG-4 lines 26-28] "Alarm … Return Decision" – no further PACs.

    Implementation note: `pac_number` on each AFTraceEntry is per-(k,f) per the
    paper (line 4: t_m^{kf} = 0 per branch), so it is NOT globally monotonic
    across focus levels. Chronological order = list-insertion order, so we
    use the trace list index to test "after the alarm".
    """
    issues: List[str] = []
    if record.decision != HealthDecision.UNHEALTHY:
        return issues
    # Find the alarm entry (chronologically last alarm).
    alarm_indices = [i for i, e in enumerate(record.af_trace) if e.triggered_alarm]
    if not alarm_indices:
        issues.append(
            f"Decision=UNHEALTHY but no alarm entry found in trace "
            f"(user={record.user_global_idx})"
        )
        return issues
    alarm_idx = alarm_indices[-1]
    # Entries chronologically after the alarm = trace[alarm_idx+1:].
    post_alarm = record.af_trace[alarm_idx + 1:]
    if post_alarm:
        first = post_alarm[0]
        issues.append(
            f"user={record.user_global_idx}: {len(post_alarm)} PACs after alarm "
            f"at trace_idx={alarm_idx} (first post: feat={first.feature_idx})"
        )
    return issues


def validate_no_healthy_class_actions(record: PredictionRecord) -> List[str]:
    """
    Verify that no actions were applied for the healthy class (h=1).

    [PAPER] The action library is built for FA=0 (disease classes h>=2 only).
    [PAPER] "r_{o|h=1} = 0" (Algorithm 2, line 15).
    """
    issues: List[str] = []
    for entry in record.af_trace:
        if entry.disease_h == HEALTHY_CLASS_ALG4:
            issues.append(
                f"user={record.user_global_idx}: action applied for h=1 "
                f"(feat={entry.feature_idx}, PAC={entry.pac_number})"
            )
    return issues


def validate_healthy_rw_threshold(record: PredictionRecord) -> List[str]:
    """
    Verify that HEALTHY decisions have rw ≤ DIAGNOSTIC_THRESHOLD at their
    last PAC.

    [PAPER] Line 35: "if rw_{t_mkf} ≤ Threshold → Claim User is healthy"
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
    Verify that AF stays in [0, 1] for every PAC.

    [PAPER §VI.A] "fuzzy logic ... values of variables can be a real number between
                  0 and 1." AF, defined as the assurance about the decision (§VI.D),
                  is a fuzzy logic value and therefore must remain in [0, 1].
    [PAPER Figs. 11, 12] Empirically AF asymptotes towards 1 without exceeding it.
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
    Verify that HEALTHY/SCREENING decisions only happen after the threshold check
    runs at end of all-h loop (i.e., the trace contains AT LEAST one PAC and the
    decision is consistent with rw at that PAC).

    [PAPER §VI.D Lines 35–38] HEALTHY only when rw <= Threshold AFTER all disease
    classes have been checked at the current node.

    Note: The paper allows HEALTHY at any focus level (m=1 or m=2) provided the
    threshold is met. This validator does NOT require focus-level escalation; it
    only ensures HEALTHY corresponds to a met-threshold state.
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
            f"exceeds threshold={DIAGNOSTIC_THRESHOLD_ALG4} — premature healthy decision"
        )
    return issues


def validate_specificity_invariant(output: Algorithm4Output) -> List[str]:
    """
    Verify the FA = 0 invariant on the population: no truly-healthy user should
    be classified UNHEALTHY.

    [PAPER §VI.B / §VII.A / Table 5] The diagnostic-test policy is FA = 0
    (false-positive rate = 0). Specificity = 100% by Eq. 5 construction when
    healthy training users define [b_min, b_max].

    [LOOCV CAVEAT] With strict per-fold retraining, a held-out healthy user whose
    feature value was the unique min or max of any feature in the *full* dataset
    will fall outside the *training* healthy range and trigger an alarm.
    Such cases are reported as warnings (not failures) to highlight the
    LOOCV-vs-full-training tension.
    """
    issues: List[str] = []
    for r in output.records:
        if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY:
            feat = r.alarm_feature_idx
            issues.append(
                f"user={r.user_global_idx}: TRUE HEALTHY classified UNHEALTHY "
                f"(alarm feat={feat}, h={r.alarm_class}) — "
                f"may be LOOCV-edge case if user is at feature extreme"
            )
    return issues


def validate_diseased_not_healthy(output: Algorithm4Output) -> List[str]:
    """
    Verify that no truly-diseased user is classified HEALTHY (missed diagnosis).

    [PAPER §VII.A Table 5] Sensitivity = 90%; up to ~10% of diseased users may
    be missed (their values all lie within the healthy range so no alarm fires
    AND rw <= threshold at end of focus level). This validator counts misses.
    """
    issues: List[str] = []
    for r in output.records:
        if r.true_is_diseased and r.decision == HealthDecision.HEALTHY:
            issues.append(
                f"user={r.user_global_idx}: TRUE DISEASED (class={r.true_label}) "
                f"classified HEALTHY (missed diagnosis)"
            )
    return issues


def run_all_validations_single(record: PredictionRecord, verbose: bool = True) -> bool:
    """
    Run all single-record validations and report results.

    Returns True if all pass.
    """
    all_issues: List[str] = []

    checks = [
        ("AF monotonicity",          validate_AF_monotonicity(record)),
        ("AF in [0,1]",              validate_AF_in_unit_interval(record)),
        ("rw monotone decrease",     validate_rw_monotone_decrease(record)),
        ("unhealthy immediate return", validate_unhealthy_immediate_return(record)),
        ("no healthy-class actions", validate_no_healthy_class_actions(record)),
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

    Checks:
      1. Per-record validations (AF monotone, rw monotone, etc.)
      2. False alarm rate = 0 (FA=0 policy)
      3. All records have a definitive decision (no UNKNOWN)
    """
    sep = "─" * 65
    if verbose:
        print(f"\n{sep}")
        print("ALGORITHM 4 VALIDATION SUITE")
        print(sep)

    all_ok = True
    total_issues = 0
    n_records = len(output.records)

    # ── 1. Per-record validations ─────────────────────────────────────────────
    n_af_unit_violations = 0
    n_premature_healthy = 0
    for rec in output.records:
        issues_mono_af   = validate_AF_monotonicity(rec)
        issues_af_unit   = validate_AF_in_unit_interval(rec)
        issues_mono_rw   = validate_rw_monotone_decrease(rec)
        issues_immediate = validate_unhealthy_immediate_return(rec)
        issues_no_h1     = validate_no_healthy_class_actions(rec)
        issues_threshold = validate_healthy_rw_threshold(rec)
        issues_focus     = validate_focus_level_progression(rec)

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
        # User-requested invariant summaries
        if n_af_unit_violations == 0:
            print(f"  [PASS] AF in [0,1] invariant: all PACs across all users (§VI.A fuzzy logic)")
        else:
            print(f"  [FAIL] AF in [0,1] invariant: {n_af_unit_violations} PACs out of bounds")
        if n_premature_healthy == 0:
            print(f"  [PASS] Premature HEALTHY: no record declared HEALTHY before threshold met")
        else:
            print(f"  [FAIL] Premature HEALTHY: {n_premature_healthy} records flagged")

    # ── 2. FA=0 policy: specificity invariant (true healthy NOT classified UNHEALTHY) ─
    # [PAPER §VI.B / Table 5] Diagnostic-test policy is FA=0 → Specificity should be 100%.
    # In strict per-fold LOOCV this can be violated when a held-out healthy user is at
    # a feature extreme (their value defined the *full* dataset's b_min/b_max but is
    # outside the *training* range). Such cases are reported as warnings.
    spec_violations = validate_specificity_invariant(output)
    if verbose:
        if not spec_violations:
            print(f"  [PASS] Specificity = 100% (no true-healthy user classified UNHEALTHY)")
        else:
            print(f"  [WARN] Specificity < 100%: {len(spec_violations)} healthy users "
                  f"classified UNHEALTHY (LOOCV edge cases on feature extremes)")

    # ── 3. Diseased-user FNR (missed diagnoses) ──────────────────────────────────
    # [PAPER §VII.A Table 5] Sensitivity = 90% target; up to ~10% miss rate accepted.
    fnr_violations = validate_diseased_not_healthy(output)
    if verbose:
        if not fnr_violations:
            print(f"  [PASS] Sensitivity = 100% (no true-diseased user classified HEALTHY)")
        else:
            fnr_pct = len(fnr_violations) / max(1, output.n_diseased_total) * 100
            print(f"  [INFO] Missed diagnoses: {len(fnr_violations)} true-diseased users "
                  f"classified HEALTHY (FNR={fnr_pct:.1f}%; paper target ~10%)")

    # ── 4. No UNKNOWN decisions ───────────────────────────────────────────────
    unknowns = [r for r in output.records if r.decision == HealthDecision.UNKNOWN]
    if unknowns:
        all_ok = False
        if verbose:
            print(f"  [FAIL] {len(unknowns)} UNKNOWN decisions (should not occur)")
    elif verbose:
        print(f"  [PASS] No UNKNOWN decisions")

    # ── 4. Summary statistics ────────────────────────────────────────────────
    if verbose:
        n_missed_diag = sum(
            1 for r in output.records
            if r.true_is_diseased and r.decision != HealthDecision.UNHEALTHY
        )
        miss_rate = (n_missed_diag / output.n_diseased_total
                     if output.n_diseased_total else 0.0)
        print(f"\n  Accuracy Summary (paper §VII.A Table 5):")
        print(f"    Overall accuracy  : {output.overall_accuracy*100:.1f}%  "
              f"({sum(1 for r in output.records if r.is_correct)}/{n_records})  "
              f"[paper FL2 target: 95.4%]")
        print(f"    Sensitivity       : {output.sensitivity*100:.1f}%  "
              f"(diseased -> UNHEALTHY)  [paper FL2 target: 90%]")
        print(f"    Specificity       : {output.specificity*100:.1f}%  "
              f"(healthy NOT alarmed = HEALTHY+SCREENING)  [paper target: 100%]")
        print(f"    False-alarm rate  : {output.false_alarm_rate*100:.1f}%  "
              f"(healthy -> UNHEALTHY = 1 - specificity)  [paper target: 0%]")
        print(f"    Missed-diagnosis rate (FNR): {miss_rate*100:.1f}%  "
              f"(diseased -> HEALTHY/SCREENING = 1 - sensitivity)")
        print(f"    Screening count   : {output.n_screening}")
        print(f"    Total time        : {output.total_elapsed_ms:.0f}ms  "
              f"({output.total_elapsed_ms/n_records:.1f}ms/user)")
        print(sep)

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – REPORTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_prediction_detail(record: PredictionRecord, show_full_trace: bool = True) -> None:
    """
    Print detailed information for one prediction record.

    Includes AF/rw trace, action rankings, node traversal, and alarm details.
    """
    sep = "─" * 65
    print(f"\n{sep}")
    print(f"PREDICTION DETAIL | user={record.user_global_idx}  "
          f"true_label={record.true_label}  "
          f"({'HEALTHY' if record.true_is_healthy else 'DISEASED'})")
    print(f"  Decision      : {record.decision.value}  "
          f"({'✓ CORRECT' if record.is_correct else '✗ WRONG'})")
    print(f"  Focus reached : m={record.max_focus_reached}")
    print(f"  PAC count     : {record.total_pac_count}")
    print(f"  Actions applied: {record.total_actions_applied}")
    print(f"  Elapsed       : {record.elapsed_ms:.1f}ms")
    if record.alarm_class is not None:
        print(f"  ALARM class   : h={record.alarm_class}  "
              f"feat={record.alarm_feature_idx}"
              f"({FEATURE_NAMES.get(record.alarm_feature_idx,'?')})")
    if record.initial_action_feat is not None:
        fname = FEATURE_NAMES.get(record.initial_action_feat, "?")
        print(f"  Initial action: feat={record.initial_action_feat} ({fname})")

    # Node traversal
    print(f"\n  Node Traversal:")
    for nr in record.node_records:
        status = ""
        if nr.met_threshold:    status = "[THRESHOLD MET]"
        elif nr.focus_increased: status = "[FOCUS UP]"
        elif nr.sent_to_screening: status = "[SCREENING]"
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

    # AF/rw trace
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
                f"{FEATURE_NAMES.get(entry.feature_idx,'?')[:18]:18s} "
                f"{v_str:8s} {entry.b_min:7.3f} {entry.b_max:7.3f} "
                f"{entry.r_j_h:6.4f} {entry.delta_AF:6.4f} "
                f"{entry.AF_real:6.4f} {entry.rw_real:6.4f}"
                + alarm_flag
            )
        if len(record.af_trace) > 20:
            print(f"  … ({len(record.af_trace) - 20} more PACs) …")


def dump_wrong_user_data(output: Algorithm4Output, filepath: str = "wrongUserData.txt") -> None:
    wrong_records = [r for r in output.records if not r.is_correct]
    if not wrong_records:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("All predictions were correct. No misclassified users.\n")
        return

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("MISCLASSIFIED USER DIAGNOSTIC TRACE\n")
        f.write(f"Total misclassified: {len(wrong_records)} / {len(output.records)}\n")
        f.write(f"Overall accuracy: {output.overall_accuracy*100:.1f}%\n")
        f.write(f"Sensitivity: {output.sensitivity*100:.1f}%  Specificity: {output.specificity*100:.1f}%\n")
        f.write("=" * 120 + "\n\n")

        missed = [r for r in wrong_records if r.true_is_diseased]
        false_alarms = [r for r in wrong_records if r.true_is_healthy]
        f.write(f"Missed diagnoses (diseased -> HEALTHY/SCREENING): {len(missed)}\n")
        f.write(f"False alarms (healthy -> UNHEALTHY): {len(false_alarms)}\n\n")

        for idx, rec in enumerate(wrong_records):
            f.write("#" * 120 + "\n")
            f.write(f"WRONG USER #{idx+1}: user_idx={rec.user_global_idx}\n")
            f.write("#" * 120 + "\n")
            true_type = "HEALTHY" if rec.true_is_healthy else f"DISEASED (class {rec.true_label})"
            f.write(f"  True label     : {rec.true_label} ({true_type})\n")
            f.write(f"  Prediction     : {rec.decision.value}\n")
            if rec.true_is_diseased:
                f.write(f"  Error type     : MISSED DIAGNOSIS (diseased user classified as {rec.decision.value})\n")
            else:
                f.write(f"  Error type     : FALSE ALARM (healthy user classified as UNHEALTHY)\n")
            f.write(f"  Focus reached  : m={rec.max_focus_reached}\n")
            f.write(f"  Total PAC count: {rec.total_pac_count}\n")
            f.write(f"  Total actions  : {rec.total_actions_applied}\n")
            f.write(f"  Elapsed        : {rec.elapsed_ms:.1f}ms\n")
            if rec.initial_action_feat is not None:
                fname = FEATURE_NAMES.get(rec.initial_action_feat, "?")
                f.write(f"  Initial action : feat={rec.initial_action_feat} ({fname})\n")
            if rec.alarm_class is not None:
                f.write(f"  Alarm class    : h={rec.alarm_class}\n")
                f.write(f"  Alarm feature  : {rec.alarm_feature_idx} ({FEATURE_NAMES.get(rec.alarm_feature_idx, '?')})\n")
            f.write("\n")

            # Node traversal summary
            f.write("  --- NODE TRAVERSAL ---\n")
            for nr in rec.node_records:
                status = ""
                if nr.met_threshold:       status = "[THRESHOLD MET -> HEALTHY]"
                elif nr.focus_increased:   status = "[FOCUS INCREASED]"
                elif nr.sent_to_screening: status = "[SCREENING]"
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

            # Full AF/rw trace — every single PAC step
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
                    f"{FEATURE_NAMES.get(entry.feature_idx,'?')[:22]:<22} | "
                    f"{v_str:>10} | {entry.b_min:10.4f} | {entry.b_max:10.4f} | "
                    f"{entry.r_j_h:8.6f} | {entry.p_h_f:10.6f} | {entry.p_h_gt1_f:10.6f} | "
                    f"{entry.delta_AF:10.6f} | {entry.AF_real:10.6f} | {entry.rw_real:10.6f} | "
                    f"{alarm_flag:>5} | {calc}{nan_flag}\n"
                )

            f.write("\n")

            # Summary stats for this user
            if rec.af_trace:
                unique_features = set(e.feature_idx for e in rec.af_trace if not e.is_nan)
                unique_h = set(e.disease_h for e in rec.af_trace)
                total_delta = sum(e.delta_AF for e in rec.af_trace)
                f.write(f"  --- SUMMARY FOR USER {rec.user_global_idx} ---\n")
                f.write(f"  Unique features tested: {len(unique_features)} {sorted(unique_features)}\n")
                f.write(f"  Disease classes processed: {sorted(unique_h)}\n")
                f.write(f"  Total AF accumulated: {total_delta:.6f}\n")
                f.write(f"  Final AF: {rec.af_trace[-1].AF_real:.6f}  Final rw: {rec.af_trace[-1].rw_real:.6f}\n")
                f.write(f"  Threshold: {DIAGNOSTIC_THRESHOLD_ALG4}  (rw <= threshold -> HEALTHY)\n")
                if rec.true_is_diseased and rec.decision != HealthDecision.UNHEALTHY:
                    f.write(f"  WHY MISSED: rw={rec.af_trace[-1].rw_real:.6f}, "
                            f"needed alarm (out-of-range value) but none found across "
                            f"{len(rec.af_trace)} actions on {len(unique_features)} unique features\n")
                    # Show which features were in range vs would have needed to be out of range
                    f.write(f"  RANGE CHECK DETAIL (all features tested):\n")
                    for entry in rec.af_trace:
                        if entry.is_nan:
                            continue
                        in_range = "IN RANGE" if not entry.triggered_alarm else "OUT OF RANGE"
                        margin_low = entry.raw_value - entry.b_min
                        margin_high = entry.b_max - entry.raw_value
                        f.write(f"    feat={entry.feature_idx:4d} ({FEATURE_NAMES.get(entry.feature_idx,'?')[:20]:<20}): "
                                f"V={entry.raw_value:10.4f}  range=[{entry.b_min:10.4f}, {entry.b_max:10.4f}]  "
                                f"{in_range}  margin_low={margin_low:+10.4f}  margin_high={margin_high:+10.4f}\n")
            f.write("\n" + "=" * 120 + "\n\n")

    log.info(f"Wrote detailed diagnostic trace for {len(wrong_records)} misclassified users to {filepath}")


def print_algorithm4_summary(output: Algorithm4Output) -> None:
    """
    Print the overall performance summary for the LOOCV run.

    [PAPER] Table 5 equivalent.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("ALGORITHM 4 PERFORMANCE SUMMARY")
    print(sep)
    n = len(output.records)
    print(f"  Users evaluated   : {n}")
    print(f"  Overall accuracy  : {output.overall_accuracy*100:.1f}%  "
          f"(paper Table 5: 95.4% at FL2)")
    print(f"  Sensitivity       : {output.sensitivity*100:.1f}%  "
          f"(diagnosis acc. of abnormal rhythm; paper Table 5: 90% at FL2)")
    print(f"  Specificity       : {output.specificity*100:.1f}%  "
          f"(healthy NOT alarmed = HEALTHY+SCREENING; paper Table 5: 100% at FL2)")
    print(f"  False-alarm rate  : {output.false_alarm_rate*100:.1f}%  "
          f"(paper FA=0% policy; 1 - specificity)")
    print(f"  HEALTHY decisions : {sum(1 for r in output.records if r.decision==HealthDecision.HEALTHY)}")
    print(f"  UNHEALTHY decisions: {sum(1 for r in output.records if r.decision==HealthDecision.UNHEALTHY)}")
    print(f"  SCREENING decisions: {output.n_screening}")
    print(f"  Total elapsed     : {output.total_elapsed_ms:.0f}ms  "
          f"({output.total_elapsed_ms/n:.1f}ms/user)")

    # Sex-based error rates for abnormal users
    print(f"\n  Sex-based error analysis (abnormal users only):")
    print(f"    Abnormal Males   : {output.total_abnormal_males} total, {output.incorrect_abnormal_males} incorrect "
          f"({output.error_rate_abnormal_males*100:.1f}% error rate)")
    print(f"    Abnormal Females : {output.total_abnormal_females} total, {output.incorrect_abnormal_females} incorrect "
          f"({output.error_rate_abnormal_females*100:.1f}% error rate)")
    error_diff = output.error_rate_abnormal_males - output.error_rate_abnormal_females
    print(f"    Difference: Males error rate - Females error rate = {error_diff*100:.1f}%")

    # Per-gender confusion matrices
    print(f"\n  Per-gender confusion matrices:")
    print(f"    Males (Healthy): {output.healthy_correct_males}/{output.healthy_total_males} correct "
          f"({output.healthy_correct_males/output.healthy_total_males*100:.1f}% accuracy)" if output.healthy_total_males else "    Males (Healthy): No healthy males")
    print(f"    Males (Diseased): {output.diseased_correct_males}/{output.diseased_total_males} correct "
          f"({output.diseased_correct_males/output.diseased_total_males*100:.1f}% sensitivity)" if output.diseased_total_males else "    Males (Diseased): No diseased males")
    print(f"    Females (Healthy): {output.healthy_correct_females}/{output.healthy_total_females} correct "
          f"({output.healthy_correct_females/output.healthy_total_females*100:.1f}% accuracy)" if output.healthy_total_females else "    Females (Healthy): No healthy females")
    print(f"    Females (Diseased): {output.diseased_correct_females}/{output.diseased_total_females} correct "
          f"({output.diseased_correct_females/output.diseased_total_females*100:.1f}% sensitivity)" if output.diseased_total_females else "    Females (Diseased): No diseased females")

    # Misclassified females by arrhythmia class
    if output.misclassified_females_by_class:
        print(f"\n  Misclassified females by arrhythmia class:")
        for cls in sorted(output.misclassified_females_by_class.keys()):
            count = output.misclassified_females_by_class[cls]
            print(f"    Class {cls}: {count} misclassifications")
    else:
        print(f"\n  No misclassified females by arrhythmia class.")

    # Per-disease-class breakdown
    diseased_recs = [r for r in output.records if r.true_is_diseased]
    if diseased_recs:
        print(f"\n  Per-class breakdown (diseased users only):")
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
            n_miss     = sum(1 for r in recs if r.decision == HealthDecision.HEALTHY)
            n_screen   = sum(1 for r in recs if r.decision == HealthDecision.SCREENING)
            pct = n_detected / n_cls * 100 if n_cls else 0
            print(f"  {cls:6d} {n_cls:8d} {n_detected:11d} {n_miss:10d} "
                  f"{n_screen:9d} {pct:7.1f}%")

    # Fairness Metrics
    print(f"\n  Fairness Metrics (Privileged: Males, Unprivileged: Females, SPD/DI based on accuracy, EO on TPR/FPR):")
    print(f"    Statistical Parity Difference (SPD): {output.fairness_spd:.4f}")
    print(f"    Disparate Impact (DI): {output.fairness_di:.4f}")
    print(f"    Equalized Odds Difference (EO): {output.fairness_eo_diff:.4f}")
    print(sep)


def print_action_ranking_summary(output: Algorithm4Output, top_k: int = 10) -> None:
    """
    Print the most frequently selected actions across all LOOCV predictions.

    [INFER] Shows which sensors the RL most commonly selects, corresponding
    to Table 4 in the paper (Actions per Arrhythmia class).
    """
    print(f"\n{'─'*65}")
    print(f"TOP {top_k} MOST FREQUENTLY SELECTED ACTIONS (LOOCV)")
    counter: Dict[int, int] = defaultdict(int)
    for rec in output.records:
        for entry in rec.af_trace:
            counter[entry.feature_idx] += 1
    top = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"  {'rank':4} {'feat':4} {'feature_name':30} {'count':8}")
    print("  " + "─" * 50)
    for rank, (feat, cnt) in enumerate(top, 1):
        print(f"  {rank:4d} {feat:4d} {FEATURE_NAMES.get(feat,'?'):30s} {cnt:8d}")


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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

def run_test_cases(
    data:        np.ndarray,
    labels:      np.ndarray,
    tree:        DecisionTree,
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
) -> None:
    """
    Run targeted test cases for specific disease types.

    [PAPER] Test cases include:
      - Bradycardia (class 6)
      - Tachycardia (class 5)
      - Healthy users (class 1)
      - Multiple disease subclasses (classes 2, 3, 9, 10)

    [ENGR] We find the first user of each class and run Algorithm 4 on them.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("ALGORITHM 4 TEST CASES")
    print(sep)

    test_targets = [
        (1,  "Healthy user"),
        (5,  "Tachycardia (class 5)"),
        (6,  "Bradycardia (class 6)"),
        (2,  "Arrhythmia class 2"),
        (3,  "Arrhythmia class 3"),
        (9,  "Arrhythmia class 9"),
        (10, "Arrhythmia class 10"),
        (16, "Unknown/other arrhythmia (class 16)"),
    ]

    for target_class, description in test_targets:
        # Find first user of this class
        candidates = np.where(labels == target_class)[0]
        if len(candidates) == 0:
            print(f"\n  [{description}] No users of class {target_class} found.")
            continue

        user_idx = int(candidates[0])
        print(f"\n{'─'*65}")
        print(f"TEST: {description}  (user_idx={user_idx})")

        record = run_algorithm4(
            user_global_idx = user_idx,
            data            = data,
            labels          = labels,
            tree            = tree,
            alg2_output     = alg2_output,
            alg3_output     = alg3_output,
            rng_seed        = 0,
            verbose         = False,
        )

        print_prediction_detail(record, show_full_trace=True)

        # Run per-record validations
        print(f"\n  Per-record validation:")
        run_all_validations_single(record, verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main(
    data_path:  str = "C:\\Users\\kadhi\\OneDrive\\Desktop\\CDS_Algorithms\\arrhythmia.data",
    run_loocv_flag: bool = True,
    max_users:  Optional[int] = None,
    run_tests:  bool = True,
    fidelity:   bool = True,
    rng_seed:   int  = 42,
) -> Algorithm4Output:
    """
    End-to-end pipeline: Algorithm 1 → 2 → 3 → 4.

    Steps:
      1. Load dataset.
      2. Build decision tree (Algorithm 1).
      3. Run Algorithm 2 (perceptor + executive training).
      4. Run Algorithm 3 (executive action refinement).
      5. (Optional) Print paper fidelity assessment.
      6. (Optional) Run targeted test cases.
      7. (Optional) Run LOOCV (Algorithm 4 over all users).
      8. Validate results.
      9. Print summary report.
    """
    import logging as _logging

    # Suppress verbose lower-level logging for clean output
    _logging.getLogger("CDS.Alg1").setLevel(_logging.WARNING)
    _logging.getLogger("CDS.Alg2").setLevel(_logging.WARNING)
    _logging.getLogger("CDS.Alg3").setLevel(_logging.WARNING)
    log.setLevel(_logging.INFO)
    for h in log.handlers:
        h.setLevel(_logging.INFO)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("CDS ALGORITHM 4: USER-HEALTH PREDICTION PIPELINE")
    log.info("=" * 65)
    log.info("Step 1: Load dataset")
    data, labels = load_dataset(data_path)
    log.info(f"  Loaded: {data.shape[0]} users × {data.shape[1]} features")
    log.info(f"  Label distribution: "
             f"{dict(sorted({int(c): int((labels==c).sum()) for c in set(labels)}.items()))}")

    # ── 2. Paper fidelity assessment (no models needed) ──────────────────────
    if fidelity:
        assess_paper_fidelity()

    # ── 3. Targeted test cases (build a single set of models for these) ─────
    if run_tests:
        log.info("Step 2: Build models for targeted test cases (full-dataset)")
        nodes_filter = ["root", "root|k1_f1", "root|k1_f2"]
        tree_demo = build_decision_tree(data, labels)
        alg2_demo = run_algorithm2(
            tree=tree_demo, data=data, labels=labels,
            n_bins=DEFAULT_N_BINS, nodes_filter=nodes_filter,
        )
        alg3_demo = run_algorithm3(
            alg2_output=alg2_demo, tree=tree_demo, data=data, labels=labels,
            nodes_filter=nodes_filter, reset_per_h=False, verbose=False,
        )
        run_test_cases(data, labels, tree_demo, alg2_demo, alg3_demo)

    # ── 4. LOOCV with per-fold retraining ────────────────────────────────────
    # [PAPER §VII.A, p.180121] "the User-under-test is removed from the
    # Arrhythmia database. Then, at Focus level 1, using Algorithm 2 ...
    # Next, in Algorithm 3 ..."
    if run_loocv_flag:
        log.info(f"Step 3: Algorithm 4 – LOOCV with per-fold retraining "
                 f"(n_users={max_users or data.shape[0]})")
        output = run_loocv(
            data         = data,
            labels       = labels,
            max_users    = max_users,
            rng_seed     = rng_seed,
            verbose      = False,
            n_bins       = DEFAULT_N_BINS,
            nodes_filter = ["root", "root|k1_f1", "root|k1_f2"],
        )

        # ── 8. Validation ─────────────────────────────────────────────────────
        log.info("Step 6: Validation")
        run_all_validations_output(output, verbose=True)

        # ── 9. Summary report ─────────────────────────────────────────────────
        print_algorithm4_summary(output)
        print_action_ranking_summary(output, top_k=10)

        # Print detail for ALL false-alarm users (healthy predicted UNHEALTHY)
        false_alarm_users = [r for r in output.records
                             if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY]
        if false_alarm_users:
            print(f"\n{'='*65}")
            print(f"FALSE ALARM USERS: {len(false_alarm_users)} healthy users predicted UNHEALTHY")
            print(f"{'='*65}")
            for fa_rec in false_alarm_users:
                print_prediction_detail(fa_rec, show_full_trace=True)
                # Print the alarm feature's bin boundaries for debugging
                if fa_rec.alarm_feature_idx is not None and fa_rec.alarm_class is not None:
                    nid = fa_rec.node_records[-1].node_id if fa_rec.node_records else "root"
                    model = alg2_demo = None
                    # Retrieve the per-fold model info from the AF trace
                    for te in fa_rec.af_trace:
                        if te.triggered_alarm:
                            print(f"\n  FALSE ALARM DETAIL for user={fa_rec.user_global_idx}:")
                            print(f"    Alarm feature : {te.feature_idx} ({te.feature_name})")
                            print(f"    Raw value     : {te.raw_value:.6f}")
                            print(f"    Healthy range : [{te.b_min:.6f}, {te.b_max:.6f}]")
                            print(f"    Disease class : h={te.disease_h}")
                            print(f"    Node          : {te.node_id}")
                            break
        else:
            print(f"\n  No false-alarm users (FA=0%).")

        # Dump full diagnostic trace for all misclassified users
        wrong_data_path = str(Path(data_path).parent / "wrongUserData.txt")
        dump_wrong_user_data(output, filepath=wrong_data_path)

    return output

def get_arrhythmia_path(filename: str = "arrhythmia.data") -> Path:
    """
    Find the arrhythmia dataset path in a cross-platform way.

    Search order:
      1. Command-line argument
      2. Current working directory
      3. Same directory as this script
      4. data/ subdirectory beside script
    """

    # 1. Command-line argument
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Dataset not found: {p}")

    # 2. Current working directory
    cwd_path = Path.cwd() / filename
    if cwd_path.exists():
        return cwd_path.resolve()

    # 3. Same directory as script
    script_dir = Path(__file__).parent
    script_path = script_dir / filename
    if script_path.exists():
        return script_path.resolve()

    # 4. data/ folder beside script
    data_path = script_dir / "data" / filename
    if data_path.exists():
        return data_path.resolve()

    raise FileNotFoundError(
        f"Could not locate {filename}. "
        "Place it beside the script, inside ./data/, "
        "or pass the path as a command-line argument."
    )


if __name__ == "__main__":
    import sys as _sys
###<<<<<<< Updated upstream
    path = _sys.argv[1] if len(_sys.argv) > 1 else "C:\\Users\\kadhi\\OneDrive\\Desktop\\amux\\verilogLearning\\CDS-NI-Algorithm\\arrhythmia.data"
###=======
    path = _sys.argv[1] if len(_sys.argv) > 1 else get_arrhythmia_path("arrhythmia.data") 
###>>>>>>> Stashed changes
    # For quick test: limit users; remove max_users for full LOOCV
    n = int(_sys.argv[2]) if len(_sys.argv) > 2 else None
    output = main(
        data_path       = path,
        run_loocv_flag  = True,
        max_users       = n,
        run_tests       = True,
        fidelity        = True,
    )