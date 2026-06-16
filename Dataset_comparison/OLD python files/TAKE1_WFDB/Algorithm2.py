"""
================================================================================
Algorithm 2: CDS Perceptor and Executive Training Mode
REVISED FOR:
  • UCI Arrhythmia tabular dataset
  • WFDB-based datasets via tabularized feature extraction
================================================================================

PURPOSE
-------
Algorithm 2 operates on the DecisionTree produced by Algorithm 1. For each
tree node (focus level m, branching feature k, branch f) it:

  PERCEPTOR TRAINING (lines 3-14)
    • Extracts raw feature data for users in this node
    • Computes global min/max across all users
    • Computes discretization step ΔB
    • Assigns users to bins (B̂ = discretized B)
    • Estimates Bayesian probability tables P(B̂|h), P(h,f),
      P(B̂), P(h|B̂) for all health classes h
    • Extracts the healthy range [b_min, b_max]
    • Computes N_m^{kf} = (b_max - b_min) / ΔB

  EXECUTIVE TRAINING for FA = 0 (lines 15-23)
    • Computes action weights r_{o|h} for each disease class h
    • Registers actions c_{mh|o}(f, FA=0) for informative features

REVISION GOALS
--------------
This revision preserves the original Algorithm 2 structure and functions while:
  1. Removing hard-coded assumptions about:
       • 279 feature columns
       • Arrhythmia-only feature names
       • class 1 / sex feature fixed column assumptions
  2. Making all feature/label/sex metadata derive from DecisionTree.schema
  3. Supporting WFDB-derived tabular matrices produced by Algorithm 1's
     generalized loading layer
  4. Preserving all print / validation / reporting functionality
  5. Keeping function and class names intact whenever possible

INTEGRATION
-----------
Algorithm 2 consumes:
  • DecisionTree from Algorithm 1
  • raw data matrix (N × F)
  • labels vector

It does NOT rebuild the decision tree.
================================================================================
"""

from __future__ import annotations

import logging
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd

RUN_GENDER_FEATURE_ANALYSIS = True
_GENDER_ANALYSIS_ALREADY_RAN = False
SEX_FEATURE_INDEX = 1   # legacy fallback only; schema-driven logic preferred

# ── Import Algorithm 1 components ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from Algorithm1 import (
    DecisionTree, TreeNode, BranchDef, FeatureKind,
    load_dataset, load_dataset_bundle, build_decision_tree, classify_features,
    FEATURE_NAMES, HEALTHY_CLASS, DIAGNOSTIC_THRESHOLD, U_MIN,
    DatasetSchema, DatasetBundle,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger(name: str = "CDS.Alg2") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)-7s | %(name)s | %(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG)
    h.setFormatter(fmt)
    log.addHandler(h)
    log.propagate = False
    return log

log = _build_logger()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N_BINS: int = 10
LAPLACE_EPSILON: float = 0.0
HEALTHY_RANGE_PERCENTILE: float = 100.0

# [LEGACY DEFAULTS] kept for backward compatibility; generalized code now
# derives classes dynamically from the dataset / schema.
ALL_DISEASE_CLASSES: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16)
ALL_CLASSES: Tuple[int, ...] = (1,) + ALL_DISEASE_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS: REWEIGHING PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
try:
    from fairness_config import ENABLE_REWEIGHING
except Exception:
    ENABLE_REWEIGHING = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2B – SCHEMA / METADATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _schema_from_tree(tree: DecisionTree) -> Optional[DatasetSchema]:
    """Return schema from tree if present."""
    return getattr(tree, "schema", None)


def _feature_name(feature_idx: int, schema: Optional[DatasetSchema]) -> str:
    """
    Generalized feature-name lookup.
    """
    if schema is not None:
        return schema.feature_name(feature_idx)
    return FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")


def _healthy_class(schema: Optional[DatasetSchema]) -> int:
    """
    Healthy class from schema or legacy fallback.
    """
    if schema is not None:
        return schema.healthy_class
    return HEALTHY_CLASS


def _sex_feature_index(schema: Optional[DatasetSchema]) -> Optional[int]:
    """
    Sex feature index from schema if available.
    """
    if schema is not None:
        return schema.sex_feature_index
    return SEX_FEATURE_INDEX


def _male_code(schema: Optional[DatasetSchema]) -> Optional[float]:
    if schema is not None:
        return schema.male_code
    return 0.0


def _female_code(schema: Optional[DatasetSchema]) -> Optional[float]:
    if schema is not None:
        return schema.female_code
    return 1.0


def _all_classes_from_labels(labels: np.ndarray) -> List[int]:
    """Return sorted list of all labels present in labels."""
    return sorted(int(x) for x in np.unique(labels).tolist())


def _all_disease_classes_from_labels(labels: np.ndarray,
                                     schema: Optional[DatasetSchema]) -> List[int]:
    """Return all non-healthy classes present."""
    hcls = _healthy_class(schema)
    return sorted(int(x) for x in np.unique(labels).tolist() if int(x) != hcls)


# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS: REWEIGHING PRE-PROCESSING (GENERALIZED)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reweighing_weights(
    labels_valid: np.ndarray,
    sex_valid: np.ndarray,
    protected_value: int = 1,
    healthy_class: int = HEALTHY_CLASS,
) -> np.ndarray:
    """
    Compute Kamiran & Calders (2012) reweighing weights for demographic parity.

    Parameters
    ----------
    labels_valid : class labels for valid users in the current node.
    sex_valid    : sex codes aligned with labels_valid.
    protected_value : which sex code is treated as protected.
    healthy_class : label representing healthy class.

    Returns
    -------
    weights : shape (n_valid,)
    """
    n = len(labels_valid)
    if n == 0:
        return np.ones(0)

    y_binary = (labels_valid == healthy_class).astype(int)

    p_s1 = (sex_valid == protected_value).sum() / n
    p_s0 = 1.0 - p_s1
    p_y1 = y_binary.sum() / n
    p_y0 = 1.0 - p_y1

    weights = np.ones(n, dtype=float)

    for s_val, p_s in [(protected_value, p_s1), (1 - protected_value, p_s0)]:
        for y_val, p_y in [(1, p_y1), (0, p_y0)]:
            mask = (sex_valid == s_val) & (y_binary == y_val)
            p_sy = mask.sum() / n
            if p_sy > 0:
                weights[mask] = (p_s * p_y) / p_sy

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiscretizationResult:
    """
    Result of discretizing one feature for one tree node.
    """
    feature_idx:       int
    feature_name:      str
    node_id:           str
    b_raw_min:         float
    b_raw_max:         float
    delta_b:           float
    bin_edges:         np.ndarray
    n_bins:            int
    bin_assignments:   np.ndarray
    valid_mask:        np.ndarray
    valid_user_rows:   np.ndarray
    bin_counts_all:    np.ndarray
    is_binary:         bool
    is_degenerate:     bool
    _raw_values_valid: Optional[np.ndarray] = None

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def bin_midpoints(self) -> np.ndarray:
        edges = self.bin_edges
        return 0.5 * (edges[:-1] + edges[1:])

    def bin_label(self, b: int) -> str:
        lo = self.bin_edges[b]
        hi = self.bin_edges[b + 1]
        if np.isinf(hi):
            return f"[{lo:.3g}, ∞)"
        return f"[{lo:.3g}, {hi:.3g})"


@dataclass
class BayesianTables:
    """
    Bayesian probability tables for one (node, feature) pair.
    """
    class_labels:      List[int]
    p_bin_given_h:     np.ndarray
    p_h_and_f:         np.ndarray
    p_bin:             np.ndarray
    p_h_given_bin:     np.ndarray
    n_users_per_class: Dict[int, int]

    @property
    def n_bins(self) -> int:
        return self.p_bin.shape[0]

    @property
    def n_classes(self) -> int:
        return len(self.class_labels)

    def class_col(self, h: int) -> int:
        try:
            return self.class_labels.index(h)
        except ValueError:
            return -1

    def p_bin_given_class(self, h: int) -> np.ndarray:
        col = self.class_col(h)
        if col < 0:
            return np.zeros(self.n_bins)
        return self.p_bin_given_h[:, col]

    def prevalence(self, h: int) -> float:
        col = self.class_col(h)
        if col < 0:
            return 0.0
        return float(self.p_h_and_f[col])


@dataclass
class HealthyRangeResult:
    """
    Healthy range [b_min, b_max] for one (node, feature) pair.
    """
    b_min_healthy:      float
    b_max_healthy:      float
    n_kf:               float
    n_healthy_valid:    int
    fallback_used:      bool


@dataclass
class PerceptorModelEntry:
    """
    One entry in the Perceptor Model Library.
    """
    node_id:          str
    focus_level:      int
    branching_feat_k: int
    branch_f:         int
    feature_idx:      int
    feature_name:     str
    n_users_node:     int
    disc:             DiscretizationResult
    bayes:            BayesianTables
    healthy_range:    HealthyRangeResult

    def summary_str(self) -> str:
        hr = self.healthy_range
        d  = self.disc
        return (f"  [{self.node_id}|o={self.feature_idx}({self.feature_name})]  "
                f"users={self.n_users_node}  valid={d.n_valid}  "
                f"bins={d.n_bins}  ΔB={d.delta_b:.3g}  "
                f"range=[{hr.b_min_healthy:.4g}, {hr.b_max_healthy:.4g}]  "
                f"N_kf={hr.n_kf:.2f}  "
                f"{'FALLBACK' if hr.fallback_used else ''}")


@dataclass
class ExecutiveActionEntry:
    """
    One entry in the Executive Action Library (FA = 0 policy).
    """
    node_id:           str
    focus_level:       int
    branching_feat_k:  int
    branch_f:          int
    feature_idx:       int
    feature_name:      str
    disease_class:     int
    action_weight:     float
    p_below_normal:    float
    p_above_normal:    float
    p_h_and_f:         float
    action_label:      str

    def summary_str(self) -> str:
        return (f"  [{self.node_id}|o={self.feature_idx}|h={self.disease_class}]  "
                f"r={self.action_weight:.4f}  "
                f"p_below={self.p_below_normal:.4f}  "
                f"p_above={self.p_above_normal:.4f}  "
                f"prev={self.p_h_and_f:.4f}  "
                f"-> {self.action_label}")


@dataclass
class Algorithm2Output:
    """
    Complete output of Algorithm 2.
    """
    perceptor_library:  List[PerceptorModelEntry]   = field(default_factory=list)
    executive_library:  List[ExecutiveActionEntry]  = field(default_factory=list)
    perceptor_index:    Dict[Tuple[str,int], PerceptorModelEntry] = field(default_factory=dict)
    executive_index:    Dict[Tuple[str,int,int], ExecutiveActionEntry] = field(default_factory=dict)
    n_nodes_processed:  int   = 0
    n_perceptor_entries:int   = 0
    n_executive_entries:int   = 0

    def get_model(self, node_id: str, feature_idx: int) -> Optional[PerceptorModelEntry]:
        return self.perceptor_index.get((node_id, feature_idx))

    def get_action(self, node_id: str, feature_idx: int, disease_h: int) -> Optional[ExecutiveActionEntry]:
        return self.executive_index.get((node_id, feature_idx, disease_h))

    def actions_for_node(self, node_id: str) -> List[ExecutiveActionEntry]:
        return [e for e in self.executive_library if e.node_id == node_id]

    def top_actions(self, node_id: str, disease_h: int, top_k: int = 10) -> List[ExecutiveActionEntry]:
        acts = [e for e in self.executive_library
                if e.node_id == node_id and e.disease_class == disease_h]
        return sorted(acts, key=lambda e: e.action_weight, reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – DISCRETIZATION (Algorithm 2, Lines 4–7)
# ─────────────────────────────────────────────────────────────────────────────

def _n_bins_for_node(n_valid: int, is_binary: bool) -> int:
    """
    Determine bin count using Sturges' rule.
    """
    if is_binary:
        return 2
    if n_valid < 2:
        return 1
    return int(np.ceil(1 + np.log2(n_valid))) + 18


def compute_discretization(feature_idx:   int,
                            node:          TreeNode,
                            data:          np.ndarray,
                            feature_kinds: Dict[int, FeatureKind],
                            n_bins_target: int = DEFAULT_N_BINS,
                            schema:        Optional[DatasetSchema] = None,
                            ) -> Optional[DiscretizationResult]:
    """
    Compute the discretization of feature `feature_idx` for `node`.

    [REVISED] Uses schema-driven feature naming and no fixed feature-count assumptions.
    """
    fname  = _feature_name(feature_idx, schema)
    n_node = len(node.user_indices)

    raw_values  = data[node.user_indices, feature_idx]
    valid_mask  = ~np.isnan(raw_values)
    valid_rows  = np.where(valid_mask)[0]
    valid_vals  = raw_values[valid_mask]
    n_valid     = len(valid_vals)

    if n_valid == 0:
        log.debug(f"    feat {feature_idx}({fname}): ALL NaN in node {node.node_id!r} - skip")
        return None

    b_raw_min = float(valid_vals.min())
    b_raw_max = float(valid_vals.max())
    is_binary = (feature_kinds[feature_idx] == FeatureKind.BINARY)

    is_degenerate = (b_raw_min == b_raw_max)
    if is_degenerate:
        delta_b   = 1.0
        bin_edges = np.array([b_raw_min - 0.5, b_raw_min + 0.5])
        n_bins    = 1
        bin_asgn  = np.zeros(n_valid, dtype=int)
        bin_counts = np.array([n_valid])
        log.debug(f"    feat {feature_idx}({fname}): DEGENERATE (all={b_raw_min}), 1 bin")
        return DiscretizationResult(
            feature_idx=feature_idx, feature_name=fname, node_id=node.node_id,
            b_raw_min=b_raw_min, b_raw_max=b_raw_max, delta_b=delta_b,
            bin_edges=bin_edges, n_bins=n_bins,
            bin_assignments=bin_asgn, valid_mask=valid_mask,
            valid_user_rows=valid_rows, bin_counts_all=bin_counts,
            is_binary=is_binary, is_degenerate=True,
        )

    n_bins_requested = _n_bins_for_node(n_valid, is_binary)

    if is_binary:
        if n_bins_requested <= 1:
            bin_edges = np.array([-0.5, 1.5])
            n_bins = 1
            delta_b = 2.0
        else:
            bin_edges = np.array([-0.5, 0.5, 1.5])
            n_bins = 2
            delta_b = 1.0
    else:
        bin_edges = np.linspace(b_raw_min, b_raw_max, n_bins_requested + 1)
        n_bins = n_bins_requested
        delta_b = (b_raw_max - b_raw_min) / n_bins

    bin_asgn = np.searchsorted(bin_edges[1:], valid_vals, side='right')
    bin_asgn = np.clip(bin_asgn, 0, n_bins - 1)
    bin_counts = np.bincount(bin_asgn, minlength=n_bins)

    log.debug(
        f"    feat {feature_idx}({fname}): "
        f"range=[{b_raw_min:.3g},{b_raw_max:.3g}]  "
        f"dB={delta_b:.3g}  bins={n_bins}  valid={n_valid}/{n_node}  "
        f"{'BIN' if is_binary else 'CONT'}"
    )

    return DiscretizationResult(
        feature_idx=feature_idx, feature_name=fname, node_id=node.node_id,
        b_raw_min=b_raw_min, b_raw_max=b_raw_max, delta_b=delta_b,
        bin_edges=bin_edges, n_bins=n_bins,
        bin_assignments=bin_asgn, valid_mask=valid_mask,
        valid_user_rows=valid_rows, bin_counts_all=bin_counts,
        is_binary=is_binary, is_degenerate=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – BAYESIAN PROBABILITY ESTIMATION (Algorithm 2, Lines 8–10)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bayesian_tables(disc:            DiscretizationResult,
                             node:            TreeNode,
                             labels:          np.ndarray,
                             class_labels:    Optional[List[int]] = None,
                             N_total:         int = 0,
                             laplace_eps:     float = LAPLACE_EPSILON,
                             instance_weights: Optional[np.ndarray] = None,
                             schema:          Optional[DatasetSchema] = None,
                             ) -> BayesianTables:
    """
    Compute the four Bayesian probability tables for one (node, feature) pair.

    [REVISED] Uses dynamic classes / schema-driven healthy class.
    """
    if class_labels is None:
        class_labels = sorted(node.health_dist.keys())

    n_classes = len(class_labels)
    n_bins    = disc.n_bins
    global_indices_valid = node.user_indices[disc.valid_user_rows]
    labels_valid         = labels[global_indices_valid]

    counts_per_class_bin = np.zeros((n_classes, n_bins), dtype=float)
    users_per_class      = np.zeros(n_classes, dtype=int)

    for ci, cls in enumerate(class_labels):
        class_mask = (labels_valid == cls)
        users_per_class[ci] = int(class_mask.sum())
        if users_per_class[ci] == 0:
            continue
        class_bins = disc.bin_assignments[class_mask]
        if instance_weights is not None:
            class_weights = instance_weights[class_mask]
            w_bin_counts = np.zeros(n_bins, dtype=float)
            for b_idx in range(n_bins):
                bin_mask = (class_bins == b_idx)
                w_bin_counts[b_idx] = class_weights[bin_mask].sum()
            counts_per_class_bin[ci] = w_bin_counts
        else:
            counts_per_class_bin[ci] = np.bincount(class_bins, minlength=n_bins)

    p_bin_given_h = np.zeros((n_bins, n_classes), dtype=float)
    for ci, cls in enumerate(class_labels):
        n_cls = users_per_class[ci]
        if n_cls == 0:
            p_bin_given_h[:, ci] = 1.0 / n_bins
        else:
            raw = counts_per_class_bin[ci] + laplace_eps
            p_bin_given_h[:, ci] = raw / raw.sum()

    # [PAPER-like interpretation] branch-local prevalence
    p_h_and_f = np.zeros(n_classes, dtype=float)
    for ci, cls in enumerate(class_labels):
        if node.n_users > 0:
            p_h_and_f[ci] = node.health_dist.get(cls, 0) / node.n_users
        else:
            p_h_and_f[ci] = 0.0

    p_bin = p_bin_given_h @ p_h_and_f
    p_bin_sum = p_bin.sum()
    if p_bin_sum > 0:
        p_bin /= p_bin_sum

    p_h_given_bin = np.zeros((n_bins, n_classes), dtype=float)
    for b in range(n_bins):
        denom = p_bin[b]
        if denom < 1e-300:
            p_h_given_bin[b] = 1.0 / n_classes
        else:
            p_h_given_bin[b] = (p_bin_given_h[b] * p_h_and_f) / denom

    hcls = _healthy_class(schema)
    log.debug(
        f"    Bayesian tables: {n_bins}bins × {n_classes}classes  "
        f"evidence_sum={p_bin_sum:.4f}  "
        f"healthy_prior={p_h_and_f[class_labels.index(hcls)]:.4f}"
        if hcls in class_labels else ""
    )

    return BayesianTables(
        class_labels     = class_labels,
        p_bin_given_h    = p_bin_given_h,
        p_h_and_f        = p_h_and_f,
        p_bin            = p_bin,
        p_h_given_bin    = p_h_given_bin,
        n_users_per_class= dict(zip(class_labels, users_per_class.tolist())),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – HEALTHY RANGE EXTRACTION (Algorithm 2, Lines 11–14)
# ─────────────────────────────────────────────────────────────────────────────

def compute_healthy_range(disc:   DiscretizationResult,
                           node:   TreeNode,
                           labels: np.ndarray,
                           data:   Optional[np.ndarray] = None,
                           schema: Optional[DatasetSchema] = None,
                           ) -> HealthyRangeResult:
    """
    Extract the healthy feature range [b_min, b_max] for one (node, feature).

    [REVISED] Healthy class comes from schema.
    """
    healthy_cls = _healthy_class(schema)

    global_valid  = node.user_indices[disc.valid_user_rows]
    labels_valid  = labels[global_valid]
    healthy_mask  = (labels_valid == healthy_cls)
    n_healthy     = int(healthy_mask.sum())

    fallback_used = False

    if n_healthy > 0:
        raw_vals_valid = disc._raw_values_valid
        healthy_vals   = raw_vals_valid[healthy_mask]

        if HEALTHY_RANGE_PERCENTILE >= 100.0:
            b_min = float(healthy_vals.min())
            b_max = float(healthy_vals.max())
        else:
            tail = (100.0 - HEALTHY_RANGE_PERCENTILE) / 2.0
            b_min = float(np.percentile(healthy_vals, tail))
            b_max = float(np.percentile(healthy_vals, 100.0 - tail))

        n_kf  = (b_max - b_min) / disc.delta_b if disc.delta_b > 0 else 1.0

    else:
        fallback_used = True
        b_min, b_max = disc.b_raw_min, disc.b_raw_max
        n_kf = float(disc.n_bins)
        log.debug(f"    HealthyRange FALLBACK (no healthy users): "
                  f"using full observed range [{b_min}, {b_max}]")

    log.debug(
        f"    HealthyRange: [{b_min:.4g}, {b_max:.4g}]  N_kf={n_kf:.2f}  "
        f"healthy_valid={n_healthy}  {'[FALLBACK]' if fallback_used else ''}"
    )

    return HealthyRangeResult(
        b_min_healthy   = b_min,
        b_max_healthy   = b_max,
        n_kf            = n_kf,
        n_healthy_valid = n_healthy,
        fallback_used   = fallback_used,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – EXECUTIVE ACTION LIBRARY (Algorithm 2, Lines 15–23)
# ─────────────────────────────────────────────────────────────────────────────

def compute_executive_actions(disc:           DiscretizationResult,
                               healthy_range:  HealthyRangeResult,
                               bayes:          BayesianTables,
                               node:           TreeNode,
                               labels:         np.ndarray,
                               data:           Optional[np.ndarray] = None,
                               schema:         Optional[DatasetSchema] = None,
                               ) -> List[ExecutiveActionEntry]:
    """
    Compute executive action weights for one (node, feature) pair.

    [REVISED] Uses schema-driven healthy class and optional schema-driven sex.
    """
    actions = []
    healthy_cls = _healthy_class(schema)
    b_min = healthy_range.b_min_healthy
    b_max = healthy_range.b_max_healthy

    global_valid = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_valid]

    if b_min > b_max:
        return actions

    healthy_mask = (labels_valid == healthy_cls)

    if healthy_mask.sum() == 0:
        return actions

    healthy_bins = disc.bin_assignments[healthy_mask]
    min_healthy_bin = int(healthy_bins.min())
    max_healthy_bin = int(healthy_bins.max())

    instance_weights = None
    sidx = _sex_feature_index(schema)
    fcode = _female_code(schema)

    if ENABLE_REWEIGHING and data is not None and sidx is not None:
        sex_valid = data[global_valid, sidx]
        # Protected group default: female if coded, else 1
        protected_value = int(fcode) if fcode is not None and not np.isnan(fcode) else 1
        instance_weights = compute_reweighing_weights(
            labels_valid=labels_valid,
            sex_valid=sex_valid,
            protected_value=protected_value,
            healthy_class=healthy_cls,
        )

    for cls in bayes.class_labels:
        if cls == healthy_cls:
            continue

        p_h_f = bayes.prevalence(cls)
        if p_h_f <= 0:
            continue

        if instance_weights is not None:
            class_mask = (labels_valid == cls)
            class_weights = instance_weights[class_mask]
            class_bins = disc.bin_assignments[class_mask]
            w_total = class_weights.sum()
            if w_total <= 0:
                continue
            w_bin_counts = np.zeros(disc.n_bins, dtype=float)
            for b_idx in range(disc.n_bins):
                bin_mask = (class_bins == b_idx)
                w_bin_counts[b_idx] = class_weights[bin_mask].sum()
            p_bin_h_weighted = w_bin_counts / w_total
            p_below = float(p_bin_h_weighted[:min_healthy_bin].sum()) if min_healthy_bin > 0 else 0.0
            p_above = float(p_bin_h_weighted[max_healthy_bin + 1:].sum()) if max_healthy_bin < disc.n_bins - 1 else 0.0
        else:
            p_bin_h = bayes.p_bin_given_class(cls)
            p_below = float(p_bin_h[:min_healthy_bin].sum()) if min_healthy_bin > 0 else 0.0
            p_above = float(p_bin_h[max_healthy_bin + 1:].sum()) if max_healthy_bin < disc.n_bins - 1 else 0.0

        r_o_h = p_below + p_above

        if (p_below > 0 or p_above > 0) and p_h_f > 0:
            action_label = f"sensor_for_feat_{disc.feature_idx}({disc.feature_name})"
            entry = ExecutiveActionEntry(
                node_id          = node.node_id,
                focus_level      = node.focus_level,
                branching_feat_k = node.branching_feat_k,
                branch_f         = node.branch_f,
                feature_idx      = disc.feature_idx,
                feature_name     = disc.feature_name,
                disease_class    = cls,
                action_weight    = r_o_h,
                p_below_normal   = p_below,
                p_above_normal   = p_above,
                p_h_and_f        = p_h_f,
                action_label     = action_label,
            )
            actions.append(entry)

    return actions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – ALGORITHM 2 CORE: PER-NODE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm2_for_node(node:          TreeNode,
                             data:          np.ndarray,
                             labels:        np.ndarray,
                             feature_kinds: Dict[int, FeatureKind],
                             class_labels:  Optional[List[int]] = None,
                             n_bins:        int = DEFAULT_N_BINS,
                             schema:        Optional[DatasetSchema] = None,
                             ) -> Tuple[List[PerceptorModelEntry],
                                        List[ExecutiveActionEntry]]:
    """
    Run Algorithm 2 for a single tree node.

    [REVISED] Schema-aware.
    """
    if class_labels is None:
        class_labels = sorted(node.health_dist.keys())

    perceptor_entries: List[PerceptorModelEntry]  = []
    executive_entries: List[ExecutiveActionEntry] = []

    n_feat = len(node.feature_indices)
    log.debug(
        f"\n  Node {node.node_id!r}: m={node.focus_level}  "
        f"users={node.n_users}  features={n_feat}  "
        f"classes={list(class_labels)}"
    )

    node_reweigh_weights = None
    sidx = _sex_feature_index(schema)
    if ENABLE_REWEIGHING and data is not None and sidx is not None:
        node_global_indices = node.user_indices
        node_labels = labels[node_global_indices]
        node_sex = data[node_global_indices, sidx]
        protected_value = int(_female_code(schema)) if _female_code(schema) is not None else 1
        node_reweigh_weights = compute_reweighing_weights(
            labels_valid=node_labels,
            sex_valid=node_sex,
            protected_value=protected_value,
            healthy_class=_healthy_class(schema),
        )

    for feature_o in node.feature_indices:
        disc = compute_discretization(
            feature_idx   = feature_o,
            node          = node,
            data          = data,
            feature_kinds = feature_kinds,
            n_bins_target = n_bins,
            schema        = schema,
        )
        if disc is None:
            continue

        raw_values_node  = data[node.user_indices, feature_o]
        raw_values_valid = raw_values_node[disc.valid_mask]
        disc._raw_values_valid = raw_values_valid

        valid_weights = None
        if node_reweigh_weights is not None:
            valid_weights = node_reweigh_weights[disc.valid_user_rows]

        bayes = compute_bayesian_tables(
            disc             = disc,
            node             = node,
            labels           = labels,
            N_total          = len(labels),
            class_labels     = list(class_labels),
            instance_weights = valid_weights,
            schema           = schema,
        )

        healthy_range = compute_healthy_range(
            disc   = disc,
            node   = node,
            labels = labels,
            schema = schema,
        )

        run_gender_feature_analysis_once(
            data=data,
            labels=labels,
            gender_feature_idx=sidx if sidx is not None else 1,
            feature_names=schema.feature_names if schema is not None else FEATURE_NAMES,
            healthy_class=_healthy_class(schema),
            male_code=_male_code(schema),
            female_code=_female_code(schema),
        )

        entry = PerceptorModelEntry(
            node_id          = node.node_id,
            focus_level      = node.focus_level,
            branching_feat_k = node.branching_feat_k,
            branch_f         = node.branch_f,
            feature_idx      = feature_o,
            feature_name     = disc.feature_name,
            n_users_node     = node.n_users,
            disc             = disc,
            bayes            = bayes,
            healthy_range    = healthy_range,
        )
        perceptor_entries.append(entry)

        act_entries = compute_executive_actions(
            disc          = disc,
            healthy_range = healthy_range,
            bayes         = bayes,
            node          = node,
            labels        = labels,
            data          = data,
            schema        = schema,
        )
        executive_entries.extend(act_entries)

    log.debug(
        f"  Node {node.node_id!r}: "
        f"perceptor_entries={len(perceptor_entries)}  "
        f"executive_entries={len(executive_entries)}"
    )
    return perceptor_entries, executive_entries


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – ALGORITHM 2 MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm2(tree:          DecisionTree,
                   data:          np.ndarray,
                   labels:        np.ndarray,
                   n_bins:        int = DEFAULT_N_BINS,
                   nodes_filter:  Optional[List[str]] = None,
                   class_labels:  Optional[List[int]] = None,
                   ) -> Algorithm2Output:
    """
    Run Algorithm 2 for all nodes in the decision tree.

    [REVISED] Health classes are inferred from labels if not supplied.
    """
    schema = _schema_from_tree(tree)

    log.info("=" * 72)
    log.info("Algorithm 2: CDS Perceptor and Executive Training")
    log.info(f"  n_bins={n_bins}  threshold={tree.threshold}")
    log.info(f"  tree_depth={tree.depth()}  total_nodes={tree.count_nodes()}")
    if schema is not None:
        log.info(f"  dataset={schema.dataset_name} ({schema.source_format})")
    log.info("=" * 72)

    feature_kinds = tree.feature_kinds

    if class_labels is None:
        class_labels = _all_classes_from_labels(labels)
    log.info(f"  Health classes: {class_labels}")

    output = Algorithm2Output()
    nodes_processed = 0

    for m in sorted(tree.nodes_by_level.keys()):
        nodes_at_m = tree.nodes_by_level[m]
        log.info(f"\n{'─'*60}")
        log.info(f"Processing level m={m}: {len(nodes_at_m)} nodes …")

        for node in nodes_at_m:
            if nodes_filter and node.node_id not in nodes_filter:
                continue

            perc_entries, exec_entries = run_algorithm2_for_node(
                node          = node,
                data          = data,
                labels        = labels,
                feature_kinds = feature_kinds,
                class_labels  = list(class_labels),
                n_bins        = n_bins,
                schema        = schema,
            )

            for e in perc_entries:
                output.perceptor_library.append(e)
                output.perceptor_index[(e.node_id, e.feature_idx)] = e

            for e in exec_entries:
                output.executive_library.append(e)
                output.executive_index[(e.node_id, e.feature_idx, e.disease_class)] = e

            nodes_processed += 1

        log.info(f"  Level m={m}: cumulative entries -> "
                 f"perceptor={len(output.perceptor_library)}  "
                 f"executive={len(output.executive_library)}")

    output.n_nodes_processed   = nodes_processed
    output.n_perceptor_entries = len(output.perceptor_library)
    output.n_executive_entries = len(output.executive_library)

    log.info(f"\n{'='*72}")
    log.info(f"Algorithm 2 complete.")
    log.info(f"  Nodes processed       : {nodes_processed}")
    log.info(f"  Perceptor entries     : {output.n_perceptor_entries}")
    log.info(f"  Executive entries     : {output.n_executive_entries}")
    log.info("=" * 72)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – VALIDATION FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

def validate_probability_normalisation(output: Algorithm2Output,
                                        tol: float = 1e-6) -> Dict[str, int]:
    """
    Verify probability normalisation invariants for all perceptor entries.
    """
    counts = {"V1_pass": 0, "V1_fail": 0,
              "V2_pass": 0, "V2_fail": 0,
              "V3_pass": 0, "V3_fail": 0}

    for entry in output.perceptor_library:
        b = entry.bayes

        col_sums = b.p_bin_given_h.sum(axis=0)
        for ci, s in enumerate(col_sums):
            if abs(s - 1.0) < tol:
                counts["V1_pass"] += 1
            else:
                counts["V1_fail"] += 1
                log.debug(f"V1 FAIL: {entry.node_id} feat={entry.feature_idx} "
                          f"class={b.class_labels[ci]}  sum={s:.6f}")

        ev_sum = b.p_bin.sum()
        if abs(ev_sum - 1.0) < tol:
            counts["V2_pass"] += 1
        else:
            counts["V2_fail"] += 1
            log.debug(f"V2 FAIL: {entry.node_id} feat={entry.feature_idx} "
                      f"evidence_sum={ev_sum:.6f}")

        row_sums = b.p_h_given_bin.sum(axis=1)
        for b_idx, s in enumerate(row_sums):
            if abs(s - 1.0) < tol:
                counts["V3_pass"] += 1
            else:
                counts["V3_fail"] += 1

    return counts


def validate_fa_zero_invariant(output: Algorithm2Output,
                                 data:   np.ndarray,
                                 labels: np.ndarray) -> Dict[str, int]:
    """
    Verify the FA = 0 invariant for all perceptor entries.
    """
    counts = {"FA0_pass": 0, "FA0_fail": 0, "FA0_fallback": 0}

    for entry in output.perceptor_library:
        hr   = entry.healthy_range
        disc = entry.disc

        if hr.fallback_used:
            counts["FA0_fallback"] += 1
            continue

        raw_vals = getattr(disc, '_raw_values_valid', None)
        if raw_vals is None:
            counts["FA0_pass"] += 1
            continue

        n_healthy_valid = entry.healthy_range.n_healthy_valid
        if n_healthy_valid == 0:
            counts["FA0_fallback"] += 1
            continue

        counts["FA0_pass"] += 1

    return counts


def validate_bayesian_consistency(output: Algorithm2Output,
                                   tol: float = 1e-6) -> Dict[str, int]:
    """
    Verify Bayesian consistency: Eq. 4 holds for each (entry, bin).
    """
    counts = {"bayes_pass": 0, "bayes_fail": 0}
    for entry in output.perceptor_library:
        b = entry.bayes
        for b_idx in range(b.n_bins):
            ev = b.p_bin[b_idx]
            if ev < 1e-300:
                continue
            recomputed = (b.p_bin_given_h[b_idx] * b.p_h_and_f) / ev
            stored     = b.p_h_given_bin[b_idx]
            max_diff   = float(np.abs(recomputed - stored).max())
            if max_diff < tol:
                counts["bayes_pass"] += 1
            else:
                counts["bayes_fail"] += 1
                log.debug(f"Bayes FAIL: {entry.node_id} feat={entry.feature_idx} "
                          f"bin={b_idx}  max_diff={max_diff:.2e}")
    return counts


def validate_healthy_range_coverage(output: Algorithm2Output) -> Dict[str, int]:
    """
    Verify that the healthy range is non-degenerate and contained in [B_min, B_max].
    """
    counts = {"range_ok": 0, "range_inverted": 0, "range_outside": 0}
    for entry in output.perceptor_library:
        hr = entry.healthy_range
        d  = entry.disc
        if hr.b_min_healthy > hr.b_max_healthy:
            counts["range_inverted"] += 1
            log.debug(f"Range INVERTED: {entry.node_id} feat={entry.feature_idx} "
                      f"[{hr.b_min_healthy},{hr.b_max_healthy}]")
        elif hr.b_min_healthy < d.b_raw_min - 1e-9 or hr.b_max_healthy > d.b_raw_max + 1e-9:
            counts["range_outside"] += 1
            log.debug(f"Range OUTSIDE: {entry.node_id} feat={entry.feature_idx} "
                      f"healthy=[{hr.b_min_healthy},{hr.b_max_healthy}] "
                      f"global=[{d.b_raw_min},{d.b_raw_max}]")
        else:
            counts["range_ok"] += 1
    return counts


def validate_action_weights(output: Algorithm2Output,
                             healthy_class: int = HEALTHY_CLASS) -> Dict[str, int]:
    """
    Verify action weight invariants.
    """
    counts = {"weight_ok": 0, "weight_bad": 0, "healthy_in_exec": 0}
    for e in output.executive_library:
        if e.disease_class == healthy_class:
            counts["healthy_in_exec"] += 1
        r = e.action_weight
        if 0 <= r <= 1.0 + 1e-9 and abs(r - (e.p_below_normal + e.p_above_normal)) < 1e-9:
            counts["weight_ok"] += 1
        else:
            counts["weight_bad"] += 1
            log.debug(f"Weight FAIL: {e.node_id} feat={e.feature_idx} h={e.disease_class} "
                      f"r={r} != {e.p_below_normal+e.p_above_normal}")
    return counts


def run_all_validations(output: Algorithm2Output,
                         data:   np.ndarray,
                         labels: np.ndarray,
                         schema: Optional[DatasetSchema] = None) -> None:
    """
    Run all validation checks and print a summary report.
    """
    print("\n" + "=" * 72)
    print("ALGORITHM 2: VALIDATION REPORT")
    print("=" * 72)

    checks = {
        "V1–V3: Probability Normalisation": validate_probability_normalisation(output),
        "Bayesian Eq.4 Consistency":         validate_bayesian_consistency(output),
        "FA=0 Invariant":                    validate_fa_zero_invariant(output, data, labels),
        "Healthy Range Coverage":            validate_healthy_range_coverage(output),
        "Action Weight Bounds":              validate_action_weights(output, healthy_class=_healthy_class(schema)),
    }
    all_ok = True
    for name, result in checks.items():
        fails = sum(v for k,v in result.items() if "fail" in k or "bad" in k
                    or "inverted" in k or "outside" in k or "healthy_in_exec" in k)
        status = "✓ PASS" if fails == 0 else f"✗ {fails} FAIL(s)"
        print(f"  {name:45s}  {status}")
        if fails > 0:
            all_ok = False
        for k, v in result.items():
            print(f"      {k}: {v}")
    print("─" * 72)
    print(f"  Overall: {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – INSPECTION AND REPORTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def print_algorithm2_summary(output: Algorithm2Output) -> None:
    """Print high-level summary of Algorithm 2 output."""
    print("\n" + "=" * 72)
    print("ALGORITHM 2: OUTPUT SUMMARY")
    print("=" * 72)
    print(f"  Nodes processed:     {output.n_nodes_processed}")
    print(f"  Perceptor entries:   {output.n_perceptor_entries}")
    print(f"  Executive entries:   {output.n_executive_entries}")

    if output.executive_library:
        weights = [e.action_weight for e in output.executive_library]
        weights_arr = np.array(weights)
        print(f"\n  Action weight distribution:")
        print(f"    min={weights_arr.min():.4f}  max={weights_arr.max():.4f}  "
              f"mean={weights_arr.mean():.4f}  median={np.median(weights_arr):.4f}")
        bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
        hist, _ = np.histogram(weights_arr, bins=bins)
        for i in range(len(hist)):
            print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:5d} entries")
    print("=" * 72)


def print_perceptor_entry(entry: PerceptorModelEntry,
                           show_full_tables: bool = False) -> None:
    """Print detailed information for one perceptor model entry."""
    disc = entry.disc
    hr   = entry.healthy_range
    b    = entry.bayes

    print(f"\n{'─'*60}")
    print(f"PERCEPTOR MODEL ENTRY")
    print(f"  Node     : {entry.node_id!r}")
    print(f"  m / k / f: {entry.focus_level} / {entry.branching_feat_k} / {entry.branch_f}")
    print(f"  Feature  : {entry.feature_idx} ({entry.feature_name})")
    print(f"  Users    : {entry.n_users_node} total  "
          f"/ {disc.n_valid} valid  "
          f"/ {hr.n_healthy_valid} healthy-valid")

    print(f"\n  Discretization:")
    print(f"    B_min={disc.b_raw_min:.4g}  B_max={disc.b_raw_max:.4g}  "
          f"ΔB={disc.delta_b:.4g}  n_bins={disc.n_bins}  "
          f"{'BINARY' if disc.is_binary else 'CONTINUOUS'}  "
          f"{'DEGENERATE' if disc.is_degenerate else ''}")
    print(f"    Bin counts: {disc.bin_counts_all.tolist()}")

    print(f"\n  Healthy Range (FA=0):")
    print(f"    b_min={hr.b_min_healthy:.4g}  b_max={hr.b_max_healthy:.4g}  "
          f"N_kf={hr.n_kf:.2f}  {'[FALLBACK]' if hr.fallback_used else ''}")

    print(f"\n  Prevalence P(h,f) per class:")
    for ci, cls in enumerate(b.class_labels):
        print(f"    h={cls:2d}: P(h,f)={b.p_h_and_f[ci]:.4f}  "
              f"n_users={b.n_users_per_class.get(cls,0)}")

    if show_full_tables:
        print(f"\n  P(B̂|h) table (rows=bins, cols=classes):")
        header = "  bin   " + "".join(f"  h={c:2d}" for c in b.class_labels)
        print(header)
        for b_idx in range(min(disc.n_bins, 20)):
            row = f"  {b_idx:3d}   " + "".join(
                f"  {b.p_bin_given_h[b_idx, ci]:6.4f}" for ci in range(b.n_classes)
            )
            print(row)

        print(f"\n  Evidence P(B̂):")
        print("  " + "  ".join(f"{v:.4f}" for v in b.p_bin))


def print_executive_top_actions(output:     Algorithm2Output,
                                  node_id:    str,
                                  disease_h:  int,
                                  top_k:      int = 15) -> None:
    """Print top-k actions for (node, disease) sorted by action weight."""
    acts = output.top_actions(node_id, disease_h, top_k)
    print(f"\n{'─'*60}")
    print(f"TOP {top_k} ACTIONS  |  node={node_id!r}  |  disease h={disease_h}")
    if not acts:
        print("  (no actions found)")
        return
    print(f"  {'rank':4} {'feat':4} {'feature_name':25} {'r_{o|h}':8} "
          f"{'p_below':8} {'p_above':8} {'P(h,f)':8}")
    print("  " + "─" * 73)
    for rank, act in enumerate(acts, 1):
        print(f"  {rank:4d} {act.feature_idx:4d} {act.feature_name:25s} "
              f"{act.action_weight:8.4f} {act.p_below_normal:8.4f} "
              f"{act.p_above_normal:8.4f} {act.p_h_and_f:8.4f}")


def print_normal_ranges_table(output:    Algorithm2Output,
                                node_id:  str,
                                max_rows: int = 30) -> None:
    """Print the perceptor model normal-range table for a node."""
    entries = [e for e in output.perceptor_library if e.node_id == node_id]
    entries.sort(key=lambda e: e.feature_idx)
    print(f"\n{'─'*72}")
    print(f"PERCEPTOR MODEL LIBRARY  |  node={node_id!r}  "
          f"({len(entries)} features)")
    print(f"  {'feat':4} {'name':25} {'b_min_h':10} {'b_max_h':10} "
          f"{'N_kf':6} {'ΔB':8} {'n_bins':6} {'valid':6}")
    print("  " + "─" * 72)
    for e in entries[:max_rows]:
        hr = e.healthy_range
        d  = e.disc
        print(f"  {e.feature_idx:4d} {e.feature_name:25s} "
              f"{hr.b_min_healthy:10.4g} {hr.b_max_healthy:10.4g} "
              f"{hr.n_kf:6.2f} {d.delta_b:8.4g} {d.n_bins:6d} {d.n_valid:6d}")
    if len(entries) > max_rows:
        print(f"  … ({len(entries) - max_rows} more entries) …")


def print_executive_summary_by_disease(output:   Algorithm2Output,
                                         node_id:  str) -> None:
    """Print a per-disease summary of available actions for a node."""
    acts   = output.actions_for_node(node_id)
    by_cls = defaultdict(list)
    for a in acts:
        by_cls[a.disease_class].append(a)
    print(f"\n{'─'*60}")
    print(f"EXECUTIVE ACTION LIBRARY SUMMARY  |  node={node_id!r}")
    print(f"  {'h':3} {'n_actions':10} {'best_feat':30} {'best_r':8}")
    print("  " + "─" * 56)
    for cls in sorted(by_cls.keys()):
        cls_acts = sorted(by_cls[cls], key=lambda a: a.action_weight, reverse=True)
        n_act = len(cls_acts)
        if n_act > 0:
            best = cls_acts[0]
            best_str = f"{best.feature_name}(o={best.feature_idx})"
        else:
            best_str = "—"
            best     = None
        print(f"  {cls:3d} {n_act:10d} {best_str:30s} "
              f"{best.action_weight:.4f}" if best else f"  {cls:3d} {n_act:10d}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main(
    data_path: Optional[str] = str(Path(__file__).parent / "data" / "arrhythmia.data"),
    run_full:  bool = False,
    dataset_type: str = "arrhythmia",
    wfdb_db_dir: Optional[str] = None,
    wfdb_record_names: Optional[List[str]] = None,
    wfdb_label_map: Optional[Dict[str, int]] = None,
    wfdb_pn_dir: Optional[str] = None,
) -> Algorithm2Output:
    """
    End-to-end execution: Algorithm 1 -> Algorithm 2 -> Validation -> Reports.

    Supports:
      • UCI Arrhythmia
      • WFDB tabularized datasets
    """
    import logging as _logging
    _logging.getLogger("CDS.Alg1").setLevel(_logging.WARNING)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    log.info("Step 1: Load dataset")
    bundle = load_dataset_bundle(
        path=data_path,
        dataset_type=dataset_type,
        wfdb_db_dir=wfdb_db_dir,
        wfdb_record_names=wfdb_record_names,
        wfdb_label_map=wfdb_label_map,
        wfdb_pn_dir=wfdb_pn_dir,
    )
    data, labels, schema = bundle.data, bundle.labels, bundle.schema
    log.info(f"  Loaded: {data.shape[0]} users × {data.shape[1]} features")
    log.info(f"  Dataset: {schema.dataset_name} ({schema.source_format})")

    # ── 2. Run Algorithm 1 ────────────────────────────────────────────────────
    log.info("Step 2: Build Decision Tree (Algorithm 1)")
    tree = build_decision_tree(data, labels, schema=schema)
    log.info(f"  Tree: depth={tree.depth()}  nodes={tree.count_nodes()}")

    # ── 3. Select nodes to process ────────────────────────────────────────────
    if run_full:
        nodes_filter = None
        log.info("Step 3: Processing ALL nodes in the tree")
    else:
        key_nodes = ["root"]
        if schema.sex_feature_index is not None:
            sex_nodes = [n.node_id for n in tree.nodes_by_level.get(2, [])
                         if n.branching_feat_k == schema.sex_feature_index]
            key_nodes.extend(sex_nodes[:2])
        nodes_filter = [n for n in key_nodes if n in tree.all_nodes]
        log.info(f"Step 3: Processing {len(nodes_filter)} key nodes: {nodes_filter}")

    # ── 4. Run Algorithm 2 ────────────────────────────────────────────────────
    log.info("Step 4: Algorithm 2 – Perceptor and Executive Training")
    output = run_algorithm2(
        tree         = tree,
        data         = data,
        labels       = labels,
        n_bins       = DEFAULT_N_BINS,
        nodes_filter = nodes_filter,
    )

    # ── 5. Validation ─────────────────────────────────────────────────────────
    log.info("Step 5: Validation")
    run_all_validations(output, data, labels, schema=schema)

    # ── 6. Reporting ──────────────────────────────────────────────────────────
    log.info("Step 6: Reporting")
    print_algorithm2_summary(output)
    print_normal_ranges_table(output, "root", max_rows=20)

    if schema.sex_feature_index is not None:
        for node_id in [n for n in (nodes_filter or []) if n != "root"]:
            if node_id in [e.node_id for e in output.perceptor_library]:
                print_normal_ranges_table(output, node_id, max_rows=15)
                print_executive_summary_by_disease(output, node_id)

    for node_id in (["root"] if not run_full else ["root"]):
        disease_candidates = _all_disease_classes_from_labels(labels, schema)
        for h in disease_candidates[:4]:
            print_executive_top_actions(output, node_id, h, top_k=10)

    # Detailed perceptor entry for sex feature or feature 0 fallback
    target_feat = schema.sex_feature_index if schema.sex_feature_index is not None else 0
    entry = output.get_model("root", target_feat)
    if entry:
        print(f"\n{'='*60}")
        print(f"DETAILED ENTRY: Feature {target_feat} ({entry.feature_name}) at ROOT")
        print_perceptor_entry(entry, show_full_tables=True)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL ONE-SHOT GENDER FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_gender_feature_analysis_once(
    data: np.ndarray,
    labels: np.ndarray,
    gender_feature_idx: int,
    feature_names: Dict[int, str],
    healthy_class: int = 1,
    significance_threshold: float = 0.05,
    male_code: Optional[float] = 0.0,
    female_code: Optional[float] = 1.0,
) -> None:
    """
    One-shot global analysis:
      • Computes male/female feature importance separately
      • Computes permutation-importance differences
      • Prints statistically separated features
      • Prints healthy ranges by gender

    [REVISED] Uses schema-driven male/female coding where possible.
    """
    global RUN_GENDER_FEATURE_ANALYSIS
    global _GENDER_ANALYSIS_ALREADY_RAN

    if not RUN_GENDER_FEATURE_ANALYSIS:
        return
    if _GENDER_ANALYSIS_ALREADY_RAN:
        return

    _GENDER_ANALYSIS_ALREADY_RAN = True
    RUN_GENDER_FEATURE_ANALYSIS = False

    # If sex feature is unavailable / invalid, skip quietly.
    if gender_feature_idx is None or gender_feature_idx < 0 or gender_feature_idx >= data.shape[1]:
        print("\n" + "=" * 80)
        print("GLOBAL GENDER FEATURE ANALYSIS")
        print("=" * 80)
        print("[INFO] No valid sex/gender feature available; skipping analysis.")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("GLOBAL GENDER FEATURE ANALYSIS")
    print("=" * 80)

    try:
        from sklearn.inspection import permutation_importance
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("[ERROR] sklearn not installed.")
        return

    gender_values = data[:, gender_feature_idx]
    valid_gender_mask = ~np.isnan(gender_values)

    # Generalized coding
    mcode = male_code if male_code is not None else 0.0
    fcode = female_code if female_code is not None else 1.0

    male_mask   = (gender_values == mcode) & valid_gender_mask
    female_mask = (gender_values == fcode) & valid_gender_mask

    if female_mask.sum() == 0 or male_mask.sum() == 0:
        print("[WARN] Missing male/female populations.")
        return

    feature_indices = [
        i for i in range(data.shape[1])
        if i != gender_feature_idx
    ]

    X = data[:, feature_indices]
    # More robust median fill column-wise
    X_filled = X.copy()
    for col in range(X_filled.shape[1]):
        col_vals = X_filled[:, col]
        mask = np.isnan(col_vals)
        if mask.any():
            median_val = np.nanmedian(col_vals)
            if np.isnan(median_val):
                median_val = 0.0
            col_vals[mask] = median_val
            X_filled[:, col] = col_vals

    y = (labels != healthy_class).astype(int)

    X_male = X_filled[male_mask]
    y_male = y[male_mask]

    X_female = X_filled[female_mask]
    y_female = y[female_mask]

    model_male = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model_female = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    if len(X_male) == 0 or len(X_female) == 0:
        print("[WARN] Insufficient male/female samples for analysis.")
        return

    model_male.fit(X_male, y_male)
    model_female.fit(X_female, y_female)

    perm_male = permutation_importance(
        model_male,
        X_male,
        y_male,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    perm_female = permutation_importance(
        model_female,
        X_female,
        y_female,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    male_importance = perm_male.importances_mean
    female_importance = perm_female.importances_mean
    diff = np.abs(male_importance - female_importance)

    ranked = np.argsort(diff)[::-1]

    print("\nTOP FEATURES WITH GENDER IMPORTANCE DIFFERENCES")
    print("-" * 80)
    for rank_idx in ranked[:25]:
        feat_idx = feature_indices[rank_idx]
        fname = feature_names.get(feat_idx, f"feature_{feat_idx}")
        print(
            f"{fname:<40} "
            f"Male={male_importance[rank_idx]:.6f}  "
            f"Female={female_importance[rank_idx]:.6f}  "
            f"Diff={diff[rank_idx]:.6f}"
        )

    print("\n" + "-" * 80)
    print("HEALTHY RANGES BY GENDER")
    print("-" * 80)

    healthy_mask = (labels == healthy_class)
    healthy_male_mask = healthy_mask & male_mask
    healthy_female_mask = healthy_mask & female_mask

    for feat_idx in feature_indices:
        col = data[:, feat_idx]

        male_vals = col[healthy_male_mask]
        female_vals = col[healthy_female_mask]

        male_vals = male_vals[~np.isnan(male_vals)]
        female_vals = female_vals[~np.isnan(female_vals)]

        if len(male_vals) == 0 or len(female_vals) == 0:
            continue

        male_min = np.min(male_vals)
        male_max = np.max(male_vals)
        female_min = np.min(female_vals)
        female_max = np.max(female_vals)

        fname = feature_names.get(feat_idx, f"feature_{feat_idx}")

        print(
            f"{fname:<40} "
            f"M:[{male_min:.3f}, {male_max:.3f}]  "
            f"F:[{female_min:.3f}, {female_max:.3f}]"
        )

    print("\n[INFO] Gender feature analysis completed.")
    print("=" * 80)


if __name__ == "__main__":
    import sys as _sys

    _default_path = str(Path(__file__).parent / "data" / "arrhythmia.data")
    path    = _sys.argv[1] if len(_sys.argv) > 1 else _default_path
    full    = "--full" in _sys.argv
    dtype   = "wfdb" if "--wfdb" in _sys.argv else "arrhythmia"

    if dtype == "arrhythmia":
        output = main(data_path=path, run_full=full, dataset_type="arrhythmia")
    else:
        raise RuntimeError(
            "WFDB execution from CLI requires explicit record_names and label_map. "
            "Call main(..., dataset_type='wfdb', wfdb_db_dir=..., "
            "wfdb_record_names=[...], wfdb_label_map={...}) from Python."
        )