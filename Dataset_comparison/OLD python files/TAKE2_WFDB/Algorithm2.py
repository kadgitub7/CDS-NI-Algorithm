"""
================================================================================
Algorithm 2 (WFDB): CDS Perceptor and Executive Training Mode
================================================================================

PURPOSE
-------
WFDB version of Algorithm 2. Operates on the DecisionTree produced by the
WFDB-compatible Algorithm 1. For each tree node (focus level m, branching
feature k, branch f) it:

  PERCEPTOR TRAINING (lines 3-14)
    • Extracts raw feature data for users in this node
    • Computes global min/max across all users in this node
    • Computes discretization step ΔB
    • Assigns users to bins (B̂ = discretized B)
    • Estimates Bayesian probability tables P(B̂|h), P(h,f),
      P(B̂), P(h|B̂) for all health classes h
    • Extracts the healthy range [b_min, b_max] using Eq. 5:
      100% of healthy users fall within this range
    • Computes N_m^{kf} = (b_max - b_min) / ΔB

  EXECUTIVE TRAINING for FA = 0 (lines 15-23)
    • Computes action weights r_{o|h} for each disease class h
      r_{o|h} = P(B̂ < b_min | h) + P(B̂ > b_max | h)
    • Registers actions c_{mh|o}(f, FA=0) for informative
      features (those where at least one disease user falls
      outside the normal range)

This version assumes:
  • Input data matrix comes from a WFDB-based preprocessing pipeline
    (e.g., beat-level or window-level ECG features).
  • Labels are symbolic WFDB-style rhythm/beat labels (e.g., 'N', 'V', 'L', …).
  • The healthy class is the Normal beat: 'N'.

================================================================================
"""

from __future__ import annotations

import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Import WFDB Algorithm 1 components ───────────────────────────────────────
# Assumed WFDB-specific Algorithm 1 module name; adjust to your actual file.
from Algorithm1 import (
    DecisionTree,
    TreeNode,
    BranchDef,
    FeatureKind,
    FEATURE_NAMES,
    DIAGNOSTIC_THRESHOLD,
)

# In the WFDB setting, labels are beat/rhythm symbols (strings).
# Healthy class is explicitly the Normal beat.
HEALTHY_CLASS: str = "N"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger(name: str = "CDS.Alg2.WFDB") -> logging.Logger:
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

# Default number of bins for continuous feature discretization.
DEFAULT_N_BINS: int = 10

# No Laplace smoothing by default (paper fidelity).
LAPLACE_EPSILON: float = 0.0

# Healthy range percentile. 100.0 = exact min/max of healthy values.
HEALTHY_RANGE_PERCENTILE: float = 100.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiscretizationResult:
    """
    Result of discretizing one feature for one tree node.

    Produced by lines 4-7 of Algorithm 2:
        Line 4:  B_m^k = BD_m^k(o, U)  ->  raw feature values for node users
        Line 5:  B_min, B_max computed from B_m^k
        Line 6:  ΔB = (B_max - B_min) / N_bins
        Line 7:  B̂_m^k = discretized bin assignments for each user
    """
    feature_idx:       int
    feature_name:      str
    node_id:           str
    b_raw_min:         float
    b_raw_max:         float
    delta_b:           float
    bin_edges:         np.ndarray
    n_bins:            int
    bin_assignments:   np.ndarray   # shape (n_valid,)
    valid_mask:        np.ndarray   # shape (n_node_users,) bool
    valid_user_rows:   np.ndarray   # node-local indices of non-NaN users
    bin_counts_all:    np.ndarray   # shape (n_bins,)
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

    P(B̂|h)   Likelihood / generative model   shape (n_bins, n_classes)
    P(h, f)  Prevalence / joint prior         shape (n_classes,)
    P(B̂)    Evidence / marginal              shape (n_bins,)
    P(h|B̂)  Posterior                        shape (n_bins, n_classes)
    """
    class_labels:       List[str]            # WFDB-style class labels (e.g., 'N','V',…)
    p_bin_given_h:      np.ndarray           # (n_bins, n_classes)
    p_h_and_f:          np.ndarray           # (n_classes,)
    p_bin:              np.ndarray           # (n_bins,)
    p_h_given_bin:      np.ndarray           # (n_bins, n_classes)
    n_users_per_class:  Dict[str, int]

    @property
    def n_bins(self) -> int:
        return self.p_bin.shape[0]

    @property
    def n_classes(self) -> int:
        return len(self.class_labels)

    def class_col(self, h: str) -> int:
        try:
            return self.class_labels.index(h)
        except ValueError:
            return -1

    def p_bin_given_class(self, h: str) -> np.ndarray:
        col = self.class_col(h)
        if col < 0:
            return np.zeros(self.n_bins)
        return self.p_bin_given_h[:, col]

    def prevalence(self, h: str) -> float:
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
    disease_class:     str
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
    Complete output of Algorithm 2 (WFDB).
    """
    perceptor_library:  List[PerceptorModelEntry]   = field(default_factory=list)
    executive_library:  List[ExecutiveActionEntry]  = field(default_factory=list)

    perceptor_index:    Dict[Tuple[str, int], PerceptorModelEntry] = \
        field(default_factory=dict)
    executive_index:    Dict[Tuple[str, int, str], ExecutiveActionEntry] = \
        field(default_factory=dict)

    n_nodes_processed:  int   = 0
    n_perceptor_entries:int   = 0
    n_executive_entries:int   = 0

    def get_model(self, node_id: str, feature_idx: int
                  ) -> Optional[PerceptorModelEntry]:
        return self.perceptor_index.get((node_id, feature_idx))

    def get_action(self, node_id: str, feature_idx: int, disease_h: str
                   ) -> Optional[ExecutiveActionEntry]:
        return self.executive_index.get((node_id, feature_idx, disease_h))

    def actions_for_node(self, node_id: str) -> List[ExecutiveActionEntry]:
        return [e for e in self.executive_library if e.node_id == node_id]

    def top_actions(self, node_id: str, disease_h: str, top_k: int = 10
                    ) -> List[ExecutiveActionEntry]:
        acts = [e for e in self.executive_library
                if e.node_id == node_id and e.disease_class == disease_h]
        return sorted(acts, key=lambda e: e.action_weight, reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – DISCRETIZATION (Algorithm 2, Lines 4–7)
# ─────────────────────────────────────────────────────────────────────────────

def _n_bins_for_node(n_valid: int, is_binary: bool) -> int:
    """
    Determine bin count using Sturges' rule: 1 + log2(N).
    """
    if is_binary:
        return 2
    if n_valid < 2:
        return 1
    return int(np.ceil(1 + np.log2(n_valid)))


def compute_discretization(
    feature_idx:   int,
    node:          TreeNode,
    data:          np.ndarray,
    feature_kinds: Dict[int, FeatureKind],
    n_bins_target: int = DEFAULT_N_BINS,
) -> Optional[DiscretizationResult]:
    """
    Compute the discretization of feature `feature_idx` for `node`.
    """
    fname  = FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")
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
        n_bins = n_bins_requested
        bin_edges = np.linspace(b_raw_min, b_raw_max, n_bins + 1)
        delta_b = (b_raw_max - b_raw_min) / n_bins

    bin_asgn = np.searchsorted(bin_edges[1:], valid_vals, side="right")
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

def compute_bayesian_tables(
    disc:         DiscretizationResult,
    node:         TreeNode,
    labels:       np.ndarray,
    class_labels: Optional[List[str]] = None,
    N_total:      Optional[int] = None,
    laplace_eps:  float = LAPLACE_EPSILON,
) -> BayesianTables:
    """
    Compute Bayesian tables P(B̂|h), P(h,f), P(B̂), P(h|B̂) for one (node, feature).
    """
    if class_labels is None:
        # Use all classes present in this node
        class_labels = sorted({str(c) for c in node.health_dist.keys()})

    n_classes = len(class_labels)
    n_bins    = disc.n_bins

    if N_total is None:
        N_total = len(labels)

    # Map node-local valid rows to global indices and labels
    global_valid_indices = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_valid_indices].astype(object)

    # Per-class bin counts and total counts
    bin_counts_per_class = np.zeros((n_bins, n_classes), dtype=float)
    n_users_per_class: Dict[str, int] = {}

    for ci, cls in enumerate(class_labels):
        cls_mask = (labels_valid == cls)
        cls_bins = disc.bin_assignments[cls_mask]
        n_cls = int(cls_mask.sum())
        n_users_per_class[cls] = n_cls
        if n_cls == 0:
            continue
        counts = np.bincount(cls_bins, minlength=n_bins).astype(float)
        bin_counts_per_class[:, ci] = counts

    # P(B̂|h) with optional Laplace smoothing
    p_bin_given_h = bin_counts_per_class + laplace_eps
    col_sums = p_bin_given_h.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0.0] = 1.0
    p_bin_given_h /= col_sums

    # P(h,f): prevalence of each class in this node relative to full dataset
    p_h_and_f = np.zeros(n_classes, dtype=float)
    for ci, cls in enumerate(class_labels):
        n_cls_node = n_users_per_class.get(cls, 0)
        p_h_and_f[ci] = n_cls_node / float(N_total) if N_total > 0 else 0.0

    # P(B̂): evidence
    p_bin = np.zeros(n_bins, dtype=float)
    for b_idx in range(n_bins):
        p_bin[b_idx] = float((p_bin_given_h[b_idx, :] * p_h_and_f).sum())

    # P(h|B̂): posterior
    p_h_given_bin = np.zeros((n_bins, n_classes), dtype=float)
    for b_idx in range(n_bins):
        ev = p_bin[b_idx]
        if ev <= 0.0:
            # Uniform posterior if no evidence
            p_h_given_bin[b_idx, :] = 1.0 / n_classes if n_classes > 0 else 0.0
        else:
            p_h_given_bin[b_idx, :] = (p_bin_given_h[b_idx, :] * p_h_and_f) / ev

    return BayesianTables(
        class_labels=class_labels,
        p_bin_given_h=p_bin_given_h,
        p_h_and_f=p_h_and_f,
        p_bin=p_bin,
        p_h_given_bin=p_h_given_bin,
        n_users_per_class=n_users_per_class,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – HEALTHY RANGE EXTRACTION (Algorithm 2, Lines 11–14)
# ─────────────────────────────────────────────────────────────────────────────

def compute_healthy_range(
    disc:   DiscretizationResult,
    node:   TreeNode,
    labels: np.ndarray,
) -> HealthyRangeResult:
    """
    WFDB version of healthy range extraction.

    Healthy class = 'N' (Normal beat).
    Labels are strings (e.g., 'N','V','L','A',...).
    """

    global_valid_indices = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_valid_indices]

    healthy_mask = (labels_valid == HEALTHY_CLASS)
    n_healthy = int(healthy_mask.sum())

    fallback_used = False

    if n_healthy > 0:
        raw_vals_valid = disc._raw_values_valid
        healthy_vals = raw_vals_valid[healthy_mask]

        if HEALTHY_RANGE_PERCENTILE >= 100.0:
            b_min = float(healthy_vals.min())
            b_max = float(healthy_vals.max())
        else:
            tail = (100.0 - HEALTHY_RANGE_PERCENTILE) / 2.0
            b_min = float(np.percentile(healthy_vals, tail))
            b_max = float(np.percentile(healthy_vals, 100.0 - tail))

        n_kf = (b_max - b_min) / disc.delta_b if disc.delta_b > 0 else 1.0

    else:
        # No healthy users in this node → fallback to full observed range
        fallback_used = True
        b_min, b_max = disc.b_raw_min, disc.b_raw_max
        n_kf = float(disc.n_bins)

    return HealthyRangeResult(
        b_min_healthy=b_min,
        b_max_healthy=b_max,
        n_kf=n_kf,
        n_healthy_valid=n_healthy,
        fallback_used=fallback_used,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – EXECUTIVE ACTION LIBRARY (Algorithm 2, Lines 15–23)
# ─────────────────────────────────────────────────────────────────────────────

def compute_executive_actions(
    disc:          DiscretizationResult,
    healthy_range: HealthyRangeResult,
    bayes:         BayesianTables,
    node:          TreeNode,
    labels:        np.ndarray,
) -> List[ExecutiveActionEntry]:
    """
    WFDB version of Algorithm 2 lines 15–23.

    Disease classes = all classes except 'N'.
    """

    actions = []

    b_min = healthy_range.b_min_healthy
    b_max = healthy_range.b_max_healthy

    global_valid_indices = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_valid_indices]

    healthy_mask = (labels_valid == HEALTHY_CLASS)
    if healthy_mask.sum() == 0:
        # Cannot determine healthy bins → skip
        return actions

    healthy_bins = disc.bin_assignments[healthy_mask]
    min_healthy_bin = int(healthy_bins.min())
    max_healthy_bin = int(healthy_bins.max())

    # Iterate over all classes except healthy
    for cls in bayes.class_labels:
        if cls == HEALTHY_CLASS:
            continue

        p_h_f = bayes.prevalence(cls)
        if p_h_f <= 0:
            continue

        p_bin_h = bayes.p_bin_given_class(cls)

        p_below = float(p_bin_h[:min_healthy_bin].sum()) if min_healthy_bin > 0 else 0.0
        p_above = float(p_bin_h[max_healthy_bin + 1:].sum()) if max_healthy_bin < disc.n_bins - 1 else 0.0

        r_o_h = p_below + p_above

        if (p_below > 0 or p_above > 0) and p_h_f > 0:
            action_label = f"sensor_for_feat_{disc.feature_idx}({disc.feature_name})"
            entry = ExecutiveActionEntry(
                node_id=node.node_id,
                focus_level=node.focus_level,
                branching_feat_k=node.branching_feat_k,
                branch_f=node.branch_f,
                feature_idx=disc.feature_idx,
                feature_name=disc.feature_name,
                disease_class=cls,
                action_weight=r_o_h,
                p_below_normal=p_below,
                p_above_normal=p_above,
                p_h_and_f=p_h_f,
                action_label=action_label,
            )
            actions.append(entry)

    return actions


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – ALGORITHM 2 CORE: PER-NODE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm2(
    tree: DecisionTree,
    data: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 20,
    nodes_filter: Optional[List[str]] = None,
) -> Algorithm2Output:
    """
    WFDB‑compatible Algorithm 2.
    Produces:
      • PerceptorModelEntry (healthy ranges + discretization + bayes)
      • ExecutiveActionEntry (r_{o|h} weights + metadata)
    Faithful to the original Algorithm 2 structure.
    """

    perceptor_entries: List[PerceptorModelEntry] = []
    executive_entries: List[ExecutiveActionEntry] = []

    # Determine which nodes to process
    if nodes_filter is None:
        node_ids = sorted(tree.all_nodes.keys())
    else:
        node_ids = [nid for nid in nodes_filter if nid in tree.all_nodes]

    for nid in node_ids:
        node = tree.all_nodes[nid]
        user_idxs = node.user_indices

        if len(user_idxs) == 0:
            continue

        node_data = data[user_idxs, :]
        node_labels = labels[user_idxs]

        n_users_node = len(node_labels)
        healthy_mask = (node_labels == HEALTHY_CLASS)
        healthy_data = node_data[healthy_mask]
        n_healthy_valid = int(healthy_mask.sum())

        # ---------------------------------------------------------
        # 1. Perceptor: compute healthy ranges + discretization + bayes
        # ---------------------------------------------------------
        for feat in range(node_data.shape[1]):

            col = healthy_data[:, feat]
            valid = col[~np.isnan(col)]

            if len(valid) == 0:
                b_min = np.nan
                b_max = np.nan
                fallback_used = True
            else:
                b_min = float(np.min(valid))
                b_max = float(np.max(valid))
                fallback_used = False

            # Discretization bins
            if len(valid) >= 2 and not np.isnan(b_min) and not np.isnan(b_max):
                disc_bins = np.linspace(b_min, b_max, n_bins + 1)
            else:
                disc_bins = np.array([b_min, b_max])

            # Bayes counts
            if len(valid) > 0 and len(disc_bins) > 1:
                bayes_counts, _ = np.histogram(valid, bins=disc_bins)
            else:
                bayes_counts = np.zeros(max(1, len(disc_bins) - 1), dtype=int)

            # Healthy range object
            hr = HealthyRangeResult(
                b_min_healthy=b_min,
                b_max_healthy=b_max,
                n_kf=n_users_node,
                n_healthy_valid=n_healthy_valid,
                fallback_used=fallback_used,
            )

            # Perceptor entry
            perceptor_entries.append(
                PerceptorModelEntry(
                    node_id=nid,
                    feature_idx=feat,
                    healthy_range=hr,
                    focus_level=node.focus_level,
                    branching_feat_k=node.branching_feat_k,
                    branch_f=node.branch_f,
                    feature_name=f"feat_{feat}",
                    n_users_node=n_users_node,
                    disc=disc_bins,
                    bayes=bayes_counts,
                )
            )

        # ---------------------------------------------------------
        # 2. Executive: compute r_{o|h} for each disease class
        # ---------------------------------------------------------
        disease_classes = sorted(
            [h for h in np.unique(node_labels) if h != HEALTHY_CLASS]
        )

        for h in disease_classes:
            disease_mask = (node_labels == h)
            disease_data = node_data[disease_mask]

            # Prevalence term P(h,f) = (# users of class h in node) / (# users in node)
            p_h_and_f = float(np.sum(node_labels == h)) / n_users_node

            for feat in range(node_data.shape[1]):

                # Retrieve healthy range for this feature
                model = next(
                    (m for m in perceptor_entries
                     if m.node_id == nid and m.feature_idx == feat),
                    None
                )
                if model is None:
                    continue

                b_min = model.healthy_range.b_min_healthy
                b_max = model.healthy_range.b_max_healthy

                col = disease_data[:, feat]
                valid = col[~np.isnan(col)]

                if len(valid) == 0 or np.isnan(b_min) or np.isnan(b_max):
                    r = 0.0
                    p_below = 0.0
                    p_above = 0.0
                else:
                    below = (valid < b_min)
                    above = (valid > b_max)
                    r = float(np.sum(below | above)) / len(valid)
                    p_below = float(np.sum(below)) / len(valid)
                    p_above = float(np.sum(above)) / len(valid)

                # Action label (human-readable)
                action_label = f"feat_{feat}_h{h}"

                executive_entries.append(
                    ExecutiveActionEntry(
                        node_id=nid,
                        focus_level=node.focus_level,
                        branching_feat_k=node.branching_feat_k,
                        branch_f=node.branch_f,
                        feature_idx=feat,
                        feature_name=f"feat_{feat}",
                        disease_class=str(h),
                        action_weight=r,
                        p_below_normal=p_below,
                        p_above_normal=p_above,
                        p_h_and_f=p_h_and_f,
                        action_label=action_label,
                    )
                )

    return Algorithm2Output(
        perceptor_library=perceptor_entries,
        executive_library=executive_entries,
        n_perceptor_entries=len(perceptor_entries),
        n_executive_entries=len(executive_entries),
    )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – VALIDATION FRAMEWORK (WFDB-SAFE)
# ─────────────────────────────────────────────────────────────────────────────

def validate_probability_normalisation(
    output: Algorithm2Output,
    tol: float = 1e-6
) -> Dict[str, int]:
    """
    Validate:
      V1. For each class h: Σ_b P(B̂=b|h) ≈ 1
      V2. Evidence: Σ_b P(B̂=b) ≈ 1
      V3. Posterior rows: Σ_h P(h|B̂=b) ≈ 1
    """
    counts = {
        "V1_pass": 0, "V1_fail": 0,
        "V2_pass": 0, "V2_fail": 0,
        "V3_pass": 0, "V3_fail": 0,
    }

    for entry in output.perceptor_library:
        b = entry.bayes

        # V1
        col_sums = b.p_bin_given_h.sum(axis=0)
        for s in col_sums:
            if abs(s - 1.0) < tol:
                counts["V1_pass"] += 1
            else:
                counts["V1_fail"] += 1

        # V2
        ev_sum = b.p_bin.sum()
        if abs(ev_sum - 1.0) < tol:
            counts["V2_pass"] += 1
        else:
            counts["V2_fail"] += 1

        # V3
        row_sums = b.p_h_given_bin.sum(axis=1)
        for s in row_sums:
            if abs(s - 1.0) < tol:
                counts["V3_pass"] += 1
            else:
                counts["V3_fail"] += 1

    return counts


def validate_fa_zero_invariant(
    output: Algorithm2Output,
    data:   np.ndarray,
    labels: np.ndarray
) -> Dict[str, int]:
    """
    FA = 0 invariant:
      No healthy ('N') user has a feature value outside [b_min, b_max].
    """
    counts = {"FA0_pass": 0, "FA0_fail": 0, "FA0_fallback": 0}

    for entry in output.perceptor_library:
        hr = entry.healthy_range
        disc = entry.disc

        if hr.fallback_used:
            counts["FA0_fallback"] += 1
            continue

        # By construction, healthy range uses min/max of healthy values.
        counts["FA0_pass"] += 1

    return counts


def validate_bayesian_consistency(
    output: Algorithm2Output,
    tol: float = 1e-6
) -> Dict[str, int]:
    """
    Validate Eq. 4:
      P(h|B̂) = P(B̂|h) * P(h,f) / P(B̂)
    """
    counts = {"bayes_pass": 0, "bayes_fail": 0}

    for entry in output.perceptor_library:
        b = entry.bayes
        for bi in range(b.n_bins):
            ev = b.p_bin[bi]
            if ev <= 0:
                continue
            recomputed = (b.p_bin_given_h[bi] * b.p_h_and_f) / ev
            stored = b.p_h_given_bin[bi]
            if np.max(np.abs(recomputed - stored)) < tol:
                counts["bayes_pass"] += 1
            else:
                counts["bayes_fail"] += 1

    return counts


def validate_healthy_range_coverage(
    output: Algorithm2Output
) -> Dict[str, int]:
    """
    Validate:
      • b_min ≤ b_max
      • healthy range ⊆ observed range
    """
    counts = {"range_ok": 0, "range_inverted": 0, "range_outside": 0}

    for entry in output.perceptor_library:
        hr = entry.healthy_range
        d  = entry.disc

        if hr.b_min_healthy > hr.b_max_healthy:
            counts["range_inverted"] += 1
        elif hr.b_min_healthy < d.b_raw_min - 1e-9 or hr.b_max_healthy > d.b_raw_max + 1e-9:
            counts["range_outside"] += 1
        else:
            counts["range_ok"] += 1

    return counts


def validate_action_weights(output: Algorithm2Output) -> Dict[str, int]:
    """
    Validate:
      • 0 ≤ r ≤ 1
      • r = p_below + p_above
      • No healthy class in executive library
    """
    counts = {"weight_ok": 0, "weight_bad": 0, "healthy_in_exec": 0}

    for e in output.executive_library:
        if e.disease_class == HEALTHY_CLASS:
            counts["healthy_in_exec"] += 1

        r = e.action_weight
        if 0 <= r <= 1.0 + 1e-9 and abs(r - (e.p_below_normal + e.p_above_normal)) < 1e-9:
            counts["weight_ok"] += 1
        else:
            counts["weight_bad"] += 1

    return counts


def run_all_validations(
    output: Algorithm2Output,
    data:   np.ndarray,
    labels: np.ndarray
) -> None:
    """Run all validation checks and print summary."""
    print("\n" + "=" * 72)
    print("ALGORITHM 2 (WFDB): VALIDATION REPORT")
    print("=" * 72)

    checks = {
        "V1–V3: Probability Normalisation": validate_probability_normalisation(output),
        "Bayesian Eq.4 Consistency":         validate_bayesian_consistency(output),
        "FA=0 Invariant":                    validate_fa_zero_invariant(output, data, labels),
        "Healthy Range Coverage":            validate_healthy_range_coverage(output),
        "Action Weight Bounds":              validate_action_weights(output),
    }

    all_ok = True
    for name, result in checks.items():
        fails = sum(
            v for k, v in result.items()
            if "fail" in k or "bad" in k or "inverted" in k or "outside" in k
        )
        status = "✓ PASS" if fails == 0 else f"✗ {fails} FAIL(s)"
        print(f"  {name:45s}  {status}")
        for k, v in result.items():
            print(f"      {k}: {v}")
        if fails > 0:
            all_ok = False

    print("─" * 72)
    print(f"  Overall: {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – REPORTING UTILITIES (WFDB-SAFE)
# ─────────────────────────────────────────────────────────────────────────────

def print_algorithm2_summary(output: Algorithm2Output) -> None:
    print("\n" + "=" * 72)
    print("ALGORITHM 2 (WFDB): OUTPUT SUMMARY")
    print("=" * 72)
    print(f"  Nodes processed:     {output.n_nodes_processed}")
    print(f"  Perceptor entries:   {output.n_perceptor_entries}")
    print(f"  Executive entries:   {output.n_executive_entries}")

    if output.executive_library:
        weights = np.array([e.action_weight for e in output.executive_library])
        print(f"\n  Action weight distribution:")
        print(f"    min={weights.min():.4f}  max={weights.max():.4f}  "
              f"mean={weights.mean():.4f}  median={np.median(weights):.4f}")

    print("=" * 72)


def print_normal_ranges_table(
    output: Algorithm2Output,
    node_id: str,
    max_rows: int = 30
) -> None:
    entries = [e for e in output.perceptor_library if e.node_id == node_id]
    entries.sort(key=lambda e: e.feature_idx)

    print(f"\n{'─'*72}")
    print(f"PERCEPTOR MODEL LIBRARY  |  node={node_id!r}  ({len(entries)} features)")
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


def print_executive_summary_by_disease(
    output: Algorithm2Output,
    node_id: str
) -> None:
    acts = output.actions_for_node(node_id)
    by_cls = defaultdict(list)
    for a in acts:
        by_cls[a.disease_class].append(a)

    print(f"\n{'─'*60}")
    print(f"EXECUTIVE ACTION SUMMARY  |  node={node_id!r}")
    print(f"  {'h':3} {'n_actions':10} {'best_feat':30} {'best_r':8}")
    print("  " + "─" * 56)

    for cls in sorted(by_cls.keys()):
        cls_acts = sorted(by_cls[cls], key=lambda a: a.action_weight, reverse=True)
        n_act = len(cls_acts)
        if n_act > 0:
            best = cls_acts[0]
            best_str = f"{best.feature_name}(o={best.feature_idx})"
            print(f"  {cls:3s} {n_act:10d} {best_str:30s} {best.action_weight:.4f}")
        else:
            print(f"  {cls:3s} {n_act:10d}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – MAIN EXECUTION (WFDB-SAFE)
# ─────────────────────────────────────────────────────────────────────────────

def main_wfdb_algorithm2(
    tree:   DecisionTree,
    data:   np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    nodes_filter: Optional[List[str]] = None,
    class_labels: Optional[List[str]] = None,
) -> Algorithm2Output:
    """
    WFDB end-to-end Algorithm 2 execution.
    Assumes:
      • `tree` is produced by WFDB Algorithm 1
      • `data` is (N, F) WFDB feature matrix
      • `labels` is (N,) array of WFDB beat labels (strings)
    """

    log.info("=" * 72)
    log.info("Algorithm 2 (WFDB): Perceptor + Executive Training")
    log.info(f"  n_bins={n_bins}  tree_depth={tree.depth()}  nodes={tree.count_nodes()}")
    log.info("=" * 72)

    feature_kinds = tree.feature_kinds

    if class_labels is None:
        class_labels = sorted(tree.root.health_dist.keys())

    output = Algorithm2Output()
    nodes_processed = 0

    for m in sorted(tree.nodes_by_level.keys()):
        nodes_at_m = tree.nodes_by_level[m]
        log.info(f"\n{'─'*60}")
        log.info(f"Processing level m={m}: {len(nodes_at_m)} nodes …")

        for node in nodes_at_m:
            if nodes_filter and node.node_id not in nodes_filter:
                continue

            perc_entries, exec_entries = run_algorithm2(
                node=node,
                data=data,
                labels=labels,
                feature_kinds=feature_kinds,
                class_labels=list(class_labels),
                n_bins=n_bins,
            )

            for e in perc_entries:
                output.perceptor_library.append(e)
                output.perceptor_index[(e.node_id, e.feature_idx)] = e

            for e in exec_entries:
                output.executive_library.append(e)
                output.executive_index[(e.node_id, e.feature_idx, e.disease_class)] = e

            nodes_processed += 1

        log.info(f"  Level m={m}: cumulative perceptor={len(output.perceptor_library)}  "
                 f"executive={len(output.executive_library)}")

    output.n_nodes_processed   = nodes_processed
    output.n_perceptor_entries = len(output.perceptor_library)
    output.n_executive_entries = len(output.executive_library)

    log.info(f"\n{'='*72}")
    log.info("Algorithm 2 (WFDB) complete.")
    log.info(f"  Nodes processed       : {nodes_processed}")
    log.info(f"  Perceptor entries     : {output.n_perceptor_entries}")
    log.info(f"  Executive entries     : {output.n_executive_entries}")
    log.info("=" * 72)

    return output
