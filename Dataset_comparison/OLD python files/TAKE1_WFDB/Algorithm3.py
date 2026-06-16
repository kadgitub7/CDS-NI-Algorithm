"""
================================================================================
Algorithm 3: CDS Executive Actions Refining  (FA = 0 Policy)
REVISED FOR:
  • UCI Arrhythmia tabular dataset
  • WFDB-based datasets via tabularized feature extraction
================================================================================

PURPOSE
-------
Algorithm 3 performs a greedy set-cover pass over the Executive Action Library
produced by Algorithm 2. Its goal is to prune features (sensors) that do not
add *new* discriminating information about unhealthy users beyond what has
already been identified by higher-weight features.

This revision preserves the full original Algorithm 3 structure while adding:
  1. Dataset-schema awareness via Algorithm 1 / 2 abstractions
  2. Removal of hard-coded Arrhythmia-only assumptions:
       • feature count / naming
       • healthy class = 1 only
       • UCI-only loading path assumptions
  3. Compatibility with WFDB-derived tabularized feature matrices
  4. Backward compatibility with the original Arrhythmia flow
  5. Optional end-to-end execution using generalized loaders

IMPORTANT WFDB NOTE
-------------------
Algorithm 3 consumes:
  • DecisionTree from Algorithm 1
  • Algorithm2Output from Algorithm 2
  • Raw tabular data matrix (N × F)
  • Labels vector

For WFDB datasets, this means Algorithm 1 must already have converted waveform
records into fixed-length feature vectors compatible with the CDS pipeline.
Algorithm 3 itself remains feature-matrix based and therefore does NOT directly
read waveform records.

================================================================================
"""

from __future__ import annotations

import copy
import logging
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np

# ── Import from Algorithms 1 & 2 ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from Algorithm1 import (
    DecisionTree,
    TreeNode,
    HEALTHY_CLASS,
    FEATURE_NAMES,
    DIAGNOSTIC_THRESHOLD,
    DatasetSchema,
    DatasetBundle,
    load_dataset,
    load_dataset_bundle,
    build_decision_tree,
)
from Algorithm2 import (
    Algorithm2Output,
    ExecutiveActionEntry,
    PerceptorModelEntry,
    run_algorithm2,
    DEFAULT_N_BINS,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger(name: str = "CDS.Alg3") -> logging.Logger:
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
# SECTION 1B – SCHEMA / DATASET GENERALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _schema_from_tree(tree: DecisionTree) -> Optional[DatasetSchema]:
    """Return dataset schema from tree if available."""
    return getattr(tree, "schema", None)


def _healthy_class(schema: Optional[DatasetSchema]) -> int:
    """
    Healthy-class lookup with backward-compatible fallback.
    """
    if schema is not None:
        return schema.healthy_class
    return HEALTHY_CLASS


def _feature_name(feature_idx: int, schema: Optional[DatasetSchema]) -> str:
    """
    Schema-aware feature-name lookup.
    """
    if schema is not None:
        return schema.feature_name(feature_idx)
    return FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")


def _all_classes_from_labels(labels: np.ndarray) -> List[int]:
    """
    Sorted class labels present in the provided label vector.
    """
    return sorted(int(x) for x in np.unique(labels).tolist())


def _all_disease_classes_from_node(node: TreeNode,
                                   schema: Optional[DatasetSchema]) -> List[int]:
    """
    Disease classes present in a tree node, excluding the healthy class.
    """
    hcls = _healthy_class(schema)
    return sorted(int(h) for h in node.health_dist if int(h) != hcls and node.health_dist[h] > 0)


def _describe_dataset(schema: Optional[DatasetSchema],
                      data: Optional[np.ndarray] = None,
                      labels: Optional[np.ndarray] = None) -> str:
    """
    Build a compact dataset description string for logs/reports.
    """
    if schema is not None:
        base = f"{schema.dataset_name} ({schema.source_format})"
        if data is not None and labels is not None:
            return (f"{base}: N={data.shape[0]} users/records, "
                    f"F={data.shape[1]} features, "
                    f"classes={sorted(np.unique(labels).tolist())}")
        return base
    if data is not None and labels is not None:
        return f"LegacyDataset: N={data.shape[0]}, F={data.shape[1]}, classes={sorted(np.unique(labels).tolist())}"
    return "unknown-dataset"


def _node_feature_count(node: TreeNode) -> int:
    """
    Convenience accessor; explicit helper retained for generalized debugging.
    """
    return len(node.feature_indices)


def _node_user_count(node: TreeNode) -> int:
    """
    Convenience accessor; explicit helper retained for generalized debugging.
    """
    return len(node.user_indices)


def _wfdb_or_tabular_guard(data: np.ndarray, labels: np.ndarray) -> None:
    """
    Validate that Algorithm 3 received a tabularized feature matrix.

    WFDB support in Algorithms 1–4 depends on WFDB records having already been
    converted into fixed-length per-record feature vectors. This helper enforces
    the resulting tabular contract only; it does not load WFDB directly.
    """
    if data.ndim != 2:
        raise ValueError(
            f"Algorithm 3 expects a 2D tabular feature matrix (N,F). Got shape={data.shape}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"Algorithm 3 expects a 1D labels vector. Got shape={labels.shape}."
        )
    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Algorithm 3 expects data and labels to align on rows. "
            f"Got data.shape[0]={data.shape[0]} and labels.shape[0]={labels.shape[0]}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RefinementRecord:
    """
    Audit record for one (node, disease class, feature) decision in Algorithm 3.

    [PAPER] Each record maps to the body of the inner two loops of Algorithm 3.
    """
    node_id:             str
    focus_level:         int
    disease_class:       int
    feature_idx:         int
    feature_name:        str
    action_weight_orig:  float
    action_weight_final: float
    b_min:               float
    b_max:               float
    n_users_node:        int
    n_users_outside:     int
    s_cumulative:        int
    buffer_before:       int
    buffer_after:        int
    was_retained:        bool
    prune_reason:        str
    h_rank:              int
    h_order:             int
    reset_mode:          str

    def summary_line(self) -> str:
        status = "RETAIN" if self.was_retained else "PRUNE "
        return (
            f"  [{self.node_id}|h={self.disease_class:2d}|"
            f"o={self.feature_idx:3d}({self.feature_name[:16]:16s})] "
            f"{status}  r={self.action_weight_orig:.4f}  "
            f"n_out={self.n_users_outside:3d}  "
            f"s={self.s_cumulative:3d}  buf_before={self.buffer_before:3d}  "
            f"{'✓' if self.was_retained else '✗'}"
            + (f"  [{self.prune_reason}]" if not self.was_retained else "")
        )


@dataclass
class NodeRefinementSummary:
    """
    Per-node summary statistics for Algorithm 3.
    """
    node_id:             str
    focus_level:         int
    n_disease_classes:   int
    n_actions_before:    int
    n_actions_retained:  int
    n_actions_removed:   int
    retention_rate:      float
    final_buffer:        int
    n_users_flagged:     int
    per_disease:         Dict[int, Tuple[int, int]]


@dataclass
class Algorithm3Output:
    """
    Complete output of Algorithm 3.
    """
    refined_actions:    List[ExecutiveActionEntry]  = field(default_factory=list)
    removed_actions:    List[ExecutiveActionEntry]  = field(default_factory=list)
    refinement_log:     List[RefinementRecord]      = field(default_factory=list)
    node_summaries:     Dict[str, NodeRefinementSummary] = field(default_factory=dict)
    reset_mode:         str                         = "global"
    n_nodes_processed:  int                         = 0
    schema:             Optional[DatasetSchema]     = None
    dataset_name:       str                         = ""

    def retained_for_node(self, node_id: str) -> List[ExecutiveActionEntry]:
        return [a for a in self.refined_actions if a.node_id == node_id]

    def retained_for_node_disease(
        self, node_id: str, disease_h: int
    ) -> List[ExecutiveActionEntry]:
        acts = [a for a in self.refined_actions
                if a.node_id == node_id and a.disease_class == disease_h]
        return sorted(acts, key=lambda e: e.action_weight, reverse=True)

    def log_for_node(self, node_id: str) -> List[RefinementRecord]:
        return [r for r in self.refinement_log if r.node_id == node_id]

    def n_retained(self, node_id: str, disease_h: Optional[int] = None) -> int:
        acts = self.retained_for_node(node_id)
        if disease_h is not None:
            acts = [a for a in acts if a.disease_class == disease_h]
        return len(acts)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – VALIDATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def validate_alg2_prerequisites(
    alg2_output: Algorithm2Output,
    node: TreeNode,
    schema: Optional[DatasetSchema] = None,
) -> Tuple[bool, List[str]]:
    """
    Verify that Algorithm 2 has been run for the given node before running
    Algorithm 3.
    """
    issues: List[str] = []
    nid = node.node_id

    perc_entries = [e for e in alg2_output.perceptor_library if e.node_id == nid]
    if not perc_entries:
        issues.append(f"No perceptor entries for node {nid!r}")

    exec_entries = [e for e in alg2_output.executive_library if e.node_id == nid]
    if not exec_entries:
        issues.append(f"No executive entries for node {nid!r}")

    for e in perc_entries:
        hr = e.healthy_range
        if not np.isfinite(hr.b_min_healthy) or not np.isfinite(hr.b_max_healthy):
            issues.append(
                f"Non-finite healthy range for node {nid!r} feat {e.feature_idx} "
                f"({_feature_name(e.feature_idx, schema)}): "
                f"[{hr.b_min_healthy}, {hr.b_max_healthy}]"
            )

    for e in exec_entries:
        if e.action_weight < 0:
            issues.append(
                f"Negative action weight for node {nid!r} "
                f"feat {e.feature_idx} ({_feature_name(e.feature_idx, schema)}) "
                f"h={e.disease_class}: {e.action_weight}"
            )

    return (len(issues) == 0), issues


def validate_fa0_constraint(retained: List[ExecutiveActionEntry],
                            schema: Optional[DatasetSchema] = None) -> List[str]:
    """
    Verify the FA=0 policy is preserved in retained actions.
    """
    violations: List[str] = []
    healthy_cls = _healthy_class(schema)
    for a in retained:
        if a.disease_class == healthy_cls:
            violations.append(
                f"FA=0 violation: action for h={healthy_cls} (healthy class) "
                f"retained at node {a.node_id!r} feat {a.feature_idx}"
            )
    return violations


def validate_output_consistency(
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
) -> List[str]:
    """
    Cross-check Algorithm 3 output against Algorithm 2 for consistency.
    """
    issues: List[str] = []

    alg2_index: Dict[Tuple[str, int, int], float] = {
        (e.node_id, e.feature_idx, e.disease_class): e.action_weight
        for e in alg2_output.executive_library
    }

    seen_keys: set = set()
    for a in alg3_output.refined_actions:
        key = (a.node_id, a.feature_idx, a.disease_class)
        if key not in alg2_index:
            issues.append(
                f"Retained action {key} was NOT in Algorithm 2 library"
            )
        orig_w = alg2_index.get(key, 0.0)
        if a.action_weight > orig_w + 1e-9:
            issues.append(
                f"Retained action {key}: weight {a.action_weight:.4f} "
                f"> original {orig_w:.4f}"
            )
        if key in seen_keys:
            issues.append(f"Duplicate retained action for key {key}")
        seen_keys.add(key)

    for a in alg3_output.removed_actions:
        if a.action_weight != 0.0:
            issues.append(
                f"Removed action (node={a.node_id!r}, feat={a.feature_idx}, "
                f"h={a.disease_class}) has non-zero weight {a.action_weight}"
            )

    return issues


def validate_newinf_monotonicity(log_records: List[RefinementRecord]) -> List[str]:
    """
    Verify that s_cumulative is monotonically non-decreasing within each
    Algorithm 3 run.
    """
    issues: List[str] = []
    if not log_records:
        return issues
    prev_s = 0
    for i, rec in enumerate(log_records):
        if rec.s_cumulative < prev_s:
            issues.append(
                f"Monotonicity violated at log position {i}: "
                f"s_cumulative went from {prev_s} -> {rec.s_cumulative} "
                f"(node={rec.node_id!r}, h={rec.disease_class}, "
                f"feat={rec.feature_idx})"
            )
        prev_s = rec.s_cumulative
    return issues


def validate_feature_indices_in_bounds(
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
    n_features: int,
) -> List[str]:
    """
    Generalized dimensionality validation for tabularized Arrhythmia/WFDB data.
    """
    issues: List[str] = []
    for e in alg2_output.executive_library:
        if not (0 <= e.feature_idx < n_features):
            issues.append(
                f"Algorithm2 executive feature index out of bounds: "
                f"node={e.node_id!r}, feat={e.feature_idx}, n_features={n_features}"
            )
    for e in alg3_output.refined_actions:
        if not (0 <= e.feature_idx < n_features):
            issues.append(
                f"Algorithm3 refined feature index out of bounds: "
                f"node={e.node_id!r}, feat={e.feature_idx}, n_features={n_features}"
            )
    for e in alg3_output.removed_actions:
        if not (0 <= e.feature_idx < n_features):
            issues.append(
                f"Algorithm3 removed feature index out of bounds: "
                f"node={e.node_id!r}, feat={e.feature_idx}, n_features={n_features}"
            )
    return issues


def run_all_validations(
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
    tree: DecisionTree,
    data: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> bool:
    """
    Run all Algorithm 3 validation checks and report results.
    """
    all_ok = True
    separator = "─" * 70
    schema = _schema_from_tree(tree)

    if verbose:
        print(f"\n{separator}")
        print("ALGORITHM 3 VALIDATION SUITE")
        print(separator)

    nodes_processed = list({e.node_id for e in alg3_output.refined_actions}
                           | {e.node_id for e in alg3_output.removed_actions})
    pre_issues_total = 0
    for nid in nodes_processed:
        if nid not in tree.all_nodes:
            if verbose:
                print(f"  [WARN] Node {nid!r} not in tree")
            continue
        node = tree.all_nodes[nid]
        ok, issues = validate_alg2_prerequisites(alg2_output, node, schema=schema)
        if not ok:
            for iss in issues:
                if verbose:
                    print(f"  [FAIL] Prerequisite: {iss}")
            pre_issues_total += len(issues)
            all_ok = False
    if verbose and pre_issues_total == 0:
        print(f"  [PASS] Prerequisite check: all {len(nodes_processed)} nodes have Alg2 data")

    fa0_issues = validate_fa0_constraint(alg3_output.refined_actions, schema=schema)
    if fa0_issues:
        for iss in fa0_issues:
            if verbose:
                print(f"  [FAIL] FA=0: {iss}")
        all_ok = False
    elif verbose:
        print(f"  [PASS] FA=0 constraint: no healthy-class actions in retained set")

    cons_issues = validate_output_consistency(alg2_output, alg3_output)
    if cons_issues:
        for iss in cons_issues:
            if verbose:
                print(f"  [FAIL] Consistency: {iss}")
        all_ok = False
    elif verbose:
        n_ret = len(alg3_output.refined_actions)
        n_rem = len(alg3_output.removed_actions)
        print(f"  [PASS] Output consistency: {n_ret} retained, {n_rem} removed")

    for nid in nodes_processed:
        node_log = alg3_output.log_for_node(nid)
        mono_issues = validate_newinf_monotonicity(node_log)
        if mono_issues:
            for iss in mono_issues:
                if verbose:
                    print(f"  [FAIL] Monotonicity ({nid!r}): {iss}")
            all_ok = False
    if all_ok and verbose:
        print(f"  [PASS] newinf monotonicity: s_cumulative is non-decreasing")

    for nid, summary in alg3_output.node_summaries.items():
        n_users = tree.all_nodes[nid].n_users if nid in tree.all_nodes else 0
        if summary.n_users_flagged > n_users:
            if verbose:
                print(
                    f"  [FAIL] Coverage ({nid!r}): "
                    f"flagged {summary.n_users_flagged} > node size {n_users}"
                )
            all_ok = False
    if all_ok and verbose:
        print(f"  [PASS] Coverage sanity: flagged users <= node size")

    if data is not None:
        feat_issues = validate_feature_indices_in_bounds(
            alg2_output=alg2_output,
            alg3_output=alg3_output,
            n_features=data.shape[1],
        )
        if feat_issues:
            all_ok = False
            if verbose:
                for iss in feat_issues:
                    print(f"  [FAIL] Dimensionality: {iss}")
        elif verbose:
            print(f"  [PASS] Feature-index dimensionality: all indices within [0, {data.shape[1]-1}]")

    if verbose:
        status = "[ALL PASS]" if all_ok else "[SOME FAILURES]"
        print(f"\n  -> Validation result: {status}")
        print(separator)

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – CORE HELPER: RAW VALUE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def _get_raw_value(global_user_idx: int, feature_idx: int, data: np.ndarray) -> float:
    """
    BD_m^k(o, u) – raw feature value for user u at feature o.

    [PAPER] "BD_m^k(o_m^k(f_m^k), u)" refers to the raw measurement for user u
    in the training database for feature o at focus level m in node (k, f).

    [ENGR]  For UCI Arrhythmia or WFDB-tabularized datasets, this is simply
            data[global_user_idx, feature_idx].
    """
    return float(data[global_user_idx, feature_idx])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – CORE HELPER: RANGE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _is_outside_range(value: float, b_min: float, b_max: float) -> bool:
    """
    [PAPER] Algorithm 3, line 6:
        if BD_m^k(o, u) > b_max^{km}(f, o)  OR  BD_m^k(o, u) < b_min^{km}(f, o)

    [ENGR]  NaN returns False.
    [ENGR]  Strict inequalities (> and <) are used as written in the paper.
    """
    if np.isnan(value):
        return False
    return (value > b_max) or (value < b_min)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – CORE ALGORITHM 3 LOGIC (ONE NODE)
# ─────────────────────────────────────────────────────────────────────────────

def _refine_one_node(
    node:          TreeNode,
    alg2_output:   Algorithm2Output,
    data:          np.ndarray,
    labels:        np.ndarray,
    reset_per_h:   bool,
    verbose:       bool = False,
    schema:        Optional[DatasetSchema] = None,
) -> Tuple[List[ExecutiveActionEntry], List[ExecutiveActionEntry], List[RefinementRecord]]:
    """
    Run Algorithm 3 for a single tree node.
    """
    nid  = node.node_id
    m    = node.focus_level
    n    = len(node.user_indices)

    healthy_cls = _healthy_class(schema)
    disease_classes: List[int] = sorted(
        [h for h in node.health_dist if h != healthy_cls and node.health_dist[h] > 0]
    )

    if not disease_classes:
        log.debug(f"  Node {nid!r}: no disease classes present -> skip")
        return [], [], []

    log.debug(
        f"\n{'─'*60}\n"
        f"  Algorithm 3 | node={nid!r}  m={m}  n_users={n}  "
        f"disease_classes={disease_classes}"
    )

    newinf: np.ndarray = np.zeros(n, dtype=bool)
    buffer: int        = 0

    retained:   List[ExecutiveActionEntry] = []
    removed:    List[ExecutiveActionEntry] = []
    log_records: List[RefinementRecord]   = []
    h_order_counter: int = 0

    for h in disease_classes:

        if reset_per_h:
            buffer = 0
            newinf = np.zeros(len(node.user_indices), dtype=bool)
            if verbose:
                log.debug(f"    [per-h reset] h={h}: buffer->0, newinf->empty")

        h_actions: List[ExecutiveActionEntry] = [
            copy.copy(e)
            for e in alg2_output.executive_library
            if e.node_id == nid and e.disease_class == h
        ]

        if not h_actions:
            log.debug(f"    h={h}: no actions in executive library -> skip")
            continue

        h_actions_sorted: List[ExecutiveActionEntry] = sorted(
            [a for a in h_actions if a.action_weight > 0.0],
            key=lambda e: e.action_weight,
            reverse=True,
        )

        n_zero = len(h_actions) - len(h_actions_sorted)
        if n_zero > 0:
            log.debug(
                f"    h={h}: {len(h_actions)} actions, "
                f"{n_zero} zero-weight removed before sorting"
            )
            for a in h_actions:
                if a.action_weight == 0.0:
                    a_copy = copy.copy(a)
                    a_copy.action_weight = 0.0
                    removed.append(a_copy)

        if not h_actions_sorted:
            log.debug(f"    h={h}: all actions have r=0 after filtering -> skip")
            continue

        log.debug(
            f"    h={h}: {len(h_actions_sorted)} non-zero actions to process, "
            f"top feature: feat={h_actions_sorted[0].feature_idx} "
            f"({_feature_name(h_actions_sorted[0].feature_idx, schema)}) "
            f"r={h_actions_sorted[0].action_weight:.4f}"
        )

        for h_rank, action in enumerate(h_actions_sorted):
            o    = action.feature_idx
            o_name = action.feature_name if action.feature_name else _feature_name(o, schema)

            model_entry: Optional[PerceptorModelEntry] = alg2_output.get_model(nid, o)

            if model_entry is None:
                log.debug(
                    f"      o={o}({o_name}): no perceptor model -> skip "
                    f"(all-NaN in node?)"
                )
                removed.append(action)
                log_records.append(RefinementRecord(
                    node_id=nid, focus_level=m, disease_class=h,
                    feature_idx=o, feature_name=o_name,
                    action_weight_orig=action.action_weight, action_weight_final=0.0,
                    b_min=float('nan'), b_max=float('nan'),
                    n_users_node=n, n_users_outside=0,
                    s_cumulative=int(newinf.sum()),
                    buffer_before=buffer, buffer_after=buffer,
                    was_retained=False, prune_reason="no perceptor model",
                    h_rank=h_rank, h_order=h_order_counter,
                    reset_mode="per_h" if reset_per_h else "global",
                ))
                h_order_counter += 1
                continue

            b_min: float = model_entry.healthy_range.b_min_healthy
            b_max: float = model_entry.healthy_range.b_max_healthy

            n_newly_flagged_this_iter = 0

            raw_vals = data[node.user_indices, o]
            valid_mask = ~np.isnan(raw_vals)
            outside_mask = (raw_vals > b_max) | (raw_vals < b_min)
            flag_mask = valid_mask & outside_mask

            new_flags = flag_mask & ~newinf
            n_newly_flagged_this_iter = int(new_flags.sum())
            newinf |= flag_mask

            s: int = int(newinf.sum())

            buffer_before = buffer

            if s <= buffer:
                orig_weight = action.action_weight
                action.action_weight = 0.0
                removed.append(action)

                if buffer == 0 and s == 0:
                    prune_reason = "s=0: feature flags no users outside range"
                else:
                    prune_reason = (
                        f"s={s} ≤ Buffer={buffer}: no new user coverage"
                    )

                was_retained = False
                log.debug(
                    f"      PRUNE  h={h:2d} o={o:3d}({o_name[:15]:15s}) "
                    f"r={orig_weight:.4f}  "
                    f"s={s:3d}  buffer={buffer:3d}  -> {prune_reason}"
                )
            else:
                retained.append(action)
                prune_reason = ""
                was_retained = True
                log.debug(
                    f"      RETAIN h={h:2d} o={o:3d}({o_name[:15]:15s}) "
                    f"r={action.action_weight:.4f}  "
                    f"s={s:3d}  buffer={buffer:3d}  "
                    f"(+{n_newly_flagged_this_iter} new users)"
                )

            buffer = s

            log_records.append(RefinementRecord(
                node_id              = nid,
                focus_level          = m,
                disease_class        = h,
                feature_idx          = o,
                feature_name         = o_name,
                action_weight_orig   = next(
                                           (e.action_weight for e in alg2_output.executive_library
                                            if e.node_id == nid and e.feature_idx == o and e.disease_class == h),
                                           0.0,
                                       ),
                action_weight_final  = action.action_weight,
                b_min                = b_min,
                b_max                = b_max,
                n_users_node         = n,
                n_users_outside      = n_newly_flagged_this_iter,
                s_cumulative         = s,
                buffer_before        = buffer_before,
                buffer_after         = buffer,
                was_retained         = was_retained,
                prune_reason         = prune_reason,
                h_rank               = h_rank,
                h_order              = h_order_counter,
                reset_mode           = "per_h" if reset_per_h else "global",
            ))
            h_order_counter += 1

    log.debug(
        f"  Node {nid!r}: retained={len(retained)}  removed={len(removed)}  "
        f"final_buffer={buffer}  users_flagged={int(newinf.sum())}"
    )
    return retained, removed, log_records


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – NODE SUMMARY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_node_summary(
    node:        TreeNode,
    retained:    List[ExecutiveActionEntry],
    removed:     List[ExecutiveActionEntry],
    log_records: List[RefinementRecord],
    alg2_output: Algorithm2Output,
    schema:      Optional[DatasetSchema] = None,
) -> NodeRefinementSummary:
    """
    Compute per-node summary statistics for Algorithm 3.
    """
    nid = node.node_id
    all_actions_before = [e for e in alg2_output.executive_library if e.node_id == nid]
    n_before   = len(all_actions_before)
    n_retained = len(retained)
    n_removed  = len(removed)

    healthy_cls = _healthy_class(schema)
    disease_classes = sorted(
        {h for h in node.health_dist if h != healthy_cls and node.health_dist[h] > 0}
    )

    per_disease: Dict[int, Tuple[int, int]] = {}
    for h in disease_classes:
        h_before   = len([e for e in all_actions_before if e.disease_class == h])
        h_retained = len([e for e in retained          if e.disease_class == h])
        per_disease[h] = (h_before, h_retained)

    node_log = [r for r in log_records if r.node_id == nid]
    if node_log:
        final_buffer    = max(r.buffer_after  for r in node_log)
        n_users_flagged = max(r.s_cumulative  for r in node_log)
    else:
        final_buffer    = 0
        n_users_flagged = 0

    return NodeRefinementSummary(
        node_id            = nid,
        focus_level        = node.focus_level,
        n_disease_classes  = len(disease_classes),
        n_actions_before   = n_before,
        n_actions_retained = n_retained,
        n_actions_removed  = n_removed,
        retention_rate     = (n_retained / n_before) if n_before > 0 else 0.0,
        final_buffer       = final_buffer,
        n_users_flagged    = n_users_flagged,
        per_disease        = per_disease,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm3(
    alg2_output:   Algorithm2Output,
    tree:          DecisionTree,
    data:          np.ndarray,
    labels:        np.ndarray,
    nodes_filter:  Optional[List[str]] = None,
    reset_per_h:   bool = False,
    verbose:       bool = False,
) -> Algorithm3Output:
    """
    Run Algorithm 3: Actions Refining for the CDS Executive.
    """
    _wfdb_or_tabular_guard(data, labels)
    schema = _schema_from_tree(tree)

    log.info("=" * 70)
    log.info("Algorithm 3: CDS Executive Actions Refining (FA = 0)")
    log.info(f"  dataset     = {_describe_dataset(schema, data, labels)}")
    log.info(f"  reset_per_h = {reset_per_h}  "
             f"({'per-disease pruning' if reset_per_h else 'global accumulation [legacy]'})")
    log.info("=" * 70)

    nodes_with_actions: set = {e.node_id for e in alg2_output.executive_library}

    if nodes_filter is not None:
        nodes_to_process = [nid for nid in nodes_filter if nid in nodes_with_actions]
        log.info(
            f"  nodes_filter={nodes_filter} -> "
            f"processing {len(nodes_to_process)} nodes"
        )
    else:
        nodes_to_process = sorted(nodes_with_actions)
        log.info(f"  Processing all {len(nodes_to_process)} nodes with executive actions")

    all_retained:    List[ExecutiveActionEntry]  = []
    all_removed:     List[ExecutiveActionEntry]  = []
    all_log_records: List[RefinementRecord]      = []
    all_summaries:   Dict[str, NodeRefinementSummary] = {}
    n_processed = 0

    for nid in nodes_to_process:
        if nid not in tree.all_nodes:
            log.warning(f"  Node {nid!r} not found in tree – skipping")
            continue

        node = tree.all_nodes[nid]
        log.info(f"\n  Processing node {nid!r}  m={node.focus_level}  "
                 f"n_users={node.n_users}  n_features={node.n_features}")

        ok, issues = validate_alg2_prerequisites(alg2_output, node, schema=schema)
        if not ok:
            log.warning(f"  Prerequisite issues for node {nid!r}:")
            for iss in issues:
                log.warning(f"    {iss}")
            log.warning(f"  -> Skipping node {nid!r}")
            continue

        retained, removed, log_records = _refine_one_node(
            node       = node,
            alg2_output= alg2_output,
            data       = data,
            labels     = labels,
            reset_per_h= reset_per_h,
            verbose    = verbose,
            schema     = schema,
        )

        all_retained.extend(retained)
        all_removed.extend(removed)
        all_log_records.extend(log_records)

        summary = _compute_node_summary(
            node=node, retained=retained, removed=removed,
            log_records=log_records, alg2_output=alg2_output,
            schema=schema,
        )
        all_summaries[nid] = summary

        log.info(
            f"  -> Node {nid!r}: retained={summary.n_actions_retained}/"
            f"{summary.n_actions_before}  "
            f"({summary.retention_rate*100:.1f}%)  "
            f"flagged_users={summary.n_users_flagged}/{node.n_users}"
        )
        n_processed += 1

    output = Algorithm3Output(
        refined_actions   = all_retained,
        removed_actions   = all_removed,
        refinement_log    = all_log_records,
        node_summaries    = all_summaries,
        reset_mode        = "per_h" if reset_per_h else "global",
        n_nodes_processed = n_processed,
        schema            = schema,
        dataset_name      = schema.dataset_name if schema is not None else "unknown",
    )

    n_total_before   = len(alg2_output.executive_library)
    n_total_retained = len(all_retained)
    n_total_removed  = len(all_removed)

    log.info("\n" + "=" * 70)
    log.info("Algorithm 3 complete")
    log.info(f"  Nodes processed   : {n_processed}")
    log.info(f"  Actions before    : {n_total_before}")
    log.info(f"  Actions retained  : {n_total_retained}")
    log.info(f"  Actions removed   : {n_total_removed}")
    if n_total_before > 0:
        log.info(
            f"  Retention rate    : "
            f"{n_total_retained/n_total_before*100:.1f}%"
        )
    log.info("=" * 70)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_algorithm3_summary(output: Algorithm3Output) -> None:
    """Print top-level summary of Algorithm 3 results."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("ALGORITHM 3 SUMMARY: Executive Actions Refining")
    print(sep)
    print(f"  Dataset          : {output.dataset_name or 'unknown'}")
    print(f"  Mode             : {output.reset_mode}")
    print(f"  Nodes processed  : {output.n_nodes_processed}")
    print(f"  Actions retained : {len(output.refined_actions)}")
    print(f"  Actions removed  : {len(output.removed_actions)}")
    total = len(output.refined_actions) + len(output.removed_actions)
    if total > 0:
        print(f"  Retention rate   : {len(output.refined_actions)/total*100:.1f}%")

    print(f"\n  Per-node breakdown:")
    print(f"  {'node_id':30s} {'m':3} {'before':8} {'retain':8} "
          f"{'rate%':7} {'flag_u':7} {'buf':5}")
    print(f"  {'─'*65}")
    for nid, s in sorted(output.node_summaries.items()):
        print(f"  {nid:30s} {s.focus_level:3d} {s.n_actions_before:8d} "
              f"{s.n_actions_retained:8d} {s.retention_rate*100:7.1f} "
              f"{s.n_users_flagged:7d} {s.final_buffer:5d}")


def print_refinement_detail(
    output:    Algorithm3Output,
    node_id:   str,
    disease_h: Optional[int] = None,
    max_rows:  int = 40,
) -> None:
    """
    Print detailed decision log for a given node (and optionally disease class).
    """
    sep = "─" * 78
    title = f"REFINEMENT DETAIL | node={node_id!r}"
    if disease_h is not None:
        title += f" | disease h={disease_h}"
    print(f"\n{sep}\n{title}\n{sep}")

    records = output.log_for_node(node_id)
    if disease_h is not None:
        records = [r for r in records if r.disease_class == disease_h]

    if not records:
        print("  (no records)")
        return

    print(f"  {'h':3} {'rank':5} {'feat':4} {'name':20} {'r_orig':8} "
          f"{'b_min':8} {'b_max':8} {'n_out':6} {'s':5} {'buf_b':6} {'status':7}")
    print(f"  {'─'*85}")

    for rec in records[:max_rows]:
        status = "RETAIN" if rec.was_retained else "PRUNE "
        print(
            f"  {rec.disease_class:3d} {rec.h_rank:5d} {rec.feature_idx:4d} "
            f"{rec.feature_name[:20]:20s} {rec.action_weight_orig:8.4f} "
            f"{rec.b_min:8.3g} {rec.b_max:8.3g} "
            f"{rec.n_users_outside:6d} {rec.s_cumulative:5d} "
            f"{rec.buffer_before:6d} {status}"
        )
        if not rec.was_retained and rec.prune_reason:
            print(f"       └─ {rec.prune_reason}")

    if len(records) > max_rows:
        print(f"  … ({len(records) - max_rows} more rows) …")


def print_per_disease_retained(
    output:  Algorithm3Output,
    node_id: str,
) -> None:
    """Print retained actions per disease class for a node."""
    sep = "─" * 70
    print(f"\n{sep}\nRETAINED ACTIONS PER DISEASE | node={node_id!r}\n{sep}")
    retained = output.retained_for_node(node_id)
    if not retained:
        print("  (none retained)")
        return

    by_h: Dict[int, List[ExecutiveActionEntry]] = defaultdict(list)
    for a in retained:
        by_h[a.disease_class].append(a)

    for h in sorted(by_h):
        h_acts = sorted(by_h[h], key=lambda e: e.action_weight, reverse=True)
        print(f"\n  Disease h={h}  ({len(h_acts)} retained actions):")
        print(f"    {'rank':4} {'feat':4} {'name':25} {'r_{o|h}':8} "
              f"{'p_below':8} {'p_above':8}")
        print(f"    {'─'*65}")
        for rank, act in enumerate(h_acts, 1):
            print(
                f"    {rank:4d} {act.feature_idx:4d} {act.feature_name:25s} "
                f"{act.action_weight:8.4f} {act.p_below_normal:8.4f} "
                f"{act.p_above_normal:8.4f}"
            )


def compare_alg2_vs_alg3(
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
    node_id:     str,
    disease_h:   int,
) -> None:
    """
    Side-by-side comparison of Algorithm 2 and Algorithm 3 action lists.
    """
    sep = "─" * 78
    print(f"\n{sep}")
    print(f"ALG2 vs ALG3 COMPARISON  |  node={node_id!r}  h={disease_h}")
    print(sep)

    alg2_acts = sorted(
        [e for e in alg2_output.executive_library
         if e.node_id == node_id and e.disease_class == disease_h],
        key=lambda e: e.action_weight, reverse=True,
    )
    alg3_acts = {a.feature_idx for a in alg3_output.retained_for_node_disease(node_id, disease_h)}

    print(f"  Alg2 actions: {len(alg2_acts)}  |  "
          f"Alg3 retained: {len(alg3_acts)}  |  "
          f"Pruned: {len(alg2_acts) - len(alg3_acts)}")
    print(f"\n  {'rank':4} {'feat':4} {'name':22} {'r':8} {'status':7} {'note':20}")
    print(f"  {'─'*70}")

    log_map: Dict[int, RefinementRecord] = {
        r.feature_idx: r
        for r in alg3_output.log_for_node(node_id)
        if r.disease_class == disease_h
    }

    for rank, act in enumerate(alg2_acts, 1):
        status = "RETAIN" if act.feature_idx in alg3_acts else "PRUNE "
        log_rec = log_map.get(act.feature_idx)
        note = ""
        if log_rec and not log_rec.was_retained:
            note = log_rec.prune_reason[:20] if log_rec.prune_reason else ""
        print(
            f"  {rank:4d} {act.feature_idx:4d} {act.feature_name[:22]:22s} "
            f"{act.action_weight:8.4f} {status} {note}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_coverage(
    alg3_output: Algorithm3Output,
    tree:        DecisionTree,
    labels:      np.ndarray,
    data:        np.ndarray,
    alg2_output: Algorithm2Output,
) -> Dict[str, dict]:
    """
    Analyse how well the RETAINED actions cover abnormal users per node.
    """
    results: Dict[str, dict] = {}

    for act in alg3_output.refined_actions:
        nid = act.node_id
        h   = act.disease_class
        o   = act.feature_idx

        if nid not in tree.all_nodes:
            continue
        node = tree.all_nodes[nid]

        model_entry = alg2_output.get_model(nid, o)
        if model_entry is None:
            continue

        b_min = model_entry.healthy_range.b_min_healthy
        b_max = model_entry.healthy_range.b_max_healthy

        h_users = [u for u in node.user_indices if labels[u] == h]
        h_vals  = data[h_users, o]
        valid   = ~np.isnan(h_vals)
        n_total = int(valid.sum())
        if n_total == 0:
            n_flag = 0
            sens   = 0.0
        else:
            n_flag = int(((h_vals[valid] > b_max) | (h_vals[valid] < b_min)).sum())
            sens   = n_flag / n_total

        results.setdefault(nid, {}).setdefault(h, {})[o] = {
            "sensitivity":      sens,
            "n_flagged":        n_flag,
            "n_diseased_valid": n_total,
            "feature_name":     act.feature_name,
            "action_weight":    act.action_weight,
        }

    return results


def print_coverage_analysis(
    coverage: Dict[str, dict],
    node_id:  str,
    top_k:    int = 10,
) -> None:
    """Print coverage analysis for a given node."""
    if node_id not in coverage:
        print(f"\nNo coverage data for node {node_id!r}")
        return

    sep = "─" * 72
    print(f"\n{sep}\nCOVERAGE ANALYSIS  |  node={node_id!r}\n{sep}")

    for h in sorted(coverage[node_id]):
        feats = coverage[node_id][h]
        print(f"\n  Disease h={h}  ({len(feats)} retained features):")
        print(f"    {'feat':4} {'name':22} {'sens%':7} {'flagged':9} "
              f"{'total':7} {'r':8}")
        print(f"    {'─'*60}")
        sorted_feats = sorted(feats.items(),
                              key=lambda kv: kv[1]["sensitivity"], reverse=True)
        for o, info in sorted_feats[:top_k]:
            print(
                f"    {o:4d} {info['feature_name'][:22]:22s} "
                f"{info['sensitivity']*100:7.1f} "
                f"{info['n_flagged']:9d} "
                f"{info['n_diseased_valid']:7d} "
                f"{info['action_weight']:8.4f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – FIDELITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

def assess_paper_fidelity(output: Algorithm3Output) -> None:
    """
    Evaluate and document the fidelity of this implementation to the paper.
    """
    sep = "=" * 72
    print(f"\n{sep}")
    print("PAPER FIDELITY ASSESSMENT – Algorithm 3")
    print(sep)

    print(f"""
ALGORITHM 3 PSEUDOCODE COVERAGE
─────────────────────────────────
[PAPER -> Code] Mapping:

  Line 1  for h = 2 to H do
            -> disease_classes = sorted([h for h in node.health_dist if h != healthy_class])
            -> for h in disease_classes:
            FIDELITY: ✓ Exact under generalized healthy-class lookup.

  Line 2  r_buf = r_{{O_m^{{kf}}|h}}
            -> h_actions = [copy(e) for e in alg2_output.executive_library
                           if e.node_id==nid and e.disease_class==h]
            FIDELITY: ✓ Exact lookup from Algorithm 2 executive library.

  Line 3  Sort r_buf descending, remove 0-value elements
            -> h_actions_sorted = sorted(
                [a for a in h_actions if a.action_weight > 0],
                key=lambda e: e.action_weight, reverse=True)
            FIDELITY: ✓ Exact.

  Lines 4-9  for o in features; for u in users: if outside -> newinf_u = 1
            -> vectorized tabular matrix check over data[node.user_indices, o]
            FIDELITY: ✓ Exact, generalized to any tabularized dataset.

  Line 10  s = summation of newinf
            -> s = int(newinf.sum())
            FIDELITY: ✓ Exact cumulative sum.

  Line 11  if summation of s <= Buffer then
            -> if s <= buffer:
            FIDELITY: ✓ Exact (≤ as written in paper).

  Lines 12-13  c_{{mh|o}} ← {{}}; r_{{o|h}} = 0
            -> action.action_weight = 0.0; removed.append(action)
            FIDELITY: ✓ Exact.

  Line 15  Buffer = s
            -> buffer = s
            FIDELITY: ✓ Exact.

GENERALIZATION NOTES
────────────────────
  • Feature naming is schema-driven; no Arrhythmia-only FEATURE_NAMES dependency.
  • Healthy class is schema-driven; no fixed assumption that healthy class = 1.
  • User count and feature count are inferred from the provided tabular dataset.
  • WFDB compatibility is achieved through Algorithm 1's tabularization layer.

AMBIGUITIES AND RESOLUTIONS
─────────────────────────────
  [AMBIG-1] Scope of buffer/newinf initialisation.
    Paper shows init BEFORE both loops, suggesting global accumulation.
    Legacy default remains: reset_per_h=False for global accumulation.

  [AMBIG-2] "for o = set O_m^k(f)" – all features or only those with nonzero r?
    Resolution: only features with r > 0 after line 3 filtering.

  [AMBIG-3] Strict vs non-strict inequalities in line 6.
    Paper uses strict inequalities.
    Implementation preserves exact strict inequalities.

  [ENGR-1] NaN handling.
    Paper does not address missing values.
    Implementation: NaN raw value -> user not flagged.

  [ENGR-2] Node independence.
    State is isolated per node, consistent with node-specific user sets.

  [ENGR-3] WFDB support.
    Algorithm 3 remains tabular and therefore relies on Algorithm 1's WFDB
    summarization stage to produce one fixed-length feature vector per record.

OUTPUT SNAPSHOT
────────────────
  dataset={output.dataset_name or 'unknown'}
  reset_mode={output.reset_mode}
  n_nodes_processed={output.n_nodes_processed}
  n_retained={len(output.refined_actions)}
  n_removed={len(output.removed_actions)}
""")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main(
    data_path:    Optional[str] = str(Path(__file__).parent / "data" / "arrhythmia.data"),
    run_full:     bool  = False,
    reset_per_h:  bool  = False,
    verbose_alg3: bool  = False,
    dataset_type: str   = "arrhythmia",
    wfdb_db_dir: Optional[str] = None,
    wfdb_record_names: Optional[List[str]] = None,
    wfdb_label_map: Optional[Dict[str, int]] = None,
    wfdb_pn_dir: Optional[str] = None,
) -> Algorithm3Output:
    """
    End-to-end pipeline:
      Algorithm 1 -> Algorithm 2 -> Algorithm 3 -> Reports

    Supports:
      • dataset_type='arrhythmia'
      • dataset_type='wfdb'
    """
    import logging as _logging
    _logging.getLogger("CDS.Alg1").setLevel(_logging.WARNING)
    _logging.getLogger("CDS.Alg2").setLevel(_logging.WARNING)

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
    _wfdb_or_tabular_guard(data, labels)
    log.info(f"  Loaded: {data.shape[0]} users/records × {data.shape[1]} features")
    log.info(f"  Dataset: {schema.dataset_name} ({schema.source_format})")

    log.info("Step 2: Algorithm 1 – Build Decision Tree")
    tree = build_decision_tree(data, labels, schema=schema)
    log.info(f"  Tree: depth={tree.depth()}  nodes={tree.count_nodes()}")

    if run_full:
        nodes_filter = None
        log.info("Step 3: Processing ALL nodes")
    else:
        nodes_filter = ["root"]
        if schema.sex_feature_index is not None:
            sex_nodes = [n.node_id for n in tree.nodes_by_level.get(2, [])
                         if n.branching_feat_k == schema.sex_feature_index]
            nodes_filter.extend(sex_nodes[:2])
        nodes_filter = [n for n in nodes_filter if n in tree.all_nodes]
        log.info(f"Step 3: Key nodes: {nodes_filter}")

    log.info("Step 4: Algorithm 2 – Perceptor and Executive Training")
    alg2_output = run_algorithm2(
        tree         = tree,
        data         = data,
        labels       = labels,
        n_bins       = DEFAULT_N_BINS,
        nodes_filter = nodes_filter,
    )
    log.info(
        f"  Alg2: {alg2_output.n_perceptor_entries} perceptor, "
        f"{alg2_output.n_executive_entries} executive entries"
    )

    log.info("Step 5: Algorithm 3 – Actions Refining")
    alg3_output = run_algorithm3(
        alg2_output  = alg2_output,
        tree         = tree,
        data         = data,
        labels       = labels,
        nodes_filter = nodes_filter,
        reset_per_h  = reset_per_h,
        verbose      = verbose_alg3,
    )

    log.info("Step 6: Validation")
    run_all_validations(alg2_output, alg3_output, tree, data=data, verbose=True)

    log.info("Step 7: Reports")
    print_algorithm3_summary(alg3_output)

    for nid in (nodes_filter or ["root"]):
        if nid not in alg3_output.node_summaries:
            continue
        disease_candidates = _all_disease_classes_from_node(tree.all_nodes[nid], schema)
        for h in disease_candidates[:3]:
            print_refinement_detail(alg3_output, nid, disease_h=h, max_rows=15)
        print_per_disease_retained(alg3_output, nid)
        for h in disease_candidates[:3]:
            compare_alg2_vs_alg3(alg2_output, alg3_output, nid, h)

    log.info("Step 8: Coverage Analysis")
    coverage = analyse_coverage(alg3_output, tree, labels, data, alg2_output)
    for nid in (nodes_filter or ["root"]):
        print_coverage_analysis(coverage, nid, top_k=8)

    assess_paper_fidelity(alg3_output)

    return alg3_output


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 – COMPATIBILITY SELF-TEST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_minimal_synthetic_bundle(n_users: int = 8,
                                    n_features: int = 6,
                                    healthy_class: int = 1) -> DatasetBundle:
    """
    Small synthetic tabular bundle useful for smoke-testing generalized logic.

    This helper does NOT replace real evaluation. It exists only to sanity-check
    that the Algorithm 3 pipeline accepts arbitrary tabular dimensionalities,
    which is especially relevant for WFDB-derived feature matrices.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_users, n_features)).astype(float)
    labels = np.asarray([healthy_class, healthy_class, 2, 2, 3, 3, healthy_class, 2], dtype=int)[:n_users]

    feature_names = {i: f"feat_{i}" for i in range(n_features)}
    schema = DatasetSchema(
        dataset_name="SyntheticTabular",
        source_format="synthetic",
        n_features=n_features,
        healthy_class=healthy_class,
        label_values=sorted(np.unique(labels).tolist()),
        feature_names=feature_names,
        sex_feature_index=1 if n_features > 1 else None,
        male_code=0.0,
        female_code=1.0,
    )
    return DatasetBundle(data=data, labels=labels, schema=schema)


def smoke_test_algorithm3_generalized() -> bool:
    """
    Lightweight compatibility smoke test.

    Purpose
    -------
    Verify that Algorithm 3 can be imported and run on a non-Arrhythmia tabular
    dataset shape without relying on hard-coded user counts or feature counts.

    Returns
    -------
    True on successful end-to-end smoke execution, else False.
    """
    try:
        bundle = _build_minimal_synthetic_bundle()
        tree = build_decision_tree(bundle.data, bundle.labels, schema=bundle.schema, max_m=2)
        alg2 = run_algorithm2(tree=tree, data=bundle.data, labels=bundle.labels, nodes_filter=["root"])
        _ = run_algorithm3(
            alg2_output=alg2,
            tree=tree,
            data=bundle.data,
            labels=bundle.labels,
            nodes_filter=["root"],
            reset_per_h=False,
            verbose=False,
        )
        return True
    except Exception as exc:
        log.warning(f"Algorithm 3 generalized smoke test failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 – PATH HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_arrhythmia_path(filename: str = "arrhythmia.data") -> Path:
    """
    Find the arrhythmia dataset path in a cross-platform way.

    Search order:
      1. Command-line argument
      2. Current working directory
      3. Same directory as this script
      4. data/ subdirectory beside script
    """
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Dataset not found: {p}")

    cwd_path = Path.cwd() / filename
    if cwd_path.exists():
        return cwd_path.resolve()

    script_dir = Path(__file__).parent
    script_path = script_dir / filename
    if script_path.exists():
        return script_path.resolve()

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

    path     = _sys.argv[1] if len(_sys.argv) > 1 else str(Path(__file__).parent / "data" / "arrhythmia.data")
    full     = "--full"     in _sys.argv
    per_h    = "--per-h"    in _sys.argv
    verbose  = "--verbose"  in _sys.argv
    dtype    = "wfdb" if "--wfdb" in _sys.argv else "arrhythmia"

    # Optional smoke test
    if "--smoke-test" in _sys.argv:
        ok = smoke_test_algorithm3_generalized()
        print(f"Algorithm 3 generalized smoke test: {'PASS' if ok else 'FAIL'}")

    if dtype == "arrhythmia":
        alg3_out = main(
            data_path=path,
            run_full=full,
            reset_per_h=per_h,
            verbose_alg3=verbose,
            dataset_type="arrhythmia",
        )
    else:
        raise RuntimeError(
            "WFDB CLI execution requires explicit record_names and label_map. "
            "Call main(..., dataset_type='wfdb', wfdb_db_dir=..., "
            "wfdb_record_names=[...], wfdb_label_map={...}, wfdb_pn_dir=...) from Python."
        )