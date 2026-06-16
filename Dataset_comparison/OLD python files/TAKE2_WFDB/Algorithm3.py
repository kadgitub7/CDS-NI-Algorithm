# Algorithm3_WFDB.py

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Imports from WFDB-based Algorithms 1 & 2 ─────────────────────────────────
from Algorithm1 import (
    DecisionTree,
    TreeNode,
    HEALTHY_CLASS,
    FEATURE_NAMES,
)
from Algorithm2 import (
    Algorithm2Output,
    ExecutiveActionEntry,
    PerceptorModelEntry,
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
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(fmt)
    log.addHandler(h)
    log.propagate = False
    return log

log = _build_logger()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RefinementRecord:
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
    node_id:             str
    focus_level:         int
    n_disease_classes:   int
    n_actions_before:    int
    n_actions_retained:  int
    n_actions_removed:   int
    retention_rate:      float
    final_buffer:        int
    n_users_flagged:     int
    per_disease:         Dict[int, Tuple[int, int]]   # h -> (before, retained)


@dataclass
class Algorithm3Output:
    refined_actions:    List[ExecutiveActionEntry]  = field(default_factory=list)
    removed_actions:    List[ExecutiveActionEntry]  = field(default_factory=list)
    refinement_log:     List[RefinementRecord]      = field(default_factory=list)
    node_summaries:     Dict[str, NodeRefinementSummary] = field(default_factory=dict)
    reset_mode:         str                         = "global"
    n_nodes_processed:  int                         = 0

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
) -> Tuple[bool, List[str]]:
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
                f"Non-finite healthy range for node {nid!r} feat {e.feature_idx}: "
                f"[{hr.b_min_healthy}, {hr.b_max_healthy}]"
            )

    for e in exec_entries:
        if e.action_weight < 0:
            issues.append(
                f"Negative action weight for node {nid!r} "
                f"feat {e.feature_idx} h={e.disease_class}: {e.action_weight}"
            )

    return (len(issues) == 0), issues


def validate_fa0_constraint(retained: List[ExecutiveActionEntry]) -> List[str]:
    violations: List[str] = []
    for a in retained:
        if a.disease_class == HEALTHY_CLASS:
            violations.append(
                f"FA=0 violation: action for h={HEALTHY_CLASS} "
                f"retained at node {a.node_id!r} feat {a.feature_idx}"
            )
    return violations


def validate_output_consistency(
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
) -> List[str]:
    issues: List[str] = []

    alg2_index: Dict[Tuple[str, int, int], float] = {
        (e.node_id, e.feature_idx, e.disease_class): e.action_weight
        for e in alg2_output.executive_library
    }

    seen_keys: set = set()
    for a in alg3_output.refined_actions:
        key = (a.node_id, a.feature_idx, a.disease_class)
        if key not in alg2_index:
            issues.append(f"Retained action {key} was NOT in Algorithm 2 library")
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
    issues: List[str] = []
    if not log_records:
        return issues
    prev_s = 0
    for i, rec in enumerate(log_records):
        if rec.s_cumulative < prev_s:
            issues.append(
                f"Monotonicity violated at log position {i}: "
                f"s_cumulative {prev_s} -> {rec.s_cumulative} "
                f"(node={rec.node_id!r}, h={rec.disease_class}, "
                f"feat={rec.feature_idx})"
            )
        prev_s = rec.s_cumulative
    return issues


def run_all_validations(
    alg2_output: Algorithm2Output,
    alg3_output: Algorithm3Output,
    tree: DecisionTree,
    verbose: bool = True,
) -> bool:
    all_ok = True
    sep = "─" * 70

    if verbose:
        print(f"\n{sep}")
        print("ALGORITHM 3 VALIDATION SUITE")
        print(sep)

    nodes_processed = list({e.node_id for e in alg3_output.refined_actions}
                           | {e.node_id for e in alg3_output.removed_actions})
    pre_issues_total = 0
    for nid in nodes_processed:
        if nid not in tree.all_nodes:
            if verbose:
                print(f"  [WARN] Node {nid!r} not in tree")
            continue
        node = tree.all_nodes[nid]
        ok, issues = validate_alg2_prerequisites(alg2_output, node)
        if not ok:
            for iss in issues:
                if verbose:
                    print(f"  [FAIL] Prerequisite: {iss}")
            pre_issues_total += len(issues)
            all_ok = False
    if verbose and pre_issues_total == 0:
        print(f"  [PASS] Prerequisite check: all {len(nodes_processed)} nodes have Alg2 data")

    fa0_issues = validate_fa0_constraint(alg3_output.refined_actions)
    if fa0_issues:
        for iss in fa0_issues:
            if verbose:
                print(f"  [FAIL] FA=0: {iss}")
        all_ok = False
    elif verbose:
        print("  [PASS] FA=0 constraint: no healthy-class actions in retained set")

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
        print("  [PASS] newinf monotonicity: s_cumulative is non-decreasing")

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
        print("  [PASS] Coverage sanity: flagged users <= node size")

    if verbose:
        status = "[ALL PASS]" if all_ok else "[SOME FAILURES]"
        print(f"\n  -> Validation result: {status}")
        print(sep)

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – CORE NODE REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _refine_one_node(
    node:          TreeNode,
    alg2_output:   Algorithm2Output,
    data:          np.ndarray,
    labels:        np.ndarray,
    reset_per_h:   bool,
    verbose:       bool = False,
) -> Tuple[List[ExecutiveActionEntry], List[ExecutiveActionEntry], List[RefinementRecord]]:
    nid  = node.node_id
    m    = node.focus_level
    n    = len(node.user_indices)

    disease_classes: List[int] = sorted(
        [h for h in node.health_dist if h != HEALTHY_CLASS and node.health_dist[h] > 0]
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
            f"r={h_actions_sorted[0].action_weight:.4f}"
        )

        for h_rank, action in enumerate(h_actions_sorted):
            o      = action.feature_idx
            o_name = action.feature_name

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
                    prune_reason = f"s={s} ≤ Buffer={buffer}: no new user coverage"

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
# SECTION 5 – NODE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _compute_node_summary(
    node:        TreeNode,
    retained:    List[ExecutiveActionEntry],
    removed:     List[ExecutiveActionEntry],
    log_records: List[RefinementRecord],
    alg2_output: Algorithm2Output,
) -> NodeRefinementSummary:
    nid = node.node_id
    all_actions_before = [e for e in alg2_output.executive_library if e.node_id == nid]
    n_before   = len(all_actions_before)
    n_retained = len(retained)
    n_removed  = len(removed)

    disease_classes = sorted(
        {h for h in node.health_dist if h != HEALTHY_CLASS and node.health_dist[h] > 0}
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
# SECTION 6 – MAIN ENTRY POINT (WFDB CONTEXT)
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
    Run Algorithm 3 (Executive Actions Refining) on top of WFDB-derived features.

    Parameters
    ----------
    alg2_output  : Output of WFDB-based Algorithm 2.
    tree         : Decision tree from WFDB-based Algorithm 1.
    data         : Raw feature matrix (e.g., WFDB-derived ECG features).
    labels       : Class labels aligned with rows of `data`.
    nodes_filter : Optional list of node_ids to process; None -> all nodes
                   that have executive actions.
    reset_per_h  : If True, buffer/newinf reset per disease class (per-h mode).
                   If False, global accumulation across disease classes.
    verbose      : If True, log detailed per-(h,o) decisions.
    """
    log.info("=" * 70)
    log.info("Algorithm 3: CDS Executive Actions Refining (FA = 0)")
    log.info(f"  reset_per_h = {reset_per_h}  "
             f"({'per-disease pruning' if reset_per_h else 'global accumulation'})")
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
                 f"n_users={node.n_users}")

        ok, issues = validate_alg2_prerequisites(alg2_output, node)
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
        )

        all_retained.extend(retained)
        all_removed.extend(removed)
        all_log_records.extend(log_records)

        summary = _compute_node_summary(
            node=node, retained=retained, removed=removed,
            log_records=log_records, alg2_output=alg2_output,
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
# SECTION 7 – REPORTING HELPERS (OPTIONAL)
# ─────────────────────────────────────────────────────────────────────────────

def print_algorithm3_summary(output: Algorithm3Output) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("ALGORITHM 3 SUMMARY: Executive Actions Refining")
    print(sep)
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
