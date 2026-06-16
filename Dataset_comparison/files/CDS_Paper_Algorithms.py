"""
================================================================================
Algorithm 1: Creating the CDS Decision Tree
================================================================================

PURPOSE
-------
Algorithm 1 builds the multi-level decision tree that partitions the user
population into progressively finer demographic/physiological sub-groups.
Each sub-group receives its own set of Bayesian normal-range models (done in
Algorithm 2). The tree topology is determined entirely by:
  • which features produce branches that each have >= u_min users  (Eq. 2)
  • a feature-ordering exclusion rule that removes redundant tree paths (line 9)

NOTE ON DATA SOURCE
-------------------
This version imports `wfdb` and provides WFDB-aware dataset loading.
Algorithm 1 still requires a tabular feature matrix of shape (N, 279) and
labels of shape (N,). WFDB records themselves are waveform data, so unless you
already have a 279-feature table derived from WFDB records, you must add a
feature-extraction stage before calling the tree builder.

PAPER FIDELITY NOTATION
-----------------------
Every implementation choice is tagged with one of:
  [PAPER]  – directly stated in the paper
  [INFER]  – logically required by the paper but left unspecified
  [ENGR]   – engineering choice with justification
================================================================================
"""

from __future__ import annotations

import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import wfdb


def _build_logger(name: str = "CDS.Alg1") -> logging.Logger:
    """
    One STDOUT handler for INFO/DEBUG messages.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)-7s | %(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG)
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger


log = _build_logger()

# [PAPER] Eq. 3: u_min = 5 / Threshold
DIAGNOSTIC_THRESHOLD: float = 0.025
U_MIN: int = math.ceil(5 / DIAGNOSTIC_THRESHOLD)

COMPLEXITY_THRESHOLD: int = 16

HEALTHY_CLASS: int = 1
LABEL_COL_IDX: int = 279
N_FEATURES: int = 279


class FeatureKind(Enum):
    """
    [PAPER] Algorithm 1 lines 3-4 distinguish two kinds of features:
      BINARY     – nominal 0/1-type variables
      CONTINUOUS – numeric variables requiring discretisation

    [ENGR] Detection is data-driven:
      if non-NaN unique values are a subset of {0,1} -> BINARY else CONTINUOUS.
    """
    BINARY = auto()
    CONTINUOUS = auto()


def _build_ecg_channel_names() -> Dict[int, str]:
    names: Dict[int, str] = {
        0: "Age", 1: "Sex", 2: "Height", 3: "Weight",
        4: "QRS_dur", 5: "PR_int", 6: "QT_int", 7: "T_int",
        8: "P_int", 9: "QRS_angle", 10: "T_angle", 11: "P_angle",
        12: "QRST_angle", 13: "J_angle", 14: "Heart_rate",
    }
    channels = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    wave_labels = [
        "Q_wid", "R_wid", "S_wid", "Rp_wid", "Sp_wid", "N_defl",
        "Rag_R", "Diph_R", "Rag_P", "Diph_P", "Rag_T", "Diph_T",
    ]
    amp_labels = [
        "JJ_amp", "Q_amp", "R_amp", "S_amp", "Rp_amp", "Sp_amp",
        "P_amp", "T_amp", "QRSA", "QRSTA",
    ]
    for i, ch in enumerate(channels):
        base_w = 15 + i * 12
        for j, lbl in enumerate(wave_labels):
            names[base_w + j] = f"{ch}_{lbl}"
        base_a = 159 + i * 10
        for j, lbl in enumerate(amp_labels):
            names[base_a + j] = f"{ch}_{lbl}"
    return names


FEATURE_NAMES: Dict[int, str] = _build_ecg_channel_names()
MISSING_VALUE_COLS: FrozenSet[int] = frozenset({10, 11, 12, 13, 14})


def classify_features(data: np.ndarray) -> Dict[int, FeatureKind]:
    """
    Classify each feature as BINARY or CONTINUOUS.
    """
    kinds: Dict[int, FeatureKind] = {}
    for col in range(N_FEATURES):
        col_data = data[:, col]
        valid = col_data[~np.isnan(col_data)]
        if len(valid) == 0:
            kinds[col] = FeatureKind.CONTINUOUS
            continue
        unique_vals = set(valid.tolist())
        if unique_vals.issubset({0.0, 1.0}):
            kinds[col] = FeatureKind.BINARY
        else:
            kinds[col] = FeatureKind.CONTINUOUS
    return kinds


def load_wfdb_dataset(
    wfdb_database: str,
    ann_ext: str = "atr",
    max_records: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a WFDB dataset by database identifier, not by local path.

    Parameters
    ----------
    wfdb_database : str
        WFDB / PhysioNet database identifier, e.g. "mitdb".
    ann_ext : str
        Annotation extension to read, default "atr".
    max_records : Optional[int]
        If set, limit the number of records loaded.

    Returns
    -------
    data : np.ndarray
        Tabular feature matrix.
    labels : np.ndarray
        Integer labels derived from annotation symbols.

    IMPORTANT
    ---------
    Algorithm 1 requires shape (N, 279). This loader validates that requirement.
    """
    if any(sep in wfdb_database for sep in ("\\", "/")) or ":" in wfdb_database:
        raise ValueError(
            f"Expected a WFDB database identifier such as 'mitdb', not a local path: "
            f"{wfdb_database!r}"
        )

    try:
        record_list = wfdb.get_record_list(wfdb_database)
    except Exception as e:
        raise ValueError(
            f"Could not retrieve record list for WFDB database {wfdb_database!r}: {e}"
        )

    if not record_list:
        raise ValueError(f"WFDB database {wfdb_database!r} returned no records.")

    if max_records is not None:
        record_list = record_list[:max_records]

    log.info(f"WFDB loader: database={wfdb_database!r}  records_found={len(record_list)}")

    features: List[np.ndarray] = []
    label_symbols: List[str] = []
    feature_dims_seen: List[int] = []
    skipped_records: List[Tuple[str, str]] = []

    for rec_name in record_list:
        try:
            rec = wfdb.rdrecord(rec_name, pn_dir=wfdb_database)
        except Exception as e:
            skipped_records.append((rec_name, f"rdrecord failed: {e}"))
            continue

        sig = getattr(rec, "p_signal", None)

        if sig is None:
            try:
                sig = getattr(rec, "d_signal", None)
                if sig is not None:
                    sig = np.asarray(sig, dtype=float)
            except Exception:
                sig = None

        if sig is None:
            skipped_records.append((rec_name, "no p_signal/d_signal available"))
            continue

        sig = np.asarray(sig, dtype=float)

        if sig.ndim == 1:
            sig = sig[:, None]

        if sig.ndim != 2 or sig.shape[0] == 0 or sig.shape[1] == 0:
            skipped_records.append((rec_name, f"invalid signal shape {sig.shape}"))
            continue

        try:
            feat = np.concatenate([
                np.nanmean(sig, axis=0),
                np.nanstd(sig, axis=0),
                np.nanmin(sig, axis=0),
                np.nanmax(sig, axis=0),
            ]).astype(float)
        except Exception as e:
            skipped_records.append((rec_name, f"feature extraction failed: {e}"))
            continue

        if feat.size == 0:
            skipped_records.append((rec_name, "empty feature vector"))
            continue

        label_sym = "NO_ANN"
        try:
            ann = wfdb.rdann(rec_name, ann_ext, pn_dir=wfdb_database)
            symbols = getattr(ann, "symbol", None)
            if symbols is not None and len(symbols) > 0:
                counts: Dict[str, int] = {}
                for s in symbols:
                    counts[s] = counts.get(s, 0) + 1
                label_sym = max(counts, key=counts.get)
        except Exception:
            pass

        features.append(feat)
        label_symbols.append(label_sym)
        feature_dims_seen.append(feat.size)

    if not features:
        msg = (
            f"WFDB loader found {len(record_list)} records in database {wfdb_database!r}, "
            f"but zero readable records were converted into features."
        )
        if skipped_records:
            preview = "\n".join(
                f"  - {name}: {reason}" for name, reason in skipped_records[:10]
            )
            msg += "\nFirst skipped records:\n" + preview
        raise ValueError(msg)

    dim_counts: Dict[int, int] = {}
    for d in feature_dims_seen:
        dim_counts[d] = dim_counts.get(d, 0) + 1

    target_dim = max(dim_counts, key=dim_counts.get)

    filtered_features: List[np.ndarray] = []
    filtered_labels: List[str] = []
    n_dropped_dim_mismatch = 0

    for feat, lab in zip(features, label_symbols):
        if feat.size == target_dim:
            filtered_features.append(feat)
            filtered_labels.append(lab)
        else:
            n_dropped_dim_mismatch += 1

    if not filtered_features:
        raise ValueError(
            "WFDB loader extracted records, but none shared a common feature dimension. "
            f"Observed dimensions: {dict(sorted(dim_counts.items()))}"
        )

    data = np.vstack(filtered_features)

    if data.shape[1] != N_FEATURES:
        raise ValueError(
            f"WFDB feature extraction produced {data.shape[1]} features, but Algorithm 1 "
            f"requires exactly {N_FEATURES}. Add a WFDB-to-tabular feature extraction stage "
            f"that outputs shape (N, {N_FEATURES})."
        )

    unique_syms = sorted(set(filtered_labels))
    sym_to_int = {s: i + 1 for i, s in enumerate(unique_syms)}
    labels = np.array([sym_to_int[s] for s in filtered_labels], dtype=int)

    log.info(
        f"WFDB loader: loaded {data.shape[0]} records, feature_dim={data.shape[1]}, "
        f"classes={len(unique_syms)}, dropped_dim_mismatch={n_dropped_dim_mismatch}, "
        f"skipped_unreadable={len(skipped_records)}"
    )

    return data, labels


def load_dataset(
    wfdb_database: str,
    ann_ext: str = "atr",
    max_records: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    return load_wfdb_dataset(
        wfdb_database=wfdb_database,
        ann_ext=ann_ext,
        max_records=max_records,
    )


@dataclass
class BranchDef:
    """
    Defines membership in a single branch for a given feature.
    """
    feature_idx: int
    branch_idx: int
    kind: FeatureKind
    value_set: Optional[FrozenSet[float]] = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    label: str = ""

    def contains(self, value: float) -> bool:
        if np.isnan(value):
            return False
        if self.kind == FeatureKind.BINARY:
            return value in self.value_set  # type: ignore[operator]
        return self.lo <= value < self.hi  # type: ignore[operator]


def _fsd_binary(feature_idx: int, user_indices: np.ndarray, data: np.ndarray) -> List[BranchDef]:
    col = data[user_indices, feature_idx]
    valid = col[~np.isnan(col)]
    values_present = sorted(set(valid.tolist()))
    if len(values_present) < 2:
        return []

    fname = FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")
    branches = []
    for bidx, v in enumerate(values_present, start=1):
        if feature_idx == 1:
            lbl = "male" if v == 0.0 else "female"
        else:
            lbl = f"{fname}={int(v)}"
        branches.append(
            BranchDef(
                feature_idx=feature_idx,
                branch_idx=bidx,
                kind=FeatureKind.BINARY,
                value_set=frozenset({v}),
                label=lbl,
            )
        )
    return branches


def _fsd_continuous_median(feature_idx: int, user_indices: np.ndarray, data: np.ndarray) -> List[BranchDef]:
    """
    Create 2 branch definitions for a CONTINUOUS feature using a median split.

    Actual intervals used:
      branch 1 = [-inf, median)
      branch 2 = [median, +inf)
    """
    col_vals = data[user_indices, feature_idx]
    valid = col_vals[~np.isnan(col_vals)]
    if len(valid) == 0:
        return []
    median_val = float(np.median(valid))
    if float(valid.min()) == float(valid.max()):
        return []

    fname = FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")
    b1 = BranchDef(
        feature_idx=feature_idx,
        branch_idx=1,
        kind=FeatureKind.CONTINUOUS,
        lo=-np.inf,
        hi=median_val,
        label=f"{fname}<{median_val}",
    )
    b2 = BranchDef(
        feature_idx=feature_idx,
        branch_idx=2,
        kind=FeatureKind.CONTINUOUS,
        lo=median_val,
        hi=np.inf,
        label=f"{fname}>={median_val}",
    )
    return [b1, b2]


def compute_fsd_branches(
    feature_idx: int,
    user_indices: np.ndarray,
    data: np.ndarray,
    kind: FeatureKind,
) -> List[BranchDef]:
    if kind == FeatureKind.BINARY:
        return _fsd_binary(feature_idx, user_indices, data)
    return _fsd_continuous_median(feature_idx, user_indices, data)


def filter_users_by_branch(
    user_indices: np.ndarray,
    branch: BranchDef,
    data: np.ndarray,
) -> np.ndarray:
    """
    Return the subset of user_indices satisfying the branch condition.
    Uses vectorized NumPy operations.
    """
    col_vals = data[user_indices, branch.feature_idx]

    if branch.kind == FeatureKind.BINARY:
        val = next(iter(branch.value_set))
        mask = (col_vals == val)
    else:
        mask = (col_vals >= branch.lo) & (col_vals < branch.hi)

    mask = mask & ~np.isnan(col_vals)
    return user_indices[mask]


def health_distribution(user_indices: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    classes = labels[user_indices]
    dist = defaultdict(int)
    for c in classes:
        dist[int(c)] += 1
    return dict(sorted(dist.items()))


def compute_branch_probability(branch_user_count: int, parent_user_count: int) -> float:
    if parent_user_count == 0:
        return 0.0
    return branch_user_count / parent_user_count


def check_branching_condition(
    branch_user_count: int,
    parent_user_count: int,
    threshold: float = DIAGNOSTIC_THRESHOLD,
) -> Tuple[bool, int, int]:
    """
    Eq. 2 reduces exactly to branch_count >= u_min.
    """
    u_min_val = math.ceil(5 / threshold)
    lhs = branch_user_count
    return (lhs >= u_min_val), lhs, u_min_val


def compute_child_feature_set(
    parent_feature_indices: List[int],
    branching_feature_k: int,
    focus_level_m: int,
) -> List[int]:
    if focus_level_m <= 2:
        return list(parent_feature_indices)
    return [f for f in parent_feature_indices if f > branching_feature_k]


@dataclass
class TreeNode:
    node_id: str
    focus_level: int
    branching_feat_k: int
    branch_f: int
    branch_def: Optional[BranchDef]
    user_indices: np.ndarray
    feature_indices: List[int]
    branch_prob: float
    health_dist: Dict[int, int]
    children_by_k: Dict[int, List["TreeNode"]] = field(default_factory=dict)
    is_leaf: bool = False
    prune_reason: str = ""

    @property
    def n_users(self) -> int:
        return len(self.user_indices)

    @property
    def n_features(self) -> int:
        return len(self.feature_indices)

    @property
    def n_healthy(self) -> int:
        return self.health_dist.get(HEALTHY_CLASS, 0)

    @property
    def n_diseased(self) -> int:
        return self.n_users - self.n_healthy

    @property
    def all_children(self) -> List["TreeNode"]:
        out = []
        for lst in self.children_by_k.values():
            out.extend(lst)
        return out

    def add_child(self, child: "TreeNode") -> None:
        k = child.branching_feat_k
        if k not in self.children_by_k:
            self.children_by_k[k] = []
        self.children_by_k[k].append(child)

    def __repr__(self) -> str:
        return (
            f"TreeNode({self.node_id!r}, m={self.focus_level}, "
            f"k={self.branching_feat_k}, f={self.branch_f}, "
            f"n_users={self.n_users}, n_feat={self.n_features})"
        )


@dataclass
class DecisionTree:
    root: TreeNode
    nodes_by_level: Dict[int, List[TreeNode]] = field(default_factory=dict)
    all_nodes: Dict[str, TreeNode] = field(default_factory=dict)
    valid_branches: Dict[int, int] = field(default_factory=dict)
    pruned_branches: Dict[int, int] = field(default_factory=dict)
    feature_kinds: Dict[int, FeatureKind] = field(default_factory=dict)
    threshold: float = DIAGNOSTIC_THRESHOLD
    u_min: int = U_MIN

    def register(self, node: TreeNode) -> None:
        m = node.focus_level
        if m not in self.nodes_by_level:
            self.nodes_by_level[m] = []
        self.nodes_by_level[m].append(node)
        self.all_nodes[node.node_id] = node

    def depth(self) -> int:
        return max(self.nodes_by_level.keys()) if self.nodes_by_level else 1

    def count_nodes(self) -> int:
        return len(self.all_nodes)


def _try_split_node(
    parent: TreeNode,
    feature_k: int,
    kind: FeatureKind,
    data: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Tuple[List[TreeNode], int]:
    m_child = parent.focus_level + 1

    branch_defs = compute_fsd_branches(
        feature_idx=feature_k,
        user_indices=parent.user_indices,
        data=data,
        kind=kind,
    )
    if len(branch_defs) < 2:
        return [], 0

    branch_user_sets: List[Tuple[BranchDef, np.ndarray]] = []
    for bdef in branch_defs:
        branch_users = filter_users_by_branch(parent.user_indices, bdef, data)
        branch_user_sets.append((bdef, branch_users))

    created: List[TreeNode] = []
    n_pruned = 0
    for bdef, busers in branch_user_sets:
        passes, lhs, u_min_val = check_branching_condition(
            branch_user_count=len(busers),
            parent_user_count=parent.n_users,
            threshold=threshold,
        )
        if not passes:
            n_pruned += 1
            continue

        p_f = compute_branch_probability(len(busers), parent.n_users)
        feats = compute_child_feature_set(
            parent_feature_indices=parent.feature_indices,
            branching_feature_k=feature_k,
            focus_level_m=m_child,
        )
        child_id = f"{parent.node_id}|k{feature_k}_f{bdef.branch_idx}"
        child = TreeNode(
            node_id=child_id,
            focus_level=m_child,
            branching_feat_k=feature_k,
            branch_f=bdef.branch_idx,
            branch_def=bdef,
            user_indices=busers,
            feature_indices=feats,
            branch_prob=p_f,
            health_dist=health_distribution(busers, labels),
        )
        created.append(child)

    if len(created) == 1:
        created[0].is_leaf = True
        created[0].prune_reason = "Single-branch split; further branching disabled."

    return created, n_pruned


def _expand_node(
    parent: TreeNode,
    data: np.ndarray,
    labels: np.ndarray,
    kinds: Dict[int, FeatureKind],
    threshold: float,
    tree: DecisionTree,
) -> int:
    m_child = parent.focus_level + 1
    n_created = 0
    n_pruned = 0
    valid_ks = []

    log.debug(
        f"  Expanding {parent.node_id!r} -> m={m_child}, "
        f"trying {len(parent.feature_indices)} features …"
    )

    for k in sorted(parent.feature_indices):
        kind = kinds[k]
        children, pruned = _try_split_node(
            parent=parent,
            feature_k=k,
            kind=kind,
            data=data,
            labels=labels,
            threshold=threshold,
        )
        n_pruned += pruned
        if children:
            valid_ks.append(k)
            n_created += len(children)
            for child in children:
                parent.add_child(child)
                tree.register(child)

    if m_child not in tree.valid_branches:
        tree.valid_branches[m_child] = 0
        tree.pruned_branches[m_child] = 0
    tree.valid_branches[m_child] += n_created
    tree.pruned_branches[m_child] += n_pruned

    log.info(
        f"  m={parent.focus_level}->{m_child} | node {parent.node_id!r} | "
        f"valid splits: {len(valid_ks)} features -> {n_created} children | "
        f"pruned branches: {n_pruned}"
    )
    if not valid_ks:
        parent.is_leaf = True
        parent.prune_reason = f"No feature split passes Eq.2 at m={m_child}"
    return n_created


def build_decision_tree(
    data: np.ndarray,
    labels: np.ndarray,
    threshold: float = DIAGNOSTIC_THRESHOLD,
    max_m: int = 2,
) -> DecisionTree:
    log.info("=" * 72)
    log.info("Algorithm 1: Building CDS Decision Tree")
    log.info(f"  threshold = {threshold},  u_min = {math.ceil(5/threshold)}")
    log.info(f"  N_users   = {len(labels)},  N_features = {data.shape[1]}")
    log.info("=" * 72)

    if data.shape[1] != N_FEATURES:
        raise ValueError(
            f"Algorithm 1 expects exactly {N_FEATURES} features, but got {data.shape[1]}. "
            "If you are using WFDB waveform records, you must first extract/construct "
            "a 279-feature tabular representation."
        )

    kinds = classify_features(data)
    n_binary = sum(1 for k in kinds.values() if k == FeatureKind.BINARY)
    n_cont = sum(1 for k in kinds.values() if k == FeatureKind.CONTINUOUS)
    log.info(f"Feature classification: {n_binary} binary, {n_cont} continuous.")

    all_user_indices = np.arange(len(labels), dtype=int)
    all_feature_indices = list(range(N_FEATURES))
    root = TreeNode(
        node_id="root",
        focus_level=1,
        branching_feat_k=-1,
        branch_f=0,
        branch_def=None,
        user_indices=all_user_indices,
        feature_indices=all_feature_indices,
        branch_prob=1.0,
        health_dist=health_distribution(all_user_indices, labels),
    )

    tree = DecisionTree(
        root=root,
        feature_kinds=kinds,
        threshold=threshold,
        u_min=math.ceil(5 / threshold),
    )
    tree.register(root)
    tree.nodes_by_level[1] = [root]

    log.info(f"Root node: {root.n_users} users, {root.n_features} features.")
    log.info(f"  Health distribution: {root.health_dist}")

    current_level_nodes = [root]

    for m in range(2, max_m + 1):
        log.info(f"\n{'─'*60}")
        log.info(f"FOCUS LEVEL m = {m}: expanding {len(current_level_nodes)} parent node(s) …")

        next_level_nodes = []
        total_new = 0

        for parent_node in current_level_nodes:
            n_new = _expand_node(
                parent=parent_node,
                data=data,
                labels=labels,
                kinds=kinds,
                threshold=threshold,
                tree=tree,
            )
            total_new += n_new
            next_level_nodes.extend(parent_node.all_children)

        log.info(f"  Total new nodes at m={m}: {total_new}")

        if total_new == 0:
            log.info(f"  No valid branches at m={m}. Tree construction complete.")
            break

        current_level_nodes = next_level_nodes

    log.info(f"\n{'='*72}")
    log.info(f"Algorithm 1 complete. Tree depth: {tree.depth()}  |  Total nodes: {tree.count_nodes()}")
    return tree


def print_tree_summary(tree: DecisionTree) -> None:
    print("\n" + "=" * 90)
    print(f"{'CDS DECISION TREE SUMMARY':^90}")
    print(
        f"  threshold={tree.threshold}, u_min={tree.u_min}, depth={tree.depth()}, "
        f"total_nodes={tree.count_nodes()}"
    )
    print("=" * 90)
    print(f"{'m':>3}  {'k':>4}  {'f':>3}  {'|U|':>5}  {'%H':>6}  {'P_f':>6}  {'#feat':>5}  label")
    print("-" * 90)
    for m in sorted(tree.nodes_by_level.keys()):
        for node in tree.nodes_by_level[m]:
            pct_h = 100 * node.n_healthy / node.n_users if node.n_users else 0
            lbl = node.branch_def.label if node.branch_def else "—root—"
            print(
                f"{m:>3}  {node.branching_feat_k:>4}  {node.branch_f:>3}  "
                f"{node.n_users:>5}  {pct_h:>5.1f}%  {node.branch_prob:>6.4f}  "
                f"{node.n_features:>5}  {lbl}"
            )
        if m < max(tree.nodes_by_level.keys()):
            print()
    print("=" * 90)


def print_node_details(node: TreeNode, feature_kinds: Dict[int, FeatureKind]) -> None:
    print(f"\n{'─'*60}")
    print(f"NODE: {node.node_id}")
    print(f"  Focus level m     : {node.focus_level}")
    print(
        f"  Branching feat k  : {node.branching_feat_k} "
        f"({FEATURE_NAMES.get(node.branching_feat_k,'root')})"
    )
    print(f"  Branch f          : {node.branch_f}")
    print(f"  Branch definition : {node.branch_def.label if node.branch_def else 'root'}")
    print(f"  Users (|U|)       : {node.n_users}")
    print(f"  Healthy/Diseased  : {node.n_healthy} / {node.n_diseased}")
    print(f"  Branch probability: {node.branch_prob:.6f}")
    print(f"  Available features: {node.n_features}")
    print(f"  Health distribution:")
    for cls, cnt in sorted(node.health_dist.items()):
        tag = " ← HEALTHY" if cls == HEALTHY_CLASS else ""
        print(f"    Class {cls:2d}: {cnt:3d}{tag}")
    print(f"  Children (by branching feature):")
    if node.children_by_k:
        for k, children in sorted(node.children_by_k.items()):
            kname = FEATURE_NAMES.get(k, f"feat_{k}")
            labels_str = ", ".join(
                f"f={c.branch_f}({c.branch_def.label if c.branch_def else '?'})->{c.n_users}u"
                for c in children
            )
            print(f"    k={k} ({kname}): {labels_str}")
    else:
        print(f"    (leaf node – {node.prune_reason or 'no further splits'})")


def print_level_statistics(tree: DecisionTree) -> None:
    print("\n" + "=" * 60)
    print(f"{'LEVEL STATISTICS':^60}")
    print("=" * 60)
    print(f"{'m':>3}  {'nodes':>6}  {'valid_br':>9}  {'pruned_br':>11}  {'avg_|U|':>8}")
    print("-" * 60)
    for m in sorted(tree.nodes_by_level.keys()):
        nodes = tree.nodes_by_level[m]
        avg_u = np.mean([n.n_users for n in nodes]) if nodes else 0
        valid = tree.valid_branches.get(m, 0)
        pruned = tree.pruned_branches.get(m, 0)
        print(f"{m:>3}  {len(nodes):>6}  {valid:>9}  {pruned:>11}  {avg_u:>8.1f}")
    print("=" * 60)


def print_branching_features_at_level(tree: DecisionTree, m: int) -> None:
    nodes_at_m = tree.nodes_by_level.get(m, [])
    if not nodes_at_m:
        print(f"\nNo nodes at focus level m={m}.")
        return

    by_parent_k: Dict[str, Dict[int, List[TreeNode]]] = defaultdict(lambda: defaultdict(list))
    for node in nodes_at_m:
        parent_id = node.node_id.rsplit("|", 1)[0]
        by_parent_k[parent_id][node.branching_feat_k].append(node)

    print(f"\n{'='*72}")
    print(f"VALID BRANCHING FEATURES AT m={m}  (u_min={tree.u_min})")
    print(f"{'='*72}")
    print(f"{'k':>4}  {'feature':20}  {'kind':12}  {'branches':>8}  branch_sizes")
    print("-" * 72)

    seen_ks: Set[int] = set()
    for parent_id, k_map in sorted(by_parent_k.items()):
        for k, children in sorted(k_map.items()):
            if k in seen_ks:
                continue
            seen_ks.add(k)
            fname = FEATURE_NAMES.get(k, f"feat_{k}")
            kind = tree.feature_kinds.get(k, FeatureKind.CONTINUOUS)
            sizes = ", ".join(f"{c.branch_def.label}:{c.n_users}" for c in children)
            print(f"{k:>4}  {fname:20}  {kind.name:12}  {len(children):>8}  {sizes}")
    print(f"{'='*72}")
    print(f"Total unique valid branching features at m={m}: {len(seen_ks)}")


def print_sex_branch_details(tree: DecisionTree) -> None:
    SEX_K = 1
    nodes_at_2 = tree.nodes_by_level.get(2, [])
    sex_nodes = [n for n in nodes_at_2 if n.branching_feat_k == SEX_K]

    print("\n" + "=" * 60)
    print(f"{'SEX-BRANCHED NODES (paper case study)':^60}")
    print("=" * 60)
    if not sex_nodes:
        print("  Sex (k=1) did not produce valid branches!")
        return

    for node in sex_nodes:
        lbl = node.branch_def.label if node.branch_def else "?"
        print(f"  Branch f={node.branch_f}  ({lbl})")
        print(f"    Users       : {node.n_users}")
        print(f"    Branch prob : {node.branch_prob:.4f}")
        print(f"    Healthy     : {node.n_healthy} ({100*node.n_healthy/node.n_users:.1f}%)")
        print(f"    Diseased    : {node.n_diseased} ({100*node.n_diseased/node.n_users:.1f}%)")
        print(f"    Health dist : {node.health_dist}")
        passes_eq2, lhs, u_min_val = check_branching_condition(
            node.n_users, tree.root.n_users, tree.threshold
        )
        print(
            f"    Eq.2 check  : ceil({tree.root.n_users}×{node.branch_prob:.4f})"
            f" = {lhs} >= {u_min_val}? {passes_eq2}"
        )
        print(f"    Features    : {node.n_features}")
        print()


def explain_level3_impossibility(tree: DecisionTree) -> None:
    print("\n" + "=" * 72)
    print("WHY FOCUS LEVEL 3 IS IMPOSSIBLE (Eq. 2 proof)")
    print("=" * 72)
    u_min = tree.u_min
    nodes_at_2 = tree.nodes_by_level.get(2, [])
    if not nodes_at_2:
        print("  No level-2 nodes to analyse.")
        return

    for parent in nodes_at_2:
        best_possible = parent.n_users // 2
        print(f"\n  Parent node: {parent.node_id!r}  |  {parent.n_users} users")
        print(f"    Best-case branch size (50/50 split): {best_possible}")
        print(f"    u_min required                     : {u_min}")
        print(f"    Best-case >= u_min?                : {best_possible >= u_min}")
        print(
            f"    -> LEVEL 3 {'POSSIBLE' if best_possible >= u_min else 'IMPOSSIBLE'} "
            f"for this parent node."
        )


def validate_user_partition(tree: DecisionTree) -> None:
    print("\n" + "=" * 60)
    print("USER-PARTITION VALIDATION")
    print("=" * 60)
    all_ok = True

    for m in sorted(tree.nodes_by_level.keys()):
        if m == 1:
            continue
        groups: Dict[Tuple[str, int], List[TreeNode]] = defaultdict(list)
        for node in tree.nodes_by_level[m]:
            parent_id = node.node_id.rsplit("|", 1)[0]
            groups[(parent_id, node.branching_feat_k)].append(node)

        for (parent_id, k), siblings in groups.items():
            parent = tree.all_nodes.get(parent_id)
            if parent is None:
                continue
            union_sets = [set(s.user_indices.tolist()) for s in siblings]
            sibling_union = set().union(*union_sets)
            ok_disjoint = True
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    inter = union_sets[i] & union_sets[j]
                    if inter:
                        ok_disjoint = False
                        print(
                            f"  FAIL disjoint: m={m} k={k} "
                            f"f={siblings[i].branch_f}∩f={siblings[j].branch_f} "
                            f"has {len(inter)} common users!"
                        )
            ok_subset = sibling_union.issubset(set(parent.user_indices.tolist()))
            status = "PASS" if (ok_disjoint and ok_subset) else "FAIL"
            if not (ok_disjoint and ok_subset):
                all_ok = False
            print(
                f"  m={m} | k={k} ({FEATURE_NAMES.get(k,'?')}) | "
                f"parent={parent.n_users}u | children={[s.n_users for s in siblings]} | "
                f"union={len(sibling_union)} | disjoint={ok_disjoint} | "
                f"subset={ok_subset} -> {status}"
            )

    print(f"\n  Overall validation: {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")


def validate_eq2(tree: DecisionTree) -> None:
    print("\n" + "=" * 60)
    print("EQ. 2 VALIDATION (all non-root nodes)")
    print("=" * 60)
    all_ok = True
    for m in sorted(tree.nodes_by_level.keys()):
        if m == 1:
            continue
        for node in tree.nodes_by_level[m]:
            parent_id = node.node_id.rsplit("|", 1)[0]
            parent = tree.all_nodes.get(parent_id)
            if parent is None:
                continue
            passes, lhs, u_min_val = check_branching_condition(
                node.n_users, parent.n_users, tree.threshold
            )
            if not passes:
                all_ok = False
                print(f"  FAIL: {node.node_id!r} | lhs={lhs} < u_min={u_min_val}")
            else:
                print(f"  PASS: {node.node_id!r} | {lhs} >= {u_min_val}")
    if all_ok:
        print("  All nodes satisfy Eq. 2 ✓")


def validate_feature_exclusion(tree: DecisionTree) -> None:
    print("\n" + "=" * 60)
    print("FEATURE EXCLUSION VALIDATION (line 9, m>2 nodes only)")
    print("=" * 60)
    violations = 0
    for m in sorted(tree.nodes_by_level.keys()):
        if m <= 2:
            print(f"  m={m}: exclusion rule does NOT apply (line 8 condition: m>2)")
            continue
        for node in tree.nodes_by_level[m]:
            k = node.branching_feat_k
            bad = [f for f in node.feature_indices if f <= k]
            if bad:
                violations += 1
                print(f"  FAIL: {node.node_id!r} | k={k} but features <= k present: {bad}")
            else:
                print(
                    f"  PASS: {node.node_id!r} | k={k} | "
                    f"features all > k ✓ ({node.n_features} remaining)"
                )
    if violations == 0:
        print("  All m>2 nodes satisfy feature exclusion rule ✓")


def main(
    wfdb_database: str = "mitdb",
    ann_ext: str = "atr",
    max_records: Optional[int] = None,
) -> DecisionTree:
    data, labels = load_dataset(
        wfdb_database=wfdb_database,
        ann_ext=ann_ext,
        max_records=max_records,
    )
    tree = build_decision_tree(data, labels)

    print_tree_summary(tree)
    print_level_statistics(tree)
    print_branching_features_at_level(tree, m=2)
    print_sex_branch_details(tree)
    explain_level3_impossibility(tree)
    validate_eq2(tree)
    validate_feature_exclusion(tree)

    for node in tree.nodes_by_level.get(2, []):
        if node.branching_feat_k == 1:
            print_node_details(node, tree.feature_kinds)

    return tree


if __name__ == "__main__":
    wfdb_database = sys.argv[1] if len(sys.argv) > 1 else "mitdb"
    tree = main(wfdb_database=wfdb_database)