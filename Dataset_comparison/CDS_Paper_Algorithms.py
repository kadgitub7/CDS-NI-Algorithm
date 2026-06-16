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

Algorithm 1 still requires a tabular feature matrix of shape (N, D) and labels
of shape (N,), where D is inferred from the feature-extraction stage. WFDB
records themselves are waveform data, so unless you already have a tabular
feature table derived from WFDB records, you must add a feature-extraction stage
before calling the tree builder.

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
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
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

# WFDB convention adopted for this implementation:
# "N" (Normal beat) is treated as the healthy class.
WFDB_HEALTHY_SYMBOL: str = "N"


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


@dataclass
class DatasetSchema:
    """
    Generic schema/configuration for tabular data derived from WFDB or other sources.

    Attributes
    ----------
    feature_names : Optional[Dict[int, str]]
        Optional explicit feature-name mapping. If omitted, generic names
        feat_0, feat_1, ... are used.
    expected_n_features : Optional[int]
        Optional fixed dimensionality constraint. If None, the code accepts the
        inferred tabular width.
    healthy_class_ids : Optional[FrozenSet[int]]
        Optional set of integer class IDs to be treated as "healthy".
        In this WFDB version, this is derived automatically from the record-level
        class mapping whenever the WFDB symbol "N" is present.
    sex_feature_index : Optional[int]
        Optional index of a sex/gender feature, if one exists in the tabular data.
    sex_value_labels : Optional[Dict[float, str]]
        Optional map from numeric sex-feature values to display labels.
        Example: {0.0: "male", 1.0: "female"}.
    missing_value_cols : FrozenSet[int]
        Optional metadata only; not used in core tree logic.
    """
    feature_names: Optional[Dict[int, str]] = None
    expected_n_features: Optional[int] = None
    healthy_class_ids: Optional[FrozenSet[int]] = None
    sex_feature_index: Optional[int] = None
    sex_value_labels: Optional[Dict[float, str]] = None
    missing_value_cols: FrozenSet[int] = frozenset()

    def get_feature_name(self, feature_idx: int) -> str:
        if self.feature_names and feature_idx in self.feature_names:
            return self.feature_names[feature_idx]
        return f"feat_{feature_idx}"


@dataclass
class LabelMetadata:
    """
    Metadata extracted from WFDB annotations so the caller can inspect all labels
    and later decide which classes should be considered healthy or diseased.

    In this implementation:
      - WFDB symbol "N" is treated as healthy if present in record-level labels.
      - All other record-level classes are treated as unhealthy.
    """
    record_to_symbol: Dict[str, str] = field(default_factory=dict)
    symbol_counts: Dict[str, int] = field(default_factory=dict)
    symbol_to_int: Dict[str, int] = field(default_factory=dict)
    int_to_symbol: Dict[int, str] = field(default_factory=dict)
    all_symbols_seen_in_annotations: List[str] = field(default_factory=list)
    records_without_annotations: List[str] = field(default_factory=list)
    records_with_failed_annotation_read: List[str] = field(default_factory=list)


@dataclass
class LoadedDataset:
    data: np.ndarray
    labels: np.ndarray
    schema: DatasetSchema
    label_metadata: LabelMetadata


def make_generic_feature_names(n_features: int) -> Dict[int, str]:
    return {i: f"feat_{i}" for i in range(n_features)}


def initialize_dataset_schema(
    n_features: int,
    feature_names: Optional[Dict[int, str]] = None,
    expected_n_features: Optional[int] = None,
    healthy_class_ids: Optional[FrozenSet[int]] = None,
    sex_feature_index: Optional[int] = None,
    sex_value_labels: Optional[Dict[float, str]] = None,
    missing_value_cols: Optional[FrozenSet[int]] = None,
) -> DatasetSchema:
    """
    Create a generic schema for the inferred tabular dataset.
    """
    resolved_feature_names = (
        feature_names if feature_names is not None else make_generic_feature_names(n_features)
    )
    return DatasetSchema(
        feature_names=resolved_feature_names,
        expected_n_features=expected_n_features,
        healthy_class_ids=healthy_class_ids,
        sex_feature_index=sex_feature_index,
        sex_value_labels=sex_value_labels,
        missing_value_cols=missing_value_cols or frozenset(),
    )


def classify_features(data: np.ndarray) -> Dict[int, FeatureKind]:
    """
    Classify each feature as BINARY or CONTINUOUS.
    """
    n_features = data.shape[1]
    kinds: Dict[int, FeatureKind] = {}
    for col in range(n_features):
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


def _extract_simple_record_features(sig: np.ndarray) -> np.ndarray:
    """
    [ENGR] Minimal generic feature extraction from a WFDB signal matrix.

    Produces a tabular vector by concatenating per-channel:
      mean, std, min, max

    If sig has shape (T, C), output has shape (4*C,).
    """
    if sig.ndim == 1:
        sig = sig[:, None]

    if sig.ndim != 2 or sig.shape[0] == 0 or sig.shape[1] == 0:
        raise ValueError(f"Invalid signal shape: {sig.shape}")

    return np.concatenate([
        np.nanmean(sig, axis=0),
        np.nanstd(sig, axis=0),
        np.nanmin(sig, axis=0),
        np.nanmax(sig, axis=0),
    ]).astype(float)


def _derive_record_label_symbol(
    rec_name: str,
    wfdb_database: str,
    ann_ext: str,
) -> Tuple[str, List[str], bool]:
    """
    Derive one record-level label symbol from WFDB annotations.

    Returns
    -------
    label_symbol : str
        A single representative symbol for the record. Here we use the most
        frequent annotation symbol if present; otherwise 'NO_ANN'.
    all_symbols : List[str]
        All annotation symbols observed for that record.
    ann_read_failed : bool
        Whether annotation reading failed entirely.
    """
    try:
        ann = wfdb.rdann(rec_name, ann_ext, pn_dir=wfdb_database)
        symbols = getattr(ann, "symbol", None)
        if symbols is None or len(symbols) == 0:
            return "NO_ANN", [], False

        counts = Counter(symbols)
        label_symbol = max(counts, key=counts.get)
        return label_symbol, list(symbols), False
    except Exception:
        return "NO_ANN", [], True


def _infer_healthy_class_ids_from_wfdb_symbols(
    symbol_to_int: Dict[str, int],
    healthy_symbol: str = WFDB_HEALTHY_SYMBOL,
) -> Optional[FrozenSet[int]]:
    """
    Infer healthy class IDs from the WFDB record-level symbol mapping.

    Assumption requested by user:
      - "N" (Normal beat) is healthy
      - every other symbol is unhealthy
    """
    if healthy_symbol in symbol_to_int:
        return frozenset({symbol_to_int[healthy_symbol]})
    return None


def _describe_health_mapping(
    symbol_to_int: Dict[str, int],
    healthy_class_ids: Optional[FrozenSet[int]],
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    healthy: List[Tuple[int, str]] = []
    unhealthy: List[Tuple[int, str]] = []

    int_to_symbol = {i: s for s, i in symbol_to_int.items()}
    for class_id in sorted(int_to_symbol.keys()):
        symbol = int_to_symbol[class_id]
        if healthy_class_ids is not None and class_id in healthy_class_ids:
            healthy.append((class_id, symbol))
        else:
            unhealthy.append((class_id, symbol))
    return healthy, unhealthy


def load_wfdb_dataset(
    wfdb_database: str,
    ann_ext: str = "atr",
    max_records: Optional[int] = None,
    expected_n_features: Optional[int] = None,
    feature_names: Optional[Dict[int, str]] = None,
    healthy_class_ids: Optional[FrozenSet[int]] = None,
    sex_feature_index: Optional[int] = None,
    sex_value_labels: Optional[Dict[float, str]] = None,
) -> LoadedDataset:
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
    expected_n_features : Optional[int]
        If provided, validates the extracted tabular width against this value.
        If None, any inferred width is accepted.
    feature_names : Optional[Dict[int, str]]
        Optional feature-name map.
    healthy_class_ids : Optional[FrozenSet[int]]
        Optional set of integer class IDs considered healthy. If omitted, this
        WFDB implementation infers healthy_class_ids by treating "N" as healthy.
    sex_feature_index : Optional[int]
        Optional sex/gender feature index.
    sex_value_labels : Optional[Dict[float, str]]
        Optional labels for values in the sex/gender feature.

    Returns
    -------
    LoadedDataset
        Contains:
        - data: tabular feature matrix
        - labels: integer class labels
        - schema: generic schema
        - label_metadata: all discovered annotation-symbol metadata

    IMPORTANT
    ---------
    Algorithm 1 requires tabular features. WFDB waveform records therefore need
    a feature-extraction stage; this loader includes a simple generic one.
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
    kept_record_names: List[str] = []
    skipped_records: List[Tuple[str, str]] = []

    label_meta = LabelMetadata()

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

        try:
            feat = _extract_simple_record_features(sig)
        except Exception as e:
            skipped_records.append((rec_name, f"feature extraction failed: {e}"))
            continue

        if feat.size == 0:
            skipped_records.append((rec_name, "empty feature vector"))
            continue

        label_sym, all_syms, ann_failed = _derive_record_label_symbol(
            rec_name=rec_name,
            wfdb_database=wfdb_database,
            ann_ext=ann_ext,
        )

        if ann_failed:
            label_meta.records_with_failed_annotation_read.append(rec_name)
        elif len(all_syms) == 0:
            label_meta.records_without_annotations.append(rec_name)

        for s in all_syms:
            label_meta.symbol_counts[s] = label_meta.symbol_counts.get(s, 0) + 1

        features.append(feat)
        label_symbols.append(label_sym)
        feature_dims_seen.append(feat.size)
        kept_record_names.append(rec_name)
        label_meta.record_to_symbol[rec_name] = label_sym

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
    filtered_record_names: List[str] = []
    n_dropped_dim_mismatch = 0

    for feat, lab, rec_name in zip(features, label_symbols, kept_record_names):
        if feat.size == target_dim:
            filtered_features.append(feat)
            filtered_labels.append(lab)
            filtered_record_names.append(rec_name)
        else:
            n_dropped_dim_mismatch += 1

    if not filtered_features:
        raise ValueError(
            "WFDB loader extracted records, but none shared a common feature dimension. "
            f"Observed dimensions: {dict(sorted(dim_counts.items()))}"
        )

    data = np.vstack(filtered_features)

    if expected_n_features is not None and data.shape[1] != expected_n_features:
        raise ValueError(
            f"WFDB feature extraction produced {data.shape[1]} features, but "
            f"expected_n_features={expected_n_features}."
        )

    unique_syms = sorted(set(filtered_labels))
    sym_to_int = {s: i + 1 for i, s in enumerate(unique_syms)}
    int_to_sym = {i: s for s, i in sym_to_int.items()}
    labels = np.array([sym_to_int[s] for s in filtered_labels], dtype=int)

    label_meta.symbol_to_int = sym_to_int
    label_meta.int_to_symbol = int_to_sym
    label_meta.all_symbols_seen_in_annotations = sorted(label_meta.symbol_counts.keys())

    if healthy_class_ids is None:
        healthy_class_ids = _infer_healthy_class_ids_from_wfdb_symbols(
            sym_to_int, healthy_symbol=WFDB_HEALTHY_SYMBOL
        )

    schema = initialize_dataset_schema(
        n_features=data.shape[1],
        feature_names=feature_names,
        expected_n_features=expected_n_features,
        healthy_class_ids=healthy_class_ids,
        sex_feature_index=sex_feature_index,
        sex_value_labels=sex_value_labels,
    )

    log.info(
        f"WFDB loader: loaded {data.shape[0]} records, feature_dim={data.shape[1]}, "
        f"record_level_classes={len(unique_syms)}, dropped_dim_mismatch={n_dropped_dim_mismatch}, "
        f"skipped_unreadable={len(skipped_records)}"
    )

    if label_meta.symbol_counts:
        top_symbols = sorted(label_meta.symbol_counts.items(), key=lambda x: (-x[1], x[0]))[:15]
        log.info(f"Top annotation symbols observed across records: {top_symbols}")
    else:
        log.info("No readable annotation symbols were collected from WFDB annotations.")

    healthy_desc, unhealthy_desc = _describe_health_mapping(sym_to_int, schema.healthy_class_ids)
    log.info(f"Healthy classes (WFDB 'N' only): {healthy_desc if healthy_desc else 'none detected'}")
    log.info(f"Unhealthy classes (all others): {unhealthy_desc if unhealthy_desc else 'none detected'}")

    return LoadedDataset(
        data=data,
        labels=labels,
        schema=schema,
        label_metadata=label_meta,
    )


def load_dataset(
    wfdb_database: str,
    ann_ext: str = "atr",
    max_records: Optional[int] = None,
    expected_n_features: Optional[int] = None,
    feature_names: Optional[Dict[int, str]] = None,
    healthy_class_ids: Optional[FrozenSet[int]] = None,
    sex_feature_index: Optional[int] = None,
    sex_value_labels: Optional[Dict[float, str]] = None,
) -> LoadedDataset:
    return load_wfdb_dataset(
        wfdb_database=wfdb_database,
        ann_ext=ann_ext,
        max_records=max_records,
        expected_n_features=expected_n_features,
        feature_names=feature_names,
        healthy_class_ids=healthy_class_ids,
        sex_feature_index=sex_feature_index,
        sex_value_labels=sex_value_labels,
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


def _fsd_binary(
    feature_idx: int,
    user_indices: np.ndarray,
    data: np.ndarray,
    schema: DatasetSchema,
) -> List[BranchDef]:
    col = data[user_indices, feature_idx]
    valid = col[~np.isnan(col)]
    values_present = sorted(set(valid.tolist()))
    if len(values_present) < 2:
        return []

    fname = schema.get_feature_name(feature_idx)
    branches = []
    for bidx, v in enumerate(values_present, start=1):
        if schema.sex_feature_index is not None and feature_idx == schema.sex_feature_index:
            if schema.sex_value_labels and v in schema.sex_value_labels:
                lbl = schema.sex_value_labels[v]
            else:
                lbl = f"{fname}={v:g}"
        else:
            if float(v).is_integer():
                lbl = f"{fname}={int(v)}"
            else:
                lbl = f"{fname}={v:g}"

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


def _fsd_continuous_median(
    feature_idx: int,
    user_indices: np.ndarray,
    data: np.ndarray,
    schema: DatasetSchema,
) -> List[BranchDef]:
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

    fname = schema.get_feature_name(feature_idx)
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
    schema: DatasetSchema,
) -> List[BranchDef]:
    if kind == FeatureKind.BINARY:
        return _fsd_binary(feature_idx, user_indices, data, schema)
    return _fsd_continuous_median(feature_idx, user_indices, data, schema)


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


def class_distribution(user_indices: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    classes = labels[user_indices]
    dist = defaultdict(int)
    for c in classes:
        dist[int(c)] += 1
    return dict(sorted(dist.items()))


def compute_healthy_count(
    class_dist: Dict[int, int],
    healthy_class_ids: Optional[FrozenSet[int]],
) -> Optional[int]:
    if healthy_class_ids is None:
        return None
    return sum(cnt for cls, cnt in class_dist.items() if cls in healthy_class_ids)


def compute_unhealthy_count(
    class_dist: Dict[int, int],
    healthy_class_ids: Optional[FrozenSet[int]],
) -> Optional[int]:
    if healthy_class_ids is None:
        return None
    return sum(cnt for cls, cnt in class_dist.items() if cls not in healthy_class_ids)


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
    class_dist: Dict[int, int]
    healthy_class_ids: Optional[FrozenSet[int]] = None
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
    def n_healthy(self) -> Optional[int]:
        return compute_healthy_count(self.class_dist, self.healthy_class_ids)

    @property
    def n_unhealthy(self) -> Optional[int]:
        return compute_unhealthy_count(self.class_dist, self.healthy_class_ids)

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
    schema: DatasetSchema
    label_metadata: Optional[LabelMetadata] = None
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
    schema: DatasetSchema,
) -> Tuple[List[TreeNode], int]:
    m_child = parent.focus_level + 1

    branch_defs = compute_fsd_branches(
        feature_idx=feature_k,
        user_indices=parent.user_indices,
        data=data,
        kind=kind,
        schema=schema,
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
            class_dist=class_distribution(busers, labels),
            healthy_class_ids=schema.healthy_class_ids,
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
            schema=tree.schema,
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
    schema: Optional[DatasetSchema] = None,
    threshold: float = DIAGNOSTIC_THRESHOLD,
    max_m: int = 2,
    label_metadata: Optional[LabelMetadata] = None,
) -> DecisionTree:
    n_features = data.shape[1]

    if schema is None:
        schema = initialize_dataset_schema(n_features=n_features)

    log.info("=" * 72)
    log.info("Algorithm 1: Building CDS Decision Tree")
    log.info(f"  threshold = {threshold},  u_min = {math.ceil(5/threshold)}")
    log.info(f"  N_users   = {len(labels)},  N_features = {data.shape[1]}")
    log.info("=" * 72)

    if schema.expected_n_features is not None and data.shape[1] != schema.expected_n_features:
        raise ValueError(
            f"Algorithm 1 expected {schema.expected_n_features} features, "
            f"but got {data.shape[1]}."
        )

    kinds = classify_features(data)
    n_binary = sum(1 for k in kinds.values() if k == FeatureKind.BINARY)
    n_cont = sum(1 for k in kinds.values() if k == FeatureKind.CONTINUOUS)
    log.info(f"Feature classification: {n_binary} binary, {n_cont} continuous.")

    all_user_indices = np.arange(len(labels), dtype=int)
    all_feature_indices = list(range(n_features))
    root = TreeNode(
        node_id="root",
        focus_level=1,
        branching_feat_k=-1,
        branch_f=0,
        branch_def=None,
        user_indices=all_user_indices,
        feature_indices=all_feature_indices,
        branch_prob=1.0,
        class_dist=class_distribution(all_user_indices, labels),
        healthy_class_ids=schema.healthy_class_ids,
    )

    tree = DecisionTree(
        root=root,
        schema=schema,
        label_metadata=label_metadata,
        feature_kinds=kinds,
        threshold=threshold,
        u_min=math.ceil(5 / threshold),
    )
    tree.register(root)
    tree.nodes_by_level[1] = [root]

    log.info(f"Root node: {root.n_users} users, {root.n_features} features.")
    log.info(f"  Class distribution: {root.class_dist}")

    if schema.healthy_class_ids is not None:
        log.info(
            f"  Healthy users: {root.n_healthy} | Unhealthy users: {root.n_unhealthy} "
            f"(healthy = WFDB symbol '{WFDB_HEALTHY_SYMBOL}' only)"
        )

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


def print_wfdb_label_report(label_metadata: LabelMetadata, schema: DatasetSchema) -> None:
    print("\n" + "=" * 90)
    print(f"{'WFDB LABEL / ANNOTATION REPORT':^90}")
    print("=" * 90)

    print("Record-level label mapping (symbol -> integer class id):")
    if label_metadata.symbol_to_int:
        for sym, idx in sorted(label_metadata.symbol_to_int.items(), key=lambda x: x[1]):
            tag = "  <- HEALTHY" if schema.healthy_class_ids and idx in schema.healthy_class_ids else ""
            print(f"  class_id={idx:>3}  symbol={sym!r}{tag}")
    else:
        print("  No record-level labels were derived.")

    print("\nHealthy / unhealthy grouping used:")
    if schema.healthy_class_ids is not None:
        healthy, unhealthy = _describe_health_mapping(label_metadata.symbol_to_int, schema.healthy_class_ids)
        print(f"  Healthy classes   : {healthy}")
        print(f"  Unhealthy classes : {unhealthy}")
    else:
        print("  Healthy grouping could not be inferred because WFDB symbol 'N' was not present.")

    print("\nAll annotation symbols observed across readable annotations:")
    if label_metadata.all_symbols_seen_in_annotations:
        print("  " + ", ".join(repr(s) for s in label_metadata.all_symbols_seen_in_annotations))
    else:
        print("  None")

    print("\nAnnotation symbol counts:")
    if label_metadata.symbol_counts:
        for sym, cnt in sorted(label_metadata.symbol_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {sym!r:>8}: {cnt}")
    else:
        print("  No annotation symbols counted.")

    print("\nRecords without readable/usable annotations:")
    print(f"  no_annotations={len(label_metadata.records_without_annotations)}")
    print(f"  annotation_read_failures={len(label_metadata.records_with_failed_annotation_read)}")
    print("=" * 90)


def print_tree_summary(tree: DecisionTree) -> None:
    print("\n" + "=" * 90)
    print(f"{'CDS DECISION TREE SUMMARY':^90}")
    print(
        f"  threshold={tree.threshold}, u_min={tree.u_min}, depth={tree.depth()}, "
        f"total_nodes={tree.count_nodes()}"
    )
    print("=" * 90)

    has_healthy = tree.schema.healthy_class_ids is not None
    if has_healthy:
        print(f"{'m':>3}  {'k':>4}  {'f':>3}  {'|U|':>5}  {'%H':>6}  {'P_f':>6}  {'#feat':>5}  label")
    else:
        print(f"{'m':>3}  {'k':>4}  {'f':>3}  {'|U|':>5}  {'P_f':>6}  {'#feat':>5}  label")
    print("-" * 90)

    for m in sorted(tree.nodes_by_level.keys()):
        for node in tree.nodes_by_level[m]:
            lbl = node.branch_def.label if node.branch_def else "—root—"
            if has_healthy:
                nh = node.n_healthy or 0
                pct_h = 100 * nh / node.n_users if node.n_users else 0
                print(
                    f"{m:>3}  {node.branching_feat_k:>4}  {node.branch_f:>3}  "
                    f"{node.n_users:>5}  {pct_h:>5.1f}%  {node.branch_prob:>6.4f}  "
                    f"{node.n_features:>5}  {lbl}"
                )
            else:
                print(
                    f"{m:>3}  {node.branching_feat_k:>4}  {node.branch_f:>3}  "
                    f"{node.n_users:>5}  {node.branch_prob:>6.4f}  "
                    f"{node.n_features:>5}  {lbl}"
                )
        if m < max(tree.nodes_by_level.keys()):
            print()
    print("=" * 90)


def print_node_details(node: TreeNode, feature_kinds: Dict[int, FeatureKind], schema: DatasetSchema) -> None:
    print(f"\n{'─'*60}")
    print(f"NODE: {node.node_id}")
    print(f"  Focus level m     : {node.focus_level}")
    print(
        f"  Branching feat k  : {node.branching_feat_k} "
        f"({schema.get_feature_name(node.branching_feat_k) if node.branching_feat_k >= 0 else 'root'})"
    )
    print(f"  Branch f          : {node.branch_f}")
    print(f"  Branch definition : {node.branch_def.label if node.branch_def else 'root'}")
    print(f"  Users (|U|)       : {node.n_users}")

    if node.n_healthy is not None and node.n_unhealthy is not None:
        print(f"  Healthy/Unhealthy : {node.n_healthy} / {node.n_unhealthy}")
    else:
        print("  Healthy/Unhealthy : not defined (healthy_class_ids not configured)")

    print(f"  Branch probability: {node.branch_prob:.6f}")
    print(f"  Available features: {node.n_features}")
    print(f"  Class distribution:")
    for cls, cnt in sorted(node.class_dist.items()):
        print(f"    Class {cls:2d}: {cnt:3d}")
    print(f"  Children (by branching feature):")
    if node.children_by_k:
        for k, children in sorted(node.children_by_k.items()):
            kname = schema.get_feature_name(k)
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

    seen_ks: set[int] = set()
    for parent_id, k_map in sorted(by_parent_k.items()):
        for k, children in sorted(k_map.items()):
            if k in seen_ks:
                continue
            seen_ks.add(k)
            fname = tree.schema.get_feature_name(k)
            kind = tree.feature_kinds.get(k, FeatureKind.CONTINUOUS)
            sizes = ", ".join(f"{c.branch_def.label}:{c.n_users}" for c in children)
            print(f"{k:>4}  {fname:20}  {kind.name:12}  {len(children):>8}  {sizes}")
    print(f"{'='*72}")
    print(f"Total unique valid branching features at m={m}: {len(seen_ks)}")


def print_sex_branch_details(tree: DecisionTree) -> None:
    if tree.schema.sex_feature_index is None:
        print("\n" + "=" * 60)
        print(f"{'SEX/GENDER-BRANCHED NODES':^60}")
        print("=" * 60)
        print("  No sex_feature_index configured; skipping.")
        return

    sex_k = tree.schema.sex_feature_index
    nodes_at_2 = tree.nodes_by_level.get(2, [])
    sex_nodes = [n for n in nodes_at_2 if n.branching_feat_k == sex_k]

    print("\n" + "=" * 60)
    print(f"{'SEX/GENDER-BRANCHED NODES':^60}")
    print("=" * 60)
    if not sex_nodes:
        print(f"  Configured sex feature (k={sex_k}) did not produce valid branches.")
        return

    for node in sex_nodes:
        lbl = node.branch_def.label if node.branch_def else "?"
        print(f"  Branch f={node.branch_f}  ({lbl})")
        print(f"    Users       : {node.n_users}")
        print(f"    Branch prob : {node.branch_prob:.4f}")
        if node.n_healthy is not None and node.n_unhealthy is not None:
            print(f"    Healthy     : {node.n_healthy} ({100*node.n_healthy/node.n_users:.1f}%)")
            print(f"    Unhealthy   : {node.n_unhealthy} ({100*node.n_unhealthy/node.n_users:.1f}%)")
        print(f"    Class dist  : {node.class_dist}")
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
    print("WHY FOCUS LEVEL 3 MAY BE IMPOSSIBLE (Eq. 2 proof)")
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
                f"  m={m} | k={k} ({tree.schema.get_feature_name(k)}) | "
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
    expected_n_features: Optional[int] = None,
    feature_names: Optional[Dict[int, str]] = None,
    sex_feature_index: Optional[int] = None,
    sex_value_labels: Optional[Dict[float, str]] = None,
) -> DecisionTree:
    loaded = load_dataset(
        wfdb_database=wfdb_database,
        ann_ext=ann_ext,
        max_records=max_records,
        expected_n_features=expected_n_features,
        feature_names=feature_names,
        healthy_class_ids=None,  # infer from WFDB 'N'
        sex_feature_index=sex_feature_index,
        sex_value_labels=sex_value_labels,
    )

    tree = build_decision_tree(
        data=loaded.data,
        labels=loaded.labels,
        schema=loaded.schema,
        label_metadata=loaded.label_metadata,
    )

    print_wfdb_label_report(loaded.label_metadata, loaded.schema)
    print_tree_summary(tree)
    print_level_statistics(tree)
    print_branching_features_at_level(tree, m=2)
    print_sex_branch_details(tree)
    explain_level3_impossibility(tree)
    validate_user_partition(tree)
    validate_eq2(tree)
    validate_feature_exclusion(tree)

    for node in tree.nodes_by_level.get(2, []):
        if loaded.schema.sex_feature_index is not None and node.branching_feat_k == loaded.schema.sex_feature_index:
            print_node_details(node, tree.feature_kinds, loaded.schema)

    return tree


if __name__ == "__main__":
    wfdb_database = sys.argv[1] if len(sys.argv) > 1 else "mitdb"
    tree = main(wfdb_database=wfdb_database)