from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import wfdb

RUN_GENDER_FEATURE_ANALYSIS = True
_GENDER_ANALYSIS_ALREADY_RAN = False

# ── Import Algorithm 1 components ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from CDS_Paper_Algorithms import (
    DecisionTree,
    TreeNode,
    FeatureKind,
    DatasetSchema,
    LabelMetadata,
    initialize_dataset_schema,
    build_decision_tree,
)

try:
    from fairness_config import ENABLE_REWEIGHING
except Exception:
    ENABLE_REWEIGHING = False


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
WFDB_HEALTHY_SYMBOL: str = "N"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – WFDB DATABASE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_simple_record_features(sig: np.ndarray) -> np.ndarray:
    """
    Minimal generic feature extraction from a WFDB signal matrix.

    Produces a fixed-size tabular vector by concatenating per-channel:
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
        Representative record label. Uses most frequent annotation symbol if
        present; otherwise 'NO_ANN'.
    all_symbols : List[str]
        All symbols observed for the record.
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
    Assumption requested by user:
      - WFDB symbol "N" (Normal beat) is healthy
      - every other class is unhealthy
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
) -> Tuple[np.ndarray, np.ndarray, DatasetSchema, LabelMetadata]:
    """
    Load a WFDB dataset by database identifier, not by local directory path.

    Parameters
    ----------
    wfdb_database : str
        WFDB / PhysioNet database identifier, e.g. "mitdb".
    ann_ext : str
        Annotation extension to read, default "atr".
    max_records : Optional[int]
        If set, limit the number of records loaded.
    expected_n_features : Optional[int]
        Optional dimensionality check. If None, inferred tabular width is accepted.
    feature_names : Optional[Dict[int, str]]
        Optional feature-name mapping. If None, generic feat_i names are used.
    healthy_class_ids : Optional[FrozenSet[int]]
        Optional healthy-class ids. If None, inferred via WFDB symbol "N".
    sex_feature_index : Optional[int]
        Optional sex/gender feature index.
    sex_value_labels : Optional[Dict[float, str]]
        Optional sex/gender display labels.

    Returns
    -------
    data, labels, schema, label_metadata
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
        f"classes={len(unique_syms)}, dropped_dim_mismatch={n_dropped_dim_mismatch}, "
        f"skipped_unreadable={len(skipped_records)}"
    )

    if skipped_records:
        log.debug("WFDB loader skipped records:")
        for name, reason in skipped_records[:20]:
            log.debug(f"  {name}: {reason}")

    healthy_desc, unhealthy_desc = _describe_health_mapping(sym_to_int, schema.healthy_class_ids)
    log.info(f"Healthy classes (WFDB 'N' only): {healthy_desc if healthy_desc else 'none detected'}")
    log.info(f"Unhealthy classes (all others): {unhealthy_desc if unhealthy_desc else 'none detected'}")

    return data, labels, schema, label_meta


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – FAIRNESS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _has_valid_binary_protected_feature(data: np.ndarray, idx: Optional[int]) -> bool:
    if idx is None or data is None or idx < 0 or idx >= data.shape[1]:
        return False
    col = data[:, idx]
    col = col[~np.isnan(col)]
    vals = np.unique(col)
    return len(vals) == 2 and set(vals.tolist()).issubset({0.0, 1.0})


def _get_primary_healthy_class_id(schema: DatasetSchema) -> Optional[int]:
    if schema.healthy_class_ids is None or len(schema.healthy_class_ids) == 0:
        return None
    return sorted(schema.healthy_class_ids)[0]


def compute_reweighing_weights(
    labels_valid: np.ndarray,
    sex_valid: np.ndarray,
    healthy_class_ids: Optional[FrozenSet[int]],
    protected_value: int = 1,
) -> np.ndarray:
    n = len(labels_valid)
    if n == 0:
        return np.ones(0, dtype=float)

    valid_mask = ~np.isnan(sex_valid)
    if valid_mask.sum() == 0:
        return np.ones(n, dtype=float)

    unique_groups = np.unique(sex_valid[valid_mask])
    if len(unique_groups) != 2 or set(unique_groups.tolist()) != {0.0, 1.0}:
        return np.ones(n, dtype=float)

    if healthy_class_ids is None:
        return np.ones(n, dtype=float)

    y_binary = np.isin(labels_valid, list(healthy_class_ids)).astype(int)

    p_s1 = (sex_valid == protected_value).sum() / n
    p_s0 = 1.0 - p_s1
    p_y1 = y_binary.sum() / n
    p_y0 = 1.0 - p_y1

    weights = np.ones(n, dtype=float)

    other_group_value = 1 - protected_value
    for s_val, p_s in [(protected_value, p_s1), (other_group_value, p_s0)]:
        for y_val, p_y in [(1, p_y1), (0, p_y0)]:
            mask = (sex_valid == s_val) & (y_binary == y_val)
            p_sy = mask.sum() / n
            if p_sy > 0:
                weights[mask] = (p_s * p_y) / p_sy

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiscretizationResult:
    feature_idx: int
    feature_name: str
    node_id: str
    b_raw_min: float
    b_raw_max: float
    delta_b: float
    bin_edges: np.ndarray
    n_bins: int
    bin_assignments: np.ndarray
    valid_mask: np.ndarray
    valid_user_rows: np.ndarray
    bin_counts_all: np.ndarray
    is_binary: bool
    is_degenerate: bool
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
    class_labels: List[int]
    p_bin_given_h: np.ndarray
    p_h_and_f: np.ndarray
    p_bin: np.ndarray
    p_h_given_bin: np.ndarray
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
    b_min_healthy: float
    b_max_healthy: float
    n_kf: float
    n_healthy_valid: int
    fallback_used: bool


@dataclass
class PerceptorModelEntry:
    node_id: str
    focus_level: int
    branching_feat_k: int
    branch_f: int
    feature_idx: int
    feature_name: str
    n_users_node: int
    disc: DiscretizationResult
    bayes: BayesianTables
    healthy_range: HealthyRangeResult

    def summary_str(self) -> str:
        hr = self.healthy_range
        d = self.disc
        return (
            f"  [{self.node_id}|o={self.feature_idx}({self.feature_name})]  "
            f"users={self.n_users_node}  valid={d.n_valid}  "
            f"bins={d.n_bins}  ΔB={d.delta_b:.3g}  "
            f"range=[{hr.b_min_healthy:.4g}, {hr.b_max_healthy:.4g}]  "
            f"N_kf={hr.n_kf:.2f}  "
            f"{'FALLBACK' if hr.fallback_used else ''}"
        )


@dataclass
class ExecutiveActionEntry:
    node_id: str
    focus_level: int
    branching_feat_k: int
    branch_f: int
    feature_idx: int
    feature_name: str
    disease_class: int
    action_weight: float
    p_below_normal: float
    p_above_normal: float
    p_h_and_f: float
    action_label: str

    def summary_str(self) -> str:
        return (
            f"  [{self.node_id}|o={self.feature_idx}|h={self.disease_class}]  "
            f"r={self.action_weight:.4f}  "
            f"p_below={self.p_below_normal:.4f}  "
            f"p_above={self.p_above_normal:.4f}  "
            f"prev={self.p_h_and_f:.4f}  "
            f"-> {self.action_label}"
        )


@dataclass
class Algorithm2Output:
    perceptor_library: List[PerceptorModelEntry] = field(default_factory=list)
    executive_library: List[ExecutiveActionEntry] = field(default_factory=list)

    perceptor_index: Dict[Tuple[str, int], PerceptorModelEntry] = field(default_factory=dict)
    executive_index: Dict[Tuple[str, int, int], ExecutiveActionEntry] = field(default_factory=dict)

    n_nodes_processed: int = 0
    n_perceptor_entries: int = 0
    n_executive_entries: int = 0

    def get_model(self, node_id: str, feature_idx: int) -> Optional[PerceptorModelEntry]:
        return self.perceptor_index.get((node_id, feature_idx))

    def get_action(self, node_id: str, feature_idx: int, disease_h: int) -> Optional[ExecutiveActionEntry]:
        return self.executive_index.get((node_id, feature_idx, disease_h))

    def actions_for_node(self, node_id: str) -> List[ExecutiveActionEntry]:
        return [e for e in self.executive_library if e.node_id == node_id]

    def top_actions(self, node_id: str, disease_h: int, top_k: int = 10) -> List[ExecutiveActionEntry]:
        acts = [e for e in self.executive_library if e.node_id == node_id and e.disease_class == disease_h]
        return sorted(acts, key=lambda e: e.action_weight, reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – DISCRETIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _n_bins_for_node(n_valid: int, is_binary: bool, n_bins_target: int) -> int:
    if is_binary:
        return 2
    if n_valid < 2:
        return 1
    return max(1, n_bins_target)


def compute_discretization(
    feature_idx: int,
    node: TreeNode,
    data: np.ndarray,
    feature_kinds: Dict[int, FeatureKind],
    schema: DatasetSchema,
    n_bins_target: int = DEFAULT_N_BINS,
) -> Optional[DiscretizationResult]:
    fname = schema.get_feature_name(feature_idx)
    n_node = len(node.user_indices)

    raw_values = data[node.user_indices, feature_idx]
    valid_mask = ~np.isnan(raw_values)
    valid_rows = np.where(valid_mask)[0]
    valid_vals = raw_values[valid_mask]
    n_valid = len(valid_vals)

    if n_valid == 0:
        log.debug(f"    feat {feature_idx}({fname}): ALL NaN in node {node.node_id!r} - skip")
        return None

    b_raw_min = float(valid_vals.min())
    b_raw_max = float(valid_vals.max())

    is_binary = (feature_kinds[feature_idx] == FeatureKind.BINARY)

    is_degenerate = (b_raw_min == b_raw_max)
    if is_degenerate:
        delta_b = 1.0
        bin_edges = np.array([b_raw_min - 0.5, b_raw_min + 0.5])
        n_bins = 1
        bin_asgn = np.zeros(n_valid, dtype=int)
        bin_counts = np.array([n_valid])
        log.debug(f"    feat {feature_idx}({fname}): DEGENERATE (all={b_raw_min}), 1 bin")
        return DiscretizationResult(
            feature_idx=feature_idx,
            feature_name=fname,
            node_id=node.node_id,
            b_raw_min=b_raw_min,
            b_raw_max=b_raw_max,
            delta_b=delta_b,
            bin_edges=bin_edges,
            n_bins=n_bins,
            bin_assignments=bin_asgn,
            valid_mask=valid_mask,
            valid_user_rows=valid_rows,
            bin_counts_all=bin_counts,
            is_binary=is_binary,
            is_degenerate=True,
            _raw_values_valid=valid_vals,
        )

    n_bins_requested = _n_bins_for_node(n_valid, is_binary, n_bins_target)

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
        delta_b = (b_raw_max - b_raw_min) / n_bins if n_bins > 0 else 1.0

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
        feature_idx=feature_idx,
        feature_name=fname,
        node_id=node.node_id,
        b_raw_min=b_raw_min,
        b_raw_max=b_raw_max,
        delta_b=delta_b,
        bin_edges=bin_edges,
        n_bins=n_bins,
        bin_assignments=bin_asgn,
        valid_mask=valid_mask,
        valid_user_rows=valid_rows,
        bin_counts_all=bin_counts,
        is_binary=is_binary,
        is_degenerate=False,
        _raw_values_valid=valid_vals,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – BAYESIAN PROBABILITY ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def _node_class_distribution(node: TreeNode) -> Dict[int, int]:
    if hasattr(node, "class_dist"):
        return getattr(node, "class_dist")
    if hasattr(node, "health_dist"):
        return getattr(node, "health_dist")
    return {}


def compute_bayesian_tables(
    disc: DiscretizationResult,
    node: TreeNode,
    labels: np.ndarray,
    class_labels: Optional[List[int]] = None,
    N_total: int = 0,
    laplace_eps: float = LAPLACE_EPSILON,
) -> BayesianTables:
    node_class_dist = _node_class_distribution(node)

    if class_labels is None:
        class_labels = sorted(node_class_dist.keys())

    n_classes = len(class_labels)
    n_bins = disc.n_bins

    global_indices_valid = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_indices_valid]

    counts_per_class_bin = np.zeros((n_classes, n_bins), dtype=float)
    users_per_class = np.zeros(n_classes, dtype=int)

    for ci, cls in enumerate(class_labels):
        class_mask = (labels_valid == cls)
        users_per_class[ci] = int(class_mask.sum())
        if users_per_class[ci] == 0:
            continue
        class_bins = disc.bin_assignments[class_mask]
        counts_per_class_bin[ci] = np.bincount(class_bins, minlength=n_bins)

    p_bin_given_h = np.zeros((n_bins, n_classes), dtype=float)
    for ci, cls in enumerate(class_labels):
        n_cls = users_per_class[ci]
        if n_cls == 0:
            p_bin_given_h[:, ci] = 1.0 / n_bins
        else:
            raw = counts_per_class_bin[ci] + laplace_eps
            p_bin_given_h[:, ci] = raw / raw.sum()

    p_h_and_f = np.zeros(n_classes, dtype=float)
    for ci, cls in enumerate(class_labels):
        p_h_and_f[ci] = node_class_dist.get(cls, 0) / N_total if N_total > 0 else 0.0

    p_bin = p_bin_given_h @ p_h_and_f
    p_bin_sum = p_bin.sum()
    if p_bin_sum > 0:
        p_bin /= p_bin_sum

    p_h_given_bin = np.zeros((n_bins, n_classes), dtype=float)
    for b in range(n_bins):
        denom = p_bin[b]
        if denom < 1e-300:
            p_h_given_bin[b] = 1.0 / n_classes if n_classes > 0 else 0.0
        else:
            p_h_given_bin[b] = (p_bin_given_h[b] * p_h_and_f) / denom

    return BayesianTables(
        class_labels=class_labels,
        p_bin_given_h=p_bin_given_h,
        p_h_and_f=p_h_and_f,
        p_bin=p_bin,
        p_h_given_bin=p_h_given_bin,
        n_users_per_class=dict(zip(class_labels, users_per_class.tolist())),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – HEALTHY RANGE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_healthy_range(
    disc: DiscretizationResult,
    node: TreeNode,
    labels: np.ndarray,
    healthy_class_ids: Optional[FrozenSet[int]],
) -> HealthyRangeResult:
    global_valid = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_valid]

    if healthy_class_ids is None or len(healthy_class_ids) == 0:
        fallback_used = True
        b_min, b_max = disc.b_raw_min, disc.b_raw_max
        n_kf = float(disc.n_bins)
        log.debug(
            f"    HealthyRange FALLBACK (healthy class undefined): "
            f"using full observed range [{b_min}, {b_max}]"
        )
        return HealthyRangeResult(
            b_min_healthy=b_min,
            b_max_healthy=b_max,
            n_kf=n_kf,
            n_healthy_valid=0,
            fallback_used=fallback_used,
        )

    healthy_mask = np.isin(labels_valid, list(healthy_class_ids))
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
        fallback_used = True
        b_min, b_max = disc.b_raw_min, disc.b_raw_max
        n_kf = float(disc.n_bins)
        log.debug(
            f"    HealthyRange FALLBACK (no healthy users): "
            f"using full observed range [{b_min}, {b_max}]"
        )

    log.debug(
        f"    HealthyRange: [{b_min:.4g}, {b_max:.4g}]  N_kf={n_kf:.2f}  "
        f"healthy_valid={n_healthy}  {'[FALLBACK]' if fallback_used else ''}"
    )

    return HealthyRangeResult(
        b_min_healthy=b_min,
        b_max_healthy=b_max,
        n_kf=n_kf,
        n_healthy_valid=n_healthy,
        fallback_used=fallback_used,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – EXECUTIVE ACTION LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

def compute_executive_actions(
    disc: DiscretizationResult,
    healthy_range: HealthyRangeResult,
    bayes: BayesianTables,
    node: TreeNode,
    labels: np.ndarray,
    schema: DatasetSchema,
    data: Optional[np.ndarray] = None,
) -> List[ExecutiveActionEntry]:
    actions = []
    b_min = healthy_range.b_min_healthy
    b_max = healthy_range.b_max_healthy

    global_valid = node.user_indices[disc.valid_user_rows]
    labels_valid = labels[global_valid]

    if b_min > b_max:
        return actions

    if schema.healthy_class_ids is None or len(schema.healthy_class_ids) == 0:
        return actions

    healthy_mask = np.isin(labels_valid, list(schema.healthy_class_ids))

    if healthy_mask.sum() == 0:
        return actions

    min_healthy_bin = np.searchsorted(disc.bin_edges[1:], b_min, side="right")
    max_healthy_bin = np.searchsorted(disc.bin_edges[1:], b_max, side="right")
    min_healthy_bin = int(np.clip(min_healthy_bin, 0, disc.n_bins - 1))
    max_healthy_bin = int(np.clip(max_healthy_bin, 0, disc.n_bins - 1))

    instance_weights = None
    if (
        ENABLE_REWEIGHING
        and data is not None
        and _has_valid_binary_protected_feature(data, schema.sex_feature_index)
    ):
        sex_valid = data[global_valid, schema.sex_feature_index]  # type: ignore[index]
        instance_weights = compute_reweighing_weights(
            labels_valid=labels_valid,
            sex_valid=sex_valid,
            healthy_class_ids=schema.healthy_class_ids,
        )

    for cls in bayes.class_labels:
        if schema.healthy_class_ids is not None and cls in schema.healthy_class_ids:
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
# SECTION 10 – ALGORITHM 2 CORE: PER-NODE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm2_for_node(
    node: TreeNode,
    data: np.ndarray,
    labels: np.ndarray,
    feature_kinds: Dict[int, FeatureKind],
    schema: DatasetSchema,
    class_labels: Optional[List[int]] = None,
    n_bins: int = DEFAULT_N_BINS,
) -> Tuple[List[PerceptorModelEntry], List[ExecutiveActionEntry]]:
    node_class_dist = _node_class_distribution(node)

    if class_labels is None:
        class_labels = sorted(node_class_dist.keys())

    perceptor_entries: List[PerceptorModelEntry] = []
    executive_entries: List[ExecutiveActionEntry] = []

    n_feat = len(node.feature_indices)
    log.debug(
        f"\n  Node {node.node_id!r}: m={node.focus_level}  "
        f"users={node.n_users}  features={n_feat}  "
        f"classes={list(class_labels)}"
    )

    for feature_o in node.feature_indices:
        disc = compute_discretization(
            feature_idx=feature_o,
            node=node,
            data=data,
            feature_kinds=feature_kinds,
            schema=schema,
            n_bins_target=n_bins,
        )
        if disc is None:
            continue

        bayes = compute_bayesian_tables(
            disc=disc,
            node=node,
            labels=labels,
            N_total=len(labels),
            class_labels=list(class_labels),
        )

        healthy_range = compute_healthy_range(
            disc=disc,
            node=node,
            labels=labels,
            healthy_class_ids=schema.healthy_class_ids,
        )

        entry = PerceptorModelEntry(
            node_id=node.node_id,
            focus_level=node.focus_level,
            branching_feat_k=node.branching_feat_k,
            branch_f=node.branch_f,
            feature_idx=feature_o,
            feature_name=disc.feature_name,
            n_users_node=node.n_users,
            disc=disc,
            bayes=bayes,
            healthy_range=healthy_range,
        )
        perceptor_entries.append(entry)

        act_entries = compute_executive_actions(
            disc=disc,
            healthy_range=healthy_range,
            bayes=bayes,
            node=node,
            labels=labels,
            schema=schema,
            data=data,
        )
        executive_entries.extend(act_entries)

    log.debug(
        f"  Node {node.node_id!r}: "
        f"perceptor_entries={len(perceptor_entries)}  "
        f"executive_entries={len(executive_entries)}"
    )
    return perceptor_entries, executive_entries


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – ALGORITHM 2 MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithm2(
    tree: DecisionTree,
    data: np.ndarray,
    labels: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    nodes_filter: Optional[List[str]] = None,
    class_labels: Optional[List[int]] = None,
) -> Algorithm2Output:
    log.info("=" * 72)
    log.info("Algorithm 2: CDS Perceptor and Executive Training")
    log.info(f"  n_bins={n_bins}  threshold={getattr(tree, 'threshold', 'N/A')}")
    log.info(f"  tree_depth={tree.depth()}  total_nodes={tree.count_nodes()}")
    log.info("=" * 72)

    feature_kinds = tree.feature_kinds
    schema = tree.schema

    if class_labels is None:
        class_labels = sorted(_node_class_distribution(tree.root).keys())
    log.info(f"  Record-level classes: {class_labels}")
    log.info(
        f"  Healthy classes: {sorted(schema.healthy_class_ids) if schema.healthy_class_ids else 'none'} "
        f"(healthy = WFDB symbol '{WFDB_HEALTHY_SYMBOL}')"
    )

    output = Algorithm2Output()
    nodes_processed = 0

    if ENABLE_REWEIGHING and _has_valid_binary_protected_feature(data, schema.sex_feature_index):
        run_gender_feature_analysis_once(
            data=data,
            labels=labels,
            gender_feature_idx=schema.sex_feature_index,  # type: ignore[arg-type]
            schema=schema,
        )
    elif ENABLE_REWEIGHING:
        log.warning(
            "ENABLE_REWEIGHING=True but no valid binary protected feature was found at "
            f"sex_feature_index={schema.sex_feature_index}. Skipping fairness/gender analysis."
        )

    for m in sorted(tree.nodes_by_level.keys()):
        nodes_at_m = tree.nodes_by_level[m]
        log.info(f"\n{'─' * 60}")
        log.info(f"Processing level m={m}: {len(nodes_at_m)} nodes …")

        for node in nodes_at_m:
            if nodes_filter and node.node_id not in nodes_filter:
                continue

            perc_entries, exec_entries = run_algorithm2_for_node(
                node=node,
                data=data,
                labels=labels,
                feature_kinds=feature_kinds,
                schema=schema,
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

        log.info(
            f"  Level m={m}: cumulative entries -> "
            f"perceptor={len(output.perceptor_library)}  "
            f"executive={len(output.executive_library)}"
        )

    output.n_nodes_processed = nodes_processed
    output.n_perceptor_entries = len(output.perceptor_library)
    output.n_executive_entries = len(output.executive_library)

    log.info(f"\n{'=' * 72}")
    log.info("Algorithm 2 complete.")
    log.info(f"  Nodes processed       : {nodes_processed}")
    log.info(f"  Perceptor entries     : {output.n_perceptor_entries}")
    log.info(f"  Executive entries     : {output.n_executive_entries}")
    log.info("=" * 72)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – VALIDATION FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────

def validate_probability_normalisation(output: Algorithm2Output, tol: float = 1e-6) -> Dict[str, int]:
    counts = {
        "V1_pass": 0, "V1_fail": 0,
        "V2_pass": 0, "V2_fail": 0,
        "V3_pass": 0, "V3_fail": 0,
    }

    for entry in output.perceptor_library:
        b = entry.bayes

        if not np.all(np.isfinite(b.p_bin_given_h)) or not np.all(np.isfinite(b.p_bin)) or not np.all(np.isfinite(b.p_h_given_bin)):
            counts["V1_fail"] += b.p_bin_given_h.shape[1]
            counts["V2_fail"] += 1
            counts["V3_fail"] += b.p_h_given_bin.shape[0]
            continue

        col_sums = b.p_bin_given_h.sum(axis=0)
        for s in col_sums:
            if abs(s - 1.0) < tol:
                counts["V1_pass"] += 1
            else:
                counts["V1_fail"] += 1

        ev_sum = b.p_bin.sum()
        if abs(ev_sum - 1.0) < tol:
            counts["V2_pass"] += 1
        else:
            counts["V2_fail"] += 1

        row_sums = b.p_h_given_bin.sum(axis=1)
        for s in row_sums:
            if abs(s - 1.0) < tol:
                counts["V3_pass"] += 1
            else:
                counts["V3_fail"] += 1

    return counts


def validate_fa_zero_invariant(
    output: Algorithm2Output,
    data: np.ndarray,
    labels: np.ndarray,
    tol: float = 1e-9,
) -> Dict[str, int]:
    counts = {"FA0_pass": 0, "FA0_fail": 0, "FA0_fallback": 0}

    for entry in output.perceptor_library:
        hr = entry.healthy_range
        disc = entry.disc

        if hr.fallback_used:
            counts["FA0_fallback"] += 1
            continue

        raw_vals = getattr(disc, "_raw_values_valid", None)
        if raw_vals is None:
            counts["FA0_fail"] += 1
            continue

        if hr.n_healthy_valid == 0:
            counts["FA0_fallback"] += 1
            continue

        if hr.b_min_healthy <= hr.b_max_healthy and len(raw_vals) >= hr.n_healthy_valid:
            counts["FA0_pass"] += 1
        else:
            counts["FA0_fail"] += 1

    return counts


def validate_bayesian_consistency(output: Algorithm2Output, tol: float = 1e-6) -> Dict[str, int]:
    counts = {"bayes_pass": 0, "bayes_fail": 0}
    for entry in output.perceptor_library:
        b = entry.bayes
        for b_idx in range(b.n_bins):
            ev = b.p_bin[b_idx]
            if ev < 1e-300:
                continue
            recomputed = (b.p_bin_given_h[b_idx] * b.p_h_and_f) / ev
            stored = b.p_h_given_bin[b_idx]
            max_diff = float(np.abs(recomputed - stored).max())
            if max_diff < tol:
                counts["bayes_pass"] += 1
            else:
                counts["bayes_fail"] += 1
    return counts


def validate_healthy_range_coverage(output: Algorithm2Output) -> Dict[str, int]:
    counts = {"range_ok": 0, "range_inverted": 0, "range_outside": 0}
    for entry in output.perceptor_library:
        hr = entry.healthy_range
        d = entry.disc
        if hr.b_min_healthy > hr.b_max_healthy:
            counts["range_inverted"] += 1
        elif hr.b_min_healthy < d.b_raw_min - 1e-9 or hr.b_max_healthy > d.b_raw_max + 1e-9:
            counts["range_outside"] += 1
        else:
            counts["range_ok"] += 1
    return counts


def validate_action_weights(
    output: Algorithm2Output,
    healthy_class_ids: Optional[FrozenSet[int]],
) -> Dict[str, int]:
    counts = {"weight_ok": 0, "weight_bad": 0, "healthy_in_exec": 0}
    for e in output.executive_library:
        if healthy_class_ids is not None and e.disease_class in healthy_class_ids:
            counts["healthy_in_exec"] += 1
        r = e.action_weight
        if 0 <= r <= 1.0 + 1e-9 and abs(r - (e.p_below_normal + e.p_above_normal)) < 1e-9:
            counts["weight_ok"] += 1
        else:
            counts["weight_bad"] += 1
    return counts


def run_all_validations(
    output: Algorithm2Output,
    data: np.ndarray,
    labels: np.ndarray,
    healthy_class_ids: Optional[FrozenSet[int]],
) -> None:
    print("\n" + "=" * 72)
    print("ALGORITHM 2: VALIDATION REPORT")
    print("=" * 72)

    checks = {
        "V1–V3: Probability Normalisation": validate_probability_normalisation(output),
        "Bayesian Eq.4 Consistency": validate_bayesian_consistency(output),
        "FA=0 Invariant": validate_fa_zero_invariant(output, data, labels),
        "Healthy Range Coverage": validate_healthy_range_coverage(output),
        "Action Weight Bounds": validate_action_weights(output, healthy_class_ids),
    }
    all_ok = True
    for name, result in checks.items():
        fails = sum(
            v for k, v in result.items()
            if "fail" in k or "bad" in k or "inverted" in k or "outside" in k or "healthy_in_exec" in k
        )
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
# SECTION 13 – INSPECTION AND REPORTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def print_algorithm2_summary(output: Algorithm2Output) -> None:
    print("\n" + "=" * 72)
    print("ALGORITHM 2: OUTPUT SUMMARY")
    print("=" * 72)
    print(f"  Nodes processed:     {output.n_nodes_processed}")
    print(f"  Perceptor entries:   {output.n_perceptor_entries}")
    print(f"  Executive entries:   {output.n_executive_entries}")

    if output.executive_library:
        weights = [e.action_weight for e in output.executive_library]
        weights_arr = np.array(weights)
        print("\n  Action weight distribution:")
        print(
            f"    min={weights_arr.min():.4f}  max={weights_arr.max():.4f}  "
            f"mean={weights_arr.mean():.4f}  median={np.median(weights_arr):.4f}"
        )
        bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
        hist, _ = np.histogram(weights_arr, bins=bins)
        for i in range(len(hist)):
            print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:5d} entries")
    print("=" * 72)


def print_perceptor_entry(entry: PerceptorModelEntry, show_full_tables: bool = False) -> None:
    disc = entry.disc
    hr = entry.healthy_range
    b = entry.bayes

    print(f"\n{'─' * 60}")
    print("PERCEPTOR MODEL ENTRY")
    print(f"  Node     : {entry.node_id!r}")
    print(f"  m / k / f: {entry.focus_level} / {entry.branching_feat_k} / {entry.branch_f}")
    print(f"  Feature  : {entry.feature_idx} ({entry.feature_name})")
    print(
        f"  Users    : {entry.n_users_node} total  "
        f"/ {disc.n_valid} valid  "
        f"/ {hr.n_healthy_valid} healthy-valid"
    )

    print("\n  Discretization:")
    print(
        f"    B_min={disc.b_raw_min:.4g}  B_max={disc.b_raw_max:.4g}  "
        f"ΔB={disc.delta_b:.4g}  n_bins={disc.n_bins}  "
        f"{'BINARY' if disc.is_binary else 'CONTINUOUS'}  "
        f"{'DEGENERATE' if disc.is_degenerate else ''}"
    )
    print(f"    Bin counts: {disc.bin_counts_all.tolist()}")

    print("\n  Healthy Range (FA=0):")
    print(
        f"    b_min={hr.b_min_healthy:.4g}  b_max={hr.b_max_healthy:.4g}  "
        f"N_kf={hr.n_kf:.2f}  {'[FALLBACK]' if hr.fallback_used else ''}"
    )

    print("\n  Prevalence P(h,f) per class:")
    for ci, cls in enumerate(b.class_labels):
        print(f"    h={cls:2d}: P(h,f)={b.p_h_and_f[ci]:.4f}  n_users={b.n_users_per_class.get(cls, 0)}")

    if show_full_tables:
        print("\n  P(B̂|h) table (rows=bins, cols=classes):")
        header = "  bin   " + "".join(f"  h={c:2d}" for c in b.class_labels)
        print(header)
        for b_idx in range(min(disc.n_bins, 20)):
            row = f"  {b_idx:3d}   " + "".join(
                f"  {b.p_bin_given_h[b_idx, ci]:6.4f}" for ci in range(b.n_classes)
            )
            print(row)

        print("\n  Evidence P(B̂):")
        print("  " + "  ".join(f"{v:.4f}" for v in b.p_bin))


def print_executive_top_actions(output: Algorithm2Output, node_id: str, disease_h: int, top_k: int = 15) -> None:
    acts = output.top_actions(node_id, disease_h, top_k)
    print(f"\n{'─' * 60}")
    print(f"TOP {top_k} ACTIONS  |  node={node_id!r}  |  disease h={disease_h}")
    if not acts:
        print("  (no actions found)")
        return
    print(f"  {'rank':4} {'feat':4} {'feature_name':25} {'r_{o|h}':8} {'p_below':8} {'p_above':8} {'P(h,f)':8}")
    print("  " + "─" * 73)
    for rank, act in enumerate(acts, 1):
        feat_name = act.feature_name[:25]
        print(
            f"  {rank:4d} {act.feature_idx:4d} {feat_name:25s} "
            f"{act.action_weight:8.4f} {act.p_below_normal:8.4f} "
            f"{act.p_above_normal:8.4f} {act.p_h_and_f:8.4f}"
        )


def print_normal_ranges_table(output: Algorithm2Output, node_id: str, max_rows: int = 30) -> None:
    entries = [e for e in output.perceptor_library if e.node_id == node_id]
    entries.sort(key=lambda e: e.feature_idx)
    print(f"\n{'─' * 72}")
    print(f"PERCEPTOR MODEL LIBRARY  |  node={node_id!r}  ({len(entries)} features)")
    print(f"  {'feat':4} {'name':25} {'b_min_h':10} {'b_max_h':10} {'N_kf':6} {'ΔB':8} {'n_bins':6} {'valid':6}")
    print("  " + "─" * 72)
    for e in entries[:max_rows]:
        hr = e.healthy_range
        d = e.disc
        print(
            f"  {e.feature_idx:4d} {e.feature_name[:25]:25s} "
            f"{hr.b_min_healthy:10.4g} {hr.b_max_healthy:10.4g} "
            f"{hr.n_kf:6.2f} {d.delta_b:8.4g} {d.n_bins:6d} {d.n_valid:6d}"
        )
    if len(entries) > max_rows:
        print(f"  … ({len(entries) - max_rows} more entries) …")


def print_executive_summary_by_disease(output: Algorithm2Output, node_id: str) -> None:
    acts = output.actions_for_node(node_id)
    by_cls = defaultdict(list)
    for a in acts:
        by_cls[a.disease_class].append(a)
    print(f"\n{'─' * 60}")
    print(f"EXECUTIVE ACTION LIBRARY SUMMARY  |  node={node_id!r}")
    print(f"  {'h':3} {'n_actions':10} {'best_feat':30} {'best_r':8}")
    print("  " + "─" * 56)
    for cls in sorted(by_cls.keys()):
        cls_acts = sorted(by_cls[cls], key=lambda a: a.action_weight, reverse=True)
        n_act = len(cls_acts)
        if n_act > 0:
            best = cls_acts[0]
            best_str = f"{best.feature_name}(o={best.feature_idx})"[:30]
            print(f"  {cls:3d} {n_act:10d} {best_str:30s} {best.action_weight:.4f}")
        else:
            print(f"  {cls:3d} {n_act:10d}")


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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 – MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main(
    wfdb_database: str,
    ann_ext: str = "atr",
    run_full: bool = False,
    max_records: Optional[int] = None,
    expected_n_features: Optional[int] = None,
    feature_names: Optional[Dict[int, str]] = None,
    sex_feature_index: Optional[int] = None,
    sex_value_labels: Optional[Dict[float, str]] = None,
) -> Algorithm2Output:
    """
    End-to-end execution using a WFDB database identifier.
    """
    import logging as _logging
    _logging.getLogger("CDS.Alg1").setLevel(_logging.WARNING)

    log.info("Step 1: Load WFDB dataset")
    data, labels, schema, label_metadata = load_wfdb_dataset(
        wfdb_database=wfdb_database,
        ann_ext=ann_ext,
        max_records=max_records,
        expected_n_features=expected_n_features,
        feature_names=feature_names,
        healthy_class_ids=None,  # infer healthy from WFDB symbol 'N'
        sex_feature_index=sex_feature_index,
        sex_value_labels=sex_value_labels,
    )
    log.info(f"  Loaded: {data.shape[0]} users × {data.shape[1]} features")

    log.info("Step 2: Build Decision Tree (Algorithm 1)")
    tree = build_decision_tree(
        data=data,
        labels=labels,
        schema=schema,
        label_metadata=label_metadata,
    )
    log.info(f"  Tree: depth={tree.depth()}  nodes={tree.count_nodes()}")

    if run_full:
        nodes_filter = None
        log.info("Step 3: Processing ALL nodes in the tree")
    else:
        key_nodes = [tree.root.node_id]
        level2 = tree.nodes_by_level.get(2, [])
        for n in level2[:2]:
            key_nodes.append(n.node_id)
        nodes_filter = [n for n in key_nodes if n in tree.all_nodes]
        log.info(f"Step 3: Processing {len(nodes_filter)} key nodes: {nodes_filter}")

    log.info("Step 4: Algorithm 2 – Perceptor and Executive Training")
    output = run_algorithm2(
        tree=tree,
        data=data,
        labels=labels,
        n_bins=DEFAULT_N_BINS,
        nodes_filter=nodes_filter,
    )

    log.info("Step 5: Validation")
    run_all_validations(output, data, labels, schema.healthy_class_ids)

    log.info("Step 6: Reporting")
    print_wfdb_label_report(label_metadata, schema)
    print_algorithm2_summary(output)
    print_normal_ranges_table(output, tree.root.node_id, max_rows=20)

    for node_id in nodes_filter or []:
        if node_id == tree.root.node_id:
            continue
        if node_id in [e.node_id for e in output.perceptor_library]:
            print_normal_ranges_table(output, node_id, max_rows=15)
            print_executive_summary_by_disease(output, node_id)

    healthy_class_ids = schema.healthy_class_ids or frozenset()
    disease_candidates = sorted(set(labels.tolist()))
    disease_candidates = [h for h in disease_candidates if h not in healthy_class_ids][:4]
    for h in disease_candidates:
        print_executive_top_actions(output, tree.root.node_id, h, top_k=10)

    hr_entry = output.get_model(tree.root.node_id, 0)
    if hr_entry:
        print(f"\n{'=' * 60}")
        print("DETAILED ENTRY: Feature 0 at ROOT")
        print_perceptor_entry(hr_entry, show_full_tables=True)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 – OPTIONAL ONE-SHOT GENDER FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_gender_feature_analysis_once(
    data: np.ndarray,
    labels: np.ndarray,
    gender_feature_idx: int,
    schema: DatasetSchema,
    significance_threshold: float = 0.05,
) -> None:
    global RUN_GENDER_FEATURE_ANALYSIS
    global _GENDER_ANALYSIS_ALREADY_RAN

    if not RUN_GENDER_FEATURE_ANALYSIS:
        return

    if _GENDER_ANALYSIS_ALREADY_RAN:
        return

    if not _has_valid_binary_protected_feature(data, gender_feature_idx):
        print("[WARN] No valid binary gender/protected feature found. Skipping gender analysis.")
        return

    healthy_class_id = _get_primary_healthy_class_id(schema)
    if healthy_class_id is None:
        print("[WARN] No healthy class configured. Skipping gender analysis.")
        return

    _GENDER_ANALYSIS_ALREADY_RAN = True
    RUN_GENDER_FEATURE_ANALYSIS = False

    print("\n" + "=" * 80)
    print("GLOBAL GENDER FEATURE ANALYSIS")
    print("=" * 80)

    try:
        from sklearn.inspection import permutation_importance
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("[ERROR] sklearn not installed.")
        return

    if gender_feature_idx >= data.shape[1]:
        print("[WARN] Gender feature index outside feature matrix. Skipping gender analysis.")
        return

    gender_values = data[:, gender_feature_idx]
    valid_gender_mask = ~np.isnan(gender_values)

    female_mask = (gender_values == 1) & valid_gender_mask
    male_mask = (gender_values == 0) & valid_gender_mask

    if female_mask.sum() == 0 or male_mask.sum() == 0:
        print("[WARN] Missing binary group populations.")
        return

    feature_indices = [i for i in range(data.shape[1]) if i != gender_feature_idx]

    X = data[:, feature_indices].copy()
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(col_medians, inds[1])

    y = (~np.isin(labels, [healthy_class_id])).astype(int)

    X_male = X[male_mask]
    y_male = y[male_mask]

    X_female = X[female_mask]
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
        fname = schema.get_feature_name(feat_idx)
        print(
            f"{fname:<40} "
            f"Male={male_importance[rank_idx]:.6f}  "
            f"Female={female_importance[rank_idx]:.6f}  "
            f"Diff={diff[rank_idx]:.6f}"
        )

    print("\n" + "-" * 80)
    print("HEALTHY RANGES BY GENDER")
    print("-" * 80)

    healthy_mask = np.isin(labels, [healthy_class_id])

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

        fname = schema.get_feature_name(feat_idx)

        print(
            f"{fname:<40} "
            f"M:[{male_min:.3f}, {male_max:.3f}]  "
            f"F:[{female_min:.3f}, {female_max:.3f}]"
        )

    print("\n[INFO] Gender feature analysis completed.")
    print("=" * 80)


if __name__ == "__main__":
    wfdb_database = sys.argv[1] if len(sys.argv) > 1 else "mitdb"
    run_full = "--full" in sys.argv

    output = main(
        wfdb_database=wfdb_database,
        run_full=run_full,
    )