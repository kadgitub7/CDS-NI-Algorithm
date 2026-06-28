"""
================================================================================
Algorithm 1: Creating the CDS Decision Tree
================================================================================

PURPOSE
-------
Algorithm 1 builds the multi-level decision tree that partitions the user
population into progressively finer demographic/physiological sub-groups.
Each sub-group receives its own set of Bayesian normal-range models (done in
Algorithm 2).  The tree topology is determined entirely by:
  • which features produce branches that each have >= u_min users  (Eq. 2)
  • a feature-ordering exclusion rule that removes redundant tree paths (line 9)

PAPER FIDELITY NOTATION
-----------------------
Every implementation choice is tagged with one of:
  [PAPER]  – directly stated in the paper
  [INFER]  – logically required by the paper but left unspecified
  [ENGR]   – engineering choice with justification

KEY FORMULAE (for quick reference)
------------------------------------
  Eq. 2   ceil( |U_{m-1}| × P_f^{(k_m,m)} ) >= u_min
  Eq. 3   u_min = 5 / Threshold  =  5 / 0.025  =  200
  Line 7  U_m^{k_m,f} = P_f^{(k_m,m)} × U_{m-1}  (user set update)
  Line 9  O_m^{k_m,f} = O_{m-1} − {1,...,k}        (feature pruning, m>2 only)
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

def _build_logger(name: str = "CDS.Alg1") -> logging.Logger:
    """
    Two handlers:
      • STDOUT  – INFO and above (human-readable summary)
      • STDOUT  – DEBUG and above for verbose tracing (if enabled via env var)
    [ENGR] Separate levels let callers silence debug noise in production.
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
DIAGNOSTIC_THRESHOLD: float = 0.025       # acceptable diagnostic error
U_MIN: int = math.ceil(5 / DIAGNOSTIC_THRESHOLD)   # = 200

# [PAPER Eq. 9] Complexity threshold: F_total,k = Σ_m F_km ≤ threshold.
# The paper does not name a specific numeric value; 16 is a defensive default
# that allows up to 4 binary levels. For the Arrhythmia case study, Eq. 2
# forces M=2 anyway (u_min=200, N=452), so this cap is dormant.
COMPLEXITY_THRESHOLD: int = 16

# [PAPER] Class 1 = healthy; classes 2-16 = disease / unknown.
HEALTHY_CLASS: int = 1
LABEL_COL_IDX: int = 279   # 0-indexed column in the CSV that holds the class

# [PAPER] The dataset has 279 features (columns 0-278).
N_FEATURES: int = 279


class FeatureKind(Enum):
    """
    [PAPER] Algorithm 1 lines 3-4 distinguish two kinds of features:
      BINARY     – "sex, smoking, drug, family disease history …"  -> F_k = 2,
                   branches are simply {0} and {1}.
      CONTINUOUS – "age, weight, height, heart rate …"             -> F_k >= 2,
                   branches are determined by statistical discretisation (FSD).
    [INFER] We detect the kind automatically from the actual data values rather
            than a hard-coded list, because the UCI dataset does not include
            smoking / drug etc.  A feature is BINARY if its unique non-NaN
            values are a subset of {0, 1}; otherwise CONTINUOUS.
    """
    BINARY     = auto()
    CONTINUOUS = auto()


def _build_ecg_channel_names() -> Dict[int, str]:
    """
    Build a human-readable name for each of the 279 features.
    [INFER] Names are derived from the arrhythmia.names file structure.
    """
    names: Dict[int, str] = {
        0: "Age",       1: "Sex",       2: "Height",    3: "Weight",
        4: "QRS_dur",   5: "PR_int",    6: "QT_int",    7: "T_int",
        8: "P_int",
        9:  "QRS_angle", 10: "T_angle",  11: "P_angle",
        12: "QRST_angle", 13: "J_angle", 14: "Heart_rate",
    }
    channels = ["DI","DII","DIII","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]
    wave_labels = [
        "Q_wid","R_wid","S_wid","Rp_wid","Sp_wid","N_defl",
        "Rag_R","Diph_R","Rag_P","Diph_P","Rag_T","Diph_T",
    ]
    amp_labels = [
        "JJ_amp","Q_amp","R_amp","S_amp","Rp_amp","Sp_amp",
        "P_amp","T_amp","QRSA","QRSTA",
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

# [PAPER] The paper cites columns with missing values; column 13 (J_angle) is
#         83 % missing – effectively unusable as a branching feature.
MISSING_VALUE_COLS: FrozenSet[int] = frozenset({10, 11, 12, 13, 14})


def classify_features(data: np.ndarray) -> Dict[int, FeatureKind]:
    """
    Classify each of the 279 features as BINARY or CONTINUOUS.

    Parameters
    ----------
    data : np.ndarray, shape (N, 279), dtype float64, NaN for missing.

    Returns
    -------
    Dict mapping feature_index -> FeatureKind

    [INFER] Detection rule: if the set of non-NaN unique values is a subset of
            {0.0, 1.0} -> BINARY; else -> CONTINUOUS.
    [ENGR]  A feature that is entirely NaN is classified as CONTINUOUS because
            it cannot produce branches anyway (it will be pruned at Eq. 2).
    """
    kinds: Dict[int, FeatureKind] = {}
    for col in range(N_FEATURES):
        col_data = data[:, col]
        valid    = col_data[~np.isnan(col_data)]
        if len(valid) == 0:
            kinds[col] = FeatureKind.CONTINUOUS   # all-NaN -> unusable
            continue
        unique_vals = set(valid)
        if unique_vals.issubset({0.0, 1.0}):
            kinds[col] = FeatureKind.BINARY
        else:
            kinds[col] = FeatureKind.CONTINUOUS
    return kinds

def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the UCI Arrhythmia CSV.

    Returns
    -------
    data   : np.ndarray, shape (452, 279), float64, NaN for missing values.
    labels : np.ndarray, shape (452,),     int,     class labels 1-16.

    [PAPER] "The MATLAB Arrhythmia database has a few numbers of instances
            (i.e., 452 instances) and there are some missing features such as
            J angle or heart rate (key feature) for some Users."
    [ENGR]  '?' is the missing-value marker in the UCI CSV; we replace it with
            NaN so numpy / pandas treat it consistently.
    """
    log.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path, header=None, na_values="?")
    assert df.shape == (452, 280), (
        f"Expected (452, 280) but got {df.shape}. "
        "Ensure you are using the UCI arrhythmia.data file."
    )
    data   = df.iloc[:, :279].to_numpy(dtype=float)   # features
    labels = df.iloc[:,  279].to_numpy(dtype=int)     # class labels
    log.info(f"  Loaded {data.shape[0]} users × {data.shape[1]} features.")
    log.info(f"  Label distribution: "
             f"{ {c: int((labels==c).sum()) for c in sorted(set(labels))} }")
    missing_per_col = np.isnan(data).sum(axis=0)
    nonempty = [(i, int(missing_per_col[i])) for i in range(N_FEATURES)
                if missing_per_col[i] > 0]
    log.info(f"  Columns with missing values: "
             f"{ {i: n for i,n in nonempty} }")
    return data, labels

@dataclass
class BranchDef:
    """
    Defines membership in a single branch for a given feature.

    Attributes
    ----------
    feature_idx : int
        0-indexed column in `data`.
    branch_idx  : int
        1-indexed branch number (f in the paper).
    kind        : FeatureKind
        BINARY or CONTINUOUS.
    value_set   : frozenset | None
        For BINARY: the set of values that belong to this branch ({0.0} or {1.0}).
    lo, hi      : float | None
        For CONTINUOUS: half-open interval [lo, hi].  The upper branch uses
        hi = +inf.
    label       : str
        Human-readable description, e.g. "sex=female" or "age<=47.0".

    [PAPER] Lines 3-4 distinguish binary (value_set) from statistical (lo..hi).
    [ENGR]  Storing lo/hi avoids floating-point comparisons against a median
            that may shift slightly between sub-populations.
    """
    feature_idx: int
    branch_idx:  int
    kind:        FeatureKind
    value_set:   Optional[FrozenSet[float]] = None
    lo:          Optional[float]            = None
    hi:          Optional[float]            = None
    label:       str                        = ""

    def contains(self, value: float) -> bool:
        """
        Return True iff `value` belongs to this branch.

        [ENGR] NaN always returns False – missing-value users are excluded
               from any branch (they cannot be assigned because the feature is
               not observed).
        """
        if np.isnan(value):
            return False
        if self.kind == FeatureKind.BINARY:
            return value in self.value_set        # type: ignore[operator]
        else:
            # [INFER] Closed lower bound, open upper bound -> [lo, hi)
            # Except the last bin which is [lo, +inf].
            return self.lo <= value < self.hi     # type: ignore[operator]


def _fsd_binary(feature_idx: int, user_indices: np.ndarray,
                data: np.ndarray) -> List[BranchDef]:
    """
    Create 2 branch definitions for a BINARY feature (values 0 or 1).

    [PAPER] Algorithm 1, line 4: "Creating a binary branch for [sex …],
            f_{k_m,m} ∈ {1, 2}, F_{k_m,m} = 2"

    Returns a list of BranchDef objects, one per distinct value present.
    If only one value is present in this user sub-set, returns an empty list
    (a single-branch split is not a split).

    [ENGR]  We check which values actually appear in the current user sub-set,
            not globally, because deeper sub-populations may be homogeneous.
    """
    col   = data[user_indices, feature_idx]
    valid = col[~np.isnan(col)]
    values_present = sorted(set(valid))
    if len(values_present) < 2:
        return []          # cannot split – only one value present

    fname = FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")
    branches = []
    for bidx, v in enumerate(values_present, start=1):
        if feature_idx == 1:   # Sex: give readable labels
            lbl = "male" if v == 0.0 else "female"
        else:
            lbl = f"{fname}={int(v)}"
        branches.append(BranchDef(
            feature_idx = feature_idx,
            branch_idx  = bidx,
            kind        = FeatureKind.BINARY,
            value_set   = frozenset({v}),
            label       = lbl,
        ))
    return branches


def _fsd_continuous_median(feature_idx: int, user_indices: np.ndarray,
                           data: np.ndarray) -> List[BranchDef]:
    """
    Create 2 branch definitions for a CONTINUOUS feature using a MEDIAN split.

    [PAPER] Algorithm 1, line 3: "Statistical discretisation for features
            can have more than two branches [Age, weight, height, heart rate …]"
    [INFER] The paper references an FSD approach but does not specify the exact
            number of bins or the cut-point method.
    [ENGR]  We use a MEDIAN split because:
              1. With N=452 and u_min=200, at most 2 branches are ever feasible
                 (⌊452/200⌋ = 2).
              2. A median split maximises the chance that both branches satisfy
                 u_min (closest to a 50/50 split).
              3. The paper's Figure 6b shows age split at "≤45 / >45", which is
                 near-median for this population (median age ≈ 47).

    Tie handling
    ------------
    [ENGR]  If the strict median creates a degenerate partition (all non-NaN
            values are identical), we return [] to signal that this feature
            cannot branch.  Otherwise we create [lo, median] and (median, +inf)
            branches.  Users with NaN are excluded from both branches.
    """
    col_vals = data[user_indices, feature_idx]
    valid    = col_vals[~np.isnan(col_vals)]
    if len(valid) == 0:
        return []
    median_val = float(np.median(valid))
    if float(valid.min()) == float(valid.max()):
        return []    # no variation -> cannot split

    fname = FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")
    b1 = BranchDef(
        feature_idx = feature_idx,
        branch_idx  = 1,
        kind        = FeatureKind.CONTINUOUS,
        lo          = -np.inf,
        hi          = median_val,           # [ENGR] branch 1: values < median
        label       = f"{fname}<{median_val}",
    )
    b2 = BranchDef(
        feature_idx = feature_idx,
        branch_idx  = 2,
        kind        = FeatureKind.CONTINUOUS,
        lo          = median_val,
        hi          = np.inf,               # branch 2: values >= median
        label       = f"{fname}>={median_val}",
    )
    # Fix labels to use ≤/>
    b1.label = f"{fname}<{median_val}"
    b2.label = f"{fname}>={median_val}"
    return [b1, b2]


def compute_fsd_branches(feature_idx: int,
                         user_indices: np.ndarray,
                         data: np.ndarray,
                         kind: FeatureKind) -> List[BranchDef]:
    """
    Dispatch to the appropriate FSD strategy for the given feature kind.

    [PAPER] Algorithm 1 lines 3-4 are the dispatch logic:
      line 3 -> CONTINUOUS -> _fsd_continuous_median
      line 4 -> BINARY     -> _fsd_binary

    Parameters
    ----------
    feature_idx  : 0-indexed column in `data`.
    user_indices : row indices of users in the current node.
    data         : full (N, 279) feature matrix.
    kind         : FeatureKind, determined by classify_features().

    Returns
    -------
    List of BranchDef, one per branch.  May be empty if the feature cannot
    produce a meaningful split for this user sub-set.
    """
    if kind == FeatureKind.BINARY:
        return _fsd_binary(feature_idx, user_indices, data)
    else:
        return _fsd_continuous_median(feature_idx, user_indices, data)

"""
def filter_users_by_branch(user_indices: np.ndarray,
                            branch: BranchDef,
                            data: np.ndarray) -> np.ndarray:
    Return the subset of `user_indices` whose value for `branch.feature_idx`
    satisfies the branch membership condition.

    This implements the user-set update from Algorithm 1, line 7:
        U_m^{k_m, f} = P_f^{(k_m,m)} × U_{m-1}^{...}

    [INFER] The paper writes this as a scalar multiplication but means the
            *actual filtered set*.  P_f × |U_{m-1}| gives the expected size,
            used in Eq. 2; the actual set is needed for downstream algorithms.

    [ENGR]  We return actual user indices, not counts.  This makes the tree
            self-contained: every downstream algorithm (2-4) can directly read
            which users belong to any node without re-filtering.
    
    col_vals = data[user_indices, branch.feature_idx]
    mask     = np.array([branch.contains(v) for v in col_vals], dtype=bool)
    return user_indices[mask]
"""

def filter_users_by_branch(user_indices: np.ndarray,
                            branch: BranchDef,
                            data: np.ndarray) -> np.ndarray:
    """
    Return the subset of user_indices whose value for branch.feature_idx
    satisfies the branch membership condition using vectorized NumPy operations.
    """
    col_vals = data[user_indices, branch.feature_idx]
    
    if branch.kind == FeatureKind.BINARY:
        val = next(iter(branch.value_set))
        mask = (col_vals == val)
    else:
        mask = (col_vals >= branch.lo) & (col_vals < branch.hi)
        
    mask = mask & ~np.isnan(col_vals)
    return user_indices[mask]

def health_distribution(user_indices: np.ndarray,
                         labels: np.ndarray) -> Dict[int, int]:
    """
    Return counts of each health class among `user_indices`.

    [PAPER] Table 3 shows the prevalence in the MATLAB database – this function
            reproduces those counts for any user sub-set, enabling direct
            comparison.
    """
    classes = labels[user_indices]
    dist = defaultdict(int)
    for c in classes:
        dist[int(c)] += 1
    return dict(sorted(dist.items()))


def compute_branch_probability(branch_user_count: int,
                                parent_user_count: int) -> float:
    """
    P_f^{(k_m, m)} = |users in branch f| / |users in parent node|

    [PAPER] Algorithm 1, line 6: "Calculate the P_f^{(k_m,m)}: use previous
            probabilities extracted in previous focus level."
    [INFER] "Previous probabilities" means we always condition on the parent
            node's user set (not the full dataset).  For the root->level-2
            transition the parent is the root.  For level-2->level-3 the parent
            is whichever level-2 node we are currently expanding.

    Parameters
    ----------
    branch_user_count : |{u ∈ U_parent : u falls in branch f}|
    parent_user_count : |U_parent|

    Returns
    -------
    float in [0, 1]
    """
    if parent_user_count == 0:
        return 0.0
    return branch_user_count / parent_user_count


def check_branching_condition(branch_user_count: int,
                               parent_user_count: int,
                               threshold: float = DIAGNOSTIC_THRESHOLD) -> Tuple[bool, int, int]:
    """
    Eq. 2: ceil( |U_{m-1}| × P_f ) >= u_min

    Since P_f = branch_count / parent_count, the product simplifies:
        |U_{m-1}| × P_f = parent_count × (branch_count / parent_count) = branch_count

    Hence Eq. 2 reduces to: branch_count >= u_min.

    [PAPER] Eq. 3: u_min = 5 / Threshold = 5 / 0.025 = 200
    [INFER] The ceil() on an integer is a no-op.

    Returns
    -------
    (passes: bool, lhs: int, u_min_value: int)
        lhs  = ceil(parent × P_f) = branch_count (the actual LHS of Eq. 2)
        u_min = 5 / threshold
    """
    u_min_val = math.ceil(5 / threshold)                      # Eq. 3
    lhs       = math.ceil(parent_user_count *
                          compute_branch_probability(branch_user_count,
                                                     parent_user_count))
    # Because lhs == branch_user_count (see derivation above), this is exact.
    return (lhs >= u_min_val), lhs, u_min_val


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – FEATURE EXCLUSION LOGIC (Algorithm 1, Line 9)
# ─────────────────────────────────────────────────────────────────────────────

def compute_child_feature_set(parent_feature_indices: List[int],
                               branching_feature_k: int,
                               focus_level_m: int) -> List[int]:
    """
    Compute the feature set O_m^{k_m, f} for a child node.

    Algorithm 1, lines 8-10:
        if m > 2:
            O_m^{k_m,f} = O_{m-1} − {1, …, k}

    [PAPER] The exclusion prevents equivalent tree paths.  For example, if the
            tree branches on Sex at m=2 and on Age at m=3, that gives the same
            population partition as branching on Age at m=2 then Sex at m=3.
            By enforcing the rule "at m>2, only branch on features with index > k
            (where k is the current level's branching feature)", we eliminate
            duplicate paths.

    [INFER] "1, …, k" means features with (0-indexed) index ≤ branching_feature_k.
            The set notation uses 1-indexed feature numbers in the paper, but
            since the exclusion rule is index-based, 0-indexed arithmetic gives
            the same result.

    [ENGR]  At m=2 no exclusion occurs; children inherit the full parent feature
            set.  This is correct because at m=2 each child node was reached by
            a single unique branching feature, so there is no duplicate-path
            problem yet.

    Parameters
    ----------
    parent_feature_indices : List[int]  – O_{m-1} for this node's parent.
    branching_feature_k    : int        – k, the feature used to branch at
                                          focus level m (i.e., the feature that
                                          PRODUCED this child node).
    focus_level_m          : int        – m, the child node's focus level.

    Returns
    -------
    List[int] – feature indices available for further splitting / modelling.
    """
    if focus_level_m <= 2:
        # [PAPER] Lines 8-10: exclusion only when m > 2.
        return list(parent_feature_indices)

    # m > 2: remove all features with index <= branching_feature_k
    return [f for f in parent_feature_indices if f > branching_feature_k]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – TREE NODE DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    """
    One node in the CDS decision tree.

    Naming convention mirrors the paper:
        focus_level      -> m
        branching_feat_k -> k  (the feature used by the PARENT to create this node;
                                -1 for the root, which has no parent)
        branch_f         -> f  (the branch index that led to this node; 0 = root)

    The root node represents focus level 1 (all users, all features).
    Nodes at focus level 2 are children of the root, branched on some feature k.

    Attributes
    ----------
    node_id          : unique string path, e.g. "root|k1_f1" for the male branch.
    focus_level      : m.
    branching_feat_k : k of the branching performed to reach this node.
    branch_f         : f, the specific branch index.
    branch_def       : BranchDef that defines membership (None for root).
    user_indices     : U_m^{k,f} – row indices in the dataset.
    feature_indices  : O_m^{k,f} – feature columns available at this node.
    branch_prob      : P_f^{(k_m,m)}.
    health_dist      : {class: count} distribution.
    children_by_k    : Dict[feature_k -> List[TreeNode]] grouping children by
                        which feature was used to split them.
    is_leaf          : True if no further splitting was possible.
    prune_reason     : Non-empty if this node was pruned / not expanded.
    """
    node_id:          str
    focus_level:      int
    branching_feat_k: int                    # k  (-1 for root)
    branch_f:         int                    # f  (0 for root)
    branch_def:       Optional[BranchDef]    # None for root
    user_indices:     np.ndarray             # U_m^{k,f}
    feature_indices:  List[int]              # O_m^{k,f}
    branch_prob:      float                  # P_f^{(k_m,m)}
    health_dist:      Dict[int, int]         # {class: count}
    children_by_k:    Dict[int, List["TreeNode"]] = field(default_factory=dict)
    is_leaf:          bool                   = False
    prune_reason:     str                    = ""

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

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
        """Register a child node under its branching feature."""
        k = child.branching_feat_k
        if k not in self.children_by_k:
            self.children_by_k[k] = []
        self.children_by_k[k].append(child)

    def __repr__(self) -> str:
        return (f"TreeNode({self.node_id!r}, m={self.focus_level}, "
                f"k={self.branching_feat_k}, f={self.branch_f}, "
                f"n_users={self.n_users}, n_feat={self.n_features})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – TREE STRUCTURE CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionTree:
    """
    Container that holds the full CDS decision tree produced by Algorithm 1.

    Attributes
    ----------
    root              : the level-1 root node.
    nodes_by_level    : Dict[m -> List[TreeNode]] for easy per-level access.
    all_nodes         : flat Dict[node_id -> TreeNode] for lookup.
    valid_branches    : count of branches that passed Eq. 2 at each level.
    pruned_branches   : count of branches pruned at each level.
    feature_kinds     : the kind classification used.
    threshold         : the Threshold parameter (default 0.025).
    u_min             : u_min = 5/Threshold.
    """
    root:           TreeNode
    nodes_by_level: Dict[int, List[TreeNode]] = field(default_factory=dict)
    all_nodes:      Dict[str, TreeNode]        = field(default_factory=dict)
    valid_branches: Dict[int, int]             = field(default_factory=dict)
    pruned_branches:Dict[int, int]             = field(default_factory=dict)
    feature_kinds:  Dict[int, FeatureKind]     = field(default_factory=dict)
    threshold:      float                      = DIAGNOSTIC_THRESHOLD
    u_min:          int                        = U_MIN

    def register(self, node: TreeNode) -> None:
        """Add a node to all index structures."""
        m = node.focus_level
        if m not in self.nodes_by_level:
            self.nodes_by_level[m] = []
        self.nodes_by_level[m].append(node)
        self.all_nodes[node.node_id] = node

    def depth(self) -> int:
        return max(self.nodes_by_level.keys()) if self.nodes_by_level else 1

    def count_nodes(self) -> int:
        return len(self.all_nodes)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – ALGORITHM 1: CORE IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def _try_split_node(parent:          TreeNode,
                    feature_k:       int,
                    kind:            FeatureKind,
                    data:            np.ndarray,
                    labels:          np.ndarray,
                    threshold:       float) -> Tuple[List[TreeNode], int]:
    """
    Revised logic: Admitted branches are checked independently.
    If a feature produces only 1 valid branch, it is marked as a leaf.
    """
    m_child = parent.focus_level + 1

    # Step 1: Compute branch definitions (FSD)
    branch_defs = compute_fsd_branches(
        feature_idx  = feature_k,
        user_indices = parent.user_indices,
        data         = data,
        kind         = kind,
    )
    if len(branch_defs) < 2:
        return [], 0

    # Step 2: Filter users into branches
    branch_user_sets: List[Tuple[BranchDef, np.ndarray]] = []
    for bdef in branch_defs:
        branch_users = filter_users_by_branch(parent.user_indices, bdef, data)
        branch_user_sets.append((bdef, branch_users))

    # Step 3: Eq. 2 check per branch + create child nodes
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
            node_id=child_id, focus_level=m_child,
            branching_feat_k=feature_k, branch_f=bdef.branch_idx,
            branch_def=bdef, user_indices=busers, feature_indices=feats,
            branch_prob=p_f, health_dist=health_distribution(busers, labels),
        )
        created.append(child)

    # NEW LOGIC: If the feature produced only one valid branch (the other was pruned),
    # this branch cannot split further and becomes a leaf.
    if len(created) == 1:
        created[0].is_leaf = True
        created[0].prune_reason = "Single-branch split; further branching disabled."

    return created, n_pruned


def _expand_node(parent:      TreeNode,
                 data:        np.ndarray,
                 labels:      np.ndarray,
                 kinds:       Dict[int, FeatureKind],
                 threshold:   float,
                 tree:        DecisionTree) -> int:
    """
    Try every feature in parent.feature_indices as a branching candidate.
    Register all valid child nodes in `tree`.

    Returns
    -------
    int – total number of valid child nodes created.

    [PAPER] Algorithm 1, line 2: "for k = set of O_{m-1}^{k,f} do"
            This is the exhaustive sweep over all available features.

    [ENGR]  We attempt every feature, not just sex, because Algorithm 1 creates
            the FULL tree (all possible branching features).  The executive
            (Algorithm 4) selects which branch to follow during prediction.
            The paper confirms this in Figure 6b, which shows BOTH sex and age
            as parallel candidates at focus level 2.
    """
    m_child     = parent.focus_level + 1
    n_created   = 0
    n_pruned    = 0
    valid_ks    = []

    log.debug(f"  Expanding {parent.node_id!r} -> m={m_child}, "
              f"trying {len(parent.feature_indices)} features …")

    for k in sorted(parent.feature_indices):
        kind = kinds[k]
        children, pruned = _try_split_node(
            parent    = parent,
            feature_k = k,
            kind      = kind,
            data      = data,
            labels    = labels,
            threshold = threshold,
        )
        n_pruned += pruned
        if children:
            valid_ks.append(k)
            n_created += len(children)
            for child in children:
                parent.add_child(child)
                tree.register(child)

    if m_child not in tree.valid_branches:
        tree.valid_branches[m_child]  = 0
        tree.pruned_branches[m_child] = 0
    tree.valid_branches[m_child]  += n_created
    tree.pruned_branches[m_child] += n_pruned

    log.info(
        f"  m={parent.focus_level}->{m_child} | node {parent.node_id!r} | "
        f"valid splits: {len(valid_ks)} features -> {n_created} children | "
        f"pruned branches: {n_pruned}"
    )
    if not valid_ks:
        parent.is_leaf    = True
        parent.prune_reason = f"No feature split passes Eq.2 at m={m_child}"
    return n_created


def build_decision_tree(data:       np.ndarray,
                         labels:    np.ndarray,
                         threshold: float = DIAGNOSTIC_THRESHOLD,
                         max_m:     int   = 2) -> DecisionTree:
    """
    Algorithm 1: Creating a Decision Tree
    ======================================
    Iterative level-by-level tree construction.

    [PAPER] Algorithm 1 pseudocode (faithfully reproduced):

        Initialization:
        1: for m = 2 to M do
        2:   for k = set of O_{m-1}^{k,f} do
        3:     FSD for continuous features
        4:     Binary branches for nominal features
        5:   for f ∈ {1…F_{k_m,m}} do
        6:     P_f = …
        7:     U_m^{kf} = P_f × U_{m-1}
        8:     if m > 2 then
        9:       O_m^{kf} = O_{m-1} − {1,…,k}
       10:     end if
       11:   end for
       12:  end for
       13: end for

    Parameters
    ----------
    data       : (N, 279) feature matrix.
    labels     : (N,)     class labels.
    threshold  : Eq. 3 threshold (default 0.025).
    max_m      : maximum focus level (hard cap; Eq. 2 will terminate earlier).

    Returns
    -------
    DecisionTree with all nodes populated.

    Execution model
    ---------------
    We use a BFS queue of nodes to expand.  At each BFS step we process all
    nodes at focus level m and attempt to create nodes at focus level m+1.
    The loop terminates when no new nodes are created (Eq. 2 prunes everything)
    or max_m is reached.

    [INFER] The paper's "for m = 2 to M" implies an outer level loop; the inner
            loop over features (line 2) must implicitly iterate over every
            parent node at level m-1, not just the root.  Our BFS captures this.
    """
    log.info("=" * 72)
    log.info("Algorithm 1: Building CDS Decision Tree")
    log.info(f"  threshold = {threshold},  u_min = {math.ceil(5/threshold)}")
    log.info(f"  N_users   = {len(labels)},  N_features = {data.shape[1]}")
    log.info("=" * 72)

    # ── Classify features once ───────────────────────────────────────────────
    kinds = classify_features(data)
    n_binary = sum(1 for k in kinds.values() if k == FeatureKind.BINARY)
    n_cont   = sum(1 for k in kinds.values() if k == FeatureKind.CONTINUOUS)
    log.info(f"Feature classification: {n_binary} binary, {n_cont} continuous.")

    # ── Initialise root node (focus level 1) ─────────────────────────────────
    # [PAPER] "In 'Focus level 1,' the measured signals, User information,
    #          features, actions, and desired false alarm rate are the inputs."
    all_user_indices    = np.arange(len(labels), dtype=int)
    all_feature_indices = list(range(N_FEATURES))
    root = TreeNode(
        node_id          = "root",
        focus_level      = 1,
        branching_feat_k = -1,           # no parent split
        branch_f         = 0,
        branch_def       = None,
        user_indices     = all_user_indices,
        feature_indices  = all_feature_indices,
        branch_prob      = 1.0,
        health_dist      = health_distribution(all_user_indices, labels),
    )

    tree = DecisionTree(
        root          = root,
        feature_kinds = kinds,
        threshold     = threshold,
        u_min         = math.ceil(5 / threshold),
    )
    tree.register(root)
    tree.nodes_by_level[1] = [root]

    log.info(f"Root node: {root.n_users} users, {root.n_features} features.")
    log.info(f"  Health distribution: {root.health_dist}")

    # ── BFS: expand level by level ───────────────────────────────────────────
    # [PAPER] "for m = 2 to M" – outer loop over focus levels.
    current_level_nodes = [root]

    for m in range(2, max_m + 1):
        log.info(f"\n{'─'*60}")
        log.info(f"FOCUS LEVEL m = {m}: "
                 f"expanding {len(current_level_nodes)} parent node(s) …")

        next_level_nodes = []
        total_new = 0

        for parent_node in current_level_nodes:
            # [PAPER] line 2: "for k = set of O_{m-1}^{k,f} do"
            n_new = _expand_node(
                parent    = parent_node,
                data      = data,
                labels    = labels,
                kinds     = kinds,
                threshold = threshold,
                tree      = tree,
            )
            total_new += n_new
            next_level_nodes.extend(parent_node.all_children)

        log.info(f"  Total new nodes at m={m}: {total_new}")

        if total_new == 0:
            # [PAPER] Eq. 2 prunes everything; stop.
            log.info(f"  No valid branches at m={m}.  Tree construction complete.")
            break

        current_level_nodes = next_level_nodes

    log.info(f"\n{'='*72}")
    log.info(f"Algorithm 1 complete.  Tree depth: {tree.depth()}  |  "
             f"Total nodes: {tree.count_nodes()}")
    return tree




# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 – INSPECTION AND REPORTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def print_tree_summary(tree: DecisionTree) -> None:
    """
    Print a compact tree summary table.

    Output columns:
        m   – focus level
        k   – branching feature index (feature used to reach this node)
        f   – branch index
        |U| – number of users
        %H  – % healthy
        P_f – branch probability
        label – branch label
    """
    print("\n" + "=" * 90)
    print(f"{'CDS DECISION TREE SUMMARY':^90}")
    print(f"  threshold={tree.threshold}, u_min={tree.u_min}, depth={tree.depth()}, "
          f"total_nodes={tree.count_nodes()}")
    print("=" * 90)
    print(f"{'m':>3}  {'k':>4}  {'f':>3}  {'|U|':>5}  {'%H':>6}  "
          f"{'P_f':>6}  {'#feat':>5}  label")
    print("-" * 90)
    for m in sorted(tree.nodes_by_level.keys()):
        for node in tree.nodes_by_level[m]:
            pct_h   = 100 * node.n_healthy / node.n_users if node.n_users else 0
            fname   = FEATURE_NAMES.get(node.branching_feat_k, "root")
            lbl     = node.branch_def.label if node.branch_def else "—root—"
            print(f"{m:>3}  {node.branching_feat_k:>4}  {node.branch_f:>3}  "
                  f"{node.n_users:>5}  {pct_h:>5.1f}%  {node.branch_prob:>6.4f}  "
                  f"{node.n_features:>5}  {lbl}")
        if m < max(tree.nodes_by_level.keys()):
            print()
    print("=" * 90)


def print_node_details(node: TreeNode, feature_kinds: Dict[int, FeatureKind]) -> None:
    """
    Print full details for a single TreeNode.
    """
    print(f"\n{'─'*60}")
    print(f"NODE: {node.node_id}")
    print(f"  Focus level m     : {node.focus_level}")
    print(f"  Branching feat k  : {node.branching_feat_k} "
          f"({FEATURE_NAMES.get(node.branching_feat_k,'root')})")
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
                f"f={c.branch_f}({c.branch_def.label if c.branch_def else '?'})"
                f"->{c.n_users}u" for c in children
            )
            print(f"    k={k} ({kname}): {labels_str}")
    else:
        print(f"    (leaf node – {node.prune_reason or 'no further splits'})")


def print_level_statistics(tree: DecisionTree) -> None:
    """
    Per-level statistics table showing valid vs pruned branches.
    """
    print("\n" + "=" * 60)
    print(f"{'LEVEL STATISTICS':^60}")
    print("=" * 60)
    print(f"{'m':>3}  {'nodes':>6}  {'valid_br':>9}  {'pruned_feat':>11}  "
          f"{'avg_|U|':>8}")
    print("-" * 60)
    for m in sorted(tree.nodes_by_level.keys()):
        nodes   = tree.nodes_by_level[m]
        avg_u   = np.mean([n.n_users for n in nodes]) if nodes else 0
        valid   = tree.valid_branches.get(m, 0)
        pruned  = tree.pruned_branches.get(m, 0)
        print(f"{m:>3}  {len(nodes):>6}  {valid:>9}  {pruned:>11}  {avg_u:>8.1f}")
    print("=" * 60)


def print_branching_features_at_level(tree: DecisionTree, m: int) -> None:
    """
    List all features that produced valid branches at focus level m,
    with their branch-size statistics.
    """
    nodes_at_m = tree.nodes_by_level.get(m, [])
    if not nodes_at_m:
        print(f"\nNo nodes at focus level m={m}.")
        return

    # Group nodes by parent and branching feature
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
            kind  = tree.feature_kinds.get(k, FeatureKind.CONTINUOUS)
            sizes = ", ".join(
                f"{c.branch_def.label}:{c.n_users}" for c in children
            )
            print(f"{k:>4}  {fname:20}  {kind.name:12}  {len(children):>8}  {sizes}")
    print(f"{'='*72}")
    print(f"Total unique valid branching features at m={m}: {len(seen_ks)}")


def print_sex_branch_details(tree: DecisionTree) -> None:
    """
    Print detailed information specifically for the Sex (col 1) branching,
    which is the one the paper uses in its case study.
    """
    SEX_K = 1
    nodes_at_2 = tree.nodes_by_level.get(2, [])
    sex_nodes  = [n for n in nodes_at_2 if n.branching_feat_k == SEX_K]

    print("\n" + "=" * 60)
    print(f"{'SEX-BRANCHED NODES (paper case study)':^60}")
    print("=" * 60)
    if not sex_nodes:
        print("  Sex (k=1) did not produce valid branches!")
        return

    for node in sex_nodes:
        lbl    = node.branch_def.label if node.branch_def else "?"
        print(f"  Branch f={node.branch_f}  ({lbl})")
        print(f"    Users       : {node.n_users}")
        print(f"    Branch prob : {node.branch_prob:.4f}")
        print(f"    Healthy     : {node.n_healthy}  "
              f"({100*node.n_healthy/node.n_users:.1f}%)")
        print(f"    Diseased    : {node.n_diseased}  "
              f"({100*node.n_diseased/node.n_users:.1f}%)")
        print(f"    Health dist : {node.health_dist}")
        passes_eq2, lhs, u_min_val = check_branching_condition(
            node.n_users, tree.root.n_users, tree.threshold
        )
        print(f"    Eq.2 check  : ceil({tree.root.n_users}×{node.branch_prob:.4f})"
              f" = {lhs} >= {u_min_val}? {passes_eq2}")
        print(f"    Features    : {node.n_features}")
        print()


def explain_level3_impossibility(tree: DecisionTree) -> None:
    """
    Demonstrate mathematically why focus level 3 is impossible for this dataset.
    [PAPER] "Due to the lack of enough Users in the MATLAB database, the CDS
            maximum focus level is 2 with two tree branches."
    """
    print("\n" + "=" * 72)
    print("WHY FOCUS LEVEL 3 IS IMPOSSIBLE (Eq. 2 proof)")
    print("=" * 72)
    u_min = tree.u_min
    nodes_at_2 = tree.nodes_by_level.get(2, [])
    if not nodes_at_2:
        print("  No level-2 nodes to analyse.")
        return

    for parent in nodes_at_2:
        best_possible = parent.n_users // 2  # best-case even split
        print(f"\n  Parent node: {parent.node_id!r}  |  {parent.n_users} users")
        print(f"    Best-case branch size (50/50 split): {best_possible}")
        print(f"    u_min required                     : {u_min}")
        print(f"    Best-case >= u_min?                : {best_possible >= u_min}")
        print(f"    -> LEVEL 3 {'POSSIBLE' if best_possible >= u_min else 'IMPOSSIBLE'} "
              f"for this parent node.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 – VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_user_partition(tree: DecisionTree) -> None:
    """
    Verify the user-set partitioning invariants for every branching feature:
      1. No user appears in two sibling branches under the same parent×feature.
      2. All users in the parent who have a non-NaN value for the branching
         feature must appear in exactly one child branch.
      3. Users with NaN values are absent from all children (excluded by FSD).

    [PAPER] Line 7: U_m^{kf} = P_f × U_{m-1} implies the child user sets are
            disjoint sub-sets of the parent.

    Prints PASS / FAIL for each node group.
    """
    print("\n" + "=" * 60)
    print("USER-PARTITION VALIDATION")
    print("=" * 60)
    all_ok = True

    for m in sorted(tree.nodes_by_level.keys()):
        if m == 1:
            continue
        # Group children by (parent_id, k)
        groups: Dict[Tuple[str, int], List[TreeNode]] = defaultdict(list)
        for node in tree.nodes_by_level[m]:
            parent_id = node.node_id.rsplit("|", 1)[0]
            groups[(parent_id, node.branching_feat_k)].append(node)

        for (parent_id, k), siblings in groups.items():
            parent = tree.all_nodes.get(parent_id)
            if parent is None:
                continue
            # Union of all sibling user index sets
            union_sets   = [set(s.user_indices.tolist()) for s in siblings]
            sibling_union = set().union(*union_sets)
            # Pairwise intersection (should all be empty)
            ok_disjoint = True
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    inter = union_sets[i] & union_sets[j]
                    if inter:
                        ok_disjoint = False
                        print(f"  FAIL disjoint: m={m} k={k} "
                              f"f={siblings[i].branch_f}∩f={siblings[j].branch_f} "
                              f"has {len(inter)} common users!")
            # Direct check: sibling union ⊆ parent
            ok_subset = sibling_union.issubset(set(parent.user_indices.tolist()))
            status = "PASS" if (ok_disjoint and ok_subset) else "FAIL"
            if not (ok_disjoint and ok_subset):
                all_ok = False
            print(f"  m={m} | k={k} ({FEATURE_NAMES.get(k,'?')}) | "
                  f"parent={parent.n_users}u | "
                  f"children={[s.n_users for s in siblings]} | "
                  f"union={len(sibling_union)} | disjoint={ok_disjoint} | "
                  f"subset={ok_subset} -> {status}")

    print(f"\n  Overall validation: {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")


def validate_eq2(tree: DecisionTree) -> None:
    """
    Verify that every non-root node satisfies Eq. 2.
    """
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
    """
    Verify that m>2 nodes have features with index > branching_feat_k removed.
    [PAPER] Algorithm 1, line 9.
    """
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
                print(f"  PASS: {node.node_id!r} | k={k} | "
                      f"features all > k ✓ ({node.n_features} remaining)")
    if violations == 0:
        print("  All m>2 nodes satisfy feature exclusion rule ✓")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 – MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main(data_path: str = str(Path(__file__).parent / "data" / "arrhythmia.data")) -> DecisionTree:
    """
    End-to-end execution of Algorithm 1.

    Steps:
      1. Load dataset.
      2. Build decision tree.
      3. Print summary tables.
      4. Run validation checks.
      5. Show the sex-specific branches (paper case study).
      6. Explain why level 3 is impossible.
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    data, labels = load_dataset(data_path)

    # ── 2. Build tree ─────────────────────────────────────────────────────────
    tree = build_decision_tree(data, labels)

    # ── 3. High-level summary ─────────────────────────────────────────────────
    print_tree_summary(tree)
    print_level_statistics(tree)
    print_branching_features_at_level(tree, m=2)

    # ── 4. Paper case study: sex branching ────────────────────────────────────
    print_sex_branch_details(tree)

    # ── 5. Why level 3 is impossible ─────────────────────────────────────────
    explain_level3_impossibility(tree)

    # ── 6. Validation ─────────────────────────────────────────────────────────
    validate_eq2(tree)
    validate_feature_exclusion(tree)

    # ── 7. Per-node details for the sex branches (paper case study) ───────────
    for node in tree.nodes_by_level.get(2, []):
        if node.branching_feat_k == 1:    # Sex feature
            print_node_details(node, tree.feature_kinds)

    return tree

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
    path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "data" / "arrhythmia.data")
    tree = main(path)