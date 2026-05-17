"""
================================================================================
Algorithm1_forcedBranch.py
================================================================================

FORCED SEX BRANCHING -- correct interpretation
-----------------------------------------------
Sex is not treated as a feature inside the decision tree.  Instead, a user's
sex determines WHICH tree they enter BEFORE any features are examined.

  Step 0 (routing):  Read user sex -> male tree OR female tree
  Step 1 (Level 1):  Root of the selected tree contains only users of that sex
  Step 2+ (Level 2+): Standard CDS branching on all remaining features
                       (Sex is excluded from each tree because it is constant)

This produces two fully independent CDS decision trees:
  - male_tree  : built on the male subset of the dataset
  - female_tree: built on the female subset of the dataset

Within each tree the standard Algorithm 1 rules apply:
  - Every feature (except Sex) that satisfies Eq.2 produces valid branches.
  - Line-9 exclusion removes features with index <= k at m > 2.
  - Users already assigned to the correct tree at Level 1 based on sex.

USAGE
-----
    from Algorithm1_forcedBranch import (
        build_forced_sex_forest,
        load_dataset_flexible,
        route_user,
        get_nodes_for_user,
    )

    data, labels = load_dataset_flexible("arrhythmia_augmented.data")
    forest = build_forced_sex_forest(data, labels, max_m=2)

    # Route a user to their sex-specific tree
    sex_label, tree = route_user(forest, data[user_idx])

    # Find which nodes contain this user at each level of their tree
    nodes_by_level = get_nodes_for_user(tree, user_idx)

PAPER FIDELITY
--------------
  [PAPER]  -- directly from the paper
  [FORCED] -- deliberate departure: sex is a routing mechanism, not a feature
  [ENGR]   -- engineering choice
================================================================================
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import shared infrastructure from the original Algorithm 1 file.
# ---------------------------------------------------------------------------
from CDS_Paper_Algorithms import (
    # Constants
    DIAGNOSTIC_THRESHOLD,
    HEALTHY_CLASS,
    N_FEATURES,
    FEATURE_NAMES,
    # Enums and data classes
    FeatureKind,
    BranchDef,
    TreeNode,
    DecisionTree,
    # Pure helper functions (unchanged)
    classify_features,
    compute_fsd_branches,
    filter_users_by_branch,
    health_distribution,
    compute_branch_probability,
    check_branching_condition,
    # Reporting utilities (unchanged)
    print_tree_summary,
    print_level_statistics,
    print_node_details,
    # Logger builder
    _build_logger,
)

log = _build_logger("CDS.Alg1.ForcedSex")

# ---------------------------------------------------------------------------
# Sex encoding constants
# ---------------------------------------------------------------------------
SEX_FEATURE_IDX: int  = 1      # column 1 in the 279-feature space
MALE_VALUE:      float = 0.0   # arrhythmia.data encoding
FEMALE_VALUE:    float = 1.0


# ============================================================================
# FLEXIBLE DATASET LOADER
# Accepts any N x 280 file (original 452 rows OR augmented 952 rows).
# ============================================================================

def load_dataset_flexible(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load any N x 280 arrhythmia-format CSV.

    Identical to CDS_Paper_Algorithms.load_dataset except the 452-row
    assertion is removed so the 952-row augmented dataset also works.

    Returns
    -------
    data   : (N, 279) float64, NaN for '?'
    labels : (N,)     int class labels 1-16
    """
    log.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path, header=None, na_values="?")
    n_rows, n_cols = df.shape
    if n_cols != 280:
        raise ValueError(
            f"Expected 280 columns (279 features + 1 label) but got {n_cols}."
        )
    data   = df.iloc[:, :279].to_numpy(dtype=float)
    labels = df.iloc[:,  279].to_numpy(dtype=int)
    log.info(f"  Loaded {n_rows} users x 279 features.")
    log.info(f"  Label distribution: "
             f"{ {c: int((labels == c).sum()) for c in sorted(set(labels))} }")
    missing_per_col = np.isnan(data).sum(axis=0)
    nonempty = {i: int(missing_per_col[i]) for i in range(N_FEATURES)
                if missing_per_col[i] > 0}
    if nonempty:
        log.info(f"  Columns with missing values: {nonempty}")
    sex_col  = data[:, SEX_FEATURE_IDX]
    n_male   = int((sex_col == MALE_VALUE).sum())
    n_female = int((sex_col == FEMALE_VALUE).sum())
    log.info(f"  Sex breakdown: {n_male} male  {n_female} female  "
             f"({n_female / n_rows * 100:.1f}% female)")
    return data, labels


# ============================================================================
# FORCED SEX FOREST -- container for both sex-specific trees
# ============================================================================

@dataclass
class ForcedSexForest:
    """
    Two independent CDS decision trees, one per sex.

    Attributes
    ----------
    male_tree      : DecisionTree built on male users only.
    female_tree    : DecisionTree built on female users only.
    male_indices   : Indices (into the full dataset) of male users.
    female_indices : Indices (into the full dataset) of female users.
    n_users        : Total users in the full dataset.
    threshold      : Eq.2 threshold used to build both trees.
    """
    male_tree:      DecisionTree
    female_tree:    DecisionTree
    male_indices:   np.ndarray
    female_indices: np.ndarray
    n_users:        int
    threshold:      float


# ============================================================================
# STANDARD FEATURE SET COMPUTATION (original line-9 rule, no Sex forcing)
# Used inside each sex-specific tree where Sex is excluded entirely.
# ============================================================================

def compute_child_feature_set_standard(
    parent_feature_indices: List[int],
    branching_feature_k:    int,
    focus_level_m:          int,
) -> List[int]:
    """
    [PAPER] Algorithm 1, line 9: at m > 2, remove features with index <= k.

    This is the original CDS rule with no modifications.  Sex is not re-added
    because it was excluded from the feature set at the root of each sex-
    specific tree and should never appear within those trees.

    Parameters
    ----------
    parent_feature_indices : O_{m-1} for the parent node.
    branching_feature_k    : k, the feature that produced this child.
    focus_level_m          : m, the child's focus level.
    """
    if focus_level_m <= 2:
        # Line-9 exclusion only applies at m > 2
        return list(parent_feature_indices)
    # m > 2: remove features with index <= k
    return [f for f in parent_feature_indices if f > branching_feature_k]


# ============================================================================
# NODE SPLITTING (standard, no Sex injection)
# ============================================================================

def _try_split_node_standard(
    parent:    TreeNode,
    feature_k: int,
    kind:      FeatureKind,
    data:      np.ndarray,
    labels:    np.ndarray,
    threshold: float,
) -> Tuple[List[TreeNode], int]:
    """
    Attempt to split `parent` on `feature_k` using standard CDS rules.

    Identical to the original _try_split_node from CDS_Paper_Algorithms.
    Uses compute_child_feature_set_standard (no Sex re-injection).

    Returns
    -------
    (created_children, n_pruned)
    """
    m_child = parent.focus_level + 1

    # Compute branch definitions via FSD
    branch_defs = compute_fsd_branches(
        feature_idx  = feature_k,
        user_indices = parent.user_indices,
        data         = data,
        kind         = kind,
    )
    if len(branch_defs) < 2:
        return [], 0

    # Assign users to each branch
    branch_user_sets: List[Tuple[BranchDef, np.ndarray]] = []
    for bdef in branch_defs:
        branch_users = filter_users_by_branch(parent.user_indices, bdef, data)
        branch_user_sets.append((bdef, branch_users))

    # Eq.2 check per branch
    created:  List[TreeNode] = []
    n_pruned: int = 0

    for bdef, busers in branch_user_sets:
        passes, lhs, u_min_val = check_branching_condition(
            branch_user_count = len(busers),
            parent_user_count = parent.n_users,
            threshold         = threshold,
        )
        if not passes:
            n_pruned += 1
            log.debug(
                f"    PRUNED k={feature_k} ({FEATURE_NAMES.get(feature_k,'?')}) "
                f"branch '{bdef.label}': |U|={len(busers)} < u_min={u_min_val}"
            )
            continue

        p_f   = compute_branch_probability(len(busers), parent.n_users)
        feats = compute_child_feature_set_standard(
            parent_feature_indices = parent.feature_indices,
            branching_feature_k    = feature_k,
            focus_level_m          = m_child,
        )

        child_id = f"{parent.node_id}|k{feature_k}_f{bdef.branch_idx}"
        child = TreeNode(
            node_id          = child_id,
            focus_level      = m_child,
            branching_feat_k = feature_k,
            branch_f         = bdef.branch_idx,
            branch_def       = bdef,
            user_indices     = busers,
            feature_indices  = feats,
            branch_prob      = p_f,
            health_dist      = health_distribution(busers, labels),
        )
        created.append(child)
        log.debug(
            f"    VALID   k={feature_k} ({FEATURE_NAMES.get(feature_k,'?')}) "
            f"branch '{bdef.label}' -> {len(busers)} users"
        )

    # If only one branch survived, that child is a leaf (sibling pruned)
    if len(created) == 1:
        created[0].is_leaf = True
        created[0].prune_reason = "Only one branch passed Eq.2 (sibling pruned)."

    return created, n_pruned


# ============================================================================
# NODE EXPANDER (standard, tries all features in parent's feature set)
# ============================================================================

def _expand_node_standard(
    parent:    TreeNode,
    data:      np.ndarray,
    labels:    np.ndarray,
    kinds:     Dict[int, FeatureKind],
    threshold: float,
    tree:      DecisionTree,
) -> int:
    """
    Expand `parent` by trying every feature in parent.feature_indices.

    Sex is not in any node's feature_indices inside a sex-specific tree
    (it was excluded at the root), so it is never tried here.

    Returns
    -------
    int -- number of valid child nodes created.
    """
    m_child   = parent.focus_level + 1
    n_created = 0
    n_pruned  = 0
    valid_ks: List[int] = []

    log.debug(
        f"  Expanding {parent.node_id!r} -> m={m_child} | "
        f"{len(parent.feature_indices)} candidates ..."
    )

    for k in sorted(parent.feature_indices):
        kind = kinds.get(k, FeatureKind.CONTINUOUS)
        children, pruned = _try_split_node_standard(
            parent    = parent,
            feature_k = k,
            kind      = kind,
            data      = data,
            labels    = labels,
            threshold = threshold,
        )
        n_pruned  += pruned
        if children:
            valid_ks.append(k)
            n_created += len(children)
            for child in children:
                parent.add_child(child)
                tree.register(child)

    tree.valid_branches[m_child]  = tree.valid_branches.get(m_child,  0) + n_created
    tree.pruned_branches[m_child] = tree.pruned_branches.get(m_child, 0) + n_pruned

    log.info(
        f"  {parent.node_id!r} m={parent.focus_level}->{m_child} | "
        f"valid_features={len(valid_ks)} -> {n_created} children | "
        f"pruned_branches={n_pruned}"
    )

    if not valid_ks:
        parent.is_leaf      = True
        parent.prune_reason = f"No feature passed Eq.2 at m={m_child}"

    return n_created


# ============================================================================
# SEX-SPECIFIC TREE BUILDER
# ============================================================================

def build_sex_specific_tree(
    data:             np.ndarray,
    labels:           np.ndarray,
    sex_user_indices: np.ndarray,
    sex_label:        str,
    threshold:        float = DIAGNOSTIC_THRESHOLD,
    max_m:            int   = 2,
) -> DecisionTree:
    """
    Build one CDS decision tree for users of a single sex.

    The root of this tree contains only the users in `sex_user_indices`.
    Sex (col 1) is excluded from the feature set because all users here
    share the same sex value -- it would never produce a valid split.

    [PAPER] Algorithm 1 rules apply within this tree:
      - Every feature (except Sex) that passes Eq.2 creates branches.
      - Line-9 exclusion removes features with index <= k at m > 2.

    Parameters
    ----------
    data             : Full (N, 279) feature matrix.
    labels           : Full (N,) class labels.
    sex_user_indices : Indices into `data`/`labels` for this sex group.
    sex_label        : 'male' or 'female' (used in node IDs and logging).
    threshold        : Eq.2 diagnostic threshold.
    max_m            : Maximum focus level to attempt.

    Returns
    -------
    DecisionTree whose root_node contains only users of the given sex.
    """
    u_min_val = math.ceil(5 / threshold)

    # Exclude Sex from feature candidates -- constant within this group
    feature_indices = [i for i in range(N_FEATURES) if i != SEX_FEATURE_IDX]

    # Classify features using the full dataset for accurate statistics
    kinds = classify_features(data)
    kinds.pop(SEX_FEATURE_IDX, None)   # remove Sex so it is never tried

    root_id = f"root_{sex_label}"
    root = TreeNode(
        node_id          = root_id,
        focus_level      = 1,
        branching_feat_k = -1,
        branch_f         = 0,
        branch_def       = None,
        user_indices     = sex_user_indices,
        feature_indices  = feature_indices,
        branch_prob      = 1.0,
        health_dist      = health_distribution(sex_user_indices, labels),
    )

    tree = DecisionTree(
        root          = root,
        feature_kinds = kinds,
        threshold     = threshold,
        u_min         = u_min_val,
    )
    tree.register(root)
    tree.nodes_by_level[1] = [root]

    log.info(f"  [{sex_label}] Level 1 (root): {root.n_users} users | "
             f"{len(feature_indices)} features (Sex excluded) | "
             f"u_min={u_min_val}")
    log.info(f"  [{sex_label}] Health dist: {root.health_dist}")

    current_level_nodes = [root]

    for m in range(2, max_m + 1):
        log.info(f"\n  [{sex_label}] {'-' * 56}")
        log.info(f"  [{sex_label}] FOCUS LEVEL m={m}: "
                 f"expanding {len(current_level_nodes)} node(s) ...")

        next_level_nodes: List[TreeNode] = []
        total_new = 0

        for parent_node in current_level_nodes:
            n_new = _expand_node_standard(
                parent    = parent_node,
                data      = data,
                labels    = labels,
                kinds     = kinds,
                threshold = threshold,
                tree      = tree,
            )
            total_new        += n_new
            next_level_nodes.extend(parent_node.all_children)

        log.info(f"  [{sex_label}] m={m}: {total_new} new nodes created.")

        if total_new == 0:
            log.info(f"  [{sex_label}] No valid splits at m={m}. Tree complete.")
            break

        current_level_nodes = next_level_nodes

    log.info(f"\n  [{sex_label}] Tree complete: "
             f"depth={tree.depth()}  nodes={tree.count_nodes()}")
    return tree


# ============================================================================
# FORCED SEX FOREST BUILDER
# ============================================================================

def build_forced_sex_forest(
    data:      np.ndarray,
    labels:    np.ndarray,
    threshold: float = DIAGNOSTIC_THRESHOLD,
    max_m:     int   = 2,
) -> ForcedSexForest:
    """
    Algorithm 1 with forced sex branching: build two sex-specific CDS trees.

    [FORCED] The sex of the user determines which tree they enter BEFORE any
    features are examined.  Inside each tree, the standard CDS Algorithm 1
    rules apply on all features except Sex.

    Structure
    ---------
    User arrives
        |
        +-- Sex = 0 (male)   --> male_tree  (Level 1: 426 male users)
        +-- Sex = 1 (female) --> female_tree (Level 1: 526 female users)

    Within each tree:
        Level 1: root with all users of that sex
        Level 2: branches for every feature (except Sex) satisfying Eq.2
        Level 3: further branches within Level 2 nodes (if u_min allows)

    Parameters
    ----------
    data      : (N, 279) feature matrix (NaN for missing values).
    labels    : (N,)     int class labels (1=healthy, 2-16=disease).
    threshold : Eq.2 diagnostic threshold (default 0.025 -> u_min=200).
    max_m     : Maximum focus level within each sex-specific tree (default 2).
                Note: with 426 male / 526 female users and u_min=200,
                Level 3 requires a Level 2 node with >=400 users, which is
                unlikely -- max_m=2 is the practical limit for most splits.

    Returns
    -------
    ForcedSexForest with male_tree and female_tree.
    """
    u_min_val = math.ceil(5 / threshold)

    male_indices   = np.where(data[:, SEX_FEATURE_IDX] == MALE_VALUE)[0]
    female_indices = np.where(data[:, SEX_FEATURE_IDX] == FEMALE_VALUE)[0]

    log.info("=" * 72)
    log.info("Algorithm 1 (FORCED SEX FOREST)")
    log.info("  Two independent CDS trees, one per sex.")
    log.info(f"  threshold={threshold}  u_min={u_min_val}  max_m={max_m}")
    log.info(f"  Total users: {len(labels)} "
             f"(male={len(male_indices)}, female={len(female_indices)})")
    log.info("=" * 72)

    log.info("\n[1/2] Building MALE tree ...")
    male_tree = build_sex_specific_tree(
        data             = data,
        labels           = labels,
        sex_user_indices = male_indices,
        sex_label        = "male",
        threshold        = threshold,
        max_m            = max_m,
    )

    log.info("\n[2/2] Building FEMALE tree ...")
    female_tree = build_sex_specific_tree(
        data             = data,
        labels           = labels,
        sex_user_indices = female_indices,
        sex_label        = "female",
        threshold        = threshold,
        max_m            = max_m,
    )

    log.info("\n" + "=" * 72)
    log.info("Forest complete.")
    log.info(f"  male_tree  : depth={male_tree.depth()}  "
             f"nodes={male_tree.count_nodes()}")
    log.info(f"  female_tree: depth={female_tree.depth()}  "
             f"nodes={female_tree.count_nodes()}")
    log.info("=" * 72)

    return ForcedSexForest(
        male_tree      = male_tree,
        female_tree    = female_tree,
        male_indices   = male_indices,
        female_indices = female_indices,
        n_users        = len(labels),
        threshold      = threshold,
    )


# ============================================================================
# USER ROUTING
# ============================================================================

def route_user(
    forest:   ForcedSexForest,
    user_row: np.ndarray,
) -> Tuple[str, Optional[DecisionTree]]:
    """
    Determine which sex-specific tree this user should traverse.

    This is Step 0 in forced-sex processing.  The user's sex value (col 1)
    is the only information needed -- no other features are examined here.

    Parameters
    ----------
    forest   : ForcedSexForest built by build_forced_sex_forest.
    user_row : 1D array of length 279 (one user's feature vector).

    Returns
    -------
    (sex_label, tree) where:
      sex_label : 'male', 'female', or 'unknown' (if sex is NaN)
      tree      : the corresponding DecisionTree, or None if unknown
    """
    sex = user_row[SEX_FEATURE_IDX]
    if np.isnan(sex):
        return 'unknown', None
    if sex == MALE_VALUE:
        return 'male', forest.male_tree
    elif sex == FEMALE_VALUE:
        return 'female', forest.female_tree
    return 'unknown', None


def get_nodes_for_user(
    tree:     DecisionTree,
    user_idx: int,
) -> Dict[int, List[TreeNode]]:
    """
    Find every node at each focus level that contains `user_idx`.

    Because each level contains branches for many different features, a user
    belongs to multiple nodes at the same level -- one node per feature where
    the user falls in a passing branch.

    Parameters
    ----------
    tree     : A sex-specific DecisionTree (from ForcedSexForest).
    user_idx : Index into the FULL dataset (consistent with user_indices
               stored in each TreeNode).

    Returns
    -------
    Dict[focus_level -> List[TreeNode]] containing this user.
    """
    result: Dict[int, List[TreeNode]] = {}
    user_idx_set = {user_idx}
    for m, nodes in tree.nodes_by_level.items():
        containing = [n for n in nodes
                      if user_idx_set.issubset(set(n.user_indices.tolist()))]
        if containing:
            result[m] = containing
    return result


# ============================================================================
# ASCII-SAFE VALIDATE_EQ2
# CDS_Paper_Algorithms.validate_eq2 uses a Unicode checkmark (U+2713) that
# cp1252 cannot encode on Windows consoles. This override uses only ASCII.
# ============================================================================

def validate_eq2(tree: DecisionTree) -> None:
    """Verify every non-root node satisfies Eq.2. ASCII-safe."""
    print("\n" + "=" * 60)
    print("EQ. 2 VALIDATION (all non-root nodes)")
    print("=" * 60)
    all_ok = True
    fails  = 0
    for m in sorted(tree.nodes_by_level.keys()):
        if m == 1:
            continue
        for node in tree.nodes_by_level[m]:
            parent_id = node.node_id.rsplit("|", 1)[0]
            parent    = tree.all_nodes.get(parent_id)
            if parent is None:
                continue
            passes, lhs, u_min_val = check_branching_condition(
                node.n_users, parent.n_users, tree.threshold
            )
            if not passes:
                all_ok = False
                fails += 1
                print(f"  FAIL: {node.node_id!r} | {lhs} < u_min={u_min_val}")
    if all_ok:
        print("  All nodes satisfy Eq. 2 (OK)")
    else:
        print(f"  {fails} node(s) failed Eq. 2")


# ============================================================================
# REPORTING HELPERS
# ============================================================================

def print_forest_summary(forest: ForcedSexForest) -> None:
    """Print a side-by-side summary of both sex-specific trees."""
    print("\n" + "=" * 72)
    print("FORCED SEX FOREST SUMMARY")
    print("=" * 72)
    print(f"  Total users in dataset : {forest.n_users}")
    print(f"  Male users  (tree 1)   : {len(forest.male_indices)}")
    print(f"  Female users (tree 2)  : {len(forest.female_indices)}")
    print(f"  Threshold              : {forest.threshold}")
    print(f"  u_min                  : {math.ceil(5 / forest.threshold)}")

    for sex_label, tree in [("MALE", forest.male_tree),
                             ("FEMALE", forest.female_tree)]:
        print(f"\n  -- {sex_label} TREE --")
        print(f"  Depth : {tree.depth()}")
        print(f"  Nodes : {tree.count_nodes()}")
        print(f"  {'Level':<8}  {'Nodes':>7}  {'Users (root)':>14}")
        print(f"  {'-'*34}")
        for m in sorted(tree.nodes_by_level.keys()):
            nodes      = tree.nodes_by_level[m]
            root_users = tree.root.n_users
            print(f"  {m:<8}  {len(nodes):>7}  {root_users:>14}")


def print_user_routing(
    forest:   ForcedSexForest,
    data:     np.ndarray,
    user_idx: int,
    labels:   np.ndarray,
) -> None:
    """
    Show how a specific user is routed through the forced-sex forest.

    Prints which tree they enter, then which nodes contain them at each
    focus level of that tree.
    """
    user_row   = data[user_idx]
    sex_label, tree = route_user(forest, user_row)
    true_class = int(labels[user_idx])

    print(f"\n{'=' * 60}")
    print(f"USER ROUTING  (user_idx={user_idx})")
    print(f"  Sex        : {sex_label}  (value={user_row[SEX_FEATURE_IDX]})")
    print(f"  True class : {true_class}")
    print(f"  Tree       : {sex_label} tree")
    print(f"{'=' * 60}")

    if tree is None:
        print("  Cannot route: sex is unknown (NaN).")
        return

    nodes_by_level = get_nodes_for_user(tree, user_idx)

    for m in sorted(tree.nodes_by_level.keys()):
        nodes_here = nodes_by_level.get(m, [])
        print(f"\n  Level {m}:  {len(nodes_here)} node(s) containing user {user_idx}")
        for n in nodes_here[:5]:    # show at most 5 to keep output readable
            feat_name = FEATURE_NAMES.get(n.branching_feat_k, f"k{n.branching_feat_k}")
            lbl       = n.branch_def.label if n.branch_def else "root"
            print(f"    {n.node_id!r}  [{feat_name}: {lbl}]  {n.n_users} users")
        if len(nodes_here) > 5:
            print(f"    ... and {len(nodes_here) - 5} more")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main_forced(data_path: str) -> ForcedSexForest:
    """
    End-to-end execution using forced-sex branching.

    Steps:
      1. Load dataset (accepts augmented or original size).
      2. Build male and female sex-specific CDS trees.
      3. Print forest summary.
      4. Validate Eq.2 on both trees.
      5. Show routing examples for a male and a female user.
    """
    # 1. Load
    data, labels = load_dataset_flexible(data_path)

    # 2. Build (max_m=2 is the effective limit given u_min=200 with ~426/526 users)
    forest = build_forced_sex_forest(data, labels, max_m=2)

    # 3. Summary
    print_forest_summary(forest)

    # 4. Validate Eq.2 on each tree
    print("\n-- Male tree Eq.2 validation --")
    validate_eq2(forest.male_tree)
    print("\n-- Female tree Eq.2 validation --")
    validate_eq2(forest.female_tree)

    # 5. Routing examples: first male and first female in the dataset
    sex_col    = data[:, SEX_FEATURE_IDX]
    male_idx   = int(np.where(sex_col == MALE_VALUE)[0][0])   \
                 if (sex_col == MALE_VALUE).any()   else None
    female_idx = int(np.where(sex_col == FEMALE_VALUE)[0][0]) \
                 if (sex_col == FEMALE_VALUE).any() else None

    for idx in filter(None.__ne__, [male_idx, female_idx]):
        print_user_routing(forest, data, idx, labels)

    return forest


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _base = Path(__file__).parent
        _candidates = [
            str(_base / "genderPlottingAlgorithms" / "arrhythmia_augmented.data"),
            str(_base / "arrhythmia_augmented.data"),
            str(_base / "arrhythmia.data"),
        ]
        _path = next((p for p in _candidates if Path(p).exists()), None)
        if _path is None:
            print("ERROR: could not find arrhythmia.data or arrhythmia_augmented.data")
            sys.exit(1)
    else:
        _path = sys.argv[1]

    main_forced(_path)
