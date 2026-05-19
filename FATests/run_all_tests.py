"""
FATests/run_all_tests.py
========================
Sequential test runner for all targeted experiments.

Enable/disable individual tests by setting the flags below to True/False.
Results are appended to FATests/results.txt after each test completes.

Each test runs the full LOOCV pipeline (452 users) with the specified
modification and records: Accuracy, Sensitivity, Specificity, False Alarm Rate,
plus the features responsible for any healthy-user false alarms.

USAGE:
    cd <project_root>
    python FATests/run_all_tests.py
"""

import sys
import os
import time
import datetime
import logging

# ──────────────────────────────────────────────────────────────────────────────
# ENABLE FLAGS — set True/False to control which tests run
# ──────────────────────────────────────────────────────────────────────────────

# Test 1: LOOCV + bin edge definition
ENABLE_TEST_1A_PER_FOLD_BINNING      = True   # Per-fold binning (baseline)
ENABLE_TEST_1B_GLOBAL_BINNING        = True   # Global binning

# Test 2: Histogram probability smoothing
ENABLE_TEST_2_EPSILON_SMOOTHING      = True   # Add epsilon to bin counts

# Test 3: Healthy-range relaxation
ENABLE_TEST_3A_ONE_BIN_MARGIN        = True   # ± one bin width margin
ENABLE_TEST_3B_PERCENTILE_5_95       = True   # 5–95% percentile bounds
ENABLE_TEST_3C_PERCENTILE_1_99       = True   # 1–99% percentile bounds
ENABLE_TEST_3D_IQR_BOUNDS            = True   # IQR-based bounds

# Test 4: Singleton / rare binary feature handling
ENABLE_TEST_4A_EXCLUDE_RARE          = True   # Exclude rare features
ENABLE_TEST_4B_DOWNWEIGHT_RARE       = True   # Downweight rare features
ENABLE_TEST_4C_MIN_SUPPORT           = True   # Minimum healthy-support count

# Test 5: Feature/action selection leakage check
ENABLE_TEST_5A_FOLD_WISE_SELECTION   = True   # Fold-wise (proper LOOCV, baseline)
ENABLE_TEST_5B_GLOBAL_SELECTION      = True   # Global feature/action selection

# Test 6: Focus-level comparison
ENABLE_TEST_6A_FOCUS_LEVEL_1         = True   # Focus level 1 only
ENABLE_TEST_6B_FOCUS_LEVEL_2_SEX     = True   # Focus level 2, sex-based branching

# Test 7: Validation granularity check
ENABLE_TEST_7A_LEAVE_ONE_SUBJECT_OUT = True   # Leave-one-subject-out
ENABLE_TEST_7B_LEAVE_ONE_RECORD_OUT  = True   # Leave-one-record-out (same if 1 record/subject)

# Test 8: Screening vs false alarm counting
ENABLE_TEST_8_SCREENING_SEPARATION   = True   # Separate screening from false alarms

# ──────────────────────────────────────────────────────────────────────────────

# Add parent directory to path so we can import the algorithm modules
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import random
import copy

from CDS_Paper_Algorithms import (
    load_dataset, build_decision_tree, classify_features,
    DecisionTree, TreeNode, FeatureKind,
    FEATURE_NAMES, HEALTHY_CLASS, DIAGNOSTIC_THRESHOLD, U_MIN, N_FEATURES,
)
from Algorithm2 import (
    Algorithm2Output, ExecutiveActionEntry, PerceptorModelEntry,
    run_algorithm2, DEFAULT_N_BINS, compute_discretization, compute_bayesian_tables,
    compute_healthy_range, compute_executive_actions,
    DiscretizationResult, BayesianTables, HealthyRangeResult,
    ALL_DISEASE_CLASSES, ALL_CLASSES, LAPLACE_EPSILON,
)
from Algorithm3 import run_algorithm3, Algorithm3Output
from Algorithm4 import (
    run_algorithm4, run_loocv, Algorithm4Output, HealthDecision,
    PredictionRecord, DIAGNOSTIC_THRESHOLD_ALG4, ALL_DISEASE_CLASSES as ALG4_DISEASE_CLASSES,
    SEX_FEATURE_INDEX_ALG4,
)
from Algorithm1_forcedBranch import build_sex_specific_tree

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.txt")
DATA_PATH = os.path.join(PROJ_ROOT, "arrhythmia.data")

RNG_SEED = 42

# Suppress verbose training logs
for name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3", "CDS.Alg1.ForcedSex", "CDS.Alg4"):
    logging.getLogger(name).setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

_summary_rows = []

def write_result(test_name: str, metrics: dict, fa_features: list, notes: str = ""):
    """Append one test result to results.txt and record for summary table."""
    _summary_rows.append((test_name, metrics))
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"TEST: {test_name}\n")
        f.write(f"TIME: {datetime.datetime.now().isoformat()}\n")
        f.write(f"{'─'*80}\n")
        f.write(f"  Accuracy:       {metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Sensitivity:    {metrics['sensitivity']*100:.2f}%\n")
        f.write(f"  Specificity:    {metrics['specificity']*100:.2f}%\n")
        f.write(f"  False Alarm Rate: {metrics['false_alarm_rate']*100:.2f}%\n")
        f.write(f"  Screening count:  {metrics.get('n_screening', 'N/A')}\n")
        f.write(f"  Healthy total:    {metrics.get('n_healthy_total', 'N/A')}\n")
        f.write(f"  Diseased total:   {metrics.get('n_diseased_total', 'N/A')}\n")
        if fa_features:
            f.write(f"  FA-causing features (healthy users flagged UNHEALTHY):\n")
            for feat_info in fa_features:
                f.write(f"    - User {feat_info['user']}: feat={feat_info['feat_idx']} "
                        f"({feat_info['feat_name']}), value={feat_info['value']:.4f}, "
                        f"range=[{feat_info['b_min']:.4f}, {feat_info['b_max']:.4f}]\n")
        else:
            f.write(f"  FA-causing features: None (0 false alarms)\n")
        if notes:
            f.write(f"  Notes: {notes}\n")
        f.write(f"{'='*80}\n")
    print(f"  -> Result written to {RESULTS_FILE}")


def extract_metrics(output: Algorithm4Output) -> dict:
    """Extract standard metrics from Algorithm4Output."""
    return {
        "accuracy": output.overall_accuracy,
        "sensitivity": output.sensitivity,
        "specificity": output.specificity,
        "false_alarm_rate": output.false_alarm_rate,
        "n_screening": output.n_screening,
        "n_healthy_total": output.n_healthy_total,
        "n_diseased_total": output.n_diseased_total,
        "n_healthy_correct": output.n_healthy_correct,
        "n_diseased_correct": output.n_diseased_correct,
    }


def extract_fa_features(output: Algorithm4Output, data: np.ndarray) -> list:
    """Extract features responsible for false alarms on healthy users."""
    fa_features = []
    for rec in output.records:
        if rec.true_is_healthy and rec.decision == HealthDecision.UNHEALTHY:
            feat_idx = rec.alarm_feature_idx
            feat_name = FEATURE_NAMES.get(feat_idx, f"feat_{feat_idx}") if feat_idx is not None else "unknown"
            raw_val = float(data[rec.user_global_idx, feat_idx]) if feat_idx is not None else float('nan')
            b_min = float('nan')
            b_max = float('nan')
            for entry in rec.af_trace:
                if entry.triggered_alarm:
                    b_min = entry.b_min
                    b_max = entry.b_max
                    break
            fa_features.append({
                "user": rec.user_global_idx,
                "feat_idx": feat_idx,
                "feat_name": feat_name,
                "value": raw_val,
                "b_min": b_min,
                "b_max": b_max,
            })
    return fa_features


def compute_metrics_from_records(records, data):
    """Compute metrics manually from a list of PredictionRecord objects."""
    n_healthy_correct = sum(1 for r in records if r.true_is_healthy and r.decision != HealthDecision.UNHEALTHY)
    n_healthy_total = sum(1 for r in records if r.true_is_healthy)
    n_diseased_correct = sum(1 for r in records if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY)
    n_diseased_total = sum(1 for r in records if r.true_is_diseased)
    n_screening = sum(1 for r in records if r.decision == HealthDecision.SCREENING)
    n_total = len(records)
    n_correct = n_healthy_correct + n_diseased_correct
    n_fa = sum(1 for r in records if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY)

    return {
        "accuracy": n_correct / n_total if n_total else 0.0,
        "sensitivity": n_diseased_correct / n_diseased_total if n_diseased_total else 0.0,
        "specificity": n_healthy_correct / n_healthy_total if n_healthy_total else 0.0,
        "false_alarm_rate": n_fa / n_healthy_total if n_healthy_total else 0.0,
        "n_screening": n_screening,
        "n_healthy_total": n_healthy_total,
        "n_diseased_total": n_diseased_total,
        "n_healthy_correct": n_healthy_correct,
        "n_diseased_correct": n_diseased_correct,
    }


def extract_fa_features_from_records(records, data):
    """Extract FA features from raw records list."""
    fa_features = []
    for rec in records:
        if rec.true_is_healthy and rec.decision == HealthDecision.UNHEALTHY:
            feat_idx = rec.alarm_feature_idx
            feat_name = FEATURE_NAMES.get(feat_idx, f"feat_{feat_idx}") if feat_idx is not None else "unknown"
            raw_val = float(data[rec.user_global_idx, feat_idx]) if feat_idx is not None else float('nan')
            b_min = float('nan')
            b_max = float('nan')
            for entry in rec.af_trace:
                if entry.triggered_alarm:
                    b_min = entry.b_min
                    b_max = entry.b_max
                    break
            fa_features.append({
                "user": rec.user_global_idx,
                "feat_idx": feat_idx,
                "feat_name": feat_name,
                "value": raw_val,
                "b_min": b_min,
                "b_max": b_max,
            })
    return fa_features


# ──────────────────────────────────────────────────────────────────────────────
# MODIFIED LOOCV PIPELINES
# ──────────────────────────────────────────────────────────────────────────────

def run_standard_loocv(data, labels, max_focus=2, force_sex_branch=False,
                       n_bins=DEFAULT_N_BINS, augment=False):
    """Run the standard per-fold LOOCV pipeline. Returns list of PredictionRecord."""
    from Algorithm1_forcedBranch import build_sex_specific_tree

    n_total = data.shape[0]
    records = []
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    for i in range(n_total):
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[i] = False
        train_data = data[train_mask]
        train_labels = labels[train_mask]

        if force_sex_branch:
            test_user_sex = data[i, SEX_FEATURE_INDEX_ALG4]
            if test_user_sex == 0:
                sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 0)[0]
                tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "male")
            else:
                sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 1)[0]
                tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "female")
        else:
            tree_i = build_decision_tree(train_data, train_labels)

        root_id = tree_i.root.node_id
        nodes_filter_i = [root_id]
        if force_sex_branch:
            if 2 in tree_i.nodes_by_level:
                for child in tree_i.nodes_by_level[2]:
                    if not child.is_leaf:
                        nodes_filter_i.append(child.node_id)
        else:
            sex_k = SEX_FEATURE_INDEX_ALG4
            sex_children = [n for n in tree_i.nodes_by_level.get(2, [])
                           if n.branching_feat_k == sex_k and not n.is_leaf]
            if len(sex_children) >= 2:
                for child in sex_children:
                    nodes_filter_i.append(child.node_id)

        alg2_i = run_algorithm2(tree_i, train_data, train_labels,
                                n_bins=n_bins, nodes_filter=nodes_filter_i)
        alg3_i = run_algorithm3(alg2_i, tree_i, train_data, train_labels,
                                nodes_filter=nodes_filter_i, reset_per_h=False)

        pred = run_algorithm4(i, data, labels, tree_i, alg2_i, alg3_i,
                              rng_seed=RNG_SEED, train_data=train_data,
                              train_labels=train_labels)
        records.append(pred)

        if (i + 1) % 50 == 0 or i == n_total - 1:
            n_correct = sum(1 for r in records if r.is_correct)
            print(f"    Progress: {i+1}/{n_total}  acc={n_correct/(i+1)*100:.1f}%")

    return records


def run_loocv_focus_level_1_only(data, labels, force_sex_branch=False, n_bins=DEFAULT_N_BINS):
    """Run LOOCV but only at focus level 1 (root node only, no level-2 branching)."""
    from Algorithm1_forcedBranch import build_sex_specific_tree

    n_total = data.shape[0]
    records = []
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    for i in range(n_total):
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[i] = False
        train_data = data[train_mask]
        train_labels = labels[train_mask]

        if force_sex_branch:
            test_user_sex = data[i, SEX_FEATURE_INDEX_ALG4]
            if test_user_sex == 0:
                sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 0)[0]
                tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "male")
            else:
                sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 1)[0]
                tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "female")
        else:
            tree_i = build_decision_tree(train_data, train_labels)

        root_id = tree_i.root.node_id
        # Only root node — no level-2 nodes
        nodes_filter_i = [root_id]

        alg2_i = run_algorithm2(tree_i, train_data, train_labels,
                                n_bins=n_bins, nodes_filter=nodes_filter_i)
        alg3_i = run_algorithm3(alg2_i, tree_i, train_data, train_labels,
                                nodes_filter=nodes_filter_i, reset_per_h=False)

        pred = run_algorithm4(i, data, labels, tree_i, alg2_i, alg3_i,
                              rng_seed=RNG_SEED, train_data=train_data,
                              train_labels=train_labels)
        records.append(pred)

        if (i + 1) % 50 == 0 or i == n_total - 1:
            n_correct = sum(1 for r in records if r.is_correct)
            print(f"    Progress: {i+1}/{n_total}  acc={n_correct/(i+1)*100:.1f}%")

    return records


def run_loocv_global_binning(data, labels, force_sex_branch=False, n_bins=DEFAULT_N_BINS):
    """
    Test 1B: Train once on the full dataset, then test each user against
    those globally-trained models. This is NOT proper LOOCV — the test user's
    data is included in training. The point is to see whether performance
    changes because bin boundaries are fixed (no per-fold shift).
    """
    from Algorithm1_forcedBranch import build_sex_specific_tree

    n_total = data.shape[0]
    records = []
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    # Train once on the full dataset
    print("    Training once on full dataset (global binning)...")
    if force_sex_branch:
        male_idx = np.where(data[:, SEX_FEATURE_INDEX_ALG4] == 0)[0]
        female_idx = np.where(data[:, SEX_FEATURE_INDEX_ALG4] == 1)[0]
        global_tree_male = build_sex_specific_tree(data, labels, male_idx, "male")
        global_tree_female = build_sex_specific_tree(data, labels, female_idx, "female")

        nodes_male = [global_tree_male.root.node_id]
        if 2 in global_tree_male.nodes_by_level:
            for child in global_tree_male.nodes_by_level[2]:
                if not child.is_leaf:
                    nodes_male.append(child.node_id)

        nodes_female = [global_tree_female.root.node_id]
        if 2 in global_tree_female.nodes_by_level:
            for child in global_tree_female.nodes_by_level[2]:
                if not child.is_leaf:
                    nodes_female.append(child.node_id)

        alg2_male = run_algorithm2(global_tree_male, data, labels,
                                    n_bins=n_bins, nodes_filter=nodes_male)
        alg2_female = run_algorithm2(global_tree_female, data, labels,
                                      n_bins=n_bins, nodes_filter=nodes_female)
        alg3_male = run_algorithm3(alg2_male, global_tree_male, data, labels,
                                    nodes_filter=nodes_male, reset_per_h=False)
        alg3_female = run_algorithm3(alg2_female, global_tree_female, data, labels,
                                      nodes_filter=nodes_female, reset_per_h=False)
    else:
        global_tree = build_decision_tree(data, labels)
        root_id = global_tree.root.node_id
        nodes_filter = [root_id]
        sex_children = [n for n in global_tree.nodes_by_level.get(2, [])
                       if n.branching_feat_k == SEX_FEATURE_INDEX_ALG4 and not n.is_leaf]
        if len(sex_children) >= 2:
            for child in sex_children:
                nodes_filter.append(child.node_id)
        alg2_global = run_algorithm2(global_tree, data, labels,
                                      n_bins=n_bins, nodes_filter=nodes_filter)
        alg3_global = run_algorithm3(alg2_global, global_tree, data, labels,
                                      nodes_filter=nodes_filter, reset_per_h=False)

    print("    Global models computed. Running per-user predictions...")

    # Test each user against the globally trained models
    for i in range(n_total):
        if force_sex_branch:
            test_user_sex = data[i, SEX_FEATURE_INDEX_ALG4]
            if test_user_sex == 0:
                tree_i = global_tree_male
                alg2_i = alg2_male
                alg3_i = alg3_male
            else:
                tree_i = global_tree_female
                alg2_i = alg2_female
                alg3_i = alg3_female
        else:
            tree_i = global_tree
            alg2_i = alg2_global
            alg3_i = alg3_global

        pred = run_algorithm4(i, data, labels, tree_i, alg2_i, alg3_i,
                              rng_seed=RNG_SEED)
        records.append(pred)

        if (i + 1) % 50 == 0 or i == n_total - 1:
            n_correct = sum(1 for r in records if r.is_correct)
            print(f"    Progress: {i+1}/{n_total}  acc={n_correct/(i+1)*100:.1f}%")

    return records


def run_loocv_epsilon_smoothing(data, labels, epsilon=1e-6, force_sex_branch=False,
                                 n_bins=DEFAULT_N_BINS):
    """
    Test 2: Add epsilon smoothing to all bin counts/probabilities before
    normalization in Algorithm 2's Bayesian tables.

    Since LAPLACE_EPSILON is captured as a default arg at definition time,
    we monkey-patch compute_bayesian_tables to inject our epsilon.
    """
    from Algorithm1_forcedBranch import build_sex_specific_tree
    import Algorithm2 as alg2_mod

    n_total = data.shape[0]
    records = []
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    # Monkey-patch: replace compute_bayesian_tables with a wrapper that passes our epsilon
    original_fn = alg2_mod.compute_bayesian_tables

    def patched_bayesian_tables(disc, node, labels, class_labels=None, N_total=0,
                                 laplace_eps=epsilon):
        return original_fn(disc=disc, node=node, labels=labels,
                          class_labels=class_labels, N_total=N_total,
                          laplace_eps=laplace_eps)

    alg2_mod.compute_bayesian_tables = patched_bayesian_tables

    try:
        for i in range(n_total):
            train_mask = np.ones(n_total, dtype=bool)
            train_mask[i] = False
            train_data = data[train_mask]
            train_labels = labels[train_mask]

            if force_sex_branch:
                test_user_sex = data[i, SEX_FEATURE_INDEX_ALG4]
                if test_user_sex == 0:
                    sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 0)[0]
                    tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "male")
                else:
                    sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 1)[0]
                    tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "female")
            else:
                tree_i = build_decision_tree(train_data, train_labels)

            root_id = tree_i.root.node_id
            nodes_filter_i = [root_id]
            if force_sex_branch:
                if 2 in tree_i.nodes_by_level:
                    for child in tree_i.nodes_by_level[2]:
                        if not child.is_leaf:
                            nodes_filter_i.append(child.node_id)

            alg2_i = run_algorithm2(tree_i, train_data, train_labels,
                                    n_bins=n_bins, nodes_filter=nodes_filter_i)
            alg3_i = run_algorithm3(alg2_i, tree_i, train_data, train_labels,
                                    nodes_filter=nodes_filter_i, reset_per_h=False)

            pred = run_algorithm4(i, data, labels, tree_i, alg2_i, alg3_i,
                                  rng_seed=RNG_SEED, train_data=train_data,
                                  train_labels=train_labels)
            records.append(pred)

            if (i + 1) % 50 == 0 or i == n_total - 1:
                n_correct = sum(1 for r in records if r.is_correct)
                print(f"    Progress: {i+1}/{n_total}  acc={n_correct/(i+1)*100:.1f}%")
    finally:
        alg2_mod.compute_bayesian_tables = original_fn

    return records


def run_loocv_relaxed_range(data, labels, mode="one_bin", force_sex_branch=False,
                             n_bins=DEFAULT_N_BINS):
    """
    Test 3: Relax the healthy range boundaries.

    Modes:
    - "one_bin": ± one bin width margin
    - "percentile_5_95": 5–95% percentile of ALL users (not just healthy)
    - "percentile_1_99": 1–99% percentile
    - "iqr": IQR-based bounds (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    """
    from Algorithm1_forcedBranch import build_sex_specific_tree

    n_total = data.shape[0]
    records = []
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    for i in range(n_total):
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[i] = False
        train_data = data[train_mask]
        train_labels = labels[train_mask]

        if force_sex_branch:
            test_user_sex = data[i, SEX_FEATURE_INDEX_ALG4]
            if test_user_sex == 0:
                sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 0)[0]
                tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "male")
            else:
                sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 1)[0]
                tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "female")
        else:
            tree_i = build_decision_tree(train_data, train_labels)

        root_id = tree_i.root.node_id
        nodes_filter_i = [root_id]
        if force_sex_branch:
            if 2 in tree_i.nodes_by_level:
                for child in tree_i.nodes_by_level[2]:
                    if not child.is_leaf:
                        nodes_filter_i.append(child.node_id)

        alg2_i = run_algorithm2(tree_i, train_data, train_labels,
                                n_bins=n_bins, nodes_filter=nodes_filter_i)

        # Post-hoc relaxation of healthy ranges
        for entry in alg2_i.perceptor_library:
            node = None
            for n in [tree_i.root] + [c for lvl in tree_i.nodes_by_level.values() for c in lvl]:
                if n.node_id == entry.node_id:
                    node = n
                    break
            if node is None:
                continue

            feat_idx = entry.feature_idx
            hr = entry.healthy_range
            disc = entry.disc

            if mode == "one_bin":
                margin = disc.delta_b
                hr.b_min_healthy -= margin
                hr.b_max_healthy += margin

            elif mode == "percentile_5_95":
                raw_vals = train_data[node.user_indices, feat_idx]
                valid = raw_vals[~np.isnan(raw_vals)]
                if len(valid) > 0:
                    hr.b_min_healthy = float(np.percentile(valid, 5))
                    hr.b_max_healthy = float(np.percentile(valid, 95))

            elif mode == "percentile_1_99":
                raw_vals = train_data[node.user_indices, feat_idx]
                valid = raw_vals[~np.isnan(raw_vals)]
                if len(valid) > 0:
                    hr.b_min_healthy = float(np.percentile(valid, 1))
                    hr.b_max_healthy = float(np.percentile(valid, 99))

            elif mode == "iqr":
                raw_vals = train_data[node.user_indices, feat_idx]
                valid = raw_vals[~np.isnan(raw_vals)]
                if len(valid) > 0:
                    q1 = float(np.percentile(valid, 25))
                    q3 = float(np.percentile(valid, 75))
                    iqr = q3 - q1
                    hr.b_min_healthy = q1 - 1.5 * iqr
                    hr.b_max_healthy = q3 + 1.5 * iqr

        alg3_i = run_algorithm3(alg2_i, tree_i, train_data, train_labels,
                                nodes_filter=nodes_filter_i, reset_per_h=False)

        pred = run_algorithm4(i, data, labels, tree_i, alg2_i, alg3_i,
                              rng_seed=RNG_SEED, train_data=train_data,
                              train_labels=train_labels)
        records.append(pred)

        if (i + 1) % 50 == 0 or i == n_total - 1:
            n_correct = sum(1 for r in records if r.is_correct)
            print(f"    Progress: {i+1}/{n_total}  acc={n_correct/(i+1)*100:.1f}%")

    return records


def run_loocv_rare_feature_handling(data, labels, mode="exclude",
                                     min_support=5, downweight_factor=0.1,
                                     force_sex_branch=False, n_bins=DEFAULT_N_BINS):
    """
    Test 4: Handle singleton/rare binary features in healthy population.

    Modes:
    - "exclude": Remove features where one binary value appears < min_support times among healthy
    - "downweight": Multiply action weight by downweight_factor for rare features
    - "min_support": Require minimum healthy-support count before feature can be used
    """
    from Algorithm1_forcedBranch import build_sex_specific_tree

    n_total = data.shape[0]
    records = []
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    for i in range(n_total):
        train_mask = np.ones(n_total, dtype=bool)
        train_mask[i] = False
        train_data = data[train_mask]
        train_labels = labels[train_mask]

        if force_sex_branch:
            test_user_sex = data[i, SEX_FEATURE_INDEX_ALG4]
            if test_user_sex == 0:
                sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 0)[0]
                tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "male")
            else:
                sex_indices = np.where(train_data[:, SEX_FEATURE_INDEX_ALG4] == 1)[0]
                tree_i = build_sex_specific_tree(train_data, train_labels, sex_indices, "female")
        else:
            tree_i = build_decision_tree(train_data, train_labels)

        root_id = tree_i.root.node_id
        nodes_filter_i = [root_id]
        if force_sex_branch:
            if 2 in tree_i.nodes_by_level:
                for child in tree_i.nodes_by_level[2]:
                    if not child.is_leaf:
                        nodes_filter_i.append(child.node_id)

        alg2_i = run_algorithm2(tree_i, train_data, train_labels,
                                n_bins=n_bins, nodes_filter=nodes_filter_i)

        # Identify rare features per node
        feature_kinds = classify_features(train_data)
        rare_features_per_node = {}

        for entry in alg2_i.perceptor_library:
            node = None
            for n in [tree_i.root] + [c for lvl in tree_i.nodes_by_level.values() for c in lvl]:
                if n.node_id == entry.node_id:
                    node = n
                    break
            if node is None:
                continue

            feat_idx = entry.feature_idx
            kind = feature_kinds.get(feat_idx, FeatureKind.CONTINUOUS)

            if kind == FeatureKind.BINARY:
                healthy_mask = train_labels[node.user_indices] == HEALTHY_CLASS
                healthy_vals = train_data[node.user_indices[healthy_mask], feat_idx]
                healthy_valid = healthy_vals[~np.isnan(healthy_vals)]

                if len(healthy_valid) > 0:
                    count_0 = (healthy_valid == 0).sum()
                    count_1 = (healthy_valid == 1).sum()
                    min_count = min(count_0, count_1)

                    if min_count < min_support:
                        if entry.node_id not in rare_features_per_node:
                            rare_features_per_node[entry.node_id] = set()
                        rare_features_per_node[entry.node_id].add(feat_idx)

        # Apply the modification to the executive library
        if mode == "exclude":
            alg2_i.executive_library = [
                e for e in alg2_i.executive_library
                if e.feature_idx not in rare_features_per_node.get(e.node_id, set())
            ]
            # Rebuild index
            alg2_i.executive_index = {
                (e.node_id, e.feature_idx, e.disease_class): e
                for e in alg2_i.executive_library
            }

        elif mode == "downweight":
            for e in alg2_i.executive_library:
                if e.feature_idx in rare_features_per_node.get(e.node_id, set()):
                    e.action_weight *= downweight_factor

        elif mode == "min_support":
            # Remove actions where the feature's healthy support is below threshold
            for entry in alg2_i.perceptor_library:
                node = None
                for n in [tree_i.root] + [c for lvl in tree_i.nodes_by_level.values() for c in lvl]:
                    if n.node_id == entry.node_id:
                        node = n
                        break
                if node is None:
                    continue
                if entry.healthy_range.n_healthy_valid < min_support:
                    # Remove all executive actions for this (node, feature)
                    alg2_i.executive_library = [
                        e for e in alg2_i.executive_library
                        if not (e.node_id == entry.node_id and e.feature_idx == entry.feature_idx)
                    ]
            alg2_i.executive_index = {
                (e.node_id, e.feature_idx, e.disease_class): e
                for e in alg2_i.executive_library
            }

        alg3_i = run_algorithm3(alg2_i, tree_i, train_data, train_labels,
                                nodes_filter=nodes_filter_i, reset_per_h=False)

        pred = run_algorithm4(i, data, labels, tree_i, alg2_i, alg3_i,
                              rng_seed=RNG_SEED, train_data=train_data,
                              train_labels=train_labels)
        records.append(pred)

        if (i + 1) % 50 == 0 or i == n_total - 1:
            n_correct = sum(1 for r in records if r.is_correct)
            print(f"    Progress: {i+1}/{n_total}  acc={n_correct/(i+1)*100:.1f}%")

    return records


def run_loocv_global_selection(data, labels, force_sex_branch=False, n_bins=DEFAULT_N_BINS):
    """
    Test 5B: Compute feature ranking and action library ONCE on full dataset,
    then reuse across all LOOCV folds (testing for information leakage).
    """
    from Algorithm1_forcedBranch import build_sex_specific_tree

    n_total = data.shape[0]
    records = []
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    # Precompute global tree, Alg2, Alg3 on full dataset
    print("    Precomputing global models on full dataset (information leakage test)...")
    if force_sex_branch:
        male_idx = np.where(data[:, SEX_FEATURE_INDEX_ALG4] == 0)[0]
        female_idx = np.where(data[:, SEX_FEATURE_INDEX_ALG4] == 1)[0]
        global_tree_male = build_sex_specific_tree(data, labels, male_idx, "male")
        global_tree_female = build_sex_specific_tree(data, labels, female_idx, "female")

        nodes_male = [global_tree_male.root.node_id]
        if 2 in global_tree_male.nodes_by_level:
            for child in global_tree_male.nodes_by_level[2]:
                if not child.is_leaf:
                    nodes_male.append(child.node_id)

        nodes_female = [global_tree_female.root.node_id]
        if 2 in global_tree_female.nodes_by_level:
            for child in global_tree_female.nodes_by_level[2]:
                if not child.is_leaf:
                    nodes_female.append(child.node_id)

        alg2_male = run_algorithm2(global_tree_male, data, labels,
                                    n_bins=n_bins, nodes_filter=nodes_male)
        alg2_female = run_algorithm2(global_tree_female, data, labels,
                                      n_bins=n_bins, nodes_filter=nodes_female)
        alg3_male = run_algorithm3(alg2_male, global_tree_male, data, labels,
                                    nodes_filter=nodes_male, reset_per_h=False)
        alg3_female = run_algorithm3(alg2_female, global_tree_female, data, labels,
                                      nodes_filter=nodes_female, reset_per_h=False)
    else:
        global_tree = build_decision_tree(data, labels)
        root_id = global_tree.root.node_id
        nodes_filter = [root_id]
        sex_children = [n for n in global_tree.nodes_by_level.get(2, [])
                       if n.branching_feat_k == SEX_FEATURE_INDEX_ALG4 and not n.is_leaf]
        if len(sex_children) >= 2:
            for child in sex_children:
                nodes_filter.append(child.node_id)

        alg2_global = run_algorithm2(global_tree, data, labels,
                                      n_bins=n_bins, nodes_filter=nodes_filter)
        alg3_global = run_algorithm3(alg2_global, global_tree, data, labels,
                                      nodes_filter=nodes_filter, reset_per_h=False)

    print("    Global models computed. Running per-user predictions...")

    for i in range(n_total):
        if force_sex_branch:
            test_user_sex = data[i, SEX_FEATURE_INDEX_ALG4]
            if test_user_sex == 0:
                tree_i = global_tree_male
                alg2_i = alg2_male
                alg3_i = alg3_male
            else:
                tree_i = global_tree_female
                alg2_i = alg2_female
                alg3_i = alg3_female
        else:
            tree_i = global_tree
            alg2_i = alg2_global
            alg3_i = alg3_global

        pred = run_algorithm4(i, data, labels, tree_i, alg2_i, alg3_i,
                              rng_seed=RNG_SEED)
        records.append(pred)

        if (i + 1) % 50 == 0 or i == n_total - 1:
            n_correct = sum(1 for r in records if r.is_correct)
            print(f"    Progress: {i+1}/{n_total}  acc={n_correct/(i+1)*100:.1f}%")

    return records


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TEST RUNNER
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("CDS-NI Algorithm: Targeted Experiment Suite")
    print("="*80)

    # Clear results file
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("CDS-NI Algorithm — Targeted Experiment Results\n")
        f.write(f"Run started: {datetime.datetime.now().isoformat()}\n")
        f.write("="*80 + "\n")

    # Summary rows are collected automatically by write_result() into _summary_rows

    # Load dataset
    print("\nLoading dataset...")
    data, labels = load_dataset(DATA_PATH)
    print(f"  Dataset: {data.shape[0]} users x {data.shape[1]} features")
    n_healthy = (labels == HEALTHY_CLASS).sum()
    n_diseased = (labels != HEALTHY_CLASS).sum()
    print(f"  Healthy: {n_healthy}, Diseased: {n_diseased}")

    # Cache for the standard baseline LOOCV result (reused by tests 1A, 5A, 6B, 7A, 7B, 8)
    baseline_records = None

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1A: Per-fold binning (baseline)
    # ══════════════════════════════════════════════════════════════════════════
    # Run baseline once if any test needs it
    needs_baseline = any([
        ENABLE_TEST_1A_PER_FOLD_BINNING, ENABLE_TEST_5A_FOLD_WISE_SELECTION,
        ENABLE_TEST_6B_FOCUS_LEVEL_2_SEX, ENABLE_TEST_7A_LEAVE_ONE_SUBJECT_OUT,
        ENABLE_TEST_7B_LEAVE_ONE_RECORD_OUT, ENABLE_TEST_8_SCREENING_SEPARATION,
    ])
    if needs_baseline and baseline_records is None:
        print("\n" + "━"*80)
        print("Running BASELINE LOOCV (shared by tests 1A, 5A, 6B, 7A, 7B, 8)...")
        print("━"*80)
        t0 = time.time()
        baseline_records = run_standard_loocv(data, labels, force_sex_branch=False)
        baseline_elapsed = time.time() - t0
        print(f"  Baseline complete in {baseline_elapsed:.1f}s")

    if ENABLE_TEST_1A_PER_FOLD_BINNING:
        print("\n" + "━"*80)
        print("TEST 1A: Per-fold binning (baseline — bin edges computed per LOOCV fold)")
        print("━"*80)
        records = baseline_records
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("1A: Per-fold binning (baseline)", metrics, fa_feats,
                     f"Standard LOOCV with per-fold retraining.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1B: Global binning
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_1B_GLOBAL_BINNING:
        print("\n" + "━"*80)
        print("TEST 1B: Global binning (bin edges from full dataset)")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_global_binning(data, labels, force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("1B: Global binning (train once, test all)", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Trained once on full dataset, tested each user against global models.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2: Epsilon smoothing
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_2_EPSILON_SMOOTHING:
        print("\n" + "━"*80)
        print("TEST 2: Histogram probability smoothing (epsilon=1e-6)")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_epsilon_smoothing(data, labels, epsilon=1e-6,
                                               force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("2: Epsilon smoothing (eps=1e-6)", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Laplace smoothing added to bin probabilities.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3A: One-bin margin
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_3A_ONE_BIN_MARGIN:
        print("\n" + "━"*80)
        print("TEST 3A: Healthy-range relaxation — ± one bin width margin")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_relaxed_range(data, labels, mode="one_bin",
                                           force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("3A: ± One-bin-width margin", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Healthy range expanded by ±delta_B.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3B: 5–95% percentile
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_3B_PERCENTILE_5_95:
        print("\n" + "━"*80)
        print("TEST 3B: Healthy-range relaxation — 5–95% percentile (all users)")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_relaxed_range(data, labels, mode="percentile_5_95",
                                           force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("3B: 5-95% percentile bounds", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Healthy range set to 5-95 percentile of all node users.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3C: 1–99% percentile
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_3C_PERCENTILE_1_99:
        print("\n" + "━"*80)
        print("TEST 3C: Healthy-range relaxation — 1–99% percentile (all users)")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_relaxed_range(data, labels, mode="percentile_1_99",
                                           force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("3C: 1-99% percentile bounds", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Healthy range set to 1-99 percentile of all node users.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3D: IQR-based bounds
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_3D_IQR_BOUNDS:
        print("\n" + "━"*80)
        print("TEST 3D: Healthy-range relaxation — IQR-based bounds")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_relaxed_range(data, labels, mode="iqr",
                                           force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("3D: IQR-based bounds", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Healthy range set to [Q1-1.5*IQR, Q3+1.5*IQR].")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 4A: Exclude rare binary features
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_4A_EXCLUDE_RARE:
        print("\n" + "━"*80)
        print("TEST 4A: Singleton handling — exclude rare binary features (value appears <=1 time)")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_rare_feature_handling(data, labels, mode="exclude",
                                                   min_support=2, force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("4A: Exclude singleton binary features", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Binary features where one value appears <=1 time among healthy users removed.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 4B: Downweight rare binary features
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_4B_DOWNWEIGHT_RARE:
        print("\n" + "━"*80)
        print("TEST 4B: Singleton handling — downweight rare features (factor=0.1)")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_rare_feature_handling(data, labels, mode="downweight",
                                                   min_support=2, downweight_factor=0.1,
                                                   force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("4B: Downweight rare features (0.1x)", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Rare binary features' action weights multiplied by 0.1.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 4C: Minimum healthy-support count
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_4C_MIN_SUPPORT:
        print("\n" + "━"*80)
        print("TEST 4C: Singleton handling — minimum healthy-support count (>=5)")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_rare_feature_handling(data, labels, mode="min_support",
                                                   min_support=5, force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("4C: Min healthy support (>=5)", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Features with <5 healthy users with valid values excluded.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 5A: Fold-wise feature/action selection (proper LOOCV, same as 1A baseline)
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_5A_FOLD_WISE_SELECTION:
        print("\n" + "━"*80)
        print("TEST 5A: Fold-wise feature/action selection (proper LOOCV)")
        print("━"*80)
        records = baseline_records
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("5A: Fold-wise selection (proper LOOCV)", metrics, fa_feats,
                     f"All Alg1-3 retrained per fold. Same as 1A baseline.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 5B: Global feature/action selection (leakage test)
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_5B_GLOBAL_SELECTION:
        print("\n" + "━"*80)
        print("TEST 5B: Global feature/action selection (leakage test)")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_global_selection(data, labels, force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("5B: Global selection (information leakage)", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Alg1-3 trained ONCE on full dataset, reused for all folds.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 6A: Focus level 1 only
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_6A_FOCUS_LEVEL_1:
        print("\n" + "━"*80)
        print("TEST 6A: Focus level 1 only (no sex-based branching at level 2)")
        print("━"*80)
        t0 = time.time()
        records = run_loocv_focus_level_1_only(data, labels, force_sex_branch=False)
        elapsed = time.time() - t0
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("6A: Focus level 1 only", metrics, fa_feats,
                     f"Elapsed: {elapsed:.1f}s. Only root node used, no level-2 branching.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 6B: Focus level 2 with sex-based branching
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_6B_FOCUS_LEVEL_2_SEX:
        print("\n" + "━"*80)
        print("TEST 6B: Focus level 2 with sex-based branching (paper's approach)")
        print("━"*80)
        records = baseline_records
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("6B: Focus level 2 + sex branching", metrics, fa_feats,
                     f"Full pipeline with forced sex branching at level 2. Same as baseline.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 7A: Leave-one-subject-out
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_7A_LEAVE_ONE_SUBJECT_OUT:
        print("\n" + "━"*80)
        print("TEST 7A: Leave-one-subject-out (standard LOOCV — 1 record per subject)")
        print("━"*80)
        records = baseline_records
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("7A: Leave-one-subject-out", metrics, fa_feats,
                     f"UCI Arrhythmia has 1 record per subject; LOSO = LORO. Same as baseline.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 7B: Leave-one-record-out
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_7B_LEAVE_ONE_RECORD_OUT:
        print("\n" + "━"*80)
        print("TEST 7B: Leave-one-record-out (identical to 7A for this dataset)")
        print("━"*80)
        records = baseline_records
        metrics = compute_metrics_from_records(records, data)
        fa_feats = extract_fa_features_from_records(records, data)
        write_result("7B: Leave-one-record-out", metrics, fa_feats,
                     f"Identical to 7A because UCI Arrhythmia has exactly 1 record per subject "
                     f"(452 subjects, 452 rows). Same as baseline.")
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 8: Screening vs false alarm counting
    # ══════════════════════════════════════════════════════════════════════════
    if ENABLE_TEST_8_SCREENING_SEPARATION:
        print("\n" + "━"*80)
        print("TEST 8: Screening vs false alarm separation")
        print("━"*80)
        records = baseline_records

        # Compute detailed breakdown
        n_total = len(records)
        n_healthy_total = sum(1 for r in records if r.true_is_healthy)
        n_diseased_total = sum(1 for r in records if r.true_is_diseased)

        # Healthy users breakdown
        healthy_as_healthy = sum(1 for r in records if r.true_is_healthy and r.decision == HealthDecision.HEALTHY)
        healthy_as_screening = sum(1 for r in records if r.true_is_healthy and r.decision == HealthDecision.SCREENING)
        healthy_as_unhealthy = sum(1 for r in records if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY)

        # Diseased users breakdown
        diseased_as_unhealthy = sum(1 for r in records if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY)
        diseased_as_healthy = sum(1 for r in records if r.true_is_diseased and r.decision == HealthDecision.HEALTHY)
        diseased_as_screening = sum(1 for r in records if r.true_is_diseased and r.decision == HealthDecision.SCREENING)

        # Standard metrics (screening = soft healthy, as in paper)
        specificity_paper = (healthy_as_healthy + healthy_as_screening) / n_healthy_total if n_healthy_total else 0
        # Strict specificity (only HEALTHY counts, screening = uncertain)
        specificity_strict = healthy_as_healthy / n_healthy_total if n_healthy_total else 0

        sensitivity = diseased_as_unhealthy / n_diseased_total if n_diseased_total else 0
        fa_rate = healthy_as_unhealthy / n_healthy_total if n_healthy_total else 0

        metrics = {
            "accuracy": sum(1 for r in records if r.is_correct) / n_total if n_total else 0,
            "sensitivity": sensitivity,
            "specificity": specificity_paper,
            "false_alarm_rate": fa_rate,
            "n_screening": sum(1 for r in records if r.decision == HealthDecision.SCREENING),
            "n_healthy_total": n_healthy_total,
            "n_diseased_total": n_diseased_total,
        }
        fa_feats = extract_fa_features_from_records(records, data)

        notes = (
            f"Uses baseline LOOCV records.\n"
            f"  DETAILED BREAKDOWN:\n"
            f"  Healthy users ({n_healthy_total} total):\n"
            f"    -> Classified HEALTHY:   {healthy_as_healthy}\n"
            f"    -> Classified SCREENING: {healthy_as_screening}\n"
            f"    -> Classified UNHEALTHY (false alarm): {healthy_as_unhealthy}\n"
            f"  Diseased users ({n_diseased_total} total):\n"
            f"    -> Classified UNHEALTHY (correct alarm): {diseased_as_unhealthy}\n"
            f"    -> Classified HEALTHY (missed):          {diseased_as_healthy}\n"
            f"    -> Classified SCREENING (uncertain):     {diseased_as_screening}\n"
            f"  Specificity (paper, screening=soft-healthy): {specificity_paper*100:.2f}%\n"
            f"  Specificity (strict, only HEALTHY):          {specificity_strict*100:.2f}%\n"
            f"  If screening referrals counted as false alarms:\n"
            f"    FA rate would be: {(healthy_as_unhealthy+healthy_as_screening)/n_healthy_total*100:.2f}%\n"
        )

        write_result("8: Screening vs false alarm separation", metrics, fa_feats, notes)
        print(f"  Accuracy={metrics['accuracy']*100:.2f}% Sens={metrics['sensitivity']*100:.2f}% "
              f"Spec={metrics['specificity']*100:.2f}% FA={metrics['false_alarm_rate']*100:.2f}%")
        print(f"  Screening total: {metrics['n_screening']}")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write("\n\n" + "="*100 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("="*100 + "\n")
        header = f"{'Test':<50} {'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12} {'FA Rate':>10} {'Screening':>10}\n"
        f.write(header)
        f.write("-"*100 + "\n")
        for name, m in _summary_rows:
            row = (f"{name:<50} "
                   f"{m['accuracy']*100:>9.2f}% "
                   f"{m['sensitivity']*100:>11.2f}% "
                   f"{m['specificity']*100:>11.2f}% "
                   f"{m['false_alarm_rate']*100:>9.2f}% "
                   f"{m.get('n_screening','N/A'):>10}\n")
            f.write(row)
        f.write("="*100 + "\n")

    print("\n\n" + "="*100)
    print("ALL TESTS COMPLETE — Results saved to FATests/results.txt")
    print("="*100)


if __name__ == "__main__":
    main()
