"""
featureImportanceByGender.py

Per-gender feature importance using Algorithm 3's refined action weights.

METHOD
------
The CDS system's own mechanism for ranking feature importance is the
executive action library produced by Algorithm 2 and refined by Algorithm 3.
After Algorithm 3, a feature is "important" for disease class h if it was
RETAINED (i.e., it contributes new user coverage beyond what prior features
already achieved).

For each retained action (node, feature, disease_h), the action weight
r_{o|h} = P(B_hat < b_min | h) + P(B_hat > b_max | h) measures the
fraction of disease-h users whose feature value falls outside the healthy
range.  Higher r_{o|h} = more discriminating feature for that disease.

This script:
  1. Runs Algorithms 1-3 on the male-only and female-only sub-populations
     (using the forced-sex forest from Algorithm1_forcedBranch.py)
  2. Extracts the refined action weights from Algorithm 3 for each gender
  3. Computes per-feature importance as the max r_{o|h} across disease classes
     (the "best disease discriminator" score for each feature)
  4. Identifies features with significant importance differences between genders
  5. Computes normal ranges by gender (Figure 7 from the paper)

Outputs
-------
  feature_importance_by_gender.csv   -- full importance table (all features)
  significant_gender_differences.csv -- features with large gender gap
  normal_ranges_male.csv             -- Figure 7 values for healthy males
  normal_ranges_female.csv           -- Figure 7 values for healthy females
  normal_ranges_combined.csv         -- side-by-side comparison

Usage
-----
  python featureImportanceByGender.py
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# ── Locate the repo root ────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from CDS_Paper_Algorithms import (
    load_dataset, build_decision_tree, FEATURE_NAMES, HEALTHY_CLASS,
    DIAGNOSTIC_THRESHOLD, N_FEATURES,
)
from Algorithm2 import run_algorithm2, DEFAULT_N_BINS
from Algorithm3 import run_algorithm3

# Suppress verbose CDS logging
for _name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
_DATA_CANDIDATES = [
    os.path.join(_REPO_ROOT, "arrhythmia.data"),
    os.path.join(_HERE, "arrhythmia.data"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("arrhythmia.data not found. Check _DATA_CANDIDATES.")

SEX_COL        = 1
MALE_CODE      = 0
FEMALE_CODE    = 1
IMPORTANCE_THRESHOLD = 0.05  # flag features with |male - female| > threshold
OUTPUT_DIR     = _HERE


def _feature_name(idx: int) -> str:
    return FEATURE_NAMES.get(idx, f"feat_{idx}")


def _stats(vals: np.ndarray) -> dict:
    if len(vals) == 0:
        return dict(b_min=np.nan, b_max=np.nan,
                    mean=np.nan, std=np.nan,
                    n_valid=0, range_width=np.nan)
    return dict(
        b_min        = float(vals.min()),
        b_max        = float(vals.max()),
        mean         = float(vals.mean()),
        std          = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        n_valid      = int(len(vals)),
        range_width  = float(vals.max() - vals.min()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("FEATURE IMPORTANCE BY GENDER  —  Algorithm 3 Action Weights")
print("=" * 80)

data, labels = load_dataset(DATA_PATH)
N, D = data.shape
print(f"  Loaded {N} users, {D} features")

sex_col     = data[:, SEX_COL]
valid_sex   = ~np.isnan(sex_col)
male_mask   = (sex_col == MALE_CODE) & valid_sex
female_mask = (sex_col == FEMALE_CODE) & valid_sex

male_indices   = np.where(male_mask)[0]
female_indices = np.where(female_mask)[0]

print(f"  Males:   {len(male_indices)}")
print(f"  Females: {len(female_indices)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RUN ALGORITHMS 1-3 ON GENDER-SPECIFIC SUB-POPULATIONS
# ─────────────────────────────────────────────────────────────────────────────
# We build a full CDS tree on each gender's sub-population separately,
# then extract Algorithm 3's refined action weights as the importance metric.
# This uses the CDS system's own mechanism (not an external ML model).

print("\n" + "=" * 80)
print("STEP 1 — RUNNING CDS PIPELINE (Alg 1-3) PER GENDER")
print("=" * 80)


def run_cds_pipeline_for_subset(subset_indices, subset_label):
    """Run Algorithms 1-3 on a subset of users and return refined actions."""
    # Build data/labels for subset
    sub_data   = data[subset_indices]
    sub_labels = labels[subset_indices]

    print(f"\n  [{subset_label}] Building decision tree ({len(subset_indices)} users)...")
    tree = build_decision_tree(sub_data, sub_labels)
    print(f"  [{subset_label}] Tree depth={tree.depth()}, nodes={tree.count_nodes()}")

    print(f"  [{subset_label}] Running Algorithm 2 (perceptor + executive training)...")
    alg2_output = run_algorithm2(tree=tree, data=sub_data, labels=sub_labels,
                                 n_bins=DEFAULT_N_BINS)
    print(f"  [{subset_label}] Alg2: {alg2_output.n_perceptor_entries} perceptor, "
          f"{alg2_output.n_executive_entries} executive entries")

    print(f"  [{subset_label}] Running Algorithm 3 (actions refining)...")
    alg3_output = run_algorithm3(alg2_output=alg2_output, tree=tree,
                                 data=sub_data, labels=sub_labels,
                                 reset_per_h=False)
    print(f"  [{subset_label}] Alg3: {len(alg3_output.refined_actions)} retained, "
          f"{len(alg3_output.removed_actions)} removed")

    return alg3_output


alg3_male   = run_cds_pipeline_for_subset(male_indices, "MALE")
alg3_female = run_cds_pipeline_for_subset(female_indices, "FEMALE")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXTRACT PER-FEATURE IMPORTANCE FROM ALGORITHM 3
# ─────────────────────────────────────────────────────────────────────────────
# For each feature, its "importance" in the CDS system is the maximum
# action weight r_{o|h} across all disease classes for which it was retained.
# This measures the feature's best-case discriminating power:
#   r_{o|h} = P(B_hat outside healthy range | disease h)
# A feature with r=0 for all diseases (or removed by Algorithm 3) has zero
# importance — it doesn't help distinguish any disease from healthy.

print("\n" + "=" * 80)
print("STEP 2 — EXTRACTING IMPORTANCE FROM ALGORITHM 3 ACTION WEIGHTS")
print("=" * 80)


def extract_importance(alg3_output, label):
    """
    Extract per-feature importance from Algorithm 3's refined actions.

    Returns dict: feature_idx -> {max_weight, mean_weight, n_diseases_retained, retained_diseases}
    """
    importance = {}
    for action in alg3_output.refined_actions:
        feat = action.feature_idx
        if feat not in importance:
            importance[feat] = {
                'max_weight': 0.0,
                'sum_weight': 0.0,
                'n_diseases': 0,
                'diseases': [],
            }
        entry = importance[feat]
        entry['max_weight'] = max(entry['max_weight'], action.action_weight)
        entry['sum_weight'] += action.action_weight
        entry['n_diseases'] += 1
        entry['diseases'].append(action.disease_class)
    return importance


male_importance   = extract_importance(alg3_male, "male")
female_importance = extract_importance(alg3_female, "female")

print(f"  Male:   {len(male_importance)} features retained by Algorithm 3")
print(f"  Female: {len(female_importance)} features retained by Algorithm 3")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
# For each feature (excluding Sex), compare the Algorithm 3 importance
# between genders.

print("\n" + "=" * 80)
print("STEP 3 — BUILDING GENDER COMPARISON TABLE")
print("=" * 80)

all_features = [i for i in range(N_FEATURES) if i != SEX_COL]

importance_rows = []
for feat_idx in all_features:
    fname = _feature_name(feat_idx)

    m_info = male_importance.get(feat_idx, {'max_weight': 0.0, 'sum_weight': 0.0, 'n_diseases': 0, 'diseases': []})
    f_info = female_importance.get(feat_idx, {'max_weight': 0.0, 'sum_weight': 0.0, 'n_diseases': 0, 'diseases': []})

    m_max = m_info['max_weight']
    f_max = f_info['max_weight']
    m_mean = m_info['sum_weight'] / m_info['n_diseases'] if m_info['n_diseases'] > 0 else 0.0
    f_mean = f_info['sum_weight'] / f_info['n_diseases'] if f_info['n_diseases'] > 0 else 0.0

    abs_diff_max  = abs(m_max - f_max)
    abs_diff_mean = abs(m_mean - f_mean)

    importance_rows.append({
        'feature_idx':          feat_idx,
        'feature_name':         fname,
        'male_max_r':           m_max,
        'male_mean_r':          m_mean,
        'male_n_diseases':      m_info['n_diseases'],
        'female_max_r':         f_max,
        'female_mean_r':        f_mean,
        'female_n_diseases':    f_info['n_diseases'],
        'abs_diff_max_r':       abs_diff_max,
        'abs_diff_mean_r':      abs_diff_mean,
        'male_only':            m_max > 0 and f_max == 0,
        'female_only':          f_max > 0 and m_max == 0,
        'significant':          abs_diff_max > IMPORTANCE_THRESHOLD,
    })

importance_df = pd.DataFrame(importance_rows).sort_values('abs_diff_max_r', ascending=False)
sig_df        = importance_df[importance_df['significant']].copy()

n_sig = len(sig_df)
n_male_only  = importance_df['male_only'].sum()
n_female_only = importance_df['female_only'].sum()

print(f"\n  Features with |male_max_r - female_max_r| > {IMPORTANCE_THRESHOLD}: {n_sig}")
print(f"  Features important ONLY in males:   {n_male_only}")
print(f"  Features important ONLY in females: {n_female_only}")

print("\n  Top 25 features by gender importance difference (Algorithm 3 max r_{o|h}):")
print(
    importance_df[
        ['feature_name', 'male_max_r', 'female_max_r',
         'male_n_diseases', 'female_n_diseases', 'abs_diff_max_r', 'significant']
    ].head(25).to_string(index=False)
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — NORMAL RANGES BY GENDER (Figure 7 numerical values)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 4 — NORMAL RANGES BY GENDER (Figure 7)")
print("=" * 80)

healthy_male_mask   = (labels == HEALTHY_CLASS) & male_mask
healthy_female_mask = (labels == HEALTHY_CLASS) & female_mask

print(f"  Healthy males:   {healthy_male_mask.sum()}")
print(f"  Healthy females: {healthy_female_mask.sum()}")

male_range_rows   = []
female_range_rows = []
combined_rows     = []

for feat_idx in range(D):
    if feat_idx == SEX_COL:
        continue

    fname = _feature_name(feat_idx)
    col   = data[:, feat_idx]

    m_vals = col[healthy_male_mask]
    m_vals = m_vals[~np.isnan(m_vals)]

    f_vals = col[healthy_female_mask]
    f_vals = f_vals[~np.isnan(f_vals)]

    ms = _stats(m_vals)
    fs = _stats(f_vals)

    male_range_rows.append({'feature_idx': feat_idx, 'feature_name': fname, **ms})
    female_range_rows.append({'feature_idx': feat_idx, 'feature_name': fname, **fs})

    combined_rows.append({
        'feature_idx':          feat_idx,
        'feature_name':         fname,
        'male_b_min':           ms['b_min'],
        'male_b_max':           ms['b_max'],
        'male_mean':            ms['mean'],
        'male_std':             ms['std'],
        'male_n_valid':         ms['n_valid'],
        'male_range_width':     ms['range_width'],
        'female_b_min':         fs['b_min'],
        'female_b_max':         fs['b_max'],
        'female_mean':          fs['mean'],
        'female_std':           fs['std'],
        'female_n_valid':       fs['n_valid'],
        'female_range_width':   fs['range_width'],
        'ranges_overlap':       not (
            np.isnan(ms['b_max']) or np.isnan(fs['b_min']) or
            ms['b_max'] < fs['b_min'] or fs['b_max'] < ms['b_min']
        ),
    })

male_range_df   = pd.DataFrame(male_range_rows)
female_range_df = pd.DataFrame(female_range_rows)
combined_df     = pd.DataFrame(combined_rows)

non_overlap = combined_df[~combined_df['ranges_overlap']].copy()
print(f"\n  Features where healthy male/female ranges do NOT overlap: "
      f"{len(non_overlap)}")
if not non_overlap.empty:
    print(
        non_overlap[['feature_name', 'male_b_min', 'male_b_max',
                     'female_b_min', 'female_b_max']].to_string(index=False)
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)


def _save(df: pd.DataFrame, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"  Saved: {path}  ({len(df)} rows)")


_save(importance_df,  "feature_importance_by_gender.csv")
_save(sig_df,         "significant_gender_differences.csv")
_save(male_range_df,  "normal_ranges_male.csv")
_save(female_range_df,"normal_ranges_female.csv")
_save(combined_df,    "normal_ranges_combined.csv")


# ─────────────────────────────────────────────────────────────────────────────
# INTERPRETATION GUIDE
# ─────────────────────────────────────────────────────────────────────────────
print("""
INTERPRETATION GUIDE
====================

feature_importance_by_gender.csv
  male_max_r / female_max_r
    Maximum Algorithm 3 action weight r_{o|h} across all disease classes
    for which this feature was RETAINED after the greedy set-cover pruning.
    Higher = more discriminating for disease diagnosis in that gender.
    r_{o|h} = P(B_hat < b_min | h) + P(B_hat > b_max | h)
  male_mean_r / female_mean_r
    Mean action weight across retained disease classes.
  male_n_diseases / female_n_diseases
    Number of disease classes for which this feature was retained.
    A feature retained for many diseases is a broad-spectrum diagnostic.
  abs_diff_max_r
    |male_max_r - female_max_r|. Sorted descending. Features at the top
    have the largest gender gap in CDS diagnostic importance.
  male_only / female_only
    True if the feature was retained by Algorithm 3 for one gender but
    not the other. These features represent gender-specific diagnostic
    pathways that the unified CDS model may miss.
  significant
    True when abs_diff_max_r > 0.05 (heuristic threshold).

normal_ranges_*.csv
  b_min, b_max
    Paper Eq. 5: exact min and max of the feature for healthy users of
    that gender (NaN excluded). These are the CDS alarm boundaries.
  ranges_overlap = False
    Healthy male/female ranges are disjoint — a value normal for one
    gender can trigger a false alarm for the other under a unified model.
""")
