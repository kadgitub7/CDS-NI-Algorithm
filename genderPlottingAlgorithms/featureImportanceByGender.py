"""
featureImportanceByGender.py

Three analyses, each independent of the LOOCV loop:

  1. Per-gender permutation importance (Breiman, 2001)
     Train a RandomForest separately on the male and female sub-populations.
     Use permutation importance with n_repeats=30 to obtain a *distribution*
     of importance scores per feature, not just a point estimate.

  2. Significance of gender differences
     For each feature, the 30 male-model importance values and the 30
     female-model importance values are compared with a two-sample Welch
     t-test (unequal variances assumed; appropriate because the models were
     trained on populations of different sizes).  Raw p-values are
     corrected for multiple testing with the Benjamini-Hochberg FDR
     procedure (Benjamini & Hochberg, 1995).

  3. Normal ranges by gender — numerical Figure 7 from the paper
     For every feature, compute b_min/b_max (paper Eq. 5: the exact raw
     min/max of healthy users' values) separately for healthy males and
     healthy females.  Outputs two CSV files that reproduce the numerical
     content of Figure 7 split by gender.

Outputs
-------
  feature_importance_by_gender.csv   — full importance table (all features)
  significant_gender_differences.csv — only features with FDR q < 0.05
  normal_ranges_male.csv             — Figure 7 values for healthy males
  normal_ranges_female.csv           — Figure 7 values for healthy females
  normal_ranges_combined.csv         — side-by-side comparison (all features)

Usage
-----
  python featureImportanceByGender.py
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# ── Locate the worktree that contains CDS_Paper_Algorithms ────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_WORKTREE   = os.path.normpath(os.path.join(
    _HERE, "..", ".claude", "worktrees", "nifty-hoover-f39cac"
))
if _WORKTREE not in sys.path:
    sys.path.insert(0, _WORKTREE)

from CDS_Paper_Algorithms import load_dataset, FEATURE_NAMES  # type: ignore

from sklearn.ensemble        import RandomForestClassifier
from sklearn.inspection      import permutation_importance
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH      = r"C:\Users\kadhi\OneDrive\Desktop\CDS_Algorithms\arrhythmia.data"
HEALTHY_CLASS  = 1
SEX_COL        = 1      # column index of the Sex feature
# Gender encoding observed in Algorithm2.py (lines 1611-1612):
#   female_mask = (gender_values == 1)
#   male_mask   = (gender_values == 0)
MALE_CODE      = 0
FEMALE_CODE    = 1

N_ESTIMATORS   = 200    # trees per RandomForest
N_REPEATS      = 30     # permutation repeats — gives distribution over importance
RANDOM_STATE   = 42
TEST_SIZE      = 0.25   # fraction held out for permutation-importance evaluation
FDR_ALPHA      = 0.05   # Benjamini-Hochberg significance level

OUTPUT_DIR     = _HERE  # write CSVs alongside this script

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _feature_name(idx: int) -> str:
    return FEATURE_NAMES.get(idx, f"feat_{idx}")


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg (1995) FDR correction.

    Returns q-values (adjusted p-values) in the same order as the input.
    The procedure controls the expected proportion of false discoveries among
    rejected hypotheses at level FDR_ALPHA.
    """
    m     = len(p_values)
    order = np.argsort(p_values)           # indices that sort p ascending

    # Raw adjusted values: q_(k) = p_(k) * m/k  for rank k = 1..m
    q_sorted = p_values[order] * m / np.arange(1, m + 1)

    # Enforce monotone non-decreasing from right to left (BH step 3).
    # q_(k) = min(q_(k), q_(k+1), ..., q_(m))
    # np.minimum.accumulate on a copy of the reversed array, then reverse back.
    # NOTE: q[order][::-1] uses fancy indexing and returns a copy; writing to
    # `out=` on such an expression silently discards the result.  We therefore
    # work on an explicit intermediate ndarray.
    q_rev = q_sorted[::-1].copy()                 # reversed copy  (writable)
    np.minimum.accumulate(q_rev, out=q_rev)        # in-place on the copy
    q_sorted = q_rev[::-1]                         # un-reverse

    # Map back to the original feature order and clamp to [0, 1]
    q = np.empty(m)
    q[order] = q_sorted
    return np.clip(q, 0.0, 1.0)


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


def _impute_median(X: np.ndarray) -> np.ndarray:
    """
    Column-wise median imputation.
    This is a lightweight preprocessing step — the imputation is done
    only to allow the RandomForest to train without NaN errors.  The
    healthy-range computation in Part 3 always uses the original raw
    values (NaNs excluded explicitly).
    """
    out = X.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any() and (~nan_mask).any():
            out[nan_mask, j] = np.nanmedian(col)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("FEATURE IMPORTANCE BY GENDER  —  CDS Arrhythmia Dataset")
print("=" * 80)

data, labels = load_dataset(DATA_PATH)
N, D = data.shape
print(f"  Loaded {N} users, {D} features")
print(f"  Healthy (class {HEALTHY_CLASS}): {(labels == HEALTHY_CLASS).sum()}")
print(f"  Disease (class > 1):             {(labels != HEALTHY_CLASS).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — GENDER SPLIT
# ─────────────────────────────────────────────────────────────────────────────
sex_col          = data[:, SEX_COL]
valid_sex        = ~np.isnan(sex_col)
male_mask        = (sex_col == MALE_CODE)   & valid_sex
female_mask      = (sex_col == FEMALE_CODE) & valid_sex

print(f"\n  Males   (code {MALE_CODE}): {male_mask.sum()}")
print(f"  Females (code {FEMALE_CODE}): {female_mask.sum()}")

if male_mask.sum() == 0 or female_mask.sum() == 0:
    raise RuntimeError("One gender group is empty — check MALE_CODE / FEMALE_CODE.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD FEATURE MATRIX  (exclude Sex column)
# ─────────────────────────────────────────────────────────────────────────────
# Exclude the Sex column itself; it is the grouping variable, not a predictor.
feature_indices = [i for i in range(D) if i != SEX_COL]   # 278 features
feature_names_list = [_feature_name(i) for i in feature_indices]

X_all = data[:, feature_indices]          # (N, 278)  — may contain NaN
X_imp = _impute_median(X_all)             # (N, 278)  — NaN-free, for RF

# Binary target: 0 = healthy, 1 = any disease
y = (labels != HEALTHY_CLASS).astype(int)

X_male   = X_imp[male_mask]
y_male   = y[male_mask]
X_female = X_imp[female_mask]
y_female = y[female_mask]

# ── Train / test split per gender ────────────────────────────────────────────
# Permutation importance MUST be evaluated on held-out data.  If evaluated
# in-sample on a large RandomForest (200 trees, 278 features), the model
# overfits and permuting any single feature leaves AUC ≈ 1.0 — every
# importance collapses to machine-epsilon noise (~1e-17).  The held-out
# split ensures the measured AUC drop reflects genuine generalisation.
X_male_tr, X_male_te, y_male_tr, y_male_te = train_test_split(
    X_male, y_male,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y_male,
)
X_female_tr, X_female_te, y_female_tr, y_female_te = train_test_split(
    X_female, y_female,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y_female,
)
print(f"  Male   train={len(y_male_tr)}  test={len(y_male_te)}")
print(f"  Female train={len(y_female_tr)}  test={len(y_female_te)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN GENDER-SPECIFIC RANDOM FORESTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 1 / 2 — TRAINING GENDER-SPECIFIC RANDOM FORESTS")
print("=" * 80)

rf_male = RandomForestClassifier(
    n_estimators = N_ESTIMATORS,
    random_state = RANDOM_STATE,
    n_jobs       = -1,
    class_weight = "balanced",   # handles class imbalance within each group
)
rf_male.fit(X_male_tr, y_male_tr)
print(f"  Male   RF trained  (n={len(y_male_tr)}, disease_rate={y_male_tr.mean():.2%})")

rf_female = RandomForestClassifier(
    n_estimators = N_ESTIMATORS,
    random_state = RANDOM_STATE,
    n_jobs       = -1,
    class_weight = "balanced",
)
rf_female.fit(X_female_tr, y_female_tr)
print(f"  Female RF trained  (n={len(y_female_tr)}, disease_rate={y_female_tr.mean():.2%})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PERMUTATION IMPORTANCE  (Breiman, 2001)
# ─────────────────────────────────────────────────────────────────────────────
# permutation_importance shuffles one feature at a time and measures the
# drop in model accuracy.  n_repeats=30 gives a distribution (30 values) per
# feature, not just a point estimate.  This distribution is essential for
# the significance test in Step 5.
#
# We evaluate each model on its OWN gender's held-out test set.
# This measures "how much does this feature generalise for diagnosing disease
# in people of this gender".  In-sample evaluation on a large RF (200 trees,
# 278 features) produces near-zero importances because the overfit model
# compensates for any permuted feature using the remaining 277 correlated
# ones, keeping AUC ≈ 1.0 regardless of which feature is shuffled.
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Running permutation importance — male model  "
      f"(n_repeats={N_REPEATS}, this may take ~1 min)...")
perm_m = permutation_importance(
    rf_male, X_male_te, y_male_te,
    n_repeats    = N_REPEATS,
    random_state = RANDOM_STATE,
    n_jobs       = -1,
    scoring      = "roc_auc",   # AUC drop is a stable importance metric
)

print("  Running permutation importance — female model  "
      f"(n_repeats={N_REPEATS})...")
perm_f = permutation_importance(
    rf_female, X_female_te, y_female_te,
    n_repeats    = N_REPEATS,
    random_state = RANDOM_STATE,
    n_jobs       = -1,
    scoring      = "roc_auc",
)

# perm_m.importances shape: (n_features, n_repeats)
male_imp_mean   = perm_m.importances_mean    # (278,)
male_imp_std    = perm_m.importances_std
female_imp_mean = perm_f.importances_mean    # (278,)
female_imp_std  = perm_f.importances_std

print("  Permutation importance complete.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SIGNIFICANCE OF GENDER DIFFERENCES
# ─────────────────────────────────────────────────────────────────────────────
# For each feature j, test H0: importance_male == importance_female.
#
# We have 30 independent importance values for each group (one per repeat).
# A two-sample Welch t-test is appropriate:
#   - Two independent samples (different models, different data)
#   - Unequal sample sizes may occur if a repeat produced equal accuracy
#   - Welch does not assume equal variance across groups
#
# The test statistic is:
#   t = (mean_m - mean_f) / sqrt(var_m/30 + var_f/30)
#
# After testing all 278 features, apply Benjamini-Hochberg FDR correction.
# This controls the expected false discovery rate at 5%, which is more
# appropriate than Bonferroni for exploratory analysis with 278 tests.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 2 / 2 — SIGNIFICANCE TESTING  (Welch t-test + BH-FDR correction)")
print("=" * 80)

p_raw = np.empty(len(feature_indices))

for j in range(len(feature_indices)):
    male_dist   = perm_m.importances[j, :]    # 30 values
    female_dist = perm_f.importances[j, :]    # 30 values
    _, p_raw[j] = ttest_ind(male_dist, female_dist, equal_var=False, alternative='two-sided')

# ttest_ind returns NaN for features where both distributions are constant
# (std=0, all 30 values identical).  A single NaN in the array poisons
# np.minimum.accumulate and makes every q-value NaN.  Treat untestable
# features as p=1 for BH, then restore NaN afterwards.
nan_p_mask  = np.isnan(p_raw)
p_for_bh    = p_raw.copy()
p_for_bh[nan_p_mask] = 1.0
q_values = _benjamini_hochberg(p_for_bh)
q_values[nan_p_mask] = np.nan   # restore NaN — these features were untestable

abs_diff = np.abs(male_imp_mean - female_imp_mean)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD FULL IMPORTANCE TABLE
# ─────────────────────────────────────────────────────────────────────────────
importance_rows = []
for j, feat_idx in enumerate(feature_indices):
    importance_rows.append({
        'feature_idx':      feat_idx,
        'feature_name':     feature_names_list[j],
        'male_imp_mean':    float(male_imp_mean[j]),
        'male_imp_std':     float(male_imp_std[j]),
        'female_imp_mean':  float(female_imp_mean[j]),
        'female_imp_std':   float(female_imp_std[j]),
        'abs_difference':   float(abs_diff[j]),
        'p_raw':            float(p_raw[j]),
        'q_fdr':            float(q_values[j]),
        'significant':      bool(q_values[j] < FDR_ALPHA),
    })

importance_df = pd.DataFrame(importance_rows).sort_values('abs_difference', ascending=False)
sig_df        = importance_df[importance_df['significant']].copy()

n_sig = len(sig_df)
print(f"\n  Features with significant gender-importance difference "
      f"(FDR q < {FDR_ALPHA}): {n_sig} / {len(feature_indices)}")

print("\n  Top 20 by absolute importance difference:")
print(
    importance_df[
        ['feature_name', 'male_imp_mean', 'female_imp_mean',
         'abs_difference', 'p_raw', 'q_fdr', 'significant']
    ].head(20).to_string(index=False)
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — NORMAL RANGES BY GENDER  (Figure 7 numerical values)
# ─────────────────────────────────────────────────────────────────────────────
# Paper Eq. 5: b_min = min(values of feature o for healthy users)
#              b_max = max(values of feature o for healthy users)
#
# We reproduce this for EVERY feature, split into healthy males and
# healthy females.  The paper's Figure 7 plots these ranges; here we
# emit the raw numbers into CSV files so they can be inspected or
# re-plotted.
#
# Additional statistics (mean, std, n_valid, range_width) are included
# because they were part of the paper's healthy-range characterisation
# even if not all are shown in Figure 7.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 3 / 3 — NORMAL RANGES BY GENDER  (Figure 7 numerical output)")
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

    fname    = _feature_name(feat_idx)
    col      = data[:, feat_idx]

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
        # Overlap flag: do the male and female normal ranges overlap?
        # Non-overlap means the healthy range itself is gender-specific.
        'ranges_overlap':       not (
            np.isnan(ms['b_max']) or np.isnan(fs['b_min']) or
            ms['b_max'] < fs['b_min'] or fs['b_max'] < ms['b_min']
        ),
    })

male_range_df   = pd.DataFrame(male_range_rows)
female_range_df = pd.DataFrame(female_range_rows)
combined_df     = pd.DataFrame(combined_rows)

# Features where healthy ranges do NOT overlap (most gender-specific)
non_overlap = combined_df[~combined_df['ranges_overlap']].copy()
print(f"\n  Features where healthy male/female ranges do NOT overlap: "
      f"{len(non_overlap)}")
if not non_overlap.empty:
    print(
        non_overlap[['feature_name', 'male_b_min', 'male_b_max',
                     'female_b_min', 'female_b_max']].to_string(index=False)
    )

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SAVE ALL OUTPUTS
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
# STEP 8 — INTERPRETATION GUIDE
# ─────────────────────────────────────────────────────────────────────────────
print("""
INTERPRETATION GUIDE
====================

feature_importance_by_gender.csv
  male_imp_mean / female_imp_mean
    Mean AUC drop when that feature is permuted in the gender-specific model.
    Higher = more important for disease diagnosis in that gender.
  male_imp_std / female_imp_std
    Standard deviation across the 30 permutation repeats.
    High std relative to mean → unstable importance (noisy feature).
  abs_difference
    |male_imp_mean - female_imp_mean|.  Sorted descending in this file.
  p_raw
    Two-tailed Welch t-test p-value comparing the 30-repeat distributions.
  q_fdr
    Benjamini-Hochberg adjusted p-value (FDR ≤ 5%).
  significant
    True when q_fdr < 0.05.  Use this column, not p_raw, to avoid inflated
    false-discovery due to testing 278 features simultaneously.

significant_gender_differences.csv
  Subset of the above: only features where the gender difference in
  permutation importance is statistically significant after FDR correction.

normal_ranges_male.csv / normal_ranges_female.csv
  b_min, b_max
    Paper Eq. 5: exact min and max of the feature for healthy users of that
    gender (NaN excluded).  These are the values the CDS uses as alarm
    boundaries in Algorithm 4.
  range_width = b_max - b_min
    Narrow range → tightly regulated feature in healthy users of that gender.
  n_valid
    Number of healthy users with a non-missing value for this feature.

normal_ranges_combined.csv
  Side-by-side male and female ranges for all features.
  ranges_overlap = False when [male_b_min, male_b_max] and
    [female_b_min, female_b_max] are disjoint.  A non-overlapping range
    means a value that is normal for one gender can be alarming for the other
    — these features strongly motivate gender-specific models.
""")
