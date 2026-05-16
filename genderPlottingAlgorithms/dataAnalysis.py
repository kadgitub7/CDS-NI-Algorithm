import os
import pandas as pd
import numpy as np
from math import isnan
import warnings

from scipy.stats import chi2_contingency, fisher_exact

from sklearn.experimental import enable_iterative_imputer  # noqa: F401  must precede import below
from sklearn.impute import IterativeImputer

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DATA_PATH = next(
    (p for p in [
        os.path.join(_REPO_ROOT, "arrhythmia.data"),
        os.path.join(_HERE, "arrhythmia.data"),
    ] if os.path.exists(p)),
    None,
)
if DATA_PATH is None:
    raise FileNotFoundError("arrhythmia.data not found. Place it in the repo root.")
N_IMPUTATIONS = 5
RANDOM_STATE = 42
ALPHA = 0.05   # significance level for informative-missingness tests

# ============================================================
# LOAD DATA
# ============================================================
# UCI Arrhythmia dataset uses '?' for missing values

df = pd.read_csv(DATA_PATH, header=None, na_values='?')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"Total missing values: {df.isna().sum().sum()}")
print()

# ============================================================
# FEATURE DEFINITIONS
# ============================================================
# According to the UCI Arrhythmia dataset:
# Column 0  -> Age
# Column 1  -> Sex (0 = male, 1 = female)   ← correct per Algorithm2/4 convention
# Last col  -> Class label

SEX_COLUMN = 1
CLASS_COLUMN = df.columns[-1]

df['gender'] = df[SEX_COLUMN].map({0: 'Male', 1: 'Female'})

print("Gender distribution:")
print(df['gender'].value_counts(dropna=False))
print()

# ============================================================
# 1. ANALYZE MISSING DATA PATTERNS BY GENDER
# ============================================================
print("=" * 80)
print("MISSING DATA ANALYSIS BY GENDER")
print("=" * 80)

# Columns to skip: the class label and the derived gender column.
# We cannot use df.columns[:-1] here because after appending 'gender',
# df.columns[-1] is 'gender', NOT the class label — the class label would
# be silently included in the loop.  Use an explicit exclusion set instead.
_SKIP_COLS = {CLASS_COLUMN, 'gender', SEX_COLUMN}

missing_summary = []

for col in df.columns:
    if col in _SKIP_COLS:
        continue

    total_missing = df[col].isna().sum()

    if total_missing == 0:
        continue

    female_mask = df['gender'] == 'Female'
    male_mask   = df['gender'] == 'Male'

    female_missing = df.loc[female_mask, col].isna().sum()
    male_missing   = df.loc[male_mask,   col].isna().sum()

    female_total = int(female_mask.sum())
    male_total   = int(male_mask.sum())

    female_missing_pct = female_missing / female_total if female_total > 0 else np.nan
    male_missing_pct   = male_missing   / male_total   if male_total   > 0 else np.nan

    missing_summary.append({
        'feature':            col,
        'total_missing':      total_missing,
        'female_missing':     female_missing,
        'male_missing':       male_missing,
        'female_missing_pct': female_missing_pct,
        'male_missing_pct':   male_missing_pct,
    })

missing_df = pd.DataFrame(missing_summary)

if not missing_df.empty:
    missing_df = missing_df.sort_values('total_missing', ascending=False)
    print(missing_df.head(20).to_string(index=False))
else:
    print("No missing data detected.")

print()

# ============================================================
# 2. TEST WHETHER MISSINGNESS IS INFORMATIVE
# ============================================================
# For each feature with missing values, build a 2×2 contingency table:
#
#              missing   observed
#   Female  [  a          b  ]
#   Male    [  c          d  ]
#
# Decision rule (Fisher, 1922 / Cochran, 1954):
#   - If ALL expected cell counts >= 5  → chi-square test (asymptotic)
#   - Otherwise                         → Fisher's exact test (exact)
#
# A feature is "informative" (MAR not MCAR) when the pattern of
# missingness differs significantly between genders (p < ALPHA).
# ============================================================

print("=" * 80)
print("INFORMATIVE MISSINGNESS TESTING  (chi-square / Fisher's exact)")
print("=" * 80)

informative_results = []

for col in df.columns:
    if col in _SKIP_COLS:
        continue

    missing_indicator = df[col].isna()

    if missing_indicator.sum() == 0:
        continue

    female_mask = df['gender'] == 'Female'
    male_mask   = df['gender'] == 'Male'

    # Cell counts for the 2×2 table
    a = int(missing_indicator[female_mask].sum())       # female missing
    b = int((~missing_indicator)[female_mask].sum())    # female observed
    c = int(missing_indicator[male_mask].sum())         # male missing
    d = int((~missing_indicator)[male_mask].sum())      # male observed

    female_missing_rate = a / (a + b) if (a + b) > 0 else np.nan
    male_missing_rate   = c / (c + d) if (c + d) > 0 else np.nan
    difference          = male_missing_rate - female_missing_rate

    contingency = np.array([[a, b], [c, d]])

    # Cochran's rule: use chi-square when all expected counts >= 5
    row_totals = contingency.sum(axis=1, keepdims=True)   # shape (2,1)
    col_totals = contingency.sum(axis=0, keepdims=True)   # shape (1,2)
    n_total    = contingency.sum()

    if n_total == 0:
        continue

    expected = row_totals * col_totals / n_total  # expected cell counts

    if (expected >= 5).all():
        # Chi-square test — asymptotically valid for large expected counts
        chi2, p_value, dof, _ = chi2_contingency(contingency, correction=False)
        test_used = 'chi2'
    else:
        # Fisher's exact test — exact p-value, valid for any cell size
        _, p_value = fisher_exact(contingency, alternative='two-sided')
        chi2       = np.nan
        test_used  = 'fisher'

    informative = p_value < ALPHA

    informative_results.append({
        'feature':              col,
        'female_missing':       a,
        'female_observed':      b,
        'male_missing':         c,
        'male_observed':        d,
        'female_missing_rate':  female_missing_rate,
        'male_missing_rate':    male_missing_rate,
        'difference':           difference,
        'test':                 test_used,
        'chi2_stat':            chi2,
        'p_value':              p_value,
        'significant':          informative,
    })

informative_df = pd.DataFrame(informative_results)

if not informative_df.empty:
    informative_df = informative_df.sort_values('p_value', ascending=True)

    print("Features ranked by p-value (smallest = strongest evidence of MAR):")
    print(
        informative_df[
            ['feature', 'female_missing_rate', 'male_missing_rate',
             'difference', 'test', 'p_value', 'significant']
        ].head(20).to_string(index=False)
    )

    n_sig = informative_df['significant'].sum()
    print()
    print(f"Features with significant gender-missingness association "
          f"(p < {ALPHA}): {n_sig}")
else:
    print("No informative missingness detected.")

print()

# ============================================================
# 3. MULTIPLE IMPUTATION — MICE (Rubin, 1987)
# ============================================================
# IterativeImputer implements MICE (Multivariate Imputation by
# Chained Equations), also known as "fully conditional specification".
#
# For each feature with missing values, a regression model is fitted
# using all OTHER features as predictors.  Missing values are filled
# with draws from the posterior predictive distribution of that model.
# This is repeated in rounds until convergence, producing one complete
# dataset per run.
#
# We run IterativeImputer N_IMPUTATIONS times with different seeds so
# each completed dataset reflects a different draw from the posterior.
# The imputed datasets then feed into Rubin's combination rules in
# Section 4.
#
# Reference: van Buuren & Groothuis-Oudshoorn (2011). MICE:
#            Multivariate Imputation by Chained Equations in R.
#            Journal of Statistical Software 45(3).
# ============================================================

print("=" * 80)
print("MULTIPLE IMPUTATION ANALYSIS  (MICE via IterativeImputer)")
print("=" * 80)

# Drop the added 'gender' column for imputation; keep class label so that
# the imputer can exploit class-feature correlations.
numeric_df = df.drop(columns=['gender']).copy()

imputed_datasets = []

for i in range(N_IMPUTATIONS):
    print(f"Running MICE imputation {i + 1}/{N_IMPUTATIONS}  "
          f"(seed={RANDOM_STATE + i})...")

    imputer = IterativeImputer(
        max_iter      = 10,          # rounds of chained-equation cycling
        random_state  = RANDOM_STATE + i,
        initial_strategy = 'mean',   # initialise missing cells with column mean
        imputation_order = 'roman',  # left-to-right column order per round
    )

    imputed_array = imputer.fit_transform(numeric_df.values)
    imputed_df    = pd.DataFrame(
        imputed_array, columns=numeric_df.columns, index=numeric_df.index
    )
    imputed_datasets.append(imputed_df)

print()
print("MICE imputations complete.")
print()

# ============================================================
# 4. ASSESS IMPACT OF IMPUTATION (RUBIN, 1987)
# ============================================================
# For each originally-missing cell, we track how much the imputed mean
# varies across the M completed datasets.
#
# Rubin's combination rules:
#   Q̄  = (1/M) Σ Q̂_i                     pooled point estimate
#   Ū  = (1/M) Σ Û_i                       within-imputation variance
#   B  = 1/(M-1) Σ (Q̂_i − Q̄)²             between-imputation variance
#   T  = Ū + (1 + 1/M) B                   total variance (Rubin 1987, eq 3.1)
#
# A large B relative to Ū signals that the missing-data mechanism
# introduces substantial uncertainty — the imputed values vary
# widely across runs because the model cannot pin down the true value.
# ============================================================

print("=" * 80)
print("IMPUTATION IMPACT ASSESSMENT  (Rubin combination rules)")
print("=" * 80)

impact_results = []

for col in numeric_df.columns:

    missing_mask = numeric_df[col].isna()

    if missing_mask.sum() == 0:
        continue

    observed_values = numeric_df.loc[~missing_mask, col]

    imputed_means     = []
    imputed_variances = []

    for imputed_df in imputed_datasets:
        imputed_values = imputed_df.loc[missing_mask, col]
        imputed_means.append(imputed_values.mean())
        imputed_variances.append(imputed_values.var(ddof=1) if len(imputed_values) > 1 else 0.0)

    pooled_mean      = float(np.mean(imputed_means))
    within_variance  = float(np.mean(imputed_variances))               # Ū
    between_variance = float(np.var(imputed_means, ddof=1))            # B
    total_variance   = within_variance + (1 + 1 / N_IMPUTATIONS) * between_variance  # T

    impact_results.append({
        'feature':             col,
        'missing_count':       int(missing_mask.sum()),
        'observed_mean':       float(observed_values.mean()),
        'pooled_imputed_mean': pooled_mean,
        'mean_difference':     pooled_mean - float(observed_values.mean()),
        'within_variance':     within_variance,
        'between_variance':    between_variance,
        'total_variance':      total_variance,
    })

impact_df = pd.DataFrame(impact_results)

if not impact_df.empty:
    impact_df = impact_df.sort_values(
        'mean_difference', key=np.abs, ascending=False
    )
    print("Features most affected by imputation:")
    print(impact_df.head(20).to_string(index=False))

print()

# ============================================================
# 5. SAVE OUTPUTS
# ============================================================

missing_df.to_csv('missingness_by_gender.csv', index=False)
informative_df.to_csv('informative_missingness_tests.csv', index=False)
impact_df.to_csv('multiple_imputation_impact.csv', index=False)

print("=" * 80)
print("OUTPUT FILES GENERATED")
print("=" * 80)
print("1. missingness_by_gender.csv")
print("2. informative_missingness_tests.csv")
print("3. multiple_imputation_impact.csv")
print()

# ============================================================
# 6. INTERPRETATION GUIDE
# ============================================================

print("=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print("""
1. MISSINGNESS BY GENDER
   - Compare female_missing_pct vs male_missing_pct per feature
   - Large differences indicate unequal data collection by gender

2. INFORMATIVE MISSINGNESS  (chi-square / Fisher's exact)
   - p < 0.05: evidence that missingness is NOT random w.r.t. gender (MAR)
   - 'chi2' used when all expected cell counts >= 5 (Cochran's rule)
   - 'fisher' used when any expected cell count < 5 (exact test)
   - difference > 0: missingness more common in males
   - difference < 0: missingness more common in females

3. MULTIPLE IMPUTATION (MICE)
   - IterativeImputer fits a regression model per feature using all
     other features as predictors, preserving inter-feature correlations
   - 5 independent completed datasets created (different random seeds)

4. RUBIN (1987) COMBINATION RULES
   - within_variance  (Ū): mean imputation variance across runs
   - between_variance (B): variance of imputed means across runs
   - total_variance   (T = Ū + (1+1/M)B): full uncertainty in imputed value
   - Large B/T ratio: missing data mechanism dominates uncertainty
   - Large mean_difference: imputed distribution differs from observed
""")
