import pandas as pd
import numpy as np
from math import isnan
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "arrhythmia.data"
N_IMPUTATIONS = 5
RANDOM_STATE = 42

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
# Column 1  -> Sex (0 = female, 1 = male)
# Last col  -> Class label

SEX_COLUMN = 1
CLASS_COLUMN = df.columns[-1]

# Create readable gender labels
# Dataset convention:
# 0 = female
# 1 = male

df['gender'] = df[SEX_COLUMN].map({0: 'Female', 1: 'Male'})

print("Gender distribution:")
print(df['gender'].value_counts(dropna=False))
print()

# ============================================================
# 1. ANALYZE MISSING DATA PATTERNS BY GENDER
# ============================================================
print("=" * 80)
print("MISSING DATA ANALYSIS BY GENDER")
print("=" * 80)

missing_summary = []

for col in df.columns[:-1]:  # exclude added gender column
    if col == 'gender':
        continue

    total_missing = df[col].isna().sum()

    if total_missing == 0:
        continue

    female_mask = df['gender'] == 'Female'
    male_mask = df['gender'] == 'Male'

    female_missing = df.loc[female_mask, col].isna().sum()
    male_missing = df.loc[male_mask, col].isna().sum()

    female_total = female_mask.sum()
    male_total = male_mask.sum()

    female_missing_pct = female_missing / female_total if female_total > 0 else np.nan
    male_missing_pct = male_missing / male_total if male_total > 0 else np.nan

    missing_summary.append({
        'feature': col,
        'total_missing': total_missing,
        'female_missing': female_missing,
        'male_missing': male_missing,
        'female_missing_pct': female_missing_pct,
        'male_missing_pct': male_missing_pct
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
# Lightweight implementation using only pandas/numpy.
# We compare missingness percentages between genders.
# ============================================================

print("=" * 80)
print("INFORMATIVE MISSINGNESS TESTING")
print("=" * 80)

informative_results = []

for col in df.columns[:-1]:
    if col == 'gender':
        continue

    missing_indicator = df[col].isna()

    if missing_indicator.sum() == 0:
        continue

    female_mask = df['gender'] == 'Female'
    male_mask = df['gender'] == 'Male'

    female_missing_rate = missing_indicator[female_mask].mean()
    male_missing_rate = missing_indicator[male_mask].mean()

    difference = male_missing_rate - female_missing_rate

    # Heuristic threshold for informative missingness
    informative = abs(difference) > 0.05

    informative_results.append({
        'feature': col,
        'female_missing_rate': female_missing_rate,
        'male_missing_rate': male_missing_rate,
        'difference': difference,
        'informative_missingness': informative
    })

informative_df = pd.DataFrame(informative_results)

if not informative_df.empty:
    informative_df = informative_df.sort_values(
        'difference',
        key=np.abs,
        ascending=False
    )

    print("Features with strongest gender-based missingness differences:")
    print(informative_df.head(20).to_string(index=False))

    significant_features = informative_df[
        informative_df['informative_missingness'] == True
    ]

    print()
    print(f"Potentially informative missingness features: {len(significant_features)}")

else:
    print("No informative missingness detected.")

print()

# ============================================================
# 3. MULTIPLE IMPUTATION (RUBIN, 1987)

# ============================================================
# Lightweight implementation using random sampling imputation.
# This avoids external ML/statistical libraries.
#
# Rubin framework:
#   - Generate multiple completed datasets
#   - Add randomness during imputation
#   - Compare variability across imputations
# ============================================================

print("=" * 80)
print("MULTIPLE IMPUTATION ANALYSIS")
print("=" * 80)

numeric_df = df.drop(columns=['gender']).copy()

imputed_datasets = []

for i in range(N_IMPUTATIONS):
    print(f"Running imputation {i+1}/{N_IMPUTATIONS}...")

    imputed_df = numeric_df.copy()

    np.random.seed(RANDOM_STATE + i)

    for col in imputed_df.columns:

        missing_mask = imputed_df[col].isna()

        if missing_mask.sum() == 0:
            continue

        observed_values = imputed_df.loc[~missing_mask, col]

        if len(observed_values) == 0:
            continue

        mean = observed_values.mean()
        std = observed_values.std()

        if pd.isna(std) or std == 0:
            std = 1e-6

        random_values = np.random.normal(
            loc=mean,
            scale=std,
            size=missing_mask.sum()
        )

        imputed_df.loc[missing_mask, col] = random_values

    imputed_datasets.append(imputed_df)

print()
print("Multiple imputations complete.")
print()

# ============================================================
# 4. ASSESS IMPACT OF IMPUTATION
# ============================================================
# Compare original observed distributions with imputed distributions
# ============================================================

print("=" * 80)
print("IMPUTATION IMPACT ASSESSMENT")
print("=" * 80)

impact_results = []

for col in numeric_df.columns:

    missing_mask = numeric_df[col].isna()

    if missing_mask.sum() == 0:
        continue

    observed_values = numeric_df.loc[~missing_mask, col]

    imputed_means = []
    imputed_variances = []

    for imputed_df in imputed_datasets:
        imputed_values = imputed_df.loc[missing_mask, col]

        imputed_means.append(imputed_values.mean())
        imputed_variances.append(imputed_values.var())

    pooled_mean = np.mean(imputed_means)
    within_variance = np.mean(imputed_variances)
    between_variance = np.var(imputed_means, ddof=1)

    # Rubin variance combination rule
    total_variance = within_variance + (1 + 1 / N_IMPUTATIONS) * between_variance

    impact_results.append({
        'feature': col,
        'missing_count': missing_mask.sum(),
        'observed_mean': observed_values.mean(),
        'pooled_imputed_mean': pooled_mean,
        'mean_difference': pooled_mean - observed_values.mean(),
        'within_variance': within_variance,
        'between_variance': between_variance,
        'total_variance': total_variance
    })

impact_df = pd.DataFrame(impact_results)

if not impact_df.empty:
    impact_df = impact_df.sort_values(
        'mean_difference',
        key=np.abs,
        ascending=False
    )

    print("Features most affected by imputation:")
    print(impact_df.head(20).to_string(index=False))

print()

# ============================================================
# 5. OPTIONAL: SAVE OUTPUTS
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
print(
    """
1. MISSINGNESS BY GENDER
   - Compare female_missing_pct vs male_missing_pct
   - Large differences suggest demographic imbalance

2. INFORMATIVE MISSINGNESS
   - Compare female_missing_rate vs male_missing_rate
   - Large differences suggest demographic bias
   - difference > 0:
       Missingness more common in males
   - difference < 0:
       Missingness more common in females

3. MULTIPLE IMPUTATION IMPACT
   - Large mean_difference:
       Imputation substantially changes the feature distribution
   - Large between_variance:
       Imputation uncertainty is high
   - Large total_variance:
       Missing data strongly affects inference

4. RUBIN (1987) FRAMEWORK
   - Multiple completed datasets are generated
   - Parameter uncertainty propagates through analysis
   - Between-imputation variance quantifies uncertainty from missing data
"""
)
