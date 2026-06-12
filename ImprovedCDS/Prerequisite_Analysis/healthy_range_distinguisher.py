"""
CDS Healthy-Range LOOCV Analysis
=================================

For each feature, the healthy range [b_min, b_max] is defined as the
min/max value observed among HEALTHY (class == 1) users in the
TRAINING set only (LOOCV: training set = all users except the one
being tested). Because the training set changes for every fold, the
healthy range for each feature is recomputed for every left-out user.

For each left-out user, we count:
  - InsideRange : number of features where the user's value falls
                  within [b_min, b_max] (computed from that fold's
                  training set)
  - OutsideRange: number of features where the user's value falls
                  outside [b_min, b_max]

Missing values ('?' in the dataset) are skipped for that feature
(neither inside nor outside is counted).

Output columns:
  User, OutsideRange, InsideRange, Class, Healthy
"""

import pandas as pd
import numpy as np

DATA_PATH = r'C:\\Users\\kadhi\\OneDrive\\Desktop\\amux\\verilogLearning\\CDS-NI-Algorithm\\data\\arrhythmia.data'
OUTPUT_PATH = r'C:\\Users\\kadhi\\OneDrive\\Desktop\\amux\\verilogLearning\\CDS-NI-Algorithm\\ImprovedCDS\\Prerequisite_Analysis\\outputs\\loocv_results.csv'
N_FEATURES = 279  # columns 0..278 are features, column 279 is class label


def load_data(path):
    df = pd.read_csv(path, header=None, na_values='?')
    X = df.iloc[:, :N_FEATURES].astype(float)
    y = df.iloc[:, N_FEATURES].astype(int)
    return X, y


def run_loocv(X, y):
    n_samples = len(X)
    results = []

    for i in range(n_samples):
        # --- Build this fold's training set (everyone except user i) ---
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False

        train_X = X.iloc[train_mask]
        train_y = y.iloc[train_mask]

        # Only healthy (class == 1) users in THIS fold's training set
        healthy_train = train_X[train_y == 1]

        # Healthy range per feature, recomputed for this fold
        b_min = healthy_train.min(skipna=True)
        b_max = healthy_train.max(skipna=True)

        # --- Evaluate the left-out user against this fold's range ---
        test_x = X.iloc[i]

        inside = 0
        outside = 0
        for f in range(N_FEATURES):
            val = test_x.iloc[f]
            lo = b_min.iloc[f]
            hi = b_max.iloc[f]

            if pd.isna(val) or pd.isna(lo) or pd.isna(hi):
                continue  # skip missing values / undefined ranges

            if lo <= val <= hi:
                inside += 1
            else:
                outside += 1

        results.append({
            'User': i + 1,
            'OutsideRange': outside,
            'InsideRange': inside,
            'Class': int(y.iloc[i]),
            'Healthy': 1 if y.iloc[i] == 1 else 0
        })

    return pd.DataFrame(results)


def summarize(res_df):
    healthy = res_df[res_df['Healthy'] == 1]
    unhealthy = res_df[res_df['Healthy'] == 0]

    print("\n--- Summary ---")
    print(f"Healthy users (class 1):   {len(healthy)}")
    print(f"Unhealthy users (class>1): {len(unhealthy)}")
    print(f"\nAvg features INSIDE range  - Healthy: {healthy['InsideRange'].mean():.2f}, "
          f"Unhealthy: {unhealthy['InsideRange'].mean():.2f}")
    print(f"Avg features OUTSIDE range - Healthy: {healthy['OutsideRange'].mean():.2f}, "
          f"Unhealthy: {unhealthy['OutsideRange'].mean():.2f}")


def main():
    X, y = load_data(DATA_PATH)
    res_df = run_loocv(X, y)
    res_df.to_csv(OUTPUT_PATH, index=False)

    print(res_df.head(10).to_string(index=False))
    print("...")
    print(res_df.tail(10).to_string(index=False))

    summarize(res_df)
    print(f"\nFull results written to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()