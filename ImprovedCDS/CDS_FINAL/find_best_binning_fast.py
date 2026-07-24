"""Fast search for best binning demonstration feature.

Goal: find feature where target values cluster tightly but equal-width
bins spread them across many bins. Supervised bins would concentrate them.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'arrhythmia.data')
df = pd.read_csv(data_path, header=None, na_values='?')
df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
df = df.dropna()
labels = df.iloc[:, -1].values
features = df.iloc[:, :-1].values
n_samples, n_features = features.shape

classes_to_check = [2, 3, 4, 5, 6, 9, 10]
N_BINS = 6

results = []

for cls in classes_to_check:
    is_target = labels == cls
    n_target = is_target.sum()
    if n_target < 5:
        continue

    for f_idx in range(n_features):
        col = features[:, f_idx]
        n_unique = len(np.unique(col))
        if n_unique < 6:
            continue

        full_range = col.max() - col.min()
        if full_range < 1e-6:
            continue

        target_vals = col[is_target]
        rest_vals = col[~is_target]

        # Equal-width bin edges
        ew_edges = np.linspace(col.min(), col.max(), N_BINS + 1)

        # Which equal-width bins contain target values?
        t_bins_ew = np.digitize(target_vals, ew_edges[1:-1])
        ew_bins_with_target = len(np.unique(t_bins_ew))

        # Equal-width max purity
        ew_purities = []
        for b in range(N_BINS):
            t_in = np.sum(t_bins_ew == b)
            r_in = np.sum(np.digitize(rest_vals, ew_edges[1:-1]) == b)
            total = t_in + r_in
            if total > 0 and t_in > 0:
                ew_purities.append(t_in / total)
        ew_max_pur = max(ew_purities) if ew_purities else 0

        # How clustered are target values?
        target_iqr = np.percentile(target_vals, 75) - np.percentile(target_vals, 25)
        cluster_tightness = 1 - (target_iqr / full_range) if full_range > 0 else 0

        # Could a single bin capture most targets?
        # Find the densest window for target vals
        sorted_t = np.sort(target_vals)
        best_capture = 0
        best_window = (0, 0)
        window_size = int(0.7 * n_target)  # 70% of targets
        if window_size >= 2:
            for i in range(len(sorted_t) - window_size + 1):
                width = sorted_t[i + window_size - 1] - sorted_t[i]
                # Count rest values in this window
                r_in_window = np.sum((rest_vals >= sorted_t[i]) & (rest_vals <= sorted_t[i + window_size - 1]))
                purity = window_size / (window_size + r_in_window) if (window_size + r_in_window) > 0 else 0
                if purity > best_capture:
                    best_capture = purity
                    best_window = (sorted_t[i], sorted_t[i + window_size - 1])

        # Score: we want high EW spread (target in many bins) + high potential purity
        # + tight clustering
        score = (ew_bins_with_target * 0.5) + (best_capture * 3) + (cluster_tightness * 2) - (ew_max_pur * 2)

        results.append({
            'class': cls,
            'feature': f_idx,
            'n_target': n_target,
            'ew_bins_with_target': ew_bins_with_target,
            'ew_max_purity': ew_max_pur,
            'best_window_purity': best_capture,
            'best_window': best_window,
            'cluster_tightness': cluster_tightness,
            'target_iqr': target_iqr,
            'full_range': full_range,
            'score': score,
        })

results.sort(key=lambda x: x['score'], reverse=True)

print("TOP 25 features for binning demonstration:")
print(f"{'Cls':>3} {'Feat':>5} {'n_t':>4} {'EW_bins':>7} {'EW_pur':>7} {'Win_pur':>7} {'Tight':>6} {'Score':>6}  Window")
print("-" * 80)
for r in results[:25]:
    w = r['best_window']
    print(f"{r['class']:>3} {r['feature']:>5} {r['n_target']:>4} "
          f"{r['ew_bins_with_target']:>7} {r['ew_max_purity']:>7.3f} "
          f"{r['best_window_purity']:>7.3f} {r['cluster_tightness']:>6.3f} "
          f"{r['score']:>6.2f}  [{w[0]:.0f}, {w[1]:.0f}]")
