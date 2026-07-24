"""
Find the best feature for demonstrating supervised vs equal-width binning.

We want a feature where:
- Target values cluster in a narrow band
- Equal-width bins split that cluster across MANY bins (bad)
- Supervised bins capture the cluster in 1-2 pure bins (good)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'arrhythmia.data')
df = pd.read_csv(data_path, header=None, na_values='?')
df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
df = df.dropna()
labels = df.iloc[:, -1].values
features = df.iloc[:, :-1].values
n_samples, n_features = features.shape

classes_to_check = [1, 2, 3, 4, 5, 6, 9, 10]

def equal_width_bins(vals, n_bins=6):
    lo, hi = vals.min(), vals.max()
    if lo == hi:
        return [lo - 0.5, hi + 0.5]
    edges = np.linspace(lo, hi, n_bins + 1)
    return edges

def supervised_bins_chi2(feat_vals, is_target, max_bins=6):
    sorted_idx = np.argsort(feat_vals)
    sorted_vals = feat_vals[sorted_idx]
    sorted_target = is_target[sorted_idx]
    n = len(sorted_vals)

    candidates = []
    for i in range(1, n):
        if sorted_vals[i] != sorted_vals[i-1]:
            candidates.append((sorted_vals[i-1] + sorted_vals[i]) / 2)

    if len(candidates) == 0:
        return equal_width_bins(feat_vals)

    edges = [sorted_vals[0] - 0.001, sorted_vals[-1] + 0.001]

    for _ in range(max_bins - 1):
        best_score = -1
        best_split = None
        for c in candidates:
            if c in edges:
                continue
            trial_edges = sorted(edges + [c])
            bins = np.digitize(feat_vals, trial_edges[1:-1])
            ct = pd.crosstab(bins, is_target)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            if (ct.values < 1).any():
                continue
            chi2, p, _, _ = chi2_contingency(ct)
            if chi2 > best_score:
                best_score = chi2
                best_split = c
        if best_split is None:
            break
        edges.append(best_split)
        edges.sort()

    return np.array(edges)

def bin_purity(feat_vals, is_target, edges):
    bin_idx = np.digitize(feat_vals, edges[1:-1])
    purities = []
    target_spread = 0
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        t_count = is_target[mask].sum()
        total = mask.sum()
        purity = t_count / total if total > 0 else 0
        purities.append(purity)
        if t_count > 0:
            target_spread += 1
    max_purity = max(purities) if purities else 0
    return max_purity, target_spread, purities

results = []

for cls in classes_to_check:
    is_target = (labels == cls).astype(int)
    n_target = is_target.sum()
    if n_target < 5:
        continue

    for f_idx in range(n_features):
        col = features[:, f_idx]
        if np.std(col) < 1e-6:
            continue
        if len(np.unique(col)) < 4:
            continue

        ew_edges = equal_width_bins(col, 6)
        sv_edges = supervised_bins_chi2(col, is_target, 6)

        ew_max_pur, ew_spread, ew_purs = bin_purity(col, is_target, ew_edges)
        sv_max_pur, sv_spread, sv_purs = bin_purity(col, is_target, sv_edges)

        # We want: supervised concentrates target (low spread, high purity)
        #          equal-width spreads target (high spread, low purity)
        score = (ew_spread - sv_spread) + (sv_max_pur - ew_max_pur) * 3

        target_vals = col[is_target == 1]
        target_range = target_vals.max() - target_vals.min()
        full_range = col.max() - col.min()
        cluster_ratio = target_range / full_range if full_range > 0 else 1

        results.append({
            'class': cls,
            'feature': f_idx,
            'n_target': n_target,
            'ew_max_purity': ew_max_pur,
            'sv_max_purity': sv_max_pur,
            'ew_target_spread': ew_spread,
            'sv_target_spread': sv_spread,
            'cluster_ratio': cluster_ratio,
            'score': score,
            'target_mean': target_vals.mean(),
            'target_std': target_vals.std(),
        })

results.sort(key=lambda x: x['score'], reverse=True)

print("TOP 20 features for binning demonstration:")
print(f"{'Class':>5} {'Feat':>5} {'n_tgt':>5} {'EW_pur':>7} {'SV_pur':>7} {'EW_spr':>6} {'SV_spr':>6} {'clust':>6} {'score':>6}")
print("-" * 65)
for r in results[:20]:
    print(f"{r['class']:>5} {r['feature']:>5} {r['n_target']:>5} "
          f"{r['ew_max_purity']:>7.3f} {r['sv_max_purity']:>7.3f} "
          f"{r['ew_target_spread']:>6} {r['sv_target_spread']:>6} "
          f"{r['cluster_ratio']:>6.3f} {r['score']:>6.2f}")
