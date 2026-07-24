"""Find features where equal-width bins split the target cluster down the middle."""
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

N_BINS = 6
results = []

for cls in [2, 3, 4, 5, 6, 9, 10]:
    is_target = labels == cls
    n_target = is_target.sum()
    if n_target < 5:
        continue

    for f_idx in range(n_features):
        col = features[:, f_idx]
        if len(np.unique(col)) < 6:
            continue
        full_range = col.max() - col.min()
        if full_range < 1e-6:
            continue

        tv = col[is_target]
        rv = col[~is_target]
        target_median = np.median(tv)

        ew_edges = np.linspace(col.min(), col.max(), N_BINS + 1)

        # How many EW edges fall INSIDE the target cluster (between p10 and p90)?
        t_p10, t_p90 = np.percentile(tv, 10), np.percentile(tv, 90)
        t_range = t_p90 - t_p10
        if t_range < 1e-6:
            continue

        edges_in_cluster = sum(1 for e in ew_edges[1:-1] if t_p10 <= e <= t_p90)

        # Separation: how distinct is target from rest?
        r_in_cluster = np.sum((rv >= t_p10) & (rv <= t_p90))
        separation = 1 - (r_in_cluster / len(rv)) if len(rv) > 0 else 0

        # Cluster tightness (IQR / full range)
        iqr = np.percentile(tv, 75) - np.percentile(tv, 25)
        tightness = 1 - (iqr / full_range)

        # Best metric: edges cutting through cluster + good separation + tight cluster
        score = edges_in_cluster * 2 + separation * 3 + tightness * 1

        # Target purity in EW bins
        t_ew_bins = np.digitize(tv, ew_edges[1:-1])
        ew_spread = len(np.unique(t_ew_bins))

        results.append({
            'class': cls, 'feature': f_idx, 'n_target': n_target,
            'edges_in_cluster': edges_in_cluster,
            'ew_spread': ew_spread,
            'separation': separation,
            'tightness': tightness,
            'target_p10': t_p10, 'target_p90': t_p90,
            'score': score,
        })

results.sort(key=lambda x: x['score'], reverse=True)

print(f"{'Cls':>3} {'Feat':>5} {'n_t':>4} {'EW_cut':>6} {'EW_spr':>6} {'Sep':>5} {'Tight':>5} {'Score':>6}  Cluster[p10..p90]")
print("-" * 80)
for r in results[:30]:
    print(f"{r['class']:>3} {r['feature']:>5} {r['n_target']:>4} "
          f"{r['edges_in_cluster']:>6} {r['ew_spread']:>6} "
          f"{r['separation']:>5.2f} {r['tightness']:>5.2f} "
          f"{r['score']:>6.2f}  [{r['target_p10']:.0f}..{r['target_p90']:.0f}]")
