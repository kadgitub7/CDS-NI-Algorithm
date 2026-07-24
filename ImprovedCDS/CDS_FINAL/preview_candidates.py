"""Preview top binning candidates as scatter plots to visually pick the best one."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'arrhythmia.data')
df = pd.read_csv(data_path, header=None, na_values='?')
df = df.dropna(axis=1, thresh=int(0.8 * len(df)))
df = df.dropna()
labels = df.iloc[:, -1].values
features = df.iloc[:, :-1].values

candidates = [
    (3, 111), (3, 110), (9, 124), (9, 151), (9, 19),
    (4, 75), (10, 90), (3, 99), (3, 112),
]

N_BINS = 6

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for idx, (cls, f_idx) in enumerate(candidates):
    ax = axes[idx]
    col = features[:, f_idx]
    is_target = labels == cls
    is_healthy = labels == 1
    is_other = ~is_target & ~is_healthy

    ax.scatter(np.where(is_other)[0], col[is_other],
               c='#cccccc', s=8, alpha=0.3, label='Other Disease', zorder=1)
    ax.scatter(np.where(is_healthy)[0], col[is_healthy],
               c='#3498db', s=10, alpha=0.4, label='Healthy', zorder=2)
    ax.scatter(np.where(is_target)[0], col[is_target],
               c='#e74c3c', s=20, alpha=0.9, label=f'Class {cls}',
               zorder=3, edgecolors='black', linewidths=0.3)

    ew_edges = np.linspace(col.min(), col.max(), N_BINS + 1)
    for e in ew_edges:
        ax.axhline(y=e, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    tv = col[is_target]
    ax.set_title(f'Class {cls}, F{f_idx} (n={is_target.sum()}) '
                 f'range=[{tv.min():.0f},{tv.max():.0f}]', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_xlabel('Patient Index')
    ax.set_ylabel('Value')

plt.suptitle('Candidate features with equal-width bin lines overlaid', fontsize=13)
plt.tight_layout()
fig.savefig('preview_binning_candidates.png', dpi=120, bbox_inches='tight')
print("Saved preview_binning_candidates.png")
