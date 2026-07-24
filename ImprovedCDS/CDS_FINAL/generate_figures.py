"""
Generate two publication-quality figures in classAnalysis scatter plot style:
1. Binning: Feature 75, Class 4 — EW bins vs supervised bins
2. OVR: Feature 14, Class 6 — binary coloring vs OVR 3-color
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import cds_dualAF as m

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'font.family': 'serif',
})

data, labels = m.load_data(str(Path(__file__).parent.parent / 'data' / 'arrhythmia.data'))


def compute_supervised_bins(vv, is_tgt, max_bins=6, min_support=3):
    """Matches the actual CDS-OVR _supervised_bin_edges algorithm:
    greedy chi-squared splits with early stopping (gain < 0.5)
    and min_support constraint per bin."""
    vv = vv[~np.isnan(vv)]
    n = len(vv)
    vmin, vmax = float(vv.min()), float(vv.max())
    if vmin == vmax or n < 2 * min_support:
        return np.array([vmin - 0.5, vmax + 0.5])

    sort_idx = np.argsort(vv)
    sv = vv[sort_idx]
    st = is_tgt[sort_idx] if not isinstance(is_tgt, np.ndarray) else np.asarray(is_tgt)[sort_idx]
    st = st.astype(float)
    edges = [vmin, vmax]

    for _ in range(max_bins - 1):
        best_gain, best_split, best_seg = 0.0, None, None
        for seg_i in range(len(edges) - 1):
            lo, hi = edges[seg_i], edges[seg_i + 1]
            if seg_i == 0:
                mask = sv <= hi
            elif seg_i == len(edges) - 2:
                mask = sv > lo
            else:
                mask = (sv > lo) & (sv <= hi)
            seg_vals, seg_targ = sv[mask], st[mask]
            n_seg = len(seg_vals)
            if n_seg < 2 * min_support:
                continue
            n_t = seg_targ.sum()
            n_r = n_seg - n_t
            if n_t == 0 or n_r == 0:
                continue
            candidates = np.where(seg_vals[:-1] != seg_vals[1:])[0]
            if len(candidates) == 0:
                continue
            cum_t = np.cumsum(seg_targ)
            for ci in candidates:
                n_left, n_right = ci + 1, n_seg - ci - 1
                if n_left < min_support or n_right < min_support:
                    continue
                t_left = cum_t[ci]
                t_right = n_t - t_left
                r_left, r_right = n_left - t_left, n_right - t_right
                e_tl = n_left * n_t / n_seg
                e_tr = n_right * n_t / n_seg
                e_rl = n_left * n_r / n_seg
                e_rr = n_right * n_r / n_seg
                chi2 = sum((o - e)**2 / e for o, e in
                           [(t_left, e_tl), (t_right, e_tr),
                            (r_left, e_rl), (r_right, e_rr)] if e > 0)
                if chi2 > best_gain:
                    best_gain = chi2
                    best_split = (seg_vals[ci] + seg_vals[ci + 1]) / 2.0
                    best_seg = seg_i
        if best_split is None or best_gain < 0.5:
            break
        edges.insert(best_seg + 1, best_split)

    edges[0] = vmin - 1e-10
    edges[-1] = vmax + 1e-10
    return np.array(sorted(set(edges)))


def make_scatter_3color(ax, col, labels, target_class, valid):
    is_target = (labels == target_class) & valid
    is_healthy = (labels == 1) & valid
    is_other = ~is_target & ~is_healthy & valid
    ax.scatter(np.where(is_other)[0], col[is_other],
               c='#e8a317', s=8, alpha=0.35, label='Other Unhealthy', zorder=1)
    ax.scatter(np.where(is_healthy)[0], col[is_healthy],
               c='#3498db', s=10, alpha=0.4, label='Healthy', zorder=2)
    ax.scatter(np.where(is_target)[0], col[is_target],
               c='#e74c3c', s=20, alpha=0.9,
               label=f'Class {target_class} (n={is_target.sum()})',
               zorder=3, edgecolors='black', linewidths=0.3)
    ax.set_xlabel('Patient Index', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)


def make_scatter_binary(ax, col, labels, valid):
    is_healthy = (labels == 1) & valid
    is_unhealthy = (labels != 1) & valid
    ax.scatter(np.where(is_unhealthy)[0], col[is_unhealthy],
               c='#e74c3c', s=10, alpha=0.4,
               label=f'Unhealthy (n={is_unhealthy.sum()})', zorder=2)
    ax.scatter(np.where(is_healthy)[0], col[is_healthy],
               c='#3498db', s=10, alpha=0.4,
               label=f'Healthy (n={is_healthy.sum()})', zorder=3)
    ax.set_xlabel('Patient Index', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)


# ── Figure 1: Binning — Feature 136, Class 5 ──
# Supervised achieves 100% purity vs EW's 6%

BINNING_FEAT = 136
BINNING_CLASS = 5
RARE_CLASSES = {4, 5, 9}
MIN_SUPPORT = 2 if BINNING_CLASS in RARE_CLASSES else 3

col = data[:, BINNING_FEAT]
valid = ~np.isnan(col)
fv = col[valid]
tv = ((labels == BINNING_CLASS) & valid)[valid].astype(int)
nv = valid.sum()
max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), 6)

print(f"=== Binning: Feature {BINNING_FEAT}, Class {BINNING_CLASS} ===")
target_vals = fv[tv.astype(bool)]
print(f"  Target values: {sorted(target_vals)}")
print(f"  Target range: [{target_vals.min()}, {target_vals.max()}]")
print(f"  Full range: [{fv.min()}, {fv.max()}]")
print(f"  Max bins (Sturges): {max_nb}")

ew_edges = np.linspace(fv.min(), fv.max(), max_nb + 1)
sv_edges = compute_supervised_bins(fv, tv, max_nb, MIN_SUPPORT)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

make_scatter_3color(ax1, col, labels, BINNING_CLASS, valid)
for e in ew_edges:
    ax1.axhline(y=e, color='#2ca02c', linewidth=1.2, linestyle='--', alpha=0.7)
ax1.set_title(f'(a) Equal-Width Bins ({max_nb} bins)', fontsize=11)
ax1.legend(fontsize=8, loc='upper right')

make_scatter_3color(ax2, col, labels, BINNING_CLASS, valid)
for e in sv_edges:
    ax2.axhline(y=e, color='#d62728', linewidth=1.2, linestyle='--', alpha=0.7)
n_sv = len(sv_edges) - 1
ax2.set_title(f'(b) Supervised Chi-Squared Bins ({n_sv} bins)', fontsize=11)
ax2.legend(fontsize=8, loc='upper right')

fig.suptitle(f'Feature {BINNING_FEAT} — Class {BINNING_CLASS}: '
             f'Equal-Width vs Supervised Binning', fontsize=12, y=1.0)
plt.tight_layout()
fig.savefig('fig_binning_comparison.png', bbox_inches='tight')

print(f"  EW edges ({max_nb} bins): {[f'{e:.1f}' for e in ew_edges]}")
print(f"  SV edges ({n_sv} bins): {[f'{e:.1f}' for e in sv_edges]}")
for label, edges in [("EW", ew_edges), ("SV", sv_edges)]:
    bins_arr = np.digitize(fv, edges[1:-1])
    for b in range(len(edges)-1):
        mask = bins_arr == b
        n_t = tv[mask].sum()
        n_tot = mask.sum()
        if n_tot > 0:
            print(f"  {label} [{edges[b]:.0f},{edges[b+1]:.0f}]: "
                  f"{n_t}/{n_tot} target ({n_t/n_tot*100:.1f}%)")
print("Saved fig_binning_comparison.png")
plt.close()


# ── Figure 2: OVR — Feature 14, Class 6 ──

OVR_FEAT = 14
OVR_CLASS = 6

col = data[:, OVR_FEAT]
valid = ~np.isnan(col)

print(f"\n=== OVR: Feature {OVR_FEAT}, Class {OVR_CLASS} ===")
is_target = (labels == OVR_CLASS) & valid
target_vals = col[is_target]
print(f"  Target values range: [{target_vals.min()}, {target_vals.max()}]")
print(f"  Target mean: {target_vals.mean():.1f}")
print(f"  Healthy mean: {col[(labels==1)&valid].mean():.1f}")
print(f"  Other unhealthy mean: {col[(labels!=1)&(labels!=OVR_CLASS)&valid].mean():.1f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

make_scatter_binary(ax1, col, labels, valid)
ax1.set_title('(a) Binary: Healthy vs All Unhealthy', fontsize=11)
ax1.legend(fontsize=8, loc='upper right')

make_scatter_3color(ax2, col, labels, OVR_CLASS, valid)
ax2.set_title(f'(b) OVR: Class {OVR_CLASS} vs Other Unhealthy vs Healthy', fontsize=11)
ax2.legend(fontsize=8, loc='upper right')

fig.suptitle(f'Feature {OVR_FEAT} — Binary vs One-vs-Rest Class Separation', fontsize=12, y=1.0)
plt.tight_layout()
fig.savefig('fig_ovr_separation.png', bbox_inches='tight')
print("Saved fig_ovr_separation.png")
plt.close()
