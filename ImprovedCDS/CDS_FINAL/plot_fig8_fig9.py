"""Generate Figure 8 (Ablation Impact) and Figure 9 (Hyperparameter Sensitivity)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
})

BASE = r"C:\Users\kadhi\OneDrive\Desktop\amux\verilogLearning\CDS-NI-Algorithm\ImprovedCDS\CDS_FINAL"

with open(os.path.join(BASE, "ablation_full_data.json")) as f:
    full = json.load(f)
with open(os.path.join(BASE, "ablation_missing_data.json")) as f:
    missing = json.load(f)

ablation_sig = full["ablation_significance"]

all_ablations = {}
for key in ablation_sig:
    all_ablations[key] = ablation_sig[key]
for key in missing:
    all_ablations[key] = missing[key]

labels_map = {
    "ratio_scoring":       "Ratio Scoring",
    "dual_af":             "Dual AF",
    "supervised_binning":  "Supervised Binning",
    "fisher_weighting":    "Fisher Weighting",
    "per_class_thresholds":"Per-Class Thresholds",
    "correlation_filter":  "Correlation Filter",
    "sex_branching":       "Sex Branching",
    "healthy_bar":         "Healthy Bar",
    "against_scale":       "Against Scale",
    "laplace_smoothing":   "Laplace Smoothing",
    "rare_class_params":   "Rare Class Params",
}

components = []
for key, data in all_ablations.items():
    base = np.array(data["baseline_accs"])
    abl = np.array(data["ablated_accs"])
    diff = base - abl
    components.append({
        "key": key,
        "label": labels_map.get(key, key),
        "delta": diff.mean(),
        "std": diff.std(ddof=1),
        "p": data["p_value"],
        "sig": data["p_value"] < 0.05,
    })

components.sort(key=lambda c: c["delta"])

labels = [c["label"] for c in components]
deltas = [c["delta"] for c in components]
stds = [c["std"] for c in components]
sigs = [c["sig"] for c in components]
pvals = [c["p"] for c in components]

SIG_COLOR = "#2E7D32"
NONSIG_COLOR = "#9E9E9E"
colors = [SIG_COLOR if s else NONSIG_COLOR for s in sigs]

fig, ax = plt.subplots(figsize=(5.5, 3.8))

bars = ax.barh(
    range(len(labels)), deltas,
    xerr=stds, height=0.55,
    color=colors, edgecolor='none',
    error_kw=dict(elinewidth=0.7, capsize=2.5, capthick=0.5, color='#333'),
    zorder=3,
)

ax.axvline(x=0, color='#333', linewidth=0.5, zorder=2)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Accuracy change when component removed (pp)")

for i, (d, p, s) in enumerate(zip(deltas, pvals, sigs)):
    if s:
        marker = "**" if p < 0.01 else "*"
        offset = d + stds[i] + 0.3 if d > 0 else d - stds[i] - 0.3
        ha = 'left' if d > 0 else 'right'
        ax.text(offset, i, marker, ha=ha, va='center',
                fontsize=9, fontweight='bold', color=SIG_COLOR)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=SIG_COLOR, label='Significant (p < 0.05)'),
    Patch(facecolor=NONSIG_COLOR, label='Not significant'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True,
          edgecolor='#ccc', fancybox=False, framealpha=0.95)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.6, zorder=1)

fig.tight_layout()
fig8_path = os.path.join(BASE, "fig8_ablation_impact.png")
fig.savefig(fig8_path)
fig.savefig(fig8_path.replace('.png', '.pdf'))
plt.close(fig)
print(f"Figure 8 saved: {fig8_path}")


# ================================================================
# FIGURE 9: Hyperparameter Sensitivity (3 panels)
# ================================================================

corr = full["sensitivity_CORR_THRESHOLD"]
bins = full["sensitivity_MAX_BINS"]
feat = full["sensitivity_FEATURES_PER_CLASS"]

OPTIMAL = {
    "corr": 0.8,
    "bins": 6,
    "feat": 18,
}

LINE_COLOR = "#1565C0"
OPT_COLOR = "#C62828"
FILL_ALPHA = 0.12

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=False)

# Panel (a): CORR_THRESHOLD
ax = axes[0]
xs = sorted([float(k) for k in corr.keys()])
means = [corr[str(x) if str(x) in corr else f"{x:.1f}"]["mean"] for x in xs]
stds_c = [corr[str(x) if str(x) in corr else f"{x:.1f}"]["std"] for x in xs]
means = np.array(means)
stds_c = np.array(stds_c)

ax.fill_between(xs, means - stds_c, means + stds_c, alpha=FILL_ALPHA, color=LINE_COLOR, linewidth=0)
ax.plot(xs, means, 'o-', color=LINE_COLOR, markersize=4, zorder=3)

opt_idx = xs.index(OPTIMAL["corr"])
ax.plot(xs[opt_idx], means[opt_idx], 's', color=OPT_COLOR, markersize=7, zorder=4, markeredgecolor='white', markeredgewidth=0.8)
ax.annotate(f'{means[opt_idx]:.2f}%', (xs[opt_idx], means[opt_idx]),
            textcoords="offset points", xytext=(0, 10), ha='center',
            fontsize=7.5, color=OPT_COLOR, fontweight='bold')

ax.set_xlabel(r"$\rho_{\max}$ (Correlation Threshold)")
ax.set_ylabel("10-Fold CV Accuracy (%)")
ax.set_title("(a)", fontweight='normal', loc='left')
ax.set_xticks(xs)

# Panel (b): MAX_BINS
ax = axes[1]
xs = sorted([int(k) for k in bins.keys()])
means = np.array([bins[str(x)]["mean"] for x in xs])
stds_b = np.array([bins[str(x)]["std"] for x in xs])

ax.fill_between(xs, means - stds_b, means + stds_b, alpha=FILL_ALPHA, color=LINE_COLOR, linewidth=0)
ax.plot(xs, means, 'o-', color=LINE_COLOR, markersize=4, zorder=3)

opt_idx = xs.index(OPTIMAL["bins"])
ax.plot(xs[opt_idx], means[opt_idx], 's', color=OPT_COLOR, markersize=7, zorder=4, markeredgecolor='white', markeredgewidth=0.8)
ax.annotate(f'{means[opt_idx]:.2f}%', (xs[opt_idx], means[opt_idx]),
            textcoords="offset points", xytext=(0, 10), ha='center',
            fontsize=7.5, color=OPT_COLOR, fontweight='bold')

ax.set_xlabel(r"$B_{\max}$ (Maximum Bins)")
ax.set_title("(b)", fontweight='normal', loc='left')
ax.set_xticks(xs)

# Panel (c): FEATURES_PER_CLASS
ax = axes[2]
xs = sorted([int(k) for k in feat.keys()])
means = np.array([feat[str(x)]["mean"] for x in xs])
stds_f = np.array([feat[str(x)]["std"] for x in xs])

ax.fill_between(xs, means - stds_f, means + stds_f, alpha=FILL_ALPHA, color=LINE_COLOR, linewidth=0)
ax.plot(xs, means, 'o-', color=LINE_COLOR, markersize=4, zorder=3)

opt_idx = xs.index(OPTIMAL["feat"])
ax.plot(xs[opt_idx], means[opt_idx], 's', color=OPT_COLOR, markersize=7, zorder=4, markeredgecolor='white', markeredgewidth=0.8)
ax.annotate(f'{means[opt_idx]:.2f}%', (xs[opt_idx], means[opt_idx]),
            textcoords="offset points", xytext=(0, 10), ha='center',
            fontsize=7.5, color=OPT_COLOR, fontweight='bold')

ax.set_xlabel(r"$K$ (Features per Class)")
ax.set_title("(c)", fontweight='normal', loc='left')
ax.set_xticks(xs)

ax.annotate("Insensitive to $K$:\naccuracy constant\nacross all values",
            xy=(20, means[opt_idx]), xycoords='data',
            xytext=(22, means[opt_idx] - 0.55), textcoords='data',
            ha='center', va='top', fontsize=7, fontstyle='italic',
            color='#555',
            bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#bbb', lw=0.5))

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.6)
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

fig.tight_layout(w_pad=2.0)
fig9_path = os.path.join(BASE, "fig9_hyperparameter_sensitivity.png")
fig.savefig(fig9_path)
fig.savefig(fig9_path.replace('.png', '.pdf'))
plt.close(fig)
print(f"Figure 9 saved: {fig9_path}")
