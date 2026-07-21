"""Generate corrected slide 3 diagram.
Same as the original chart, just no multiclass bar for the base model."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "presentation_diagrams")
os.makedirs(OUT_DIR, exist_ok=True)

NAVY = "#1E2761"
ICE_BLUE = "#4A90D9"
ACCENT_RED = "#E74C3C"
ACCENT_GREEN = "#27AE60"
MID_GREY = "#7F8C8D"
DARK_TEXT = "#2C3E50"
GOLD = "#F39C12"

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': False,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

# All variants from results_summary.csv, 10-fold CV column
labels = [
    'Base CDS',
    'OVR\nBaseline',
    'Supervised\nBinning',
    'Corr.\nFilter',
    'Dual AF',
    'Fisher\nWeighting',
    'Ratio\nScoring',
    'Healthy\nBar',
    'Rare Class\nParams',
    'Remove\nClasses',
    'Laplace',
    'Class\nThresholds',
    'Final\n(all combined)'
]

# 10-fold CV multiclass accuracy
accuracies = [
    None,   # Base CDS — binary only, no multiclass bar
    28.5,   # 01_ovr_baseline
    26.8,   # 02_supervised_binning
    31.6,   # 03_correlation_filter
    62.6,   # 04_dual_af
    63.3,   # 05_fisher_weighting
    48.5,   # 06_ratio_scoring
    67.0,   # 07_healthy_bar
    28.3,   # 08_rare_class_params
    25.5,   # 09_remove_classes
    28.1,   # 10_laplace
    54.2,   # 11_class_thresholds
    86.8,   # 99_final
]

fig, ax = plt.subplots(figsize=(15, 6.5))
x = np.arange(len(labels))

colors = []
for i, val in enumerate(accuracies):
    if i == 0:
        colors.append('#D5DBDB')      # Base — no bar drawn
    elif i == len(labels) - 1:
        colors.append(ACCENT_GREEN)   # Final
    else:
        colors.append(ICE_BLUE)       # OVR variants

bar_vals = [v if v is not None else 0 for v in accuracies]
bars = ax.bar(x[1:], bar_vals[1:], color=colors[1:], edgecolor='white',
              linewidth=1.5, width=0.7, zorder=3)

# Label each bar with its value
for bar, val in zip(bars, bar_vals[1:]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color=DARK_TEXT)

# Base CDS — no bar, just a text annotation
ax.text(0, 3, 'Binary only\n(no multiclass)', ha='center', va='bottom',
        fontsize=10, fontweight='bold', color=GOLD,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9E6',
                  edgecolor=GOLD, alpha=0.95))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('10-Fold CV Multiclass Accuracy (%)', fontsize=13)
ax.set_title('Independent Changes and Their Effect on Accuracy',
             fontsize=15, fontweight='bold', color=NAVY, pad=15)
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, zorder=0)

fig.text(0.5, -0.03,
         'Base = original CDS (binary healthy/unhealthy only). '
         'All others have OVR implemented alongside. '
         'Many changes depend on each other to be effective.',
         ha='center', fontsize=10, color=MID_GREY, style='italic')

path = os.path.join(OUT_DIR, 'slide_03_independent_changes.png')
fig.savefig(path, dpi=200, facecolor='white', edgecolor='none',
            bbox_inches='tight', pad_inches=0.3)
plt.close(fig)
print(f"Saved: {path}")
