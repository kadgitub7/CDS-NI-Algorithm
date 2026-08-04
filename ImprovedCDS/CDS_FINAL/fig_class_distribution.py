"""Figure: UCI Arrhythmia Dataset — Class distribution by sex."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "old", "data", "arrhythmia.data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

data = np.genfromtxt(DATA_PATH, delimiter=',', filling_values=np.nan)
labels = data[:, -1].astype(int)
sex = data[:, 1]  # 0=male, 1=female per arrhythmia.names

CLASS_INFO = {
    1:  "Normal",
    2:  "Ischemic Changes",
    3:  "Old Ant. MI",
    4:  "Old Inf. MI",
    5:  "Sinus Tachy",
    6:  "Sinus Brady",
    7:  "PVC",
    8:  "Supravent. PC",
    9:  "LBBB",
    10: "RBBB",
    14: "LV Hypertrophy",
    15: "A-Fib/Flutter",
    16: "Other",
}

all_classes = sorted(set(labels))
x_labels = [f"{CLASS_INFO.get(c, f'Cls {c}')} ({c})" for c in all_classes]

male_counts = []
female_counts = []
for c in all_classes:
    mask = (labels == c)
    male_counts.append(int((mask & (sex == 0)).sum()))
    female_counts.append(int((mask & (sex == 1)).sum()))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'axes.linewidth': 0.6,
})

fig, ax = plt.subplots(figsize=(8, 4.0))
x = np.arange(len(all_classes))
width = 0.35

bars_m = ax.bar(x - width/2, male_counts, width, label='Male', color='#2196F3', alpha=0.85, edgecolor='white', linewidth=0.5)
bars_f = ax.bar(x + width/2, female_counts, width, label='Female', color='#E91E63', alpha=0.85, edgecolor='white', linewidth=0.5)

for bar_group in [bars_m, bars_f]:
    for bar in bar_group:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, str(int(h)),
                    ha='center', va='bottom', fontsize=7, fontweight='normal')

ax.set_ylabel("Number of Patients")
ax.set_xlabel("Arrhythmia Class")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=40, ha='right')
ax.legend(loc='upper right', frameon=True, edgecolor='#ccc', fancybox=False, framealpha=0.95)
ax.set_ylim(0, max(max(male_counts), max(female_counts)) * 1.15)
ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
ax.grid(axis='y', alpha=0.3, linewidth=0.4)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

total = sum(male_counts) + sum(female_counts)
ax.text(0.72, 0.92, f"N = {total}  ({sum(male_counts)} M / {sum(female_counts)} F)",
        transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#ccc', lw=0.5))

fig.tight_layout()
path = os.path.join(OUT_DIR, "fig_class_distribution.png")
fig.savefig(path, dpi=300, bbox_inches='tight')
fig.savefig(path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close(fig)
print(f"Saved: {path}")
