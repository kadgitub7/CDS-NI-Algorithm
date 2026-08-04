import matplotlib.pyplot as plt
import numpy as np
import os

# Set font styling for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300
})

# --- Data Extraction (Using Max values instead of Mean) ---
# Training percentages
x_ratios = np.array([50, 60, 70, 75, 80, 85, 90])
x_labels = ['50/50', '60/40', '70/30', '75/25', '80/20', '85/15', '90/10']

# Random Split Max Data (from Table C2: Multi Best, and Binary Max values)
rand_multi_max = np.array([85.1, 86.23, 87.2, 86.54, 89.29, 90.48, 92.86])
rand_bin_max   = np.array([87.5, 92.22, 90.4, 91.35, 94.05, 96.83, 97.62])

# Stratified Split Max Data (from Table C7: Multi Best, and Binary Best equivalents)
strat_x = np.array([50, 60, 70, 80, 90])
strat_multi_max = np.array([81.52, 85.71, 85.94, 89.29, 91.11])
strat_bin_max   = np.array([84.83, 89.29, 87.5, 92.86, 95.56])

# --- Plotting ---
fig, ax = plt.subplots(figsize=(7, 4.5))

# --- Plot Random Split (Solid Lines) ---
ax.plot(x_ratios, rand_multi_max, '-o', color='#1f77b4', linewidth=1.5, label='Multiclass Max (Random)')
ax.plot(x_ratios, rand_bin_max, '-s', color='#ff7f0e', linewidth=1.5, label='Binary Max (Random)')

# --- Plot Stratified Split (Dashed Lines) ---
ax.plot(strat_x, strat_multi_max, '--o', color='#1f77b4', alpha=0.7, linewidth=1.2, label='Multiclass Max (Stratified)')
ax.plot(strat_x, strat_bin_max, '--s', color='#ff7f0e', alpha=0.7, linewidth=1.2, label='Binary Max (Stratified)')

# --- Formatting ---
ax.set_title('FIGURE 6: Maximum Accuracy vs. Training Set Size', pad=12, fontweight='bold')
ax.set_xlabel('Training Percentage / Split Ratio')
ax.set_ylabel('Maximum Accuracy (%)')

# Customize x-ticks
ax.set_xticks(x_ratios)
ax.set_xticklabels(x_labels)

ax.set_ylim(75, 100)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='none')

plt.tight_layout()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'comparison_splits.png')

# Save figure strictly as PNG format
plt.savefig(file_path, format='png', bbox_inches='tight')

plt.show()