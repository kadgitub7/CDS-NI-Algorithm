import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "arrhythmia.data"
OUTPUT_FOLDER = "featureGraphs"

# Healthy class = 1
# Unhealthy classes = 2-16

HEALTHY_CLASS = 1
UNHEALTHY_CLASSES = list(range(2, 17))

# ============================================================
# Load Dataset
# ============================================================

# Dataset uses '?' for missing values
df = pd.read_csv(DATASET_PATH, header=None, na_values='?')

# Last column is the class label
class_col = df.columns[-1]

# Feature columns (279 features)
feature_cols = df.columns[:-1]

# ============================================================
# Create Output Directory
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# Generate Plots
# ============================================================

for feature_idx, feature in enumerate(feature_cols):

    # Extract feature values and labels
    feature_values = df[feature]
    labels = df[class_col]

    # Remove NaN rows for this feature
    valid_mask = ~feature_values.isna()

    x = np.arange(valid_mask.sum())
    y = feature_values[valid_mask].values
    valid_labels = labels[valid_mask].values

    # Determine colors
    colors = [
        'blue' if label == HEALTHY_CLASS else 'red'
        for label in valid_labels
    ]

    # ========================================================
    # Create Plot
    # ========================================================

    plt.figure(figsize=(12, 6))

    plt.scatter(
        x,
        y,
        c=colors,
        alpha=0.7,
        s=20
    )

    plt.title(f'Feature {feature_idx + 1}')
    plt.xlabel('User Index')
    plt.ylabel('Feature Value')

    # Create legend manually
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0],
               marker='o',
               color='w',
               label='Healthy (Class 1)',
               markerfacecolor='blue',
               markersize=8),

        Line2D([0], [0],
               marker='o',
               color='w',
               label='Unhealthy (Class 2-16)',
               markerfacecolor='red',
               markersize=8)
    ]

    plt.legend(handles=legend_elements)

    plt.grid(True, alpha=0.3)

    # ========================================================
    # Save Plot
    # ========================================================

    output_path = os.path.join(
        OUTPUT_FOLDER,
        f'feature_{feature_idx + 1}.png'
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

print("\nAll feature graphs generated successfully.")