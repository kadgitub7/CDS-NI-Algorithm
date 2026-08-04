import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define class labels and correct matrix data
classes = ['Cls 1', 'Cls 2', 'Cls 3', 'Cls 4', 'Cls 5', 'Cls 6', 'Cls 9', 'Cls 10']

cm = np.array([
    [224, 11,  0,  0,  2,  2,  0,  6],  # Class 1
    [  8, 34,  0,  1,  0,  0,  0,  1],  # Class 2
    [  0,  1, 14,  0,  0,  0,  0,  0],  # Class 3
    [  3,  1,  0, 11,  0,  0,  0,  0],  # Class 4
    [  5,  0,  0,  0,  8,  0,  0,  0],  # Class 5
    [  2,  0,  1,  1,  0, 21,  0,  0],  # Class 6
    [  0,  1,  0,  0,  0,  0,  8,  0],  # Class 9
    [  8,  1,  0,  0,  0,  0,  0,  41]   # Class 10
])

plt.figure(figsize=(7, 6), dpi=300)
sns.set_theme(style="white", font="sans-serif")

ax = sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    cbar=True,
    square=True,
    linewidths=0.5,
    linecolor='lightgray',
    cbar_kws={'label': '', 'shrink': 0.8},
    annot_kws={'size': 9}
)

ax.set_xlabel('Predicted Class', fontsize=11, labelpad=8, fontweight='bold')
ax.set_ylabel('True Class', fontsize=11, labelpad=8, fontweight='bold')

ax.set_xticks(np.arange(len(classes)) + 0.5)
ax.set_yticks(np.arange(len(classes)) + 0.5)
ax.set_xticklabels(classes, fontsize=10, rotation=45, ha='right')
ax.set_yticklabels(classes, fontsize=10, rotation=0)

plt.tight_layout()

# Save to the script's current directory as 'confusion matric.png'
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'confusion matric.png')

plt.savefig(file_path, format='png', bbox_inches='tight')
plt.show()

print(f"File successfully saved to: {file_path}")