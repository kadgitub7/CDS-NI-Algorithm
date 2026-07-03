"""Per-disease-class feature distribution analysis.

Generates scatter plots for each feature, one set per disease class.
Set 0: Healthy (blue) vs All Unhealthy (red)  — same as original graphs
Sets 1-13: Class X (red) vs Everyone Else (gray), with Healthy highlighted (blue)

This reveals whether individual disease classes cluster in feature space
in ways that are invisible when all unhealthy classes are lumped together.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import cds_dualAF as m


def load():
    data, labels = m.load_data(str(Path(__file__).parent / 'data' / 'arrhythmia.data'))
    is_bin = m.classify_features(data)
    return data, labels, is_bin


def plot_feature_class(data, labels, feat_idx, class_id, ax, is_bin_feat):
    """Plot one feature for one class comparison.

    class_id=0  → Healthy (1) vs All Unhealthy
    class_id=2+ → Class X (red) vs Healthy (blue) vs Other Unhealthy (gray)
    """
    col = data[:, feat_idx]
    valid = ~np.isnan(col)

    if class_id == 0:
        # Original view: healthy vs all unhealthy
        h_mask = (labels == 1) & valid
        u_mask = (labels != 1) & valid

        ax.scatter(np.where(u_mask)[0], col[u_mask],
                   c='#e74c3c', s=12, alpha=0.6, label='Unhealthy', zorder=2)
        ax.scatter(np.where(h_mask)[0], col[h_mask],
                   c='#3498db', s=12, alpha=0.6, label='Healthy', zorder=3)
        ax.set_title(f'Feature {feat_idx} — Healthy vs All Unhealthy', fontsize=10)
    else:
        # Per-class view
        h_mask = (labels == 1) & valid
        c_mask = (labels == class_id) & valid
        o_mask = (labels != 1) & (labels != class_id) & valid
        n_class = int(c_mask.sum())

        ax.scatter(np.where(o_mask)[0], col[o_mask],
                   c='#cccccc', s=8, alpha=0.3, label='Other Unhealthy', zorder=1)
        ax.scatter(np.where(h_mask)[0], col[h_mask],
                   c='#3498db', s=10, alpha=0.4, label='Healthy', zorder=2)
        ax.scatter(np.where(c_mask)[0], col[c_mask],
                   c='#e74c3c', s=20, alpha=0.9, label=f'Class {class_id} (n={n_class})',
                   zorder=3, edgecolors='black', linewidths=0.3)
        ax.set_title(f'Feature {feat_idx} — Class {class_id} (n={n_class}) vs Rest',
                     fontsize=10)

    ax.set_xlabel('User Index', fontsize=8)
    ax.set_ylabel('Value', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(labelsize=7)


def generate_all(features=None, classes=None):
    data, labels, is_bin = load()

    all_classes = sorted(set(labels))
    disease_classes = [c for c in all_classes if c != 1]

    if classes is None:
        class_ids = [0] + [int(c) for c in disease_classes]
    else:
        class_ids = classes

    if features is None:
        features = list(range(m.N_FEAT))

    out_root = Path(__file__).parent / 'output' / 'classAnalysis'

    for cid in class_ids:
        if cid == 0:
            folder = out_root / 'healthy_vs_unhealthy'
        else:
            n = int((labels == cid).sum())
            folder = out_root / f'class_{cid:02d}_n{n}'

        folder.mkdir(parents=True, exist_ok=True)
        print(f'Generating class {cid} ({folder.name})...')

        for fi in features:
            col = data[:, fi]
            if np.all(np.isnan(col)):
                continue

            fig, ax = plt.subplots(figsize=(10, 4))
            plot_feature_class(data, labels, fi, cid, ax, is_bin[fi])
            fig.tight_layout()
            fig.savefig(folder / f'feature_{fi}.png', dpi=100)
            plt.close(fig)

        print(f'  → {len(features)} features saved to {folder}')


def generate_summary(top_n=30):
    """Generate a compact summary: for each disease class, show the top_n
    features where that class is most separated from healthy.

    Outputs one multi-panel figure per class showing the most discriminative
    features, ranked by separation (difference in means / pooled std).
    """
    data, labels, is_bin = load()

    all_classes = sorted(set(labels))
    disease_classes = [int(c) for c in all_classes if c != 1]

    out_root = Path(__file__).parent / 'output' / 'classAnalysis'
    out_root.mkdir(parents=True, exist_ok=True)

    h_mask = labels == 1

    for cid in disease_classes:
        c_mask = labels == cid
        n_class = int(c_mask.sum())
        if n_class < 2:
            continue

        scores = []
        for fi in range(m.N_FEAT):
            col = data[:, fi]
            hv = col[h_mask & ~np.isnan(col)]
            cv = col[c_mask & ~np.isnan(col)]
            if len(hv) < 5 or len(cv) < 2:
                continue
            pooled_std = np.sqrt((np.var(hv) * len(hv) + np.var(cv) * len(cv))
                                 / (len(hv) + len(cv)))
            if pooled_std < 1e-9:
                continue
            sep = abs(np.mean(cv) - np.mean(hv)) / pooled_std
            scores.append((fi, sep))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_feats = scores[:top_n]

        n_cols = 5
        n_rows = (len(top_feats) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else (axes if hasattr(axes, '__len__') else [axes])

        for idx, (fi, sep) in enumerate(top_feats):
            ax = axes[idx]
            plot_feature_class(data, labels, fi, cid, ax, is_bin[fi])
            ax.set_title(f'F{fi} (sep={sep:.2f})', fontsize=9)

        for idx in range(len(top_feats), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'Class {cid} (n={n_class}) — Top {len(top_feats)} Most '
                     f'Discriminative Features vs Healthy',
                     fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(out_root / f'summary_class_{cid:02d}_n{n_class}.png',
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Class {cid:2d} (n={n_class:3d}): saved summary with {len(top_feats)} features')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Per-class feature analysis')
    parser.add_argument('--mode', choices=['all', 'summary', 'both'], default='summary',
                        help='all=every feature×class PNG, summary=compact multi-panel per class')
    parser.add_argument('--classes', nargs='+', type=int, default=None,
                        help='Specific class IDs to generate (default: all)')
    parser.add_argument('--features', nargs='+', type=int, default=None,
                        help='Specific feature indices (default: all 279)')
    args = parser.parse_args()

    if args.mode in ('summary', 'both'):
        generate_summary()
    if args.mode in ('all', 'both'):
        generate_all(features=args.features, classes=args.classes)
