"""
Generate publication-quality figures for the CDS-OVR paper.
- Figure 5: Per-Class AUC grouped bar chart (3 protocols)
- Figure 6: Confusion matrix with improved contrast
- Figure 7: Accuracy vs training set size (mean + error bars, with 10-fold CV)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(OUT_DIR, "results_auc.json"), "r") as f:
    auc_data = json.load(f)

with open(os.path.join(OUT_DIR, "results_all_splits.json"), "r") as f:
    splits_data = json.load(f)

with open(os.path.join(OUT_DIR, "results_confusion.json"), "r") as f:
    conf_data = json.load(f)

CLASS_LABELS = {
    "1": "Normal (1)", "2": "Ischemic (2)", "3": "Old Ant. MI (3)",
    "4": "Old Inf. MI (4)", "5": "Sinus Tachy (5)", "6": "Sinus Brady (6)",
    "9": "LBBB (9)", "10": "RBBB (10)"
}
CLASS_ORDER = ["1", "2", "3", "4", "5", "6", "9", "10"]


def make_figure5():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8,
        'axes.linewidth': 0.6,
    })

    protocols = {"10-Fold CV": "10fold", "90/10 Split": "90_10", "60/40 Split": "60_40"}
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(CLASS_ORDER))
    width = 0.24
    offsets = [-width, 0, width]

    for idx, (label, key) in enumerate(protocols.items()):
        seed_data = auc_data[key]["seeds"]
        means = []
        err_lo = []
        err_hi = []
        for c in CLASS_ORDER:
            all_vals = [s["per_class_auc"][c] for s in seed_data]
            valid = [v for v in all_vals if v > 0]
            if len(valid) >= 2:
                m = np.mean(valid)
                s = np.std(valid, ddof=1)
            elif len(valid) == 1:
                m = valid[0]
                s = 0.0
            else:
                m = 0.0
                s = 0.0
            means.append(m)
            err_lo.append(min(s, m))
            err_hi.append(min(s, 1.0 - m))

        ax.bar(x + offsets[idx], means, width, label=label,
               color=colors[idx], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offsets[idx], means, yerr=[err_lo, err_hi], fmt="none",
                    ecolor="#333", elinewidth=0.7, capsize=2, capthick=0.5)

    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("Mean AUC-ROC")
    ax.set_xlabel("Class")
    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER],
                       rotation=30, ha="right")
    ax.legend(loc="lower left", frameon=True, edgecolor="#ccc",
              fancybox=False, framealpha=0.95)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.axhline(y=1.0, color='#ccc', linewidth=0.4, linestyle='-', zorder=0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_per_class_auc.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 5 saved: {path}")


def make_figure6():
    cm = np.array(conf_data["matrix"], dtype=int)
    n = len(CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    boundaries = [0, 0.5, 1.5, 3.5, 7.5, 15.5, 30.5, 60.5, 250]
    cmap = plt.cm.Blues
    norm = BoundaryNorm(boundaries, cmap.N)
    im = ax.imshow(cm, cmap=cmap, norm=norm, aspect="auto")

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            color = "white" if val > 15 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=11, color=color, fontweight=weight)

    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                     fill=False, edgecolor="#1565C0", linewidth=2))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Cls {c}" for c in CLASS_ORDER], fontsize=9)
    ax.set_yticklabels([CLASS_LABELS[c] for c in CLASS_ORDER], fontsize=9)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Count", fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_confusion_matrix.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 6 saved: {path}")


def make_figure7():
    # JSON keys don't always match display labels
    split_keys_json = ["50/50", "60/40", "70/30", "75/25", "80/19", "85/15", "90/9"]
    x_labels = ["50/50", "60/40", "70/30", "75/25", "80/20", "85/15", "90/10", "10-Fold"]

    rand_multi_mean = [splits_data[k]["summary"]["multi_mean"] for k in split_keys_json]
    rand_multi_std = [splits_data[k]["summary"]["multi_std"] for k in split_keys_json]
    rand_bin_mean = [splits_data[k]["summary"]["binary_mean"] for k in split_keys_json]
    rand_bin_std = [splits_data[k]["summary"]["binary_std"] for k in split_keys_json]

    # 10-fold CV from seed data
    tenfold = splits_data.get("10-fold CV", {}).get("summary", {})
    rand_multi_mean.append(tenfold.get("multi_mean", 84.66))
    rand_multi_std.append(tenfold.get("multi_std", 0.93))
    rand_bin_mean.append(tenfold.get("binary_mean", 87.72))
    rand_bin_std.append(tenfold.get("binary_std", 0.67))

    strat = {
        "50/50":  (77.21, 1.74, 81.37), "60/40":  (80.36, 2.94, 84.29),
        "70/30":  (81.09, 2.83, 85.70), "80/20":  (82.26, 5.26, 86.19),
        "90/10":  (85.56, 4.58, 89.56), "10-Fold": (85.32, 0.62, 88.56),
    }
    strat_multi = [strat.get(k, (None,))[0] for k in x_labels]
    strat_bin = [strat.get(k, (None, None, None))[2] for k in x_labels]

    # Remove None entries for 75/25 and 85/15
    x = np.arange(len(x_labels))
    strat_x = [i for i, k in enumerate(x_labels) if k in strat]
    strat_m_vals = [v for v in strat_multi if v is not None]
    strat_b_vals = [v for v in strat_bin if v is not None]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.errorbar(x, rand_multi_mean, yerr=rand_multi_std, fmt="o-",
                color="#1565C0", linewidth=1.8, markersize=6, capsize=4,
                label="Multiclass Mean (Random)", elinewidth=1)
    ax.errorbar(x, rand_bin_mean, yerr=rand_bin_std, fmt="s-",
                color="#E65100", linewidth=1.8, markersize=6, capsize=4,
                label="Binary Mean (Random)", elinewidth=1)

    ax.plot(strat_x, strat_m_vals, "o--", color="#1565C0", linewidth=1.2,
            markersize=5, alpha=0.6, label="Multiclass Mean (Stratified)")
    ax.plot(strat_x, strat_b_vals, "s--", color="#E65100", linewidth=1.2,
            markersize=5, alpha=0.6, label="Binary Mean (Stratified)")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel("Training/Testing Protocol", fontsize=11)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=11)
    ax.set_ylim(74, 94)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig7_accuracy_vs_split.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 7 saved: {path}")


def print_tables():
    print("\n" + "=" * 100)
    print("TABLE 8: Results Across 8 Evaluation Protocols (add 10-Fold CV as final row)")
    print("=" * 100)
    header = f"{'Split':<12} {'Multi Mean':>10} {'Multi Std':>10} {'Multi Best':>10} {'Multi Worst':>11} {'Bin Mean':>10} {'Bin Std':>9} {'Bin Best':>9} {'Bin Worst':>10}"
    print(header)
    print("-" * 103)
    json_keys = ["50/50", "60/40", "70/30", "75/25", "80/19", "85/15", "90/9"]
    display_keys = ["50/50", "60/40", "70/30", "75/25", "80/20", "85/15", "90/10"]
    for k, dk in zip(json_keys, display_keys):
        s = splits_data[k]["summary"]
        k = dk  # use display name
        print(f"{k:<12} {s['multi_mean']:>9.2f}% {s['multi_std']:>9.2f}% {s['multi_best']:>9.2f}% {s['multi_worst']:>10.2f}% {s['binary_mean']:>9.2f}% {s['binary_std']:>8.2f}% {s['binary_best']:>8.2f}% {s['binary_worst']:>9.2f}%")

    multi = [86.78, 84.13, 85.82, 84.62, 84.13, 83.89, 84.62, 83.41, 84.86, 84.38]
    bina = [88.70, 87.74, 88.70, 87.26, 87.74, 87.26, 87.74, 87.02, 88.22, 86.78]
    print(f"{'10-Fold CV':<12} {np.mean(multi):>9.2f}% {np.std(multi):>9.2f}% {max(multi):>9.2f}% {min(multi):>10.2f}% {np.mean(bina):>9.2f}% {np.std(bina):>8.2f}% {max(bina):>8.2f}% {min(bina):>9.2f}%")

    print("\n" + "=" * 100)
    print("TABLE 9: Stratified vs. Random Comparison (ALL available protocols)")
    print("=" * 100)
    rows = [
        ("10-fold CV", 84.66, 0.93, 85.32, 0.62, 87.72, 88.56),
        ("90/10",      86.19, 4.97, 85.56, 4.58, 88.57, 89.56),
        ("80/20",      85.47, 2.43, 82.26, 5.26, 87.98, 86.19),
        ("70/30",      81.36, 3.75, 81.09, 2.83, 85.12, 85.70),
        ("60/40",      80.18, 3.06, 80.36, 2.94, 84.49, 84.29),
        ("50/50",      79.38, 2.85, 77.21, 1.74, 83.03, 81.37),
    ]
    header = f"{'Protocol':<14} {'Random Multi (std)':<22} {'Strat Multi (std)':<22} {'Diff':<10} {'Rand Bin':>9} {'Strat Bin':>10}"
    print(header)
    print("-" * 90)
    for proto, rm, rs, sm, ss, rb, sb in rows:
        diff = sm - rm
        print(f"{proto:<14} {rm:.2f}% ({rs:.2f}%)       {sm:.2f}% ({ss:.2f}%)       {diff:>+.2f} pp  {rb:>8.2f}% {sb:>9.2f}%")
    print("\nNOTE: 85/15 and 75/25 stratified results not in provided data.")
    print("Run task_stratified.py with those splits to add them.")


if __name__ == "__main__":
    make_figure5()
    make_figure6()
    make_figure7()
    print_tables()
