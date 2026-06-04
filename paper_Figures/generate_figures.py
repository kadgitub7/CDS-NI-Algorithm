"""
Paper figure generation for CDS-NI hybrid arrhythmia classifier.

Generates 8 publication-quality figures from Enhanced_model and resource_analysis data.
Outputs saved to paper_Figures/figures/ as both PNG (300 dpi) and PDF.
"""

import csv
import re
import math
import os
from pathlib import Path
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import matplotlib.lines as mlines
import numpy as np

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ENHANCED_DIR = BASE_DIR / "Enhanced_model" / "results"
RESOURCE_DIR = BASE_DIR / "resource_analysis" / "results"
RESOURCE_CSV = RESOURCE_DIR / "comparison.csv"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# color palette
COLORS = {
    "cds": "#2196F3",
    "ann": "#FF9800",
    "rnn": "#4CAF50",
    "cascade_ann": "#E91E63",
    "cascade_rnn": "#9C27B0",
    "vote_ann": "#00BCD4",
    "vote_rnn": "#607D8B",
    "conf_ann": "#795548",
    "conf_rnn": "#CDDC39",
    "stack_ann": "#F44336",
    "stack_rnn": "#FF5722",
    "sel_ann": "#3F51B5",
    "sel_rnn": "#009688",
}

MODE_ORDER = [
    "CDS-only", "ANN-only", "RNN-only",
    "Cascade(ANN)", "Cascade(RNN)",
    "Vote(CDS+ANN)", "Vote(CDS+RNN)",
    "Confidence(CDS+ANN)", "Confidence(CDS+RNN)",
    "Stacked(CDS->ANN)", "Stacked(CDS->RNN)",
    "Selector(CDS+ANN)", "Selector(CDS+RNN)",
]

MODE_COLORS = [
    COLORS["cds"], COLORS["ann"], COLORS["rnn"],
    COLORS["cascade_ann"], COLORS["cascade_rnn"],
    COLORS["vote_ann"], COLORS["vote_rnn"],
    COLORS["conf_ann"], COLORS["conf_rnn"],
    COLORS["stack_ann"], COLORS["stack_rnn"],
    COLORS["sel_ann"], COLORS["sel_rnn"],
]

MODE_SHORT = [
    "CDS", "ANN", "RNN",
    "Cas-A", "Cas-R",
    "Vote-A", "Vote-R",
    "Conf-A", "Conf-R",
    "Stk-A", "Stk-R",
    "Sel-A", "Sel-R",
]


# ── Data Loading ────────────────────────────────────────────────────────────

def load_resource_csv():
    """Load the resource_analysis comparison CSV."""
    data = OrderedDict()
    with open(RESOURCE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row["Mode"]
            for k, v in row.items():
                if k != "Mode":
                    try:
                        row[k] = float(v)
                    except ValueError:
                        pass
            data[mode] = row
    return data


def parse_enhanced_file(filepath):
    """Parse an Enhanced_model result file for confusion matrix and decision sources."""
    text = filepath.read_text(encoding="utf-8")
    result = {}

    m = re.search(r"RESULTS?:?\s*(.+)", text)
    if m:
        result["mode"] = m.group(1).strip()

    # Confusion matrix
    cm_pattern = r"Actual Healthy\s+(\d+)\s+(\d+)\s*\n\s*Actual Unhealthy\s+(\d+)\s+(\d+)"
    cm = re.search(cm_pattern, text)
    if cm:
        result["cm"] = np.array([
            [int(cm.group(1)), int(cm.group(2))],
            [int(cm.group(3)), int(cm.group(4))]
        ])

    # Decision sources
    sources = []
    in_source = False
    for line in text.splitlines():
        if "DECISION SOURCE BREAKDOWN" in line:
            in_source = True
            continue
        if in_source and "PER-CLASS" in line:
            break
        if in_source:
            sm = re.match(r"\s{2}(\S[\w\-]+(?:\s*[\w\-]+)*)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%", line)
            if sm:
                sources.append({
                    "source": sm.group(1).strip(),
                    "count": int(sm.group(2)),
                    "pct": float(sm.group(3)),
                    "accuracy": float(sm.group(4)),
                })
    result["sources"] = sources

    # Per-class
    per_class = {}
    for cm_match in re.finditer(
        r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\S+)",
        text, re.MULTILINE
    ):
        cls_id = int(cm_match.group(1))
        per_class[cls_id] = {
            "correct": int(cm_match.group(2)),
            "total": int(cm_match.group(3)),
            "accuracy": float(cm_match.group(4)),
            "label": cm_match.group(5),
        }
    result["per_class"] = per_class

    return result


def load_all_enhanced():
    """Load all Enhanced_model result files."""
    files = {
        "CDS-only": "CDS-only.txt",
        "ANN-only": "ANN-only.txt",
        "RNN-only": "RNN-only.txt",
        "Cascade(ANN)": "Cascade_ANN.txt",
        "Cascade(RNN)": "Cascade_RNN.txt",
        "Vote(CDS+ANN)": "Vote_CDS_ANN.txt",
        "Vote(CDS+RNN)": "Vote_CDS_RNN.txt",
        "Confidence(CDS+ANN)": "Confidence_CDS_ANN.txt",
        "Confidence(CDS+RNN)": "Confidence_CDS_RNN.txt",
        "Stacked(CDS->ANN)": "Stacked_CDS_to_ANN.txt",
        "Stacked(CDS->RNN)": "Stacked_CDS_to_RNN.txt",
        "Selector(CDS+ANN)": "Selector_CDS_ANN.txt",
        "Selector(CDS+RNN)": "Selector_CDS_RNN.txt",
    }
    data = {}
    for mode, fname in files.items():
        fpath = ENHANCED_DIR / fname
        if fpath.exists():
            data[mode] = parse_enhanced_file(fpath)
    return data


def wilson_ci(correct, total, z=1.96):
    if total == 0:
        return (0, 0, 0)
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return (max(0, center - margin), p, min(1, center + margin))


# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
resource_data = load_resource_csv()
enhanced_data = load_all_enhanced()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: System Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Fig. 1: Hybrid CDS + ML Arrhythmia Classification Pipeline", fontsize=13, fontweight="bold", pad=15)

    box_kw = dict(boxstyle="round,pad=0.3", linewidth=1.5)

    def draw_box(x, y, w, h, text, color, fontsize=9, textcolor="white"):
        rect = FancyBboxPatch((x, y), w, h, **box_kw, facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor)

    def draw_arrow(x1, y1, x2, y2, text="", color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, text, ha="center", va="bottom", fontsize=7,
                    fontstyle="italic", color=color)

    # Input
    draw_box(0.3, 4.8, 2.2, 0.8, "ECG Signal\n(452 Users, UCI)", "#78909C", 9)

    # Feature extraction
    draw_box(3.2, 4.8, 2.5, 0.8, "Feature Extraction\n(279 features)", "#546E7A", 9)
    draw_arrow(2.5, 5.2, 3.2, 5.2, "", "black")

    # Algorithm 3 pruning
    draw_box(6.4, 4.8, 2.5, 0.8, "Alg. 3: Feature\nPruning (~30 kept)", "#455A64", 9)
    draw_arrow(5.7, 5.2, 6.4, 5.2, "", "black")

    # CDS Algorithm 4
    draw_box(1.0, 2.8, 2.8, 1.2, "CDS Algorithm 4\nRange Check\n+ Assurance Factor", "#2196F3", 9)
    draw_arrow(7.65, 4.8, 2.4, 4.0, "Pruned\nfeatures", "#37474F")

    # Decision paths from CDS
    draw_box(0.2, 0.8, 1.5, 0.8, "HEALTHY\n(AF > threshold)", "#4CAF50", 8)
    draw_box(1.9, 0.8, 1.5, 0.8, "ALARM\n(Out of range)", "#F44336", 8)
    draw_box(3.6, 0.8, 1.8, 0.8, "SCREENING\n(Uncertain)", "#FF9800", 8)

    draw_arrow(1.6, 2.8, 0.95, 1.6, "Confident\nhealthy", "#4CAF50")
    draw_arrow(2.4, 2.8, 2.65, 1.6, "Feature\nalarm", "#F44336")
    draw_arrow(3.4, 2.8, 4.5, 1.6, "Uncertain", "#FF9800")

    # ML branch
    draw_box(5.8, 2.8, 2.8, 1.2, "ML Classifier\n(ANN or RNN)\nTrained on residuals", "#E91E63", 9)
    draw_arrow(5.4, 1.2, 5.8, 3.2, "Route to ML", "#E91E63")

    # ML outputs
    draw_box(6.0, 0.8, 1.2, 0.8, "Healthy", "#4CAF50", 8)
    draw_box(7.4, 0.8, 1.4, 0.8, "Unhealthy", "#F44336", 8)
    draw_arrow(6.6, 2.8, 6.6, 1.6, "", "#E91E63")
    draw_arrow(7.8, 2.8, 8.1, 1.6, "", "#E91E63")

    # Final decision
    draw_box(3.0, -0.2, 4.0, 0.6, "Final Classification Decision", "#1A237E", 10)
    draw_arrow(0.95, 0.8, 4.0, 0.4, "", "#666")
    draw_arrow(2.65, 0.8, 4.5, 0.4, "", "#666")
    draw_arrow(6.6, 0.8, 5.8, 0.4, "", "#666")
    draw_arrow(8.1, 0.8, 6.5, 0.4, "", "#666")

    # Annotations
    ax.text(0.5, 2.2, "~24.8% of users", fontsize=7, ha="center", color="#4CAF50", fontstyle="italic")
    ax.text(2.4, 2.2, "~53.1% of users", fontsize=7, ha="center", color="#F44336", fontstyle="italic")
    ax.text(4.5, 2.2, "~22.1% of users", fontsize=7, ha="center", color="#FF9800", fontstyle="italic")

    save_fig(fig, "fig1_system_architecture")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Accuracy Bar Chart with CIs
# ═══════════════════════════════════════════════════════════════════════════

def fig2_accuracy_comparison():
    fig, ax = plt.subplots(figsize=(12, 5))

    accuracies = []
    ci_lo = []
    ci_hi = []
    for mode in MODE_ORDER:
        acc = float(resource_data[mode]["Accuracy%"])
        n = int(float(resource_data[mode]["N_Users"]))
        correct = round(acc * n / 100)
        lo, mid, hi = wilson_ci(correct, n)
        accuracies.append(acc)
        ci_lo.append(acc - lo * 100)
        ci_hi.append(hi * 100 - acc)

    x = np.arange(len(MODE_ORDER))
    bars = ax.bar(x, accuracies, color=MODE_COLORS, edgecolor="black", linewidth=0.5, width=0.7)
    ax.errorbar(x, accuracies, yerr=[ci_lo, ci_hi], fmt="none", ecolor="black",
                capsize=3, capthick=1, elinewidth=1)

    # CDS-only reference line
    cds_acc = float(resource_data["CDS-only"]["Accuracy%"])
    ax.axhline(y=cds_acc, color="#2196F3", linestyle="--", linewidth=1, alpha=0.7, label=f"CDS-only ({cds_acc:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(MODE_SHORT, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fig. 2: Classification Accuracy Comparison (452 Users, LOOCV)", fontweight="bold")
    ax.set_ylim(30, 85)
    ax.legend(loc="lower right")

    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci_hi[i] + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, "fig2_accuracy_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Confusion Matrices
# ═══════════════════════════════════════════════════════════════════════════

def fig3_confusion_matrices():
    models = ["CDS-only", "ANN-only", "Cascade(ANN)"]
    titles = ["CDS-only (74.6%)", "ANN-only (70.6%)", "Cascade(ANN) (74.8%)"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Fig. 3: Binary Confusion Matrices — CDS vs ANN vs Best Hybrid", fontweight="bold", y=1.02)

    for idx, (mode, title) in enumerate(zip(models, titles)):
        ax = axes[idx]
        cm = enhanced_data[mode]["cm"]
        total = cm.sum()

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=250)

        labels = ["Healthy", "Unhealthy"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100
                color = "white" if val > 150 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    plt.tight_layout()
    save_fig(fig, "fig3_confusion_matrices")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Radar / Spider Chart
# ═══════════════════════════════════════════════════════════════════════════

def fig4_radar_chart():
    modes_to_plot = ["CDS-only", "ANN-only", "Cascade(ANN)", "Cascade(RNN)"]
    colors_radar = [COLORS["cds"], COLORS["ann"], COLORS["cascade_ann"], COLORS["cascade_rnn"]]

    metrics = ["Accuracy", "Sensitivity", "Specificity", "F1 Score", "Speed\n(inv. time)", "Efficiency\n(inv. memory)"]
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    max_time = max(float(resource_data[m]["Avg_Time_Per_User_ms"]) for m in modes_to_plot)
    max_mem = max(float(resource_data[m]["Peak_Memory_MB"]) for m in modes_to_plot)

    for mode, color in zip(modes_to_plot, colors_radar):
        rd = resource_data[mode]
        values = [
            float(rd["Accuracy%"]),
            float(rd["Sensitivity%"]),
            float(rd["Specificity%"]),
            float(rd["F1%"]),
            (1 - float(rd["Avg_Time_Per_User_ms"]) / max_time) * 100,
            (1 - float(rd["Peak_Memory_MB"]) / max_mem) * 100,
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=mode, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7, color="grey")
    ax.set_title("Fig. 4: Multi-Objective Comparison\n(Top 4 Modes)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    save_fig(fig, "fig4_radar_chart")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Decision Source Pie Charts
# ═══════════════════════════════════════════════════════════════════════════

def fig5_decision_sources():
    modes_to_plot = ["CDS-only", "Cascade(ANN)", "Cascade(RNN)",
                     "Confidence(CDS+ANN)", "Vote(CDS+ANN)"]

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    fig.suptitle("Fig. 5: Decision Source Distribution — How Often Each Mode Uses CDS vs ML",
                 fontweight="bold", y=1.05)

    source_colors = {
        "CDS-ALARM": "#F44336", "CDS-HEALTHY": "#4CAF50", "CDS-SCREENING": "#FF9800",
        "ANN": "#E91E63", "RNN": "#9C27B0",
        "VOTE-AGREE-HEALTHY": "#4CAF50", "VOTE-AGREE-UNHEALTHY": "#F44336",
        "VOTE-TIE-CDS": "#2196F3",
        "CONF-CDS": "#2196F3", "CONF-ANN": "#E91E63", "CONF-HEALTHY": "#4CAF50",
        "SEL-CDS": "#FF9800", "SEL-CDS-ALARM": "#F44336", "SEL-CDS-HEALTHY": "#4CAF50",
    }

    for idx, mode in enumerate(modes_to_plot):
        ax = axes[idx]
        sources = enhanced_data[mode]["sources"]
        if not sources:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        labels = [s["source"] for s in sources]
        sizes = [s["count"] for s in sources]
        colors = [source_colors.get(s["source"], "#9E9E9E") for s in sources]

        short_labels = []
        for s in sources:
            name = s["source"]
            name = name.replace("CDS-", "").replace("VOTE-", "V-").replace("CONF-", "C-").replace("SEL-", "S-")
            short_labels.append(f"{name}\n{s['count']}")

        wedges, texts = ax.pie(sizes, labels=short_labels, colors=colors,
                               startangle=90, textprops={"fontsize": 7})
        ax.set_title(mode.replace("(CDS+", "(").replace("Confidence", "Conf."),
                     fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "fig5_decision_sources")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Computational Cost Bar Chart
# ═══════════════════════════════════════════════════════════════════════════

def fig6_computational_cost():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig. 6: Computational Cost and Inference Time Comparison", fontweight="bold", y=1.02)

    # Panel A: Multiplications per user (log scale)
    mults = []
    for mode in MODE_ORDER:
        mults.append(float(resource_data[mode]["Avg_Mults_Per_User"]))

    x = np.arange(len(MODE_ORDER))
    bars = ax1.bar(x, mults, color=MODE_COLORS, edgecolor="black", linewidth=0.5, width=0.7)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODE_SHORT, rotation=45, ha="right")
    ax1.set_ylabel("Avg. Multiplications / User (log scale)")
    ax1.set_title("(a) Computational Cost", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, which="both")

    for bar, val in zip(bars, mults):
        if val > 1e6:
            label = f"{val:.1e}"
        else:
            label = f"{val:,.0f}"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                 label, ha="center", va="bottom", fontsize=6, rotation=60)

    # Panel B: Time per user
    times = []
    cds_times = []
    ml_times = []
    for mode in MODE_ORDER:
        rd = resource_data[mode]
        t_cds = float(rd["Avg_CDS_Train_ms"]) + float(rd["Avg_CDS_Predict_ms"])
        t_ml = float(rd["Avg_ML_Train_ms"]) + float(rd["Avg_ML_Predict_ms"])
        cds_times.append(t_cds)
        ml_times.append(t_ml)
        times.append(t_cds + t_ml)

    ax2.bar(x, cds_times, color="#2196F3", edgecolor="black", linewidth=0.5, width=0.7, label="CDS time")
    ax2.bar(x, ml_times, bottom=cds_times, color="#E91E63", edgecolor="black", linewidth=0.5, width=0.7, label="ML time")
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODE_SHORT, rotation=45, ha="right")
    ax2.set_ylabel("Avg. Time / User (ms)")
    ax2.set_title("(b) Inference Time Breakdown", fontweight="bold")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    for i, t in enumerate(times):
        ax2.text(x[i], t + 50, f"{t:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "fig6_computational_cost")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Per-Class Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig7_perclass_heatmap():
    class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16]
    class_labels = [
        "Healthy\n(245)", "D2\n(44)", "D3\n(15)", "D4\n(15)", "D5\n(13)",
        "D6\n(25)", "D7\n(3)", "D8\n(2)", "D9\n(9)", "D10\n(50)",
        "D14\n(4)", "D15\n(5)", "D16\n(22)"
    ]

    matrix = []
    for mode in MODE_ORDER:
        row = []
        pc = enhanced_data[mode]["per_class"]
        for cid in class_ids:
            if cid in pc:
                row.append(pc[cid]["accuracy"])
            else:
                row.append(0.0)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(class_ids)))
    ax.set_xticklabels(class_labels, fontsize=8)
    ax.set_yticks(np.arange(len(MODE_ORDER)))
    ax.set_yticklabels(MODE_SHORT, fontsize=9)

    for i in range(len(MODE_ORDER)):
        for j in range(len(class_ids)):
            val = matrix[i, j]
            color = "white" if val < 40 or val > 90 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7,
                    fontweight="bold", color=color)

    ax.set_title("Fig. 7: Per-Class Accuracy Heatmap (% correct per disease class)",
                 fontweight="bold", pad=10)
    ax.set_xlabel("Disease Class (sample count)")
    ax.set_ylabel("Classification Mode")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)")

    plt.tight_layout()
    save_fig(fig, "fig7_perclass_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Device Architecture Block Diagram
# ═══════════════════════════════════════════════════════════════════════════

def fig8_device_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Fig. 8: Wearable Device Architecture — Embedded vs FPGA Implementation",
                 fontweight="bold", y=0.98)

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    box_kw = dict(boxstyle="round,pad=0.25", linewidth=1.2)

    def draw_box(ax, x, y, w, h, text, color, fontsize=8, textcolor="white"):
        rect = FancyBboxPatch((x, y), w, h, **box_kw, facecolor=color, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, linespacing=1.3)

    def draw_arrow_simple(ax, x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.15, my, text, fontsize=6, fontstyle="italic", color="#555")

    # ── Track A: Embedded ──
    ax = axes[0]
    ax.set_title("Track A: Embedded (ARM Cortex-M4F)", fontsize=11, fontweight="bold", pad=10)

    draw_box(ax, 0.3, 8.0, 2.5, 1.2, "MAX86150\nECG + PPG", "#0D47A1", 8)
    draw_box(ax, 0.3, 6.3, 2.5, 1.2, "MAX30205\nTemperature", "#1565C0", 8)
    draw_box(ax, 0.3, 4.6, 2.5, 1.2, "BMI160\nIMU", "#1976D2", 8)

    draw_box(ax, 4.0, 4.0, 5.5, 5.5, "", "#E3F2FD", textcolor="black")
    ax.text(6.75, 9.2, "Arduino Nano 33 BLE Sense", fontsize=9, ha="center",
            fontweight="bold", color="#0D47A1")
    ax.text(6.75, 8.7, "nRF52840 | 64 MHz | 256 KB RAM", fontsize=7, ha="center", color="#555")

    draw_box(ax, 4.5, 7.5, 4.5, 0.8, "DSP Filters + Feature Extract", "#42A5F5", 8)
    draw_box(ax, 4.5, 6.2, 2.0, 0.8, "CDS Alg. 4\nLookup", "#2196F3", 8)
    draw_box(ax, 7.0, 6.2, 2.0, 0.8, "ANN Forward\nPass", "#E91E63", 8)
    draw_box(ax, 4.5, 4.8, 4.5, 0.8, "Cascade Decision Logic", "#1A237E", 8)

    draw_arrow_simple(ax, 2.8, 8.6, 4.5, 7.9, "I2C")
    draw_arrow_simple(ax, 2.8, 6.9, 4.5, 7.5, "I2C")
    draw_arrow_simple(ax, 2.8, 5.2, 4.5, 5.2, "SPI")
    draw_arrow_simple(ax, 6.75, 7.5, 5.5, 7.0)
    draw_arrow_simple(ax, 6.75, 7.5, 8.0, 7.0)
    draw_arrow_simple(ax, 5.5, 6.2, 6.0, 5.6)
    draw_arrow_simple(ax, 8.0, 6.2, 7.7, 5.6)

    draw_box(ax, 4.0, 2.5, 2.5, 1.0, "BLE 5.0\nTransmit", "#00897B", 8)
    draw_box(ax, 7.0, 2.5, 2.5, 1.0, "Phone\nApp", "#00695C", 8)
    draw_arrow_simple(ax, 6.75, 4.8, 5.25, 3.5)
    draw_arrow_simple(ax, 6.5, 3.0, 7.0, 3.0, "BLE")

    draw_box(ax, 0.3, 2.5, 2.5, 1.0, "LiPo 500 mAh\n~6 days", "#F57F17", 8)
    draw_arrow_simple(ax, 2.8, 3.0, 4.0, 3.0, "3.7V")

    ax.text(5.0, 1.8, "Latency: ~1 ms | Power: ~3.5 mA | BOM: ~$70",
            fontsize=8, ha="center", color="#333", fontstyle="italic")

    # ── Track B: FPGA ──
    ax = axes[1]
    ax.set_title("Track B: FPGA (Parallel Hardware)", fontsize=11, fontweight="bold", pad=10)

    draw_box(ax, 0.3, 8.0, 2.5, 1.2, "MAX86150\nECG + PPG", "#4A148C", 8)
    draw_box(ax, 0.3, 6.3, 2.5, 1.2, "MAX30205\nTemperature", "#6A1B9A", 8)
    draw_box(ax, 0.3, 4.6, 2.5, 1.2, "BMI160\nIMU", "#7B1FA2", 8)

    draw_box(ax, 4.0, 4.0, 5.5, 5.5, "", "#F3E5F5", textcolor="black")
    ax.text(6.75, 9.2, "Tang Nano 9K / iCE40 UP5K", fontsize=9, ha="center",
            fontweight="bold", color="#4A148C")
    ax.text(6.75, 8.7, "8,640 LUTs | 44% util. | 12 MHz", fontsize=7, ha="center", color="#555")

    draw_box(ax, 4.5, 7.5, 4.5, 0.8, "IIR Filters + R-Peak (parallel)", "#AB47BC", 8)
    draw_box(ax, 4.5, 6.2, 2.0, 0.8, "CDS Range\nCheck (1 clk)", "#9C27B0", 8)
    draw_box(ax, 7.0, 6.2, 2.0, 0.8, "ANN MAC\n(1504 clks)", "#E91E63", 8)
    draw_box(ax, 4.5, 4.8, 4.5, 0.8, "Cascade Decision (1 clk comb.)", "#4A148C", 8)

    draw_arrow_simple(ax, 2.8, 8.6, 4.5, 7.9, "I2C")
    draw_arrow_simple(ax, 2.8, 6.9, 4.5, 7.5, "I2C")
    draw_arrow_simple(ax, 2.8, 5.2, 4.5, 5.2, "SPI")
    draw_arrow_simple(ax, 6.75, 7.5, 5.5, 7.0)
    draw_arrow_simple(ax, 6.75, 7.5, 8.0, 7.0)
    draw_arrow_simple(ax, 5.5, 6.2, 6.0, 5.6)
    draw_arrow_simple(ax, 8.0, 6.2, 7.7, 5.6)

    draw_box(ax, 4.0, 2.5, 2.5, 1.0, "nRF52832\nBLE Module", "#00897B", 8)
    draw_box(ax, 7.0, 2.5, 2.5, 1.0, "Phone\nApp", "#00695C", 8)
    draw_arrow_simple(ax, 6.75, 4.8, 5.25, 3.5)
    draw_arrow_simple(ax, 6.5, 3.0, 7.0, 3.0, "UART")

    draw_box(ax, 0.3, 2.5, 2.5, 1.0, "LiPo 500 mAh\n~9.5 days", "#F57F17", 8)
    draw_arrow_simple(ax, 2.8, 3.0, 4.0, 3.0, "3.7V")

    ax.text(5.0, 1.8, "Latency: ~128 us | Power: ~2.2 mA | BOM: ~$66",
            fontsize=8, ha="center", color="#333", fontstyle="italic")

    plt.tight_layout()
    save_fig(fig, "fig8_device_architecture")


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════

def save_fig(fig, name):
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, format="png")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    print(f"  Saved: {png_path.name} + {pdf_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\nGenerating paper figures...\n")

    print("Fig 1: System architecture...")
    fig1_architecture()

    print("Fig 2: Accuracy comparison bar chart...")
    fig2_accuracy_comparison()

    print("Fig 3: Confusion matrices...")
    fig3_confusion_matrices()

    print("Fig 4: Radar/spider chart...")
    fig4_radar_chart()

    print("Fig 5: Decision source pie charts...")
    fig5_decision_sources()

    print("Fig 6: Computational cost...")
    fig6_computational_cost()

    print("Fig 7: Per-class heatmap...")
    fig7_perclass_heatmap()

    print("Fig 8: Device architecture...")
    fig8_device_architecture()

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done.")
