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
    "tgll_t1": "#1B5E20",
    "tgll_t12": "#2E7D32",
    "tgll_t123": "#388E3C",
}

MODE_ORDER = [
    "CDS-only", "ANN-only", "RNN-only",
    "Cascade(ANN)", "Cascade(RNN)",
    "Vote(CDS+ANN)", "Vote(CDS+RNN)",
    "Confidence(CDS+ANN)", "Confidence(CDS+RNN)",
    "Stacked(CDS->ANN)", "Stacked(CDS->RNN)",
    "Selector(CDS+ANN)", "Selector(CDS+RNN)",
    "AlarmRefine(ANN)", "AFGated(ANN)", "DisagreeRules(ANN)",
    "AlarmSpecialist(ANN)", "TripleCascade",
    "TGLLNet-T1+CDS", "TGLLNet-T12+CDS", "TGLLNet-T123+CDS",
    "TGLLNet-T1-only", "TGLLNet-T12-only", "TGLLNet-T123-only",
]

MODE_COLORS = [
    COLORS["cds"], COLORS["ann"], COLORS["rnn"],
    COLORS["cascade_ann"], COLORS["cascade_rnn"],
    COLORS["vote_ann"], COLORS["vote_rnn"],
    COLORS["conf_ann"], COLORS["conf_rnn"],
    COLORS["stack_ann"], COLORS["stack_rnn"],
    COLORS["sel_ann"], COLORS["sel_rnn"],
    "#D50000", "#00C853", "#2962FF", "#AA00FF", "#FF6D00",
    COLORS["tgll_t1"], COLORS["tgll_t12"], COLORS["tgll_t123"],
    "#66BB6A", "#81C784", "#A5D6A7",  # standalone tgllnet (lighter greens)
]

MODE_SHORT = [
    "CDS", "ANN", "RNN",
    "Cas-A", "Cas-R",
    "Vote-A", "Vote-R",
    "Conf-A", "Conf-R",
    "Stk-A", "Stk-R",
    "Sel-A", "Sel-R",
    "ARef", "AFG", "DRule", "ASpec", "TriCas",
    "TGL-1", "TGL-12", "TGL-123",
    "TGL1s", "TGL12s", "TGL123s",
]


# ── Data Loading ────────────────────────────────────────────────────────────

def _parse_resource_txt(filepath):
    """Parse a single resource_analysis .txt file into a dict matching the CSV columns."""
    text = filepath.read_text(encoding="utf-8")
    row = {}

    m = re.search(r"(?:RESOURCE ANALYSIS|RESULTS):\s*(.+)", text)
    row["Mode"] = m.group(1).strip() if m else filepath.stem

    m = re.search(r"Users evaluated:\s*(\d+)", text)
    row["N_Users"] = float(m.group(1)) if m else 0

    patterns = {
        "Accuracy%": r"Accuracy:\s*([\d.]+)%",
        "Sensitivity%": r"Sensitivity.*?:\s*([\d.]+)%",
        "Specificity%": r"Specificity:\s*([\d.]+)%",
        "PPV%": r"Precision \(PPV\):\s*([\d.]+)%",
        "NPV%": r"Neg Pred Value.*?:\s*([\d.]+)%",
        "F1%": r"F1 Score:\s*([\d.]+)%",
        "Total_Time_s": r"Total wall-clock:\s*([\d.]+)s",
        "Avg_Time_Per_User_ms": r"Avg per user:\s*([\d.]+)ms",
        "Avg_CDS_Train_ms": r"Avg CDS train:\s*([\d.]+)ms",
        "Avg_CDS_Predict_ms": r"Avg CDS predict:\s*([\d.]+)ms",
        "Avg_ML_Train_ms": r"Avg ML train:\s*([\d.]+)ms",
        "Avg_ML_Predict_ms": r"Avg ML predict:\s*([\d.]+)ms",
        "Peak_Memory_MB": r"Peak memory:\s*([\d.]+)\s*MB",
        "Avg_Memory_MB": r"Avg memory per fold:\s*([\d.]+)\s*MB",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        row[key] = float(m.group(1)) if m else 0.0

    # Handle total time from Enhanced_model format too
    if row["Total_Time_s"] == 0:
        m = re.search(r"Total time:\s*([\d.]+)s", text)
        if m:
            row["Total_Time_s"] = float(m.group(1))
        m = re.search(r"\((\d+)ms/user\)", text)
        if m:
            row["Avg_Time_Per_User_ms"] = float(m.group(1))

    for key, pat in [
        ("Total_Multiplications", r"Total multiplications:\s*([\d,]+)"),
        ("CDS_Mults", r"CDS total:\s*([\d,]+)"),
        ("ML_Train_Mults", r"ML training total:\s*([\d,]+)"),
        ("ML_Infer_Mults", r"ML inference total:\s*([\d,]+)"),
    ]:
        m = re.search(pat, text)
        row[key] = float(m.group(1).replace(",", "")) if m else 0.0

    m = re.search(r"Avg per user:\s*([\d,]+)\n", text)
    row["Avg_Mults_Per_User"] = float(m.group(1).replace(",", "")) if m else 0.0

    return row


def load_resource_csv():
    """Load resource data by parsing individual .txt result files directly."""
    data = OrderedDict()

    # Scan resource_analysis/results/
    if RESOURCE_DIR.exists():
        print(f"  Scanning {RESOURCE_DIR}")
        for fpath in sorted(RESOURCE_DIR.glob("*.txt")):
            if fpath.name in ("comparison.txt",):
                continue
            row = _parse_resource_txt(fpath)
            if row["N_Users"] > 0:
                data[row["Mode"]] = row
                print(f"    {row['Mode']:30s}  acc={row['Accuracy%']:.1f}%")

    # Scan Enhanced_model/results/ for any modes not already found
    if ENHANCED_DIR.exists():
        print(f"  Scanning {ENHANCED_DIR}")
        for fpath in sorted(ENHANCED_DIR.glob("*.txt")):
            if fpath.name in ("comparison.txt",):
                continue
            row = _parse_resource_txt(fpath)
            if row["N_Users"] > 0 and row["Mode"] not in data:
                data[row["Mode"]] = row
                print(f"    {row['Mode']:30s}  acc={row['Accuracy%']:.1f}%")

    print(f"  Loaded resource data for {len(data)} modes")
    return data


def parse_enhanced_file(filepath):
    """Parse an Enhanced_model or resource_analysis result file for confusion matrix, sources, per-class."""
    text = filepath.read_text(encoding="utf-8")
    result = {}

    m = re.search(r"(?:RESULTS|RESOURCE ANALYSIS):\s*(.+)", text)
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
    else:
        # Derive from per-class data if no explicit confusion matrix
        result["cm"] = None

    # Decision sources — extract from DECISION SOURCE BREAKDOWN section
    sources = []
    in_source_section = False
    for line in text.splitlines():
        if "DECISION SOURCE BREAKDOWN" in line:
            in_source_section = True
            continue
        if in_source_section and ("PER-CLASS" in line or "MISCLASSIFICATION" in line):
            break
        if not in_source_section:
            continue
        # Skip header/separator lines
        if "Source" in line and "Count" in line:
            continue
        if line.strip().startswith("─") or line.strip().startswith("-") or not line.strip():
            continue
        # Match: "  ANN                                100    22.1%      78.0%"
        sm = re.match(r"^\s{2,}(\S[\S ]*\S)\s{2,}(\d+)\s+([\d.]+)%\s+([\d.]+)%", line)
        if sm:
            sources.append({
                "source": sm.group(1).strip(),
                "count": int(sm.group(2)),
                "pct": float(sm.group(3)),
                "accuracy": float(sm.group(4)),
            })
            continue
        # Match resource_analysis: "  SOURCE   :   COUNT  ( PCT%)  acc=ACC%"
        sm = re.match(r"^\s{2,}(\S[\S ]*\S)\s*:\s*(\d+)\s+\(\s*([\d.]+)%\)\s+acc=([\d.]+)%", line)
        if sm:
            sources.append({
                "source": sm.group(1).strip(),
                "count": int(sm.group(2)),
                "pct": float(sm.group(3)),
                "accuracy": float(sm.group(4)),
            })
    result["sources"] = sources

    # Per-class — try multiple formats
    per_class = {}
    # Format 1: "  Class  1 (Healthy     ): 231/245 (94.3%)"  (resource_analysis)
    for cm_match in re.finditer(
        r"Class\s+(\d+)\s+\((\S[\w\-]*)\s*\):\s*(\d+)/(\d+)\s+\(([\d.]+)%\)",
        text
    ):
        cls_id = int(cm_match.group(1))
        per_class[cls_id] = {
            "correct": int(cm_match.group(3)),
            "total": int(cm_match.group(4)),
            "accuracy": float(cm_match.group(5)),
            "label": cm_match.group(2).strip(),
        }
    # Format 2: "     1       231     245      94.3%  Healthy"  (Enhanced_model)
    if not per_class:
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

    # Derive confusion matrix from per-class if missing
    if result["cm"] is None and per_class:
        h = per_class.get(1, {"correct": 0, "total": 0})
        tn = h["correct"]
        fp = h["total"] - h["correct"]
        dis_correct = sum(v["correct"] for k, v in per_class.items() if k != 1)
        dis_total = sum(v["total"] for k, v in per_class.items() if k != 1)
        tp = dis_correct
        fn = dis_total - dis_correct
        result["cm"] = np.array([[tn, fp], [fn, tp]])

    return result


def load_all_enhanced():
    """Load all result files from both Enhanced_model and resource_analysis."""
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
        "AlarmRefine(ANN)": "AlarmRefine_ANN.txt",
        "AFGated(ANN)": "AFGated_ANN.txt",
        "DisagreeRules(ANN)": "DisagreeRules_ANN.txt",
        "AlarmSpecialist(ANN)": "AlarmSpecialist_ANN.txt",
        "TripleCascade": "TripleCascade.txt",
        "TGLLNet-T1+CDS": "TGLLNet-T1_CDS.txt",
        "TGLLNet-T12+CDS": "TGLLNet-T12_CDS.txt",
        "TGLLNet-T123+CDS": "TGLLNet-T123_CDS.txt",
        "TGLLNet-T1-only": "TGLLNet-T1-only.txt",
        "TGLLNet-T12-only": "TGLLNet-T12-only.txt",
        "TGLLNet-T123-only": "TGLLNet-T123-only.txt",
    }
    data = {}
    # Try Enhanced_model first, then resource_analysis as fallback
    for mode, fname in files.items():
        for search_dir in (ENHANCED_DIR, RESOURCE_DIR):
            fpath = search_dir / fname
            if fpath.exists() and mode not in data:
                parsed = parse_enhanced_file(fpath)
                if parsed.get("per_class") or parsed.get("cm") is not None:
                    data[mode] = parsed
                    break
    print(f"  Loaded enhanced data for {len(data)} modes")
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

def _available_modes():
    """Return only MODE_ORDER entries present in the resource CSV data."""
    return [m for m in MODE_ORDER if m in resource_data]


def fig2_accuracy_comparison():
    fig, ax = plt.subplots(figsize=(14, 5.5))

    avail = _available_modes()
    colors = [MODE_COLORS[MODE_ORDER.index(m)] for m in avail]
    shorts = [MODE_SHORT[MODE_ORDER.index(m)] for m in avail]

    accuracies = []
    ci_lo = []
    ci_hi = []
    for mode in avail:
        acc = float(resource_data[mode]["Accuracy%"])
        n = int(float(resource_data[mode]["N_Users"]))
        correct = round(acc * n / 100)
        lo, mid, hi = wilson_ci(correct, n)
        accuracies.append(acc)
        ci_lo.append(acc - lo * 100)
        ci_hi.append(hi * 100 - acc)

    x = np.arange(len(avail))
    bars = ax.bar(x, accuracies, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax.errorbar(x, accuracies, yerr=[ci_lo, ci_hi], fmt="none", ecolor="black",
                capsize=3, capthick=1, elinewidth=1)

    # CDS-only reference line
    cds_acc = float(resource_data["CDS-only"]["Accuracy%"])
    ax.axhline(y=cds_acc, color="#2196F3", linestyle="--", linewidth=1, alpha=0.7, label=f"CDS-only ({cds_acc:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(shorts, rotation=45, ha="right")
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
    # Auto-select: best hybrid, best standalone, and CDS-only (always shown)
    available = {m: d for m, d in enhanced_data.items() if d.get("cm") is not None}
    if not available:
        print("  Skip fig3: no confusion matrix data")
        return

    # Rank by accuracy from resource_data
    def _acc(mode):
        if mode in resource_data:
            return float(resource_data[mode].get("Accuracy%", 0))
        return 0.0

    standalones = [m for m in available if "-only" in m]
    hybrids = [m for m in available if "-only" not in m]

    # Pick: CDS-only (baseline), best standalone ML, best hybrid — up to 4
    picks = []
    if "CDS-only" in available:
        picks.append("CDS-only")
    best_ml = sorted([m for m in standalones if m != "CDS-only"], key=_acc, reverse=True)
    if best_ml:
        picks.append(best_ml[0])
    best_hybrids = sorted(hybrids, key=_acc, reverse=True)
    for h in best_hybrids:
        if h not in picks:
            picks.append(h)
            if len(picks) >= 4:
                break

    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Fig. 3: Binary Confusion Matrices (auto-selected top modes)", fontweight="bold", y=1.02)

    for idx, mode in enumerate(picks):
        ax = axes[idx]
        cm = available[mode]["cm"]
        total = cm.sum()
        acc = _acc(mode)

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(250, cm.max()))

        labels = ["Healthy", "Unhealthy"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(f"{mode}\n({acc:.1f}%)", fontsize=10, fontweight="bold")

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100
                color = "white" if val > cm.max() * 0.6 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    plt.tight_layout()
    save_fig(fig, "fig3_confusion_matrices")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Radar / Spider Chart
# ═══════════════════════════════════════════════════════════════════════════

def fig4_radar_chart():
    # Auto-select: 3 standalone + top 2 hybrids by accuracy (max 5 total)
    avail = [m for m in resource_data if float(resource_data[m].get("Accuracy%", 0)) > 0]
    if len(avail) < 2:
        print("  Skip fig4: need at least 2 modes with data")
        return

    def _acc(m):
        return float(resource_data[m].get("Accuracy%", 0))

    standalones = sorted([m for m in avail if "-only" in m], key=_acc, reverse=True)
    hybrids = sorted([m for m in avail if "-only" not in m], key=_acc, reverse=True)

    modes_to_plot = []
    for m in standalones[:3]:
        modes_to_plot.append(m)
    for m in hybrids:
        if m not in modes_to_plot:
            modes_to_plot.append(m)
            if len(modes_to_plot) >= 5:
                break

    all_colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0",
                  "#00BCD4", "#795548", "#F44336"]

    # Check if time/memory data exists
    has_resources = any(float(resource_data[m].get("Avg_Time_Per_User_ms", 0)) > 0 for m in modes_to_plot)

    if has_resources:
        metrics = ["Accuracy", "Sensitivity", "Specificity", "F1 Score", "Speed\n(inv. time)", "Efficiency\n(inv. memory)"]
    else:
        metrics = ["Accuracy", "Sensitivity", "Specificity", "F1 Score", "PPV"]

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    if has_resources:
        max_time = max(max(float(resource_data[m].get("Avg_Time_Per_User_ms", 1)) for m in modes_to_plot), 1)
        max_mem = max(max(float(resource_data[m].get("Peak_Memory_MB", 1)) for m in modes_to_plot), 1)

    for i, mode in enumerate(modes_to_plot):
        rd = resource_data[mode]
        if has_resources:
            values = [
                float(rd["Accuracy%"]),
                float(rd["Sensitivity%"]),
                float(rd["Specificity%"]),
                float(rd["F1%"]),
                (1 - float(rd.get("Avg_Time_Per_User_ms", 0)) / max_time) * 100,
                (1 - float(rd.get("Peak_Memory_MB", 0)) / max_mem) * 100,
            ]
        else:
            values = [
                float(rd["Accuracy%"]),
                float(rd["Sensitivity%"]),
                float(rd["Specificity%"]),
                float(rd["F1%"]),
                float(rd.get("PPV%", 0)),
            ]
        values += values[:1]
        color = all_colors[i % len(all_colors)]
        ax.plot(angles, values, "o-", linewidth=2, label=mode, color=color, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7, color="grey")
    ax.set_title(f"Fig. 4: Multi-Objective Comparison\n(Top {len(modes_to_plot)} Modes — auto-selected)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    save_fig(fig, "fig4_radar_chart")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Decision Source Pie Charts
# ═══════════════════════════════════════════════════════════════════════════

def fig5_decision_sources():
    # Auto-select: all modes that have source breakdown data with >1 source
    modes_with_sources = [m for m, d in enhanced_data.items()
                          if d.get("sources") and len(d["sources"]) > 1]

    if not modes_with_sources:
        # Try modes with exactly 1 source too (standalones)
        modes_with_sources = [m for m, d in enhanced_data.items()
                              if d.get("sources") and len(d["sources"]) >= 1]

    if not modes_with_sources:
        print("  Skip fig5: no decision source data found in any result files")
        return

    # Sort by accuracy, pick up to 6
    def _acc(m):
        if m in resource_data:
            return float(resource_data[m].get("Accuracy%", 0))
        return 0.0

    modes_to_plot = sorted(modes_with_sources, key=_acc, reverse=True)[:6]
    n = len(modes_to_plot)

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Fig. 5: Decision Source Distribution (auto-selected modes with source data)",
                 fontweight="bold", y=1.05)

    # Color map for any source name
    source_palette = {
        "CDS-ALARM": "#F44336", "CDS-HEALTHY": "#4CAF50", "CDS-SCREENING": "#FF9800",
        "ANN": "#E91E63", "RNN": "#9C27B0", "ANN-only": "#FF9800", "RNN-only": "#4CAF50",
        "CDS-only": "#2196F3",
        "VOTE-AGREE-HEALTHY": "#66BB6A", "VOTE-AGREE-UNHEALTHY": "#EF5350",
        "VOTE-AGREE": "#66BB6A", "VOTE-TIE-CDS": "#42A5F5",
        "CONF-CDS": "#42A5F5", "CONF-ANN": "#E91E63", "CONF-RNN": "#9C27B0",
        "CONF-HEALTHY": "#66BB6A",
        "STACKED-ANN": "#E91E63", "STACKED-RNN": "#9C27B0",
        "SEL-CDS": "#FF9800", "SEL-CDS-ALARM": "#F44336", "SEL-CDS-HEALTHY": "#66BB6A",
        "SEL-ANN": "#E91E63", "SEL-RNN": "#9C27B0",
    }
    fallback_colors = ["#9E9E9E", "#757575", "#BDBDBD", "#616161", "#E0E0E0"]

    for idx, mode in enumerate(modes_to_plot):
        ax = axes[idx]
        sources = enhanced_data[mode]["sources"]

        labels_list = [s["source"] for s in sources]
        sizes = [s["count"] for s in sources]
        colors = []
        for s in sources:
            name = s["source"]
            if name in source_palette:
                colors.append(source_palette[name])
            else:
                colors.append(fallback_colors[len(colors) % len(fallback_colors)])

        short_labels = []
        for s in sources:
            name = s["source"]
            name = (name.replace("CDS-", "").replace("VOTE-", "V-")
                    .replace("CONF-", "C-").replace("SEL-", "S-")
                    .replace("STACKED-", "Stk-"))
            short_labels.append(f"{name}\n{s['count']}")

        wedges, texts = ax.pie(sizes, labels=short_labels, colors=colors,
                               startangle=90, textprops={"fontsize": 7})
        acc = _acc(mode)
        title = mode.replace("(CDS+", "(").replace("Confidence", "Conf.")
        ax.set_title(f"{title}\n({acc:.1f}%)", fontsize=8, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, "fig5_decision_sources")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Computational Cost Bar Chart
# ═══════════════════════════════════════════════════════════════════════════

def fig6_computational_cost():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig. 6: Computational Cost and Inference Time Comparison", fontweight="bold", y=1.02)

    avail = _available_modes()
    colors = [MODE_COLORS[MODE_ORDER.index(m)] for m in avail]
    shorts = [MODE_SHORT[MODE_ORDER.index(m)] for m in avail]

    # Panel A: Multiplications per user (log scale)
    mults = []
    for mode in avail:
        mults.append(float(resource_data[mode]["Avg_Mults_Per_User"]))

    x = np.arange(len(avail))
    bars = ax1.bar(x, mults, color=colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(shorts, rotation=45, ha="right")
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
    for mode in avail:
        rd = resource_data[mode]
        t_cds = float(rd["Avg_CDS_Train_ms"]) + float(rd["Avg_CDS_Predict_ms"])
        t_ml = float(rd["Avg_ML_Train_ms"]) + float(rd["Avg_ML_Predict_ms"])
        cds_times.append(t_cds)
        ml_times.append(t_ml)
        times.append(t_cds + t_ml)

    ax2.bar(x, cds_times, color="#2196F3", edgecolor="black", linewidth=0.5, width=0.7, label="CDS time")
    ax2.bar(x, ml_times, bottom=cds_times, color="#E91E63", edgecolor="black", linewidth=0.5, width=0.7, label="ML time")
    ax2.set_xticks(x)
    ax2.set_xticklabels(shorts, rotation=45, ha="right")
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

    avail_e = [m for m in MODE_ORDER if m in enhanced_data]
    shorts_e = [MODE_SHORT[MODE_ORDER.index(m)] for m in avail_e]

    matrix = []
    for mode in avail_e:
        row = []
        pc = enhanced_data[mode]["per_class"]
        for cid in class_ids:
            if cid in pc:
                row.append(pc[cid]["accuracy"])
            else:
                row.append(0.0)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, max(5, len(avail_e) * 0.45)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(class_ids)))
    ax.set_xticklabels(class_labels, fontsize=8)
    ax.set_yticks(np.arange(len(avail_e)))
    ax.set_yticklabels(shorts_e, fontsize=9)

    for i in range(len(avail_e)):
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
