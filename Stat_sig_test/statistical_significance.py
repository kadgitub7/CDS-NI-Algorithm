"""
Statistical significance testing for CDS arrhythmia classifier comparisons.

Tests:
  1. McNemar's test — paired comparison of two classifiers on the same 452 users
  2. Wilson confidence intervals — 95% CI on accuracy (better than normal approx for n=452)
  3. Per-class McNemar — tests whether improvement is concentrated in specific disease classes

Analyzes both Enhanced_model and resource_analysis result folders.
"""

import os
import re
import math
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# ── scipy is optional; fall back to manual computation ──────────────────────
try:
    from scipy.stats import chi2
    def chi2_sf(x, df):
        return chi2.sf(x, df)
except ImportError:
    def chi2_sf(x, df):
        """Survival function for chi-squared with df=1 using Wilson-Hilferty."""
        if x <= 0:
            return 1.0
        if df != 1:
            raise ValueError("Fallback only supports df=1")
        z = math.sqrt(x)
        # P(Z > z) using error function
        return 0.5 * math.erfc(z / math.sqrt(2))

# ── Constants ───────────────────────────────────────────────────────────────
N_USERS = 452
ALPHA = 0.05
Z_ALPHA = 1.96  # for 95% CI

BASE_DIR = Path(__file__).resolve().parent.parent
ENHANCED_DIR = BASE_DIR / "Enhanced_model" / "results"
RESOURCE_DIR = BASE_DIR / "resource_analysis" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

RESULT_FILES = [
    "CDS-only.txt",
    "ANN-only.txt",
    "RNN-only.txt",
    "Cascade_ANN.txt",
    "Cascade_RNN.txt",
    "Vote_CDS_ANN.txt",
    "Vote_CDS_RNN.txt",
    "Confidence_CDS_ANN.txt",
    "Confidence_CDS_RNN.txt",
    "Stacked_CDS_to_ANN.txt",
    "Stacked_CDS_to_RNN.txt",
    "Selector_CDS_ANN.txt",
    "Selector_CDS_RNN.txt",
    "AlarmRefine_ANN.txt",
    "AFGated_ANN.txt",
    "DisagreeRules_ANN.txt",
    "AlarmSpecialist_ANN.txt",
    "TripleCascade.txt",
    "TGLLNet-T1_CDS.txt",
    "TGLLNet-T12_CDS.txt",
    "TGLLNet-T123_CDS.txt",
    "TGLLNet-T1-only.txt",
    "TGLLNet-T12-only.txt",
    "TGLLNet-T123-only.txt",
]

CLASS_LABELS = {
    1: "Healthy", 2: "Disease-2", 3: "Disease-3", 4: "Disease-4",
    5: "Disease-5", 6: "Disease-6", 7: "Disease-7", 8: "Disease-8",
    9: "Disease-9", 10: "Disease-10", 14: "Disease-14", 15: "Disease-15",
    16: "Disease-16",
}


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_result_file(filepath):
    """Parse a result .txt file and return structured data."""
    text = filepath.read_text(encoding="utf-8")
    result = {}

    # Mode name
    m = re.search(r"RESULTS?:?\s*(?:RESOURCE ANALYSIS:?)?\s*(.+)", text)
    if m:
        result["mode"] = m.group(1).strip()

    # Accuracy fraction
    m = re.search(r"Accuracy:\s+([\d.]+)%\s+\((\d+)/(\d+)\)", text)
    if not m:
        m = re.search(r"Accuracy:\s+([\d.]+)%", text)
    if m:
        result["accuracy_pct"] = float(m.group(1))
        if m.lastindex >= 3:
            result["correct"] = int(m.group(2))
            result["total"] = int(m.group(3))
        else:
            result["correct"] = round(float(m.group(1)) * N_USERS / 100)
            result["total"] = N_USERS

    # Sensitivity, Specificity, F1
    for metric in ["Sensitivity", "Specificity", "Precision", "F1 Score"]:
        pat = re.escape(metric) + r"[^:]*:\s+([\d.]+)%"
        sm = re.search(pat, text)
        if sm:
            result[metric.lower().replace(" ", "_")] = float(sm.group(1))

    # Per-class breakdown
    per_class = {}
    # Enhanced_model format:  Class   Correct   Total   Accuracy  Label
    for cm in re.finditer(
        r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\S+)",
        text, re.MULTILINE
    ):
        cls_id = int(cm.group(1))
        per_class[cls_id] = {
            "correct": int(cm.group(2)),
            "total": int(cm.group(3)),
            "accuracy": float(cm.group(4)),
            "label": cm.group(5),
        }
    # resource_analysis format:  Class  10 (Disease-10  ): 39/50 (78.0%)
    if not per_class:
        for cm in re.finditer(
            r"Class\s+(\d+)\s+\([^)]+\):\s+(\d+)/(\d+)\s+\(([\d.]+)%\)",
            text
        ):
            cls_id = int(cm.group(1))
            per_class[cls_id] = {
                "correct": int(cm.group(2)),
                "total": int(cm.group(3)),
                "accuracy": float(cm.group(4)),
                "label": CLASS_LABELS.get(cls_id, f"Class-{cls_id}"),
            }
    result["per_class"] = per_class

    # Misclassification trace — extract user IDs and their true class
    errors = {}  # user_id -> true_class
    in_trace = False
    for line in text.splitlines():
        if "MISCLASSIFICATION TRACE" in line:
            in_trace = True
            continue
        if in_trace:
            # Match lines like:  8       H      D2  CDS-ALARM ...
            em = re.match(
                r"\s+(\d+)\s+(H|D\d+)\s+(H|D\d+)\s+\S+", line
            )
            if em:
                uid = int(em.group(1))
                true_raw = em.group(2)
                if true_raw == "H":
                    true_cls = 1
                else:
                    true_cls = int(true_raw[1:])
                errors[uid] = true_cls
            elif line.strip().startswith("====="):
                in_trace = False

    result["error_users"] = set(errors.keys())
    result["error_details"] = errors  # uid -> true_class
    return result


def parse_resource_comparison_errors(filepath):
    """Parse ERROR OVERLAP section from resource_analysis comparison.txt."""
    text = filepath.read_text(encoding="utf-8")
    error_sets = {}
    for m in re.finditer(
        r"^\s+([\w()>+\-]+(?:\s*[\w()>+\-]+)*)\s+errors:\s+\[([^\]]*)\]",
        text, re.MULTILINE
    ):
        mode = m.group(1).strip()
        ids_str = m.group(2).strip()
        if ids_str:
            error_sets[mode] = set(int(x.strip()) for x in ids_str.split(","))
        else:
            error_sets[mode] = set()
    return error_sets


# ── Statistical Tests ───────────────────────────────────────────────────────

def mcnemar_test(errors_a, errors_b, n=N_USERS):
    """
    McNemar's test comparing two classifiers on the same dataset.

    Returns dict with contingency table cells, chi-squared statistic, and p-value.
    Uses Edwards' continuity correction for small counts.
    """
    # b = A wrong, B right (B improved over A)
    # c = A right, B wrong (B regressed from A)
    b = len(errors_a - errors_b)
    c = len(errors_b - errors_a)

    # Also compute for reference
    both_wrong = len(errors_a & errors_b)
    both_right = n - len(errors_a | errors_b)

    if b + c == 0:
        return {
            "b": b, "c": c, "both_wrong": both_wrong, "both_right": both_right,
            "chi2": 0.0, "p_value": 1.0, "p_exact": None, "significant": False,
            "direction": "tied", "note": "Identical error patterns"
        }

    # McNemar's chi-squared with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = chi2_sf(chi2_stat, df=1)

    # Also compute exact (mid-p) for small discordant counts
    if b + c < 25:
        from math import comb
        # Exact binomial test: under H0, b ~ Binomial(b+c, 0.5)
        total = b + c
        observed = min(b, c)
        p_exact = 0.0
        for k in range(observed + 1):
            p_exact += comb(total, k) * (0.5 ** total)
        p_exact *= 2  # two-sided
        p_exact = min(p_exact, 1.0)
    else:
        p_exact = None

    return {
        "b": b, "c": c,
        "both_wrong": both_wrong, "both_right": both_right,
        "chi2": chi2_stat,
        "p_value": p_value,
        "p_exact": p_exact,
        "significant": (p_exact if p_exact is not None else p_value) < ALPHA,
        "direction": "B better" if b > c else ("A better" if c > b else "tied"),
    }


def wilson_ci(correct, total, z=Z_ALPHA):
    """
    Wilson score interval for a binomial proportion.
    Better than normal approximation for small n or extreme p.
    """
    if total == 0:
        return (0.0, 0.0, 0.0)
    p_hat = correct / total
    denom = 1 + z ** 2 / total
    center = (p_hat + z ** 2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / total + z ** 2 / (4 * total ** 2))
    return (max(0.0, center - margin), p_hat, min(1.0, center + margin))


def per_class_mcnemar(errors_a_details, errors_b_details, per_class_info, n_users=N_USERS):
    """
    Per-class McNemar's test.
    errors_a_details, errors_b_details: dict of uid -> true_class for misclassified users.
    per_class_info: dict of class_id -> {total, correct, ...} from any model (for class totals).
    """
    results = {}
    for cls_id in sorted(per_class_info.keys()):
        cls_total = per_class_info[cls_id]["total"]
        label = per_class_info[cls_id].get("label", f"Class-{cls_id}")

        # Users of this class that each model got wrong
        a_wrong_cls = {uid for uid, tc in errors_a_details.items() if tc == cls_id}
        b_wrong_cls = {uid for uid, tc in errors_b_details.items() if tc == cls_id}

        b = len(a_wrong_cls - b_wrong_cls)  # A wrong, B right
        c = len(b_wrong_cls - a_wrong_cls)  # A right, B wrong
        both_wrong = len(a_wrong_cls & b_wrong_cls)
        both_right = cls_total - len(a_wrong_cls | b_wrong_cls)

        if b + c == 0:
            p_val = 1.0
            chi2_stat = 0.0
        elif b + c < 25:
            from math import comb
            total_disc = b + c
            observed = min(b, c)
            p_val = 0.0
            for k in range(observed + 1):
                p_val += comb(total_disc, k) * (0.5 ** total_disc)
            p_val = min(p_val * 2, 1.0)
            chi2_stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0
        else:
            chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
            p_val = chi2_sf(chi2_stat, df=1)

        results[cls_id] = {
            "label": label,
            "total": cls_total,
            "a_errors": len(a_wrong_cls),
            "b_errors": len(b_wrong_cls),
            "b_improved": b,
            "b_regressed": c,
            "both_wrong": both_wrong,
            "both_right": both_right,
            "p_value": p_val,
            "significant": p_val < ALPHA,
            "direction": "B better" if b > c else ("A better" if c > b else "same"),
        }
    return results


# ── Report Generation ───────────────────────────────────────────────────────

def format_report(source_name, models, baseline_key="CDS-only"):
    """Generate a full statistical significance report."""
    lines = []
    w = 100

    lines.append("=" * w)
    lines.append(f"  STATISTICAL SIGNIFICANCE ANALYSIS — {source_name}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Baseline:  {baseline_key}")
    lines.append(f"  N users:   {N_USERS}")
    lines.append(f"  Alpha:     {ALPHA}")
    lines.append("=" * w)

    # ── Section 1: Wilson Confidence Intervals ──
    lines.append("")
    lines.append("-" * w)
    lines.append("  SECTION 1: WILSON 95% CONFIDENCE INTERVALS ON ACCURACY")
    lines.append("-" * w)
    lines.append(f"  {'Model':<30s}  {'Accuracy':>8s}  {'Correct':>7s}  {'95% CI Lower':>12s}  {'95% CI Upper':>12s}  {'CI Width':>8s}")
    lines.append(f"  {'─' * 30}  {'─' * 8}  {'─' * 7}  {'─' * 12}  {'─' * 12}  {'─' * 8}")

    for name, data in models.items():
        correct = data.get("correct", 0)
        total = data.get("total", N_USERS)
        lo, mid, hi = wilson_ci(correct, total)
        lines.append(
            f"  {name:<30s}  {mid * 100:7.2f}%  {correct:>5d}/{total}  "
            f"{lo * 100:11.2f}%  {hi * 100:11.2f}%  {(hi - lo) * 100:7.2f}%"
        )

    # ── Section 2: McNemar's Tests vs Baseline ──
    baseline = models.get(baseline_key)
    if baseline is None:
        lines.append(f"\n  WARNING: Baseline '{baseline_key}' not found. Skipping McNemar tests.")
        return "\n".join(lines)

    baseline_errors = baseline["error_users"]

    lines.append("")
    lines.append("-" * w)
    lines.append(f"  SECTION 2: McNEMAR'S TEST — EACH MODEL vs {baseline_key}")
    lines.append("-" * w)
    lines.append("  Tests whether the error patterns differ significantly from the baseline.")
    lines.append("  b = baseline wrong & model right (improvement)")
    lines.append("  c = baseline right & model wrong (regression)")
    lines.append("")
    lines.append(
        f"  {'Model':<30s}  {'b':>4s}  {'c':>4s}  {'χ²':>8s}  {'p-value':>10s}  "
        f"{'p-exact':>10s}  {'Sig?':>5s}  {'Direction':>12s}"
    )
    lines.append(
        f"  {'─' * 30}  {'─' * 4}  {'─' * 4}  {'─' * 8}  {'─' * 10}  "
        f"{'─' * 10}  {'─' * 5}  {'─' * 12}"
    )

    for name, data in models.items():
        if name == baseline_key:
            continue
        result = mcnemar_test(baseline_errors, data["error_users"])
        p_exact_str = f"{result['p_exact']:.6f}" if result.get("p_exact") is not None else "N/A"
        sig_marker = "YES *" if result["significant"] else "no"
        lines.append(
            f"  {name:<30s}  {result['b']:>4d}  {result['c']:>4d}  "
            f"{result['chi2']:>8.3f}  {result['p_value']:>10.6f}  "
            f"{p_exact_str:>10s}  {sig_marker:>5s}  {result['direction']:>12s}"
        )

    # Contingency table detail for models that differ from baseline
    lines.append("")
    lines.append("  CONTINGENCY TABLE DETAILS (models with different error patterns):")
    lines.append("")
    for name, data in models.items():
        if name == baseline_key:
            continue
        result = mcnemar_test(baseline_errors, data["error_users"])
        if result["b"] == 0 and result["c"] == 0:
            continue
        lines.append(f"  {baseline_key} vs {name}:")
        lines.append(f"    Both correct:   {result['both_right']:>4d}")
        lines.append(f"    Both wrong:     {result['both_wrong']:>4d}")
        lines.append(f"    {baseline_key} wrong, {name} right (improvement): {result['b']:>4d}")
        lines.append(f"    {baseline_key} right, {name} wrong (regression):  {result['c']:>4d}")
        lines.append("")

    # ── Section 3: Per-Class McNemar's ──
    lines.append("-" * w)
    lines.append(f"  SECTION 3: PER-CLASS McNEMAR'S TEST vs {baseline_key}")
    lines.append("-" * w)
    lines.append("  Tests whether improvement/regression is concentrated in specific disease classes.")
    lines.append("  Uses exact binomial test (two-sided) when discordant pairs < 25.")
    lines.append("")

    baseline_details = baseline.get("error_details", {})
    baseline_per_class = baseline.get("per_class", {})

    for name, data in models.items():
        if name == baseline_key:
            continue
        # Skip if identical errors
        if data["error_users"] == baseline_errors:
            lines.append(f"  {name}: identical to {baseline_key} — skipped")
            lines.append("")
            continue

        model_details = data.get("error_details", {})
        pc_results = per_class_mcnemar(baseline_details, model_details, baseline_per_class)

        lines.append(f"  {baseline_key} vs {name}:")
        lines.append(
            f"    {'Class':<12s}  {'N':>5s}  {'Base Err':>8s}  {'Mod Err':>8s}  "
            f"{'Improved':>8s}  {'Regressed':>9s}  {'p-value':>10s}  {'Sig?':>5s}  {'Dir':>10s}"
        )
        lines.append(
            f"    {'─' * 12}  {'─' * 5}  {'─' * 8}  {'─' * 8}  "
            f"{'─' * 8}  {'─' * 9}  {'─' * 10}  {'─' * 5}  {'─' * 10}"
        )
        for cls_id in sorted(pc_results.keys()):
            r = pc_results[cls_id]
            sig_marker = "YES *" if r["significant"] else "no"
            lines.append(
                f"    {r['label']:<12s}  {r['total']:>5d}  {r['a_errors']:>8d}  {r['b_errors']:>8d}  "
                f"{r['b_improved']:>8d}  {r['b_regressed']:>9d}  {r['p_value']:>10.6f}  "
                f"{sig_marker:>5s}  {r['direction']:>10s}"
            )
        lines.append("")

    # ── Section 4: Summary / Interpretation ──
    lines.append("-" * w)
    lines.append("  SECTION 4: SUMMARY & INTERPRETATION")
    lines.append("-" * w)
    lines.append("")

    sig_models = []
    for name, data in models.items():
        if name == baseline_key:
            continue
        result = mcnemar_test(baseline_errors, data["error_users"])
        if result["significant"]:
            sig_models.append((name, result))

    if sig_models:
        lines.append("  Models with STATISTICALLY SIGNIFICANT differences from CDS-only (α=0.05):")
        for name, result in sig_models:
            direction = "outperforms" if result["direction"] == "B better" else "underperforms"
            p_val = result["p_exact"] if result.get("p_exact") is not None else result["p_value"]
            p_str = f"{p_val:.6f}"
            lines.append(f"    • {name} {direction} CDS-only (p={p_str}, b={result['b']}, c={result['c']})")
    else:
        lines.append("  No model shows a statistically significant difference from CDS-only at α=0.05.")
        lines.append("  This means observed accuracy differences could be due to chance variation.")

    lines.append("")
    lines.append("  Notes:")
    lines.append("  • McNemar's test is appropriate here because both classifiers are evaluated on the")
    lines.append("    exact same 452 users via LOOCV, making predictions paired.")
    lines.append("  • Wilson intervals are preferred over normal approximation for n=452 as they provide")
    lines.append("    better coverage, especially for proportions near boundaries.")
    lines.append("  • Per-class tests use exact binomial (two-sided) when discordant pairs < 25,")
    lines.append("    as chi-squared approximation is unreliable for very small counts.")
    lines.append("  • Continuity correction (Edwards') is applied to chi-squared McNemar's statistic.")
    lines.append("")
    lines.append("=" * w)

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def load_models_from_folder(results_dir):
    """Load all model results from a results folder."""
    models = {}
    for fname in RESULT_FILES:
        fpath = results_dir / fname
        if fpath.exists():
            data = parse_result_file(fpath)
            key = fname.replace(".txt", "").replace("_", " ")
            # Normalize key names to match standard naming
            key_map = {
                "CDS-only": "CDS-only",
                "ANN-only": "ANN-only",
                "RNN-only": "RNN-only",
                "Cascade ANN": "Cascade(ANN)",
                "Cascade RNN": "Cascade(RNN)",
                "Vote CDS ANN": "Vote(CDS+ANN)",
                "Vote CDS RNN": "Vote(CDS+RNN)",
                "Confidence CDS ANN": "Confidence(CDS+ANN)",
                "Confidence CDS RNN": "Confidence(CDS+RNN)",
                "Stacked CDS to ANN": "Stacked(CDS->ANN)",
                "Stacked CDS to RNN": "Stacked(CDS->RNN)",
                "Selector CDS ANN": "Selector(CDS+ANN)",
                "Selector CDS RNN": "Selector(CDS+RNN)",
                "AlarmRefine ANN": "AlarmRefine(ANN)",
                "AFGated ANN": "AFGated(ANN)",
                "DisagreeRules ANN": "DisagreeRules(ANN)",
                "AlarmSpecialist ANN": "AlarmSpecialist(ANN)",
                "TripleCascade": "TripleCascade",
                "TGLLNet-T1 CDS": "TGLLNet-T1+CDS",
                "TGLLNet-T12 CDS": "TGLLNet-T12+CDS",
                "TGLLNet-T123 CDS": "TGLLNet-T123+CDS",
                "TGLLNet-T1-only": "TGLLNet-T1-only",
                "TGLLNet-T12-only": "TGLLNet-T12-only",
                "TGLLNet-T123-only": "TGLLNet-T123-only",
            }
            standard_key = key_map.get(key, key)
            models[standard_key] = data
    return models


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Enhanced_model analysis ──
    print("Parsing Enhanced_model results...")
    enhanced_models = load_models_from_folder(ENHANCED_DIR)
    print(f"  Loaded {len(enhanced_models)} models: {', '.join(enhanced_models.keys())}")

    enhanced_report = format_report("Enhanced_model", enhanced_models)
    enhanced_path = OUTPUT_DIR / "enhanced_model_significance.txt"
    enhanced_path.write_text(enhanced_report, encoding="utf-8")
    print(f"  Saved: {enhanced_path}")

    # ── resource_analysis analysis ──
    print("Parsing resource_analysis results...")
    resource_models = load_models_from_folder(RESOURCE_DIR)
    print(f"  Loaded {len(resource_models)} models: {', '.join(resource_models.keys())}")

    resource_report = format_report("resource_analysis", resource_models)
    resource_path = OUTPUT_DIR / "resource_analysis_significance.txt"
    resource_path.write_text(resource_report, encoding="utf-8")
    print(f"  Saved: {resource_path}")

    # ── Combined summary ──
    print("Generating combined summary...")
    combined = generate_combined_summary(enhanced_models, resource_models)
    combined_path = OUTPUT_DIR / "combined_summary.txt"
    combined_path.write_text(combined, encoding="utf-8")
    print(f"  Saved: {combined_path}")

    print("\nDone.")


def generate_combined_summary(enhanced_models, resource_models):
    """Cross-validate that both folders give consistent results and produce summary."""
    lines = []
    w = 100

    lines.append("=" * w)
    lines.append("  COMBINED STATISTICAL SIGNIFICANCE SUMMARY")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * w)
    lines.append("")

    # Cross-validation
    lines.append("-" * w)
    lines.append("  CROSS-VALIDATION: Enhanced_model vs resource_analysis")
    lines.append("-" * w)
    lines.append("  Verifying that both folders produce identical classification results.")
    lines.append("")

    all_match = True
    for model_name in enhanced_models:
        if model_name in resource_models:
            e_errors = enhanced_models[model_name]["error_users"]
            r_errors = resource_models[model_name]["error_users"]
            match = e_errors == r_errors
            status = "MATCH" if match else "MISMATCH"
            if not match:
                all_match = False
            lines.append(f"  {model_name:<30s}: {status}")
        else:
            lines.append(f"  {model_name:<30s}: not in resource_analysis")

    lines.append("")
    if all_match:
        lines.append("  All models produce identical error sets across both analysis folders.")
    else:
        lines.append("  WARNING: Some models have different error sets between folders!")
    lines.append("")

    # Unified results table
    baseline_key = "CDS-only"
    baseline = enhanced_models.get(baseline_key) or resource_models.get(baseline_key)
    if baseline is None:
        lines.append("  ERROR: CDS-only baseline not found.")
        return "\n".join(lines)

    baseline_errors = baseline["error_users"]

    lines.append("-" * w)
    lines.append("  UNIFIED McNEMAR'S TEST RESULTS vs CDS-only")
    lines.append("-" * w)
    lines.append("")
    lines.append(
        f"  {'Model':<30s}  {'Acc%':>6s}  {'ΔAcc':>6s}  {'95% CI':>20s}  "
        f"{'b':>4s}  {'c':>4s}  {'p-value':>10s}  {'Sig?':>5s}"
    )
    lines.append(
        f"  {'─' * 30}  {'─' * 6}  {'─' * 6}  {'─' * 20}  "
        f"{'─' * 4}  {'─' * 4}  {'─' * 10}  {'─' * 5}"
    )

    all_models = enhanced_models if len(enhanced_models) >= len(resource_models) else resource_models
    baseline_acc = baseline["accuracy_pct"]

    for name in [baseline_key] + [k for k in all_models if k != baseline_key]:
        data = all_models[name]
        correct = data.get("correct", 0)
        total = data.get("total", N_USERS)
        lo, mid, hi = wilson_ci(correct, total)
        ci_str = f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"
        delta = data["accuracy_pct"] - baseline_acc

        if name == baseline_key:
            lines.append(
                f"  {name:<30s}  {mid * 100:5.1f}%  {'—':>6s}  {ci_str:>20s}  "
                f"{'—':>4s}  {'—':>4s}  {'(baseline)':>10s}  {'—':>5s}"
            )
        else:
            result = mcnemar_test(baseline_errors, data["error_users"])
            p_use = result.get("p_exact") if result.get("p_exact") is not None else result["p_value"]
            sig_marker = "YES *" if result["significant"] else "no"
            lines.append(
                f"  {name:<30s}  {mid * 100:5.1f}%  {delta:>+5.1f}%  {ci_str:>20s}  "
                f"{result['b']:>4d}  {result['c']:>4d}  {p_use:>10.6f}  {sig_marker:>5s}"
            )

    # Per-class summary for the best-performing hybrid
    lines.append("")
    lines.append("-" * w)
    lines.append("  PER-CLASS BREAKDOWN FOR TOP HYBRID MODELS")
    lines.append("-" * w)

    top_models = ["Cascade(ANN)", "Selector(CDS+ANN)", "Cascade(RNN)", "Selector(CDS+RNN)"]
    baseline_details = baseline.get("error_details", {})
    baseline_per_class = baseline.get("per_class", {})

    for tm in top_models:
        if tm not in all_models:
            continue
        data = all_models[tm]
        if data["error_users"] == baseline_errors:
            continue
        model_details = data.get("error_details", {})
        pc_results = per_class_mcnemar(baseline_details, model_details, baseline_per_class)

        lines.append(f"\n  CDS-only vs {tm}:")
        improving = [(cls_id, r) for cls_id, r in pc_results.items() if r["b_improved"] > 0]
        regressing = [(cls_id, r) for cls_id, r in pc_results.items() if r["b_regressed"] > 0]

        if improving:
            lines.append("    Classes where hybrid IMPROVES:")
            for cls_id, r in improving:
                sig = " (significant)" if r["significant"] else ""
                lines.append(
                    f"      {r['label']:<12s}: +{r['b_improved']} correct, "
                    f"-{r['b_regressed']} regressed, p={r['p_value']:.4f}{sig}"
                )
        if regressing:
            lines.append("    Classes where hybrid REGRESSES:")
            for cls_id, r in regressing:
                sig = " (significant)" if r["significant"] else ""
                lines.append(
                    f"      {r['label']:<12s}: +{r['b_improved']} correct, "
                    f"-{r['b_regressed']} regressed, p={r['p_value']:.4f}{sig}"
                )

    lines.append("")
    lines.append("-" * w)
    lines.append("  KEY TAKEAWAYS")
    lines.append("-" * w)
    lines.append("")

    # Auto-generate takeaways
    sig_count = 0
    for name in all_models:
        if name == baseline_key:
            continue
        result = mcnemar_test(baseline_errors, all_models[name]["error_users"])
        if result["significant"]:
            sig_count += 1

    total_compared = len(all_models) - 1
    lines.append(f"  • {sig_count} of {total_compared} models show statistically significant "
                 f"differences from CDS-only (α={ALPHA})")

    baseline_correct = baseline.get("correct", 0)
    lo, _, hi = wilson_ci(baseline_correct, N_USERS)
    lines.append(f"  • CDS-only 95% CI: [{lo * 100:.1f}%, {hi * 100:.1f}%] — any model whose CI ")
    lines.append(f"    overlaps this range cannot be claimed as definitively better or worse")

    # Check if any hybrids have non-overlapping CIs
    for name in ["Cascade(ANN)", "Selector(CDS+ANN)"]:
        if name in all_models:
            c2 = all_models[name].get("correct", 0)
            lo2, _, hi2 = wilson_ci(c2, N_USERS)
            overlap = lo2 <= hi and hi2 >= lo
            if not overlap:
                lines.append(f"  • {name} CI [{lo2 * 100:.1f}%, {hi2 * 100:.1f}%] does NOT overlap "
                             f"CDS-only CI — strong evidence of difference")
            else:
                lines.append(f"  • {name} CI [{lo2 * 100:.1f}%, {hi2 * 100:.1f}%] overlaps CDS-only "
                             f"CI — differences could be due to sampling variation")

    lines.append("")
    lines.append("=" * w)

    return "\n".join(lines)


if __name__ == "__main__":
    main()
