"""
reproducibilityReport.py
========================
Standalone reproducibility report for the CDS algorithm on the UCI
Arrhythmia database.

Runs Algorithm 4's LOOCV (per-fold retraining of Algorithms 1-3) and
prints a full gender-disaggregated performance report:

  1. Overall accuracy / sensitivity / specificity / false-alarm rate
  2. Per-gender accuracy table (male vs female)
  3. Per-gender confusion matrix (healthy / diseased breakdown)
  4. Fairness metrics: SPD, Disparate Impact, Equalized Odds difference
  5. Misclassification by arrhythmia class, split by gender
  6. Classes contributing most to female vs male error

Usage
-----
  python reproducibilityReport.py                 # full 452-user LOOCV
  python reproducibilityReport.py --max_users 50  # quick test on first 50

Outputs
-------
  reproducibility_report.txt  -- full formatted text report
  per_class_errors.csv        -- per-class error counts (male / female)
"""

import os
import sys
import argparse
import logging
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_EXT_DIR = os.path.join(_REPO_ROOT, "extensions")
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

_DATA_CANDIDATES = [
    os.path.join(_REPO_ROOT, "data", "arrhythmia.data"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("arrhythmia.data not found. Place it in the data/ directory.")

# ---------------------------------------------------------------------------
# SUPPRESS ALL CDS LOGGING -- keep output clean
# ---------------------------------------------------------------------------
for _name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3", "CDS.Alg4",
              "CDS.Alg1.ForcedSex", "CDS.Alg1.Forced"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# IMPORT CDS PIPELINE
# ---------------------------------------------------------------------------
from CDS_Paper_Algorithms import load_dataset
from Algorithm4 import (
    run_loocv,
    Algorithm4Output,
    HealthDecision,
    DEFAULT_N_BINS,
    fairness_lambda_grid_search,
    select_best_lambda,
)
import fairness_config
from extensions.adversarial_debiasing import run_adversarial_debiasing
from extensions.Fairness_EqualizedOdds import compute_bayes_optimal_predictor

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SEX_COL      = 1
MALE_CODE    = 0
FEMALE_CODE  = 1
HEALTHY_CLS  = 1

CLASS_NAMES = {
    1:  "Normal",
    2:  "Coronary Artery Disease",
    3:  "Old Anterior MI",
    4:  "Old Inferior MI",
    5:  "Sinus Tachycardia",
    6:  "Sinus Bradycardia",
    7:  "PVC",
    8:  "Supraventricular PC",
    9:  "Left Bundle Branch Block",
    10: "Right Bundle Branch Block",
    14: "LV Hypertrophy",
    15: "Atrial Fibrillation",
    16: "Others",
}

OUT_DIR_CSV     = os.path.join(_REPO_ROOT, "output", "csv")
OUT_DIR_REPORTS = os.path.join(_REPO_ROOT, "output", "reports")


# ---------------------------------------------------------------------------
# METRICS COMPUTED DIRECTLY FROM RECORDS
# (avoids relying on pre-computed fields that may have the SCREENING bug)
# ---------------------------------------------------------------------------

def compute_metrics(output: Algorithm4Output):
    """
    Compute all report metrics from raw PredictionRecord list.

    Returns a dict with all scalar metrics and per-class breakdowns.
    """
    records = output.records
    data    = output.data
    N       = len(records)

    # --- overall ---
    n_correct  = sum(1 for r in records if r.is_correct)
    n_healthy  = sum(1 for r in records if r.true_is_healthy)
    n_diseased = sum(1 for r in records if r.true_is_diseased)

    # Healthy correct = NOT alarmed (HEALTHY or SCREENING)
    n_h_correct = sum(
        1 for r in records
        if r.true_is_healthy and r.decision != HealthDecision.UNHEALTHY
    )
    # Diseased correct = ALARMED
    n_d_correct = sum(
        1 for r in records
        if r.true_is_diseased and r.decision == HealthDecision.UNHEALTHY
    )
    n_fa = sum(
        1 for r in records
        if r.true_is_healthy and r.decision == HealthDecision.UNHEALTHY
    )
    n_screening = sum(1 for r in records if r.decision == HealthDecision.SCREENING)

    overall_acc  = n_correct  / N            if N            else 0.0
    sensitivity  = n_d_correct / n_diseased  if n_diseased   else 0.0
    specificity  = n_h_correct / n_healthy   if n_healthy    else 0.0
    fa_rate      = n_fa        / n_healthy   if n_healthy    else 0.0

    # --- per gender (using != UNHEALTHY for healthy-correct, consistent with overall) ---
    def _gender_block(sex_code):
        m = [r for r in records if data[r.user_global_idx, SEX_COL] == sex_code]
        total     = len(m)
        correct   = sum(1 for r in m if r.is_correct)
        healthy   = [r for r in m if r.true_is_healthy]
        diseased  = [r for r in m if r.true_is_diseased]
        h_correct = sum(1 for r in healthy  if r.decision != HealthDecision.UNHEALTHY)
        d_correct = sum(1 for r in diseased if r.decision == HealthDecision.UNHEALTHY)
        d_wrong   = sum(1 for r in diseased if r.decision != HealthDecision.UNHEALTHY)
        screening = sum(1 for r in m if r.decision == HealthDecision.SCREENING)
        return {
            "total":          total,
            "correct":        correct,
            "accuracy":       correct / total if total else 0.0,
            "n_healthy":      len(healthy),
            "n_diseased":     len(diseased),
            "h_correct":      h_correct,
            "d_correct":      d_correct,
            "d_wrong":        d_wrong,
            "sensitivity":    d_correct / len(diseased) if diseased else 0.0,
            "specificity":    h_correct / len(healthy)  if healthy  else 0.0,
            "error_rate":     d_wrong   / len(diseased) if diseased else 0.0,
            "screening":      screening,
        }

    male_blk   = _gender_block(MALE_CODE)
    female_blk = _gender_block(FEMALE_CODE)

    # --- fairness metrics ---
    # SPD: P(Y_hat=UNHEALTHY|male) - P(Y_hat=UNHEALTHY|female)
    n_male_pred_unhealthy = sum(1 for r in records
                                if data[r.user_global_idx, SEX_COL] == MALE_CODE
                                and r.decision == HealthDecision.UNHEALTHY)
    n_female_pred_unhealthy = sum(1 for r in records
                                  if data[r.user_global_idx, SEX_COL] == FEMALE_CODE
                                  and r.decision == HealthDecision.UNHEALTHY)
    n_male_total = male_blk["total"]
    n_female_total = female_blk["total"]
    p_unhealthy_male = n_male_pred_unhealthy / n_male_total if n_male_total > 0 else 0
    p_unhealthy_female = n_female_pred_unhealthy / n_female_total if n_female_total > 0 else 0
    spd = p_unhealthy_male - p_unhealthy_female
    di  = p_unhealthy_female / p_unhealthy_male if p_unhealthy_male > 0 else float("nan")

    tpr_m = male_blk["sensitivity"]
    fpr_m = 1.0 - male_blk["specificity"]
    tpr_f = female_blk["sensitivity"]
    fpr_f = 1.0 - female_blk["specificity"]
    eo_tpr_diff = tpr_m - tpr_f
    eo_fpr_diff = fpr_m - fpr_f
    eo_diff = max(abs(eo_tpr_diff), abs(eo_fpr_diff))

    # --- Subgroup Disparity Calibration Error (SDCE) ---
    # SDCE measures the gap in calibration between subgroups.
    # For each group g, calibration error = |P(Y=1 | Y_hat=positive, G=g) - overall_positive_rate|
    # SDCE = max over groups of |group_precision - overall_precision|
    #
    # Here "positive" = UNHEALTHY prediction, Y=1 = truly diseased.
    n_pred_unhealthy_male = n_male_pred_unhealthy
    n_pred_unhealthy_female = n_female_pred_unhealthy
    n_true_pos_male = sum(
        1 for r in records
        if data[r.user_global_idx, SEX_COL] == MALE_CODE
        and r.decision == HealthDecision.UNHEALTHY
        and r.true_is_diseased
    )
    n_true_pos_female = sum(
        1 for r in records
        if data[r.user_global_idx, SEX_COL] == FEMALE_CODE
        and r.decision == HealthDecision.UNHEALTHY
        and r.true_is_diseased
    )
    n_total_pred_unhealthy = n_pred_unhealthy_male + n_pred_unhealthy_female
    n_total_true_pos = n_true_pos_male + n_true_pos_female

    precision_overall = n_total_true_pos / n_total_pred_unhealthy if n_total_pred_unhealthy > 0 else 0
    precision_male = n_true_pos_male / n_pred_unhealthy_male if n_pred_unhealthy_male > 0 else 0
    precision_female = n_true_pos_female / n_pred_unhealthy_female if n_pred_unhealthy_female > 0 else 0
    sdce = max(abs(precision_male - precision_overall), abs(precision_female - precision_overall))

    # --- per-class error counts by gender ---
    class_errors = defaultdict(lambda: {"female_wrong": 0, "male_wrong": 0,
                                        "female_total": 0, "male_total": 0})
    for r in records:
        if not r.true_is_diseased:
            continue
        cls = r.true_label
        sex = data[r.user_global_idx, SEX_COL]
        if sex == FEMALE_CODE:
            class_errors[cls]["female_total"] += 1
            if not r.is_correct:
                class_errors[cls]["female_wrong"] += 1
        elif sex == MALE_CODE:
            class_errors[cls]["male_total"] += 1
            if not r.is_correct:
                class_errors[cls]["male_wrong"] += 1

    return {
        "N":             N,
        "n_correct":     n_correct,
        "n_healthy":     n_healthy,
        "n_diseased":    n_diseased,
        "n_h_correct":   n_h_correct,
        "n_d_correct":   n_d_correct,
        "n_fa":          n_fa,
        "n_screening":   n_screening,
        "overall_acc":   overall_acc,
        "sensitivity":   sensitivity,
        "specificity":   specificity,
        "fa_rate":       fa_rate,
        "male":          male_blk,
        "female":        female_blk,
        "spd":           spd,
        "di":            di,
        "eo_diff":       eo_diff,
        "sdce":          sdce,
        "precision_male":    precision_male,
        "precision_female":  precision_female,
        "precision_overall": precision_overall,
        "class_errors":  dict(class_errors),
    }


# ---------------------------------------------------------------------------
# REPORT FORMATTER
# ---------------------------------------------------------------------------

def build_report(metrics, data, labels, max_users, elapsed_sec, adv_result=None,
                  grid_results=None, eo_result=None, eo_comparison=None):
    lines = []
    W = 72

    def h1(title):
        lines.append("=" * W)
        lines.append(title.center(W))
        lines.append("=" * W)

    def h2(title):
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    def row(*cols, widths=None):
        if widths is None:
            lines.append("  " + "  ".join(str(c) for c in cols))
        else:
            parts = []
            for c, w in zip(cols, widths):
                s = str(c)
                parts.append(s.ljust(w) if w > 0 else s.rjust(-w))
            lines.append("  " + "  ".join(parts))

    N       = metrics["N"]
    m       = metrics["male"]
    f       = metrics["female"]
    ce      = metrics["class_errors"]

    # ---- header ----
    h1("CDS ALGORITHM -- REPRODUCIBILITY REPORT")
    lines.append(f"  Dataset : UCI Arrhythmia (n={N} users)")
    lines.append(f"  Method  : Leave-One-Out Cross-Validation (per-fold retraining)")
    if max_users:
        lines.append(f"  NOTE    : Limited run (max_users={max_users} of 452)")
    lines.append(f"  Runtime : {elapsed_sec:.0f}s")
    lines.append(f"  Paper target: 95.4% overall accuracy")
    lines.append(f"  Config  : {fairness_config.summary()}")

    # ---- 1. overall ----
    h2("1. OVERALL PERFORMANCE")
    row(f"Total users evaluated : {N}")
    row(f"Correct predictions   : {metrics['n_correct']} / {N}")
    row(f"Overall accuracy      : {metrics['overall_acc']*100:.2f}%")
    row(f"Sensitivity (diseased detection)     : {metrics['sensitivity']*100:.2f}%")
    row(f"Specificity (1 - false alarm rate)   : {metrics['specificity']*100:.2f}%")
    row(f"False alarm rate                     : {metrics['fa_rate']*100:.2f}%")
    row(f"Users sent to SCREENING              : {metrics['n_screening']}")
    row(f"Healthy users  : {metrics['n_healthy']}  |  "
        f"Diseased users : {metrics['n_diseased']}")

    # ---- 2. gender accuracy ----
    h2("2. GENDER-DISAGGREGATED ACCURACY")
    row("Metric", "Male", "Female", "Diff (M-F)",
        widths=[32, -9, -9, -10])
    row("-" * 32, "-" * 9, "-" * 9, "-" * 10,
        widths=[32, -9, -9, -10])
    metrics_pairs = [
        ("Total users",          m["total"],                  f["total"],                  ""),
        ("Overall accuracy",     f"{m['accuracy']*100:.2f}%", f"{f['accuracy']*100:.2f}%",
                                 f"{(m['accuracy']-f['accuracy'])*100:+.2f}%"),
        ("Sensitivity",          f"{m['sensitivity']*100:.2f}%", f"{f['sensitivity']*100:.2f}%",
                                 f"{(m['sensitivity']-f['sensitivity'])*100:+.2f}%"),
        ("Specificity",          f"{m['specificity']*100:.2f}%", f"{f['specificity']*100:.2f}%",
                                 f"{(m['specificity']-f['specificity'])*100:+.2f}%"),
        ("Diseased error rate",  f"{m['error_rate']*100:.2f}%", f"{f['error_rate']*100:.2f}%",
                                 f"{(m['error_rate']-f['error_rate'])*100:+.2f}%"),
        ("Healthy correct",      f"{m['h_correct']}/{m['n_healthy']}",
                                 f"{f['h_correct']}/{f['n_healthy']}", ""),
        ("Diseased correct",     f"{m['d_correct']}/{m['n_diseased']}",
                                 f"{f['d_correct']}/{f['n_diseased']}", ""),
        ("Sent to SCREENING",    m["screening"],              f["screening"],              ""),
    ]
    for label, mv, fv, diff in metrics_pairs:
        row(label, mv, fv, diff, widths=[32, -9, -9, -10])

    lines.append("")
    lines.append("  Paper reported: 5% male diseased error, 14% female diseased error")

    # ---- 3. confusion matrix ----
    h2("3. PER-GENDER CONFUSION MATRIX (Diseased users only)")
    lines.append("  Decision     | Male correct / total  | Female correct / total")
    lines.append("  " + "-" * 56)
    lines.append(f"  Detected     | {m['d_correct']:>5} / {m['n_diseased']:<5} "
                 f"({m['sensitivity']*100:>5.1f}%)  | "
                 f"{f['d_correct']:>5} / {f['n_diseased']:<5} "
                 f"({f['sensitivity']*100:>5.1f}%)")
    lines.append(f"  Missed       | {m['d_wrong']:>5} / {m['n_diseased']:<5} "
                 f"({m['error_rate']*100:>5.1f}%)  | "
                 f"{f['d_wrong']:>5} / {f['n_diseased']:<5} "
                 f"({f['error_rate']*100:>5.1f}%)")

    # ---- 4. fairness ----
    h2("4. FAIRNESS METRICS  (Mehrabi et al., 2021)")
    lines.append("  Metric                        Value    Interpretation")
    lines.append("  " + "-" * 60)
    spd = metrics["spd"]
    di  = metrics["di"]
    eo  = metrics["eo_diff"]
    lines.append(f"  Statistical Parity Diff (SPD) {spd:>+7.4f}  "
                 f"{'fair (|SPD|<0.1)' if abs(spd)<0.1 else 'BIASED (|SPD|>=0.1)'}")
    lines.append(f"  Disparate Impact (DI)         {di:>7.4f}  "
                 f"{'fair (DI>=0.8)' if di>=0.8 else 'BIASED (DI<0.8)'}")
    lines.append(f"  Equalized Odds diff (EO)      {eo:>7.4f}  "
                 f"{'fair (EO<0.1)' if eo<0.1 else 'BIASED (EO>=0.1)'}")

    sdce = metrics["sdce"]
    lines.append(f"  Subgroup Disparity Cal. Err   {sdce:>7.4f}  "
                 f"{'fair (SDCE<0.05)' if sdce<0.05 else 'BIASED (SDCE>=0.05)'}")
    lines.append(f"    Precision (male UNHEALTHY)  {metrics['precision_male']:>7.4f}")
    lines.append(f"    Precision (female UNHEALTHY){metrics['precision_female']:>7.4f}")
    lines.append(f"    Precision (overall)         {metrics['precision_overall']:>7.4f}")
    lines.append("")
    lines.append("  SPD > 0 means males get UNHEALTHY predictions more often.")
    lines.append("  DI < 0.8 is the standard 80% rule threshold (EEOC guideline).")
    lines.append("  SDCE measures calibration gap between subgroups (lower=fairer).")

    # ---- 5. per-class error ----
    h2("5. MISCLASSIFICATION BY ARRHYTHMIA CLASS AND GENDER")
    lines.append(f"  {'Class':<6}  {'Name':<26}  "
                 f"{'F wrong':>8}  {'F total':>8}  {'F err%':>7}  "
                 f"{'M wrong':>8}  {'M total':>8}  {'M err%':>7}")
    lines.append("  " + "-" * 82)

    all_disease_cls = sorted(
        set(y for y in labels if y != HEALTHY_CLS)
    )
    for cls in all_disease_cls:
        err = ce.get(cls, {"female_wrong": 0, "male_wrong": 0,
                           "female_total": 0, "male_total": 0})
        fw = err["female_wrong"]
        ft = err["female_total"]
        mw = err["male_wrong"]
        mt = err["male_total"]
        fp = fw / ft * 100 if ft else float("nan")
        mp = mw / mt * 100 if mt else float("nan")
        fp_s = f"{fp:>6.1f}%" if ft else "   n/a "
        mp_s = f"{mp:>6.1f}%" if mt else "   n/a "
        name = CLASS_NAMES.get(cls, f"Class {cls}")
        lines.append(f"  {cls:<6}  {name:<26}  "
                     f"{fw:>8}  {ft:>8}  {fp_s}  "
                     f"{mw:>8}  {mt:>8}  {mp_s}")

    # ---- 6. top contributors ----
    h2("6. CLASSES CONTRIBUTING MOST TO FEMALE MISCLASSIFICATION")
    fem_cls_errs = [
        (cls, ce[cls]["female_wrong"], ce[cls]["female_total"])
        for cls in ce
        if ce[cls]["female_total"] > 0
    ]
    fem_cls_errs.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"  {'Rank':<5}  {'Class':<6}  {'Name':<26}  "
                 f"{'Wrong':>7}  {'Total':>7}  {'Err%':>7}")
    lines.append("  " + "-" * 65)
    for rank, (cls, wrong, total) in enumerate(fem_cls_errs, 1):
        pct  = wrong / total * 100 if total else 0.0
        name = CLASS_NAMES.get(cls, f"Class {cls}")
        lines.append(f"  {rank:<5}  {cls:<6}  {name:<26}  "
                     f"{wrong:>7}  {total:>7}  {pct:>6.1f}%")
        if rank >= 10:
            break

    h2("7. CLASSES CONTRIBUTING MOST TO MALE MISCLASSIFICATION")
    mal_cls_errs = [
        (cls, ce[cls]["male_wrong"], ce[cls]["male_total"])
        for cls in ce
        if ce[cls]["male_total"] > 0
    ]
    mal_cls_errs.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"  {'Rank':<5}  {'Class':<6}  {'Name':<26}  "
                 f"{'Wrong':>7}  {'Total':>7}  {'Err%':>7}")
    lines.append("  " + "-" * 65)
    for rank, (cls, wrong, total) in enumerate(mal_cls_errs, 1):
        pct  = wrong / total * 100 if total else 0.0
        name = CLASS_NAMES.get(cls, f"Class {cls}")
        lines.append(f"  {rank:<5}  {cls:<6}  {name:<26}  "
                     f"{wrong:>7}  {total:>7}  {pct:>6.1f}%")
        if rank >= 10:
            break

    # ---- 8. adversarial debiasing (if results provided) ----
    if adv_result is not None:
        h2("8. ADVERSARIAL DEBIASING (Zhang et al., 2018 adapted)")
        lines.append(f"  Adversary architecture : MLP ({fairness_config.ADVERSARIAL_HIDDEN_DIM} hidden units)")
        lines.append(f"  Training epochs        : {fairness_config.ADVERSARIAL_EPOCHS}")
        lines.append(f"  Learning rate          : {fairness_config.ADVERSARIAL_LR}")
        lines.append("")

        # Phase 1: Detection
        lines.append("  Phase 1 — DETECTION (adversary on continuous outputs)")
        lines.append(f"    Adversary input          : (risk_score, Y_true)  [equalized odds]")
        lines.append(f"    Adversary accuracy        : {adv_result.adversary_accuracy_before*100:.2f}%")
        signal = "DETECTED" if adv_result.gender_signal_detected else "not detected"
        lines.append(f"    Gender signal             : {signal}")
        lines.append("")

        # Phase 2: Calibration
        lines.append("  Phase 2 — CALIBRATION (adversary-guided threshold adjustment)")
        lines.append(f"    Iterations               : {adv_result.n_calibration_iterations}")
        lines.append(f"    Converged                : {adv_result.calibration_converged}")
        lines.append(f"    Threshold (male)         : {adv_result.threshold_male:.4f}")
        lines.append(f"    Threshold (female)       : {adv_result.threshold_female:.4f}")
        lines.append(f"    Predictions changed      : {adv_result.n_predictions_changed}")
        lines.append("")

        # Phase 3: Verification
        lines.append("  Phase 3 — VERIFICATION (fresh adversary on corrected outputs)")
        lines.append(f"    Adversary accuracy        : {adv_result.adversary_accuracy_after*100:.2f}%")
        lines.append(f"    Chance baseline            : 50.00%")
        lines.append("")

        # Before/after comparison table
        lines.append(f"  {'Metric':<30}  {'Before':>9}  {'After':>9}  {'Delta':>9}")
        lines.append("  " + "-" * 62)
        lines.append(f"  {'Adversary accuracy':<30}  "
                     f"{adv_result.adversary_accuracy_before*100:>8.2f}%  "
                     f"{adv_result.adversary_accuracy_after*100:>8.2f}%  "
                     f"{(adv_result.adversary_accuracy_after-adv_result.adversary_accuracy_before)*100:>+8.2f}%")
        lines.append(f"  {'Overall accuracy':<30}  "
                     f"{adv_result.original_overall_accuracy*100:>8.2f}%  "
                     f"{adv_result.debiased_overall_accuracy*100:>8.2f}%  "
                     f"{(adv_result.debiased_overall_accuracy-adv_result.original_overall_accuracy)*100:>+8.2f}%")
        lines.append(f"  {'Male accuracy':<30}  "
                     f"{adv_result.original_male_accuracy*100:>8.2f}%  "
                     f"{adv_result.debiased_male_accuracy*100:>8.2f}%  "
                     f"{(adv_result.debiased_male_accuracy-adv_result.original_male_accuracy)*100:>+8.2f}%")
        lines.append(f"  {'Female accuracy':<30}  "
                     f"{adv_result.original_female_accuracy*100:>8.2f}%  "
                     f"{adv_result.debiased_female_accuracy*100:>8.2f}%  "
                     f"{(adv_result.debiased_female_accuracy-adv_result.original_female_accuracy)*100:>+8.2f}%")
        gap_before = abs(adv_result.original_male_accuracy - adv_result.original_female_accuracy)
        gap_after = abs(adv_result.debiased_male_accuracy - adv_result.debiased_female_accuracy)
        lines.append(f"  {'Gender accuracy gap':<30}  "
                     f"{gap_before*100:>8.2f}%  "
                     f"{gap_after*100:>8.2f}%  "
                     f"{(gap_after-gap_before)*100:>+8.2f}%")
        lines.append("  " + "-" * 62)
        tpr_gap_b = abs(adv_result.original_tpr_male - adv_result.original_tpr_female)
        tpr_gap_a = abs(adv_result.debiased_tpr_male - adv_result.debiased_tpr_female)
        lines.append(f"  {'TPR gap (|male-female|)':<30}  "
                     f"{tpr_gap_b*100:>8.2f}%  "
                     f"{tpr_gap_a*100:>8.2f}%  "
                     f"{(tpr_gap_a-tpr_gap_b)*100:>+8.2f}%")
        fpr_gap_b = abs(adv_result.original_fpr_male - adv_result.original_fpr_female)
        fpr_gap_a = abs(adv_result.debiased_fpr_male - adv_result.debiased_fpr_female)
        lines.append(f"  {'FPR gap (|male-female|)':<30}  "
                     f"{fpr_gap_b*100:>8.2f}%  "
                     f"{fpr_gap_a*100:>8.2f}%  "
                     f"{(fpr_gap_a-fpr_gap_b)*100:>+8.2f}%")

    # ---- 9. RL fairness grid search results ----
    if grid_results is not None:
        h2("9. RL FAIRNESS LAMBDA GRID SEARCH RESULTS")
        lines.append(f"  {'Lambda':>8}  {'Accuracy':>9}  {'M err%':>7}  {'F err%':>7}  "
                     f"{'Gap':>7}  {'SPD':>8}  {'EO':>8}")
        lines.append("  " + "-" * 62)
        best_gap = min(r["gender_gap"] for r in grid_results)
        for r in grid_results:
            marker = " <-- best" if r["gender_gap"] == best_gap else ""
            lines.append(
                f"  {r['lambda']:>8.3f}  {r['accuracy']*100:>8.1f}%  "
                f"{r['male_error_rate']*100:>6.1f}%  {r['female_error_rate']*100:>6.1f}%  "
                f"{r['gender_gap']*100:>6.1f}%  {r['spd']:>+8.4f}  "
                f"{r['eo_diff']:>8.4f}{marker}"
            )
        lines.append("")
        lines.append(f"  Selected lambda = {fairness_config.FAIRNESS_LAMBDA} "
                     f"(minimises gender gap while maintaining accuracy)")

    # ---- 10. equalized odds post-processing (if results provided) ----
    if eo_result is not None:
        h2("10. EQUALIZED ODDS POST-PROCESSING (Hardt et al., 2016)")
        lines.append(f"  Method: Concave envelope intersection + randomized thresholds")
        lines.append(f"  Constraint: BOTH TPR and FPR equalized across groups")
        lines.append(f"  Loss: prevalence-weighted expected error rate")
        lines.append(f"  ROC sample points per group  : {fairness_config.EQUALIZED_ODDS_ROC_POINTS}")
        lines.append("")
        lines.append(f"  Baseline threshold (all)     : {eo_result.original_threshold:.6f}")
        lines.append(f"  Optimised threshold (male)   : {eo_result.threshold_male:.6f}")
        lines.append(f"  Optimised threshold (female) : {eo_result.threshold_female:.6f}")
        lines.append("")

        # Randomized classifier info
        rt_m = eo_result.randomized_threshold_male
        rt_f = eo_result.randomized_threshold_female
        if rt_m and not rt_m.is_deterministic:
            lines.append(f"  Male classifier   : RANDOMIZED  t_lo={rt_m.t_lo:.4f}  t_hi={rt_m.t_hi:.4f}  p={rt_m.p:.4f}")
        else:
            lines.append(f"  Male classifier   : deterministic  threshold={eo_result.threshold_male:.6f}")
        if rt_f and not rt_f.is_deterministic:
            lines.append(f"  Female classifier : RANDOMIZED  t_lo={rt_f.t_lo:.4f}  t_hi={rt_f.t_hi:.4f}  p={rt_f.p:.4f}")
        else:
            lines.append(f"  Female classifier : deterministic  threshold={eo_result.threshold_female:.6f}")
        lines.append("")

        lines.append(f"  {'Metric':<30}  {'Male':>9}  {'Female':>9}  {'|Delta|':>9}")
        lines.append("  " + "-" * 62)
        lines.append(f"  {'True Positive Rate (TPR)':<30}  "
                     f"{eo_result.tpr_male:>8.4f}   "
                     f"{eo_result.tpr_female:>8.4f}   "
                     f"{abs(eo_result.tpr_male - eo_result.tpr_female):>8.4f}")
        lines.append(f"  {'False Positive Rate (FPR)':<30}  "
                     f"{eo_result.fpr_male:>8.4f}   "
                     f"{eo_result.fpr_female:>8.4f}   "
                     f"{abs(eo_result.fpr_male - eo_result.fpr_female):>8.4f}")
        lines.append("")
        lines.append(f"  Utility loss vs. baseline        : {eo_result.utility_loss:+.6f}")

        diag = eo_result.diagnostics
        lines.append(f"  Feasible region candidates        : {diag.get('feasible_region_points_count', 'N/A')}")
        lines.append(f"  ROC points (male / female)        : "
                     f"{diag.get('roc_male_points_count', 'N/A')} / "
                     f"{diag.get('roc_female_points_count', 'N/A')}")

    # ---- 10b. BASELINE vs EQUALIZED ODDS COMPARISON ----
    if eo_comparison is not None:
        bl = eo_comparison["baseline"]
        eo = eo_comparison["equalized_odds"]

        h2("10b. BASELINE vs EQUALIZED ODDS -- SIDE-BY-SIDE COMPARISON")
        lines.append("")
        lines.append(f"  {'Metric':<32}  {'Baseline':>10}  {'Eq. Odds':>10}  {'Delta':>10}")
        lines.append("  " + "-" * 68)

        def _cmp_row(label, bv, ev):
            bs = f"{bv*100:.2f}%"
            es = f"{ev*100:.2f}%"
            ds = f"{(ev-bv)*100:+.2f}%"
            lines.append(f"  {label:<32}  {bs:>10}  {es:>10}  {ds:>10}")

        _cmp_row("Overall accuracy",   bl["accuracy"],    eo["accuracy"])
        _cmp_row("Sensitivity (TPR)",  bl["sensitivity"],  eo["sensitivity"])
        _cmp_row("Specificity (1-FPR)", bl["specificity"], eo["specificity"])
        lines.append("  " + "-" * 68)
        _cmp_row("Male accuracy",      bl["male_accuracy"],   eo["male_accuracy"])
        _cmp_row("Male sensitivity",   bl["male_sensitivity"], eo["male_sensitivity"])
        _cmp_row("Male specificity",   bl["male_specificity"], eo["male_specificity"])
        _cmp_row("Male error rate",    bl["male_error_rate"],  eo["male_error_rate"])
        lines.append("  " + "-" * 68)
        _cmp_row("Female accuracy",    bl["female_accuracy"],   eo["female_accuracy"])
        _cmp_row("Female sensitivity", bl["female_sensitivity"], eo["female_sensitivity"])
        _cmp_row("Female specificity", bl["female_specificity"], eo["female_specificity"])
        _cmp_row("Female error rate",  bl["female_error_rate"],  eo["female_error_rate"])
        lines.append("  " + "-" * 68)

        bl_gap = abs(bl["male_error_rate"] - bl["female_error_rate"])
        eo_gap = abs(eo["male_error_rate"] - eo["female_error_rate"])
        _cmp_row("Gender error gap",   bl_gap, eo_gap)

        bl_tpr_gap = abs(bl["male_sensitivity"] - bl["female_sensitivity"])
        eo_tpr_gap = abs(eo["male_sensitivity"] - eo["female_sensitivity"])
        _cmp_row("TPR gap (|M-F|)",    bl_tpr_gap, eo_tpr_gap)

        bl_fpr_gap = abs((1-bl["male_specificity"]) - (1-bl["female_specificity"]))
        eo_fpr_gap = abs((1-eo["male_specificity"]) - (1-eo["female_specificity"]))
        _cmp_row("FPR gap (|M-F|)",    bl_fpr_gap, eo_fpr_gap)

        lines.append("")
        lines.append("  Interpretation:")
        if eo_tpr_gap < bl_tpr_gap:
            lines.append(f"    TPR gap reduced: {bl_tpr_gap*100:.2f}% -> {eo_tpr_gap*100:.2f}%")
        else:
            lines.append(f"    TPR gap unchanged/increased: {bl_tpr_gap*100:.2f}% -> {eo_tpr_gap*100:.2f}%")
        if eo_fpr_gap < bl_fpr_gap:
            lines.append(f"    FPR gap reduced: {bl_fpr_gap*100:.2f}% -> {eo_fpr_gap*100:.2f}%")
        else:
            lines.append(f"    FPR gap unchanged/increased: {bl_fpr_gap*100:.2f}% -> {eo_fpr_gap*100:.2f}%")
        acc_delta = eo["accuracy"] - bl["accuracy"]
        lines.append(f"    Accuracy change: {acc_delta*100:+.2f}%")

    # ---- 11. active configuration summary ----
    h2("11. ACTIVE FAIRNESS CONFIGURATION")
    lines.append(f"  {fairness_config.summary()}")
    lines.append("")
    lines.append(f"  ENABLE_REWEIGHING            = {fairness_config.ENABLE_REWEIGHING}")
    lines.append(f"  ENABLE_FAIRNESS_RL           = {fairness_config.ENABLE_FAIRNESS_RL}")
    if fairness_config.ENABLE_FAIRNESS_RL:
        lines.append(f"    FAIRNESS_LAMBDA            = {fairness_config.FAIRNESS_LAMBDA}")
    lines.append(f"  ENABLE_ADVERSARIAL_DEBIASING = {fairness_config.ENABLE_ADVERSARIAL_DEBIASING}")
    lines.append(f"  ENABLE_EQUALIZED_ODDS        = {fairness_config.ENABLE_EQUALIZED_ODDS}")
    lines.append(f"  ENABLE_FORCED_SEX_BRANCHING  = {fairness_config.ENABLE_FORCED_SEX_BRANCHING}")
    lines.append(f"  ENABLE_DATA_AUGMENTATION     = {fairness_config.ENABLE_DATA_AUGMENTATION}")
    if fairness_config.ENABLE_DATA_AUGMENTATION:
        lines.append(f"    AUGMENTATION_STRATEGY      = {fairness_config.AUGMENTATION_STRATEGY}")
        lines.append(f"    AUGMENTATION_TARGET_N      = {fairness_config.AUGMENTATION_TARGET_N}")

    lines.append("")
    lines.append("=" * W)
    lines.append("END OF REPORT")
    lines.append("=" * W)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PER-CLASS CSV
# ---------------------------------------------------------------------------

def build_class_csv(metrics, labels):
    rows = []
    ce   = metrics["class_errors"]
    for cls in sorted(set(y for y in labels if y != HEALTHY_CLS)):
        err = ce.get(cls, {"female_wrong": 0, "male_wrong": 0,
                           "female_total": 0, "male_total": 0})
        fw, ft = err["female_wrong"], err["female_total"]
        mw, mt = err["male_wrong"],   err["male_total"]
        rows.append({
            "class":          cls,
            "name":           CLASS_NAMES.get(cls, f"Class {cls}"),
            "female_wrong":   fw,
            "female_total":   ft,
            "female_error_pct": fw / ft * 100 if ft else float("nan"),
            "male_wrong":     mw,
            "male_total":     mt,
            "male_error_pct": mw / mt * 100 if mt else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CDS Reproducibility Report")
    parser.add_argument("--max_users", type=int, default=None,
                        help="Limit LOOCV to first N users (default: all 452)")
    parser.add_argument("--rng_seed",  type=int, default=42)
    parser.add_argument("--skip_grid_search", action="store_true",
                        help="Skip lambda grid search even if ENABLE_FAIRNESS_RL is True")
    args = parser.parse_args()

    import time

    print("=" * 72)
    print("CDS REPRODUCIBILITY REPORT")
    print("=" * 72)
    print(f"Config: {fairness_config.summary()}")
    print(f"Loading data from: {DATA_PATH}")

    data, labels = load_dataset(DATA_PATH)
    n_run = args.max_users or data.shape[0]

    sex   = data[:, SEX_COL]
    print(f"Dataset: {data.shape[0]} users  |  "
          f"Male={int((sex==MALE_CODE).sum())}  "
          f"Female={int((sex==FEMALE_CODE).sum())}")

    # ── RL Fairness: Grid search for optimal lambda ──────────────────────────
    grid_results = None
    if fairness_config.ENABLE_FAIRNESS_RL and not args.skip_grid_search:
        print()
        print("=" * 72)
        print("FAIRNESS RL: Running lambda grid search ...")
        print("  This runs LOOCV for each lambda value to find the optimal tradeoff.")
        print("=" * 72)

        t_grid = time.time()
        grid_results = fairness_lambda_grid_search(
            data=data,
            labels=labels,
            max_users=args.max_users,
            rng_seed=args.rng_seed,
        )
        t_grid_elapsed = time.time() - t_grid

        best = select_best_lambda(grid_results)

        print(f"\nGrid search complete in {t_grid_elapsed:.0f}s.")
        print(f"\n  {'Lambda':>8}  {'Accuracy':>9}  {'M err%':>7}  {'F err%':>7}  {'Gap':>7}  {'SPD':>8}  {'EO':>8}")
        print(f"  {'-'*62}")
        for r in grid_results:
            marker = " <-- best" if r["lambda"] == best["lambda"] else ""
            print(f"  {r['lambda']:>8.3f}  {r['accuracy']*100:>8.1f}%  "
                  f"{r['male_error_rate']*100:>6.1f}%  {r['female_error_rate']*100:>6.1f}%  "
                  f"{r['gender_gap']*100:>6.1f}%  {r['spd']:>+8.4f}  {r['eo_diff']:>8.4f}{marker}")

        print(f"\n  Selected lambda={best['lambda']} for final run.")

        # Apply the best lambda for the final LOOCV run
        import Algorithm4 as _alg4_mod
        fairness_config.FAIRNESS_LAMBDA = best["lambda"]
        _alg4_mod.FAIRNESS_LAMBDA = best["lambda"]

    print(f"\nRunning LOOCV on {n_run} user(s) ... (this takes several minutes)")
    print()

    t0 = time.time()
    output = run_loocv(
        data         = data,
        labels       = labels,
        max_users    = args.max_users,
        rng_seed     = args.rng_seed,
        verbose      = False,
        n_bins       = DEFAULT_N_BINS,
        nodes_filter = ["root", "root|k1_f1", "root|k1_f2"],
    )
    elapsed = time.time() - t0

    print(f"LOOCV complete in {elapsed:.0f}s.  Computing metrics ...")

    metrics = compute_metrics(output)

    # Run adversarial debiasing if enabled
    adv_result = None
    if fairness_config.ENABLE_ADVERSARIAL_DEBIASING:
        print("Running adversarial debiasing ...")
        from Algorithm4 import DIAGNOSTIC_THRESHOLD_ALG4
        adv_result = run_adversarial_debiasing(
            output=output,
            data=data,
            labels=labels,
            diagnostic_threshold=DIAGNOSTIC_THRESHOLD_ALG4,
        )

    # Run equalized odds post-processing if enabled
    eo_result = None
    eo_comparison = None
    if fairness_config.ENABLE_EQUALIZED_ODDS:
        print("Running equalized odds post-processing (Hardt et al., 2016) ...")
        from Algorithm4 import DIAGNOSTIC_THRESHOLD_ALG4, compute_margin_risk_scores
        from extensions.Fairness_EqualizedOdds import apply_equalized_odds_thresholds
        records = output.records

        # Use margin-based risk scores instead of raw rw values.
        # Raw rw is bimodal (0.0 for all non-alarm users, >0.19 for alarm users)
        # with no intermediate values, making threshold adjustment meaningless.
        # Margin-based scores provide a continuous distribution where:
        #   score > 0 = feature outside healthy range (alarm)
        #   score < 0 = all features within range (healthy)
        # A baseline threshold of 0.0 reproduces original alarm-based decisions.
        risk_scores = compute_margin_risk_scores(records)
        user_labels = np.array([r.true_label for r in records])
        user_data = data[np.array([r.user_global_idx for r in records])]

        eo_result = compute_bayes_optimal_predictor(
            scores=risk_scores,
            labels=user_labels,
            data=user_data,
            baseline_threshold=0.25,  # midpoint of alarm gap in piecewise score space
            verbose=False,
            n_roc_points=fairness_config.EQUALIZED_ODDS_ROC_POINTS,
        )

        # ── Build baseline vs equalized odds comparison ───────────────────
        # Baseline predictions: original CDS decisions
        # Equalized odds: re-classify using fair thresholds
        fair_preds = apply_equalized_odds_thresholds(
            risk_scores, user_data, eo_result, seed=42,
        )

        # Compute baseline metrics from original predictions
        baseline_preds = np.array([
            1 if r.decision == HealthDecision.UNHEALTHY else 0
            for r in records
        ])
        true_binary = (user_labels != 1).astype(int)  # 1=diseased, 0=healthy
        sex_col = user_data[:, SEX_COL]

        def _compute_group_metrics(preds, truth, sex, sex_code=None):
            """Compute accuracy/sensitivity/specificity/error_rate for a group."""
            if sex_code is not None:
                mask = sex == sex_code
                preds = preds[mask]
                truth = truth[mask]
            n = len(truth)
            if n == 0:
                return {"accuracy": 0, "sensitivity": 0, "specificity": 0, "error_rate": 0}
            n_pos = int(truth.sum())          # diseased
            n_neg = n - n_pos                 # healthy
            tp = int(((preds == 1) & (truth == 1)).sum())
            tn = int(((preds == 0) & (truth == 0)).sum())
            fp = int(((preds == 1) & (truth == 0)).sum())
            fn = int(((preds == 0) & (truth == 1)).sum())
            return {
                "accuracy":    (tp + tn) / n if n else 0,
                "sensitivity": tp / n_pos if n_pos else 0,
                "specificity": tn / n_neg if n_neg else 0,
                "error_rate":  fn / n_pos if n_pos else 0,
            }

        bl_all = _compute_group_metrics(baseline_preds, true_binary, sex_col)
        bl_m   = _compute_group_metrics(baseline_preds, true_binary, sex_col, MALE_CODE)
        bl_f   = _compute_group_metrics(baseline_preds, true_binary, sex_col, FEMALE_CODE)

        eo_all = _compute_group_metrics(fair_preds, true_binary, sex_col)
        eo_m   = _compute_group_metrics(fair_preds, true_binary, sex_col, MALE_CODE)
        eo_f   = _compute_group_metrics(fair_preds, true_binary, sex_col, FEMALE_CODE)

        eo_comparison = {
            "baseline": {
                "accuracy":           bl_all["accuracy"],
                "sensitivity":        bl_all["sensitivity"],
                "specificity":        bl_all["specificity"],
                "male_accuracy":      bl_m["accuracy"],
                "male_sensitivity":   bl_m["sensitivity"],
                "male_specificity":   bl_m["specificity"],
                "male_error_rate":    bl_m["error_rate"],
                "female_accuracy":    bl_f["accuracy"],
                "female_sensitivity": bl_f["sensitivity"],
                "female_specificity": bl_f["specificity"],
                "female_error_rate":  bl_f["error_rate"],
            },
            "equalized_odds": {
                "accuracy":           eo_all["accuracy"],
                "sensitivity":        eo_all["sensitivity"],
                "specificity":        eo_all["specificity"],
                "male_accuracy":      eo_m["accuracy"],
                "male_sensitivity":   eo_m["sensitivity"],
                "male_specificity":   eo_m["specificity"],
                "male_error_rate":    eo_m["error_rate"],
                "female_accuracy":    eo_f["accuracy"],
                "female_sensitivity": eo_f["sensitivity"],
                "female_specificity": eo_f["specificity"],
                "female_error_rate":  eo_f["error_rate"],
            },
        }

        n_changed = int((fair_preds != baseline_preds).sum())
        print(f"  Equalized odds re-classified {n_changed} / {len(records)} predictions")

    report  = build_report(metrics, data, labels, args.max_users, elapsed,
                           adv_result, grid_results, eo_result, eo_comparison)
    csv_df  = build_class_csv(metrics, labels)

    # --- print to console ---
    print()
    print(report)

    # --- save files ---
    os.makedirs(OUT_DIR_REPORTS, exist_ok=True)
    os.makedirs(OUT_DIR_CSV, exist_ok=True)
    report_path = os.path.join(OUT_DIR_REPORTS, "reproducibility_report.txt")
    csv_path    = os.path.join(OUT_DIR_CSV, "per_class_errors.csv")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    csv_df.to_csv(csv_path, index=False)

    print(f"\nSaved: {report_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
