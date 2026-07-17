"""Ablation studies and supplemental evidence for claims in CDS_OVR_Comparative_Analysis.md.

Fills gaps identified in the claim audit:
  1. against_scale ablation: 0.5 vs 0.8 for rare classes
  2. Sex branching ablation: with vs without
  3. Per-class accuracy across multiple splits for Irfan comparison
  4. Feature count in Sharma simulation (101 vs 106 discrepancy)
"""
import sys, os, io, json
import numpy as np
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

from cds_ovr import (
    load_data, classify_features, run_split, run_10fold, stats, binary_acc,
    RARE_CLASSES, AGAINST_SCALE_MAP, SEX_FEAT,
    _train_ovr_node, _refine_ovr_node, _compute_af, _route_user,
    N_FEAT, HEALTHY, REMOVE_CLASSES, MIN_SUPPORT_MAP, CONF_SUPPORT_MAP,
    MAX_BINS, CORR_THRESHOLD, FEATURES_PER_CLASS, CLASS_THRESHOLDS,
    LAPLACE_ALPHA, _supervised_bin_edges, BinModel
)
import cds_ovr

OUT_DIR = os.path.dirname(__file__)
report_lines = []
evidence = {}

def log(msg=""):
    report_lines.append(msg)
    print(msg)

def section(title):
    log(f"\n{'='*80}")
    log(f"  {title}")
    log(f"{'='*80}\n")


def main():
    data, labels = load_data()
    is_bin = classify_features(data)

    # =========================================================================
    # 1. AGAINST_SCALE ABLATION
    # =========================================================================
    section("1. AGAINST_SCALE ABLATION: 0.5 vs 0.8 FOR RARE CLASSES")

    log("Testing whether against_scale=0.5 for rare classes improves accuracy")
    log("vs. uniform against_scale=0.8 for all classes.\n")

    seeds = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]

    # Save original values
    orig_scale = dict(AGAINST_SCALE_MAP)

    # Run with current settings (0.5 for rare, 0.8 for common)
    log("--- Current settings (against_scale=0.5 for rare, 0.8 for common) ---")
    current_accs = []
    current_rare_accs = []
    for seed in seeds:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, spec, sens, ba = stats(res)
        current_accs.append(100 * acc)
        # Per-class accuracy for rare classes
        for cls in sorted(RARE_CLASSES):
            cr = [r for r in res if r[1] == cls]
            if cr:
                cls_acc = 100 * sum(r[3] for r in cr) / len(cr)
                current_rare_accs.append((seed, cls, cls_acc, len(cr)))

    log(f"  10-fold CV mean: {np.mean(current_accs):.2f}% (std={np.std(current_accs):.2f}%)")

    # Run with uniform 0.8
    log("\n--- Uniform against_scale=0.8 for ALL classes ---")
    for cls in RARE_CLASSES:
        AGAINST_SCALE_MAP[cls] = 0.8

    uniform_accs = []
    uniform_rare_accs = []
    for seed in seeds:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, spec, sens, ba = stats(res)
        uniform_accs.append(100 * acc)
        for cls in sorted(RARE_CLASSES):
            cr = [r for r in res if r[1] == cls]
            if cr:
                cls_acc = 100 * sum(r[3] for r in cr) / len(cr)
                uniform_rare_accs.append((seed, cls, cls_acc, len(cr)))

    log(f"  10-fold CV mean: {np.mean(uniform_accs):.2f}% (std={np.std(uniform_accs):.2f}%)")

    # Restore
    for cls in RARE_CLASSES:
        AGAINST_SCALE_MAP[cls] = orig_scale[cls]

    diff = np.mean(current_accs) - np.mean(uniform_accs)
    log(f"\n  Difference: {diff:+.2f} pp (current - uniform)")
    log(f"  Current wins in {sum(c > u for c, u in zip(current_accs, uniform_accs))}/10 seeds")

    # Per-class breakdown for rare classes
    log("\n  Per-class accuracy for rare classes (averaged over 10 seeds):")
    for cls in sorted(RARE_CLASSES):
        cur = [a for s, c, a, n in current_rare_accs if c == cls]
        uni = [a for s, c, a, n in uniform_rare_accs if c == cls]
        log(f"    Class {cls:2d}: current={np.mean(cur):.1f}%, uniform={np.mean(uni):.1f}%, diff={np.mean(cur)-np.mean(uni):+.1f} pp")

    evidence["against_scale_ablation"] = {
        "current_mean": round(float(np.mean(current_accs)), 2),
        "uniform_mean": round(float(np.mean(uniform_accs)), 2),
        "difference_pp": round(diff, 2),
        "current_wins": sum(c > u for c, u in zip(current_accs, uniform_accs)),
    }

    # =========================================================================
    # 2. SEX BRANCHING ABLATION
    # =========================================================================
    section("2. SEX BRANCHING ABLATION: WITH vs WITHOUT")

    log("Testing whether sex-based branching (SEX_FEAT=1) improves accuracy")
    log("vs. no sex branching.\n")

    # Save original
    orig_sex = cds_ovr.SEX_FEAT

    # With sex branching (current)
    log("--- Current: sex branching enabled (SEX_FEAT=1) ---")
    sex_on_accs = []
    for seed in seeds:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        sex_on_accs.append(100 * acc)
    log(f"  10-fold CV mean: {np.mean(sex_on_accs):.2f}% (std={np.std(sex_on_accs):.2f}%)")

    # Without sex branching
    log("\n--- Ablation: sex branching disabled (SEX_FEAT=-1) ---")
    cds_ovr.SEX_FEAT = -1

    sex_off_accs = []
    for seed in seeds:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        sex_off_accs.append(100 * acc)
    log(f"  10-fold CV mean: {np.mean(sex_off_accs):.2f}% (std={np.std(sex_off_accs):.2f}%)")

    cds_ovr.SEX_FEAT = orig_sex

    sex_diff = np.mean(sex_on_accs) - np.mean(sex_off_accs)
    log(f"\n  Difference: {sex_diff:+.2f} pp (sex_on - sex_off)")
    log(f"  Sex branching wins in {sum(a > b for a, b in zip(sex_on_accs, sex_off_accs))}/10 seeds")

    evidence["sex_branching_ablation"] = {
        "with_sex_mean": round(float(np.mean(sex_on_accs)), 2),
        "without_sex_mean": round(float(np.mean(sex_off_accs)), 2),
        "difference_pp": round(sex_diff, 2),
    }

    # =========================================================================
    # 3. PER-CLASS ACCURACY ACROSS SPLITS
    # =========================================================================
    section("3. PER-CLASS ACCURACY ACROSS SPLITS (for Irfan comparison)")

    log("Per-class accuracy at 60/40 and 90/10 (best seed for overall accuracy):\n")

    for frac, tag in [(0.60, "60/40"), (0.90, "90/10")]:
        best_acc = 0
        best_res = None
        best_seed = None
        all_res_by_seed = {}
        for seed in seeds:
            res = run_split(data, labels, is_bin, seed=seed, train_frac=frac)
            acc, _, _, _ = stats(res)
            all_res_by_seed[seed] = res
            if acc > best_acc:
                best_acc = acc
                best_res = res
                best_seed = seed

        log(f"--- {tag} split (best seed={best_seed}, overall acc={100*best_acc:.1f}%) ---")
        classes = sorted(set(r[1] for r in best_res))
        per_class_data = {}
        for cls in classes:
            cr = [r for r in best_res if r[1] == cls]
            correct = sum(r[3] for r in cr)
            cls_acc = 100 * correct / len(cr) if cr else 0
            log(f"  Class {cls:2d}: {correct}/{len(cr)} = {cls_acc:.1f}%")
            per_class_data[str(cls)] = {"correct": correct, "total": len(cr), "accuracy": round(cls_acc, 1)}

        # Also compute mean per-class accuracy across all seeds
        log(f"\n  Mean per-class accuracy across 10 seeds:")
        for cls in classes:
            cls_accs = []
            for seed in seeds:
                cr = [r for r in all_res_by_seed[seed] if r[1] == cls]
                if cr:
                    cls_accs.append(100 * sum(r[3] for r in cr) / len(cr))
            log(f"  Class {cls:2d}: mean={np.mean(cls_accs):.1f}%, std={np.std(cls_accs):.1f}%, best={max(cls_accs):.1f}%, worst={min(cls_accs):.1f}%")
        log("")

        evidence[f"per_class_{tag.replace('/', '_')}"] = per_class_data

    # =========================================================================
    # 4. SHARMA FEATURE COUNT DISCREPANCY
    # =========================================================================
    section("4. SHARMA FEATURE COUNT DISCREPANCY (101 vs 106)")

    log("Sharma reports retaining 101 features with |r| > 0.1.")
    log("Our simulation gets a different count because:")
    log("  - Sharma drops column 14 first, then removes 32 rows with missing values")
    log("  - We simulate on the full 416-patient, 279-feature dataset")
    log("  - The different patient count and missing value handling changes correlations\n")

    # Simulate Sharma's exact pipeline
    from scipy.stats import pearsonr

    # Full dataset binary target
    full_binary = (labels != 1).astype(float)
    n_feat = data.shape[1]

    # Our simulation (all 416 patients, all 279 features)
    our_count = 0
    for f in range(n_feat):
        col = data[:, f]
        mask = ~np.isnan(col)
        if mask.sum() < 10:
            continue
        r, _ = pearsonr(col[mask], full_binary[mask])
        if abs(r) > 0.1:
            our_count += 1
    log(f"Our simulation (416 patients, 279 features): {our_count} features with |r| > 0.1")

    # Simulate Sharma's pipeline: drop col 13 (0-indexed = Sharma's col 14), remove missing rows
    data_sharma = np.delete(data, 13, axis=1)  # drop column 14
    mask_rows = ~np.any(np.isnan(data_sharma), axis=1)
    data_sharma = data_sharma[mask_rows]
    labels_sharma = labels[mask_rows]
    binary_sharma = (labels_sharma != 1).astype(float)

    sharma_count = 0
    for f in range(data_sharma.shape[1]):
        col = data_sharma[:, f]
        mask = ~np.isnan(col)
        if mask.sum() < 10:
            continue
        r, _ = pearsonr(col[mask], binary_sharma[mask])
        if abs(r) > 0.1:
            sharma_count += 1
    log(f"Sharma's pipeline ({data_sharma.shape[0]} patients, {data_sharma.shape[1]} features): {sharma_count} features with |r| > 0.1")

    # Also check: what if Sharma used all 452 patients (16 classes)?
    # We can't do this directly since load_data already filters classes
    log(f"\nNote: Sharma uses all 452 patients (including classes 7,8,11-13,16)")
    log(f"with binary labels, so their feature count may differ due to")
    log(f"different class composition affecting correlations.")

    evidence["sharma_feature_count"] = {
        "our_simulation": our_count,
        "sharma_pipeline_simulation": sharma_count,
        "sharma_reported": 101,
    }

    # =========================================================================
    # 5. SENSITIVITY/SPECIFICITY AT 90/10 (for Jadhav comparison)
    # =========================================================================
    section("5. CDS-OVR SENSITIVITY/SPECIFICITY AT 90/10 (all seeds)")

    log("Full sensitivity and specificity data for Jadhav comparison:\n")
    for seed in seeds:
        res = run_split(data, labels, is_bin, seed=seed, train_frac=0.90)
        acc, spec, sens, ba = stats(res)
        bacc = binary_acc(res)
        tp = sum(1 for r in res if r[1] != 1 and r[2] != 1)
        fn = sum(1 for r in res if r[1] != 1 and r[2] == 1)
        tn = sum(1 for r in res if r[1] == 1 and r[2] == 1)
        fp = sum(1 for r in res if r[1] == 1 and r[2] != 1)
        log(f"  seed={seed:2d}: binary={100*bacc:.1f}%  sens={100*sens:.1f}%  spec={100*spec:.1f}%  TP={tp} FN={fn} TN={tn} FP={fp}")

    # =========================================================================
    # Save outputs
    # =========================================================================

    report_path = os.path.join(OUT_DIR, "evidence_ablation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")

    json_path = os.path.join(OUT_DIR, "evidence_ablation_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"Data saved to {json_path}")


if __name__ == "__main__":
    main()
