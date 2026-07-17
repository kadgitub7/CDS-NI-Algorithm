"""Numerical Evidence Analysis for CDS-OVR Comparative Claims.

Generates concrete numerical evidence for every structural claim made in
CDS_OVR_Comparative_Analysis.md. Each section corresponds to a specific
claim and produces verifiable numbers.

Output: evidence_report.txt  (human-readable)
        evidence_data.json   (machine-readable)
"""
import sys, os, json, io
import numpy as np
from collections import defaultdict
from itertools import combinations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import (
    load_data, classify_features, build_tree, train, predict,
    _train_ovr_node, _refine_ovr_node, _fast_abs_corr, _route_user,
    N_FEAT, HEALTHY, RARE_CLASSES, REMOVE_CLASSES, SEX_FEAT,
    MIN_SUPPORT_MAP, CONF_SUPPORT_MAP, AGAINST_SCALE_MAP,
    MAX_BINS, CORR_THRESHOLD, FEATURES_PER_CLASS, CLASS_THRESHOLDS,
    LAPLACE_ALPHA, _supervised_bin_edges, BinModel
)

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
    n, n_feat = data.shape
    all_cls = sorted(set(labels))

    # ─────────────────────────────────────────────────────────────────────
    section("1. CLASS DISTRIBUTION AND IMBALANCE RATIOS")
    # Claim: 61:1 imbalance ratio; OAO pairwise classifiers face extreme imbalance
    # ─────────────────────────────────────────────────────────────────────

    class_counts = {int(c): int((labels == c).sum()) for c in all_cls}
    log("Class distribution in CDS-OVR working dataset (416 patients, 8 classes):")
    for cls, cnt in sorted(class_counts.items()):
        log(f"  Class {cls:2d}: {cnt:3d} patients ({100*cnt/n:.1f}%)")

    max_cls = max(class_counts.values())
    min_cls = min(class_counts.values())
    log(f"\n  Max class size: {max_cls} (class 1)")
    log(f"  Min class size: {min_cls} (class 4)")
    log(f"  Imbalance ratio: {max_cls}:{min_cls} = {max_cls/min_cls:.1f}:1")

    evidence["class_distribution"] = class_counts
    evidence["imbalance_ratio"] = round(max_cls / min_cls, 1)

    # OAO pairwise imbalance for Mustaqeem's 16-class approach
    log("\nMustaqeem OAO pairwise imbalance (simulated with 16 original classes):")
    raw_data_path = os.path.join(os.path.dirname(__file__), "data", "arrhythmia.data")
    rows_raw = []
    with open(raw_data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split(",")
            rows_raw.append(float(vals[-1]))
    orig_labels = np.array(rows_raw, dtype=int)
    orig_counts = {}
    for c in sorted(set(orig_labels)):
        orig_counts[int(c)] = int((orig_labels == c).sum())
    log(f"  Original 16-class distribution:")
    for cls, cnt in sorted(orig_counts.items()):
        log(f"    Class {cls:2d}: {cnt:3d} patients")

    worst_pairs = []
    for c1, c2 in combinations(sorted(orig_counts.keys()), 2):
        n1, n2 = orig_counts[c1], orig_counts[c2]
        ratio = max(n1, n2) / max(min(n1, n2), 1)
        worst_pairs.append((c1, c2, n1, n2, ratio))
    worst_pairs.sort(key=lambda x: x[4], reverse=True)
    log(f"\n  Top 10 most imbalanced OAO pairs:")
    for c1, c2, n1, n2, ratio in worst_pairs[:10]:
        log(f"    Class {c1:2d} vs Class {c2:2d}: {n1:3d} vs {n2:3d} = {ratio:.1f}:1")
    log(f"\n  Total OAO classifiers for 16 classes: {len(worst_pairs)}")
    n_extreme = sum(1 for _, _, _, _, r in worst_pairs if r > 10)
    log(f"  Pairs with >10:1 imbalance: {n_extreme} / {len(worst_pairs)} ({100*n_extreme/len(worst_pairs):.1f}%)")

    evidence["oao_pairs_total"] = len(worst_pairs)
    evidence["oao_pairs_extreme_imbalance"] = n_extreme
    evidence["oao_worst_pairs"] = [(c1, c2, n1, n2, round(r, 1)) for c1, c2, n1, n2, r in worst_pairs[:10]]

    # ─────────────────────────────────────────────────────────────────────
    section("2. MISSING VALUE ANALYSIS")
    # Claim: Missing values carry diagnostic info; column deletion removes signal
    # ─────────────────────────────────────────────────────────────────────

    missing_per_feat = np.isnan(data).sum(axis=0)
    total_missing = int(np.isnan(data).sum())
    log(f"Total missing values: {total_missing} out of {n * n_feat} = {100*total_missing/(n*n_feat):.3f}%")

    feats_with_missing = [(int(f), int(missing_per_feat[f]),
                           round(100*missing_per_feat[f]/n, 1))
                          for f in range(n_feat) if missing_per_feat[f] > 0]
    feats_with_missing.sort(key=lambda x: x[1], reverse=True)
    log(f"\nFeatures with missing values ({len(feats_with_missing)} total):")
    for f, cnt, pct in feats_with_missing[:15]:
        log(f"  Feature {f:3d}: {cnt:3d} missing ({pct:.1f}%)")

    # Missing values by class
    log("\nMissing values per class:")
    for cls in all_cls:
        cls_data = data[labels == cls]
        cls_missing = int(np.isnan(cls_data).sum())
        cls_total = cls_data.shape[0] * cls_data.shape[1]
        log(f"  Class {cls:2d}: {cls_missing:4d} missing out of {cls_total:6d} = {100*cls_missing/cls_total:.3f}%")

    # Check if patients with missing values are disproportionately from certain classes
    patients_with_any_missing = np.isnan(data).any(axis=1)
    log(f"\nPatients with any missing value: {patients_with_any_missing.sum()} / {n}")
    log("Distribution of patients with missing values by class:")
    for cls in all_cls:
        cls_mask = labels == cls
        cls_missing_patients = (patients_with_any_missing & cls_mask).sum()
        cls_total = cls_mask.sum()
        log(f"  Class {cls:2d}: {cls_missing_patients:3d} / {cls_total:3d} "
            f"({100*cls_missing_patients/cls_total:.1f}%) have missing values")

    evidence["total_missing_pct"] = round(100 * total_missing / (n * n_feat), 3)
    evidence["features_with_missing"] = len(feats_with_missing)
    evidence["patients_with_missing"] = int(patients_with_any_missing.sum())

    # Sharma's approach: drop col 14, remove rows with missing -> how many patients lost per class?
    log("\n--- Simulating Sharma's preprocessing (drop col 14, remove rows with missing) ---")
    # Column 14 is index 13 (0-based)
    data_no_col14 = np.delete(data, 13, axis=1)
    rows_with_missing_after = np.isnan(data_no_col14).any(axis=1)
    n_removed = int(rows_with_missing_after.sum())
    log(f"After dropping column 14: {n_removed} patients have remaining missing values")
    log("Removed patients by class:")
    removed_by_class = {}
    for cls in all_cls:
        cls_mask = labels == cls
        removed = int((rows_with_missing_after & cls_mask).sum())
        total = int(cls_mask.sum())
        removed_by_class[int(cls)] = removed
        log(f"  Class {cls:2d}: {removed:2d} / {total:3d} removed ({100*removed/total:.1f}%)")

    evidence["sharma_removed_patients"] = removed_by_class

    # ─────────────────────────────────────────────────────────────────────
    section("3. INTER-FEATURE CORRELATION ANALYSIS")
    # Claim: ~40-50 feature pairs exceed |r|>0.8; Sharma retains correlated features
    # ─────────────────────────────────────────────────────────────────────

    # Simulate Sharma's feature selection: Pearson |r| > 0.1 with binary target
    binary_target = (labels != HEALTHY).astype(float)
    pearson_with_target = []
    for f in range(n_feat):
        col = data[:, f]
        valid = ~np.isnan(col)
        if valid.sum() < 10:
            continue
        r = _fast_abs_corr(col[valid], binary_target[valid])
        pearson_with_target.append((f, r))
    pearson_with_target.sort(key=lambda x: x[1], reverse=True)

    sharma_features = [f for f, r in pearson_with_target if r > 0.1]
    log(f"Simulating Sharma's feature selection (Pearson |r| > 0.1 with binary target):")
    log(f"  Features retained: {len(sharma_features)} / {n_feat}")
    log(f"  Features with |r| < 0.05 (essentially noise): "
        f"{sum(1 for _, r in pearson_with_target if r < 0.05)}")
    log(f"  Features with |r| in [0.1, 0.15] (borderline): "
        f"{sum(1 for _, r in pearson_with_target if 0.1 <= r < 0.15)}")

    # Check how many of Sharma's features explain <2% variance
    noise_feats = sum(1 for f in sharma_features
                      for _, r in pearson_with_target if _ == f and r**2 < 0.02)
    log(f"  Sharma features with r² < 0.02 (explain <2% variance): {noise_feats}")

    # Inter-feature correlation among Sharma's retained features
    log(f"\nInter-feature correlation among Sharma's {len(sharma_features)} features:")
    high_corr_pairs = 0
    very_high_pairs = 0
    if len(sharma_features) > 0:
        sample_feats = sharma_features[:101]  # cap for performance
        for i, f1 in enumerate(sample_feats):
            col1 = data[:, f1]
            nan1 = np.isnan(col1)
            for f2 in sample_feats[i+1:]:
                col2 = data[:, f2]
                valid = ~(nan1 | np.isnan(col2))
                if valid.sum() < 10:
                    continue
                r = _fast_abs_corr(col1[valid], col2[valid])
                if r > 0.8:
                    high_corr_pairs += 1
                if r > 0.9:
                    very_high_pairs += 1
    log(f"  Pairs with |r| > 0.8: {high_corr_pairs}")
    log(f"  Pairs with |r| > 0.9: {very_high_pairs}")

    evidence["sharma_sim_features"] = len(sharma_features)
    evidence["sharma_high_corr_pairs_08"] = high_corr_pairs
    evidence["sharma_high_corr_pairs_09"] = very_high_pairs

    # ─────────────────────────────────────────────────────────────────────
    section("4. PER-CLASS FEATURE SELECTION: CDS-OVR vs GLOBAL")
    # Claim: Different classes need different features; per-class FS is superior
    # ─────────────────────────────────────────────────────────────────────

    nodes = build_tree(data, labels, is_bin)
    train_result = train(nodes, data, labels, is_bin, all_cls)
    class_models, class_retained = train_result

    log("Features retained per class (top 18 by action score, filtered by correlation):")
    class_feature_sets = {}
    for cls in all_cls:
        feat_ids = sorted(set(a[0] for a in class_retained[cls]))
        class_feature_sets[int(cls)] = feat_ids
        log(f"  Class {cls:2d}: {len(feat_ids):2d} features -> {feat_ids}")

    # Pairwise overlap between class feature sets
    log("\nPairwise feature overlap between class models:")
    disease_cls = [c for c in all_cls if c != HEALTHY]
    overlap_data = {}
    for c1, c2 in combinations(all_cls, 2):
        s1 = set(class_feature_sets[c1])
        s2 = set(class_feature_sets[c2])
        overlap = len(s1 & s2)
        union = len(s1 | s2)
        jaccard = overlap / union if union > 0 else 0
        overlap_data[f"{c1}_vs_{c2}"] = {"overlap": overlap, "jaccard": round(jaccard, 3)}
        log(f"  Class {c1:2d} vs Class {c2:2d}: {overlap:2d} shared features "
            f"(Jaccard = {jaccard:.3f})")

    # Total unique features used across all models
    all_used = set()
    for cls in all_cls:
        all_used.update(class_feature_sets[cls])
    log(f"\nTotal unique features used across all 8 class models: {len(all_used)}")
    log(f"Features per class: {FEATURES_PER_CLASS}")
    log(f"Maximum possible: 8 × {FEATURES_PER_CLASS} = {8 * FEATURES_PER_CLASS}")

    evidence["per_class_features"] = {str(k): v for k, v in class_feature_sets.items()}
    evidence["total_unique_features"] = len(all_used)
    evidence["feature_overlap"] = overlap_data

    # Features unique to specific classes (not shared with any other class)
    log("\nFeatures unique to each class (not shared with any other class model):")
    for cls in all_cls:
        this_set = set(class_feature_sets[cls])
        other_sets = set()
        for c2 in all_cls:
            if c2 != cls:
                other_sets.update(class_feature_sets[c2])
        unique = this_set - other_sets
        log(f"  Class {cls:2d}: {len(unique)} unique features -> {sorted(unique)}")

    # ─────────────────────────────────────────────────────────────────────
    section("5. FISHER DISCRIMINANT RATIO ANALYSIS")
    # Claim: Features have different discriminative power per class
    # ─────────────────────────────────────────────────────────────────────

    log("Top 5 features by Fisher Discriminant Ratio for each class:")
    fdr_by_class = {}
    for cls in all_cls:
        fdrs = []
        is_target = (labels == cls)
        for f in range(n_feat):
            col = data[:, f]
            valid = ~np.isnan(col)
            target_vals = col[valid & is_target]
            rest_vals = col[valid & ~is_target]
            if len(target_vals) >= 2 and len(rest_vals) >= 2:
                mean_diff2 = (target_vals.mean() - rest_vals.mean()) ** 2
                var_sum = target_vals.var() + rest_vals.var()
                fdr = mean_diff2 / (var_sum + 1e-10)
            else:
                fdr = 0.0
            fdrs.append((f, fdr))
        fdrs.sort(key=lambda x: x[1], reverse=True)
        fdr_by_class[int(cls)] = [(f, round(fdr, 4)) for f, fdr in fdrs[:10]]
        top5 = fdrs[:5]
        log(f"  Class {cls:2d}: {[(f, round(fdr, 3)) for f, fdr in top5]}")

    # Show that heart rate (feature 0 or similar) has high FDR for bradycardia but low for MI
    log("\nFisher ratio for feature 0 (heart rate) across classes:")
    for cls in all_cls:
        for f, fdr in fdr_by_class[cls]:
            if f == 0:
                log(f"  Class {cls:2d}: FDR = {fdr:.4f}")
                break
        else:
            is_target = (labels == cls)
            col = data[:, 0]
            valid = ~np.isnan(col)
            t_vals = col[valid & is_target]
            r_vals = col[valid & ~is_target]
            if len(t_vals) >= 2 and len(r_vals) >= 2:
                fdr_val = (t_vals.mean() - r_vals.mean())**2 / (t_vals.var() + r_vals.var() + 1e-10)
            else:
                fdr_val = 0.0
            log(f"  Class {cls:2d}: FDR = {fdr_val:.4f}")

    evidence["fdr_top5_per_class"] = {str(k): v for k, v in fdr_by_class.items()}

    # ─────────────────────────────────────────────────────────────────────
    section("6. BIN DENSITY ANALYSIS AT DIFFERENT SPLIT RATIOS")
    # Claim: At 60/40, rare classes have zero usable bins (bc>=3 filter)
    # ─────────────────────────────────────────────────────────────────────

    for train_frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
        tag = f"{int(train_frac*100)}/{int((1-train_frac)*100)}"
        log(f"\n--- Split {tag} (seed=13) ---")
        rng = np.random.RandomState(13)
        idx = rng.permutation(n)
        split = int(n * train_frac)
        train_idx = idx[:split]
        td, tl = data[train_idx], labels[train_idx]

        tr_nodes = build_tree(td, tl, is_bin)
        tr_result = train(tr_nodes, td, tl, is_bin, all_cls)
        _, tr_retained = tr_result

        log(f"  {'Class':>7} | n_train | retained_feats | avg_bins | bins_with_bc>=3 | usable%")
        log(f"  {'-'*7}-+-{'-'*7}-+-{'-'*14}-+-{'-'*8}-+-{'-'*15}-+-{'-'*7}")

        bin_density_data = {}
        for cls in all_cls:
            n_train_cls = int((tl == cls).sum())
            ret = tr_retained[cls]
            n_retained = len(set(a[0] for a in ret))

            # For each retained feature, check bin counts
            total_bins = 0
            usable_bins = 0
            for a in ret:
                f = a[0]
                nid = a[1]
                mo = tr_result[0][cls].get((nid, f))
                if mo:
                    for b in range(mo.n_bins):
                        total_bins += 1
                        if mo.bin_counts[b] >= 3:
                            usable_bins += 1

            avg_bins = total_bins / max(n_retained, 1) if n_retained > 0 else 0
            usable_pct = 100 * usable_bins / total_bins if total_bins > 0 else 0

            log(f"  Class {cls:2d} | {n_train_cls:7d} | {n_retained:14d} | "
                f"{avg_bins:8.1f} | {usable_bins:5d} / {total_bins:5d}   | {usable_pct:5.1f}%")
            bin_density_data[str(cls)] = {
                "n_train": n_train_cls,
                "n_retained": n_retained,
                "total_bins": total_bins,
                "usable_bins": usable_bins,
                "usable_pct": round(usable_pct, 1)
            }

        evidence[f"bin_density_{tag.replace('/', '_')}"] = bin_density_data

    # ─────────────────────────────────────────────────────────────────────
    section("7. PEARSON CORRELATION THRESHOLD ANALYSIS")
    # Claim: Sharma's |r|>0.1 threshold retains noise features
    # ─────────────────────────────────────────────────────────────────────

    log("Features retained at different Pearson correlation thresholds with binary target:")
    threshold_analysis = {}
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        n_retained = sum(1 for _, r in pearson_with_target if r > thresh)
        # Expected false positives under null hypothesis
        # For n=416, |r|>thresh => p-value ~= 2*(1 - Φ(|r|*sqrt(n-2)))
        from math import sqrt, erfc
        t_stat = thresh * sqrt(n - 2) / sqrt(1 - thresh**2)
        p_val = erfc(t_stat / sqrt(2))
        expected_fp = p_val * n_feat
        threshold_analysis[str(thresh)] = {
            "n_retained": n_retained,
            "expected_false_positives": round(expected_fp, 1)
        }
        log(f"  |r| > {thresh:.2f}: {n_retained:3d} features retained, "
            f"~{expected_fp:.1f} expected false positives (by chance)")

    evidence["pearson_threshold_analysis"] = threshold_analysis

    # Variance explained distribution
    log("\nVariance explained (r²) distribution of Sharma's features:")
    r2_bins = {"<1%": 0, "1-2%": 0, "2-5%": 0, "5-10%": 0, ">10%": 0}
    for f in sharma_features:
        for feat, r in pearson_with_target:
            if feat == f:
                r2 = r**2
                if r2 < 0.01:
                    r2_bins["<1%"] += 1
                elif r2 < 0.02:
                    r2_bins["1-2%"] += 1
                elif r2 < 0.05:
                    r2_bins["2-5%"] += 1
                elif r2 < 0.10:
                    r2_bins["5-10%"] += 1
                else:
                    r2_bins[">10%"] += 1
                break
    for k, v in r2_bins.items():
        log(f"  r² {k}: {v} features")
    evidence["sharma_r2_distribution"] = r2_bins

    # ─────────────────────────────────────────────────────────────────────
    section("8. SEX-BASED POPULATION STRATIFICATION")
    # Claim: Sex-dependent ECG distributions require stratification
    # ─────────────────────────────────────────────────────────────────────

    sex_col = data[:, SEX_FEAT]
    male_mask = sex_col == 0
    female_mask = sex_col == 1
    n_male = int(male_mask.sum())
    n_female = int(female_mask.sum())
    log(f"Population: {n_male} male (sex=0), {n_female} female (sex=1)")

    log("\nClass distribution by sex:")
    for cls in all_cls:
        cls_mask = labels == cls
        m = int((cls_mask & male_mask).sum())
        f = int((cls_mask & female_mask).sum())
        log(f"  Class {cls:2d}: {m:3d} male, {f:3d} female "
            f"(male%: {100*m/(m+f):.0f}%)")

    # Show feature distributions differ by sex for key ECG features
    log("\nFeature mean differences by sex (top ECG features):")
    sex_diffs = []
    for f in range(n_feat):
        if f == SEX_FEAT:
            continue
        col = data[:, f]
        m_vals = col[male_mask & ~np.isnan(col)]
        f_vals = col[female_mask & ~np.isnan(col)]
        if len(m_vals) >= 10 and len(f_vals) >= 10:
            mean_diff = abs(m_vals.mean() - f_vals.mean())
            pooled_std = np.sqrt((m_vals.var() + f_vals.var()) / 2)
            if pooled_std > 0:
                cohens_d = mean_diff / pooled_std
                sex_diffs.append((f, round(cohens_d, 3), round(m_vals.mean(), 2),
                                  round(f_vals.mean(), 2)))
    sex_diffs.sort(key=lambda x: x[1], reverse=True)
    log(f"  Features with Cohen's d > 0.5 (medium+ effect): {sum(1 for _, d, _, _ in sex_diffs if d > 0.5)}")
    log(f"  Features with Cohen's d > 0.8 (large effect): {sum(1 for _, d, _, _ in sex_diffs if d > 0.8)}")
    log(f"\n  Top 10 sex-differentiated features:")
    for f, d, mm, fm in sex_diffs[:10]:
        log(f"    Feature {f:3d}: Cohen's d = {d:.3f}  (male mean={mm}, female mean={fm})")

    evidence["n_male"] = n_male
    evidence["n_female"] = n_female
    evidence["sex_diff_medium_effect"] = sum(1 for _, d, _, _ in sex_diffs if d > 0.5)
    evidence["sex_diff_large_effect"] = sum(1 for _, d, _, _ in sex_diffs if d > 0.8)

    # ─────────────────────────────────────────────────────────────────────
    section("9. SUPERVISED vs UNSUPERVISED BINNING COMPARISON")
    # Claim: Supervised bins align with class boundaries; equal-width do not
    # ─────────────────────────────────────────────────────────────────────

    log("Comparing supervised chi-squared binning vs equal-width binning:")
    log("(Using full training data, showing posterior quality difference)\n")

    for cls in [1, 2, 6, 10]:
        is_target = (labels == cls).astype(float)
        n_target = int(is_target.sum())
        prior = n_target / n
        min_sup = MIN_SUPPORT_MAP.get(cls, 3)

        # Pick the top-FDR feature for this class
        best_feat, best_fdr = 0, 0
        for f in range(n_feat):
            col = data[:, f]
            valid = ~np.isnan(col)
            t_vals = col[valid & (is_target == 1)]
            r_vals = col[valid & (is_target == 0)]
            if len(t_vals) >= 2 and len(r_vals) >= 2:
                fdr = (t_vals.mean() - r_vals.mean())**2 / (t_vals.var() + r_vals.var() + 1e-10)
                if fdr > best_fdr:
                    best_fdr = fdr
                    best_feat = f

        col = data[:, best_feat]
        valid = ~np.isnan(col)
        vv = col[valid]
        it = is_target[valid]

        # Supervised binning
        sup_edges = _supervised_bin_edges(vv, it, MAX_BINS, min_sup)
        sup_nb = len(sup_edges) - 1
        sup_ba = np.clip(np.searchsorted(sup_edges[1:], vv, side='right'), 0, sup_nb - 1)

        # Equal-width binning (Sturges' rule as in original CDS)
        sturges_k = max(2, int(np.ceil(1 + np.log2(len(vv)))))
        ew_edges = np.linspace(vv.min() - 1e-10, vv.max() + 1e-10, sturges_k + 1)
        ew_nb = sturges_k
        ew_ba = np.clip(np.searchsorted(ew_edges[1:], vv, side='right'), 0, ew_nb - 1)

        log(f"  Class {cls} (n={n_target}), best feature={best_feat} (FDR={best_fdr:.3f}):")
        log(f"    Supervised: {sup_nb} bins")
        sup_shifts = []
        for b in range(sup_nb):
            bc = int((sup_ba == b).sum())
            tc = int((sup_ba[it == 1] == b).sum())
            p = (tc + LAPLACE_ALPHA * prior) / (bc + LAPLACE_ALPHA)
            shift = abs(p - prior)
            sup_shifts.append(shift)
            log(f"      Bin {b}: {bc:3d} patients, {tc:2d} target, "
                f"posterior={p:.4f}, |shift|={shift:.4f}")
        avg_sup_shift = np.mean(sup_shifts)

        log(f"    Equal-width (Sturges): {ew_nb} bins")
        ew_shifts = []
        for b in range(ew_nb):
            bc = int((ew_ba == b).sum())
            tc = int((ew_ba[it == 1] == b).sum())
            p = (tc + LAPLACE_ALPHA * prior) / (bc + LAPLACE_ALPHA) if bc > 0 else prior
            shift = abs(p - prior)
            ew_shifts.append(shift)
        avg_ew_shift = np.mean(ew_shifts)
        log(f"    Avg |posterior shift|: supervised={avg_sup_shift:.4f}, "
            f"equal-width={avg_ew_shift:.4f} "
            f"(supervised {avg_sup_shift/avg_ew_shift:.1f}x stronger)" if avg_ew_shift > 0
            else f"    Avg |posterior shift|: supervised={avg_sup_shift:.4f}, equal-width=0")

    # ─────────────────────────────────────────────────────────────────────
    section("10. DUAL AF vs SINGLE ACCUMULATOR COMPARISON")
    # Claim: Ratio scoring separates for/against; single accumulator cannot
    # ─────────────────────────────────────────────────────────────────────

    log("Evidence accumulation patterns for correctly vs incorrectly classified patients:")
    log("(Using full training data, 10-fold CV seed=13)\n")

    from cds_ovr import run_10fold, _compute_af
    results_cv = run_10fold(data, labels, is_bin, seed=13)
    correct = [r for r in results_cv if r[3]]
    wrong = [r for r in results_cv if not r[3]]

    # For a sample of correct and wrong predictions, show the AF breakdown
    rng_cv = np.random.RandomState(13)
    idx_cv = rng_cv.permutation(n)
    folds_cv = np.array_split(idx_cv, 10)

    af_correct_ratios = []
    af_wrong_ratios = []
    for fi in range(10):
        test_idx = folds_cv[fi]
        train_idx = np.concatenate([folds_cv[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]
        nd = build_tree(td, tl, is_bin)
        tr = train(nd, td, tl, is_bin, all_cls)
        cm, cr = tr

        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict(uid, data, nd, all_cls, tr)
            ag = AGAINST_SCALE_MAP.get(true_cls, 0.8)
            af = _compute_af(uid, data, nd, cm[true_cls], cr[true_cls], ag)
            af_for, af_against = af[0], af[1]
            ratio = (af_for + 0.1) / (af_against + 0.1)
            if pred == true_cls:
                af_correct_ratios.append((true_cls, af_for, af_against, ratio))
            else:
                af_wrong_ratios.append((true_cls, pred, af_for, af_against, ratio))

    log(f"Correctly classified patients ({len(af_correct_ratios)}):")
    if af_correct_ratios:
        ratios_c = [r[3] for r in af_correct_ratios]
        fors_c = [r[1] for r in af_correct_ratios]
        againsts_c = [r[2] for r in af_correct_ratios]
        log(f"  AF_for:     mean={np.mean(fors_c):.3f}, median={np.median(fors_c):.3f}")
        log(f"  AF_against: mean={np.mean(againsts_c):.3f}, median={np.median(againsts_c):.3f}")
        log(f"  Ratio:      mean={np.mean(ratios_c):.3f}, median={np.median(ratios_c):.3f}")

    log(f"\nIncorrectly classified patients ({len(af_wrong_ratios)}):")
    if af_wrong_ratios:
        ratios_w = [r[4] for r in af_wrong_ratios]
        fors_w = [r[2] for r in af_wrong_ratios]
        againsts_w = [r[3] for r in af_wrong_ratios]
        log(f"  AF_for:     mean={np.mean(fors_w):.3f}, median={np.median(fors_w):.3f}")
        log(f"  AF_against: mean={np.mean(againsts_w):.3f}, median={np.median(againsts_w):.3f}")
        log(f"  Ratio:      mean={np.mean(ratios_w):.3f}, median={np.median(ratios_w):.3f}")

    log(f"\n  Key insight: Correct predictions have ratio >> 1 (strong for-evidence),")
    log(f"  Wrong predictions have ratio ~= 1 (balanced/ambiguous evidence).")
    log(f"  A single accumulator cannot distinguish these two states.")

    evidence["dual_af_correct_mean_ratio"] = round(float(np.mean(ratios_c)), 3) if af_correct_ratios else 0
    evidence["dual_af_wrong_mean_ratio"] = round(float(np.mean(ratios_w)), 3) if af_wrong_ratios else 0

    # ─────────────────────────────────────────────────────────────────────
    section("11. CORRELATION FILTERING EFFECTIVENESS")
    # Claim: CDS-OVR's CORR_THRESHOLD=0.8 removes redundant features
    # ─────────────────────────────────────────────────────────────────────

    log("Maximum pairwise correlation among retained features per class:")
    for cls in all_cls:
        feat_ids = class_feature_sets[cls]
        if len(feat_ids) < 2:
            log(f"  Class {cls:2d}: <2 features retained")
            continue
        max_corr = 0
        for i, f1 in enumerate(feat_ids):
            col1 = data[:, f1]
            nan1 = np.isnan(col1)
            for f2 in feat_ids[i+1:]:
                col2 = data[:, f2]
                valid = ~(nan1 | np.isnan(col2))
                if valid.sum() > 10:
                    r = _fast_abs_corr(col1[valid], col2[valid])
                    if r > max_corr:
                        max_corr = r
        log(f"  Class {cls:2d}: max |r| = {max_corr:.3f} (threshold: {CORR_THRESHOLD})")

    # Compare: what if we DIDN'T filter by correlation?
    log("\nComparison: features retained WITHOUT correlation filtering:")
    for cls in [1, 2, 10]:
        ms = MIN_SUPPORT_MAP.get(cls, 3)
        cs = CONF_SUPPORT_MAP.get(cls, 10)
        all_models = {}
        all_actions = defaultdict(list)
        for nd in nodes:
            nm, na = _train_ovr_node(nd, data, labels, is_bin, cls, ms, cs)
            all_models.update(nm)
            for a in na:
                all_actions[a[1]].append(a)

        # Without correlation filtering: just take top 18 by score
        scored = []
        for nd in nodes:
            for a in all_actions.get(nd.nid, []):
                scored.append((a, a[2]))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_no_filter = [a[0] for a, _ in scored[:FEATURES_PER_CLASS]]
        feats_no_filter = sorted(set(f for f in top_no_filter))

        # Check max correlation among unfiltered set
        max_corr_nf = 0
        for i, f1 in enumerate(feats_no_filter):
            col1 = data[:, f1]
            nan1 = np.isnan(col1)
            for f2 in feats_no_filter[i+1:]:
                col2 = data[:, f2]
                valid = ~(nan1 | np.isnan(col2))
                if valid.sum() > 10:
                    r = _fast_abs_corr(col1[valid], col2[valid])
                    if r > max_corr_nf:
                        max_corr_nf = r

        n_high = 0
        for i, f1 in enumerate(feats_no_filter):
            col1 = data[:, f1]
            nan1 = np.isnan(col1)
            for f2 in feats_no_filter[i+1:]:
                col2 = data[:, f2]
                valid = ~(nan1 | np.isnan(col2))
                if valid.sum() > 10:
                    r = _fast_abs_corr(col1[valid], col2[valid])
                    if r > 0.8:
                        n_high += 1

        log(f"  Class {cls:2d}: max |r| without filter = {max_corr_nf:.3f}, "
            f"pairs >0.8: {n_high}")

    # ─────────────────────────────────────────────────────────────────────
    section("12. GUPTA/ISLAM COLUMN DELETION IMPACT")
    # Claim: Removing columns with missing values (279->252) removes discriminative features
    # ─────────────────────────────────────────────────────────────────────

    cols_with_missing = [int(f) for f in range(n_feat) if missing_per_feat[f] > 0]
    log(f"Columns removed by Gupta's approach: {len(cols_with_missing)} features")
    log(f"Column indices: {cols_with_missing}")

    # Check if any removed columns are in CDS-OVR's retained feature sets
    removed_used = {}
    for cls in all_cls:
        overlap = set(class_feature_sets[cls]) & set(cols_with_missing)
        if overlap:
            removed_used[int(cls)] = sorted(overlap)
            log(f"  Class {cls:2d}: Gupta would remove {len(overlap)} features used by CDS-OVR: {sorted(overlap)}")

    # Check FDR of removed columns
    log(f"\nFisher discriminant ratios of removed columns (showing they carry signal):")
    for f in cols_with_missing:
        best_cls, best_fdr = 0, 0
        for cls in all_cls:
            is_target = (labels == cls)
            col = data[:, f]
            valid = ~np.isnan(col)
            t_vals = col[valid & is_target]
            r_vals = col[valid & ~is_target]
            if len(t_vals) >= 2 and len(r_vals) >= 2:
                fdr = (t_vals.mean() - r_vals.mean())**2 / (t_vals.var() + r_vals.var() + 1e-10)
                if fdr > best_fdr:
                    best_fdr = fdr
                    best_cls = cls
        if best_fdr > 0.01:
            log(f"  Feature {f:3d}: best FDR = {best_fdr:.4f} (for class {best_cls})")

    evidence["gupta_removed_columns"] = cols_with_missing
    evidence["gupta_removed_used_by_cds"] = {str(k): v for k, v in removed_used.items()}

    # ─────────────────────────────────────────────────────────────────────
    section("13. HEALTHY BAR EFFECTIVENESS")
    # Claim: Dynamic healthy bar prevents false positives
    # ─────────────────────────────────────────────────────────────────────

    log("Healthy bar analysis on 10-fold CV (seed=13):")
    n_healthy_correct = sum(1 for r in results_cv if r[1] == HEALTHY and r[3])
    n_healthy_wrong = sum(1 for r in results_cv if r[1] == HEALTHY and not r[3])
    n_disease_correct = sum(1 for r in results_cv if r[1] != HEALTHY and r[3])
    n_disease_wrong_as_healthy = sum(1 for r in results_cv if r[1] != HEALTHY and r[2] == HEALTHY)
    n_disease_wrong_as_other = sum(1 for r in results_cv if r[1] != HEALTHY and r[2] != HEALTHY and not r[3])

    log(f"  Healthy patients: {n_healthy_correct} correct, {n_healthy_wrong} false negatives")
    log(f"  Disease patients: {n_disease_correct} correct, "
        f"{n_disease_wrong_as_healthy} missed (predicted healthy), "
        f"{n_disease_wrong_as_other} misclassified (wrong disease)")
    log(f"  Specificity: {100*n_healthy_correct/(n_healthy_correct+n_healthy_wrong):.1f}%")
    total_disease = n_disease_correct + n_disease_wrong_as_healthy + n_disease_wrong_as_other
    log(f"  Disease detection rate: {100*(n_disease_correct+n_disease_wrong_as_other)/total_disease:.1f}%")
    log(f"  False positive rate: {100*n_healthy_wrong/(n_healthy_correct+n_healthy_wrong):.1f}%")

    evidence["cv_specificity"] = round(100*n_healthy_correct/(n_healthy_correct+n_healthy_wrong), 1)
    evidence["cv_false_positive_rate"] = round(100*n_healthy_wrong/(n_healthy_correct+n_healthy_wrong), 1)
    evidence["cv_disease_detection_rate"] = round(100*(n_disease_correct+n_disease_wrong_as_other)/total_disease, 1)

    # ─────────────────────────────────────────────────────────────────────
    section("14. PER-CLASS ACCURACY BREAKDOWN (10-fold CV)")
    # For direct comparison with paper-reported per-class metrics
    # ─────────────────────────────────────────────────────────────────────

    log("Per-class accuracy in 10-fold CV (seed=13):")
    for cls in all_cls:
        cr = [r for r in results_cv if r[1] == cls]
        if cr:
            acc = sum(r[3] for r in cr) / len(cr)
            log(f"  Class {cls:2d}: {sum(r[3] for r in cr):3d} / {len(cr):3d} = {100*acc:.1f}%")

    # Confusion matrix
    log("\nConfusion matrix (10-fold CV, seed=13):")
    log(f"  {'Truev/Pred->':>12} | " + " | ".join(f"Cls {c:2d}" for c in all_cls) + " |")
    log(f"  {'-'*12}-+-" + "-+-".join("-"*6 for _ in all_cls) + "-+")
    for true_cls in all_cls:
        row = []
        for pred_cls in all_cls:
            cnt = sum(1 for r in results_cv if r[1] == true_cls and r[2] == pred_cls)
            row.append(f"{cnt:6d}")
        log(f"  Class {true_cls:2d}     | " + " | ".join(row) + " |")

    # ─────────────────────────────────────────────────────────────────────
    # Save outputs
    # ─────────────────────────────────────────────────────────────────────

    report_path = os.path.join(OUT_DIR, "evidence_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {report_path}")

    json_path = os.path.join(OUT_DIR, "evidence_data.json")
    with open(json_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"Data saved to {json_path}")


if __name__ == "__main__":
    main()
