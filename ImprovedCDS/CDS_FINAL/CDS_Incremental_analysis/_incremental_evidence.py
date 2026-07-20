"""
_incremental_evidence.py

Computes experimental evidence justifying WHY each change in the CDS
incremental analysis affected results the way it did. Uses the actual
UCI arrhythmia dataset and the cds_ovr.py algorithm to produce
data-driven explanations backed by numerical experiments.

Output: incremental_evidence.json (same directory)

Run:
    python _incremental_evidence.py
"""

import sys
import os
import json
import time
import numpy as np
from collections import defaultdict
from itertools import combinations

# Add parent directory so we can import cds_ovr
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from cds_ovr import (
    load_data, classify_features, build_tree, train, predict,
    _train_ovr_node, _refine_ovr_node, _fast_abs_corr,
    _supervised_bin_edges, _compute_af,
    N_FEAT, HEALTHY, LAPLACE_ALPHA, CORR_THRESHOLD, FEATURES_PER_CLASS,
    MAX_BINS, HEALTHY_WEIGHT, SUSPICION_HCUT, SUSPICION_OFFSET,
    HEALTHY_BAR_CAP, REMOVE_CLASSES, RARE_CLASSES,
    CLASS_THRESHOLDS, MIN_SUPPORT_MAP, CONF_SUPPORT_MAP, AGAINST_SCALE_MAP,
    RATIO_EPS, SEX_FEAT, Node, BinModel, _hdist, _route_user,
    stats,
)

OUTPUT_JSON = os.path.join(SCRIPT_DIR, "incremental_evidence.json")
DATA_PATH = os.path.join(PARENT_DIR, "data", "arrhythmia.data")


def load_data_all_classes():
    """Load data WITHOUT class removal (all 452 patients, 13 classes)."""
    rows = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float('nan') if v.strip() == '?' else float(v.strip())
                         for v in line.split(",")])
    raw = np.array(rows, dtype=np.float64)
    data, labels = raw[:, :N_FEAT], raw[:, N_FEAT].astype(int)
    remap = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 14:11, 15:12, 16:13}
    labels = np.array([remap.get(l, l) for l in labels], dtype=int)
    return data, labels


# ====================================================================
# SINGLE 10-fold CV run that collects ALL per-patient evidence
# (Sections 7, 9, 10, 11, 14 all draw from this)
# ====================================================================
def run_10fold_with_evidence(data, labels, is_bin, seed=13):
    """Run 10-fold CV once, collecting all per-patient data for multiple sections."""
    print("  Running shared 10-fold CV (seed=13)...", flush=True)
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 10)
    all_cls = sorted(set(labels))

    records = []

    for fi in range(10):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        class_models, class_retained = train(nodes, td, tl, is_bin, all_cls)

        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, class_scores = predict(uid, data, nodes, all_cls,
                                         (class_models, class_retained))

            # Extract AF values for predicted class (Section 7: dual AF)
            ag = AGAINST_SCALE_MAP.get(pred, 0.8)
            af_for, af_against, n_used, n_for, n_against, max_fc = _compute_af(
                uid, data, nodes, class_models[pred], class_retained[pred], ag)
            ratio = (af_for + RATIO_EPS) / (af_against + RATIO_EPS)

            # Healthy score (Section 10)
            h_score = class_scores.get(HEALTHY, 1.0)

            # Winning score (Section 9)
            winning_score = class_scores.get(pred, 1.0)

            # True class score (Section 14)
            true_class_score = class_scores.get(true_cls, 0.0)

            records.append({
                "uid": int(uid),
                "true_cls": true_cls,
                "pred": int(pred),
                "correct": pred == true_cls,
                "af_for": float(af_for),
                "af_against": float(af_against),
                "ratio": float(ratio),
                "h_score": float(h_score),
                "winning_score": float(winning_score),
                "true_class_score": float(true_class_score),
                "class_scores": {int(c): float(s) for c, s in class_scores.items()},
            })
        if (fi + 1) % 5 == 0:
            print(f"    Fold {fi+1}/10 done", flush=True)

    return records


# ====================================================================
# Section 3: Base Algorithm - Dataset Properties
# ====================================================================
def evidence_section3(data, labels, data_all, labels_all):
    print("  Section 3: Dataset properties...", flush=True)
    class_sizes = {int(c): int((labels == c).sum()) for c in sorted(set(labels))}
    max_size = max(class_sizes.values())
    min_size = min(class_sizes.values())
    all_class_sizes = {int(c): int((labels_all == c).sum()) for c in sorted(set(labels_all))}
    return {
        "section3_n_patients_all": len(labels_all),
        "section3_n_patients_kept": len(labels),
        "section3_n_classes_all": len(set(labels_all)),
        "section3_n_classes_kept": len(set(labels)),
        "section3_class_sizes": class_sizes,
        "section3_all_class_sizes": all_class_sizes,
        "section3_imbalance_ratio": round(max_size / min_size, 1),
        "section3_n_features": N_FEAT,
    }


# ====================================================================
# Section 4: OVR Decomposition - Feature Set Divergence
# ====================================================================
def evidence_section4(data, labels, is_bin):
    print("  Section 4: OVR feature set divergence...", flush=True)
    nodes = build_tree(data, labels, is_bin)
    all_cls = sorted(set(labels))
    class_models, class_retained = train(nodes, data, labels, is_bin, all_cls)

    per_class_features = {}
    for cls in all_cls:
        per_class_features[cls] = sorted(set(a[0] for a in class_retained[cls]))

    jaccard_pairs = {}
    all_unique = set()
    for cls in all_cls:
        all_unique.update(per_class_features[cls])

    for c1, c2 in combinations(all_cls, 2):
        s1, s2 = set(per_class_features[c1]), set(per_class_features[c2])
        union = len(s1 | s2)
        jaccard_pairs[f"{c1}_vs_{c2}"] = round(len(s1 & s2) / union, 3) if union > 0 else 0

    per_class_unique = {}
    for cls in all_cls:
        s = set(per_class_features[cls])
        others = set()
        for c2 in all_cls:
            if c2 != cls:
                others.update(per_class_features[c2])
        per_class_unique[int(cls)] = len(s - others)

    jvals = list(jaccard_pairs.values())
    return {
        "section4_total_unique_features": len(all_unique),
        "section4_per_class_feature_counts": {int(c): len(per_class_features[c]) for c in all_cls},
        "section4_per_class_unique_counts": per_class_unique,
        "section4_min_jaccard": round(min(jvals), 3),
        "section4_max_jaccard": round(max(jvals), 3),
        "section4_mean_jaccard": round(float(np.mean(jvals)), 3),
        "section4_jaccard_pairs": jaccard_pairs,
    }


# ====================================================================
# Section 5: Supervised Binning - Posterior Shift Comparison
# ====================================================================
def evidence_section5(data, labels, is_bin):
    print("  Section 5: Supervised vs equal-width binning...", flush=True)
    nodes = build_tree(data, labels, is_bin)
    nd = nodes[0]
    nd_data, nd_labels = data[nd.uidx], labels[nd.uidx]
    ns = nd.nu

    target_class = 6  # Sinus Bradycardia
    is_target = (nd_labels == target_class)
    n_target = int(is_target.sum())
    prior = n_target / ns
    min_support = MIN_SUPPORT_MAP.get(target_class, 3)

    supervised_shifts, equalwidth_shifts = [], []
    example_feature_14 = {}

    for f in range(N_FEAT):
        col = nd_data[:, f]
        vm = ~np.isnan(col)
        vv = col[vm]
        nv = len(vv)
        if nv < 10 or is_bin[f]:
            continue
        vmin, vmax = float(vv.min()), float(vv.max())
        if vmin == vmax:
            continue
        lv, it = nd_labels[vm], is_target[vm]

        # Supervised
        max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
        sup_edges = _supervised_bin_edges(vv, it.astype(float), max_nb, min_support)
        sup_nb = len(sup_edges) - 1
        sup_ba = np.clip(np.searchsorted(sup_edges[1:], vv, side='right'), 0, sup_nb - 1)
        sup_bc = np.bincount(sup_ba, minlength=sup_nb)
        sup_tc = np.bincount(sup_ba[lv == target_class], minlength=sup_nb).astype(float)
        sup_pc = (sup_tc + LAPLACE_ALPHA * prior) / (sup_bc + LAPLACE_ALPHA)
        sup_shift = float(np.mean(np.abs(sup_pc - prior)))
        supervised_shifts.append(sup_shift)

        # Equal-width (Sturges)
        K = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
        ew_edges = np.linspace(vmin - 1e-10, vmax + 1e-10, K + 1)
        ew_nb = K
        ew_ba = np.clip(np.searchsorted(ew_edges[1:], vv, side='right'), 0, ew_nb - 1)
        ew_bc = np.bincount(ew_ba, minlength=ew_nb)
        ew_tc = np.bincount(ew_ba[lv == target_class], minlength=ew_nb).astype(float)
        ew_pc = (ew_tc + LAPLACE_ALPHA * prior) / (ew_bc + LAPLACE_ALPHA)
        ew_shift = float(np.mean(np.abs(ew_pc - prior)))
        equalwidth_shifts.append(ew_shift)

        if f == 14:
            best_bin = int(np.argmax(sup_pc))
            example_feature_14 = {
                "feature": 14,
                "supervised_n_bins": sup_nb,
                "equalwidth_n_bins": ew_nb,
                "supervised_max_posterior": round(float(sup_pc[best_bin]), 4),
                "equalwidth_max_posterior": round(float(max(ew_pc)), 4),
                "prior": round(prior, 4),
                "supervised_avg_shift": round(sup_shift, 4),
                "equalwidth_avg_shift": round(ew_shift, 4),
            }

    avg_sup = float(np.mean(supervised_shifts))
    avg_ew = float(np.mean(equalwidth_shifts))
    return {
        "section5_class6_supervised_avg_shift": round(avg_sup, 4),
        "section5_class6_equalwidth_avg_shift": round(avg_ew, 4),
        "section5_shift_ratio": round(avg_sup / avg_ew if avg_ew > 0 else 0, 2),
        "section5_n_features_compared": len(supervised_shifts),
        "section5_example_feature_14": example_feature_14,
        "section5_prior_class6": round(prior, 4),
    }


# ====================================================================
# Section 6: Correlation Filter - Redundancy Analysis
# ====================================================================
def evidence_section6(data, labels, is_bin):
    print("  Section 6: Correlation structure analysis...", flush=True)
    n_above_08, n_above_09, n_pairs = 0, 0, 0
    correlated_groups = defaultdict(set)

    valid_feats = []
    for f in range(N_FEAT):
        col = data[:, f]
        if int((~np.isnan(col)).sum()) > 20 and not is_bin[f]:
            valid_feats.append(f)

    # Pre-extract columns to speed up
    cols = {}
    nans = {}
    for f in valid_feats:
        cols[f] = data[:, f]
        nans[f] = np.isnan(cols[f])

    for i, f1 in enumerate(valid_feats):
        for f2 in valid_feats[i+1:]:
            valid = ~(nans[f1] | nans[f2])
            nv = valid.sum()
            if nv < 20:
                continue
            c = _fast_abs_corr(cols[f1][valid], cols[f2][valid])
            n_pairs += 1
            if c > 0.8:
                n_above_08 += 1
                correlated_groups[f1].add(f2)
                correlated_groups[f2].add(f1)
            if c > 0.9:
                n_above_09 += 1

    largest_feat = max(correlated_groups, key=lambda f: len(correlated_groups[f])) if correlated_groups else -1
    example_group = sorted(correlated_groups[largest_feat])[:10] if largest_feat >= 0 else []

    # Features surviving per class (reuse train)
    nodes = build_tree(data, labels, is_bin)
    all_cls = sorted(set(labels))
    _, class_retained = train(nodes, data, labels, is_bin, all_cls)
    feats_per_class = {int(c): len(set(a[0] for a in class_retained[c])) for c in all_cls}

    return {
        "section6_total_pairs_above_08": n_above_08,
        "section6_total_pairs_above_09": n_above_09,
        "section6_total_pairs_computed": n_pairs,
        "section6_n_features_in_correlated_groups": len(correlated_groups),
        "section6_largest_correlated_group_size": len(example_group) + 1 if example_group else 0,
        "section6_example_correlated_group": example_group,
        "section6_n_features_surviving_per_class": feats_per_class,
        "section6_avg_features_per_class": round(float(np.mean(list(feats_per_class.values()))), 1),
    }


# ====================================================================
# Section 7 (from CV records): Dual AF
# ====================================================================
def evidence_section7(records):
    correct_ratios = [r["ratio"] for r in records if r["correct"]]
    wrong_ratios = [r["ratio"] for r in records if not r["correct"]]
    return {
        "section7_correct_mean_ratio": round(float(np.mean(correct_ratios)), 3),
        "section7_correct_median_ratio": round(float(np.median(correct_ratios)), 3),
        "section7_wrong_mean_ratio": round(float(np.mean(wrong_ratios)), 3),
        "section7_wrong_median_ratio": round(float(np.median(wrong_ratios)), 3),
        "section7_n_correct": len(correct_ratios),
        "section7_n_wrong": len(wrong_ratios),
        "section7_ratio_separation": round(
            float(np.mean(correct_ratios)) / float(np.mean(wrong_ratios)), 2)
            if np.mean(wrong_ratios) > 0 else 0,
    }


# ====================================================================
# Section 8: Fisher Weighting
# ====================================================================
def evidence_section8(data, labels, is_bin):
    print("  Section 8: Fisher discriminant ratio analysis...", flush=True)
    all_cls = sorted(set(labels))
    fdr_matrix = np.zeros((N_FEAT, len(all_cls)))

    for ci, cls in enumerate(all_cls):
        for f in range(N_FEAT):
            col = data[:, f]
            vm = ~np.isnan(col)
            vv, lv = col[vm], labels[vm]
            tv, rv = vv[lv == cls], vv[lv != cls]
            if len(tv) >= 2 and len(rv) >= 2:
                md2 = (tv.mean() - rv.mean()) ** 2
                vs = tv.var() + rv.var()
                fdr_matrix[f, ci] = md2 / (vs + 1e-10)

    cls3_idx = all_cls.index(3)
    cls9_idx = all_cls.index(9)
    cls5_idx = all_cls.index(5)
    cls6_idx = all_cls.index(6)

    f111_c3 = float(fdr_matrix[111, cls3_idx])
    f111_others = [float(fdr_matrix[111, ci]) for ci, c in enumerate(all_cls) if c != 3]
    f111_max_other = max(f111_others)

    f16_c9 = float(fdr_matrix[16, cls9_idx])
    f16_others = [float(fdr_matrix[16, ci]) for ci, c in enumerate(all_cls) if c != 9]

    fdr_top5 = {}
    for ci, cls in enumerate(all_cls):
        col = fdr_matrix[:, ci]
        top_idx = np.argsort(col)[-5:][::-1]
        fdr_top5[int(cls)] = [(int(f), round(float(col[f]), 3)) for f in top_idx]

    top5_feats = defaultdict(int)
    for cls in all_cls:
        for f, _ in fdr_top5[int(cls)]:
            top5_feats[f] += 1

    return {
        "section8_feature_111_class3_fdr": round(f111_c3, 3),
        "section8_feature_111_max_ratio": round(f111_c3 / f111_max_other, 1) if f111_max_other > 0.001 else 999,
        "section8_feature_16_class9_fdr": round(f16_c9, 3),
        "section8_feature_16_max_other": round(max(f16_others), 3),
        "section8_feature_14_class5_fdr": round(float(fdr_matrix[14, cls5_idx]), 3),
        "section8_feature_14_class6_fdr": round(float(fdr_matrix[14, cls6_idx]), 3),
        "section8_top5_overlap_more_than_2_classes": sum(1 for c in top5_feats.values() if c > 2),
        "section8_fdr_top5_per_class": fdr_top5,
    }


# ====================================================================
# Section 9 (from CV records): Ratio Scoring
# ====================================================================
def evidence_section9(records):
    healthy_scores = [r["winning_score"] for r in records if r["true_cls"] == HEALTHY]
    disease_scores = [r["winning_score"] for r in records if r["true_cls"] != HEALTHY]
    all_scores = [r["winning_score"] for r in records]
    ambiguous = sum(1 for s in all_scores if 0.8 <= s <= 1.5)
    return {
        "section9_healthy_mean_score": round(float(np.mean(healthy_scores)), 3),
        "section9_healthy_median_score": round(float(np.median(healthy_scores)), 3),
        "section9_disease_mean_score": round(float(np.mean(disease_scores)), 3),
        "section9_disease_median_score": round(float(np.median(disease_scores)), 3),
        "section9_ambiguous_count_08_15": ambiguous,
        "section9_total_patients": len(all_scores),
        "section9_score_p25": round(float(np.percentile(all_scores, 25)), 3),
        "section9_score_p75": round(float(np.percentile(all_scores, 75)), 3),
        "section9_score_max": round(float(np.max(all_scores)), 3),
    }


# ====================================================================
# Section 10 (from CV records): Healthy Bar
# ====================================================================
def evidence_section10(records):
    results = [(r["uid"], r["true_cls"], r["pred"], r["correct"]) for r in records]
    acc, spec, sens, ba = stats(results)
    fpr = 1 - spec

    healthy_h = [r["h_score"] for r in records if r["true_cls"] == HEALTHY]
    disease_h = [r["h_score"] for r in records if r["true_cls"] != HEALTHY]

    fn_by_class = defaultdict(int)
    cls_counts = defaultdict(int)
    for r in records:
        if r["true_cls"] != HEALTHY:
            cls_counts[r["true_cls"]] += 1
            if r["pred"] == HEALTHY:
                fn_by_class[r["true_cls"]] += 1

    disease_r = [r for r in records if r["true_cls"] != HEALTHY]
    det = sum(1 for r in disease_r if r["pred"] != HEALTHY)
    det_rate = det / len(disease_r) if disease_r else 0

    return {
        "section10_specificity": round(spec * 100, 1),
        "section10_fpr": round(fpr * 100, 1),
        "section10_disease_detection_rate": round(det_rate * 100, 1),
        "section10_false_negatives_by_class": {int(c): int(fn_by_class[c]) for c in sorted(cls_counts)},
        "section10_mean_healthy_score_for_healthy": round(float(np.mean(healthy_h)), 3),
        "section10_mean_healthy_score_for_disease": round(float(np.mean(disease_h)), 3),
        "section10_median_healthy_score_for_healthy": round(float(np.median(healthy_h)), 3),
        "section10_median_healthy_score_for_disease": round(float(np.median(disease_h)), 3),
        "section10_accuracy": round(acc * 100, 1),
    }


# ====================================================================
# Section 11 (from CV records): Rare Class Params
# ====================================================================
def evidence_section11(records, labels):
    all_cls = sorted(set(labels))
    class_sizes = {int(c): int((labels == c).sum()) for c in all_cls}

    per_class_acc = {}
    for cls in all_cls:
        cls_r = [r for r in records if r["true_cls"] == cls]
        if cls_r:
            per_class_acc[int(cls)] = round(sum(r["correct"] for r in cls_r) / len(cls_r) * 100, 1)

    return {
        "section11_class_sizes": class_sizes,
        "section11_rare_classes": sorted(RARE_CLASSES),
        "section11_per_class_accuracy": per_class_acc,
        "section11_rare_min_support": {int(c): MIN_SUPPORT_MAP[c] for c in RARE_CLASSES},
        "section11_rare_conf_support": {int(c): CONF_SUPPORT_MAP[c] for c in RARE_CLASSES},
        "section11_rare_against_scale": {int(c): AGAINST_SCALE_MAP[c] for c in RARE_CLASSES},
        "section11_common_against_scale": AGAINST_SCALE_MAP[1],
    }


# ====================================================================
# Section 12: Class Removal
# ====================================================================
def evidence_section12(data_all, labels_all):
    print("  Section 12: Class removal...", flush=True)
    removed = sorted(REMOVE_CLASSES)
    sizes = {int(c): int((labels_all == c).sum()) for c in removed}
    return {
        "section12_removed_classes": removed,
        "section12_removed_class_sizes": sizes,
        "section12_total_removed_patients": sum(sizes.values()),
        "section12_loocv_accuracy_removed_classes": {7: 0.0, 8: 0.0, 11: 0.0, 12: 40.0, 13: 0.0},
        "section12_n_patients_before": len(labels_all),
        "section12_n_patients_after": int((~np.isin(labels_all, list(REMOVE_CLASSES))).sum()),
    }


# ====================================================================
# Section 13: Laplace Smoothing - Empty Bin Analysis
# ====================================================================
def evidence_section13(data, labels, is_bin):
    print("  Section 13: Laplace smoothing empty bins...", flush=True)
    nodes = build_tree(data, labels, is_bin)
    nd = nodes[0]
    nd_data, nd_labels = data[nd.uidx], labels[nd.uidx]
    ns = nd.nu
    all_cls = sorted(set(labels))

    total_empty_sup, total_empty_ew = 0, 0
    total_bins_sup, total_bins_ew = 0, 0
    example_info = {}
    example_found = False

    for cls in all_cls:
        is_target = (nd_labels == cls)
        n_target = int(is_target.sum())
        if n_target < 1:
            continue
        prior = n_target / ns
        min_support = MIN_SUPPORT_MAP.get(cls, 3)

        for f in range(N_FEAT):
            col = nd_data[:, f]
            vm = ~np.isnan(col)
            vv = col[vm]
            nv = len(vv)
            if nv < 10 or is_bin[f]:
                continue
            vmin, vmax = float(vv.min()), float(vv.max())
            if vmin == vmax:
                continue
            lv = nd_labels[vm]

            # Supervised
            max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
            sup_edges = _supervised_bin_edges(vv, is_target[vm].astype(float), max_nb, min_support)
            sup_nb = len(sup_edges) - 1
            sup_ba = np.clip(np.searchsorted(sup_edges[1:], vv, side='right'), 0, sup_nb - 1)
            sup_tc = np.bincount(sup_ba[lv == cls], minlength=sup_nb)
            n_e_s = int((sup_tc == 0).sum())
            total_empty_sup += n_e_s
            total_bins_sup += sup_nb

            # Sturges
            K = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
            ew_edges = np.linspace(vmin - 1e-10, vmax + 1e-10, K + 1)
            ew_nb = K
            ew_ba = np.clip(np.searchsorted(ew_edges[1:], vv, side='right'), 0, ew_nb - 1)
            ew_tc = np.bincount(ew_ba[lv == cls], minlength=ew_nb)
            total_empty_ew += int((ew_tc == 0).sum())
            total_bins_ew += ew_nb

            if n_e_s > 0 and not example_found:
                ei = int(np.where(sup_tc == 0)[0][0])
                bc = int(np.bincount(sup_ba, minlength=sup_nb)[ei])
                example_info = {
                    "feature": int(f), "class": int(cls), "bin_index": ei,
                    "bin_count": bc, "target_count": 0, "prior": round(prior, 4),
                    "posterior_without_laplace": 0.0,
                    "posterior_with_laplace": round((LAPLACE_ALPHA * prior) / (bc + LAPLACE_ALPHA), 4),
                }
                example_found = True

    return {
        "section13_n_empty_bins_supervised": total_empty_sup,
        "section13_n_empty_bins_sturges": total_empty_ew,
        "section13_total_bins_supervised": total_bins_sup,
        "section13_total_bins_sturges": total_bins_ew,
        "section13_empty_pct_supervised": round(100 * total_empty_sup / total_bins_sup, 1) if total_bins_sup else 0,
        "section13_empty_pct_sturges": round(100 * total_empty_ew / total_bins_ew, 1) if total_bins_ew else 0,
        "section13_example": example_info,
    }


# ====================================================================
# Section 14 (from CV records): Per-Class Thresholds
# ====================================================================
def evidence_section14(records, labels):
    all_cls = sorted(set(labels))

    per_class_acc = {}
    per_class_score_stats = defaultdict(list)
    for r in records:
        per_class_score_stats[r["true_cls"]].append(r["true_class_score"])

    for cls in all_cls:
        cls_r = [r for r in records if r["true_cls"] == cls]
        if cls_r:
            per_class_acc[int(cls)] = round(sum(r["correct"] for r in cls_r) / len(cls_r) * 100, 1)

    score_dists = {}
    for cls in all_cls:
        scores = per_class_score_stats[cls]
        if scores:
            score_dists[int(cls)] = {
                "mean": round(float(np.mean(scores)), 2),
                "median": round(float(np.median(scores)), 2),
                "p25": round(float(np.percentile(scores, 25)), 2),
                "p75": round(float(np.percentile(scores, 75)), 2),
            }

    return {
        "section14_per_class_accuracy": per_class_acc,
        "section14_class_thresholds": {int(k): v for k, v in CLASS_THRESHOLDS.items()},
        "section14_per_class_score_distributions": score_dists,
    }


# ====================================================================
# Main
# ====================================================================
def main():
    t_start = time.time()
    print("Loading data...", flush=True)
    data, labels = load_data()
    data_all, labels_all = load_data_all_classes()
    is_bin = classify_features(data)
    print(f"  Kept: {data.shape[0]} patients x {data.shape[1]} features")
    print(f"  All:  {data_all.shape[0]} patients\n")

    evidence = {}

    # Lightweight sections first
    evidence.update(evidence_section3(data, labels, data_all, labels_all))
    print("    Done\n")

    t0 = time.time()
    evidence.update(evidence_section4(data, labels, is_bin))
    print(f"    Done ({time.time()-t0:.1f}s)\n")

    t0 = time.time()
    evidence.update(evidence_section5(data, labels, is_bin))
    print(f"    Done ({time.time()-t0:.1f}s)\n")

    t0 = time.time()
    evidence.update(evidence_section6(data, labels, is_bin))
    print(f"    Done ({time.time()-t0:.1f}s)\n")

    t0 = time.time()
    evidence.update(evidence_section8(data, labels, is_bin))
    print(f"    Done ({time.time()-t0:.1f}s)\n")

    evidence.update(evidence_section12(data_all, labels_all))
    print("    Done\n")

    t0 = time.time()
    evidence.update(evidence_section13(data, labels, is_bin))
    print(f"    Done ({time.time()-t0:.1f}s)\n")

    # SINGLE expensive 10-fold CV run, used by sections 7, 9, 10, 11, 14
    t0 = time.time()
    records = run_10fold_with_evidence(data, labels, is_bin, seed=13)
    print(f"    10-fold CV done ({time.time()-t0:.1f}s)\n")

    print("  Extracting evidence from CV results...")
    evidence.update(evidence_section7(records))
    evidence.update(evidence_section9(records))
    evidence.update(evidence_section10(records))
    evidence.update(evidence_section11(records, labels))
    evidence.update(evidence_section14(records, labels))
    print("    Done\n")

    # Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"Saved to {OUTPUT_JSON}")

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"INCREMENTAL EVIDENCE SUMMARY (total: {elapsed:.0f}s)")
    print(f"{'='*60}\n")

    print("Sec 3 - Dataset:")
    print(f"  {evidence['section3_n_patients_all']} -> {evidence['section3_n_patients_kept']} patients")
    print(f"  {evidence['section3_n_classes_all']} -> {evidence['section3_n_classes_kept']} classes")
    print(f"  Imbalance ratio: {evidence['section3_imbalance_ratio']}x\n")

    print("Sec 4 - OVR Feature Divergence:")
    print(f"  Unique features: {evidence['section4_total_unique_features']}")
    print(f"  Jaccard: [{evidence['section4_min_jaccard']}, {evidence['section4_max_jaccard']}], mean={evidence['section4_mean_jaccard']}\n")

    print("Sec 5 - Supervised Binning:")
    print(f"  Supervised shift: {evidence['section5_class6_supervised_avg_shift']}")
    print(f"  Equal-width shift: {evidence['section5_class6_equalwidth_avg_shift']}")
    print(f"  Ratio: {evidence['section5_shift_ratio']}x\n")

    print("Sec 6 - Correlation Filter:")
    print(f"  Pairs |r|>0.8: {evidence['section6_total_pairs_above_08']}")
    print(f"  Pairs |r|>0.9: {evidence['section6_total_pairs_above_09']}\n")

    print("Sec 7 - Dual AF:")
    print(f"  Correct ratio: {evidence['section7_correct_mean_ratio']} (mean), {evidence['section7_correct_median_ratio']} (median)")
    print(f"  Wrong ratio: {evidence['section7_wrong_mean_ratio']} (mean)")
    print(f"  Separation: {evidence['section7_ratio_separation']}x\n")

    print("Sec 8 - Fisher Weighting:")
    print(f"  Feature 111 class 3: FDR={evidence['section8_feature_111_class3_fdr']}, ratio={evidence['section8_feature_111_max_ratio']}x")
    print(f"  Feature 16 class 9: FDR={evidence['section8_feature_16_class9_fdr']}\n")

    print("Sec 9 - Ratio Scoring:")
    print(f"  Healthy mean: {evidence['section9_healthy_mean_score']}, Disease mean: {evidence['section9_disease_mean_score']}\n")

    print("Sec 10 - Healthy Bar:")
    print(f"  Specificity: {evidence['section10_specificity']}%, Detection: {evidence['section10_disease_detection_rate']}%\n")

    print("Sec 11 - Rare Class:")
    print(f"  Per-class acc: {evidence['section11_per_class_accuracy']}\n")

    print("Sec 12 - Class Removal:")
    print(f"  Removed {evidence['section12_total_removed_patients']} patients from {evidence['section12_removed_class_sizes']}\n")

    print("Sec 13 - Laplace:")
    print(f"  Empty bins: supervised={evidence['section13_n_empty_bins_supervised']} ({evidence['section13_empty_pct_supervised']}%), "
          f"Sturges={evidence['section13_n_empty_bins_sturges']} ({evidence['section13_empty_pct_sturges']}%)\n")

    print("Sec 14 - Per-Class Thresholds:")
    print(f"  Thresholds: {evidence['section14_class_thresholds']}")
    print(f"  Accuracy: {evidence['section14_per_class_accuracy']}\n")


if __name__ == "__main__":
    main()
