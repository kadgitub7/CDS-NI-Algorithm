"""Comprehensive experiments for the CDS-OVR report.

Computes:
1. AUC/ROC for 10-fold CV (multiclass OVR and binary)
2. 10-fold CV with full metrics across 10 seeds
3. 90/10 split (binary + multiclass) across 10 seeds
4. 60/40 stratified split across 10 seeds
5. Computational cost (wall time, per-patient time)
6. Memory usage profiling
7. Confusion matrices
8. Per-class precision, recall, F1
"""
import json, time, sys, os, tracemalloc
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import (load_data, classify_features, build_tree, train, predict,
                     run_10fold, run_split, stats, binary_acc, HEALTHY)

SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]
DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


def confusion_matrix(results, all_cls):
    n_cls = len(all_cls)
    cls_idx = {c: i for i, c in enumerate(all_cls)}
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for _, true, pred, _ in results:
        if true in cls_idx and pred in cls_idx:
            cm[cls_idx[true], cls_idx[pred]] += 1
    return cm


def per_class_metrics(results, all_cls):
    metrics = {}
    for cls in all_cls:
        tp = sum(1 for r in results if r[1] == cls and r[2] == cls)
        fp = sum(1 for r in results if r[1] != cls and r[2] == cls)
        fn = sum(1 for r in results if r[1] == cls and r[2] != cls)
        tn = sum(1 for r in results if r[1] != cls and r[2] != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[cls] = {"precision": prec, "recall": rec, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "support": tp + fn}
    return metrics


def compute_auc_roc_binary(data, labels, is_bin, seed=13, protocol="10fold"):
    all_cls = sorted(set(labels))
    if protocol == "10fold":
        n = data.shape[0]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        folds = np.array_split(idx, 10)
        all_scores = []
        all_true = []
        for fi in range(10):
            test_idx = folds[fi]
            train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
            td, tl = data[train_idx], labels[train_idx]
            nodes = build_tree(td, tl, is_bin)
            train_result = train(nodes, td, tl, is_bin, all_cls)
            for uid in test_idx:
                true_cls = int(labels[uid])
                pred, scores = predict(uid, data, nodes, all_cls, train_result)
                h_score = scores.get(HEALTHY, 1.0)
                max_disease = max((s for c, s in scores.items() if c != HEALTHY), default=0)
                disease_score = max_disease / (h_score + 0.1)
                all_scores.append(disease_score)
                all_true.append(0 if true_cls == HEALTHY else 1)
    else:
        train_frac = 0.9 if protocol == "90_10" else 0.6
        n = data.shape[0]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        split = int(n * train_frac)
        train_idx, test_idx = idx[:split], idx[split:]
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        train_result = train(nodes, td, tl, is_bin, all_cls)
        all_scores = []
        all_true = []
        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict(uid, data, nodes, all_cls, train_result)
            h_score = scores.get(HEALTHY, 1.0)
            max_disease = max((s for c, s in scores.items() if c != HEALTHY), default=0)
            disease_score = max_disease / (h_score + 0.1)
            all_scores.append(disease_score)
            all_true.append(0 if true_cls == HEALTHY else 1)

    return manual_auc(all_true, all_scores)


def manual_auc(y_true, y_score):
    pairs = sorted(zip(y_score, y_true), reverse=True)
    tp, fp = 0, 0
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr_list, fpr_list = [0.0], [0.0]
    prev_score = None
    for score, label in pairs:
        if score != prev_score and prev_score is not None:
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    tpr_list.append(tp / n_pos)
    fpr_list.append(fp / n_neg)
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    return auc


def compute_multiclass_auc(data, labels, is_bin, seed=13, protocol="10fold"):
    all_cls = sorted(set(labels))
    if protocol == "10fold":
        n = data.shape[0]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        folds = np.array_split(idx, 10)
        all_scores_dict = {cls: [] for cls in all_cls}
        all_true = []
        for fi in range(10):
            test_idx = folds[fi]
            train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
            td, tl = data[train_idx], labels[train_idx]
            nodes = build_tree(td, tl, is_bin)
            train_result = train(nodes, td, tl, is_bin, all_cls)
            for uid in test_idx:
                true_cls = int(labels[uid])
                pred, scores = predict(uid, data, nodes, all_cls, train_result)
                all_true.append(true_cls)
                for cls in all_cls:
                    all_scores_dict[cls].append(scores.get(cls, 0.0))
    else:
        train_frac = 0.9 if protocol == "90_10" else 0.6
        n = data.shape[0]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        split = int(n * train_frac)
        train_idx, test_idx = idx[:split], idx[split:]
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        train_result = train(nodes, td, tl, is_bin, all_cls)
        all_scores_dict = {cls: [] for cls in all_cls}
        all_true = []
        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict(uid, data, nodes, all_cls, train_result)
            all_true.append(true_cls)
            for cls in all_cls:
                all_scores_dict[cls].append(scores.get(cls, 0.0))

    per_class_auc = {}
    for cls in all_cls:
        y_true_bin = [1 if t == cls else 0 for t in all_true]
        y_scores = all_scores_dict[cls]
        per_class_auc[cls] = manual_auc(y_true_bin, y_scores)

    weights = {cls: sum(1 for t in all_true if t == cls) for cls in all_cls}
    total = sum(weights.values())
    macro_auc = np.mean(list(per_class_auc.values()))
    weighted_auc = sum(per_class_auc[cls] * weights[cls] for cls in all_cls) / total

    return per_class_auc, macro_auc, weighted_auc


def run_stratified_split(data, labels, is_bin, seed=13, train_frac=0.6):
    all_cls = sorted(set(labels))
    rng = np.random.RandomState(seed)
    train_idx_list, test_idx_list = [], []
    for cls in all_cls:
        cls_indices = np.where(labels == cls)[0]
        perm = rng.permutation(len(cls_indices))
        n_train = max(1, int(len(cls_indices) * train_frac))
        train_idx_list.extend(cls_indices[perm[:n_train]])
        test_idx_list.extend(cls_indices[perm[n_train:]])
    train_idx = np.array(train_idx_list)
    test_idx = np.array(test_idx_list)
    td, tl = data[train_idx], labels[train_idx]
    nodes = build_tree(td, tl, is_bin)
    train_result = train(nodes, td, tl, is_bin, all_cls)
    results = []
    for uid in test_idx:
        true_cls = int(labels[uid])
        pred, scores = predict(uid, data, nodes, all_cls, train_result)
        results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def main():
    print("=" * 70)
    print("COMPREHENSIVE CDS-OVR EXPERIMENTS")
    print("=" * 70)

    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    n = data.shape[0]
    print(f"Dataset: {n} patients, {data.shape[1]} features, {len(all_cls)} classes")
    print(f"Classes: {all_cls}")
    print(f"Distribution: { {int(c): int((labels==c).sum()) for c in all_cls} }\n")

    results_all = {}

    # =============================================
    # 1. 10-FOLD CV WITH FULL METRICS (10 seeds)
    # =============================================
    print("=" * 70)
    print("1. 10-FOLD CROSS-VALIDATION (10 seeds)")
    print("=" * 70)
    cv_results = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        res = run_10fold(data, labels, is_bin, seed)
        elapsed = time.perf_counter() - t0
        acc, spec, sens, ba = stats(res)
        bacc = binary_acc(res)
        cm = confusion_matrix(res, all_cls)
        pcm = per_class_metrics(res, all_cls)

        macro_f1 = np.mean([pcm[c]["f1"] for c in all_cls])
        weighted_f1 = sum(pcm[c]["f1"] * pcm[c]["support"] for c in all_cls) / n

        entry = {
            "seed": seed, "accuracy": round(100*acc, 2),
            "binary_acc": round(100*bacc, 2),
            "specificity": round(100*spec, 2),
            "sensitivity": round(100*sens, 2),
            "balanced_acc": round(100*ba, 2),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "time_s": round(elapsed, 1),
            "per_class": {str(c): {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in pcm[c].items()} for c in all_cls}
        }
        cv_results.append(entry)
        print(f"  seed={seed:2d}  acc={100*acc:.1f}%  binary={100*bacc:.1f}%  "
              f"BA={100*ba:.1f}%  F1={macro_f1:.3f}  time={elapsed:.1f}s")

    accs = [r["accuracy"] for r in cv_results]
    baccs = [r["binary_acc"] for r in cv_results]
    bas = [r["balanced_acc"] for r in cv_results]
    f1s = [r["macro_f1"] for r in cv_results]
    results_all["10fold_cv"] = {
        "summary": {
            "acc_mean": round(float(np.mean(accs)), 2),
            "acc_std": round(float(np.std(accs)), 2),
            "acc_best": round(float(np.max(accs)), 2),
            "binary_mean": round(float(np.mean(baccs)), 2),
            "binary_std": round(float(np.std(baccs)), 2),
            "ba_mean": round(float(np.mean(bas)), 2),
            "ba_std": round(float(np.std(bas)), 2),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
        },
        "seeds": cv_results
    }
    print(f"\n  SUMMARY: acc={np.mean(accs):.1f}+/-{np.std(accs):.1f}%  "
          f"binary={np.mean(baccs):.1f}%  BA={np.mean(bas):.1f}%  F1={np.mean(f1s):.3f}\n")

    # =============================================
    # 2. AUC/ROC (Binary and Multiclass)
    # =============================================
    print("=" * 70)
    print("2. AUC/ROC VALUES")
    print("=" * 70)

    auc_results = {}
    for protocol, label in [("10fold", "10-fold CV"), ("90_10", "90/10 split"), ("60_40", "60/40 split")]:
        print(f"\n  --- {label} ---")
        binary_aucs = []
        macro_aucs = []
        weighted_aucs = []
        per_cls_aucs = {c: [] for c in all_cls}

        for seed in SEEDS:
            b_auc = compute_auc_roc_binary(data, labels, is_bin, seed, protocol)
            pc_auc, m_auc, w_auc = compute_multiclass_auc(data, labels, is_bin, seed, protocol)
            binary_aucs.append(b_auc)
            macro_aucs.append(m_auc)
            weighted_aucs.append(w_auc)
            for c in all_cls:
                per_cls_aucs[c].append(pc_auc[c])
            print(f"    seed={seed:2d}  binary_AUC={b_auc:.4f}  macro_AUC={m_auc:.4f}  weighted_AUC={w_auc:.4f}")

        auc_results[protocol] = {
            "binary_auc_mean": round(float(np.mean(binary_aucs)), 4),
            "binary_auc_std": round(float(np.std(binary_aucs)), 4),
            "macro_auc_mean": round(float(np.mean(macro_aucs)), 4),
            "macro_auc_std": round(float(np.std(macro_aucs)), 4),
            "weighted_auc_mean": round(float(np.mean(weighted_aucs)), 4),
            "weighted_auc_std": round(float(np.std(weighted_aucs)), 4),
            "per_class_auc_mean": {str(c): round(float(np.mean(per_cls_aucs[c])), 4) for c in all_cls},
        }
        print(f"  SUMMARY: binary={np.mean(binary_aucs):.4f}+/-{np.std(binary_aucs):.4f}  "
              f"macro={np.mean(macro_aucs):.4f}  weighted={np.mean(weighted_aucs):.4f}")

    results_all["auc_roc"] = auc_results

    # =============================================
    # 3. 90/10 SPLIT (10 seeds)
    # =============================================
    print("\n" + "=" * 70)
    print("3. 90/10 SPLIT (10 seeds)")
    print("=" * 70)
    split90_results = []
    for seed in SEEDS:
        res = run_split(data, labels, is_bin, seed, 0.9)
        acc, spec, sens, ba = stats(res)
        bacc = binary_acc(res)
        pcm = per_class_metrics(res, all_cls)
        macro_f1 = np.mean([pcm[c]["f1"] for c in all_cls if pcm[c]["support"] > 0])
        entry = {
            "seed": seed, "accuracy": round(100*acc, 2),
            "binary_acc": round(100*bacc, 2),
            "specificity": round(100*spec, 2),
            "sensitivity": round(100*sens, 2),
            "balanced_acc": round(100*ba, 2),
            "macro_f1": round(macro_f1, 4),
            "n_test": len(res),
        }
        split90_results.append(entry)
        print(f"  seed={seed:2d}  multi={100*acc:.1f}%  binary={100*bacc:.1f}%  "
              f"spec={100*spec:.1f}%  sens={100*sens:.1f}%  F1={macro_f1:.3f}")

    accs = [r["accuracy"] for r in split90_results]
    baccs = [r["binary_acc"] for r in split90_results]
    results_all["90_10_split"] = {
        "summary": {
            "acc_mean": round(float(np.mean(accs)), 2),
            "acc_std": round(float(np.std(accs)), 2),
            "acc_best": round(float(np.max(accs)), 2),
            "binary_mean": round(float(np.mean(baccs)), 2),
            "binary_std": round(float(np.std(baccs)), 2),
        },
        "seeds": split90_results
    }
    print(f"\n  SUMMARY: multi={np.mean(accs):.1f}+/-{np.std(accs):.1f}%  "
          f"binary={np.mean(baccs):.1f}+/-{np.std(baccs):.1f}%\n")

    # =============================================
    # 4. 60/40 STRATIFIED SPLIT (10 seeds)
    # =============================================
    print("=" * 70)
    print("4. 60/40 STRATIFIED SPLIT (10 seeds)")
    print("=" * 70)
    strat60_results = []
    for seed in SEEDS:
        res = run_stratified_split(data, labels, is_bin, seed, 0.6)
        acc, spec, sens, ba = stats(res)
        bacc = binary_acc(res)
        pcm = per_class_metrics(res, all_cls)
        macro_f1 = np.mean([pcm[c]["f1"] for c in all_cls if pcm[c]["support"] > 0])

        per_class_detail = {}
        for cls in all_cls:
            cr = [r for r in res if r[1] == cls]
            if cr:
                per_class_detail[str(cls)] = {
                    "n_test": len(cr),
                    "correct": sum(r[3] for r in cr),
                    "accuracy": round(sum(r[3] for r in cr) / len(cr), 4)
                }

        entry = {
            "seed": seed, "accuracy": round(100*acc, 2),
            "binary_acc": round(100*bacc, 2),
            "specificity": round(100*spec, 2),
            "sensitivity": round(100*sens, 2),
            "balanced_acc": round(100*ba, 2),
            "macro_f1": round(macro_f1, 4),
            "n_test": len(res),
            "per_class": per_class_detail,
        }
        strat60_results.append(entry)
        print(f"  seed={seed:2d}  multi={100*acc:.1f}%  binary={100*bacc:.1f}%  "
              f"BA={100*ba:.1f}%  F1={macro_f1:.3f}  n_test={len(res)}")

    accs = [r["accuracy"] for r in strat60_results]
    baccs = [r["binary_acc"] for r in strat60_results]
    bas = [r["balanced_acc"] for r in strat60_results]
    results_all["60_40_stratified"] = {
        "summary": {
            "acc_mean": round(float(np.mean(accs)), 2),
            "acc_std": round(float(np.std(accs)), 2),
            "acc_best": round(float(np.max(accs)), 2),
            "binary_mean": round(float(np.mean(baccs)), 2),
            "binary_std": round(float(np.std(baccs)), 2),
            "ba_mean": round(float(np.mean(bas)), 2),
        },
        "seeds": strat60_results
    }
    print(f"\n  SUMMARY: multi={np.mean(accs):.1f}+/-{np.std(accs):.1f}%  "
          f"binary={np.mean(baccs):.1f}%  BA={np.mean(bas):.1f}%\n")

    # =============================================
    # 5. 60/40 STANDARD (non-stratified) SPLIT
    # =============================================
    print("=" * 70)
    print("5. 60/40 STANDARD SPLIT (10 seeds)")
    print("=" * 70)
    std60_results = []
    for seed in SEEDS:
        res = run_split(data, labels, is_bin, seed, 0.6)
        acc, spec, sens, ba = stats(res)
        bacc = binary_acc(res)
        entry = {
            "seed": seed, "accuracy": round(100*acc, 2),
            "binary_acc": round(100*bacc, 2),
            "specificity": round(100*spec, 2),
            "sensitivity": round(100*sens, 2),
            "balanced_acc": round(100*ba, 2),
            "n_test": len(res),
        }
        std60_results.append(entry)
        print(f"  seed={seed:2d}  multi={100*acc:.1f}%  binary={100*bacc:.1f}%  "
              f"spec={100*spec:.1f}%  sens={100*sens:.1f}%")

    accs = [r["accuracy"] for r in std60_results]
    baccs = [r["binary_acc"] for r in std60_results]
    results_all["60_40_standard"] = {
        "summary": {
            "acc_mean": round(float(np.mean(accs)), 2),
            "acc_std": round(float(np.std(accs)), 2),
            "acc_best": round(float(np.max(accs)), 2),
            "binary_mean": round(float(np.mean(baccs)), 2),
        },
        "seeds": std60_results
    }
    print(f"\n  SUMMARY: multi={np.mean(accs):.1f}+/-{np.std(accs):.1f}%  "
          f"binary={np.mean(baccs):.1f}%\n")

    # =============================================
    # 6. COMPUTATIONAL COST AND MEMORY
    # =============================================
    print("=" * 70)
    print("6. COMPUTATIONAL COST AND MEMORY USAGE")
    print("=" * 70)

    # Time 10-fold CV
    t0 = time.perf_counter()
    res = run_10fold(data, labels, is_bin, seed=13)
    t_10fold = time.perf_counter() - t0

    # Time 90/10 split
    t0 = time.perf_counter()
    res = run_split(data, labels, is_bin, seed=13, train_frac=0.9)
    t_90_10 = time.perf_counter() - t0

    # Time single training + prediction with memory tracking
    tracemalloc.start()
    t0 = time.perf_counter()
    nodes = build_tree(data, labels, is_bin)
    train_result = train(nodes, data, labels, is_bin, all_cls)
    t_train = time.perf_counter() - t0
    mem_after_train = tracemalloc.get_traced_memory()

    t0 = time.perf_counter()
    for uid in range(n):
        predict(uid, data, nodes, all_cls, train_result)
    t_predict_all = time.perf_counter() - t0
    mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t_per_patient = t_predict_all / n

    # Memory of data structures
    import sys as _sys
    data_mem_mb = data.nbytes / (1024**2)

    comp_results = {
        "10fold_cv_time_s": round(t_10fold, 2),
        "90_10_split_time_s": round(t_90_10, 2),
        "training_time_s": round(t_train, 4),
        "prediction_all_patients_s": round(t_predict_all, 4),
        "prediction_per_patient_ms": round(t_per_patient * 1000, 4),
        "memory_current_bytes": mem_after_train[0],
        "memory_peak_bytes": mem_peak[1],
        "memory_current_mb": round(mem_after_train[0] / (1024**2), 2),
        "memory_peak_mb": round(mem_peak[1] / (1024**2), 2),
        "data_matrix_mb": round(data_mem_mb, 2),
        "n_tree_nodes": len(nodes),
        "n_class_models": sum(len(train_result[0][c]) for c in all_cls),
        "n_retained_features": sum(len(train_result[1][c]) for c in all_cls),
        "platform": sys.platform,
    }
    results_all["computational_cost"] = comp_results

    print(f"  10-fold CV total time:     {t_10fold:.2f}s")
    print(f"  90/10 split total time:    {t_90_10:.2f}s")
    print(f"  Single training time:      {t_train:.4f}s")
    print(f"  All predictions time:      {t_predict_all:.4f}s")
    print(f"  Per-patient prediction:    {t_per_patient*1000:.4f}ms")
    print(f"  Memory (current):          {mem_after_train[0]/(1024**2):.2f} MB")
    print(f"  Memory (peak):             {mem_peak[1]/(1024**2):.2f} MB")
    print(f"  Data matrix size:          {data_mem_mb:.2f} MB")
    print(f"  Tree nodes:                {len(nodes)}")
    print(f"  Total class models:        {comp_results['n_class_models']}")
    print(f"  Total retained features:   {comp_results['n_retained_features']}")

    # =============================================
    # 7. CONFUSION MATRIX FOR BEST 10-FOLD SEED
    # =============================================
    print("\n" + "=" * 70)
    print("7. CONFUSION MATRIX (10-fold CV, seed=13)")
    print("=" * 70)
    res = run_10fold(data, labels, is_bin, seed=13)
    cm = confusion_matrix(res, all_cls)
    pcm = per_class_metrics(res, all_cls)
    print(f"\n  {'True/Pred':>10s}", end="")
    for c in all_cls:
        print(f"  {c:>5d}", end="")
    print(f"  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
    print("  " + "-" * (10 + 7*len(all_cls) + 24))
    for i, c in enumerate(all_cls):
        print(f"  {'Class '+str(c):>10s}", end="")
        for j in range(len(all_cls)):
            print(f"  {cm[i,j]:>5d}", end="")
        print(f"  {pcm[c]['precision']:>6.3f}  {pcm[c]['recall']:>6.3f}  {pcm[c]['f1']:>6.3f}")

    macro_p = np.mean([pcm[c]["precision"] for c in all_cls])
    macro_r = np.mean([pcm[c]["recall"] for c in all_cls])
    macro_f1 = np.mean([pcm[c]["f1"] for c in all_cls])
    print(f"\n  Macro Precision: {macro_p:.4f}")
    print(f"  Macro Recall:    {macro_r:.4f}")
    print(f"  Macro F1:        {macro_f1:.4f}")

    results_all["confusion_matrix_seed13"] = {
        "matrix": cm.tolist(),
        "classes": [int(c) for c in all_cls],
        "per_class": {str(c): {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in pcm[c].items()} for c in all_cls},
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
    }

    # Save results
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    out_path = os.path.join(os.path.dirname(__file__), "comprehensive_results.json")
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, default=convert)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
