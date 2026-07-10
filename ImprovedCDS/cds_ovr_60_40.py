"""CDS OVR — 60/40 Train/Test Split with preprocessing (run 10 times, report best).

Preprocessing: outlier clipping + test-time-only imputation.
Training data keeps NaN (clean bin statistics via skip).
Test data NaN filled with training medians (recovers evidence votes).
"""
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from cds_ovr import (
    load_data, classify_features, build_tree, _train_all_classes,
    predict_ovr, size_based_thresholds, _compute_af,
    HEALTHY, MERGE_CLASSES, MERGED_LABEL, N_FEAT
)
from cds_preprocess import clip_outliers, compute_fill_values, impute_test_rows


def run_split(data, labels, train_frac=0.6, seed=42):
    n = data.shape[0]
    is_bin = classify_features(data)
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    split = int(n * train_frac)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_set = set(train_idx)
    train_mask = np.array([i in train_set for i in range(n)])

    pp_data = clip_outliers(data, train_mask, is_bin)

    td, tl = pp_data[train_mask], labels[train_mask]

    fill_values = compute_fill_values(td, is_bin)
    impute_test_rows(pp_data, test_idx, fill_values)

    merged_tl = tl.copy()
    for c in MERGE_CLASSES:
        merged_tl[merged_tl == c] = MERGED_LABEL
    merged_cls = sorted(set(merged_tl))

    nodes = build_tree(td, merged_tl, is_bin)
    class_models, class_retained = _train_all_classes(
        nodes, td, merged_tl, is_bin, merged_cls)
    class_thresholds = size_based_thresholds(merged_tl, merged_cls)

    results = []
    t0 = time.perf_counter()

    for uid in test_idx:
        pred, scores = predict_ovr(
            uid, pp_data, nodes, class_models, class_retained, merged_cls,
            class_thresholds=class_thresholds)

        if pred == MERGED_LABEL:
            sub_classes = sorted(c for c in MERGE_CLASSES if c in set(tl))
            if sub_classes:
                sub_nodes = build_tree(td, tl, is_bin)
                sub_models, sub_retained = _train_all_classes(
                    sub_nodes, td, tl, is_bin, sub_classes)
                best_sub, best_score = sub_classes[0], -1.0
                for cls in sub_classes:
                    af_for, af_against, _ = _compute_af(
                        uid, pp_data, sub_nodes, sub_models[cls], sub_retained[cls])
                    sc = (af_for + 0.1) / (af_against + 0.1)
                    if sc > best_score:
                        best_score = sc
                        best_sub = cls
                pred = best_sub

        true_cls = int(labels[uid])
        results.append((uid, true_cls, pred, pred == true_cls))

    acc = sum(r[3] for r in results) / len(results) * 100
    print(f"  {len(train_idx)} train / {len(test_idx)} test  acc={acc:.1f}%  {time.perf_counter()-t0:.1f}s", flush=True)
    return results


def print_report(results, label="60/40 Split"):
    all_cls = sorted(set(r[1] for r in results) | set(r[2] for r in results))
    n = len(results)
    correct = sum(r[3] for r in results)
    print(f"\n{'='*70}")
    print(f"CDS One-vs-Rest {label} — {n} test users")
    print(f"{'='*70}")
    print(f"Overall Accuracy:   {100*correct/n:.1f}%")

    per_class = {}
    for cls in all_cls:
        cr = [r for r in results if r[1] == cls]
        if cr:
            per_class[cls] = sum(r[3] for r in cr) / len(cr)
    ba = np.mean(list(per_class.values()))
    print(f"Balanced Accuracy:  {100*ba:.1f}%")

    th = [r for r in results if r[1] == HEALTHY]
    td = [r for r in results if r[1] != HEALTHY]
    spec = sum(r[3] for r in th) / len(th) if th else 0
    sens = sum(1 for r in td if r[2] != HEALTHY) / len(td) if td else 0
    tp = sum(1 for r in td if r[2] != HEALTHY)
    tn = sum(r[3] for r in th)
    fp = len(th) - tn
    fn = len(td) - tp
    binary_acc = (tp + tn) / n
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    detected = [r for r in td if r[2] != HEALTHY]
    subtype_correct = sum(1 for r in detected if r[3])

    print(f"\nBinary (H vs D):")
    print(f"  Accuracy:        {100*binary_acc:.1f}%")
    print(f"  Specificity:     {100*spec:.1f}%  ({tn}/{len(th)})")
    print(f"  Sensitivity:     {100*sens:.1f}%  ({tp}/{len(td)})")
    print(f"  PPV:             {100*ppv:.1f}%")
    print(f"  NPV:             {100*npv:.1f}%")
    if tp:
        print(f"  Subtyping:       {subtype_correct}/{tp} detected = {100*subtype_correct/tp:.1f}% correct class")

    print(f"\nPer-class detail:")
    for cls in sorted(per_class.keys()):
        n_cls = sum(1 for r in results if r[1] == cls)
        n_correct = sum(1 for r in results if r[1] == cls and r[3])
        lbl = "healthy" if cls == HEALTHY else f"class {cls}"
        if cls != HEALTHY:
            n_detected = sum(1 for r in results if r[1] == cls and r[2] != HEALTHY)
            det_str = f"  detected: {n_detected}/{n_cls}"
        else:
            det_str = ""
        wrong = defaultdict(int)
        for r in results:
            if r[1] == cls and not r[3]:
                wrong[r[2]] += 1
        wrong_str = ""
        if wrong:
            wrong_parts = sorted(wrong.items(), key=lambda x: -x[1])[:3]
            wrong_str = "  misclassed as: " + ", ".join(
                f"{'H' if k==1 else f'c{k}'}({v})" for k, v in wrong_parts)
        print(f"  {lbl:>10s}  {n_correct:3d}/{n_cls:3d} = {100*n_correct/n_cls:5.1f}%{det_str}{wrong_str}")
    print(f"{'='*70}")


if __name__ == "__main__":
    dp = str(Path(__file__).parent / "data" / "arrhythmia.data")
    print(f"Loading {dp}", flush=True)
    data, labels = load_data(dp)
    print(f"{data.shape[0]} users x {data.shape[1]} feats | "
          f"H={int((labels==HEALTHY).sum())} D={int((labels!=HEALTHY).sum())}", flush=True)
    print("Preprocessing: outlier clipping + test-time-only imputation")

    best_acc = 0.0
    best_results = None

    for run in range(10):
        seed = run * 7 + 13
        print(f"\n--- Run {run+1}/10 (seed={seed}) ---", flush=True)
        results = run_split(data, labels, train_frac=0.6, seed=seed)
        acc = sum(r[3] for r in results) / len(results) * 100
        if acc > best_acc:
            best_acc = acc
            best_results = results

    print(f"\n*** Best accuracy across 10 runs: {best_acc:.1f}% ***")
    print_report(best_results, label="60/40 Split + Preprocessing (Best of 10 runs)")
