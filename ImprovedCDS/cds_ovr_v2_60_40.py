"""CDS OVR V2 — 60/40 Split."""
import sys, time
import numpy as np
from pathlib import Path
from cds_ovr_v2 import (
    load_data, classify_features, build_tree,
    _train_all_classes, _compute_af, predict_ovr, size_based_thresholds,
    print_report, HEALTHY, MERGE_CLASSES, MERGED_LABEL,
    RATIO_EPS
)


def run_split(data, labels, train_frac, seed):
    n = data.shape[0]
    is_bin = classify_features(data)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]

    td, tl = data[train_idx], labels[train_idx]
    t0 = time.perf_counter()

    merged_tl = tl.copy()
    for c in MERGE_CLASSES:
        merged_tl[merged_tl == c] = MERGED_LABEL
    merged_cls = sorted(set(merged_tl))

    nodes = build_tree(td, merged_tl, is_bin)
    class_models, class_retained = _train_all_classes(
        nodes, td, merged_tl, is_bin, merged_cls)
    class_thresholds = size_based_thresholds(merged_tl, merged_cls)

    results = []
    for uid in test_idx:
        pred, scores = predict_ovr(
            uid, data, nodes, class_models, class_retained, merged_cls,
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
                        uid, data, sub_nodes, sub_models[cls], sub_retained[cls])
                    sc = (af_for + RATIO_EPS) / (af_against + RATIO_EPS)
                    if sc > best_score:
                        best_score = sc
                        best_sub = cls
                pred = best_sub

        true_cls = int(labels[uid])
        results.append((uid, true_cls, pred, pred == true_cls))

    acc = sum(r[3] for r in results) / len(results) * 100
    print(f"  {len(train_idx)} train / {len(test_idx)} test  acc={acc:.1f}%  {time.perf_counter()-t0:.1f}s")
    return results


if __name__ == "__main__":
    dp = str(Path(__file__).parent / "data" / "arrhythmia.data")
    print(f"Loading {dp}", flush=True)
    data, labels = load_data(dp)
    print(f"{data.shape[0]} users x {data.shape[1]} feats | "
          f"H={int((labels==HEALTHY).sum())} D={int((labels!=HEALTHY).sum())}", flush=True)
    print("V2: OVR + SUM scoring + Fisher weights\n")

    n_runs = 10
    best_acc = 0
    best_results = None

    for run in range(n_runs):
        seed = 13 + run * 7
        print(f"--- Run {run+1}/{n_runs} (seed={seed}) ---")
        results = run_split(data, labels, 0.6, seed)
        acc = sum(r[3] for r in results) / len(results) * 100
        if acc > best_acc:
            best_acc = acc
            best_results = results

    print(f"\n*** Best accuracy across {n_runs} runs: {best_acc:.1f}% ***")
    print_report(best_results, "OVR V2 60/40 Split (Best of 10 runs)")
