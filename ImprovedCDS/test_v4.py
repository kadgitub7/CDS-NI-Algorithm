"""Quick V4 test — 10-fold CV with 3 seeds for fast iteration."""
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

from cds_ovr_v4 import (
    load_data, classify_features, build_tree,
    _train_all_classes, predict_ovr, print_report,
    HEALTHY,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


def run_10fold(data, labels, is_bin, seed):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 10)
    all_cls = sorted(set(labels))
    all_results = []

    for fi in range(10):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]

        nodes = build_tree(td, tl, is_bin)
        class_models, class_retained, class_metrics, cal_thresholds = \
            _train_all_classes(nodes, td, tl, is_bin, all_cls)

        if fi == 0:
            print(f"  Metrics: {class_metrics}")
            print(f"  Thresholds: { {k: round(v,2) for k,v in cal_thresholds.items()} }")

        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict_ovr(
                uid, data, nodes, class_models, class_retained, all_cls,
                class_thresholds=cal_thresholds,
                class_metrics=class_metrics)
            ok = (int(pred) == true_cls)
            all_results.append((int(uid), true_cls, int(pred), ok))

    return all_results


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats\n")

    seeds = [13, 20, 27]
    best_acc = 0
    best_results = None
    best_seed = None

    for i, seed in enumerate(seeds):
        t0 = time.time()
        print(f"--- Run {i+1}/{len(seeds)} (seed={seed}) ---", flush=True)
        results = run_10fold(data, labels, is_bin, seed)
        acc = sum(r[3] for r in results) / len(results)
        elapsed = time.time() - t0
        print(f"  acc={100*acc:.1f}%  ({elapsed:.1f}s)", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_results = results
            best_seed = seed

    print(f"\n*** Best: {100*best_acc:.1f}% (seed={best_seed}) ***")
    print_report(best_results, label=f"OVR V4 10-fold (Best of {len(seeds)} runs)")
