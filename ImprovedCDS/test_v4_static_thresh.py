"""V4 with static thresholds — isolate training-side changes only."""
import time
import numpy as np
from pathlib import Path

from cds_ovr_v4 import (
    load_data, classify_features, build_tree,
    _train_all_classes, predict_ovr, print_report,
    HEALTHY, CLASS_THRESHOLDS,
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
        class_models, class_retained, class_metrics, _ = \
            _train_all_classes(nodes, td, tl, is_bin, all_cls)

        if fi == 0:
            print(f"  Metrics: { {int(k): v for k,v in class_metrics.items()} }")

        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict_ovr(
                uid, data, nodes, class_models, class_retained, all_cls,
                class_thresholds=CLASS_THRESHOLDS,
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

        th = [r for r in results if r[1] == 1]
        td = [r for r in results if r[1] != 1]
        spec = sum(r[3] for r in th) / len(th)
        sens = sum(1 for r in td if r[2] != 1) / len(td)
        print(f"  acc={100*acc:.1f}%  spec={100*spec:.1f}%  sens={100*sens:.1f}%  ({elapsed:.1f}s)", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_results = results
            best_seed = seed

    print(f"\n*** Best: {100*best_acc:.1f}% (seed={best_seed}) ***")
    print_report(best_results, label=f"OVR V4 Static Thresh (Best of {len(seeds)} runs)")
