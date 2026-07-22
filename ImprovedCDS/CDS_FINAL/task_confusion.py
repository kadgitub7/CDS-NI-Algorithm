"""Confusion matrix and per-class P/R/F1 computation."""
import json, sys, os, time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import (load_data, classify_features, run_10fold, stats, binary_acc)


def main():
    t0 = time.perf_counter()
    print("Confusion Matrix and Classification Metrics", flush=True)
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    n = data.shape[0]

    print("  Running 10-fold CV seed=13...", flush=True)
    res = run_10fold(data, labels, is_bin, seed=13)
    acc, spec, sens, ba = stats(res)
    bacc = binary_acc(res)

    # Confusion matrix
    n_cls = len(all_cls)
    cls_idx = {c: i for i, c in enumerate(all_cls)}
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for _, true, pred, _ in res:
        if true in cls_idx and pred in cls_idx:
            cm[cls_idx[true], cls_idx[pred]] += 1

    # Per-class metrics
    per_class = {}
    for cls in all_cls:
        tp = sum(1 for r in res if r[1] == cls and r[2] == cls)
        fp = sum(1 for r in res if r[1] != cls and r[2] == cls)
        fn = sum(1 for r in res if r[1] == cls and r[2] != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[str(cls)] = {
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
        }

    macro_p = float(np.mean([per_class[str(c)]["precision"] for c in all_cls]))
    macro_r = float(np.mean([per_class[str(c)]["recall"] for c in all_cls]))
    macro_f1 = float(np.mean([per_class[str(c)]["f1"] for c in all_cls]))

    # Print confusion matrix
    print(f"\n  {'':>10s}", end="")
    for c in all_cls:
        print(f"  {c:>5d}", end="")
    print(f"  {'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}")
    for i, c in enumerate(all_cls):
        print(f"  {'Cls '+str(c):>10s}", end="")
        for j in range(n_cls):
            print(f"  {cm[i,j]:>5d}", end="")
        print(f"  {per_class[str(c)]['precision']:>6.3f}  {per_class[str(c)]['recall']:>6.3f}  {per_class[str(c)]['f1']:>6.3f}")

    print(f"\n  Accuracy: {100*acc:.1f}%  Binary: {100*bacc:.1f}%")
    print(f"  Macro P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}", flush=True)

    output = {
        "matrix": cm.tolist(),
        "classes": [int(c) for c in all_cls],
        "per_class": per_class,
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(100*acc, 2),
        "binary_accuracy": round(100*bacc, 2),
        "specificity": round(100*spec, 2),
        "sensitivity": round(100*sens, 2),
        "balanced_accuracy": round(100*ba, 2),
    }

    out_path = os.path.join(os.path.dirname(__file__), "results_confusion.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
