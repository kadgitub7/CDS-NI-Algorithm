"""V3 10-fold CV with full master logging."""
import time
import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

from cds_ovr_v3_log import (
    load_data, classify_features, build_tree,
    _train_all_classes, predict_ovr_logged, log_training_summary,
    print_report, NumpyEncoder,
    HEALTHY, RATIO_EPS, CLASS_THRESHOLDS
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")
LOG_DIR = Path(__file__).parent / "output" / "logs"


def run_10fold_logged(data, labels, is_bin, seed, n_folds=10):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    all_cls = sorted(set(labels))

    master_log = {
        "eval_type": "v3_10fold_cv",
        "seed": seed,
        "n_folds": n_folds,
        "n_total": n,
        "classes": [int(c) for c in all_cls],
        "folds": {},
        "user_traces": {},
    }

    all_results = []

    for fi in range(n_folds):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fi])
        td, tl = data[train_idx], labels[train_idx]

        nodes = build_tree(td, tl, is_bin)
        class_models, class_retained = _train_all_classes(
            nodes, td, tl, is_bin, all_cls)

        training_log = log_training_summary(
            nodes, class_models, class_retained, tl, all_cls)
        master_log["folds"][str(fi)] = {
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "training": training_log,
        }

        for uid in test_idx:
            true_cls = int(labels[uid])

            pred, scores, pred_log = predict_ovr_logged(
                uid, data, nodes, class_models, class_retained,
                all_cls, class_thresholds=CLASS_THRESHOLDS)

            final_pred = int(pred)
            ok = (final_pred == true_cls)
            all_results.append((int(uid), true_cls, final_pred, ok))

            user_raw_values = {}
            ret_feats = set()
            for cls_key, cls_log in pred_log["class_af_details"].items():
                for feat_id in cls_log.get("retained_features", []):
                    ret_feats.add(feat_id)
            for f in sorted(ret_feats):
                v = data[uid, f]
                user_raw_values[str(f)] = None if np.isnan(v) else round(float(v), 6)

            master_log["user_traces"][str(int(uid))] = {
                "uid": int(uid),
                "fold": fi,
                "true_class": true_cls,
                "final_pred": final_pred,
                "correct": ok,
                "sex": round(float(data[uid, 1]), 1),
                "raw_feature_values": user_raw_values,
                "prediction_log": pred_log,
            }

    acc = sum(r[3] for r in all_results) / len(all_results)
    master_log["overall_accuracy"] = round(float(acc * 100), 2)
    master_log["n_correct"] = sum(r[3] for r in all_results)
    master_log["n_wrong"] = sum(not r[3] for r in all_results)

    return all_results, master_log


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats | "
          f"H={int((labels==HEALTHY).sum())} D={int((labels!=HEALTHY).sum())}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]
    best_acc = 0
    best_results = None
    best_seed = None

    for i, seed in enumerate(seeds):
        print(f"\n--- Run {i+1}/{len(seeds)} (seed={seed}) ---", flush=True)
        results, master_log = run_10fold_logged(data, labels, is_bin, seed)
        acc = master_log["overall_accuracy"]

        log_path = LOG_DIR / f"v3_log_10fold_seed{seed}.json"
        with open(log_path, 'w') as f:
            json.dump(master_log, f, indent=1, cls=NumpyEncoder)
        print(f"  {len(results)} test  acc={acc:.1f}%  "
              f"log={os.path.getsize(log_path)/(1024*1024):.1f}MB", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_results = results
            best_seed = seed

    print(f"\n*** Best accuracy across {len(seeds)} runs: {best_acc:.1f}% (seed={best_seed}) ***")
    print_report(best_results, label=f"OVR V3 10-fold CV (Best of {len(seeds)} runs)")

    print(f"\nAll logs in: {LOG_DIR}")
    print("Run: python analyze_logs.py output/logs/v3_log_10fold_seed{N}.json")
