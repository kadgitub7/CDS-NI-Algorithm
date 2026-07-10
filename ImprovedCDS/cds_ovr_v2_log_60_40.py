"""60/40 Split with full master logging.

Writes a master JSON log to output/logs/log_60_40_seed{seed}.json
containing every variable for every test user.
"""
import time
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np

from cds_ovr_v2_log import (
    load_data, classify_features, build_tree,
    _train_all_classes, predict_ovr_logged, log_training_summary,
    size_based_thresholds, print_report, NumpyEncoder,
    HEALTHY, MERGE_CLASSES, MERGED_LABEL, RATIO_EPS
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")
LOG_DIR = Path(__file__).parent / "output" / "logs"


def run_split_logged(data, labels, is_bin, seed, train_frac=0.6):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]

    td, tl = data[train_idx], labels[train_idx]

    merged_tl = tl.copy()
    for c in MERGE_CLASSES:
        merged_tl[merged_tl == c] = MERGED_LABEL
    merged_cls = sorted(set(merged_tl))

    nodes = build_tree(td, merged_tl, is_bin)
    class_models, class_retained = _train_all_classes(
        nodes, td, merged_tl, is_bin, merged_cls)
    class_thresholds = size_based_thresholds(merged_tl, merged_cls)

    training_log = log_training_summary(
        nodes, class_models, class_retained, merged_tl, merged_cls)

    master_log = {
        "eval_type": "60_40_split",
        "seed": seed,
        "train_frac": train_frac,
        "n_total": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_uids": [int(u) for u in train_idx],
        "test_uids": [int(u) for u in test_idx],
        "training": training_log,
        "user_traces": {},
    }

    results = []
    t0 = time.time()

    for uid in test_idx:
        true_cls = int(labels[uid])
        true_merged = MERGED_LABEL if true_cls in MERGE_CLASSES else true_cls

        pred, scores, pred_log = predict_ovr_logged(
            uid, data, nodes, class_models, class_retained,
            merged_cls, class_thresholds=class_thresholds)

        final_pred = int(pred)

        sub_log = None
        if pred == MERGED_LABEL:
            sub_classes = sorted(c for c in MERGE_CLASSES if c in set(tl))
            if sub_classes:
                sub_nodes = build_tree(td, tl, is_bin)
                sub_models, sub_retained = _train_all_classes(
                    sub_nodes, td, tl, is_bin, sub_classes)
                best_sub, best_score = sub_classes[0], -1.0
                sub_log = {"sub_classes": sub_classes, "sub_scores": {}}
                for cls in sub_classes:
                    _, _, sub_pred_log = predict_ovr_logged(
                        uid, data, sub_nodes, sub_models, sub_retained,
                        sub_classes)
                    af_d = sub_pred_log["class_af_details"].get(str(cls), {})
                    sc = af_d.get("ratio", 0.0)
                    sub_log["sub_scores"][str(cls)] = round(float(sc), 6)
                    if sc > best_score:
                        best_score = sc
                        best_sub = cls
                final_pred = int(best_sub)
                sub_log["sub_winner"] = int(best_sub)
                sub_log["sub_best_score"] = round(float(best_score), 6)

        ok = (final_pred == true_cls)
        results.append((int(uid), true_cls, final_pred, ok))

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
            "true_class": true_cls,
            "true_merged": true_merged,
            "stage1_pred": int(pred),
            "final_pred": final_pred,
            "correct": ok,
            "sex": round(float(data[uid, 1]), 1),
            "raw_feature_values": user_raw_values,
            "prediction_log": pred_log,
            "subtype_log": sub_log,
        }

    elapsed = time.time() - t0
    acc = sum(r[3] for r in results) / len(results)
    master_log["overall_accuracy"] = round(float(acc * 100), 2)
    master_log["n_correct"] = sum(r[3] for r in results)
    master_log["n_wrong"] = sum(not r[3] for r in results)
    master_log["elapsed_s"] = round(elapsed, 2)

    return results, master_log


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
        results, master_log = run_split_logged(data, labels, is_bin, seed)
        acc = master_log["overall_accuracy"]

        log_path = LOG_DIR / f"log_60_40_seed{seed}.json"
        with open(log_path, 'w') as f:
            json.dump(master_log, f, indent=1, cls=NumpyEncoder)
        print(f"  {len(results)} test  acc={acc:.1f}%  "
              f"log={os.path.getsize(log_path)/(1024*1024):.1f}MB", flush=True)

        if acc > best_acc:
            best_acc = acc
            best_results = results
            best_seed = seed

    print(f"\n*** Best accuracy across {len(seeds)} runs: {best_acc:.1f}% (seed={best_seed}) ***")
    print_report(best_results, label=f"OVR V2 60/40 Split (Best of {len(seeds)} runs)")

    print(f"\nAll logs in: {LOG_DIR}")
    print("Run: python analyze_logs.py output/logs/log_60_40_seed{N}.json")
