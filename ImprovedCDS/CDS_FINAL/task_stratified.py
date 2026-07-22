"""Stratified evaluation across ALL validation methods - FULLY PARALLEL.
Runs all 60 experiments (6 protocols x 10 seeds) via ProcessPoolExecutor.
"""
import json, sys, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import (load_data, classify_features, build_tree, train, predict,
                     stats, binary_acc, HEALTHY)

SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]


def run_stratified_split(data, labels, is_bin, seed, train_frac):
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


def run_stratified_kfold(data, labels, is_bin, seed, k=10):
    all_cls = sorted(set(labels))
    rng = np.random.RandomState(seed)
    fold_indices = [[] for _ in range(k)]
    for cls in all_cls:
        cls_indices = np.where(labels == cls)[0]
        perm = rng.permutation(len(cls_indices))
        cls_folds = np.array_split(perm, k)
        for fi in range(k):
            fold_indices[fi].extend(cls_indices[cls_folds[fi]])
    fold_indices = [np.array(f) for f in fold_indices]
    results = []
    for fi in range(k):
        test_idx = fold_indices[fi]
        train_idx = np.concatenate([fold_indices[j] for j in range(k) if j != fi])
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        train_result = train(nodes, td, tl, is_bin, all_cls)
        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict(uid, data, nodes, all_cls, train_result)
            results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def compute_entry(res, all_cls, seed):
    acc, spec, sens, ba = stats(res)
    bacc = binary_acc(res)
    per_class_detail = {}
    for cls in all_cls:
        cr = [r for r in res if r[1] == cls]
        if cr:
            n_correct = int(sum(1 for r in cr if r[3]))
            n_bin_correct = int(sum(1 for r in cr if
                (cls == 1 and r[2] == 1) or (cls != 1 and r[2] != 1)))
            per_class_detail[str(cls)] = {
                "n_test": len(cr),
                "correct": n_correct,
                "multi_acc": round(n_correct / len(cr), 4),
                "binary_correct": n_bin_correct,
                "binary_acc": round(n_bin_correct / len(cr), 4),
            }
    return {
        "seed": seed,
        "accuracy": round(100*acc, 2),
        "binary_acc": round(100*bacc, 2),
        "specificity": round(100*spec, 2),
        "sensitivity": round(100*sens, 2),
        "balanced_acc": round(100*ba, 2),
        "n_test": len(res),
        "per_class": per_class_detail,
    }


def run_single(args):
    protocol, seed = args
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))

    if protocol == "10fold":
        res = run_stratified_kfold(data, labels, is_bin, seed, k=10)
    else:
        frac = float(protocol.split("_")[0]) / 100.0
        res = run_stratified_split(data, labels, is_bin, seed, frac)

    entry = compute_entry(res, all_cls, seed)
    return protocol, seed, entry


def summarize(entries):
    accs = [r["accuracy"] for r in entries]
    baccs = [r["binary_acc"] for r in entries]
    bas = [r["balanced_acc"] for r in entries]
    specs = [r["specificity"] for r in entries]
    senss = [r["sensitivity"] for r in entries]
    return {
        "acc_mean": round(float(np.mean(accs)), 2),
        "acc_std": round(float(np.std(accs)), 2),
        "acc_best": round(float(np.max(accs)), 2),
        "acc_worst": round(float(np.min(accs)), 2),
        "binary_mean": round(float(np.mean(baccs)), 2),
        "binary_std": round(float(np.std(baccs)), 2),
        "binary_best": round(float(np.max(baccs)), 2),
        "ba_mean": round(float(np.mean(bas)), 2),
        "ba_std": round(float(np.std(bas)), 2),
        "spec_mean": round(float(np.mean(specs)), 2),
        "sens_mean": round(float(np.mean(senss)), 2),
    }


def main():
    t0 = time.perf_counter()
    print("=" * 60, flush=True)
    print("STRATIFIED EVALUATION - ALL METHODS (PARALLEL)", flush=True)
    print("=" * 60, flush=True)

    protocols = ["10fold", "90_10", "80_20", "70_30", "60_40", "50_50"]
    tasks = []
    for proto in protocols:
        for seed in SEEDS:
            tasks.append((proto, seed))

    print(f"  Total tasks: {len(tasks)}", flush=True)
    print(f"  Protocols: {protocols}", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)
    print(f"  Using 10 parallel workers", flush=True)

    results_by_protocol = {p: [] for p in protocols}
    done = 0

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(run_single, t): t for t in tasks}
        for future in as_completed(futures):
            proto, seed, entry = future.result()
            results_by_protocol[proto].append(entry)
            done += 1
            print(f"  [{done}/{len(tasks)}] {proto} seed={seed}  "
                  f"multi={entry['accuracy']}%  binary={entry['binary_acc']}%  "
                  f"BA={entry['balanced_acc']}%", flush=True)

    output = {}
    proto_labels = {
        "10fold": "stratified_10fold",
        "90_10": "stratified_90_10",
        "80_20": "stratified_80_20",
        "70_30": "stratified_70_30",
        "60_40": "stratified_60_40",
        "50_50": "stratified_50_50",
    }
    for proto in protocols:
        entries = sorted(results_by_protocol[proto], key=lambda x: x["seed"])
        key = proto_labels[proto]
        s = summarize(entries)
        output[key] = {"summary": s, "seeds": entries}
        label = proto.replace("_", "/") if proto != "10fold" else "10-Fold CV"
        print(f"\n  {label}: multi={s['acc_mean']}+/-{s['acc_std']}%  "
              f"binary={s['binary_mean']}%  BA={s['ba_mean']}%", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), "results_stratified.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
