"""Parallel AUC/ROC computation across all protocols and seeds.
Uses ProcessPoolExecutor to parallelize across seeds.
Combines binary and multiclass AUC in a single prediction pass (2x speedup).
"""
import json, sys, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import (load_data, classify_features, build_tree, train, predict, HEALTHY)

SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]


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


def compute_auc_single(args):
    """Compute binary + multiclass AUC for one (seed, protocol) pair in a single pass."""
    seed, protocol = args
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    n = data.shape[0]

    all_scores_dict = {cls: [] for cls in all_cls}
    all_disease_scores = []
    all_true = []
    all_true_binary = []

    if protocol == "10fold":
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        folds = np.array_split(idx, 10)
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
                all_disease_scores.append(disease_score)
                all_true_binary.append(0 if true_cls == HEALTHY else 1)
                all_true.append(true_cls)
                for cls in all_cls:
                    all_scores_dict[cls].append(scores.get(cls, 0.0))
    else:
        train_frac = 0.9 if protocol == "90_10" else 0.6
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        split = int(n * train_frac)
        train_idx, test_idx = idx[:split], idx[split:]
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        train_result = train(nodes, td, tl, is_bin, all_cls)
        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict(uid, data, nodes, all_cls, train_result)
            h_score = scores.get(HEALTHY, 1.0)
            max_disease = max((s for c, s in scores.items() if c != HEALTHY), default=0)
            disease_score = max_disease / (h_score + 0.1)
            all_disease_scores.append(disease_score)
            all_true_binary.append(0 if true_cls == HEALTHY else 1)
            all_true.append(true_cls)
            for cls in all_cls:
                all_scores_dict[cls].append(scores.get(cls, 0.0))

    binary_auc = manual_auc(all_true_binary, all_disease_scores)

    per_class_auc = {}
    for cls in all_cls:
        y_true_bin = [1 if t == cls else 0 for t in all_true]
        per_class_auc[cls] = manual_auc(y_true_bin, all_scores_dict[cls])

    weights = {cls: sum(1 for t in all_true if t == cls) for cls in all_cls}
    total = sum(weights.values())
    macro_auc = float(np.mean(list(per_class_auc.values())))
    weighted_auc = sum(per_class_auc[cls] * weights[cls] for cls in all_cls) / total

    return {
        "seed": seed, "protocol": protocol,
        "binary_auc": round(binary_auc, 4),
        "macro_auc": round(macro_auc, 4),
        "weighted_auc": round(weighted_auc, 4),
        "per_class_auc": {str(c): round(v, 4) for c, v in per_class_auc.items()},
    }


def main():
    t0 = time.perf_counter()
    print("AUC/ROC Parallel Computation", flush=True)
    print(f"  Protocols: 10fold, 90/10, 60/40", flush=True)
    print(f"  Seeds: {SEEDS}", flush=True)

    tasks = []
    for protocol in ["10fold", "90_10", "60_40"]:
        for seed in SEEDS:
            tasks.append((seed, protocol))

    print(f"  Total tasks: {len(tasks)}", flush=True)
    print(f"  Using {min(6, len(tasks))} parallel workers", flush=True)

    results_by_protocol = {"10fold": [], "90_10": [], "60_40": []}

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(compute_auc_single, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            results_by_protocol[result["protocol"]].append(result)
            done += 1
            print(f"  [{done}/{len(tasks)}] {result['protocol']} seed={result['seed']} "
                  f"binary={result['binary_auc']:.4f} macro={result['macro_auc']:.4f}", flush=True)

    all_cls = sorted(set(load_data()[1]))
    auc_output = {}
    for protocol in ["10fold", "90_10", "60_40"]:
        entries = results_by_protocol[protocol]
        binary_aucs = [e["binary_auc"] for e in entries]
        macro_aucs = [e["macro_auc"] for e in entries]
        weighted_aucs = [e["weighted_auc"] for e in entries]
        per_cls = {str(c): [] for c in all_cls}
        for e in entries:
            for c in all_cls:
                per_cls[str(c)].append(e["per_class_auc"][str(c)])

        auc_output[protocol] = {
            "binary_auc_mean": round(float(np.mean(binary_aucs)), 4),
            "binary_auc_std": round(float(np.std(binary_aucs)), 4),
            "macro_auc_mean": round(float(np.mean(macro_aucs)), 4),
            "macro_auc_std": round(float(np.std(macro_aucs)), 4),
            "weighted_auc_mean": round(float(np.mean(weighted_aucs)), 4),
            "weighted_auc_std": round(float(np.std(weighted_aucs)), 4),
            "per_class_auc_mean": {c: round(float(np.mean(v)), 4) for c, v in per_cls.items()},
            "seeds": entries,
        }

    out_path = os.path.join(os.path.dirname(__file__), "results_auc.json")
    with open(out_path, "w") as f:
        json.dump(auc_output, f, indent=2)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
