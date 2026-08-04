"""Compute AUC/ROC for missing protocols: 50/50, 70/30, 75/25, 80/20, 85/15.
Saves to results_auc_extra.json.
"""
import json, sys, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import load_data, classify_features, build_tree, train, predict, HEALTHY

SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]

PROTOCOLS = {
    "50_50": 0.50,
    "70_30": 0.70,
    "75_25": 0.75,
    "80_20": 0.80,
    "85_15": 0.85,
}


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


def compute_one(args):
    seed, protocol_key, train_frac = args
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    n = data.shape[0]

    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]

    td, tl = data[train_idx], labels[train_idx]
    nodes = build_tree(td, tl, is_bin)
    train_result = train(nodes, td, tl, is_bin, all_cls)

    all_scores_dict = {cls: [] for cls in all_cls}
    all_disease_scores = []
    all_true = []
    all_true_binary = []

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
        "seed": seed, "protocol": protocol_key,
        "binary_auc": round(binary_auc, 4),
        "macro_auc": round(macro_auc, 4),
        "weighted_auc": round(weighted_auc, 4),
        "per_class_auc": {str(c): round(v, 4) for c, v in per_class_auc.items()},
    }


def main():
    t0 = time.perf_counter()
    print("AUC/ROC for missing protocols", flush=True)

    tasks = []
    for pkey, frac in PROTOCOLS.items():
        for seed in SEEDS:
            tasks.append((seed, pkey, frac))

    print(f"  {len(tasks)} tasks, 6 workers", flush=True)
    results_by_proto = {k: [] for k in PROTOCOLS}

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(compute_one, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            r = future.result()
            results_by_proto[r["protocol"]].append(r)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}]", flush=True)

    all_cls = sorted(set(load_data()[1]))
    output = {}
    for pkey in PROTOCOLS:
        entries = results_by_proto[pkey]
        entries.sort(key=lambda e: e["seed"])
        ba = [e["binary_auc"] for e in entries]
        ma = [e["macro_auc"] for e in entries]
        wa = [e["weighted_auc"] for e in entries]
        pc = {str(c): [e["per_class_auc"][str(c)] for e in entries] for c in all_cls}

        output[pkey] = {
            "binary_auc_mean": round(float(np.mean(ba)), 4),
            "binary_auc_std": round(float(np.std(ba, ddof=1)), 4),
            "macro_auc_mean": round(float(np.mean(ma)), 4),
            "macro_auc_std": round(float(np.std(ma, ddof=1)), 4),
            "weighted_auc_mean": round(float(np.mean(wa)), 4),
            "weighted_auc_std": round(float(np.std(wa, ddof=1)), 4),
            "per_class_auc_mean": {c: round(float(np.mean(v)), 4) for c, v in pc.items()},
            "seeds": entries,
        }

    out_path = os.path.join(os.path.dirname(__file__), "results_auc_extra.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
