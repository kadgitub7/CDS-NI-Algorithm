"""Compute per-class Precision, Recall, F1 and binary P/R/F1 for all protocols.
Saves to results_detailed_metrics.json.
"""
import json, sys, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import load_data, classify_features, build_tree, train, predict, HEALTHY

SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]

PROTOCOLS = {
    "50_50":   (0.50, "holdout"),
    "60_40":   (0.60, "holdout"),
    "70_30":   (0.70, "holdout"),
    "75_25":   (0.75, "holdout"),
    "80_20":   (0.80, "holdout"),
    "85_15":   (0.85, "holdout"),
    "90_10":   (0.90, "holdout"),
    "10fold":  (None,  "cv"),
}


def compute_one(args):
    seed, protocol_key, train_frac, mode = args
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    n = data.shape[0]

    true_all = []
    pred_all = []

    if mode == "cv":
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
                true_all.append(int(labels[uid]))
                p, _ = predict(uid, data, nodes, all_cls, train_result)
                pred_all.append(p)
    else:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        split = int(n * train_frac)
        train_idx, test_idx = idx[:split], idx[split:]
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        train_result = train(nodes, td, tl, is_bin, all_cls)
        for uid in test_idx:
            true_all.append(int(labels[uid]))
            p, _ = predict(uid, data, nodes, all_cls, train_result)
            pred_all.append(p)

    per_class = {}
    for cls in all_cls:
        tp = sum(1 for t, p in zip(true_all, pred_all) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_all, pred_all) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_all, pred_all) if t == cls and p != cls)
        support = sum(1 for t in true_all if t == cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[str(cls)] = {
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    # Binary: Normal vs Arrhythmia
    b_tp = sum(1 for t, p in zip(true_all, pred_all) if t != HEALTHY and p != HEALTHY)
    b_fp = sum(1 for t, p in zip(true_all, pred_all) if t == HEALTHY and p != HEALTHY)
    b_fn = sum(1 for t, p in zip(true_all, pred_all) if t != HEALTHY and p == HEALTHY)
    b_tn = sum(1 for t, p in zip(true_all, pred_all) if t == HEALTHY and p == HEALTHY)
    b_prec = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
    b_rec = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
    b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0
    b_spec = b_tn / (b_tn + b_fp) if (b_tn + b_fp) > 0 else 0.0
    b_sens = b_rec

    # Macro averages
    macro_prec = np.mean([v["precision"] for v in per_class.values()])
    macro_rec = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    # Weighted averages
    total_support = sum(v["support"] for v in per_class.values())
    w_prec = sum(v["precision"] * v["support"] for v in per_class.values()) / total_support
    w_rec = sum(v["recall"] * v["support"] for v in per_class.values()) / total_support
    w_f1 = sum(v["f1"] * v["support"] for v in per_class.values()) / total_support

    multiclass_acc = sum(1 for t, p in zip(true_all, pred_all) if t == p) / len(true_all)

    return {
        "seed": seed,
        "protocol": protocol_key,
        "n_test": len(true_all),
        "multiclass_acc": round(100 * multiclass_acc, 2),
        "binary_precision": round(b_prec, 4),
        "binary_recall": round(b_rec, 4),
        "binary_f1": round(b_f1, 4),
        "binary_sensitivity": round(b_sens, 4),
        "binary_specificity": round(b_spec, 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(w_prec), 4),
        "weighted_recall": round(float(w_rec), 4),
        "weighted_f1": round(float(w_f1), 4),
        "per_class": per_class,
    }


def main():
    t0 = time.perf_counter()
    print("Per-class & binary metrics for ALL protocols", flush=True)

    tasks = []
    for pkey, (frac, mode) in PROTOCOLS.items():
        for seed in SEEDS:
            tasks.append((seed, pkey, frac, mode))

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

    output = {}
    for pkey in PROTOCOLS:
        entries = sorted(results_by_proto[pkey], key=lambda e: e["seed"])

        def mean_std(key):
            vals = [e[key] for e in entries]
            return round(float(np.mean(vals)), 4), round(float(np.std(vals, ddof=1)), 4)

        m_acc, s_acc = mean_std("multiclass_acc")
        m_bp, s_bp = mean_std("binary_precision")
        m_br, s_br = mean_std("binary_recall")
        m_bf, s_bf = mean_std("binary_f1")
        m_bs, s_bs = mean_std("binary_sensitivity")
        m_bsp, s_bsp = mean_std("binary_specificity")
        m_mp, s_mp = mean_std("macro_precision")
        m_mr, s_mr = mean_std("macro_recall")
        m_mf, s_mf = mean_std("macro_f1")
        m_wp, s_wp = mean_std("weighted_precision")
        m_wr, s_wr = mean_std("weighted_recall")
        m_wf, s_wf = mean_std("weighted_f1")

        output[pkey] = {
            "summary": {
                "multiclass_acc": {"mean": m_acc, "std": s_acc},
                "binary_precision": {"mean": m_bp, "std": s_bp},
                "binary_recall": {"mean": m_br, "std": s_br},
                "binary_f1": {"mean": m_bf, "std": s_bf},
                "binary_sensitivity": {"mean": m_bs, "std": s_bs},
                "binary_specificity": {"mean": m_bsp, "std": s_bsp},
                "macro_precision": {"mean": m_mp, "std": s_mp},
                "macro_recall": {"mean": m_mr, "std": s_mr},
                "macro_f1": {"mean": m_mf, "std": s_mf},
                "weighted_precision": {"mean": m_wp, "std": s_wp},
                "weighted_recall": {"mean": m_wr, "std": s_wr},
                "weighted_f1": {"mean": m_wf, "std": s_wf},
            },
            "seeds": entries,
        }

    out_path = os.path.join(os.path.dirname(__file__), "results_detailed_metrics.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
