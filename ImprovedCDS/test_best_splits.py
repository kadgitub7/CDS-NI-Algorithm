"""Test W2-05 (SV + against_scale 0.8) on all three evaluation protocols."""
import time
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from experiment_runner import (
    load_data, classify_features, build_tree,
    train_all_cfg, predict_cfg, stats,
    HEALTHY, CLASS_THRESHOLDS,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


def train_best(nodes, td, tl, is_bin, all_cls):
    return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

def predict_best(uid, data, nodes, all_cls, tr):
    cm, cr = tr
    return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)


def run_10fold(data, labels, is_bin, seed):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 10)
    all_cls = sorted(set(labels))
    results = []
    for fi in range(10):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        tr = train_best(nodes, td, tl, is_bin, all_cls)
        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict_best(uid, data, nodes, all_cls, tr)
            results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def run_split(data, labels, is_bin, seed, train_frac):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    td, tl = data[train_idx], labels[train_idx]
    all_cls = sorted(set(labels))
    nodes = build_tree(td, tl, is_bin)
    tr = train_best(nodes, td, tl, is_bin, all_cls)
    results = []
    for uid in test_idx:
        true_cls = int(labels[uid])
        pred, scores = predict_best(uid, data, nodes, all_cls, tr)
        results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def print_report(results, label=""):
    n = len(results)
    correct = sum(r[3] for r in results)
    th = [r for r in results if r[1] == 1]
    td = [r for r in results if r[1] != 1]
    spec = sum(r[3] for r in th) / len(th) if th else 0
    sens = sum(1 for r in td if r[2] != 1) / len(td) if td else 0
    detected = [r for r in td if r[2] != 1]
    subtype_ok = sum(1 for r in detected if r[2] == r[1])
    ppv_pos = [r for r in results if r[2] != 1]
    ppv = sum(1 for r in ppv_pos if r[1] != 1) / len(ppv_pos) if ppv_pos else 0
    npv_neg = [r for r in results if r[2] == 1]
    npv = sum(1 for r in npv_neg if r[1] == 1) / len(npv_neg) if npv_neg else 0

    print(f"\n{'='*70}")
    print(f"  {label} -- {n} users")
    print(f"{'='*70}")
    print(f"  Overall Accuracy:   {100*correct/n:.1f}%")
    ba_cls = {}
    for cls in sorted(set(r[1] for r in results)):
        cr = [r for r in results if r[1] == cls]
        ba_cls[cls] = sum(r[3] for r in cr) / len(cr) if cr else 0
    ba = np.mean(list(ba_cls.values()))
    print(f"  Balanced Accuracy:  {100*ba:.1f}%")
    print(f"\n  Binary (H vs D):")
    print(f"    Accuracy:        {100*(sum(r[3] for r in th)+sum(1 for r in td if r[2]!=1))/(len(th)+len(td)):.1f}%")
    print(f"    Specificity:     {100*spec:.1f}%  ({sum(r[3] for r in th)}/{len(th)})")
    print(f"    Sensitivity:     {100*sens:.1f}%  ({len(detected)}/{len(td)})")
    print(f"    PPV:             {100*ppv:.1f}%")
    print(f"    NPV:             {100*npv:.1f}%")
    if detected:
        print(f"    Subtyping:       {subtype_ok}/{len(detected)} detected = {100*subtype_ok/len(detected):.1f}% correct class")
    print(f"\n  Per-class detail:")
    for cls in sorted(set(r[1] for r in results)):
        cr = [r for r in results if r[1] == cls]
        ok = sum(r[3] for r in cr)
        tot = len(cr)
        det = sum(1 for r in cr if r[2] != 1) if cls != 1 else ok
        mis = defaultdict(int)
        for r in cr:
            if r[2] != r[1]:
                lbl = f"c{r[2]}" if r[2] != 1 else "H"
                mis[lbl] += 1
        mis_str = ", ".join(f"{k}({v})" for k, v in sorted(mis.items(), key=lambda x: -x[1]))
        if cls == 1:
            print(f"    {'healthy':>10s}  {ok:3d}/{tot:3d} = {100*ok/tot:5.1f}%  misclassed as: {mis_str}")
        else:
            print(f"    {'class '+str(cls):>10s}  {ok:3d}/{tot:3d} = {100*ok/tot:5.1f}%  detected: {det}/{tot}  misclassed as: {mis_str}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats\n")

    seeds = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print(f"{'Protocol':20s}  {'Seed':>6s}  {'Acc':>6s}  {'Spec':>6s}  {'Sens':>6s}  {'BA':>6s}")
    print("-" * 75)

    if mode in ("10fold", "all"):
        accs = []
        best_acc, best_r = 0, None
        for seed in seeds:
            t0 = time.time()
            r = run_10fold(data, labels, is_bin, seed)
            acc, spec, sens, ba = stats(r)
            accs.append(acc)
            if acc > best_acc: best_acc, best_r = acc, r
            print(f"{'10-fold CV':20s}  {seed:6d}  {100*acc:5.1f}%  {100*spec:5.1f}%  {100*sens:5.1f}%  {100*ba:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        print(f"{'10-fold MEAN':20s}  {'---':>6s}  {100*np.mean(accs):5.1f}%")
        print_report(best_r, "10-fold CV Best")

    if mode in ("9010", "all"):
        accs = []
        best_acc, best_r = 0, None
        for seed in seeds:
            t0 = time.time()
            r = run_split(data, labels, is_bin, seed, 0.9)
            acc, spec, sens, ba = stats(r)
            accs.append(acc)
            if acc > best_acc: best_acc, best_r = acc, r
            print(f"{'90/10 split':20s}  {seed:6d}  {100*acc:5.1f}%  {100*spec:5.1f}%  {100*sens:5.1f}%  {100*ba:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        print(f"{'90/10 MEAN':20s}  {'---':>6s}  {100*np.mean(accs):5.1f}%")
        print_report(best_r, "90/10 Best")

    if mode in ("6040", "all"):
        accs = []
        best_acc, best_r = 0, None
        for seed in seeds:
            t0 = time.time()
            r = run_split(data, labels, is_bin, seed, 0.6)
            acc, spec, sens, ba = stats(r)
            accs.append(acc)
            if acc > best_acc: best_acc, best_r = acc, r
            print(f"{'60/40 split':20s}  {seed:6d}  {100*acc:5.1f}%  {100*spec:5.1f}%  {100*sens:5.1f}%  {100*ba:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        print(f"{'60/40 MEAN':20s}  {'---':>6s}  {100*np.mean(accs):5.1f}%")
        print_report(best_r, "60/40 Best")
