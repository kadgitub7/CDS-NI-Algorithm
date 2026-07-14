"""Validate W11-01 (W9-05 + class 10 threshold=3.0) across all split protocols.

Runs 10-fold CV, 90/10 split, 60/40 split each across 10 seeds.
Also computes binary accuracy for 90/10 split.
"""
import time
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from experiment_runner import (
    load_data, classify_features, build_tree, _route_user,
    train_all_cfg, train_ovr_node_cfg, refine_ovr_node_cfg,
    compute_af_cfg, predict_cfg, score_patient,
    run_10fold, stats, BinModel,
    HEALTHY, CLASS_THRESHOLDS, RATIO_EPS, HEALTHY_WEIGHT, HEALTHY_BAR_CAP,
    SUSPICION_HCUT, SUSPICION_OFFSET, MIN_SUPPORT, CONF_SUPPORT,
    N_FEAT, LAPLACE_ALPHA, U_MIN, CORR_THRESHOLD,
    _supervised_bin_edges, _fast_abs_corr,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")
RARE_CLASSES = {4, 5, 9}
SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]

W1101_THRESHOLDS = dict(CLASS_THRESHOLDS)
W1101_THRESHOLDS[10] = 3.0


def _train_with_class_params(nodes, td, tl, is_bin, all_cls,
                             min_sup_map, conf_sup_map):
    import experiment_runner as mod
    class_models, class_retained = {}, {}
    orig_ms, orig_cs = mod.MIN_SUPPORT, mod.CONF_SUPPORT

    for cls in all_cls:
        mod.MIN_SUPPORT = min_sup_map.get(cls, 3)
        mod.CONF_SUPPORT = conf_sup_map.get(cls, 10)

        cls_models = {}
        cls_actions = defaultdict(list)
        for nd in nodes:
            nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls, use_supervised_bins=True)
            cls_models.update(nm)
            for a in na:
                cls_actions[a[1]].append(a)

        cls_ret = []
        for nd in nodes:
            cls_ret.extend(refine_ovr_node_cfg(
                nd, cls_models, cls_actions.get(nd.nid, []), td,
                rank_by='score', fpc=18))

        class_models[cls] = cls_models
        class_retained[cls] = cls_ret

    mod.MIN_SUPPORT = orig_ms
    mod.CONF_SUPPORT = orig_cs
    return class_models, class_retained


def make_w1101():
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in range(1, 14)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in range(1, 14)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            ag = 0.5 if cls in RARE_CLASSES else 0.8
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=ag)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY:
                continue
            t = W1101_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t:
                continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return train, predict


def make_w905():
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in range(1, 14)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in range(1, 14)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            ag = 0.5 if cls in RARE_CLASSES else 0.8
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=ag)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY:
                continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t:
                continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return train, predict


def run_split(data, labels, is_bin, seed, predict_fn, train_fn, train_frac):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]
    td, tl = data[train_idx], labels[train_idx]
    all_cls = sorted(set(labels))
    nodes = build_tree(td, tl, is_bin)
    train_result = train_fn(nodes, td, tl, is_bin, all_cls)
    results = []
    for uid in test_idx:
        true_cls = int(labels[uid])
        pred, scores = predict_fn(uid, data, nodes, all_cls, train_result)
        results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def binary_acc(results):
    correct = 0
    for _, true_cls, pred, _ in results:
        true_bin = 0 if true_cls == HEALTHY else 1
        pred_bin = 0 if pred == HEALTHY else 1
        if true_bin == pred_bin:
            correct += 1
    return correct / len(results)


def validate_model(name, train_fn, predict_fn, data, labels, is_bin):
    print(f"\n{'='*70}")
    print(f"  VALIDATING: {name}")
    print(f"{'='*70}")

    # 10-fold CV
    print(f"\n  10-Fold CV (10 seeds):")
    print(f"  {'Seed':>6s}  {'Acc':>7s}  {'Spec':>7s}  {'Sens':>7s}")
    print(f"  {'-'*35}")
    cv_accs, cv_specs, cv_senss = [], [], []
    for s in SEEDS:
        results = run_10fold(data, labels, is_bin, s, predict_fn, train_fn)
        acc, spec, sens, ba = stats(results)
        cv_accs.append(acc)
        cv_specs.append(spec)
        cv_senss.append(sens)
        print(f"  {s:6d}  {100*acc:6.1f}%  {100*spec:6.1f}%  {100*sens:6.1f}%", flush=True)
    print(f"  {'-'*35}")
    print(f"  {'Mean':>6s}  {100*np.mean(cv_accs):6.1f}%  {100*np.mean(cv_specs):6.1f}%  {100*np.mean(cv_senss):6.1f}%")
    print(f"  {'Std':>6s}  {100*np.std(cv_accs):6.2f}%")
    print(f"  {'Best':>6s}  {100*max(cv_accs):6.1f}%")

    # 90/10 split
    print(f"\n  90/10 Split (10 seeds) — multiclass + binary:")
    print(f"  {'Seed':>6s}  {'Multi':>7s}  {'Binary':>7s}  {'Spec':>7s}  {'Sens':>7s}")
    print(f"  {'-'*45}")
    s90_accs, s90_bins = [], []
    for s in SEEDS:
        results = run_split(data, labels, is_bin, s, predict_fn, train_fn, 0.9)
        acc, spec, sens, ba = stats(results)
        ba_val = binary_acc(results)
        s90_accs.append(acc)
        s90_bins.append(ba_val)
        print(f"  {s:6d}  {100*acc:6.1f}%  {100*ba_val:6.1f}%  {100*spec:6.1f}%  {100*sens:6.1f}%", flush=True)
    print(f"  {'-'*45}")
    print(f"  {'Mean':>6s}  {100*np.mean(s90_accs):6.1f}%  {100*np.mean(s90_bins):6.1f}%")
    print(f"  {'Best':>6s}  {100*max(s90_accs):6.1f}%  {100*max(s90_bins):6.1f}%")

    # 60/40 split
    print(f"\n  60/40 Split (10 seeds):")
    print(f"  {'Seed':>6s}  {'Acc':>7s}  {'Spec':>7s}  {'Sens':>7s}")
    print(f"  {'-'*35}")
    s60_accs = []
    for s in SEEDS:
        results = run_split(data, labels, is_bin, s, predict_fn, train_fn, 0.6)
        acc, spec, sens, ba = stats(results)
        s60_accs.append(acc)
        print(f"  {s:6d}  {100*acc:6.1f}%  {100*spec:6.1f}%  {100*sens:6.1f}%", flush=True)
    print(f"  {'-'*35}")
    print(f"  {'Mean':>6s}  {100*np.mean(s60_accs):6.1f}%")
    print(f"  {'Best':>6s}  {100*max(s60_accs):6.1f}%")

    return {
        'cv_mean': np.mean(cv_accs), 'cv_best': max(cv_accs), 'cv_std': np.std(cv_accs),
        's90_mean': np.mean(s90_accs), 's90_best': max(s90_accs),
        's90_bin_mean': np.mean(s90_bins), 's90_bin_best': max(s90_bins),
        's60_mean': np.mean(s60_accs), 's60_best': max(s60_accs),
    }


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats")

    models = {
        "W11-01 (W9-05 + c10 thresh=3.0)": make_w1101(),
        "W9-05 baseline (minsup+against)": make_w905(),
    }

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode != "all":
        key = list(models.keys())[int(mode) - 1]
        models = {key: models[key]}

    all_results = {}
    for name, (train_fn, predict_fn) in models.items():
        t0 = time.time()
        r = validate_model(name, train_fn, predict_fn, data, labels, is_bin)
        elapsed = time.time() - t0
        r['time'] = elapsed
        all_results[name] = r

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — W11-01 vs W9-05 vs Benchmarks")
    print(f"{'='*70}")
    print(f"\n  Benchmarks: 10-fold=86.57%, 90/10 multi=92.07%, 90/10 bin=95.24%, 60/40=93.33%\n")
    print(f"  {'Model':<35s}  {'10-fold':>8s}  {'90/10m':>8s}  {'90/10b':>8s}  {'60/40':>8s}")
    print(f"  {'-'*75}")
    for name, r in all_results.items():
        print(f"  {name:<35s}  {100*r['cv_mean']:7.1f}%  {100*r['s90_mean']:7.1f}%  "
              f"{100*r['s90_bin_mean']:7.1f}%  {100*r['s60_mean']:7.1f}%")
    print(f"  {'-'*75}")
    print(f"  {'Benchmarks':<35s}  {'86.57%':>8s}  {'92.07%':>8s}  {'95.24%':>8s}  {'93.33%':>8s}")

    print(f"\n  Best single-seed results:")
    print(f"  {'Model':<35s}  {'10-fold':>8s}  {'90/10m':>8s}  {'90/10b':>8s}  {'60/40':>8s}")
    print(f"  {'-'*75}")
    for name, r in all_results.items():
        print(f"  {name:<35s}  {100*r['cv_best']:7.1f}%  {100*r['s90_best']:7.1f}%  "
              f"{100*r['s90_bin_best']:7.1f}%  {100*r['s60_best']:7.1f}%")
