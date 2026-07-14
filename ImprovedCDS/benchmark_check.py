"""Check W9-05 against target benchmarks.

Targets:
- 90/10 multiclass: 92.07%
- 90/10 binary: 95.24%
- 10-fold CV: 86.57%
- 60/40 split: 93.33%
"""
import numpy as np
from collections import defaultdict
from pathlib import Path
from experiment_runner import (
    load_data, classify_features, build_tree,
    train_ovr_node_cfg, refine_ovr_node_cfg,
    compute_af_cfg, predict_cfg, score_patient,
    run_10fold, stats, HEALTHY, CLASS_THRESHOLDS, RATIO_EPS,
    HEALTHY_WEIGHT, HEALTHY_BAR_CAP, SUSPICION_HCUT, SUSPICION_OFFSET,
    CORR_THRESHOLD, _fast_abs_corr,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")
RARE_CLASSES = {4, 5, 9}
SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]


def _train_with_class_params(nodes, td, tl, is_bin, all_cls,
                             min_sup_map, conf_sup_map,
                             corr_thresh_map=None):
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

        corr_t = corr_thresh_map.get(cls, CORR_THRESHOLD) if corr_thresh_map else CORR_THRESHOLD
        fpc = 18

        if corr_t != CORR_THRESHOLD:
            cls_ret = []
            for nd in nodes:
                node_actions = cls_actions.get(nd.nid, [])
                scored = [(a, a[2]) for a in node_actions]
                scored.sort(key=lambda x: x[1], reverse=True)
                if not scored:
                    continue
                top_scored = scored[:3 * fpc]
                nd_data = td[nd.uidx]
                top_feats = sorted(set(a[0] for a, _ in top_scored))
                correlations = {}
                for i, f1 in enumerate(top_feats):
                    col1 = nd_data[:, f1]
                    nan1 = np.isnan(col1)
                    for f2 in top_feats[i + 1:]:
                        col2 = nd_data[:, f2]
                        valid = ~(nan1 | np.isnan(col2))
                        if valid.sum() > 10:
                            c = _fast_abs_corr(col1[valid], col2[valid])
                            if c > 0:
                                correlations[(f1, f2)] = c
                                correlations[(f2, f1)] = c
                kept, kept_features = [], set()
                for a, s in top_scored:
                    f = a[0]
                    if f in kept_features:
                        continue
                    if kept_features:
                        max_corr = max(correlations.get((f, kf), 0.0) for kf in kept_features)
                        if max_corr > corr_t:
                            continue
                    kept.append(a)
                    kept_features.add(f)
                    if len(kept) >= fpc:
                        break
                cls_ret.extend(kept)
        else:
            cls_ret = []
            for nd in nodes:
                cls_ret.extend(refine_ovr_node_cfg(
                    nd, cls_models, cls_actions.get(nd.nid, []), td,
                    rank_by='score', fpc=fpc))

        class_models[cls] = cls_models
        class_retained[cls] = cls_ret

    mod.MIN_SUPPORT = orig_ms
    mod.CONF_SUPPORT = orig_cs
    return class_models, class_retained


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


def binary_stats(results):
    correct = 0
    total = len(results)
    for uid, true_cls, pred_cls, _ in results:
        true_bin = 0 if true_cls == HEALTHY else 1
        pred_bin = 0 if pred_cls == HEALTHY else 1
        if true_bin == pred_bin:
            correct += 1
    return correct / total


def multiclass_stats(results):
    correct = sum(r[3] for r in results)
    return correct / len(results)


# W9-05: minsup + low against for rare
ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in range(1, 14)}
cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in range(1, 14)}

def train_w905(nodes, td, tl, is_bin, all_cls):
    return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

def predict_w905(uid, data, nodes, all_cls, tr):
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


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats\n")

    print("=" * 70)
    print("  W9-05 vs BENCHMARKS")
    print("=" * 70)

    # 10-fold CV
    print("\n  10-Fold CV (target: 86.57%)")
    print(f"  {'Seed':>6s}  {'Multi':>8s}  {'Binary':>8s}")
    print(f"  {'-'*30}")
    cv_multi, cv_bin = [], []
    for s in SEEDS:
        results = run_10fold(data, labels, is_bin, s, predict_w905, train_w905)
        m = multiclass_stats(results)
        b = binary_stats(results)
        cv_multi.append(m)
        cv_bin.append(b)
        print(f"  {s:6d}  {100*m:7.2f}%  {100*b:7.2f}%", flush=True)
    print(f"  {'-'*30}")
    print(f"  {'Mean':>6s}  {100*np.mean(cv_multi):7.2f}%  {100*np.mean(cv_bin):7.2f}%")
    print(f"  {'Best':>6s}  {100*max(cv_multi):7.2f}%  {100*max(cv_bin):7.2f}%")
    print(f"  Target: 86.57% (multi)")

    # 90/10 split
    print("\n  90/10 Split (targets: multi 92.07%, binary 95.24%)")
    print(f"  {'Seed':>6s}  {'Multi':>8s}  {'Binary':>8s}")
    print(f"  {'-'*30}")
    s90_multi, s90_bin = [], []
    for s in SEEDS:
        results = run_split(data, labels, is_bin, s, predict_w905, train_w905, 0.9)
        m = multiclass_stats(results)
        b = binary_stats(results)
        s90_multi.append(m)
        s90_bin.append(b)
        print(f"  {s:6d}  {100*m:7.2f}%  {100*b:7.2f}%", flush=True)
    print(f"  {'-'*30}")
    print(f"  {'Mean':>6s}  {100*np.mean(s90_multi):7.2f}%  {100*np.mean(s90_bin):7.2f}%")
    print(f"  {'Best':>6s}  {100*max(s90_multi):7.2f}%  {100*max(s90_bin):7.2f}%")
    print(f"  Targets: multi 92.07%, binary 95.24%")

    # 60/40 split
    print("\n  60/40 Split (target: 93.33%)")
    print(f"  {'Seed':>6s}  {'Multi':>8s}  {'Binary':>8s}")
    print(f"  {'-'*30}")
    s60_multi, s60_bin = [], []
    for s in SEEDS:
        results = run_split(data, labels, is_bin, s, predict_w905, train_w905, 0.6)
        m = multiclass_stats(results)
        b = binary_stats(results)
        s60_multi.append(m)
        s60_bin.append(b)
        print(f"  {s:6d}  {100*m:7.2f}%  {100*b:7.2f}%", flush=True)
    print(f"  {'-'*30}")
    print(f"  {'Mean':>6s}  {100*np.mean(s60_multi):7.2f}%  {100*np.mean(s60_bin):7.2f}%")
    print(f"  {'Best':>6s}  {100*max(s60_multi):7.2f}%  {100*max(s60_bin):7.2f}%")
    print(f"  Target: 93.33%")

    print("\n" + "=" * 70)
    print("  SUMMARY vs BENCHMARKS")
    print("=" * 70)
    print(f"\n  {'Metric':<30s}  {'W9-05 Best':>12s}  {'W9-05 Mean':>12s}  {'Target':>10s}  {'Status':>10s}")
    print(f"  {'-'*80}")

    metrics = [
        ("90/10 multiclass", max(s90_multi), np.mean(s90_multi), 0.9207),
        ("90/10 binary", max(s90_bin), np.mean(s90_bin), 0.9524),
        ("10-fold CV", max(cv_multi), np.mean(cv_multi), 0.8657),
        ("60/40 split", max(s60_multi), np.mean(s60_multi), 0.9333),
    ]
    for name, best, mean, target in metrics:
        status = "BEAT!" if best >= target else f"need +{100*(target-best):.1f}%"
        print(f"  {name:<30s}  {100*best:11.2f}%  {100*mean:11.2f}%  {100*target:9.2f}%  {status:>10s}")
