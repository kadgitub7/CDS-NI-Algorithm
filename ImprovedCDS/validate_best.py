"""Full validation of W9-01 (combo minsup+corr) and W9-05 (minsup+low against).

Runs:
1. 10-fold CV across 10 seeds
2. 90/10 split across 10 seeds
3. 60/40 split across 10 seeds

Compares against old baselines:
- 85.3% 10-fold CV
- 92.90% 90/10 split
- 82.6% 60/40 split
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


def _train_with_class_params(nodes, td, tl, is_bin, all_cls,
                             min_sup_map, conf_sup_map,
                             corr_thresh_map=None, fpc_map=None):
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

        fpc = fpc_map.get(cls, 18) if fpc_map else 18
        corr_t = corr_thresh_map.get(cls, CORR_THRESHOLD) if corr_thresh_map else CORR_THRESHOLD

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


# ─── W9-01: Combo minsup + relaxed corr ───

def make_w901():
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in range(1, 14)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in range(1, 14)}
    ct = {cls: (0.9 if cls in RARE_CLASSES else 0.8) for cls in range(1, 14)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs, corr_thresh_map=ct)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    return train, predict


# ─── W9-05: Minsup + low against for rare ───

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


# ─── W8-03: Original winner (minsup only, for reference) ───

def make_w803():
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in range(1, 14)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in range(1, 14)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    return train, predict


# ─── Old baseline (supervised bins + against_scale=0.8) ───

def make_baseline():
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)

    return train, predict


def validate_model(name, train_fn, predict_fn, data, labels, is_bin):
    print(f"\n{'='*70}")
    print(f"  VALIDATING: {name}")
    print(f"{'='*70}")

    # 10-fold CV across 10 seeds
    print(f"\n  10-Fold CV (10 seeds):")
    print(f"  {'Seed':>6s}  {'Acc':>7s}  {'Spec':>7s}  {'Sens':>7s}  {'BA':>7s}")
    print(f"  {'-'*40}")
    cv_accs, cv_specs, cv_senss, cv_bas = [], [], [], []
    for s in SEEDS:
        results = run_10fold(data, labels, is_bin, s, predict_fn, train_fn)
        acc, spec, sens, ba = stats(results)
        cv_accs.append(acc)
        cv_specs.append(spec)
        cv_senss.append(sens)
        cv_bas.append(ba)
        print(f"  {s:6d}  {100*acc:6.1f}%  {100*spec:6.1f}%  {100*sens:6.1f}%  {100*ba:6.1f}%", flush=True)
    print(f"  {'-'*40}")
    print(f"  {'Mean':>6s}  {100*np.mean(cv_accs):6.1f}%  {100*np.mean(cv_specs):6.1f}%  "
          f"{100*np.mean(cv_senss):6.1f}%  {100*np.mean(cv_bas):6.1f}%")
    print(f"  {'Std':>6s}  {100*np.std(cv_accs):6.2f}%  {100*np.std(cv_specs):6.2f}%  "
          f"{100*np.std(cv_senss):6.2f}%  {100*np.std(cv_bas):6.2f}%")
    print(f"  {'Best':>6s}  {100*max(cv_accs):6.1f}%")

    # 90/10 split across 10 seeds
    print(f"\n  90/10 Split (10 seeds):")
    print(f"  {'Seed':>6s}  {'Acc':>7s}  {'Spec':>7s}  {'Sens':>7s}  {'BA':>7s}")
    print(f"  {'-'*40}")
    s90_accs = []
    for s in SEEDS:
        results = run_split(data, labels, is_bin, s, predict_fn, train_fn, 0.9)
        acc, spec, sens, ba = stats(results)
        s90_accs.append(acc)
        print(f"  {s:6d}  {100*acc:6.1f}%  {100*spec:6.1f}%  {100*sens:6.1f}%  {100*ba:6.1f}%", flush=True)
    print(f"  {'-'*40}")
    print(f"  {'Mean':>6s}  {100*np.mean(s90_accs):6.1f}%")
    print(f"  {'Std':>6s}  {100*np.std(s90_accs):6.2f}%")
    print(f"  {'Best':>6s}  {100*max(s90_accs):6.1f}%")

    # 60/40 split across 10 seeds
    print(f"\n  60/40 Split (10 seeds):")
    print(f"  {'Seed':>6s}  {'Acc':>7s}  {'Spec':>7s}  {'Sens':>7s}  {'BA':>7s}")
    print(f"  {'-'*40}")
    s60_accs = []
    for s in SEEDS:
        results = run_split(data, labels, is_bin, s, predict_fn, train_fn, 0.6)
        acc, spec, sens, ba = stats(results)
        s60_accs.append(acc)
        print(f"  {s:6d}  {100*acc:6.1f}%  {100*spec:6.1f}%  {100*sens:6.1f}%  {100*ba:6.1f}%", flush=True)
    print(f"  {'-'*40}")
    print(f"  {'Mean':>6s}  {100*np.mean(s60_accs):6.1f}%")
    print(f"  {'Std':>6s}  {100*np.std(s60_accs):6.2f}%")
    print(f"  {'Best':>6s}  {100*max(s60_accs):6.1f}%")

    return {
        'cv_mean': np.mean(cv_accs), 'cv_best': max(cv_accs), 'cv_std': np.std(cv_accs),
        's90_mean': np.mean(s90_accs), 's90_best': max(s90_accs),
        's60_mean': np.mean(s60_accs), 's60_best': max(s60_accs),
    }


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats")

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    models = {
        "W9-01 Combo (minsup+corr)": make_w901(),
        "W9-05 Minsup+low against": make_w905(),
        "W8-03 Minsup only": make_w803(),
        "Old baseline (supbins+ag0.8)": make_baseline(),
    }

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

    # Summary comparison
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"\n  Old baselines: 10-fold=85.3%, 90/10=92.9%, 60/40=82.6%\n")
    print(f"  {'Model':<35s}  {'10-fold':>8s}  {'90/10':>8s}  {'60/40':>8s}  {'Time':>6s}")
    print(f"  {'-'*65}")
    for name, r in all_results.items():
        print(f"  {name:<35s}  {100*r['cv_mean']:7.1f}%  {100*r['s90_mean']:7.1f}%  "
              f"{100*r['s60_mean']:7.1f}%  {r['time']:5.0f}s")
    print(f"  {'-'*65}")
    print(f"  {'Old baseline':<35s}  {'85.3%':>8s}  {'92.9%':>8s}  {'82.6%':>8s}")
