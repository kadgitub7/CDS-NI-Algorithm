"""Wave 9 — Combine and validate W8 winners, push further.

W8-03 (86.1%): Lower MIN_SUPPORT=2, CONF_SUPPORT=5 for rare classes {4,5,9}
W8-09 (85.6%): Relaxed correlation threshold 0.9 for rare classes

Strategy:
1. Combine W8-03 + W8-09 (both preprocessing changes, should stack)
2. Apply lower MIN_SUPPORT to ALL classes (not just rare)
3. Try MIN_SUPPORT=1 for rare classes (even more aggressive)
4. W8-03 + fewer bins for rare classes
5. W8-03 + asymmetric against_scale for rare
6. W8-03 + more features per rare class (fpc=24)
7. Adaptive MIN_SUPPORT based on class size
8. W8-03 + relaxed corr + fewer bins (triple combo)
9. W8-03 + lower thresholds for rare classes
10. W8-03 + breadth-weighted scoring for rare
11. Multi-seed validation of W8-03 (10 seeds)
12. W8-03 + W8-09 multi-seed validation
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


def _train_with_class_params(nodes, td, tl, is_bin, all_cls,
                             min_sup_map, conf_sup_map,
                             corr_thresh_map=None, fpc_map=None,
                             max_bins_map=None):
    """Generic training with per-class MIN_SUPPORT, CONF_SUPPORT, correlation, fpc, and bins."""
    import experiment_runner as mod
    class_models, class_retained = {}, {}
    orig_ms, orig_cs = mod.MIN_SUPPORT, mod.CONF_SUPPORT
    orig_mb = mod.MAX_BINS
    orig_ct = mod.CORR_THRESHOLD

    for cls in all_cls:
        mod.MIN_SUPPORT = min_sup_map.get(cls, orig_ms)
        mod.CONF_SUPPORT = conf_sup_map.get(cls, orig_cs)
        if max_bins_map:
            mod.MAX_BINS = max_bins_map.get(cls, 6)

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
    mod.MAX_BINS = orig_mb
    mod.CORR_THRESHOLD = orig_ct
    return class_models, class_retained


# ─── W9-01: W8-03 + W8-09 combo (lower minsup + relaxed corr for rare) ───

def exp_combo_minsup_corr(data, labels, is_bin, seed):
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in set(labels)}
    ct = {cls: (0.9 if cls in RARE_CLASSES else 0.8) for cls in set(labels)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs, corr_thresh_map=ct)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-02: Lower MIN_SUPPORT for ALL classes ───

def exp_minsup2_all(data, labels, is_bin, seed):
    ms = {cls: 2 for cls in set(labels)}
    cs = {cls: 5 for cls in set(labels)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-03: MIN_SUPPORT=1 for rare (most aggressive) ───

def exp_minsup1_rare(data, labels, is_bin, seed):
    ms = {cls: (1 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (3 if cls in RARE_CLASSES else 10) for cls in set(labels)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-04: W8-03 + fewer bins for rare ───

def exp_minsup_fewerbins(data, labels, is_bin, seed):
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in set(labels)}
    mb = {cls: (3 if cls in RARE_CLASSES else 6) for cls in set(labels)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs, max_bins_map=mb)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-05: W8-03 + lower against_scale for rare ───

def exp_minsup_lowagainst(data, labels, is_bin, seed):
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in set(labels)}

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

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-06: W8-03 + more features for rare (fpc=24) ───

def exp_minsup_morefeat(data, labels, is_bin, seed):
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in set(labels)}
    fp = {cls: (24 if cls in RARE_CLASSES else 18) for cls in set(labels)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs, fpc_map=fp)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-07: Adaptive MIN_SUPPORT by class size ───

def exp_adaptive_minsup(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        import experiment_runner as mod
        class_counts = {cls: int((tl == cls).sum()) for cls in all_cls}
        ms, cs = {}, {}
        for cls in all_cls:
            n = class_counts[cls]
            if n < 15:
                ms[cls], cs[cls] = 1, 3
            elif n < 30:
                ms[cls], cs[cls] = 2, 5
            elif n < 60:
                ms[cls], cs[cls] = 2, 8
            else:
                ms[cls], cs[cls] = 3, 10
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-08: Triple combo: minsup + relaxed corr + fewer bins for rare ───

def exp_triple_combo(data, labels, is_bin, seed):
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in set(labels)}
    ct = {cls: (0.9 if cls in RARE_CLASSES else 0.8) for cls in set(labels)}
    mb = {cls: (3 if cls in RARE_CLASSES else 6) for cls in set(labels)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs,
                                        corr_thresh_map=ct, max_bins_map=mb)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-09: W8-03 + lower thresholds for rare ───

def exp_minsup_lowthresh(data, labels, is_bin, seed):
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in set(labels)}

    low_thresh = dict(CLASS_THRESHOLDS)
    for c in RARE_CLASSES:
        if c in low_thresh:
            low_thresh[c] = max(low_thresh[c] - 1.0, 2.0)

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls,
                           against_scale=0.8, thresholds=low_thresh)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-10: W8-03 + breadth-weighted for rare only ───

def exp_minsup_breadth_rare(data, labels, is_bin, seed):
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in set(labels)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            ratio = (af[0] + RATIO_EPS) / (af[1] + RATIO_EPS)
            if cls in RARE_CLASSES:
                breadth = af[3] / max(af[3] + af[4], 1)
                class_scores[cls] = ratio * (0.5 + 0.5 * breadth)
            else:
                class_scores[cls] = ratio

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

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W9-11: Multi-seed validation of W8-03 ───

def exp_multiseed_w803(data, labels, is_bin, seed):
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in set(labels)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    seeds = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]
    all_accs = []
    for s in seeds:
        results = run_10fold(data, labels, is_bin, s, predict, train)
        acc, spec, sens, ba = stats(results)
        all_accs.append(acc)
        print(f"  Seed {s:2d}: {100*acc:5.1f}%", flush=True)
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    print(f"  Mean: {100*mean_acc:5.1f}% +/- {100*std_acc:4.1f}%", flush=True)
    return run_10fold(data, labels, is_bin, 13, predict, train)


# ─── W9-12: Multi-seed validation of W9-01 (combo) ───

def exp_multiseed_combo(data, labels, is_bin, seed):
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in set(labels)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in set(labels)}
    ct = {cls: (0.9 if cls in RARE_CLASSES else 0.8) for cls in set(labels)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs, corr_thresh_map=ct)

    def predict(uid, data, nodes, all_cls, tr):
        return predict_cfg(uid, data, nodes, tr[0], tr[1], all_cls, against_scale=0.8)

    seeds = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]
    all_accs = []
    for s in seeds:
        results = run_10fold(data, labels, is_bin, s, predict, train)
        acc, spec, sens, ba = stats(results)
        all_accs.append(acc)
        print(f"  Seed {s:2d}: {100*acc:5.1f}%", flush=True)
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    print(f"  Mean: {100*mean_acc:5.1f}% +/- {100*std_acc:4.1f}%", flush=True)
    return run_10fold(data, labels, is_bin, 13, predict, train)


experiments = [
    ("W9-01 Combo minsup+corr",        exp_combo_minsup_corr),
    ("W9-02 Minsup2 ALL classes",      exp_minsup2_all),
    ("W9-03 Minsup1 rare",             exp_minsup1_rare),
    ("W9-04 Minsup+fewer bins",        exp_minsup_fewerbins),
    ("W9-05 Minsup+low against",       exp_minsup_lowagainst),
    ("W9-06 Minsup+more feat rare",    exp_minsup_morefeat),
    ("W9-07 Adaptive minsup",          exp_adaptive_minsup),
    ("W9-08 Triple combo",             exp_triple_combo),
    ("W9-09 Minsup+low thresh rare",   exp_minsup_lowthresh),
    ("W9-10 Minsup+breadth rare",      exp_minsup_breadth_rare),
    ("W9-11 Multi-seed W8-03",         exp_multiseed_w803),
    ("W9-12 Multi-seed combo",         exp_multiseed_combo),
]

if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats\n")

    seed = 13
    if len(sys.argv) > 1:
        exp_id = int(sys.argv[1])
        experiments_run = [experiments[exp_id - 1]]
    else:
        experiments_run = experiments

    print(f"{'Experiment':42s}  {'Acc':>6s}  {'Spec':>6s}  {'Sens':>6s}  {'BA':>6s}  {'Time':>6s}")
    print("-" * 85)

    for name, exp_fn in experiments_run:
        t0 = time.time()
        results = exp_fn(data, labels, is_bin, seed)
        elapsed = time.time() - t0
        acc, spec, sens, ba = stats(results)
        print(f"{name:42s}  {100*acc:5.1f}%  {100*spec:5.1f}%  {100*ba:5.1f}%  {elapsed:5.0f}s",
              flush=True)
