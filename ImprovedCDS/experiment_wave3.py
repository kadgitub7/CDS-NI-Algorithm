"""Wave 3 experiments — radical structural changes beyond parameter tuning.
Base: supervised bins + against_scale 0.8 (85.3% on 10-fold seed=13).
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
    AGAINST_SCALE, N_FEAT, LAPLACE_ALPHA, CORR_THRESHOLD,
    _supervised_bin_edges, _fast_abs_corr,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


# ─── W3-01: Combine top winners: SV + against 0.8 + 27 healthy ───

def exp_sv08_27h(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, healthy_fpc=27)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-02: SV + against 0.8 + composite feature selection ───

def exp_sv08_composite(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, rank_by='composite')
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-03: SV + against 0.8 + healthy weight 1.15 ───

def exp_sv08_hw115(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          against_scale=0.8, healthy_weight=1.15)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-04: Ensemble bagging — train 5 models, average scores ───

def exp_ensemble_bag(data, labels, is_bin, seed):
    N_BAGS = 5

    def train(nodes, td, tl, is_bin, all_cls):
        rng = np.random.RandomState(seed + 999)
        bags = []
        n = len(td)
        for b in range(N_BAGS):
            boot_idx = rng.choice(n, size=n, replace=True)
            bd, bl = td[boot_idx], tl[boot_idx]
            b_nodes = build_tree(bd, bl, is_bin)
            cm, cr = train_all_cfg(b_nodes, bd, bl, is_bin, all_cls,
                                    use_supervised_bins=True)
            bags.append((b_nodes, cm, cr, bd, bl))
        return bags

    def predict(uid, data, nodes, all_cls, tr):
        bags = tr
        avg_scores = defaultdict(float)
        for b_nodes, cm, cr, bd, bl in bags:
            for cls in all_cls:
                af = compute_af_cfg(uid, data, b_nodes, cm[cls], cr[cls],
                                    against_scale=0.8)
                s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
                avg_scores[cls] += s / len(bags)

        h_score = avg_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in avg_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, dict(avg_scores)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-05: Two-stage: binary healthy/disease first, then subtype ───

def exp_two_stage(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                                use_supervised_bins=True)
        binary_labels = np.where(tl == HEALTHY, HEALTHY, 0).astype(int)
        binary_cls = [0, HEALTHY]
        b_nodes = build_tree(td, binary_labels, is_bin)
        bcm, bcr = train_all_cfg(b_nodes, td, binary_labels, is_bin, binary_cls,
                                  use_supervised_bins=True)
        return (cm, cr, bcm, bcr, b_nodes, binary_cls)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, bcm, bcr, b_nodes, binary_cls = tr
        h_af = compute_af_cfg(uid, data, b_nodes, bcm[HEALTHY], bcr[HEALTHY],
                              against_scale=0.8)
        d_af = compute_af_cfg(uid, data, b_nodes, bcm[0], bcr[0],
                              against_scale=0.8)
        h_score = score_patient(h_af[0], h_af[1], h_af[3], h_af[4], h_af[5], 'ratio')
        d_score = score_patient(d_af[0], d_af[1], d_af[3], d_af[4], d_af[5], 'ratio')

        if h_score > d_score * 1.5:
            return HEALTHY, {HEALTHY: h_score}

        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0) * 0.8
            if score < t: continue
            candidates[cls] = score
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-06: Adaptive features per class (more for rare classes) ───

def exp_adaptive_fpc(data, labels, is_bin, seed):
    class_counts = defaultdict(int)
    for l in labels: class_counts[l] += 1

    def train(nodes, td, tl, is_bin, all_cls):
        class_models, class_retained = {}, {}
        for cls in all_cls:
            n_cls = sum(1 for l in tl if l == cls)
            if cls == HEALTHY:
                fpc = 27
            elif n_cls < 15:
                fpc = 12
            elif n_cls < 30:
                fpc = 18
            else:
                fpc = 22
            cls_models = {}
            cls_actions_by_node = defaultdict(list)
            for nd in nodes:
                nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls,
                                             use_supervised_bins=True)
                cls_models.update(nm)
                for a in na: cls_actions_by_node[a[1]].append(a)
            cls_ret = []
            for nd in nodes:
                cls_ret.extend(refine_ovr_node_cfg(
                    nd, cls_models, cls_actions_by_node.get(nd.nid, []), td,
                    rank_by='score', fpc=fpc))
            class_models[cls] = cls_models
            class_retained[cls] = cls_ret
        return class_models, class_retained

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-07: Class-specific against_scale ───

def exp_class_against(data, labels, is_bin, seed):
    cls_against = {1: 0.8, 2: 0.5, 3: 0.6, 4: 0.5, 5: 0.5, 6: 0.5, 9: 0.6, 10: 0.6}

    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            ag_s = cls_against.get(cls, 0.6)
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=ag_s)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-08: Higher max bins (10) with supervised binning ───

def exp_more_bins(data, labels, is_bin, seed):
    MAX_BINS_10 = 10

    def train_node_10bins(node, data, labels, is_bin, target_class):
        models, actions = {}, []
        nd_data, nd_labels = data[node.uidx], labels[node.uidx]
        ns = node.nu
        n_target = int((nd_labels == target_class).sum())
        if n_target < 1: return models, actions
        prior = n_target / ns
        is_target = (nd_labels == target_class)
        for f in range(N_FEAT):
            col = nd_data[:, f]
            vm = ~np.isnan(col)
            vv = col[vm]
            nv = len(vv)
            if nv == 0: continue
            vmin, vmax = float(vv.min()), float(vv.max())
            if is_bin[f]:
                nb = 1 if vmin == vmax else 2
                edges = np.array([vmin-.5, vmin+.5]) if nb==1 else np.array([-.5,.5,1.5])
            elif vmin == vmax:
                nb, edges = 1, np.array([vmin-.5, vmin+.5])
            else:
                max_nb = min(max(2, int(np.ceil(1+np.log2(nv)))), MAX_BINS_10)
                edges = _supervised_bin_edges(vv, is_target[vm].astype(float), max_nb)
                nb = len(edges) - 1
            ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb-1)
            lv = nd_labels[vm]
            bin_counts = np.bincount(ba, minlength=nb)
            target_counts = np.bincount(ba[lv==target_class], minlength=nb).astype(float)
            p_class = (target_counts + LAPLACE_ALPHA * prior) / (bin_counts + LAPLACE_ALPHA)
            models[(node.nid, f)] = BinModel(nb, edges, bin_counts, target_counts, p_class, prior, CONF_SUPPORT)
            score = 0.0
            for b in range(nb):
                if bin_counts[b] >= MIN_SUPPORT:
                    shift = abs(p_class[b] - prior)
                    confidence = min(1.0, float(bin_counts[b])/CONF_SUPPORT)
                    score += shift * confidence
            target_vals = vv[lv==target_class]
            rest_vals = vv[lv!=target_class]
            if len(target_vals)>=2 and len(rest_vals)>=2:
                mean_diff2 = (target_vals.mean()-rest_vals.mean())**2
                var_sum = target_vals.var()+rest_vals.var()
                fisher = mean_diff2/(var_sum+1e-10)
            else:
                fisher = 0.0
            if score > 0.001:
                actions.append((f, node.nid, score, fisher))
        return models, actions

    def train(nodes, td, tl, is_bin, all_cls):
        class_models, class_retained = {}, {}
        for cls in all_cls:
            cls_models = {}
            cls_actions_by_node = defaultdict(list)
            for nd in nodes:
                nm, na = train_node_10bins(nd, td, tl, is_bin, cls)
                cls_models.update(nm)
                for a in na: cls_actions_by_node[a[1]].append(a)
            cls_ret = []
            for nd in nodes:
                cls_ret.extend(refine_ovr_node_cfg(
                    nd, cls_models, cls_actions_by_node.get(nd.nid, []), td,
                    rank_by='score', fpc=18))
            class_models[cls] = cls_models
            class_retained[cls] = cls_ret
        return class_models, class_retained

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-09: Confidence-gated — require min N features FOR before calling disease ───

def exp_confidence_gate(data, labels, is_bin, seed):
    MIN_FOR_FEATURES = 4

    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        class_n_for = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            class_n_for[cls] = af[3]

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            if class_n_for[cls] < MIN_FOR_FEATURES: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-10: Margin-based — only call disease if it beats healthy by a margin ───

def exp_margin_pred(data, labels, is_bin, seed):
    MARGIN = 1.5

    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            margin_over = score / max(h_score, 0.01)
            if margin_over < MARGIN: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-11: Lower correlation threshold (0.65) — more decorrelated features ───

def exp_decorr_065(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        class_models, class_retained = {}, {}
        for cls in all_cls:
            cls_models = {}
            cls_actions_by_node = defaultdict(list)
            for nd in nodes:
                nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls,
                                             use_supervised_bins=True)
                cls_models.update(nm)
                for a in na: cls_actions_by_node[a[1]].append(a)
            cls_ret = []
            for nd in nodes:
                node_actions = cls_actions_by_node.get(nd.nid, [])
                scored = [(a, a[2]) for a in node_actions]
                scored.sort(key=lambda x: x[1], reverse=True)
                if not scored: continue
                top_scored = scored[:54]
                nd_data = td[nd.uidx]
                top_feats = sorted(set(a[0] for a, _ in top_scored))
                correlations = {}
                for i, f1 in enumerate(top_feats):
                    col1 = nd_data[:, f1]
                    nan1 = np.isnan(col1)
                    for f2 in top_feats[i+1:]:
                        col2 = nd_data[:, f2]
                        valid = ~(nan1 | np.isnan(col2))
                        if valid.sum() > 10:
                            c = _fast_abs_corr(col1[valid], col2[valid])
                            if c > 0:
                                correlations[(f1,f2)] = c
                                correlations[(f2,f1)] = c
                kept, kept_features = [], set()
                for a, s in top_scored:
                    f = a[0]
                    if f in kept_features: continue
                    if kept_features:
                        max_corr = max(correlations.get((f, kf), 0.0) for kf in kept_features)
                        if max_corr > 0.65: continue
                    kept.append(a)
                    kept_features.add(f)
                    if len(kept) >= 18: break
                cls_ret.extend(kept)
            class_models[cls] = cls_models
            class_retained[cls] = cls_ret
        return class_models, class_retained

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W3-12: Best combo attempt: SV + against 0.8 + 27H + composite + HW 1.15 ───

def exp_best_combo(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, rank_by='composite',
                             healthy_fpc=27)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          against_scale=0.8, healthy_weight=1.15)
    return run_10fold(data, labels, is_bin, seed, predict, train)


experiments = [
    ("W3-01 SV+0.8+27H",              exp_sv08_27h),
    ("W3-02 SV+0.8+composite",        exp_sv08_composite),
    ("W3-03 SV+0.8+HW1.15",           exp_sv08_hw115),
    ("W3-04 Ensemble 5-bag",          exp_ensemble_bag),
    ("W3-05 Two-stage binary",        exp_two_stage),
    ("W3-06 Adaptive FPC",            exp_adaptive_fpc),
    ("W3-07 Class-specific against",  exp_class_against),
    ("W3-08 10 bins supervised",      exp_more_bins),
    ("W3-09 Confidence gate (4+)",    exp_confidence_gate),
    ("W3-10 Margin pred (1.5x)",      exp_margin_pred),
    ("W3-11 Decorr 0.65",            exp_decorr_065),
    ("W3-12 Best combo",             exp_best_combo),
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
        print(f"{name:42s}  {100*acc:5.1f}%  {100*spec:5.1f}%  {100*sens:5.1f}%  {100*ba:5.1f}%  {elapsed:5.0f}s",
              flush=True)
