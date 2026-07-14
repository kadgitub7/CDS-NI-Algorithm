"""Wave 8 — data-driven fixes based on error analysis.

Key findings from error analysis:
1. Error patients have drastically lower breadth (n_for/(n_for+n_against))
2. Class 5 errors: only 2-7 features FOR, 29-34 AGAINST (breadth 0.134)
3. 11 healthy->class 2 FPs: class 2 AF_for is very strong (2-3+)
4. 10 class 10->healthy FNs: class 10 scores 0.14-3.15 vs threshold 3.5
5. Rare classes (4,5) have too few training examples for reliable bins

Targeted fixes:
1. Top-K weighted scoring (top features dominate, not all features)
2. Breadth-weighted ratio (scale score by breadth)
3. Lower MIN_SUPPORT for rare classes (more features can contribute)
4. Max-of-top-K AF (use peak evidence, not sum)
5. Stricter healthy->disease false positive filter
6. Lower against_scale for rare classes only
7. Weighted Laplace by class size (less smoothing pushes posteriors harder)
8. Fewer bins for rare classes (more samples per bin)
9. Relaxed correlation filter for rare classes (keep more features)
10. AF_for only scoring for rare classes (ignore against)
11. Breadth gate + lower threshold combo
12. Top-3 dominant scoring (only top 3 features per class decide)
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
    N_FEAT, LAPLACE_ALPHA, U_MIN,
    _supervised_bin_edges, _fast_abs_corr,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")

RARE_CLASSES = {4, 5, 9}


# ─── W8-01: Top-K weighted AF (top features weighted higher) ───

def exp_topk_weighted(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            lvl_nodes = _route_user(uid, data, nodes)
            retained = cr[cls]
            models = cm[cls]
            fisher_map = {}
            for a in retained:
                fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
            max_fisher = max(fisher_map.values()) if fisher_map else 1.0

            contributions = []
            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for a in retained:
                        if a[1] != nd.nid: continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v): continue
                        mo = models.get((nd.nid, f))
                        if not mo: continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'), 0, mo.n_bins-1))
                        bc = mo.bin_counts[bi]
                        if bc < MIN_SUPPORT: continue
                        shift = mo.p_class[bi] - mo.prior
                        conf = min(1.0, bc / CONF_SUPPORT)
                        fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                        weighted = abs(shift) * conf * fw
                        contributions.append((weighted, shift >= 0, a[2]))

            contributions.sort(key=lambda x: x[2], reverse=True)
            af_for, af_against = 0.0, 0.0
            for i, (w, is_for, _) in enumerate(contributions):
                rank_weight = 1.0 / (1.0 + 0.05 * i)
                if is_for:
                    af_for += w * rank_weight
                else:
                    af_against += w * rank_weight * 0.8

            class_scores[cls] = (af_for + RATIO_EPS) / (af_against + RATIO_EPS)

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-02: Breadth-weighted ratio ───

def exp_breadth_weighted(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            ratio = (af[0] + RATIO_EPS) / (af[1] + RATIO_EPS)
            breadth = af[3] / max(af[3] + af[4], 1)
            class_scores[cls] = ratio * (0.5 + 0.5 * breadth)

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0) * 0.7
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-03: Lower MIN_SUPPORT for rare classes ───

def exp_lower_minsup(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        class_models, class_retained = {}, {}
        class_counts = {cls: int((tl == cls).sum()) for cls in all_cls}
        for cls in all_cls:
            import experiment_runner as mod
            if cls in RARE_CLASSES:
                mod.MIN_SUPPORT = 2
                mod.CONF_SUPPORT = 5
            else:
                mod.MIN_SUPPORT = 3
                mod.CONF_SUPPORT = 10
            cls_models = {}
            cls_actions = defaultdict(list)
            for nd in nodes:
                nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls, use_supervised_bins=True)
                cls_models.update(nm)
                for a in na: cls_actions[a[1]].append(a)
            cls_ret = []
            for nd in nodes:
                cls_ret.extend(refine_ovr_node_cfg(
                    nd, cls_models, cls_actions.get(nd.nid, []), td, rank_by='score', fpc=18))
            class_models[cls] = cls_models
            class_retained[cls] = cls_ret
        mod.MIN_SUPPORT = 3
        mod.CONF_SUPPORT = 10
        return class_models, class_retained

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-04: Max-of-top-K AF (peak evidence) ───

def exp_peak_evidence(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            ratio = (af[0] + RATIO_EPS) / (af[1] + RATIO_EPS)
            peak = af[5]
            class_scores[cls] = ratio * (1 + peak * 3)

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-05: Stricter FP filter (healthy needs higher margin over disease) ───

def exp_strict_fp(data, labels, is_bin, seed):
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
            if h_score > 2.0 and score < h_score * 1.8:
                continue
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-06: AF_for only for rare classes, ratio for common ───

def exp_affor_rare(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            if cls in RARE_CLASSES:
                class_scores[cls] = af[0] * 10
            else:
                class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-07: Lower against for rare, higher for common (asymmetric) ───

def exp_asymmetric_against(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            ag_scale = 0.5 if cls in RARE_CLASSES else 0.8
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=ag_scale)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-08: Fewer bins for rare classes (3 instead of 6) ───

def exp_fewer_bins_rare(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        class_models, class_retained = {}, {}
        class_counts = {cls: int((tl == cls).sum()) for cls in all_cls}
        import experiment_runner as mod
        for cls in all_cls:
            if cls in RARE_CLASSES:
                mod.MAX_BINS = 3
            else:
                mod.MAX_BINS = 6
            cls_models = {}
            cls_actions = defaultdict(list)
            for nd in nodes:
                nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls, use_supervised_bins=True)
                cls_models.update(nm)
                for a in na: cls_actions[a[1]].append(a)
            cls_ret = []
            for nd in nodes:
                cls_ret.extend(refine_ovr_node_cfg(
                    nd, cls_models, cls_actions.get(nd.nid, []), td, rank_by='score', fpc=18))
            class_models[cls] = cls_models
            class_retained[cls] = cls_ret
        mod.MAX_BINS = 6
        return class_models, class_retained

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-09: Relaxed correlation for rare classes (keep more diverse features) ───

def exp_relaxed_corr_rare(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        class_models, class_retained = {}, {}
        for cls in all_cls:
            corr_thresh = 0.9 if cls in RARE_CLASSES else 0.8
            cls_models = {}
            cls_actions = defaultdict(list)
            for nd in nodes:
                nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls, use_supervised_bins=True)
                cls_models.update(nm)
                for a in na: cls_actions[a[1]].append(a)
            cls_ret = []
            for nd in nodes:
                node_actions = cls_actions.get(nd.nid, [])
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
                        if max_corr > corr_thresh: continue
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


# ─── W8-10: Breadth gate + lower threshold ───

def exp_breadth_gate(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        class_breadth = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            class_breadth[cls] = af[3] / max(af[3] + af[4], 1)

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            breadth = class_breadth.get(cls, 0)
            if breadth > 0.5:
                t *= 0.85
            elif breadth < 0.2:
                t *= 1.15
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-11: Top-3 dominant (only top 3 features per class contribute) ───

def exp_top3_dominant(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            lvl_nodes = _route_user(uid, data, nodes)
            retained = cr[cls]
            models = cm[cls]
            fisher_map = {}
            for a in retained:
                fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
            max_fisher = max(fisher_map.values()) if fisher_map else 1.0

            contribs_for, contribs_against = [], []
            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for a in retained:
                        if a[1] != nd.nid: continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v): continue
                        mo = models.get((nd.nid, f))
                        if not mo: continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'), 0, mo.n_bins-1))
                        bc = mo.bin_counts[bi]
                        if bc < MIN_SUPPORT: continue
                        shift = mo.p_class[bi] - mo.prior
                        conf = min(1.0, bc / CONF_SUPPORT)
                        fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                        weighted = abs(shift) * conf * fw
                        if shift >= 0:
                            contribs_for.append(weighted)
                        else:
                            contribs_against.append(weighted)

            contribs_for.sort(reverse=True)
            contribs_against.sort(reverse=True)
            af_for = sum(contribs_for[:3])
            af_against = sum(contribs_against[:3]) * 0.8
            class_scores[cls] = (af_for + RATIO_EPS) / (af_against + RATIO_EPS)

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W8-12: Z-score normalized scores (relative to training distribution) ───

def exp_zscore_norm(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
        n_train = td.shape[0]
        score_stats = {}
        for cls in all_cls:
            scores = []
            for uid in range(n_train):
                af = compute_af_cfg(uid, td, nodes, cm[cls], cr[cls], against_scale=0.8)
                s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
                scores.append(s)
            scores = np.array(scores)
            score_stats[cls] = (float(scores.mean()), float(scores.std()) + 1e-6)
        return (cm, cr, score_stats)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, score_stats = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            raw = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            mean, std = score_stats[cls]
            class_scores[cls] = (raw - mean) / std

        h_score = class_scores.get(HEALTHY, 0)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            if score > 2.0 and score > h_score:
                candidates[cls] = score
            elif score > 3.0:
                candidates[cls] = score
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


experiments = [
    ("W8-01 Top-K weighted AF",        exp_topk_weighted),
    ("W8-02 Breadth-weighted ratio",   exp_breadth_weighted),
    ("W8-03 Lower MIN_SUP rare",       exp_lower_minsup),
    ("W8-04 Peak evidence boost",      exp_peak_evidence),
    ("W8-05 Strict FP filter",         exp_strict_fp),
    ("W8-06 AF_for only rare",         exp_affor_rare),
    ("W8-07 Asymmetric against",       exp_asymmetric_against),
    ("W8-08 Fewer bins rare",          exp_fewer_bins_rare),
    ("W8-09 Relaxed corr rare",        exp_relaxed_corr_rare),
    ("W8-10 Breadth gate",             exp_breadth_gate),
    ("W8-11 Top-3 dominant",           exp_top3_dominant),
    ("W8-12 Z-score normalized",       exp_zscore_norm),
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
