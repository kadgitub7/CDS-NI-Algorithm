"""Wave 7 — radical structural departures to break 85.3%.

The AF mechanism has hit an information ceiling. These experiments
bypass or fundamentally restructure the scoring pipeline.

Ideas:
1. Naive Bayes posterior (skip AF, use log-posterior directly)
2. Percentile thresholds (calibrate thresholds from training score distribution)
3. Class-pair disambiguation (train pairwise classifiers for confused classes)
4. Stacked correction (train correction model on first-pass errors)
5. Weighted vote by bin occupancy (bins with more target weight higher)
6. Per-node scoring (separate score from each tree node, ensemble)
7. Leave-one-class-out (train "not class X" and invert)
8. Top-K feature consensus (only call disease if top-K features agree)
9. Soft against (sigmoid scaling of against contribution)
10. Max posterior (use max bin posterior instead of aggregate)
11. Weighted Laplace by class size (more smoothing for rare classes)
12. Against 0.8 + relaxed thresholds for rare classes only
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
    run_10fold, stats, BinModel, Node, _hdist,
    HEALTHY, CLASS_THRESHOLDS, RATIO_EPS, HEALTHY_WEIGHT, HEALTHY_BAR_CAP,
    SUSPICION_HCUT, SUSPICION_OFFSET, MIN_SUPPORT, CONF_SUPPORT,
    N_FEAT, LAPLACE_ALPHA, U_MIN,
    _supervised_bin_edges, _fast_abs_corr,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


# ─── W7-01: Naive Bayes log-posterior (bypass AF entirely) ───

def exp_naive_bayes(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        return (cm, cr)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_log_post = {}
        for cls in all_cls:
            lvl_nodes = _route_user(uid, data, nodes)
            retained = cr[cls]
            models = cm[cls]
            log_post = 0.0
            n_used = 0
            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for a in retained:
                        if a[1] != nd.nid: continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v): continue
                        mo = models.get((nd.nid, f))
                        if not mo: continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                        0, mo.n_bins-1))
                        bc = mo.bin_counts[bi]
                        if bc < MIN_SUPPORT: continue
                        p = np.clip(mo.p_class[bi], 0.001, 0.999)
                        prior = np.clip(mo.prior, 0.001, 0.999)
                        log_lr = np.log(p) - np.log(prior)
                        log_post += log_lr
                        n_used += 1
            class_log_post[cls] = log_post

        best_disease = max((c for c in all_cls if c != HEALTHY),
                          key=lambda c: class_log_post.get(c, -999))
        h_lp = class_log_post.get(HEALTHY, 0)
        d_lp = class_log_post.get(best_disease, 0)

        if d_lp > h_lp * 0.5 and d_lp > 0.5:
            return best_disease, class_log_post
        return HEALTHY, class_log_post

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W7-02: Percentile thresholds (from training score distribution) ───

def exp_percentile_thresh(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        n_train = td.shape[0]
        pct_thresholds = {}
        for cls in all_cls:
            if cls == HEALTHY: continue
            healthy_scores = []
            for uid in range(n_train):
                if tl[uid] == HEALTHY:
                    af = compute_af_cfg(uid, td, nodes, cm[cls], cr[cls], against_scale=0.8)
                    s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
                    healthy_scores.append(s)
            if healthy_scores:
                pct_thresholds[cls] = np.percentile(healthy_scores, 95)
            else:
                pct_thresholds[cls] = CLASS_THRESHOLDS.get(cls, 3.0)
        return (cm, cr, pct_thresholds)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, pct_t = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          thresholds=pct_t, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W7-03: Class-pair disambiguation ───

def exp_pairwise(data, labels, is_bin, seed):
    CONFUSED_PAIRS = [(1, 2), (1, 6), (1, 10), (2, 6), (4, 5)]

    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        pair_models = {}
        for c1, c2 in CONFUSED_PAIRS:
            mask = (tl == c1) | (tl == c2)
            if mask.sum() < 20: continue
            pair_data = td[mask]
            pair_labels = tl[mask]
            pair_nodes = build_tree(pair_data, pair_labels, is_bin)
            p_cm, p_cr = train_all_cfg(pair_nodes, pair_data, pair_labels,
                                        is_bin, [c1, c2], use_supervised_bins=True)
            pair_models[(c1, c2)] = (pair_nodes, p_cm, p_cr, pair_data, pair_labels)
        return (cm, cr, pair_models)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, pair_models = tr
        pred, class_scores = predict_cfg(uid, data, nodes, cm, cr, all_cls,
                                         against_scale=0.8)
        for (c1, c2), (p_nodes, p_cm, p_cr, pd, pl) in pair_models.items():
            if pred in (c1, c2):
                other = c2 if pred == c1 else c1
                af_pred = compute_af_cfg(uid, data, p_nodes, p_cm[pred], p_cr[pred],
                                         against_scale=0.8)
                af_other = compute_af_cfg(uid, data, p_nodes, p_cm[other], p_cr[other],
                                          against_scale=0.8)
                s_pred = score_patient(af_pred[0], af_pred[1], af_pred[3], af_pred[4],
                                       af_pred[5], 'ratio')
                s_other = score_patient(af_other[0], af_other[1], af_other[3], af_other[4],
                                        af_other[5], 'ratio')
                if s_other > s_pred * 1.3:
                    pred = other
                    break
        return pred, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W7-04: Stacked correction (2nd pass on 1st-pass errors) ───

def exp_stacked(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        n_train = td.shape[0]
        error_patterns = defaultdict(list)
        for uid in range(n_train):
            pred, scores = predict_cfg(uid, td, nodes, cm, cr, all_cls, against_scale=0.8)
            true_cls = tl[uid]
            if pred != true_cls:
                score_vec = [scores.get(c, 0) for c in sorted(all_cls)]
                error_patterns[(pred, true_cls)].append(score_vec)
        corrections = {}
        for (pred_cls, true_cls), vecs in error_patterns.items():
            if len(vecs) < 3: continue
            vecs = np.array(vecs)
            mean_vec = vecs.mean(axis=0)
            corrections[(pred_cls, true_cls)] = (mean_vec, len(vecs))
        return (cm, cr, corrections, sorted(all_cls))

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, corrections, cls_order = tr
        pred, class_scores = predict_cfg(uid, data, nodes, cm, cr, all_cls,
                                         against_scale=0.8)
        score_vec = np.array([class_scores.get(c, 0) for c in cls_order])
        best_correction = None
        best_sim = -1
        for (p, t), (mean_vec, count) in corrections.items():
            if p != pred: continue
            if count < 3: continue
            diff = score_vec - mean_vec
            sim = 1.0 / (1.0 + np.sqrt(np.sum(diff**2)))
            if sim > best_sim:
                best_sim = sim
                best_correction = t
        if best_correction is not None and best_sim > 0.3:
            pred = best_correction
        return pred, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W7-05: Weighted vote by bin target density ───

def exp_density_vote(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True)

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

            af_for, af_against = 0.0, 0.0
            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for a in retained:
                        if a[1] != nd.nid: continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v): continue
                        mo = models.get((nd.nid, f))
                        if not mo: continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                        0, mo.n_bins-1))
                        bc = mo.bin_counts[bi]
                        if bc < MIN_SUPPORT: continue
                        tc = mo.target_counts[bi]
                        shift = mo.p_class[bi] - mo.prior
                        conf = min(1.0, bc / CONF_SUPPORT)
                        fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                        density_w = np.sqrt(max(tc, 0.1)) / np.sqrt(max(bc, 1))
                        weighted = abs(shift) * conf * fw * (1 + density_w)
                        if shift >= 0:
                            af_for += weighted
                        else:
                            af_against += weighted * 0.8

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


# ─── W7-06: Per-node ensembling (each node votes, majority wins) ───

def exp_node_ensemble(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        node_votes = defaultdict(lambda: defaultdict(float))
        lvl_nodes = _route_user(uid, data, nodes)
        for cls in all_cls:
            retained = cr[cls]
            models = cm[cls]
            fisher_map = {}
            for a in retained:
                fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
            max_fisher = max(fisher_map.values()) if fisher_map else 1.0

            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    af_for, af_against = 0.0, 0.0
                    for a in retained:
                        if a[1] != nd.nid: continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v): continue
                        mo = models.get((nd.nid, f))
                        if not mo: continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                        0, mo.n_bins-1))
                        bc = mo.bin_counts[bi]
                        if bc < MIN_SUPPORT: continue
                        shift = mo.p_class[bi] - mo.prior
                        conf = min(1.0, bc / CONF_SUPPORT)
                        fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                        weighted = abs(shift) * conf * fw
                        if shift >= 0:
                            af_for += weighted
                        else:
                            af_against += weighted * 0.8
                    node_votes[nd.nid][cls] = (af_for + RATIO_EPS) / (af_against + RATIO_EPS)

        class_scores = {}
        for cls in all_cls:
            scores = [node_votes[nid][cls] for nid in node_votes if cls in node_votes[nid]]
            class_scores[cls] = np.mean(scores) if scores else 1.0

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


# ─── W7-07: Top-K feature consensus ───

def exp_topk_consensus(data, labels, is_bin, seed):
    K = 5

    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        class_consensus = {}
        for cls in all_cls:
            lvl_nodes = _route_user(uid, data, nodes)
            retained = cr[cls]
            models = cm[cls]
            feature_votes = []
            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for a in retained:
                        if a[1] != nd.nid: continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v): continue
                        mo = models.get((nd.nid, f))
                        if not mo: continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                        0, mo.n_bins-1))
                        bc = mo.bin_counts[bi]
                        if bc < MIN_SUPPORT: continue
                        shift = mo.p_class[bi] - mo.prior
                        feature_votes.append((a[2], shift > 0))

            feature_votes.sort(key=lambda x: x[0], reverse=True)
            top_k = feature_votes[:K]
            n_for = sum(1 for _, v in top_k if v)
            class_consensus[cls] = n_for / max(len(top_k), 1)

            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            if class_consensus.get(cls, 0) < 0.6: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W7-08: Sigmoid against scaling ───

def exp_sigmoid_against(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True)

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

            af_for, af_against = 0.0, 0.0
            contribs = []
            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for a in retained:
                        if a[1] != nd.nid: continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v): continue
                        mo = models.get((nd.nid, f))
                        if not mo: continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                        0, mo.n_bins-1))
                        bc = mo.bin_counts[bi]
                        if bc < MIN_SUPPORT: continue
                        shift = mo.p_class[bi] - mo.prior
                        conf = min(1.0, bc / CONF_SUPPORT)
                        fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                        weighted = abs(shift) * conf * fw
                        if shift >= 0:
                            af_for += weighted
                        else:
                            sig = 1.0 / (1.0 + np.exp(-5 * (weighted - 0.1)))
                            af_against += weighted * sig * 0.8

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


# ─── W7-09: Max posterior (strongest single bin decides) ───

def exp_max_posterior(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            ratio = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

            lvl_nodes = _route_user(uid, data, nodes)
            max_p = 0.0
            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for a in cr[cls]:
                        if a[1] != nd.nid: continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v): continue
                        mo = cm[cls].get((nd.nid, f))
                        if not mo: continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                        0, mo.n_bins-1))
                        if mo.bin_counts[bi] >= MIN_SUPPORT:
                            p = mo.p_class[bi]
                            if p > max_p:
                                max_p = p

            class_scores[cls] = ratio * (1 + max_p)

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


# ─── W7-10: Rare-class-aware Laplace (more smoothing for rare classes) ───

def exp_rare_laplace(data, labels, is_bin, seed):
    def train_custom(nodes, td, tl, is_bin, all_cls):
        class_models, class_retained = {}, {}
        class_counts = {cls: int((tl == cls).sum()) for cls in all_cls}
        max_count = max(class_counts.values())
        for cls in all_cls:
            alpha = LAPLACE_ALPHA * (max_count / max(class_counts[cls], 1)) ** 0.5
            alpha = min(alpha, 5.0)
            import experiment_runner as mod
            saved = mod.LAPLACE_ALPHA
            mod.LAPLACE_ALPHA = alpha
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
                    rank_by='score', fpc=18))
            class_models[cls] = cls_models
            class_retained[cls] = cls_ret
            mod.LAPLACE_ALPHA = saved
        return class_models, class_retained

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train_custom)


# ─── W7-11: Relaxed thresholds for rare classes only ───

def exp_rare_relaxed(data, labels, is_bin, seed):
    RARE_CLASSES = {4, 5, 9}
    relaxed_t = dict(CLASS_THRESHOLDS)
    for cls in RARE_CLASSES:
        relaxed_t[cls] = relaxed_t.get(cls, 3.0) * 0.8

    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          thresholds=relaxed_t, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W7-12: Hybrid — AF ratio + naive Bayes tiebreaker ───

def exp_hybrid_bayes(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        class_bayes = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

            lvl_nodes = _route_user(uid, data, nodes)
            log_lr_sum = 0.0
            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for a in cr[cls]:
                        if a[1] != nd.nid: continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v): continue
                        mo = cm[cls].get((nd.nid, f))
                        if not mo: continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                        0, mo.n_bins-1))
                        if mo.bin_counts[bi] >= MIN_SUPPORT:
                            p = np.clip(mo.p_class[bi], 0.01, 0.99)
                            prior = np.clip(mo.prior, 0.01, 0.99)
                            log_lr_sum += np.log(p/(1-p)) - np.log(prior/(1-prior))
            class_bayes[cls] = log_lr_sum

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

        if len(candidates) > 1:
            for cls in list(candidates.keys()):
                if class_bayes.get(cls, 0) < 0:
                    del candidates[cls]

        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


experiments = [
    ("W7-01 Naive Bayes posterior",    exp_naive_bayes),
    ("W7-02 Percentile thresholds",    exp_percentile_thresh),
    ("W7-03 Pairwise disambiguation",  exp_pairwise),
    ("W7-04 Stacked correction",       exp_stacked),
    ("W7-05 Density-weighted vote",    exp_density_vote),
    ("W7-06 Per-node ensemble",        exp_node_ensemble),
    ("W7-07 Top-K consensus",          exp_topk_consensus),
    ("W7-08 Sigmoid against",          exp_sigmoid_against),
    ("W7-09 Max posterior boost",      exp_max_posterior),
    ("W7-10 Rare-class Laplace",       exp_rare_laplace),
    ("W7-11 Rare class relaxed thresh", exp_rare_relaxed),
    ("W7-12 Hybrid AF+Bayes",         exp_hybrid_bayes),
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
