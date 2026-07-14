"""Wave 6 — combine the best wave 5 ideas to break past 85.3%.

Winners from wave 5:
- W5-08 Confusion-aware boost: 85.3% (sens 82.5%, BA 78.1%)
- W5-10 Age-stratified tree: 85.3% (sens 82.5%, BA 78.1%)
- W5-03 Second opinion: 85.1% (sens 83.0%, BA 78.3%)

Strategy: combine these complementary mechanisms + explore variations.
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


def _build_age_tree(data, labels, is_bin):
    SEX_FEAT, AGE_FEAT = 1, 2
    n = data.shape[0]
    root = Node("root", 1, np.arange(n), _hdist(labels, np.arange(n)))
    all_nodes = [root]
    ctr = [0]
    sex_col = data[:, SEX_FEAT]
    for sex_val in sorted(set(sex_col[~np.isnan(sex_col)])):
        mask = sex_col == sex_val
        idx = np.where(mask)[0]
        if len(idx) < U_MIN: continue
        ctr[0] += 1
        sex_nd = Node(f"L2_sex{int(sex_val)}_{ctr[0]}", 2, idx, _hdist(labels, idx))
        sex_nd.bfeat = SEX_FEAT
        sex_nd.bbin = True
        sex_nd.bvset = frozenset({sex_val})
        sex_nd.parent = root
        sex_nd.ancestor_feats = frozenset({SEX_FEAT})
        root.children.append(sex_nd)
        all_nodes.append(sex_nd)
        age_col = data[idx, AGE_FEAT]
        valid_ages = age_col[~np.isnan(age_col)]
        if len(valid_ages) < 2 * U_MIN: continue
        med_age = float(np.median(valid_ages))
        for lo, hi, label in [(-np.inf, med_age, "young"), (med_age, np.inf, "old")]:
            sub_mask = (age_col > lo) if hi == np.inf else (age_col <= hi)
            sub_idx = idx[sub_mask & ~np.isnan(age_col)]
            if len(sub_idx) < U_MIN: continue
            ctr[0] += 1
            age_nd = Node(f"L3_{label}_{ctr[0]}", 3, sub_idx, _hdist(labels, sub_idx))
            age_nd.bfeat = AGE_FEAT
            age_nd.blo, age_nd.bhi = lo, hi
            age_nd.parent = sex_nd
            age_nd.ancestor_feats = frozenset({SEX_FEAT, AGE_FEAT})
            sex_nd.children.append(age_nd)
            all_nodes.append(age_nd)
    return all_nodes


# ─── W6-01: Age tree + confusion-aware boost ───

def exp_age_confusion(data, labels, is_bin, seed):
    def train(nodes_unused, td, tl, is_bin, all_cls):
        nodes = _build_age_tree(td, tl, is_bin)
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
        n_train = td.shape[0]
        class_acc = {}
        for cls in all_cls:
            correct, total = 0, 0
            for uid in range(n_train):
                if tl[uid] == cls:
                    pred, _ = predict_cfg(uid, td, nodes, cm, cr, all_cls, against_scale=0.8)
                    total += 1
                    if pred == cls: correct += 1
            class_acc[cls] = correct / max(total, 1)
        boost = {}
        for cls in all_cls:
            if cls == HEALTHY:
                boost[cls] = 1.0
            else:
                boost[cls] = 1.0 + max(0, (0.8 - class_acc[cls])) * 2.0
        return (nodes, cm, cr, boost)

    def predict(uid, data, nodes_unused, all_cls, tr):
        nodes, cm, cr, boost = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            class_scores[cls] = s * boost.get(cls, 1.0)
        h_score = class_scores.get(HEALTHY, 1.0) / boost.get(HEALTHY, 1.0)
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


# ─── W6-02: Age tree + second opinion ───

def exp_age_second_opinion(data, labels, is_bin, seed):
    def train(nodes_unused, td, tl, is_bin, all_cls):
        nodes = _build_age_tree(td, tl, is_bin)
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True, fpc=18)
        cr2 = {}
        for cls in all_cls:
            primary_feats = set(a[0] for a in cr[cls])
            cls_actions = []
            for nd in nodes:
                for key, mo in cm[cls].items():
                    if key[0] != nd.nid: continue
                    f = key[1]
                    if f in primary_feats: continue
                    score = sum(abs(mo.p_class[b] - mo.prior) * min(1.0, float(mo.bin_counts[b])/CONF_SUPPORT)
                               for b in range(mo.n_bins) if mo.bin_counts[b] >= MIN_SUPPORT)
                    if score > 0.001:
                        tv = td[nd.uidx][tl[nd.uidx]==cls, f]
                        rv = td[nd.uidx][tl[nd.uidx]!=cls, f]
                        tv, rv = tv[~np.isnan(tv)], rv[~np.isnan(rv)]
                        fisher = (tv.mean()-rv.mean())**2/(tv.var()+rv.var()+1e-10) if len(tv)>=2 and len(rv)>=2 else 0.0
                        cls_actions.append((f, nd.nid, score, fisher))
            cls_actions.sort(key=lambda x: x[2], reverse=True)
            cr2[cls] = cls_actions[:12]
        return (nodes, cm, cr, cr2)

    def predict(uid, data, nodes_unused, all_cls, tr):
        nodes, cm, cr, cr2 = tr
        pred, class_scores = predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
        if pred == HEALTHY:
            h_score = class_scores.get(HEALTHY, 1.0)
            for cls in all_cls:
                if cls == HEALTHY: continue
                t = CLASS_THRESHOLDS.get(cls, 3.0)
                margin = class_scores.get(cls, 0) / max(t, 0.1)
                if margin > 0.7 and cr2.get(cls):
                    af2 = compute_af_cfg(uid, data, nodes, cm[cls], cr2[cls], against_scale=0.8)
                    s2 = score_patient(af2[0], af2[1], af2[3], af2[4], af2[5], 'ratio')
                    if s2 > t * 1.2:
                        combined = 0.6 * class_scores[cls] + 0.4 * s2
                        if combined > max(t, min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)):
                            pred = cls
                            break
        return pred, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W6-03: Triple combo — age tree + confusion + second opinion ───

def exp_triple_combo(data, labels, is_bin, seed):
    def train(nodes_unused, td, tl, is_bin, all_cls):
        nodes = _build_age_tree(td, tl, is_bin)
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True, fpc=18)
        n_train = td.shape[0]
        class_acc = {}
        for cls in all_cls:
            correct, total = 0, 0
            for uid in range(n_train):
                if tl[uid] == cls:
                    pred, _ = predict_cfg(uid, td, nodes, cm, cr, all_cls, against_scale=0.8)
                    total += 1
                    if pred == cls: correct += 1
            class_acc[cls] = correct / max(total, 1)
        boost = {cls: 1.0 if cls == HEALTHY else 1.0 + max(0, (0.8 - class_acc[cls])) * 2.0
                 for cls in all_cls}
        cr2 = {}
        for cls in all_cls:
            primary_feats = set(a[0] for a in cr[cls])
            cls_actions = []
            for nd in nodes:
                for key, mo in cm[cls].items():
                    if key[0] != nd.nid: continue
                    f = key[1]
                    if f in primary_feats: continue
                    score = sum(abs(mo.p_class[b] - mo.prior) * min(1.0, float(mo.bin_counts[b])/CONF_SUPPORT)
                               for b in range(mo.n_bins) if mo.bin_counts[b] >= MIN_SUPPORT)
                    if score > 0.001:
                        tv = td[nd.uidx][tl[nd.uidx]==cls, f]
                        rv = td[nd.uidx][tl[nd.uidx]!=cls, f]
                        tv, rv = tv[~np.isnan(tv)], rv[~np.isnan(rv)]
                        fisher = (tv.mean()-rv.mean())**2/(tv.var()+rv.var()+1e-10) if len(tv)>=2 and len(rv)>=2 else 0.0
                        cls_actions.append((f, nd.nid, score, fisher))
            cls_actions.sort(key=lambda x: x[2], reverse=True)
            cr2[cls] = cls_actions[:12]
        return (nodes, cm, cr, boost, cr2)

    def predict(uid, data, nodes_unused, all_cls, tr):
        nodes, cm, cr, boost, cr2 = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            class_scores[cls] = s * boost.get(cls, 1.0)
        h_score = class_scores.get(HEALTHY, 1.0) / boost.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        pred = max(candidates, key=candidates.get) if candidates else HEALTHY

        if pred == HEALTHY:
            for cls in all_cls:
                if cls == HEALTHY: continue
                t = CLASS_THRESHOLDS.get(cls, 3.0)
                margin = class_scores.get(cls, 0) / max(t, 0.1)
                if margin > 0.7 and cr2.get(cls):
                    af2 = compute_af_cfg(uid, data, nodes, cm[cls], cr2[cls], against_scale=0.8)
                    s2 = score_patient(af2[0], af2[1], af2[3], af2[4], af2[5], 'ratio')
                    if s2 > t * 1.2:
                        combined = 0.6 * class_scores[cls] + 0.4 * s2
                        if combined > max(t, healthy_bar):
                            pred = cls
                            break
        return pred, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W6-04: Confusion boost with softer scaling (1.5x instead of 2x) ───

def exp_confusion_soft(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
        n_train = td.shape[0]
        class_acc = {}
        for cls in all_cls:
            correct, total = 0, 0
            for uid in range(n_train):
                if tl[uid] == cls:
                    pred, _ = predict_cfg(uid, td, nodes, cm, cr, all_cls, against_scale=0.8)
                    total += 1
                    if pred == cls: correct += 1
            class_acc[cls] = correct / max(total, 1)
        boost = {cls: 1.0 if cls == HEALTHY else 1.0 + max(0, (0.8 - class_acc[cls])) * 1.5
                 for cls in all_cls}
        return (cm, cr, boost)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, boost = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            class_scores[cls] = s * boost.get(cls, 1.0)
        h_score = class_scores.get(HEALTHY, 1.0) / boost.get(HEALTHY, 1.0)
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


# ─── W6-05: Confusion boost 3x (more aggressive) ───

def exp_confusion_strong(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
        n_train = td.shape[0]
        class_acc = {}
        for cls in all_cls:
            correct, total = 0, 0
            for uid in range(n_train):
                if tl[uid] == cls:
                    pred, _ = predict_cfg(uid, td, nodes, cm, cr, all_cls, against_scale=0.8)
                    total += 1
                    if pred == cls: correct += 1
            class_acc[cls] = correct / max(total, 1)
        boost = {cls: 1.0 if cls == HEALTHY else 1.0 + max(0, (0.8 - class_acc[cls])) * 3.0
                 for cls in all_cls}
        return (cm, cr, boost)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, boost = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            class_scores[cls] = s * boost.get(cls, 1.0)
        h_score = class_scores.get(HEALTHY, 1.0) / boost.get(HEALTHY, 1.0)
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


# ─── W6-06: Confusion boost with threshold lowering (lower threshold for weak classes) ───

def exp_confusion_threshold(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
        n_train = td.shape[0]
        class_acc = {}
        for cls in all_cls:
            correct, total = 0, 0
            for uid in range(n_train):
                if tl[uid] == cls:
                    pred, _ = predict_cfg(uid, td, nodes, cm, cr, all_cls, against_scale=0.8)
                    total += 1
                    if pred == cls: correct += 1
            class_acc[cls] = correct / max(total, 1)
        adapted_t = {}
        for cls in all_cls:
            if cls == HEALTHY: continue
            base_t = CLASS_THRESHOLDS.get(cls, 3.0)
            acc = class_acc[cls]
            if acc < 0.6:
                adapted_t[cls] = base_t * 0.75
            elif acc < 0.75:
                adapted_t[cls] = base_t * 0.85
            else:
                adapted_t[cls] = base_t
        return (cm, cr, adapted_t)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, adapted_t = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          thresholds=adapted_t, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W6-07: Second opinion with lower trigger threshold (0.5 instead of 0.7) ───

def exp_second_opinion_aggressive(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True, fpc=18)
        cr2 = {}
        for cls in all_cls:
            primary_feats = set(a[0] for a in cr[cls])
            cls_actions = []
            for nd in nodes:
                for key, mo in cm[cls].items():
                    if key[0] != nd.nid: continue
                    f = key[1]
                    if f in primary_feats: continue
                    score = sum(abs(mo.p_class[b] - mo.prior) * min(1.0, float(mo.bin_counts[b])/CONF_SUPPORT)
                               for b in range(mo.n_bins) if mo.bin_counts[b] >= MIN_SUPPORT)
                    if score > 0.001:
                        tv = td[nd.uidx][tl[nd.uidx]==cls, f]
                        rv = td[nd.uidx][tl[nd.uidx]!=cls, f]
                        tv, rv = tv[~np.isnan(tv)], rv[~np.isnan(rv)]
                        fisher = (tv.mean()-rv.mean())**2/(tv.var()+rv.var()+1e-10) if len(tv)>=2 and len(rv)>=2 else 0.0
                        cls_actions.append((f, nd.nid, score, fisher))
            cls_actions.sort(key=lambda x: x[2], reverse=True)
            cr2[cls] = cls_actions[:12]
        return (cm, cr, cr2)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, cr2 = tr
        pred, class_scores = predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
        if pred == HEALTHY:
            h_score = class_scores.get(HEALTHY, 1.0)
            for cls in all_cls:
                if cls == HEALTHY: continue
                t = CLASS_THRESHOLDS.get(cls, 3.0)
                margin = class_scores.get(cls, 0) / max(t, 0.1)
                if margin > 0.5 and cr2.get(cls):
                    af2 = compute_af_cfg(uid, data, nodes, cm[cls], cr2[cls], against_scale=0.8)
                    s2 = score_patient(af2[0], af2[1], af2[3], af2[4], af2[5], 'ratio')
                    if s2 > t:
                        combined = 0.5 * class_scores[cls] + 0.5 * s2
                        if combined > max(t, min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)):
                            pred = cls
                            break
        return pred, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W6-08: Dynamic against + second opinion ───

def exp_dynamic_second(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True, fpc=18)
        cr2 = {}
        for cls in all_cls:
            primary_feats = set(a[0] for a in cr[cls])
            cls_actions = []
            for nd in nodes:
                for key, mo in cm[cls].items():
                    if key[0] != nd.nid: continue
                    f = key[1]
                    if f in primary_feats: continue
                    score = sum(abs(mo.p_class[b] - mo.prior) * min(1.0, float(mo.bin_counts[b])/CONF_SUPPORT)
                               for b in range(mo.n_bins) if mo.bin_counts[b] >= MIN_SUPPORT)
                    if score > 0.001:
                        tv = td[nd.uidx][tl[nd.uidx]==cls, f]
                        rv = td[nd.uidx][tl[nd.uidx]!=cls, f]
                        tv, rv = tv[~np.isnan(tv)], rv[~np.isnan(rv)]
                        fisher = (tv.mean()-rv.mean())**2/(tv.var()+rv.var()+1e-10) if len(tv)>=2 and len(rv)>=2 else 0.0
                        cls_actions.append((f, nd.nid, score, fisher))
            cls_actions.sort(key=lambda x: x[2], reverse=True)
            cr2[cls] = cls_actions[:12]
        return (cm, cr, cr2)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, cr2 = tr
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
            n_for, n_against = 0, 0
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
                            af_for += weighted
                            n_for += 1
                        else:
                            af_against += weighted
                            n_against += 1
            breadth = n_for / max(n_for + n_against, 1)
            dynamic_as = 0.5 + 0.5 * (1 - breadth)
            af_against *= dynamic_as
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
        pred = max(candidates, key=candidates.get) if candidates else HEALTHY

        if pred == HEALTHY:
            for cls in all_cls:
                if cls == HEALTHY: continue
                t = CLASS_THRESHOLDS.get(cls, 3.0)
                margin = class_scores.get(cls, 0) / max(t, 0.1)
                if margin > 0.7 and cr2.get(cls):
                    af2 = compute_af_cfg(uid, data, nodes, cm[cls], cr2[cls], against_scale=0.8)
                    s2 = score_patient(af2[0], af2[1], af2[3], af2[4], af2[5], 'ratio')
                    if s2 > t * 1.2:
                        combined = 0.6 * class_scores[cls] + 0.4 * s2
                        if combined > max(t, healthy_bar):
                            pred = cls
                            break
        return pred, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W6-09: Against 0.75 (between 0.7 and 0.8) ───

def exp_against_075(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.75)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W6-10: Against 0.85 ───

def exp_against_085(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.85)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W6-11: Age tree + against 0.75 ───

def exp_age_075(data, labels, is_bin, seed):
    def train(nodes_unused, td, tl, is_bin, all_cls):
        nodes = _build_age_tree(td, tl, is_bin)
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
        return (nodes, cm, cr)
    def predict(uid, data, nodes_unused, all_cls, tr):
        nodes, cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.75)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W6-12: Age tree + against 0.85 ───

def exp_age_085(data, labels, is_bin, seed):
    def train(nodes_unused, td, tl, is_bin, all_cls):
        nodes = _build_age_tree(td, tl, is_bin)
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
        return (nodes, cm, cr)
    def predict(uid, data, nodes_unused, all_cls, tr):
        nodes, cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.85)
    return run_10fold(data, labels, is_bin, seed, predict, train)


experiments = [
    ("W6-01 Age+confusion boost",       exp_age_confusion),
    ("W6-02 Age+second opinion",        exp_age_second_opinion),
    ("W6-03 Triple combo",              exp_triple_combo),
    ("W6-04 Confusion 1.5x",           exp_confusion_soft),
    ("W6-05 Confusion 3x",             exp_confusion_strong),
    ("W6-06 Confusion+threshold",       exp_confusion_threshold),
    ("W6-07 Second opinion aggr",       exp_second_opinion_aggressive),
    ("W6-08 Dynamic+second opinion",    exp_dynamic_second),
    ("W6-09 Against 0.75",             exp_against_075),
    ("W6-10 Against 0.85",             exp_against_085),
    ("W6-11 Age+against 0.75",         exp_age_075),
    ("W6-12 Age+against 0.85",         exp_age_085),
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
