"""Wave 5 — novel mechanisms to break the 85.3% ceiling.
Focus: completely different algorithmic ideas, not parameter tuning.

Novel ideas:
1. Learned thresholds (optimize per-class thresholds on training fold)
2. Multi-resolution bins (combine supervised + equal-width evidence)
3. Second opinion (re-evaluate borderline cases with reserve features)
4. KNN backup (neighbor-informed for low-confidence predictions)
5. Feature interaction pairs (ratios/diffs of top feature pairs)
6. Posterior product scoring (multiply posteriors instead of additive AF)
7. Consensus gating (require agreement of two scoring methods)
8. Confusion-aware training (weight classes by their confusion rates)
9. Dynamic against scale (scale against by confidence level)
10. Stratified tree (age-based branching in addition to sex)
11. Bayesian posterior combination (log-odds instead of AF)
12. Calibrated probability (Platt-style sigmoid calibration on scores)
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
    N_FEAT, LAPLACE_ALPHA, CORR_THRESHOLD, MAX_BINS, U_MIN,
    _supervised_bin_edges, _fast_abs_corr,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


# ─── W5-01: Learned thresholds — optimize thresholds on training fold ───

def exp_learned_thresholds(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        n_train = td.shape[0]
        all_scores = {}
        for uid in range(n_train):
            for cls in all_cls:
                af = compute_af_cfg(uid, td, nodes, cm[cls], cr[cls], against_scale=0.8)
                s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
                all_scores[(uid, cls)] = s

        learned_t = {}
        for cls in all_cls:
            if cls == HEALTHY:
                continue
            best_acc, best_t = -1, CLASS_THRESHOLDS.get(cls, 3.0)
            for t_cand in np.arange(1.5, 8.0, 0.25):
                correct = 0
                for uid in range(n_train):
                    true_cls = tl[uid]
                    score = all_scores[(uid, cls)]
                    h_score = all_scores[(uid, HEALTHY)]
                    healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
                    effective_t = max(t_cand, healthy_bar)
                    if h_score < SUSPICION_HCUT:
                        effective_t -= SUSPICION_OFFSET
                    pred_this = score >= effective_t
                    if true_cls == cls and pred_this:
                        correct += 1
                    elif true_cls != cls and not pred_this:
                        correct += 1
                acc = correct / n_train
                if acc > best_acc:
                    best_acc, best_t = acc, t_cand
            learned_t[cls] = best_t
        return (cm, cr, learned_t)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, learned_t = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          thresholds=learned_t, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W5-02: Multi-resolution — combine supervised + equal-width bins ───

def exp_multi_resolution(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm_sv, cr_sv = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                                      use_supervised_bins=True)
        cm_ew, cr_ew = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                                      use_supervised_bins=False)
        return (cm_sv, cr_sv, cm_ew, cr_ew)

    def predict(uid, data, nodes, all_cls, tr):
        cm_sv, cr_sv, cm_ew, cr_ew = tr
        class_scores = {}
        for cls in all_cls:
            af_sv = compute_af_cfg(uid, data, nodes, cm_sv[cls], cr_sv[cls],
                                   against_scale=0.8)
            af_ew = compute_af_cfg(uid, data, nodes, cm_ew[cls], cr_ew[cls],
                                   against_scale=0.8)
            s_sv = score_patient(af_sv[0], af_sv[1], af_sv[3], af_sv[4], af_sv[5], 'ratio')
            s_ew = score_patient(af_ew[0], af_ew[1], af_ew[3], af_ew[4], af_ew[5], 'ratio')
            class_scores[cls] = 0.7 * s_sv + 0.3 * s_ew

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


# ─── W5-03: Second opinion — borderline cases get re-evaluated with reserve features ───

def exp_second_opinion(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True, fpc=18)
        cr2 = {}
        for cls in all_cls:
            primary_feats = set(a[0] for a in cr[cls])
            cls_models = cm[cls]
            cls_actions = []
            for nd in nodes:
                for key, mo in cls_models.items():
                    if key[0] != nd.nid: continue
                    f = key[1]
                    if f in primary_feats: continue
                    score = 0.0
                    for b in range(mo.n_bins):
                        if mo.bin_counts[b] >= MIN_SUPPORT:
                            shift = abs(mo.p_class[b] - mo.prior)
                            conf = min(1.0, float(mo.bin_counts[b]) / CONF_SUPPORT)
                            score += shift * conf
                    if score > 0.001:
                        target_vals = td[nd.uidx][tl[nd.uidx]==cls, f]
                        rest_vals = td[nd.uidx][tl[nd.uidx]!=cls, f]
                        tv = target_vals[~np.isnan(target_vals)]
                        rv = rest_vals[~np.isnan(rest_vals)]
                        if len(tv) >= 2 and len(rv) >= 2:
                            fisher = (tv.mean()-rv.mean())**2 / (tv.var()+rv.var()+1e-10)
                        else:
                            fisher = 0.0
                        cls_actions.append((f, nd.nid, score, fisher))
            cls_actions.sort(key=lambda x: x[2], reverse=True)
            cr2[cls] = cls_actions[:12]
        return (cm, cr, cr2)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, cr2 = tr
        pred, class_scores = predict_cfg(uid, data, nodes, cm, cr, all_cls,
                                         against_scale=0.8)
        if pred == HEALTHY:
            h_score = class_scores.get(HEALTHY, 1.0)
            for cls in all_cls:
                if cls == HEALTHY: continue
                t = CLASS_THRESHOLDS.get(cls, 3.0)
                margin = class_scores.get(cls, 0) / max(t, 0.1)
                if margin > 0.7 and cr2.get(cls):
                    af2 = compute_af_cfg(uid, data, nodes, cm[cls], cr2[cls],
                                         against_scale=0.8)
                    s2 = score_patient(af2[0], af2[1], af2[3], af2[4], af2[5], 'ratio')
                    if s2 > t * 1.2:
                        combined = 0.6 * class_scores[cls] + 0.4 * s2
                        if combined > max(t, min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)):
                            pred = cls
                            break
        return pred, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W5-04: KNN backup — use neighbors for low-confidence cases ───

def exp_knn_backup(data, labels, is_bin, seed):
    K = 5

    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        return (cm, cr, td, tl)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, td, tl = tr
        pred, class_scores = predict_cfg(uid, data, nodes, cm, cr, all_cls,
                                         against_scale=0.8)
        h_score = class_scores.get(HEALTHY, 1.0)
        disease_scores = {c: s for c, s in class_scores.items() if c != HEALTHY}
        max_disease = max(disease_scores.values()) if disease_scores else 0
        t_for_max = CLASS_THRESHOLDS.get(
            max(disease_scores, key=disease_scores.get), 3.0) if disease_scores else 3.0

        is_borderline = (pred == HEALTHY and max_disease > t_for_max * 0.6) or \
                        (pred != HEALTHY and (max_disease - t_for_max) / max(t_for_max, 0.1) < 0.3)

        if is_borderline:
            patient = data[uid]
            dists = []
            for i in range(td.shape[0]):
                diff = patient - td[i]
                valid = ~(np.isnan(patient) | np.isnan(td[i]))
                if valid.sum() > 0:
                    d = np.sqrt(np.nansum(diff[valid]**2) / valid.sum())
                    dists.append((d, tl[i]))
            dists.sort(key=lambda x: x[0])
            nn_labels = [d[1] for d in dists[:K]]
            nn_vote = defaultdict(int)
            for l in nn_labels:
                nn_vote[l] += 1
            knn_pred = max(nn_vote, key=nn_vote.get)

            if pred == HEALTHY and knn_pred != HEALTHY:
                if nn_vote[knn_pred] >= 3:
                    pred = knn_pred
            elif pred != HEALTHY and knn_pred == HEALTHY:
                if nn_vote[HEALTHY] >= 4:
                    pred = HEALTHY

        return pred, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W5-05: Feature interaction pairs — ratio features from top pairs ───

def exp_feature_interactions(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        interaction_models = {}
        for cls in all_cls:
            retained = cr[cls]
            top_feats = list(set(a[0] for a in retained))[:10]
            int_models = {}
            for nd in nodes:
                nd_data, nd_labels = td[nd.uidx], tl[nd.uidx]
                is_target = (nd_labels == cls).astype(float)
                ns = nd.nu
                n_target = int(is_target.sum())
                if n_target < 1: continue
                prior = n_target / ns
                for i, f1 in enumerate(top_feats):
                    for f2 in top_feats[i+1:]:
                        c1, c2 = nd_data[:, f1], nd_data[:, f2]
                        valid = ~(np.isnan(c1) | np.isnan(c2))
                        if valid.sum() < 20: continue
                        v2 = c2[valid]
                        if np.all(np.abs(v2) < 1e-10): continue
                        ratio = c1[valid] / (v2 + 1e-10)
                        edges = _supervised_bin_edges(ratio, is_target[valid], 4)
                        nb = len(edges) - 1
                        ba = np.clip(np.searchsorted(edges[1:], ratio, side='right'), 0, nb-1)
                        lv = nd_labels[valid]
                        bc = np.bincount(ba, minlength=nb)
                        tc = np.bincount(ba[lv==cls], minlength=nb).astype(float)
                        p_class = (tc + LAPLACE_ALPHA * prior) / (bc + LAPLACE_ALPHA)
                        score = sum(abs(p_class[b] - prior) * min(1.0, float(bc[b])/CONF_SUPPORT)
                                   for b in range(nb) if bc[b] >= MIN_SUPPORT)
                        if score > 0.05:
                            int_models[(nd.nid, f1, f2)] = BinModel(
                                nb, edges, bc, tc, p_class, prior, CONF_SUPPORT)
            interaction_models[cls] = int_models
        return (cm, cr, interaction_models)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, int_models = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            base = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            int_bonus = 0.0
            n_int = 0
            lvl_nodes = _route_user(uid, data, nodes)
            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for key, mo in int_models.get(cls, {}).items():
                        if key[0] != nd.nid: continue
                        f1, f2 = key[1], key[2]
                        v1, v2 = data[uid, f1], data[uid, f2]
                        if np.isnan(v1) or np.isnan(v2) or abs(v2) < 1e-10: continue
                        ratio = v1 / (v2 + 1e-10)
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], ratio, side='right'),
                                        0, mo.n_bins-1))
                        if mo.bin_counts[bi] >= MIN_SUPPORT:
                            shift = mo.p_class[bi] - mo.prior
                            conf = min(1.0, mo.bin_counts[bi] / CONF_SUPPORT)
                            int_bonus += shift * conf * 0.3
                            n_int += 1
            class_scores[cls] = base + int_bonus

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


# ─── W5-06: Log-odds Bayesian combination (replace additive AF) ───

def exp_log_odds(data, labels, is_bin, seed):
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

            log_odds = 0.0
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
                        p = np.clip(mo.p_class[bi], 0.01, 0.99)
                        prior = np.clip(mo.prior, 0.01, 0.99)
                        lr = np.log(p / (1 - p)) - np.log(prior / (1 - prior))
                        fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                        conf = min(1.0, bc / CONF_SUPPORT)
                        log_odds += lr * fw * conf
                        n_used += 1

            prior_cls = 0.1
            combined_logit = np.log(prior_cls / (1 - prior_cls)) + log_odds
            prob = 1.0 / (1.0 + np.exp(-np.clip(combined_logit, -20, 20)))
            class_scores[cls] = prob / max(1 - prob, 1e-6)

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


# ─── W5-07: Consensus gating — require ratio AND net to agree ───

def exp_consensus(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        ratio_scores, net_scores = {}, {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            ratio_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            net_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'net')

        h_score = ratio_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls in all_cls:
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT: t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            ratio_pass = ratio_scores[cls] >= t
            net_pass = net_scores[cls] > 0.5
            if ratio_pass and net_pass:
                candidates[cls] = (ratio_scores[cls] - t) / max(t, 0.1)
            elif ratio_pass and not net_pass:
                if ratio_scores[cls] > t * 1.5:
                    candidates[cls] = (ratio_scores[cls] - t) / max(t, 0.1) * 0.5

        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, ratio_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W5-08: Confusion-aware reweighting — upweight rare misclassified classes ───

def exp_confusion_aware(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        n_train = td.shape[0]
        class_acc = {}
        for cls in all_cls:
            correct, total = 0, 0
            for uid in range(n_train):
                if tl[uid] == cls:
                    pred, _ = predict_cfg(uid, td, nodes, cm, cr, all_cls,
                                         against_scale=0.8)
                    total += 1
                    if pred == cls:
                        correct += 1
            class_acc[cls] = correct / max(total, 1)
        boost = {}
        for cls in all_cls:
            if cls == HEALTHY:
                boost[cls] = 1.0
            else:
                acc = class_acc[cls]
                boost[cls] = 1.0 + max(0, (0.8 - acc)) * 2.0
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


# ─── W5-09: Dynamic against — scale against by confidence breadth ───

def exp_dynamic_against(data, labels, is_bin, seed):
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
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W5-10: Age-stratified tree — branch on age (feat 2) after sex ───

def exp_age_tree(data, labels, is_bin, seed):
    def build_age_tree(data, labels, is_bin):
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
            sex_nd = Node(f"L2_sex{int(sex_val)}_{ctr[0]}", 2, idx,
                         _hdist(labels, idx))
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
                if hi == np.inf:
                    sub_mask = age_col > lo
                else:
                    sub_mask = age_col <= hi
                sub_idx = idx[sub_mask & ~np.isnan(age_col)]
                if len(sub_idx) < U_MIN: continue
                ctr[0] += 1
                age_nd = Node(f"L3_{label}_{ctr[0]}", 3, sub_idx,
                             _hdist(labels, sub_idx))
                age_nd.bfeat = AGE_FEAT
                age_nd.blo, age_nd.bhi = lo, hi
                age_nd.parent = sex_nd
                age_nd.ancestor_feats = frozenset({SEX_FEAT, AGE_FEAT})
                sex_nd.children.append(age_nd)
                all_nodes.append(age_nd)
        return all_nodes

    def train(nodes_unused, td, tl, is_bin, all_cls):
        nodes = build_age_tree(td, tl, is_bin)
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        return (nodes, cm, cr)

    def predict(uid, data, nodes_unused, all_cls, tr):
        nodes, cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W5-11: Sigmoid-calibrated scores ───

def exp_calibrated(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        n_train = td.shape[0]
        calib_params = {}
        for cls in all_cls:
            scores_pos, scores_neg = [], []
            for uid in range(n_train):
                af = compute_af_cfg(uid, td, nodes, cm[cls], cr[cls], against_scale=0.8)
                s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
                if tl[uid] == cls:
                    scores_pos.append(s)
                else:
                    scores_neg.append(s)
            if not scores_pos or not scores_neg:
                calib_params[cls] = (1.0, 0.0)
                continue
            mean_pos = np.mean(scores_pos)
            mean_neg = np.mean(scores_neg)
            if abs(mean_pos - mean_neg) < 1e-6:
                calib_params[cls] = (1.0, 0.0)
                continue
            midpoint = (mean_pos + mean_neg) / 2
            scale = 2.0 / max(abs(mean_pos - mean_neg), 0.01)
            calib_params[cls] = (scale, midpoint)
        return (cm, cr, calib_params)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, calib_params = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            raw = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            scale, mid = calib_params.get(cls, (1.0, 0.0))
            calibrated = 1.0 / (1.0 + np.exp(-np.clip(scale * (raw - mid), -20, 20)))
            class_scores[cls] = calibrated * 10

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


# ─── W5-12: Geometric mean scoring — sqrt(AF_for * breadth) ───

def exp_geometric_score(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            af_for, af_ag, n_used, n_for, n_ag, max_fc = af
            ratio = (af_for + RATIO_EPS) / (af_ag + RATIO_EPS)
            breadth = n_for / max(n_for + n_ag, 1)
            class_scores[cls] = np.sqrt(ratio * (1 + breadth * 5))

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


experiments = [
    ("W5-01 Learned thresholds",       exp_learned_thresholds),
    ("W5-02 Multi-resolution bins",    exp_multi_resolution),
    ("W5-03 Second opinion",           exp_second_opinion),
    ("W5-04 KNN backup",              exp_knn_backup),
    ("W5-05 Feature interactions",     exp_feature_interactions),
    ("W5-06 Log-odds Bayesian",        exp_log_odds),
    ("W5-07 Consensus gating",         exp_consensus),
    ("W5-08 Confusion-aware boost",    exp_confusion_aware),
    ("W5-09 Dynamic against scale",    exp_dynamic_against),
    ("W5-10 Age-stratified tree",      exp_age_tree),
    ("W5-11 Sigmoid calibration",      exp_calibrated),
    ("W5-12 Geometric mean score",     exp_geometric_score),
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
