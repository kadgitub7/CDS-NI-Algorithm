"""Wave 10 — Push past 86.57% 10-fold CV. Build on W9-05 as base.

W9-05 base: MIN_SUPPORT=2/CONF_SUPPORT=5 for rare + against_scale=0.5 for rare.
Best single seed: 86.3%. Need: 86.57%.

Strategy: target the specific error patterns still remaining.
Key errors: healthy<->class2 confusion, class10->healthy FNs, rare class FNs.

New ideas:
1. Per-class against_scale optimization (tune for each class)
2. Class-specific threshold optimization via training accuracy
3. Laplace alpha tuning for rare classes (less smoothing = sharper posteriors)
4. Ensemble of 2 tree structures (sex + age branching)
5. Feature count increase for common confused classes (class 2, 10)
6. Cross-validation threshold tuning (set thresholds based on training CV)
7. Confidence-weighted voting (high-confidence features get more say)
8. Two-stage classification (binary healthy/disease first, then multiclass)
9. W9-05 + relaxed correlation (combine W9-05 with W8-09's corr trick)
10. Per-class feature count tuning
11. SMOTE-inspired: duplicate rare class training samples
12. Pairwise disambiguation for top-2 confused classes
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
    N_FEAT, LAPLACE_ALPHA, U_MIN, CORR_THRESHOLD, MAX_BINS,
    _supervised_bin_edges, _fast_abs_corr,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")
RARE_CLASSES = {4, 5, 9}


def _train_w905_base(nodes, td, tl, is_bin, all_cls):
    """W9-05 base training: lower minsup for rare classes."""
    import experiment_runner as mod
    class_models, class_retained = {}, {}
    orig_ms, orig_cs = mod.MIN_SUPPORT, mod.CONF_SUPPORT

    for cls in all_cls:
        mod.MIN_SUPPORT = 2 if cls in RARE_CLASSES else 3
        mod.CONF_SUPPORT = 5 if cls in RARE_CLASSES else 10

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


def _predict_w905_base(uid, data, nodes, all_cls, tr):
    """W9-05 base prediction: lower against_scale for rare."""
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


# ─── W10-01: Per-class against_scale optimization ───

def exp_perclass_against(data, labels, is_bin, seed):
    AGAINST_MAP = {1: 0.8, 2: 0.9, 3: 0.8, 4: 0.4, 5: 0.4, 6: 0.7, 9: 0.4, 10: 0.7}

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            ag = AGAINST_MAP.get(cls, 0.8)
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

    return run_10fold(data, labels, is_bin, seed, predict, _train_w905_base)


# ─── W10-02: Training CV threshold optimization ───

def exp_cv_thresholds(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = _train_w905_base(nodes, td, tl, is_bin, all_cls)
        n_train = td.shape[0]
        class_score_dist = {cls: [] for cls in all_cls if cls != HEALTHY}
        for uid in range(n_train):
            true_cls = int(tl[uid])
            for cls in all_cls:
                if cls == HEALTHY:
                    continue
                ag = 0.5 if cls in RARE_CLASSES else 0.8
                af = compute_af_cfg(uid, td, nodes, cm[cls], cr[cls], against_scale=ag)
                s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
                class_score_dist[cls].append((s, true_cls == cls))

        opt_thresh = dict(CLASS_THRESHOLDS)
        for cls in all_cls:
            if cls == HEALTHY:
                continue
            scores = class_score_dist[cls]
            if not scores:
                continue
            best_f1, best_t = 0, opt_thresh.get(cls, 3.0)
            for t_cand in np.arange(1.0, 8.0, 0.25):
                tp = sum(1 for s, is_t in scores if is_t and s >= t_cand)
                fp = sum(1 for s, is_t in scores if not is_t and s >= t_cand)
                fn = sum(1 for s, is_t in scores if is_t and s < t_cand)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t_cand
            opt_thresh[cls] = best_t
        return cm, cr, opt_thresh

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, opt_thresh = tr
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
            t = opt_thresh.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t:
                continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W10-03: Lower Laplace alpha for rare classes ───

def exp_low_laplace_rare(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        import experiment_runner as mod
        class_models, class_retained = {}, {}
        orig_ms, orig_cs, orig_la = mod.MIN_SUPPORT, mod.CONF_SUPPORT, mod.LAPLACE_ALPHA

        for cls in all_cls:
            mod.MIN_SUPPORT = 2 if cls in RARE_CLASSES else 3
            mod.CONF_SUPPORT = 5 if cls in RARE_CLASSES else 10
            mod.LAPLACE_ALPHA = 0.3 if cls in RARE_CLASSES else 1.0

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
        mod.LAPLACE_ALPHA = orig_la
        return class_models, class_retained

    return run_10fold(data, labels, is_bin, seed, _predict_w905_base, train)


# ─── W10-04: Two-stage: binary first, then multiclass ───

def exp_two_stage(data, labels, is_bin, seed):
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            ag = 0.5 if cls in RARE_CLASSES else 0.8
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=ag)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')

        h_score = class_scores.get(HEALTHY, 1.0)

        disease_evidence = sum(
            max(0, class_scores.get(c, 0) - CLASS_THRESHOLDS.get(c, 3.0))
            for c in all_cls if c != HEALTHY
        )
        if disease_evidence < 0.5 and h_score > 3.0:
            return HEALTHY, class_scores

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

    return run_10fold(data, labels, is_bin, seed, predict, _train_w905_base)


# ─── W10-05: More features for confused classes (class 2, 10) ───

def exp_more_feats_confused(data, labels, is_bin, seed):
    CONFUSED_CLASSES = {2, 10}
    FPC_MAP = {}
    for c in range(1, 14):
        if c in CONFUSED_CLASSES:
            FPC_MAP[c] = 24
        else:
            FPC_MAP[c] = 18

    def train(nodes, td, tl, is_bin, all_cls):
        import experiment_runner as mod
        class_models, class_retained = {}, {}
        orig_ms, orig_cs = mod.MIN_SUPPORT, mod.CONF_SUPPORT

        for cls in all_cls:
            mod.MIN_SUPPORT = 2 if cls in RARE_CLASSES else 3
            mod.CONF_SUPPORT = 5 if cls in RARE_CLASSES else 10

            cls_models = {}
            cls_actions = defaultdict(list)
            for nd in nodes:
                nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls, use_supervised_bins=True)
                cls_models.update(nm)
                for a in na:
                    cls_actions[a[1]].append(a)
            fpc = FPC_MAP.get(cls, 18)
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

    return run_10fold(data, labels, is_bin, seed, _predict_w905_base, train)


# ─── W10-06: W9-05 + relaxed correlation for rare ───

def exp_w905_relaxed_corr(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        import experiment_runner as mod
        class_models, class_retained = {}, {}
        orig_ms, orig_cs = mod.MIN_SUPPORT, mod.CONF_SUPPORT

        for cls in all_cls:
            mod.MIN_SUPPORT = 2 if cls in RARE_CLASSES else 3
            mod.CONF_SUPPORT = 5 if cls in RARE_CLASSES else 10
            corr_t = 0.9 if cls in RARE_CLASSES else 0.8

            cls_models = {}
            cls_actions = defaultdict(list)
            for nd in nodes:
                nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls, use_supervised_bins=True)
                cls_models.update(nm)
                for a in na:
                    cls_actions[a[1]].append(a)

            cls_ret = []
            fpc = 18
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
            class_models[cls] = cls_models
            class_retained[cls] = cls_ret

        mod.MIN_SUPPORT = orig_ms
        mod.CONF_SUPPORT = orig_cs
        return class_models, class_retained

    return run_10fold(data, labels, is_bin, seed, _predict_w905_base, train)


# ─── W10-07: Confidence-weighted scoring (high-confidence bins dominate) ───

def exp_confidence_weighted(data, labels, is_bin, seed):
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            ag = 0.5 if cls in RARE_CLASSES else 0.8
            lvl_nodes = _route_user(uid, data, nodes)
            retained = cr[cls]
            models = cm[cls]
            fisher_map = {}
            for a in retained:
                fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
            max_fisher = max(fisher_map.values()) if fisher_map else 1.0

            af_for, af_against = 0.0, 0.0
            n_for, n_against = 0, 0
            max_fc = 0.0
            ms = 2 if cls in RARE_CLASSES else 3
            cs = 5 if cls in RARE_CLASSES else 10

            for lvl in sorted(lvl_nodes.keys()):
                for nd in lvl_nodes[lvl]:
                    for a in retained:
                        if a[1] != nd.nid:
                            continue
                        f = a[0]
                        v = data[uid, f]
                        if np.isnan(v):
                            continue
                        mo = models.get((nd.nid, f))
                        if not mo:
                            continue
                        bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'), 0, mo.n_bins - 1))
                        bc = mo.bin_counts[bi]
                        if bc < ms:
                            continue
                        shift = mo.p_class[bi] - mo.prior
                        confidence = min(1.0, bc / cs)
                        fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                        weighted = abs(shift) * confidence * confidence * fw
                        if shift >= 0:
                            af_for += weighted
                            n_for += 1
                            if weighted > max_fc:
                                max_fc = weighted
                        else:
                            af_against += weighted * ag
                            n_against += 1

            class_scores[cls] = (af_for + RATIO_EPS) / (af_against + RATIO_EPS)

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

    return run_10fold(data, labels, is_bin, seed, predict, _train_w905_base)


# ─── W10-08: SMOTE-lite: oversample rare class in training ───

def exp_smote_rare(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        import experiment_runner as mod
        class_counts = {cls: int((tl == cls).sum()) for cls in all_cls}
        median_count = int(np.median([c for c in class_counts.values() if c > 0]))

        aug_indices = list(range(len(tl)))
        rng = np.random.RandomState(42)
        for cls in all_cls:
            if class_counts[cls] < median_count // 2:
                cls_idx = np.where(tl == cls)[0]
                n_needed = median_count // 2 - class_counts[cls]
                extra = rng.choice(cls_idx, size=n_needed, replace=True)
                aug_indices.extend(extra.tolist())

        aug_data = td[aug_indices]
        aug_labels = tl[aug_indices]
        aug_nodes = build_tree(aug_data, aug_labels, is_bin)

        class_models, class_retained = {}, {}
        orig_ms, orig_cs = mod.MIN_SUPPORT, mod.CONF_SUPPORT

        for cls in all_cls:
            mod.MIN_SUPPORT = 2 if cls in RARE_CLASSES else 3
            mod.CONF_SUPPORT = 5 if cls in RARE_CLASSES else 10

            cls_models = {}
            cls_actions = defaultdict(list)
            for nd in aug_nodes:
                nm, na = train_ovr_node_cfg(nd, aug_data, aug_labels, is_bin, cls, use_supervised_bins=True)
                cls_models.update(nm)
                for a in na:
                    cls_actions[a[1]].append(a)
            cls_ret = []
            for nd in aug_nodes:
                cls_ret.extend(refine_ovr_node_cfg(
                    nd, cls_models, cls_actions.get(nd.nid, []), aug_data,
                    rank_by='score', fpc=18))
            class_models[cls] = cls_models
            class_retained[cls] = cls_ret

        mod.MIN_SUPPORT = orig_ms
        mod.CONF_SUPPORT = orig_cs
        return class_models, class_retained

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


# ─── W10-09: Adjusted healthy weight/bar ───

def exp_adjusted_healthy(data, labels, is_bin, seed):
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            ag = 0.5 if cls in RARE_CLASSES else 0.8
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=ag)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(1.0 * h_score, 4.5)
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

    return run_10fold(data, labels, is_bin, seed, predict, _train_w905_base)


# ─── W10-10: Per-class feature count tuning ───

def exp_perclass_fpc(data, labels, is_bin, seed):
    FPC = {1: 18, 2: 22, 3: 18, 4: 14, 5: 14, 6: 18, 9: 14, 10: 22}

    def train(nodes, td, tl, is_bin, all_cls):
        import experiment_runner as mod
        class_models, class_retained = {}, {}
        orig_ms, orig_cs = mod.MIN_SUPPORT, mod.CONF_SUPPORT

        for cls in all_cls:
            mod.MIN_SUPPORT = 2 if cls in RARE_CLASSES else 3
            mod.CONF_SUPPORT = 5 if cls in RARE_CLASSES else 10

            cls_models = {}
            cls_actions = defaultdict(list)
            for nd in nodes:
                nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls, use_supervised_bins=True)
                cls_models.update(nm)
                for a in na:
                    cls_actions[a[1]].append(a)
            fpc = FPC.get(cls, 18)
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

    return run_10fold(data, labels, is_bin, seed, _predict_w905_base, train)


# ─── W10-11: Lower thresholds for class 10 (most missed disease) ───

def exp_lower_thresh_c10(data, labels, is_bin, seed):
    ADJ_THRESH = dict(CLASS_THRESHOLDS)
    ADJ_THRESH[10] = 2.5
    ADJ_THRESH[2] = 4.0

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
            t = ADJ_THRESH.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t:
                continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return run_10fold(data, labels, is_bin, seed, predict, _train_w905_base)


# ─── W10-12: All rare-class tricks combined (minsup+corr+against+laplace) ───

def exp_all_rare_tricks(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        import experiment_runner as mod
        class_models, class_retained = {}, {}
        orig_ms, orig_cs, orig_la = mod.MIN_SUPPORT, mod.CONF_SUPPORT, mod.LAPLACE_ALPHA

        for cls in all_cls:
            mod.MIN_SUPPORT = 2 if cls in RARE_CLASSES else 3
            mod.CONF_SUPPORT = 5 if cls in RARE_CLASSES else 10
            mod.LAPLACE_ALPHA = 0.5 if cls in RARE_CLASSES else 1.0
            corr_t = 0.9 if cls in RARE_CLASSES else 0.8

            cls_models = {}
            cls_actions = defaultdict(list)
            for nd in nodes:
                nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls, use_supervised_bins=True)
                cls_models.update(nm)
                for a in na:
                    cls_actions[a[1]].append(a)

            fpc = 18
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
            class_models[cls] = cls_models
            class_retained[cls] = cls_ret

        mod.MIN_SUPPORT = orig_ms
        mod.CONF_SUPPORT = orig_cs
        mod.LAPLACE_ALPHA = orig_la
        return class_models, class_retained

    return run_10fold(data, labels, is_bin, seed, _predict_w905_base, train)


experiments = [
    ("W10-01 Per-class against",       exp_perclass_against),
    ("W10-02 CV thresh optimize",      exp_cv_thresholds),
    ("W10-03 Low Laplace rare",        exp_low_laplace_rare),
    ("W10-04 Two-stage classify",      exp_two_stage),
    ("W10-05 More feats confused",     exp_more_feats_confused),
    ("W10-06 W905+relaxed corr",       exp_w905_relaxed_corr),
    ("W10-07 Confidence^2 weight",     exp_confidence_weighted),
    ("W10-08 SMOTE rare",             exp_smote_rare),
    ("W10-09 Adjusted healthy bar",    exp_adjusted_healthy),
    ("W10-10 Per-class fpc",           exp_perclass_fpc),
    ("W10-11 Lower thresh c10",        exp_lower_thresh_c10),
    ("W10-12 All rare tricks",         exp_all_rare_tricks),
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
