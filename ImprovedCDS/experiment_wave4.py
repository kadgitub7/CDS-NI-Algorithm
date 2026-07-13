"""Wave 4 — radical structural changes to break the 85.3% ceiling.
Focus: information the current algorithm throws away.
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
    AGAINST_SCALE, N_FEAT, LAPLACE_ALPHA, CORR_THRESHOLD, MAX_BINS,
    _supervised_bin_edges, _fast_abs_corr,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


# ─── W4-01: Fine-tune against_scale (0.7) ───

def exp_against_07(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.7)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W4-02: Fine-tune against_scale (0.9) ───

def exp_against_09(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.9)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W4-03: Fine-tune against_scale (1.0 — equal weight) ───

def exp_against_10(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=1.0)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W4-04: Missing value as informative signal ───

def exp_missing_signal(data, labels, is_bin, seed):
    n_missing = np.isnan(data).sum(axis=1).astype(float)
    miss_per_feat = np.isnan(data).astype(float)

    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
        miss_models = {}
        for cls in all_cls:
            is_target = (tl == cls).astype(float)
            miss_counts = np.isnan(td).sum(axis=1).astype(float)
            if miss_counts.max() == miss_counts.min(): continue
            edges = _supervised_bin_edges(miss_counts, is_target, 4)
            nb = len(edges) - 1
            ba = np.clip(np.searchsorted(edges[1:], miss_counts, side='right'), 0, nb-1)
            n_target = int(is_target.sum())
            prior = n_target / len(tl)
            bin_counts = np.bincount(ba, minlength=nb)
            target_counts = np.bincount(ba[tl==cls], minlength=nb).astype(float)
            p_class = (target_counts + LAPLACE_ALPHA * prior) / (bin_counts + LAPLACE_ALPHA)
            miss_models[cls] = BinModel(nb, edges, bin_counts, target_counts, p_class, prior, CONF_SUPPORT)
        return (cm, cr, miss_models)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, miss_models = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            base_score = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
            miss_bonus = 0.0
            if cls in miss_models:
                mo = miss_models[cls]
                n_miss = float(np.isnan(data[uid]).sum())
                bi = int(np.clip(np.searchsorted(mo.edges[1:], n_miss, side='right'), 0, mo.n_bins-1))
                if mo.bin_counts[bi] >= MIN_SUPPORT:
                    shift = mo.p_class[bi] - mo.prior
                    confidence = min(1.0, mo.bin_counts[bi] / CONF_SUPPORT)
                    miss_bonus = shift * confidence * 0.3
            class_scores[cls] = base_score + miss_bonus

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


# ─── W4-05: Score normalization by number of features used ───

def exp_score_norm(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            af_for, af_ag, n_used, n_for, n_ag, max_fc = af
            n_total = n_for + n_ag
            if n_total > 0:
                norm_for = af_for / max(n_for, 1) * 10
                norm_ag = af_ag / max(n_ag, 1) * 10
                class_scores[cls] = (norm_for + RATIO_EPS) / (norm_ag + RATIO_EPS)
            else:
                class_scores[cls] = 1.0

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


# ─── W4-06: Weighted voting by bin chi-squared quality ───

def _chi2_bin_quality(mo, bin_idx):
    bc = mo.bin_counts[bin_idx]
    tc = mo.target_counts[bin_idx]
    n_total = mo.bin_counts.sum()
    t_total = mo.target_counts.sum()
    if n_total == 0 or t_total == 0: return 0.0
    e_t = bc * t_total / n_total
    e_r = bc * (n_total - t_total) / n_total
    r = bc - tc
    chi2 = 0.0
    if e_t > 0: chi2 += (tc - e_t)**2 / e_t
    if e_r > 0: chi2 += (r - e_r)**2 / e_r
    return chi2

def exp_chi2_weighted(data, labels, is_bin, seed):
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
                        bin_idx = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'), 0, mo.n_bins-1))
                        bc = mo.bin_counts[bin_idx]
                        if bc < MIN_SUPPORT: continue
                        p_c = mo.p_class[bin_idx]
                        shift = p_c - mo.prior
                        confidence = min(1.0, bc / CONF_SUPPORT)
                        fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                        chi2_q = min(_chi2_bin_quality(mo, bin_idx), 20.0)
                        bin_weight = 1.0 + np.log1p(chi2_q) * 0.2
                        weighted = abs(shift) * confidence * fw * bin_weight
                        if shift >= 0:
                            af_for += weighted
                            n_for += 1
                        else:
                            af_against += weighted * 0.8
                            n_against += 1

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


# ─── W4-07: Purity-based scoring — use bin purity instead of shift ───

def exp_purity_score(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            lvl_nodes = _route_user(uid, data, nodes)
            retained = cr[cls]
            models = cm[cls]
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
                        purity = mo.p_class[bi]
                        if purity > mo.prior:
                            af_for += purity * min(1.0, bc / CONF_SUPPORT)
                            n_for += 1
                        elif purity < mo.prior:
                            af_against += (1.0 - purity) * min(1.0, bc / CONF_SUPPORT) * 0.8
                            n_against += 1

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


# ─── W4-08: Rank-based aggregation (replace AF with rank-voting) ───

def exp_rank_vote(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        raw_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
            raw_scores[cls] = (af[0] + RATIO_EPS) / (af[1] + RATIO_EPS)

        ranked = sorted(raw_scores.keys(), key=lambda c: raw_scores[c], reverse=True)
        rank_map = {c: i for i, c in enumerate(ranked)}

        h_rank = rank_map.get(HEALTHY, 0)
        if h_rank == 0:
            return HEALTHY, raw_scores

        top_cls = ranked[0]
        top_score = raw_scores[top_cls]
        h_score = raw_scores[HEALTHY]
        if top_cls != HEALTHY and top_score > CLASS_THRESHOLDS.get(top_cls, 3.0):
            if top_score > h_score * 1.2:
                return top_cls, raw_scores

        return HEALTHY, raw_scores

    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W4-09: Aggressive against (1.2) — very conservative ───

def exp_against_12(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=1.2)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W4-10: Feature selection by Fisher only + SV + against 0.8 ───

def exp_fisher_rank(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, rank_by='fisher')
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W4-11: More features (24 per class) + SV + against 0.8 ───

def exp_24fpc(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, fpc=24)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ─── W4-12: Fewer features (12 per class) + SV + against 0.8 ───

def exp_12fpc(data, labels, is_bin, seed):
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, fpc=12)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


experiments = [
    ("W4-01 against 0.7",             exp_against_07),
    ("W4-02 against 0.9",             exp_against_09),
    ("W4-03 against 1.0",             exp_against_10),
    ("W4-04 Missing value signal",    exp_missing_signal),
    ("W4-05 Score normalization",     exp_score_norm),
    ("W4-06 Chi2 weighted voting",    exp_chi2_weighted),
    ("W4-07 Purity scoring",          exp_purity_score),
    ("W4-08 Rank-based voting",       exp_rank_vote),
    ("W4-09 against 1.2",             exp_against_12),
    ("W4-10 Fisher rank + SV+0.8",    exp_fisher_rank),
    ("W4-11 24 features + SV+0.8",    exp_24fpc),
    ("W4-12 12 features + SV+0.8",    exp_12fpc),
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
