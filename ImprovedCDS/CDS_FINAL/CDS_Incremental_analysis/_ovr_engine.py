"""Configurable OVR engine for CDS incremental analysis.

Each variant file creates a CDSConfig and calls run_variant().
The engine handles training and prediction based on configuration flags.
"""
import numpy as np
from collections import defaultdict
from _common import (
    load_data, classify_features, build_tree, _route_user,
    _fast_abs_corr, _hdist, BinModel, run_all_evaluations,
    HEALTHY, N_FEAT, SEEDS
)


class CDSConfig:
    def __init__(self, **kw):
        # --- Binning ---
        self.binning = 'sturges'        # 'sturges' or 'supervised'
        self.max_bins = 6               # cap for supervised binning

        # --- Feature selection ---
        self.corr_threshold = None      # None=no filter; float=threshold
        self.features_per_class = None  # None=no limit; int=max features

        # --- Smoothing ---
        self.laplace_alpha = 0.0        # 0.0=no Laplace smoothing

        # --- Support parameters ---
        self.min_support = 3
        self.conf_support = 10
        self.against_scale = 0.8

        # --- Per-class support (rare class handling) ---
        self.rare_classes = None        # None=uniform; set like {4,5,9}
        self.rare_min_support = 2
        self.rare_conf_support = 5
        self.rare_against_scale = 0.5

        # --- AF computation ---
        self.af_mode = 'single'         # 'single' or 'dual'
        self.fisher = False

        # --- Final scoring ---
        self.scoring = 'simple'         # 'simple' or 'ratio'
        self.ratio_eps = 0.1

        # --- Thresholding ---
        self.threshold_mode = 'fixed'   # 'fixed' or 'healthy_bar'
        self.fixed_threshold = 0.55
        self.healthy_weight = 1.05
        self.suspicion_hcut = 2.0
        self.suspicion_offset = 0.3
        self.healthy_bar_cap = 5.0
        self.class_thresholds = None    # dict or None

        # --- Data preprocessing ---
        self.remove_classes = None      # set or None

        for k, v in kw.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown config key: {k}")
            setattr(self, k, v)

    def get_min_support(self, cls):
        if self.rare_classes and cls in self.rare_classes:
            return self.rare_min_support
        return self.min_support

    def get_conf_support(self, cls):
        if self.rare_classes and cls in self.rare_classes:
            return self.rare_conf_support
        return self.conf_support

    def get_against_scale(self, cls):
        if self.rare_classes and cls in self.rare_classes:
            return self.rare_against_scale
        return self.against_scale


# ─── Supervised binning (chi-squared) ───

def _supervised_bin_edges(vv, is_target, max_bins, min_support):
    n = len(vv)
    vmin, vmax = float(vv.min()), float(vv.max())
    if vmin == vmax or n < 2 * min_support:
        return np.array([vmin - 0.5, vmax + 0.5])

    sort_idx = np.argsort(vv)
    sv = vv[sort_idx]
    st = is_target[sort_idx]
    edges = [vmin, vmax]

    for _ in range(max_bins - 1):
        best_gain, best_split, best_seg = 0.0, None, None
        for seg_i in range(len(edges) - 1):
            lo, hi = edges[seg_i], edges[seg_i + 1]
            if seg_i == 0:
                mask = sv <= hi
            elif seg_i == len(edges) - 2:
                mask = sv > lo
            else:
                mask = (sv > lo) & (sv <= hi)
            seg_vals, seg_targ = sv[mask], st[mask]
            n_seg = len(seg_vals)
            if n_seg < 2 * min_support:
                continue
            n_t, n_r = seg_targ.sum(), n_seg - seg_targ.sum()
            if n_t == 0 or n_r == 0:
                continue
            candidates = np.where(seg_vals[:-1] != seg_vals[1:])[0]
            if len(candidates) == 0:
                continue
            cum_t = np.cumsum(seg_targ)
            for ci in candidates:
                n_left, n_right = ci + 1, n_seg - ci - 1
                if n_left < min_support or n_right < min_support:
                    continue
                t_left = cum_t[ci]
                t_right = n_t - t_left
                r_left, r_right = n_left - t_left, n_right - t_right
                e_tl = n_left * n_t / n_seg
                e_tr = n_right * n_t / n_seg
                e_rl = n_left * n_r / n_seg
                e_rr = n_right * n_r / n_seg
                chi2 = sum((o - e)**2 / e for o, e in
                           [(t_left, e_tl), (t_right, e_tr),
                            (r_left, e_rl), (r_right, e_rr)] if e > 0)
                if chi2 > best_gain:
                    best_gain = chi2
                    best_split = (seg_vals[ci] + seg_vals[ci + 1]) / 2.0
                    best_seg = seg_i
        if best_split is None or best_gain < 0.5:
            break
        edges.insert(best_seg + 1, best_split)

    edges[0] = vmin - 1e-10
    edges[-1] = vmax + 1e-10
    return np.array(sorted(set(edges)))


# ─── Training ───

def _train_ovr_node(cfg, node, data, labels, is_bin, target_class):
    models, actions = {}, []
    nd_data, nd_labels = data[node.uidx], labels[node.uidx]
    ns = node.nu
    n_target = int((nd_labels == target_class).sum())
    if n_target < 1:
        return models, actions
    prior = n_target / ns
    is_target = (nd_labels == target_class)
    ms = cfg.get_min_support(target_class)
    cs = cfg.get_conf_support(target_class)

    for f in range(N_FEAT):
        col = nd_data[:, f]
        vm = ~np.isnan(col)
        vv = col[vm]
        nv = len(vv)
        if nv == 0:
            continue
        vmin, vmax = float(vv.min()), float(vv.max())

        if is_bin[f]:
            nb = 1 if vmin == vmax else 2
            edges = np.array([vmin - .5, vmin + .5]) if nb == 1 else np.array([-.5, .5, 1.5])
        elif vmin == vmax:
            nb, edges = 1, np.array([vmin - .5, vmin + .5])
        elif cfg.binning == 'supervised':
            max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), cfg.max_bins)
            edges = _supervised_bin_edges(vv, is_target[vm].astype(float), max_nb, ms)
            nb = len(edges) - 1
        else:
            nb = max(2, int(np.ceil(1 + np.log2(nv))))
            edges = np.linspace(vmin, vmax, nb + 1)

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]
        bin_counts = np.bincount(ba, minlength=nb)
        target_counts = np.bincount(ba[lv == target_class], minlength=nb).astype(float)

        if cfg.laplace_alpha > 0:
            p_class = (target_counts + cfg.laplace_alpha * prior) / (bin_counts + cfg.laplace_alpha)
        else:
            p_class = np.divide(target_counts, bin_counts,
                                out=np.full(nb, prior), where=bin_counts > 0)

        models[(node.nid, f)] = BinModel(nb, edges, bin_counts, target_counts,
                                         p_class, prior, cs)

        score = 0.0
        for b in range(nb):
            if bin_counts[b] >= ms:
                shift = abs(p_class[b] - prior)
                confidence = min(1.0, float(bin_counts[b]) / cs)
                score += shift * confidence

        fisher = 0.0
        if cfg.fisher:
            target_vals = vv[lv == target_class]
            rest_vals = vv[lv != target_class]
            if len(target_vals) >= 2 and len(rest_vals) >= 2:
                mean_diff2 = (target_vals.mean() - rest_vals.mean()) ** 2
                var_sum = target_vals.var() + rest_vals.var()
                fisher = mean_diff2 / (var_sum + 1e-10)

        if score > 0.001:
            actions.append((f, node.nid, score, fisher))

    return models, actions


def _refine_ovr_node(cfg, node, models, node_actions, data):
    fpc = cfg.features_per_class
    ct = cfg.corr_threshold

    scored = [(a, a[2]) for a in node_actions]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []

    if ct is not None and fpc is not None:
        top_scored = scored[:3 * fpc]
        nd_data = data[node.uidx]
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
                        correlations[(f1, f2)] = c
                        correlations[(f2, f1)] = c
        kept, kept_features = [], set()
        for a, s in top_scored:
            f = a[0]
            if f in kept_features:
                continue
            raw = data[node.uidx, f]
            if (~np.isnan(raw)).sum() == 0:
                continue
            if kept_features:
                max_corr = max(correlations.get((f, kf), 0.0) for kf in kept_features)
                if max_corr > ct:
                    continue
            kept.append(a)
            kept_features.add(f)
            if len(kept) >= fpc:
                break
        return kept
    elif fpc is not None:
        return [a for a, _ in scored[:fpc]]
    else:
        return [a for a, _ in scored]


def train(cfg, nodes, data, labels, is_bin, all_cls):
    class_models, class_retained = {}, {}
    for cls in all_cls:
        cls_models = {}
        cls_actions = defaultdict(list)
        for nd in nodes:
            nm, na = _train_ovr_node(cfg, nd, data, labels, is_bin, cls)
            cls_models.update(nm)
            for a in na:
                cls_actions[a[1]].append(a)
        cls_ret = []
        for nd in nodes:
            cls_ret.extend(_refine_ovr_node(
                cfg, nd, cls_models, cls_actions.get(nd.nid, []), data))
        class_models[cls] = cls_models
        class_retained[cls] = cls_ret
    return class_models, class_retained


# ─── Prediction ───

def _compute_af(cfg, uid, data, nodes, models, retained, target_class):
    lvl_nodes = _route_user(uid, data, nodes)
    af_for, af_against = 0.0, 0.0
    n_used = 0

    fisher_map = {}
    if cfg.fisher:
        for a in retained:
            fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
    max_fisher = max(fisher_map.values()) if fisher_map else 1.0

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
                bin_idx = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                     0, mo.n_bins - 1))
                bc = mo.bin_counts[bin_idx]
                if bc < 3:
                    continue
                p_c = mo.p_class[bin_idx]
                shift = p_c - mo.prior
                confidence = min(1.0, bc / 10)

                if cfg.fisher and fisher_map:
                    fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                else:
                    fw = 1.0

                weighted = abs(shift) * confidence * fw

                if cfg.af_mode == 'dual':
                    ag = cfg.get_against_scale(target_class)
                    if shift >= 0:
                        af_for += weighted
                    else:
                        af_against += weighted * ag
                else:
                    af_for += weighted

                n_used += 1

    return af_for, af_against, n_used


def predict(cfg, uid, data, nodes, all_cls, train_result):
    class_models, class_retained = train_result
    class_scores = {}

    for cls in all_cls:
        af_for, af_against, n_used = _compute_af(
            cfg, uid, data, nodes, class_models[cls], class_retained[cls], cls)

        if cfg.scoring == 'ratio':
            eps = cfg.ratio_eps
            class_scores[cls] = (af_for + eps) / (af_against + eps)
        else:
            norm = max(n_used, 1)
            if cfg.af_mode == 'dual':
                class_scores[cls] = (af_for - af_against) / norm
            else:
                class_scores[cls] = af_for / norm

    if cfg.threshold_mode == 'healthy_bar':
        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(cfg.healthy_weight * h_score, cfg.healthy_bar_cap)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY:
                continue
            if cfg.class_thresholds:
                t = cfg.class_thresholds.get(cls, 3.0)
            else:
                t = 3.0
            if h_score < cfg.suspicion_hcut:
                t -= cfg.suspicion_offset
            t = max(t, healthy_bar)
            if score < t:
                continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
    else:
        threshold = cfg.fixed_threshold
        if cfg.class_thresholds:
            candidates = {}
            for cls, score in class_scores.items():
                if cls == HEALTHY:
                    continue
                t = cfg.class_thresholds.get(cls, threshold)
                if score > t:
                    candidates[cls] = score
            best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        elif threshold > 0:
            h_score = class_scores.get(HEALTHY, 0.0)
            candidates = {}
            for cls, score in class_scores.items():
                if cls == HEALTHY:
                    continue
                if score > h_score * threshold:
                    candidates[cls] = score
            best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        else:
            best_cls = max(class_scores, key=class_scores.get)

    return best_cls, class_scores


# ─── Runner ───

def run_variant(cfg, label="", seeds=None):
    seeds = seeds or SEEDS
    data, labels = load_data(remove_classes=cfg.remove_classes)
    is_bin = classify_features(data)
    print(f"Loaded {data.shape[0]} patients, classes: {sorted(set(labels))}")

    def _train(nodes, d, l, ib, ac):
        return train(cfg, nodes, d, l, ib, ac)

    def _predict(uid, d, nodes, ac, tr):
        return predict(cfg, uid, d, nodes, ac, tr)

    return run_all_evaluations(data, labels, is_bin, _train, _predict,
                               seeds=seeds, label=label)
