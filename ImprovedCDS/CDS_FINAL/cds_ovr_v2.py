"""CDS-OVR Arrhythmia Classifier — V2 (improved from W11-01).

Improvements over W11-01:
  - Adaptive MAX_BINS: scales bin count with per-class training size.
  - SMOTE oversampling for rare classes to improve posterior quality.
  - Concordance bonus: boosts disease scores when many features agree.
"""
import numpy as np
from collections import defaultdict
from pathlib import Path

# ─── Constants ───

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")

U_MIN = 200
HEALTHY = 1
N_FEAT = 279
SEX_FEAT = 1
LAPLACE_ALPHA = 1.0
RATIO_EPS = 0.1
CORR_THRESHOLD = 0.8
FEATURES_PER_CLASS = 18
FPC_MAP = {}
MAX_BINS = 6
HEALTHY_WEIGHT = 1.05
SUSPICION_HCUT = 2.0
SUSPICION_OFFSET = 0.3
HEALTHY_BAR_CAP = 5.0
REMOVE_CLASSES = {7, 8, 11, 12, 13}
RARE_CLASSES = {4, 5, 9}

CLASS_THRESHOLDS = {2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.0}

MIN_SUPPORT_MAP = {cls: (2 if cls in RARE_CLASSES else 3) for cls in range(1, 14)}
CONF_SUPPORT_MAP = {cls: (5 if cls in RARE_CLASSES else 10) for cls in range(1, 14)}
AGAINST_SCALE_MAP = {cls: (0.5 if cls in RARE_CLASSES else 0.8) for cls in range(1, 14)}

# V2: Interaction parameters (disabled)
INTERACTION_CLASSES = set()
INTERACTION_PAIRS = 3
INTERACTION_WEIGHT = 0.35

# V2: SMOTE parameters
SMOTE_TARGET = 0
SMOTE_K = 5

# V2: Concordance bonus parameters
CONCORDANCE_RATE = 0.0
CONCORDANCE_FLOOR = 5


# ─── Data loading ───

def load_data(path=None):
    path = path or DATA_PATH
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float('nan') if v.strip() == '?' else float(v.strip())
                         for v in line.split(",")])
    raw = np.array(rows, dtype=np.float64)
    data, labels = raw[:, :N_FEAT], raw[:, N_FEAT].astype(int)
    remap = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 14:11, 15:12, 16:13}
    labels = np.array([remap.get(l, l) for l in labels], dtype=int)
    keep = np.array([l not in REMOVE_CLASSES for l in labels])
    return data[keep], labels[keep]


def classify_features(data):
    is_bin = np.zeros(N_FEAT, dtype=bool)
    for c in range(N_FEAT):
        v = data[:, c]
        v = v[~np.isnan(v)]
        if len(v) > 0 and set(np.unique(v)).issubset({0.0, 1.0}):
            is_bin[c] = True
    return is_bin


# ─── Tree structure ───

def _hdist(labels, idx):
    d = defaultdict(int)
    for u in idx:
        d[labels[u]] += 1
    return dict(d)


class Node:
    __slots__ = ('nid', 'lvl', 'uidx', 'hdist', 'bfeat', 'bbin',
                 'bvset', 'blo', 'bhi', 'children', 'parent', 'ancestor_feats')
    def __init__(self, nid, lvl, uidx, hdist):
        self.nid, self.lvl, self.uidx, self.hdist = nid, lvl, uidx, hdist
        self.bfeat = self.bvset = self.blo = self.bhi = None
        self.bbin = False
        self.children, self.parent = [], None
        self.ancestor_feats = frozenset()

    @property
    def nu(self):
        return len(self.uidx)

    def branch_match(self, row):
        if self.bfeat is None:
            return True
        v = row[self.bfeat]
        if np.isnan(v):
            return False
        if self.bbin:
            return v in self.bvset
        if self.bhi == np.inf:
            return v > self.blo
        return v <= self.bhi


class BinModel:
    __slots__ = ('n_bins', 'edges', 'bin_counts', 'target_counts',
                 'p_class', 'prior', 'cls_conf_support')
    def __init__(self, n_bins, edges, bin_counts, target_counts,
                 p_class, prior, cls_conf_support):
        self.n_bins, self.edges = n_bins, edges
        self.bin_counts, self.target_counts = bin_counts, target_counts
        self.p_class, self.prior, self.cls_conf_support = p_class, prior, cls_conf_support


class InteractionModel:
    __slots__ = ('f1', 'f2', 'grid_counts', 'grid_target', 'grid_p_class',
                 'med1', 'med2', 'prior')
    def __init__(self, f1, f2, grid_counts, grid_target, grid_p_class,
                 med1, med2, prior):
        self.f1, self.f2 = f1, f2
        self.grid_counts, self.grid_target = grid_counts, grid_target
        self.grid_p_class = grid_p_class
        self.med1, self.med2 = med1, med2
        self.prior = prior


def build_tree(data, labels, is_bin):
    n = data.shape[0]
    root = Node("root", 1, np.arange(n), _hdist(labels, np.arange(n)))
    all_nodes = [root]
    current_level = [root]
    ctr = [0]
    while current_level:
        lvl = current_level[0].lvl
        if lvl > 1:
            break
        children = []
        for parent in current_level:
            feats = [SEX_FEAT]
            for f in feats:
                col = data[parent.uidx, f]
                vm = ~np.isnan(col)
                vv = col[vm]
                if len(vv) == 0:
                    continue
                if is_bin[f]:
                    uq = sorted(set(vv))
                    if len(uq) < 2:
                        continue
                    parts = [(parent.uidx[vm & (col == val)], frozenset({val}), None, None, True)
                             for val in uq]
                else:
                    if float(vv.min()) == float(vv.max()):
                        continue
                    med = float(np.median(vv))
                    parts = [(parent.uidx[vm & (col <= med)], None, -np.inf, med, False),
                             (parent.uidx[vm & (col > med)], None, med, np.inf, False)]
                for cu, vs, lo, hi, ib in parts:
                    if len(cu) < U_MIN:
                        continue
                    ctr[0] += 1
                    ch = Node(f"L{lvl+1}_f{f}_{ctr[0]}", lvl+1, cu, _hdist(labels, cu))
                    ch.bfeat, ch.bvset, ch.blo, ch.bhi, ch.bbin = f, vs, lo, hi, ib
                    ch.parent = parent
                    ch.ancestor_feats = parent.ancestor_feats | frozenset({f})
                    parent.children.append(ch)
                    children.append(ch)
        seen, deduped = {}, []
        for ch in children:
            key = np.sort(ch.uidx).tobytes()
            if key not in seen:
                seen[key] = True
                deduped.append(ch)
            else:
                ch.parent.children.remove(ch)
        all_nodes.extend(deduped)
        current_level = deduped
    return all_nodes


# ─── Utilities ───

def _fast_abs_corr(x, y):
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    num = (dx * dy).sum()
    den2 = (dx * dx).sum() * (dy * dy).sum()
    return abs(num / np.sqrt(den2)) if den2 > 0 else 0.0


def _route_user(uid, data, nodes):
    by_lvl = defaultdict(list)
    for nd in nodes:
        by_lvl[nd.lvl].append(nd)
    result, active = {}, set()
    for lvl in sorted(by_lvl.keys()):
        if lvl == 1:
            result[1] = [nodes[0]]
            active = {nodes[0].nid}
        else:
            matched = [nd for nd in by_lvl[lvl]
                       if nd.parent and nd.parent.nid in active and nd.branch_match(data[uid])]
            if matched:
                result[lvl] = matched
                active = {nd.nid for nd in matched}
            else:
                break
    return result


# ─── Supervised binning ───

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


def _adaptive_max_bins(n_target_train, base_max=MAX_BINS):
    if n_target_train < 5:
        return 2
    elif n_target_train < 8:
        return 3
    elif n_target_train < 12:
        return 4
    return base_max


# ─── SMOTE oversampling ───

def _smote_oversample(data, labels, is_bin, target_class, target_count, k=SMOTE_K, seed=42):
    idx = np.where(labels == target_class)[0]
    n = len(idx)
    if n >= target_count or n < 2:
        return data, labels

    n_synthetic = target_count - n
    rng = np.random.RandomState(seed)

    class_data = data[idx]
    k_actual = min(k, n - 1)

    cont_feats = np.where(~is_bin)[0]
    class_cont = class_data[:, cont_feats].copy()
    for c in range(class_cont.shape[1]):
        col = class_cont[:, c]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            mean_val = np.nanmean(col) if not nan_mask.all() else 0.0
            col[nan_mask] = mean_val

    stds = np.std(class_cont, axis=0)
    stds[stds == 0] = 1.0
    class_norm = class_cont / stds

    n_cont = len(cont_feats)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((class_norm[i] - class_norm[j]) ** 2))
            dists[i, j] = d
            dists[j, i] = d
    np.fill_diagonal(dists, np.inf)
    knn_idx = np.argsort(dists, axis=1)[:, :k_actual]

    synthetics = []
    for _ in range(n_synthetic):
        i = rng.randint(n)
        j = knn_idx[i, rng.randint(k_actual)]
        lam = rng.random()

        synth = data[idx[i]].copy()
        neighbor = data[idx[j]]

        for f in cont_feats:
            if np.isnan(synth[f]) or np.isnan(neighbor[f]):
                continue
            synth[f] = synth[f] + lam * (neighbor[f] - synth[f])

        synthetics.append(synth)

    synth_array = np.array(synthetics)
    synth_labels = np.full(n_synthetic, target_class, dtype=labels.dtype)

    return np.vstack([data, synth_array]), np.concatenate([labels, synth_labels])


def _apply_smote(data, labels, is_bin, target_count=SMOTE_TARGET, k=SMOTE_K, seed=42):
    aug_data, aug_labels = data.copy(), labels.copy()
    for cls in RARE_CLASSES:
        n_cls = (aug_labels == cls).sum()
        if n_cls < target_count and n_cls >= 2:
            aug_data, aug_labels = _smote_oversample(
                aug_data, aug_labels, is_bin, cls, target_count, k, seed)
    return aug_data, aug_labels


# ─── Training ───

def _train_ovr_node(node, data, labels, is_bin, target_class, min_support, conf_support,
                    adaptive_bins=None):
    models, actions = {}, []
    nd_data, nd_labels = data[node.uidx], labels[node.uidx]
    ns = node.nu
    n_target = int((nd_labels == target_class).sum())
    if n_target < 1:
        return models, actions
    prior = n_target / ns
    is_target = (nd_labels == target_class)

    max_bins_for_class = adaptive_bins if adaptive_bins else MAX_BINS

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
        else:
            max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), max_bins_for_class)
            edges = _supervised_bin_edges(vv, is_target[vm].astype(float), max_nb, min_support)
            nb = len(edges) - 1

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]
        bin_counts = np.bincount(ba, minlength=nb)
        target_counts = np.bincount(ba[lv == target_class], minlength=nb).astype(float)
        p_class = (target_counts + LAPLACE_ALPHA * prior) / (bin_counts + LAPLACE_ALPHA)

        models[(node.nid, f)] = BinModel(nb, edges, bin_counts, target_counts,
                                         p_class, prior, conf_support)

        score = 0.0
        for b in range(nb):
            if bin_counts[b] >= min_support:
                shift = abs(p_class[b] - prior)
                confidence = min(1.0, float(bin_counts[b]) / conf_support)
                score += shift * confidence

        target_vals = vv[lv == target_class]
        rest_vals = vv[lv != target_class]
        if len(target_vals) >= 2 and len(rest_vals) >= 2:
            mean_diff2 = (target_vals.mean() - rest_vals.mean()) ** 2
            var_sum = target_vals.var() + rest_vals.var()
            fisher = mean_diff2 / (var_sum + 1e-10)
        else:
            fisher = 0.0

        if score > 0.001:
            actions.append((f, node.nid, score, fisher))

    return models, actions


def _refine_ovr_node(node, models, node_actions, data, fpc=FEATURES_PER_CLASS):
    scored = [(a, a[2] * (1.0 + 0.5 * np.sqrt(a[3]))) for a in node_actions]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []

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
            if max_corr > CORR_THRESHOLD:
                continue
        kept.append(a)
        kept_features.add(f)
        if len(kept) >= fpc:
            break
    return kept


def _train_class_interactions(data, labels, idx, target_class, retained, is_bin, min_support):
    """Train interaction models for a specific class from its top features."""
    feat_fisher = {}
    for a in retained:
        f, nid, score, fisher = a
        if f not in feat_fisher or fisher > feat_fisher[f]:
            feat_fisher[f] = fisher

    top_feats = sorted(feat_fisher.keys(), key=lambda f: feat_fisher[f], reverse=True)[:8]
    top_feats = [f for f in top_feats if not is_bin[f]]

    if len(top_feats) < 2:
        return []

    is_target = (labels[idx] == target_class)
    n_target = is_target.sum()
    if n_target < 3:
        return []
    prior = n_target / len(idx)

    candidates = []
    for i, f1 in enumerate(top_feats):
        col1 = data[idx, f1]
        for f2 in top_feats[i+1:]:
            col2 = data[idx, f2]
            valid = ~(np.isnan(col1) | np.isnan(col2))
            nv = valid.sum()
            if nv < 20:
                continue

            v1, v2 = col1[valid], col2[valid]
            lv = labels[idx[valid]]
            it = (lv == target_class)

            med1 = float(np.median(v1))
            med2 = float(np.median(v2))

            grid_counts = np.zeros(4)
            grid_target = np.zeros(4)
            masks = [
                (v1 <= med1) & (v2 <= med2),
                (v1 <= med1) & (v2 > med2),
                (v1 > med1) & (v2 <= med2),
                (v1 > med1) & (v2 > med2),
            ]
            for qi, m in enumerate(masks):
                grid_counts[qi] = m.sum()
                grid_target[qi] = it[m].sum()

            if grid_counts.min() < min_support:
                continue

            grid_p = (grid_target + LAPLACE_ALPHA * prior) / (grid_counts + LAPLACE_ALPHA)

            # Only keep if there's a strong concentration in one quadrant
            max_shift = max(abs(grid_p[qi] - prior) for qi in range(4))
            if max_shift < 0.1:
                continue

            # Check for non-additive pattern (interaction effect)
            # Compare diagonal concentrations
            diag_diff = abs((grid_p[0] + grid_p[3])/2 - (grid_p[1] + grid_p[2])/2)
            if diag_diff < 0.03:
                continue

            score = sum(abs(grid_p[qi] - prior) * min(1.0, grid_counts[qi]/10.0)
                        for qi in range(4))

            model = InteractionModel(f1, f2, grid_counts, grid_target, grid_p,
                                     med1, med2, prior)
            candidates.append((score, model))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in candidates[:INTERACTION_PAIRS]]


def train(nodes, data, labels, is_bin, all_cls, real_counts=None):
    """Train OVR models for all classes with per-class parameters."""
    class_models, class_retained = {}, {}
    class_interactions = {}

    for cls in all_cls:
        ms = MIN_SUPPORT_MAP.get(cls, 3)
        cs = CONF_SUPPORT_MAP.get(cls, 10)

        if real_counts:
            n_real = real_counts.get(cls, int((labels[nodes[0].uidx] == cls).sum()))
        else:
            n_real = int((labels[nodes[0].uidx] == cls).sum())
        adaptive_bins = _adaptive_max_bins(n_real)

        cls_models = {}
        cls_actions = defaultdict(list)
        for nd in nodes:
            nm, na = _train_ovr_node(nd, data, labels, is_bin, cls, ms, cs, adaptive_bins)
            cls_models.update(nm)
            for a in na:
                cls_actions[a[1]].append(a)

        fpc = FPC_MAP.get(cls, FEATURES_PER_CLASS)
        cls_ret = []
        for nd in nodes:
            cls_ret.extend(_refine_ovr_node(
                nd, cls_models, cls_actions.get(nd.nid, []), data, fpc))

        # V2: Interaction models only for specified classes
        cls_interact = []
        if cls in INTERACTION_CLASSES and cls_ret:
            cls_interact = _train_class_interactions(
                data, labels, nodes[0].uidx, cls, cls_ret, is_bin, ms)

        class_models[cls] = cls_models
        class_retained[cls] = cls_ret
        class_interactions[cls] = cls_interact

    return class_models, class_retained, class_interactions


# ─── Prediction ───

def _compute_af(uid, data, nodes, models, retained, against_scale, interactions=None):
    lvl_nodes = _route_user(uid, data, nodes)
    af_for, af_against = 0.0, 0.0
    n_for, n_against, n_used = 0, 0, 0
    max_for_contrib = 0.0

    fisher_map = {}
    for a in retained:
        fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
    max_fisher = max(fisher_map.values()) if fisher_map else 1.0

    pred_min_bc = 3

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
                if bc < pred_min_bc:
                    continue
                p_c = mo.p_class[bin_idx]
                shift = p_c - mo.prior
                confidence = min(1.0, bc / 10)
                fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                weighted = abs(shift) * confidence * fw
                if shift >= 0:
                    af_for += weighted
                    n_for += 1
                    if weighted > max_for_contrib:
                        max_for_contrib = weighted
                else:
                    af_against += weighted * against_scale
                    n_against += 1
                n_used += 1

    # V2: Interaction evidence (class 10 only)
    if interactions:
        for imodel in interactions:
            v1 = data[uid, imodel.f1]
            v2 = data[uid, imodel.f2]
            if np.isnan(v1) or np.isnan(v2):
                continue

            b1 = 0 if v1 <= imodel.med1 else 1
            b2 = 0 if v2 <= imodel.med2 else 1
            qi = b1 * 2 + b2

            bc = imodel.grid_counts[qi]
            if bc < 5:
                continue

            p_c = imodel.grid_p_class[qi]
            shift = p_c - imodel.prior
            confidence = min(1.0, bc / 15.0)
            weighted = abs(shift) * confidence * INTERACTION_WEIGHT

            if shift >= 0:
                af_for += weighted
                n_for += 1
            else:
                af_against += weighted * against_scale
                n_against += 1
            n_used += 1

    return af_for, af_against, n_used, n_for, n_against, max_for_contrib


def predict(uid, data, nodes, all_cls, train_result):
    """Predict class for a single patient."""
    class_models, class_retained, class_interactions = train_result
    class_scores = {}

    for cls in all_cls:
        ag = AGAINST_SCALE_MAP.get(cls, 0.8)
        af = _compute_af(uid, data, nodes, class_models[cls], class_retained[cls], ag,
                         class_interactions.get(cls, []))
        af_for = af[0]
        n_for = af[3]
        if cls != HEALTHY and n_for > CONCORDANCE_FLOOR:
            af_for *= 1.0 + CONCORDANCE_RATE * (n_for - CONCORDANCE_FLOOR)
        class_scores[cls] = (af_for + RATIO_EPS) / (af[1] + RATIO_EPS)

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


# ─── Evaluation ───

def run_10fold(data, labels, is_bin, seed=13):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 10)
    all_cls = sorted(set(labels))
    results = []
    for fi in range(10):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]
        real_counts = {cls: int((tl == cls).sum()) for cls in all_cls}
        td, tl = _apply_smote(td, tl, is_bin, seed=seed + fi)
        nodes = build_tree(td, tl, is_bin)
        train_result = train(nodes, td, tl, is_bin, all_cls, real_counts=real_counts)
        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict(uid, data, nodes, all_cls, train_result)
            results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def run_split(data, labels, is_bin, seed=13, train_frac=0.9):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]
    td, tl = data[train_idx], labels[train_idx]
    all_cls = sorted(set(labels))
    real_counts = {cls: int((tl == cls).sum()) for cls in all_cls}
    td, tl = _apply_smote(td, tl, is_bin, seed=seed)
    nodes = build_tree(td, tl, is_bin)
    train_result = train(nodes, td, tl, is_bin, all_cls, real_counts=real_counts)
    results = []
    for uid in test_idx:
        true_cls = int(labels[uid])
        pred, scores = predict(uid, data, nodes, all_cls, train_result)
        results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def run_loocv(data, labels, is_bin):
    n = data.shape[0]
    all_cls = sorted(set(labels))
    results = []
    for i in range(n):
        train_idx = np.concatenate([np.arange(0, i), np.arange(i+1, n)])
        td, tl = data[train_idx], labels[train_idx]
        real_counts = {cls: int((tl == cls).sum()) for cls in all_cls}
        td, tl = _apply_smote(td, tl, is_bin, seed=i)
        nodes = build_tree(td, tl, is_bin)
        train_result = train(nodes, td, tl, is_bin, all_cls, real_counts=real_counts)
        true_cls = int(labels[i])
        pred, scores = predict(i, data, nodes, all_cls, train_result)
        results.append((i, true_cls, int(pred), pred == true_cls))
        if (i + 1) % 50 == 0:
            acc = sum(r[3] for r in results) / len(results)
            print(f"  LOOCV progress: {i+1}/{n} ({100*acc:.1f}% so far)", flush=True)
    return results


def stats(results):
    n = len(results)
    correct = sum(r[3] for r in results)
    th = [r for r in results if r[1] == HEALTHY]
    td = [r for r in results if r[1] != HEALTHY]
    spec = sum(r[3] for r in th) / len(th) if th else 0
    sens = sum(1 for r in td if r[2] != HEALTHY) / len(td) if td else 0
    ba_cls = {}
    for cls in sorted(set(r[1] for r in results)):
        cr = [r for r in results if r[1] == cls]
        ba_cls[cls] = sum(r[3] for r in cr) / len(cr) if cr else 0
    ba = np.mean(list(ba_cls.values()))
    return correct / n, spec, sens, ba


def binary_acc(results):
    correct = sum(1 for _, t, p, _ in results
                  if (t == HEALTHY) == (p == HEALTHY))
    return correct / len(results)


# ─── Main ───

if __name__ == "__main__":
    import time, sys

    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data()
    is_bin = classify_features(data)
    print(f"{data.shape[0]} patients x {data.shape[1]} features")
    print(f"Classes: {sorted(set(labels))}")
    print(f"Class distribution: { {int(c): int((labels==c).sum()) for c in sorted(set(labels))} }\n")

    seeds = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]
    run_seeds = seeds if "--all-seeds" in sys.argv else [int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 13]

    interactions_str = f", interactions for classes {INTERACTION_CLASSES}" if INTERACTION_CLASSES else ""
    print(f"V2: adaptive bins, {FEATURES_PER_CLASS} feat/class{interactions_str}")
    print(f"Thresholds: {CLASS_THRESHOLDS}\n")

    for seed in run_seeds:
        print(f"=== Seed {seed} ===")

        # 10-fold CV
        t0 = time.time()
        results = run_10fold(data, labels, is_bin, seed)
        elapsed = time.time() - t0
        acc, spec, sens, ba = stats(results)
        print(f"  10-fold:  {100*acc:.1f}% multi | {100*binary_acc(results):.1f}% binary | spec={100*spec:.1f}% sens={100*sens:.1f}% | {elapsed:.0f}s")

        # 90/10 split
        t0 = time.time()
        results_90 = run_split(data, labels, is_bin, seed, 0.9)
        elapsed = time.time() - t0
        acc, spec, sens, ba = stats(results_90)
        print(f"  90/10:    {100*acc:.1f}% multi | {100*binary_acc(results_90):.1f}% binary | spec={100*spec:.1f}% sens={100*sens:.1f}% | {elapsed:.0f}s")

        # 60/40 split
        t0 = time.time()
        results_60 = run_split(data, labels, is_bin, seed, 0.6)
        elapsed = time.time() - t0
        acc, spec, sens, ba = stats(results_60)
        print(f"  60/40:    {100*acc:.1f}% multi | {100*binary_acc(results_60):.1f}% binary | spec={100*spec:.1f}% sens={100*sens:.1f}% | {elapsed:.0f}s")
        print()

    if "--loocv" in sys.argv:
        print("Running LOOCV...", flush=True)
        t0 = time.time()
        results_loo = run_loocv(data, labels, is_bin)
        elapsed = time.time() - t0
        acc, spec, sens, ba = stats(results_loo)
        print(f"  LOOCV:    {100*acc:.1f}% multi | {100*binary_acc(results_loo):.1f}% binary")
        print(f"            spec={100*spec:.1f}% sens={100*sens:.1f}% bal={100*ba:.1f}% | {elapsed:.0f}s")
