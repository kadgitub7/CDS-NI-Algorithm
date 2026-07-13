"""Parallel experiment runner — tests many structural ideas on 10-fold CV seed=13.
Each experiment is a self-contained function that returns results.
"""
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")

# ─── Base constants ───
U_MIN = 200
HEALTHY = 1
N_FEAT = 279
FORCE_SEX_BRANCHING = True
SEX_FEAT = 1
LAPLACE_ALPHA = 1.0
MIN_SUPPORT = 3
CONF_SUPPORT = 10
RATIO_EPS = 0.1
CORR_THRESHOLD = 0.8
FEATURES_PER_CLASS = 18
MAX_BINS = 6
HEALTHY_WEIGHT = 1.05
SUSPICION_HCUT = 2.0
SUSPICION_OFFSET = 0.3
REMOVE_CLASSES = {7, 8, 11, 12, 13}
AGAINST_SCALE = 0.6
HEALTHY_BAR_CAP = 5.0
CLASS_THRESHOLDS = {2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.5}


# ─── Shared infrastructure ───

def load_data(path):
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
    def nu(self): return len(self.uidx)
    def branch_match(self, row):
        if self.bfeat is None: return True
        v = row[self.bfeat]
        if np.isnan(v): return False
        if self.bbin: return v in self.bvset
        if self.bhi == np.inf: return v > self.blo
        return v <= self.bhi


class BinModel:
    __slots__ = ('n_bins', 'edges', 'bin_counts', 'target_counts',
                 'p_class', 'prior', 'cls_conf_support')
    def __init__(self, n_bins, edges, bin_counts, target_counts,
                 p_class, prior, cls_conf_support):
        self.n_bins, self.edges = n_bins, edges
        self.bin_counts, self.target_counts = bin_counts, target_counts
        self.p_class, self.prior, self.cls_conf_support = p_class, prior, cls_conf_support


def build_tree(data, labels, is_bin):
    n = data.shape[0]
    root = Node("root", 1, np.arange(n), _hdist(labels, np.arange(n)))
    all_nodes = [root]
    current_level = [root]
    ctr = [0]
    while current_level:
        lvl = current_level[0].lvl
        if FORCE_SEX_BRANCHING and lvl > 1: break
        children = []
        for parent in current_level:
            feats = [SEX_FEAT] if FORCE_SEX_BRANCHING else [f for f in range(N_FEAT) if f not in parent.ancestor_feats]
            for f in feats:
                col = data[parent.uidx, f]
                vm = ~np.isnan(col)
                vv = col[vm]
                if len(vv) == 0: continue
                if is_bin[f]:
                    uq = sorted(set(vv))
                    if len(uq) < 2: continue
                    parts = [(parent.uidx[vm & (col == val)], frozenset({val}), None, None, True) for val in uq]
                else:
                    if float(vv.min()) == float(vv.max()): continue
                    med = float(np.median(vv))
                    parts = [(parent.uidx[vm & (col <= med)], None, -np.inf, med, False),
                             (parent.uidx[vm & (col > med)], None, med, np.inf, False)]
                for cu, vs, lo, hi, ib in parts:
                    if len(cu) < U_MIN: continue
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


def _fast_abs_corr(x, y):
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    num = (dx * dy).sum()
    den2 = (dx * dx).sum() * (dy * dy).sum()
    return abs(num / np.sqrt(den2)) if den2 > 0 else 0.0


def _route_user(uid, data, nodes):
    by_lvl = defaultdict(list)
    for nd in nodes: by_lvl[nd.lvl].append(nd)
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
            else: break
    return result


# ─── Supervised binning ───

def _supervised_bin_edges(vv, is_target, max_bins):
    n = len(vv)
    vmin, vmax = float(vv.min()), float(vv.max())
    if vmin == vmax or n < 2 * MIN_SUPPORT:
        return np.array([vmin - 0.5, vmax + 0.5])

    sort_idx = np.argsort(vv)
    sv = vv[sort_idx]
    st = is_target[sort_idx]
    edges = [vmin, vmax]

    for _ in range(max_bins - 1):
        best_gain, best_split, best_seg = 0.0, None, None
        for seg_i in range(len(edges) - 1):
            lo, hi = edges[seg_i], edges[seg_i + 1]
            if seg_i == 0: mask = sv <= hi
            elif seg_i == len(edges) - 2: mask = sv > lo
            else: mask = (sv > lo) & (sv <= hi)
            seg_vals, seg_targ = sv[mask], st[mask]
            n_seg = len(seg_vals)
            if n_seg < 2 * MIN_SUPPORT: continue
            n_t, n_r = seg_targ.sum(), n_seg - seg_targ.sum()
            if n_t == 0 or n_r == 0: continue
            candidates = np.where(seg_vals[:-1] != seg_vals[1:])[0]
            if len(candidates) == 0: continue
            cum_t = np.cumsum(seg_targ)
            for ci in candidates:
                n_left, n_right = ci + 1, n_seg - ci - 1
                if n_left < MIN_SUPPORT or n_right < MIN_SUPPORT: continue
                t_left = cum_t[ci]
                t_right = n_t - t_left
                r_left, r_right = n_left - t_left, n_right - t_right
                e_tl, e_tr = n_left * n_t / n_seg, n_right * n_t / n_seg
                e_rl, e_rr = n_left * n_r / n_seg, n_right * n_r / n_seg
                chi2 = sum((o - e)**2 / e for o, e in
                           [(t_left, e_tl), (t_right, e_tr), (r_left, e_rl), (r_right, e_rr)] if e > 0)
                if chi2 > best_gain:
                    best_gain = chi2
                    best_split = (seg_vals[ci] + seg_vals[ci + 1]) / 2.0
                    best_seg = seg_i
        if best_split is None or best_gain < 0.5: break
        edges.insert(best_seg + 1, best_split)

    edges[0] = vmin - 1e-10
    edges[-1] = vmax + 1e-10
    return np.array(sorted(set(edges)))


# ─── Configurable training ───

def train_ovr_node_cfg(node, data, labels, is_bin, target_class,
                       use_supervised_bins=False):
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
            edges = np.array([vmin - .5, vmin + .5]) if nb == 1 else np.array([-.5, .5, 1.5])
        elif vmin == vmax:
            nb, edges = 1, np.array([vmin - .5, vmin + .5])
        else:
            max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
            if use_supervised_bins:
                edges = _supervised_bin_edges(vv, is_target[vm].astype(float), max_nb)
                nb = len(edges) - 1
            else:
                nb = max_nb
                edges = np.linspace(vmin, vmax, nb + 1)

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]
        bin_counts = np.bincount(ba, minlength=nb)
        target_counts = np.bincount(ba[lv == target_class], minlength=nb).astype(float)
        p_class = (target_counts + LAPLACE_ALPHA * prior) / (bin_counts + LAPLACE_ALPHA)

        models[(node.nid, f)] = BinModel(nb, edges, bin_counts, target_counts, p_class, prior, CONF_SUPPORT)

        score = 0.0
        for b in range(nb):
            if bin_counts[b] >= MIN_SUPPORT:
                shift = abs(p_class[b] - prior)
                confidence = min(1.0, float(bin_counts[b]) / CONF_SUPPORT)
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


def refine_ovr_node_cfg(node, models, node_actions, data,
                        rank_by='score', fpc=18):
    if rank_by == 'fisher':
        scored = [(a, a[3]) for a in node_actions]
    elif rank_by == 'composite':
        scored = [(a, a[2] * (1.0 + np.sqrt(a[3]))) for a in node_actions]
    else:
        scored = [(a, a[2]) for a in node_actions]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored: return []

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
        if f in kept_features: continue
        raw = data[node.uidx, f]
        if (~np.isnan(raw)).sum() == 0: continue
        if kept_features:
            max_corr = max(correlations.get((f, kf), 0.0) for kf in kept_features)
            if max_corr > CORR_THRESHOLD: continue
        kept.append(a)
        kept_features.add(f)
        if len(kept) >= fpc: break
    return kept


def compute_af_cfg(uid, data, nodes, models, retained,
                   against_scale=0.6):
    lvl_nodes = _route_user(uid, data, nodes)
    af_for, af_against = 0.0, 0.0
    n_for, n_against, n_used = 0, 0, 0
    max_for_contrib = 0.0

    fisher_map = {}
    for a in retained:
        fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
    max_fisher = max(fisher_map.values()) if fisher_map else 1.0

    for lvl in sorted(lvl_nodes.keys()):
        for nd in lvl_nodes[lvl]:
            for a in retained:
                if a[1] != nd.nid: continue
                f = a[0]
                v = data[uid, f]
                if np.isnan(v): continue
                mo = models.get((nd.nid, f))
                if not mo: continue
                bin_idx = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'), 0, mo.n_bins - 1))
                bc = mo.bin_counts[bin_idx]
                if bc < MIN_SUPPORT: continue
                p_c = mo.p_class[bin_idx]
                shift = p_c - mo.prior
                confidence = min(1.0, bc / CONF_SUPPORT)
                fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                weighted = abs(shift) * confidence * fw
                if shift >= 0:
                    af_for += weighted
                    n_for += 1
                    if weighted > max_for_contrib: max_for_contrib = weighted
                else:
                    af_against += weighted * against_scale
                    n_against += 1
                n_used += 1

    return af_for, af_against, n_used, n_for, n_against, max_for_contrib


def score_patient(af_for, af_against, n_for, n_against, max_fc, scoring='ratio'):
    if scoring == 'ratio':
        return (af_for + RATIO_EPS) / (af_against + RATIO_EPS)
    elif scoring == 'net':
        return af_for - af_against
    elif scoring == 'breadth_ratio':
        ratio = (af_for + RATIO_EPS) / (af_against + RATIO_EPS)
        breadth = n_for / max(n_for + n_against, 1)
        return breadth * ratio
    elif scoring == 'af_for':
        return af_for
    elif scoring == 'breadth_net':
        breadth = n_for / max(n_for + n_against, 1)
        return breadth * (af_for - af_against)
    return (af_for + RATIO_EPS) / (af_against + RATIO_EPS)


def predict_cfg(uid, data, nodes, class_models, class_retained, all_cls,
                thresholds=None, scoring='ratio', class_scoring=None,
                healthy_scoring=None, against_scale=0.6,
                healthy_bar_cap=5.0, healthy_weight=1.05):
    thresholds = thresholds or CLASS_THRESHOLDS
    class_scores = {}

    for cls in all_cls:
        af_for, af_ag, n_used, n_for, n_ag, max_fc = compute_af_cfg(
            uid, data, nodes, class_models[cls], class_retained[cls],
            against_scale=against_scale)

        if class_scoring and cls in class_scoring:
            s = score_patient(af_for, af_ag, n_for, n_ag, max_fc, class_scoring[cls])
        elif cls == HEALTHY and healthy_scoring:
            s = score_patient(af_for, af_ag, n_for, n_ag, max_fc, healthy_scoring)
        else:
            s = score_patient(af_for, af_ag, n_for, n_ag, max_fc, scoring)
        class_scores[cls] = s

    h_score = class_scores.get(HEALTHY, 1.0)
    healthy_bar = min(healthy_weight * h_score, healthy_bar_cap)

    candidates = {}
    for cls, score in class_scores.items():
        if cls == HEALTHY: continue
        t = thresholds.get(cls, 3.0)
        if h_score < SUSPICION_HCUT:
            t -= SUSPICION_OFFSET
        t = max(t, healthy_bar)
        if score < t: continue
        candidates[cls] = (score - t) / max(t, 0.1)

    best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
    return best_cls, class_scores


def train_all_cfg(nodes, data, labels, is_bin, all_cls,
                  use_supervised_bins=False, rank_by='score',
                  fpc=18, healthy_fpc=None):
    class_models, class_retained = {}, {}
    for cls in all_cls:
        cls_fpc = (healthy_fpc or fpc) if cls == HEALTHY else fpc
        cls_models = {}
        cls_actions_by_node = defaultdict(list)
        for nd in nodes:
            nm, na = train_ovr_node_cfg(nd, data, labels, is_bin, cls,
                                        use_supervised_bins=use_supervised_bins)
            cls_models.update(nm)
            for a in na: cls_actions_by_node[a[1]].append(a)
        cls_ret = []
        for nd in nodes:
            cls_ret.extend(refine_ovr_node_cfg(
                nd, cls_models, cls_actions_by_node.get(nd.nid, []), data,
                rank_by=rank_by, fpc=cls_fpc))
        class_models[cls] = cls_models
        class_retained[cls] = cls_ret
    return class_models, class_retained


def run_10fold(data, labels, is_bin, seed, predict_fn, train_fn):
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
        nodes = build_tree(td, tl, is_bin)
        train_result = train_fn(nodes, td, tl, is_bin, all_cls)
        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict_fn(uid, data, nodes, all_cls, train_result)
            results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def stats(results):
    n = len(results)
    correct = sum(r[3] for r in results)
    th = [r for r in results if r[1] == 1]
    td = [r for r in results if r[1] != 1]
    spec = sum(r[3] for r in th) / len(th) if th else 0
    sens = sum(1 for r in td if r[2] != 1) / len(td) if td else 0
    tp = sum(1 for r in td if r[2] != 1)
    tn = sum(r[3] for r in th)
    fp = len(th) - tn
    fn = len(td) - tp
    ba_cls = {}
    for cls in sorted(set(r[1] for r in results)):
        cr = [r for r in results if r[1] == cls]
        ba_cls[cls] = sum(r[3] for r in cr) / len(cr) if cr else 0
    ba = np.mean(list(ba_cls.values()))
    return correct/n, spec, sens, ba


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def exp_baseline(data, labels, is_bin, seed):
    """V3 baseline — equal-width bins, action-score selection, ratio scoring."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_supervised_bins(data, labels, is_bin, seed):
    """Supervised binning only — bins placed to maximize chi-squared separation."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_composite_select(data, labels, is_bin, seed):
    """Composite feature selection: score * (1 + sqrt(fisher))."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, rank_by='composite')
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_supervised_composite(data, labels, is_bin, seed):
    """Supervised bins + composite feature selection."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, rank_by='composite')
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_net_scoring(data, labels, is_bin, seed):
    """Net scoring (af_for - af_against) for all classes."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, scoring='net',
                          thresholds={2:0.5, 3:1.0, 4:0.8, 5:0.5, 6:0.5, 9:1.0, 10:0.5})
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_breadth_ratio_scoring(data, labels, is_bin, seed):
    """Breadth*ratio scoring for all classes."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, scoring='breadth_ratio',
                          thresholds={2:1.5, 3:2.5, 4:2.0, 5:1.5, 6:1.5, 9:2.5, 10:1.5})
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_healthy_breadth_disease_ratio(data, labels, is_bin, seed):
    """Healthy uses breadth_ratio, disease uses ratio."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          healthy_scoring='breadth_ratio')
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_more_healthy_feats(data, labels, is_bin, seed):
    """27 features for healthy (breadth), 18 for disease."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, healthy_fpc=27)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_lower_against_scale(data, labels, is_bin, seed):
    """Reduce AGAINST_SCALE from 0.6 to 0.4 — weaken disease-against evidence."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.4)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_higher_against_scale(data, labels, is_bin, seed):
    """Increase AGAINST_SCALE from 0.6 to 0.8 — stronger disease-against."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_lower_healthy_bar(data, labels, is_bin, seed):
    """Lower healthy_bar_cap from 5.0 to 3.5 — easier to exceed with disease score."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, healthy_bar_cap=3.5)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_higher_healthy_bar(data, labels, is_bin, seed):
    """Raise healthy_bar_cap from 5.0 to 7.0 — harder to classify as disease."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, healthy_bar_cap=7.0)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_supervised_more_healthy(data, labels, is_bin, seed):
    """Supervised bins + 27 healthy features + composite selection."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, rank_by='composite',
                             healthy_fpc=27)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_per_class_scoring(data, labels, is_bin, seed):
    """Each class uses its best scoring metric (learned on training fold)."""
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls)
        n_train = td.shape[0]
        metrics = ['ratio', 'net', 'breadth_ratio', 'af_for']
        class_best = {}
        for cls in all_cls:
            best_auc, best_m = -1, 'ratio'
            for mt in metrics:
                pos, neg = [], []
                for uid in range(n_train):
                    af = compute_af_cfg(uid, td, nodes, cm[cls], cr[cls])
                    s = score_patient(af[0], af[1], af[3], af[4], af[5], mt)
                    (pos if tl[uid] == cls else neg).append(s)
                if not pos or not neg: continue
                count = sum(1 if p > n else 0.5 if p == n else 0
                           for p in pos for n in neg)
                auc = count / (len(pos) * len(neg))
                if auc > best_auc:
                    best_auc, best_m = auc, mt
            class_best[cls] = best_m
        return cm, cr, class_best

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, class_best = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          class_scoring=class_best)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_supervised_per_class(data, labels, is_bin, seed):
    """Supervised bins + per-class scoring selection."""
    def train(nodes, td, tl, is_bin, all_cls):
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls,
                               use_supervised_bins=True)
        n_train = td.shape[0]
        metrics = ['ratio', 'net', 'breadth_ratio', 'af_for']
        class_best = {}
        for cls in all_cls:
            best_auc, best_m = -1, 'ratio'
            for mt in metrics:
                pos, neg = [], []
                for uid in range(n_train):
                    af = compute_af_cfg(uid, td, nodes, cm[cls], cr[cls])
                    s = score_patient(af[0], af[1], af[3], af[4], af[5], mt)
                    (pos if tl[uid] == cls else neg).append(s)
                if not pos or not neg: continue
                count = sum(1 if p > n else 0.5 if p == n else 0
                           for p in pos for n in neg)
                auc = count / (len(pos) * len(neg))
                if auc > best_auc:
                    best_auc, best_m = auc, mt
            class_best[cls] = best_m
        return cm, cr, class_best

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr, class_best = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          class_scoring=class_best)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_lower_corr_threshold(data, labels, is_bin, seed):
    """Lower correlation threshold from 0.8 to 0.6 — more diverse features."""
    orig = globals()
    saved = CORR_THRESHOLD
    import experiment_runner as mod
    mod.CORR_THRESHOLD = 0.6
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    result = run_10fold(data, labels, is_bin, seed, predict, train)
    mod.CORR_THRESHOLD = saved
    return result


def exp_higher_max_bins(data, labels, is_bin, seed):
    """More bins (8 instead of 6) — finer granularity."""
    import experiment_runner as mod
    saved = mod.MAX_BINS
    mod.MAX_BINS = 8
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    result = run_10fold(data, labels, is_bin, seed, predict, train)
    mod.MAX_BINS = saved
    return result


def exp_fewer_bins(data, labels, is_bin, seed):
    """Fewer bins (4 instead of 6) — more samples per bin, less overfitting."""
    import experiment_runner as mod
    saved = mod.MAX_BINS
    mod.MAX_BINS = 4
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    result = run_10fold(data, labels, is_bin, seed, predict, train)
    mod.MAX_BINS = saved
    return result


def exp_no_fisher_weighting(data, labels, is_bin, seed):
    """Remove Fisher weighting from AF computation — all features equal weight."""
    def compute_af_nofw(uid, data, nodes, models, retained, against_scale=0.6):
        lvl_nodes = _route_user(uid, data, nodes)
        af_for, af_against = 0.0, 0.0
        n_for, n_against, n_used = 0, 0, 0
        max_for_contrib = 0.0
        for lvl in sorted(lvl_nodes.keys()):
            for nd in lvl_nodes[lvl]:
                for a in retained:
                    if a[1] != nd.nid: continue
                    f = a[0]
                    v = data[uid, f]
                    if np.isnan(v): continue
                    mo = models.get((nd.nid, f))
                    if not mo: continue
                    bin_idx = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'), 0, mo.n_bins - 1))
                    bc = mo.bin_counts[bin_idx]
                    if bc < MIN_SUPPORT: continue
                    shift = mo.p_class[bin_idx] - mo.prior
                    confidence = min(1.0, bc / CONF_SUPPORT)
                    weighted = abs(shift) * confidence
                    if shift >= 0:
                        af_for += weighted
                        n_for += 1
                        if weighted > max_for_contrib: max_for_contrib = weighted
                    else:
                        af_against += weighted * against_scale
                        n_against += 1
                    n_used += 1
        return af_for, af_against, n_used, n_for, n_against, max_for_contrib

    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_nofw(uid, data, nodes, cm[cls], cr[cls])
            class_scores[cls] = (af[0] + RATIO_EPS) / (af[1] + RATIO_EPS)
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


def exp_higher_laplace(data, labels, is_bin, seed):
    """Higher Laplace smoothing (2.0 vs 1.0) — more regularization."""
    import experiment_runner as mod
    saved = mod.LAPLACE_ALPHA
    mod.LAPLACE_ALPHA = 2.0
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    result = run_10fold(data, labels, is_bin, seed, predict, train)
    mod.LAPLACE_ALPHA = saved
    return result


def exp_best_combo(data, labels, is_bin, seed):
    """Best combination: supervised bins + composite select + 27 healthy feats + breadth_ratio healthy."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, rank_by='composite',
                             healthy_fpc=27)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          healthy_scoring='breadth_ratio')
    return run_10fold(data, labels, is_bin, seed, predict, train)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats\n")

    seed = 13

    experiments = [
        ("01 Baseline",                    exp_baseline),
        ("02 Supervised bins",             exp_supervised_bins),
        ("03 Composite selection",         exp_composite_select),
        ("04 Supervised+composite",        exp_supervised_composite),
        ("05 Net scoring",                 exp_net_scoring),
        ("06 Breadth*ratio scoring",       exp_breadth_ratio_scoring),
        ("07 Healthy=breadth,disease=ratio", exp_healthy_breadth_disease_ratio),
        ("08 More healthy feats (27)",     exp_more_healthy_feats),
        ("09 Against scale 0.4",           exp_lower_against_scale),
        ("10 Against scale 0.8",           exp_higher_against_scale),
        ("11 Healthy bar cap 3.5",         exp_lower_healthy_bar),
        ("12 Healthy bar cap 7.0",         exp_higher_healthy_bar),
        ("13 Supervised+27H+composite",    exp_supervised_more_healthy),
        ("14 Per-class scoring",           exp_per_class_scoring),
        ("15 Supervised+per-class",        exp_supervised_per_class),
        ("16 Corr threshold 0.6",          exp_lower_corr_threshold),
        ("17 Max bins 8",                  exp_higher_max_bins),
        ("18 Max bins 4",                  exp_fewer_bins),
        ("19 No Fisher weighting",         exp_no_fisher_weighting),
        ("20 Laplace alpha 2.0",           exp_higher_laplace),
        ("21 Best combo",                  exp_best_combo),
    ]

    if len(sys.argv) > 1:
        exp_id = int(sys.argv[1])
        experiments = [experiments[exp_id - 1]]

    print(f"{'Experiment':42s}  {'Acc':>6s}  {'Spec':>6s}  {'Sens':>6s}  {'BA':>6s}  {'Time':>6s}")
    print("-" * 85)

    for name, exp_fn in experiments:
        t0 = time.time()
        results = exp_fn(data, labels, is_bin, seed)
        elapsed = time.time() - t0
        acc, spec, sens, ba = stats(results)
        n_correct = sum(r[3] for r in results)
        print(f"{name:42s}  {100*acc:5.1f}%  {100*spec:5.1f}%  {100*sens:5.1f}%  {100*ba:5.1f}%  {elapsed:5.0f}s",
              flush=True)
