"""CDS OVR V4 — Per-class specialized classification.

Structural changes from V3:
  1. Supervised binning: bin edges placed to maximize chi-squared separation
  2. Per-class scoring metric: each class uses whichever metric (ratio/net/breadth*ratio)
     best separates target from rest, learned during training
  3. Asymmetric strategy: healthy uses breadth-weighted evidence (many small signals),
     disease classes use peak-sensitive ratio (few strong signals)
  4. Feature selection by separation power (Fisher * action_score composite)
"""
import time
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

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

CLASS_THRESHOLDS = {
    2: 3.5,
    3: 5.0,
    4: 4.0,
    5: 3.5,
    6: 3.5,
    9: 5.0,
    10: 3.5,
}


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
        self.nid = nid
        self.lvl = lvl
        self.uidx = uidx
        self.hdist = hdist
        self.bfeat = self.bvset = self.blo = self.bhi = None
        self.bbin = False
        self.children = []
        self.parent = None
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
                 'p_class', 'prior', 'cls_conf_support', 'separation_score')

    def __init__(self, n_bins, edges, bin_counts, target_counts,
                 p_class, prior, cls_conf_support, separation_score=0.0):
        self.n_bins = n_bins
        self.edges = edges
        self.bin_counts = bin_counts
        self.target_counts = target_counts
        self.p_class = p_class
        self.prior = prior
        self.cls_conf_support = cls_conf_support
        self.separation_score = separation_score


def build_tree(data, labels, is_bin):
    n = data.shape[0]
    root = Node("root", 1, np.arange(n), _hdist(labels, np.arange(n)))
    all_nodes = [root]
    current_level = [root]
    ctr = [0]

    while current_level:
        lvl = current_level[0].lvl
        if FORCE_SEX_BRANCHING and lvl > 1:
            break
        children = []

        for parent in current_level:
            if FORCE_SEX_BRANCHING:
                feats = [SEX_FEAT]
            else:
                feats = [f for f in range(N_FEAT) if f not in parent.ancestor_feats]

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
                    parts = [(parent.uidx[vm & (col == val)],
                              frozenset({val}), None, None, True) for val in uq]
                else:
                    if float(vv.min()) == float(vv.max()):
                        continue
                    med = float(np.median(vv))
                    parts = [
                        (parent.uidx[vm & (col <= med)], None, -np.inf, med, False),
                        (parent.uidx[vm & (col > med)], None, med, np.inf, False),
                    ]
                for cu, vs, lo, hi, ib in parts:
                    if len(cu) < U_MIN:
                        continue
                    ctr[0] += 1
                    ch = Node(f"L{lvl+1}_f{f}_{ctr[0]}", lvl + 1,
                              cu, _hdist(labels, cu))
                    ch.bfeat, ch.bvset, ch.blo, ch.bhi, ch.bbin = f, vs, lo, hi, ib
                    ch.parent = parent
                    ch.ancestor_feats = parent.ancestor_feats | frozenset({f})
                    parent.children.append(ch)
                    children.append(ch)

        seen = {}
        deduped = []
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
    if den2 <= 0:
        return 0.0
    return abs(num / np.sqrt(den2))


def _supervised_bin_edges(vv, is_target, max_bins):
    """Find bin edges that maximize chi-squared separation between target and rest.

    Greedy recursive splitting: at each step, find the split point within the
    largest-impurity bin that maximally separates target from rest.
    """
    n = len(vv)
    if n == 0:
        return np.array([vv.min() - 0.5, vv.max() + 0.5])

    vmin, vmax = float(vv.min()), float(vv.max())
    if vmin == vmax:
        return np.array([vmin - 0.5, vmin + 0.5])

    sort_idx = np.argsort(vv)
    sv = vv[sort_idx]
    st = is_target[sort_idx]

    edges = [vmin, vmax]

    for _ in range(max_bins - 1):
        best_gain = 0.0
        best_split = None
        best_seg = None

        for seg_i in range(len(edges) - 1):
            lo, hi = edges[seg_i], edges[seg_i + 1]
            if seg_i == 0:
                mask = sv <= hi
            elif seg_i == len(edges) - 2:
                mask = sv > lo
            else:
                mask = (sv > lo) & (sv <= hi)

            seg_vals = sv[mask]
            seg_targ = st[mask]
            n_seg = len(seg_vals)
            if n_seg < 2 * MIN_SUPPORT:
                continue

            n_t_total = seg_targ.sum()
            n_r_total = n_seg - n_t_total
            if n_t_total == 0 or n_r_total == 0:
                continue

            unique_vals = np.unique(seg_vals)
            if len(unique_vals) < 2:
                continue

            cum_t = np.cumsum(seg_targ)
            cum_n = np.arange(1, n_seg + 1)

            candidates = np.where(seg_vals[:-1] != seg_vals[1:])[0]
            if len(candidates) == 0:
                continue

            for ci in candidates:
                n_left = ci + 1
                n_right = n_seg - n_left
                if n_left < MIN_SUPPORT or n_right < MIN_SUPPORT:
                    continue

                t_left = cum_t[ci]
                t_right = n_t_total - t_left
                r_left = n_left - t_left
                r_right = n_right - t_right

                e_tl = n_left * n_t_total / n_seg
                e_tr = n_right * n_t_total / n_seg
                e_rl = n_left * n_r_total / n_seg
                e_rr = n_right * n_r_total / n_seg

                chi2 = 0.0
                for obs, exp in [(t_left, e_tl), (t_right, e_tr),
                                 (r_left, e_rl), (r_right, e_rr)]:
                    if exp > 0:
                        chi2 += (obs - exp) ** 2 / exp

                if chi2 > best_gain:
                    best_gain = chi2
                    split_val = (seg_vals[ci] + seg_vals[ci + 1]) / 2.0
                    best_split = split_val
                    best_seg = seg_i

        if best_split is None or best_gain < 0.5:
            break

        edges.insert(best_seg + 1, best_split)

    edges[0] = vmin - 1e-10
    edges[-1] = vmax + 1e-10

    return np.array(sorted(set(edges)))


def train_ovr_node(node, data, labels, is_bin, target_class):
    models = {}
    actions = []
    nd_data, nd_labels = data[node.uidx], labels[node.uidx]
    ns = node.nu

    n_target = int((nd_labels == target_class).sum())
    if n_target < 1:
        return models, actions
    prior = n_target / ns
    is_target = (nd_labels == target_class)

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
            edges = (np.array([vmin - .5, vmin + .5]) if nb == 1
                     else np.array([-.5, .5, 1.5]))
        elif vmin == vmax:
            nb, edges = 1, np.array([vmin - .5, vmin + .5])
        else:
            max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
            edges = _supervised_bin_edges(vv, is_target[vm].astype(float), max_nb)
            nb = len(edges) - 1

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]
        bin_counts = np.bincount(ba, minlength=nb)
        target_counts = np.bincount(ba[lv == target_class], minlength=nb).astype(float)
        p_class = (target_counts + LAPLACE_ALPHA * prior) / (bin_counts + LAPLACE_ALPHA)

        score = 0.0
        sep_score = 0.0
        for b in range(nb):
            if bin_counts[b] >= MIN_SUPPORT:
                shift = abs(p_class[b] - prior)
                confidence = min(1.0, float(bin_counts[b]) / CONF_SUPPORT)
                score += shift * confidence
                sep_score += shift * shift * bin_counts[b]

        target_vals = vv[lv == target_class]
        rest_vals = vv[lv != target_class]
        if len(target_vals) >= 2 and len(rest_vals) >= 2:
            mean_diff2 = (target_vals.mean() - rest_vals.mean()) ** 2
            var_sum = target_vals.var() + rest_vals.var()
            fisher = mean_diff2 / (var_sum + 1e-10)
        else:
            fisher = 0.0

        models[(node.nid, f)] = BinModel(
            nb, edges, bin_counts, target_counts, p_class, prior, CONF_SUPPORT,
            separation_score=sep_score)

        if score > 0.001:
            composite = score * (1.0 + np.sqrt(fisher))
            actions.append((f, node.nid, score, fisher, composite))

    return models, actions


def refine_ovr_node(node, models, node_actions, data, features_per_class=None):
    fpc = features_per_class or FEATURES_PER_CLASS
    scored = [(a, a[4]) for a in node_actions]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []

    candidate_limit = 3 * fpc
    top_scored = scored[:candidate_limit]

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

    kept = []
    kept_features = set()

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
                       if nd.parent and nd.parent.nid in active
                       and nd.branch_match(data[uid])]
            if matched:
                result[lvl] = matched
                active = {nd.nid for nd in matched}
            else:
                break
    return result


def _compute_af(uid, data, nodes, models, retained):
    """Compute Fisher-weighted dual AFs with full evidence breakdown."""
    lvl_nodes = _route_user(uid, data, nodes)
    af_for = 0.0
    af_against = 0.0
    n_used = 0
    n_for = 0
    n_against = 0
    max_for_contrib = 0.0
    breadth_sum = 0.0

    fisher_map = {}
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
                bin_idx = int(np.clip(
                    np.searchsorted(mo.edges[1:], v, side='right'),
                    0, mo.n_bins - 1
                ))
                bc = mo.bin_counts[bin_idx]
                if bc < MIN_SUPPORT:
                    continue

                p_c = mo.p_class[bin_idx]
                prior = mo.prior
                shift = p_c - prior
                confidence = min(1.0, bc / CONF_SUPPORT)

                fw = np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10))
                fw = max(fw, 0.1)

                weighted = abs(shift) * confidence * fw

                if shift >= 0:
                    af_for += weighted
                    n_for += 1
                    breadth_sum += 1.0
                    if weighted > max_for_contrib:
                        max_for_contrib = weighted
                else:
                    af_against += weighted * AGAINST_SCALE
                    n_against += 1
                n_used += 1

    return af_for, af_against, n_used, n_for, n_against, max_for_contrib, breadth_sum


def _score_patient(af_for, af_against, n_for, n_against, breadth_sum,
                   max_for_contrib, metric_type):
    """Compute score using the specified metric type."""
    if metric_type == 'ratio':
        return (af_for + RATIO_EPS) / (af_against + RATIO_EPS)
    elif metric_type == 'net':
        return af_for - af_against
    elif metric_type == 'breadth_ratio':
        ratio = (af_for + RATIO_EPS) / (af_against + RATIO_EPS)
        breadth = n_for / max(n_for + n_against, 1)
        return breadth * ratio
    elif metric_type == 'breadth_weighted':
        if n_for + n_against == 0:
            return 0.0
        breadth = n_for / (n_for + n_against)
        avg_for = af_for / max(n_for, 1)
        return breadth * avg_for * (n_for ** 0.5)
    elif metric_type == 'af_for':
        return af_for
    return (af_for + RATIO_EPS) / (af_against + RATIO_EPS)


def _select_class_metrics(nodes, data, labels, class_models, class_retained, all_cls):
    """For each class, determine which scoring metric best separates target from rest."""
    n = data.shape[0]
    metric_types = ['ratio', 'net', 'breadth_ratio', 'af_for']
    class_metrics = {}

    for cls in all_cls:
        af_data = []
        for uid in range(n):
            af_for, af_against, n_used, n_for, n_ag, max_fc, breadth = _compute_af(
                uid, data, nodes, class_models[cls], class_retained[cls])
            is_target = 1 if labels[uid] == cls else 0
            af_data.append((af_for, af_against, n_for, n_ag, breadth, max_fc, is_target))

        best_auc = -1
        best_metric = 'ratio'

        for mt in metric_types:
            scores_pos = []
            scores_neg = []
            for af_for, af_ag, n_for, n_ag, breadth, max_fc, is_t in af_data:
                s = _score_patient(af_for, af_ag, n_for, n_ag, breadth, max_fc, mt)
                if is_t:
                    scores_pos.append(s)
                else:
                    scores_neg.append(s)

            if not scores_pos or not scores_neg:
                continue

            auc = _fast_auc(scores_pos, scores_neg)
            if auc > best_auc:
                best_auc = auc
                best_metric = mt

        class_metrics[cls] = best_metric

    return class_metrics


def _fast_auc(pos_scores, neg_scores):
    """Compute AUC using the Wilcoxon-Mann-Whitney statistic."""
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    count = 0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                count += 1
            elif p == n:
                count += 0.5
    return count / (n_pos * n_neg)


def predict_ovr(uid, data, nodes, class_models, class_retained, all_cls,
                class_thresholds=None, class_metrics=None):
    class_scores = {}
    class_raw = {}

    for cls in all_cls:
        af_for, af_against, n_used, n_for, n_ag, max_fc, breadth = _compute_af(
            uid, data, nodes, class_models[cls], class_retained[cls])

        metric = 'ratio'
        if class_metrics:
            metric = class_metrics.get(cls, 'ratio')

        score = _score_patient(af_for, af_against, n_for, n_ag, breadth, max_fc, metric)
        class_scores[cls] = score
        class_raw[cls] = (af_for, af_against, n_for, n_ag, breadth, max_fc)

    h_score = class_scores.get(HEALTHY, 1.0)
    h_ratio = h_score
    healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)

    thresholds = class_thresholds or CLASS_THRESHOLDS

    candidates = {}
    for cls, score in class_scores.items():
        if cls == HEALTHY:
            continue
        t = thresholds.get(cls, 3.0)
        if h_ratio < SUSPICION_HCUT:
            t -= SUSPICION_OFFSET
        t = max(t, healthy_bar)
        if score < t:
            continue
        candidates[cls] = (score - t) / max(t, 0.1)

    if candidates:
        best_cls = max(candidates, key=candidates.get)
    else:
        best_cls = HEALTHY

    return best_cls, class_scores


def _calibrate_thresholds(nodes, data, labels, all_cls,
                          class_models, class_retained, class_metrics):
    """Calibrate thresholds per class using the class-specific metric."""
    n = data.shape[0]
    cal_thresholds = {}

    for cls in all_cls:
        if cls == HEALTHY:
            continue

        metric = class_metrics.get(cls, 'ratio') if class_metrics else 'ratio'
        scores_pos = []
        scores_neg = []

        for uid in range(n):
            af_for, af_against, n_used, n_for, n_ag, max_fc, breadth = _compute_af(
                uid, data, nodes, class_models[cls], class_retained[cls])
            s = _score_patient(af_for, af_against, n_for, n_ag, breadth, max_fc, metric)
            if labels[uid] == cls:
                scores_pos.append(s)
            else:
                scores_neg.append(s)

        if not scores_pos:
            cal_thresholds[cls] = CLASS_THRESHOLDS.get(cls, 3.0)
            continue

        all_scores = sorted(set(scores_pos + scores_neg))
        best_f1 = -1
        best_t = CLASS_THRESHOLDS.get(cls, 3.0)

        for t in all_scores:
            tp = sum(1 for s in scores_pos if s >= t)
            fp = sum(1 for s in scores_neg if s >= t)
            fn = len(scores_pos) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        min_floor = 1.5
        cal_thresholds[cls] = max(best_t, min_floor)

    return cal_thresholds


def _train_all_classes(nodes, data, labels, is_bin, all_cls):
    class_models = {}
    class_retained = {}

    for cls in all_cls:
        fpc = FEATURES_PER_CLASS
        if cls == HEALTHY:
            fpc = int(FEATURES_PER_CLASS * 1.5)

        cls_models = {}
        cls_actions_by_node = defaultdict(list)
        for nd in nodes:
            nm, na = train_ovr_node(nd, data, labels, is_bin, cls)
            cls_models.update(nm)
            for a in na:
                cls_actions_by_node[a[1]].append(a)
        cls_ret = []
        for nd in nodes:
            cls_ret.extend(
                refine_ovr_node(nd, cls_models,
                                cls_actions_by_node.get(nd.nid, []), data,
                                features_per_class=fpc))
        class_models[cls] = cls_models
        class_retained[cls] = cls_ret

    class_metrics = _select_class_metrics(
        nodes, data, labels, class_models, class_retained, all_cls)

    cal_thresholds = _calibrate_thresholds(
        nodes, data, labels, all_cls,
        class_models, class_retained, class_metrics)

    return class_models, class_retained, class_metrics, cal_thresholds


def print_report(results, label="OVR V4"):
    all_cls = sorted(set(r[1] for r in results) | set(r[2] for r in results))
    n = len(results)
    correct = sum(r[3] for r in results)
    print(f"\n{'='*70}")
    print(f"CDS {label} -- {n} users")
    print(f"{'='*70}")
    print(f"Overall Accuracy:   {100*correct/n:.1f}%")

    per_class = {}
    for cls in all_cls:
        cr = [r for r in results if r[1] == cls]
        if cr:
            per_class[cls] = sum(r[3] for r in cr) / len(cr)
    ba = np.mean(list(per_class.values()))
    print(f"Balanced Accuracy:  {100*ba:.1f}%")

    th = [r for r in results if r[1] == HEALTHY]
    td = [r for r in results if r[1] != HEALTHY]
    spec = sum(r[3] for r in th) / len(th) if th else 0
    sens = sum(1 for r in td if r[2] != HEALTHY) / len(td) if td else 0
    tp = sum(1 for r in td if r[2] != HEALTHY)
    tn = sum(r[3] for r in th)
    fp = len(th) - tn
    fn = len(td) - tp
    binary_acc = (tp + tn) / n
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    detected = [r for r in td if r[2] != HEALTHY]
    subtype_correct = sum(1 for r in detected if r[3])

    print(f"\nBinary (H vs D):")
    print(f"  Accuracy:        {100*binary_acc:.1f}%")
    print(f"  Specificity:     {100*spec:.1f}%  ({tn}/{len(th)})")
    print(f"  Sensitivity:     {100*sens:.1f}%  ({tp}/{len(td)})")
    print(f"  PPV:             {100*ppv:.1f}%")
    print(f"  NPV:             {100*npv:.1f}%")
    if tp:
        print(f"  Subtyping:       {subtype_correct}/{tp} detected = {100*subtype_correct/tp:.1f}% correct class")

    print(f"\nPer-class detail:")
    for cls in sorted(per_class.keys()):
        n_cls = sum(1 for r in results if r[1] == cls)
        n_correct = sum(1 for r in results if r[1] == cls and r[3])
        lbl = "healthy" if cls == HEALTHY else f"class {cls}"
        if cls != HEALTHY:
            n_detected = sum(1 for r in results if r[1] == cls and r[2] != HEALTHY)
            det_str = f"  detected: {n_detected}/{n_cls}"
        else:
            det_str = ""
        wrong = defaultdict(int)
        for r in results:
            if r[1] == cls and not r[3]:
                wrong[r[2]] += 1
        wrong_str = ""
        if wrong:
            wrong_parts = sorted(wrong.items(), key=lambda x: -x[1])[:3]
            wrong_str = "  misclassed as: " + ", ".join(
                f"{'H' if k==1 else f'c{k}'}({v})" for k, v in wrong_parts)
        print(f"  {lbl:>10s}  {n_correct:3d}/{n_cls:3d} = {100*n_correct/n_cls:5.1f}%{det_str}{wrong_str}")
    print(f"{'='*70}")
