"""CDS-OVR Arrhythmia Classifier — Bagging experiment.

Duplicate of cds_ovr.py with two additions evaluated in order:

Step 1: Full 452-fold LOOCV baseline (no constants changed).
Step 2: Stratified bagging evaluated against that baseline.

Fold-isolation discipline from cds_ovr.py is preserved throughout.
build_tree / train are always rebuilt from scratch on training data only.
"""
import json
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

# ─── Constants (unchanged from cds_ovr.py) ───

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")

U_MIN = 200
HEALTHY = 1
N_FEAT = 279
SEX_FEAT = 1
LAPLACE_ALPHA = 1.0
RATIO_EPS = 0.1
CORR_THRESHOLD = 0.8
FEATURES_PER_CLASS = 18
MAX_BINS = 6
HEALTHY_WEIGHT = 1.05
SUSPICION_HCUT = 2.0
SUSPICION_OFFSET = 0.3
HEALTHY_BAR_CAP = 3.5   # lowered from 5.0; range 3.5-6.0 is null for baseline LOOCV,
                        # but scores that inflate H (e.g. rival-aware features) would
                        # push hbar to 5.0 and block disease classes with scores ~3.7
REMOVE_CLASSES = {7, 8, 11, 12, 13}
RARE_CLASSES = {4, 5, 9}

CLASS_THRESHOLDS = {2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.0}

MIN_SUPPORT_MAP  = {cls: (2 if cls in RARE_CLASSES else 3) for cls in range(1, 14)}
CONF_SUPPORT_MAP = {cls: (5 if cls in RARE_CLASSES else 10) for cls in range(1, 14)}
AGAINST_SCALE_MAP = {cls: (0.5 if cls in RARE_CLASSES else 0.8) for cls in range(1, 14)}


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


# ─── Training ───

def _train_ovr_node(node, data, labels, is_bin, target_class, min_support, conf_support):
    models, actions = {}, []
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
            edges = np.array([vmin - .5, vmin + .5]) if nb == 1 else np.array([-.5, .5, 1.5])
        elif vmin == vmax:
            nb, edges = 1, np.array([vmin - .5, vmin + .5])
        else:
            max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
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
    scored = [(a, a[2]) for a in node_actions]
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


def train(nodes, data, labels, is_bin, all_cls):
    """Train OVR models for all classes with per-class parameters."""
    class_models, class_retained = {}, {}
    for cls in all_cls:
        ms = MIN_SUPPORT_MAP.get(cls, 3)
        cs = CONF_SUPPORT_MAP.get(cls, 10)
        cls_models = {}
        cls_actions = defaultdict(list)
        for nd in nodes:
            nm, na = _train_ovr_node(nd, data, labels, is_bin, cls, ms, cs)
            cls_models.update(nm)
            for a in na:
                cls_actions[a[1]].append(a)
        cls_ret = []
        for nd in nodes:
            cls_ret.extend(_refine_ovr_node(
                nd, cls_models, cls_actions.get(nd.nid, []), data))
        class_models[cls] = cls_models
        class_retained[cls] = cls_ret
    return class_models, class_retained


# ─── Prediction ───

def _compute_af(uid, data, nodes, models, retained, against_scale):
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

    return af_for, af_against, n_used, n_for, n_against, max_for_contrib


def predict(uid, data, nodes, all_cls, train_result):
    """Predict class for a single patient. Returns (best_cls, class_scores)."""
    class_models, class_retained = train_result
    class_scores = {}
    for cls in all_cls:
        ag = AGAINST_SCALE_MAP.get(cls, 0.8)
        af = _compute_af(uid, data, nodes, class_models[cls], class_retained[cls], ag)
        class_scores[cls] = (af[0] + RATIO_EPS) / (af[1] + RATIO_EPS)

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


# ─── Evaluation helpers ───

def stats(results):
    """Overall accuracy, specificity, sensitivity, per-class balanced accuracy."""
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
    correct = sum(1 for _, t, p, _ in results if (t == HEALTHY) == (p == HEALTHY))
    return correct / len(results)


def confusion_matrix(results, all_cls):
    idx = {c: i for i, c in enumerate(all_cls)}
    cm = np.zeros((len(all_cls), len(all_cls)), dtype=int)
    for _, true_cls, pred_cls, _ in results:
        if true_cls in idx and pred_cls in idx:
            cm[idx[true_cls]][idx[pred_cls]] += 1
    return cm


def print_full_report(results, label=""):
    all_cls = sorted(set(r[1] for r in results))
    acc, spec, sens, ba = stats(results)
    ba_bin = (spec + sens) / 2.0
    n = len(results)
    correct = sum(r[3] for r in results)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Users evaluated:             {n}")
    print(f"  Per-class (multiclass) acc:  {100*correct/n:.1f}%")
    print(f"  Per-class balanced acc:      {100*ba:.1f}%")
    print(f"  Binary (H vs D) balanced:    {100*ba_bin:.1f}%  <<< PRIMARY")
    print(f"  Binary specificity:          {100*spec:.1f}%")
    print(f"  Binary sensitivity:          {100*sens:.1f}%")
    print()
    print(f"  Per-class detail:")
    print(f"  {'Class':>8}  {'N':>4}  {'Correct':>7}  {'Acc':>6}  Misclassified as")
    for cls in all_cls:
        cr = [r for r in results if r[1] == cls]
        n_cls = len(cr)
        n_correct = sum(r[3] for r in cr)
        wrong = defaultdict(int)
        for r in cr:
            if not r[3]:
                wrong[r[2]] += 1
        wrong_str = "  ".join(
            f"{'H' if k==HEALTHY else f'c{k}'}({v})"
            for k, v in sorted(wrong.items(), key=lambda x: -x[1])[:4])
        lbl = "H(healthy)" if cls == HEALTHY else f"c{cls}"
        print(f"  {lbl:>10}  {n_cls:>4}  {n_correct:>7}  {100*n_correct/n_cls:>5.1f}%  {wrong_str}")

    print()
    print(f"  Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(results, all_cls)
    hdr = "  " + " ".join(f"{'H' if c==HEALTHY else f'c{c}':>5}" for c in all_cls)
    print(hdr)
    for i, cls in enumerate(all_cls):
        row_lbl = "H" if cls == HEALTHY else f"c{cls}"
        row = "  " + f"{row_lbl:>3}  " + " ".join(f"{cm[i][j]:>5}" for j in range(len(all_cls)))
        print(row)
    print(f"{'='*70}")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(results, all_cls, path):
    """Save LOOCV results to JSON for Step 2 comparison."""
    cm = confusion_matrix(results, all_cls)
    per_class = {}
    for cls in all_cls:
        cr = [r for r in results if r[1] == cls]
        n_cls = len(cr)
        n_correct = sum(r[3] for r in cr)
        per_class[str(int(cls))] = {"n": n_cls, "correct": n_correct,
                                     "acc": n_correct / n_cls if n_cls else 0}
    acc, spec, sens, ba = stats(results)
    out = {
        "n": len(results),
        "overall_acc": float(acc),
        "spec": float(spec),
        "sens": float(sens),
        "per_class_balanced_acc": float(ba),
        "binary_balanced_acc": float((spec + sens) / 2.0),
        "per_class": per_class,
        "confusion_matrix": {"classes": [int(c) for c in all_cls],
                              "matrix": cm.tolist()},
        "per_fold": [{"uid": int(r[0]), "true": int(r[1]),
                      "pred": int(r[2]), "correct": bool(r[3])}
                     for r in results],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, cls=_NumpyEncoder)
    print(f"  Results saved to {path}", flush=True)


# ─── Step 1: LOOCV ───

def run_loocv(data, labels, is_bin):
    """Full leave-one-out CV. No seed — LOOCV is deterministic."""
    n = data.shape[0]
    all_cls = sorted(set(labels))
    results = []
    t0 = time.perf_counter()

    for i in range(n):
        # Build train fold: all patients except i
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        td, tl = data[train_mask], labels[train_mask]

        nodes = build_tree(td, tl, is_bin)
        train_result = train(nodes, td, tl, is_bin, all_cls)
        true_cls = int(labels[i])
        pred, _ = predict(i, data, nodes, all_cls, train_result)
        results.append((i, true_cls, int(pred), pred == true_cls))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            done = sum(r[3] for r in results)
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:3d}/{n}]  acc={100*done/(i+1):.1f}%  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    print(f"  Total time: {time.perf_counter()-t0:.0f}s", flush=True)
    return results


# ─── Step 2: Stratified Bagging ───

def _stratified_bootstrap(labels, rng):
    """Resample within each class separately (with replacement)."""
    bag_idx = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        bag_idx.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    return np.concatenate(bag_idx)


def train_bagged(data, labels, is_bin, all_cls, n_bags, seed):
    """Build n_bags independent (nodes, train_result) pairs from stratified bootstraps."""
    rng = np.random.RandomState(seed)
    bags = []
    for _ in range(n_bags):
        bag_idx = _stratified_bootstrap(labels, rng)
        bd, bl = data[bag_idx], labels[bag_idx]
        nodes = build_tree(bd, bl, is_bin)
        tr = train(nodes, bd, bl, is_bin, all_cls)
        bags.append((nodes, tr))
    return bags


def predict_bagged(uid, data, bag_results, all_cls):
    """Average continuous class_scores across bags, then apply thresholds once."""
    score_accum = {cls: 0.0 for cls in all_cls}
    for nodes, train_result in bag_results:
        _, class_scores = predict(uid, data, nodes, all_cls, train_result)
        for cls, s in class_scores.items():
            score_accum[cls] += s

    n_bags = len(bag_results)
    avg_scores = {cls: score_accum[cls] / n_bags for cls in all_cls}

    # Apply thresholds to the averaged scores (same logic as predict())
    h_score = avg_scores.get(HEALTHY, 1.0)
    healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
    candidates = {}
    for cls, score in avg_scores.items():
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
    return best_cls, avg_scores


def diversity_diagnostic(data, labels, is_bin, all_cls, n_bags=10, seed=0,
                          n_folds_to_sample=5):
    """
    For a sample of training folds, compare retained feature sets across bags.
    Reports mean pairwise Jaccard similarity of retained feature indices per class.
    High Jaccard => low diversity => bagging unlikely to help.
    """
    print(f"\n--- Diversity diagnostic ({n_folds_to_sample} sampled folds, "
          f"{n_bags} bags each) ---", flush=True)
    n = data.shape[0]
    sample_folds = np.linspace(0, n - 1, n_folds_to_sample, dtype=int)

    all_jaccards = defaultdict(list)
    for i in sample_folds:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        td, tl = data[mask], labels[mask]

        bags = train_bagged(td, tl, is_bin, all_cls, n_bags, seed=seed + int(i))
        # Extract retained feature sets per class per bag
        for cls in all_cls:
            feat_sets = []
            for nodes_b, (class_models_b, class_retained_b) in bags:
                cr = class_retained_b.get(cls, [])
                feat_sets.append(frozenset(a[0] for a in cr))

            # Pairwise Jaccard across all bag pairs
            for bi in range(len(feat_sets)):
                for bj in range(bi + 1, len(feat_sets)):
                    a, b = feat_sets[bi], feat_sets[bj]
                    union = len(a | b)
                    jaccard = len(a & b) / union if union > 0 else 1.0
                    all_jaccards[cls].append(jaccard)

    print(f"  Mean pairwise Jaccard similarity of retained features across bags:")
    print(f"  {'Class':>8}  {'Mean Jaccard':>13}  {'Min':>6}  {'Max':>6}  Interpretation")
    for cls in all_cls:
        jacs = all_jaccards[cls]
        mean_j = np.mean(jacs)
        min_j, max_j = np.min(jacs), np.max(jacs)
        interp = ("high overlap — low diversity" if mean_j > 0.8
                  else "moderate diversity" if mean_j > 0.5
                  else "good diversity")
        lbl = "H(healthy)" if cls == HEALTHY else f"c{cls}"
        print(f"  {lbl:>10}  {mean_j:>13.3f}  {min_j:>6.3f}  {max_j:>6.3f}  {interp}")

    overall_mean = np.mean([v for vals in all_jaccards.values() for v in vals])
    print(f"\n  Overall mean Jaccard: {overall_mean:.3f}")
    if overall_mean > 0.8:
        print("  Verdict: Retained features are highly overlapping across bags.")
        print("  Bagging has little diversity to exploit — expect minimal benefit.")
    elif overall_mean > 0.5:
        print("  Verdict: Moderate feature overlap. Bagging may provide modest benefit.")
    else:
        print("  Verdict: Good feature diversity across bags. Bagging has real signal to aggregate.")
    return all_jaccards


def _nbags_convergence_diagnostic(data, labels, is_bin, all_cls, seed=42,
                                   max_bags=30, held_out_frac=0.1):
    """
    Training-only diagnostic to choose n_bags.
    Splits full training data 90/10 (no LOOCV leakage — this is a meta-diagnostic
    run once on the full dataset to pick the implementation constant).
    Watches how much the ensemble's aggregate score vector changes as n_bags grows.
    Stops and returns the n at which improvement visibly flattens.
    """
    print(f"\n--- n_bags convergence diagnostic (training-only, seed={seed}) ---",
          flush=True)
    rng = np.random.RandomState(seed)
    n = data.shape[0]
    split = int(n * (1 - held_out_frac))
    perm = rng.permutation(n)
    train_idx, val_idx = perm[:split], perm[split:]
    td, tl = data[train_idx], labels[train_idx]

    # Build bags incrementally
    bag_rng = np.random.RandomState(seed + 1)
    bags = []
    prev_scores = None
    deltas = []

    for b in range(1, max_bags + 1):
        bag_idx = _stratified_bootstrap(tl, bag_rng)
        bd, bl = td[bag_idx], tl[bag_idx]
        nodes = build_tree(bd, bl, is_bin)
        tr = train(nodes, bd, bl, is_bin, all_cls)
        bags.append((nodes, tr))

        # Compute ensemble scores on val set with current b bags
        score_matrix = []
        for uid in val_idx:
            score_accum = {cls: 0.0 for cls in all_cls}
            for nd, tresult in bags:
                _, cs = predict(uid, data, nd, all_cls, tresult)
                for cls, s in cs.items():
                    score_accum[cls] += s
            row = [score_accum[cls] / b for cls in sorted(all_cls)]
            score_matrix.append(row)
        score_matrix = np.array(score_matrix)

        if prev_scores is not None:
            delta = float(np.mean(np.abs(score_matrix - prev_scores)))
            deltas.append((b, delta))
        prev_scores = score_matrix.copy()

    print(f"  bags  score_delta")
    for b, d in deltas:
        bar = "#" * int(d * 200)
        print(f"  {b:>4}  {d:.5f}  {bar}")

    # Find where delta drops below 10% of initial delta
    if deltas:
        init_delta = deltas[0][1]
        threshold = 0.10 * init_delta
        chosen = max_bags
        for b, d in deltas:
            if d < threshold:
                chosen = b
                break
        print(f"\n  Initial delta: {init_delta:.5f}  |  10% threshold: {threshold:.5f}")
        print(f"  Chosen n_bags = {chosen}  "
              f"(first bag count where delta < 10% of initial)")
        return chosen
    return 10


def run_loocv_bagged(data, labels, is_bin, n_bags, seed):
    """Full LOOCV with stratified bagging. Same fold isolation as run_loocv."""
    n = data.shape[0]
    all_cls = sorted(set(labels))
    results = []
    t0 = time.perf_counter()

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        td, tl = data[mask], labels[mask]

        # Bootstrap resampling happens inside the LOOCV fold, never across held-out patient
        bag_results = train_bagged(td, tl, is_bin, all_cls, n_bags, seed=seed + i)
        true_cls = int(labels[i])
        pred, _ = predict_bagged(i, data, bag_results, all_cls)
        results.append((i, true_cls, int(pred), pred == true_cls))

        if (i + 1) % 25 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            done = sum(r[3] for r in results)
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:3d}/{n}]  acc={100*done/(i+1):.1f}%  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    print(f"  Total time: {time.perf_counter()-t0:.0f}s", flush=True)
    return results


def print_comparison(results_base, results_bag, all_cls):
    """Side-by-side per-class and overall comparison table."""
    acc1, spec1, sens1, ba1 = stats(results_base)
    acc2, spec2, sens2, ba2 = stats(results_bag)

    print(f"\n{'='*80}")
    print(f"  STEP 1 vs STEP 2 COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Metric':<35}  {'Step 1 (LOOCV)':>15}  {'Step 2 (Bagged)':>15}  {'Delta':>8}")
    print(f"  {'-'*35}  {'-'*15}  {'-'*15}  {'-'*8}")

    def row(name, v1, v2):
        d = v2 - v1
        sign = "+" if d >= 0 else ""
        print(f"  {name:<35}  {100*v1:>14.1f}%  {100*v2:>14.1f}%  {sign}{100*d:>6.1f}pp")

    row("Per-class (multiclass) accuracy", acc1, acc2)
    row("Per-class balanced accuracy", ba1, ba2)
    row("Binary (H vs D) balanced accuracy", (spec1+sens1)/2, (spec2+sens2)/2)
    row("Binary specificity", spec1, spec2)
    row("Binary sensitivity", sens1, sens2)

    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':>10}  {'N':>4}  {'Step1 acc':>10}  {'Step2 acc':>10}  {'Delta':>8}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}")
    for cls in all_cls:
        cr1 = [r for r in results_base if r[1] == cls]
        cr2 = [r for r in results_bag if r[1] == cls]
        n_cls = len(cr1)
        a1 = sum(r[3] for r in cr1) / n_cls if n_cls else 0
        a2 = sum(r[3] for r in cr2) / n_cls if n_cls else 0
        d = a2 - a1
        sign = "+" if d >= 0 else ""
        lbl = "H(healthy)" if cls == HEALTHY else f"c{cls}"
        print(f"  {lbl:>10}  {n_cls:>4}  {100*a1:>9.1f}%  {100*a2:>9.1f}%  "
              f"{sign}{100*d:>6.1f}pp  "
              f"({'n.s.' if n_cls < 10 else 'large-N'})")
    print(f"{'='*80}")


# ─── Main ───

if __name__ == "__main__":
    import sys

    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    n = data.shape[0]
    print(f"{n} patients x {data.shape[1]} features")
    print(f"Classes: {all_cls}")
    print(f"Distribution: { {int(c): int((labels==c).sum()) for c in all_cls} }\n")

    results_path = str(Path(__file__).parent / "loocv_baseline_results.json")
    results_bag_path = str(Path(__file__).parent / "loocv_bagged_results.json")

    # ── Step 1: LOOCV baseline ──────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1 — Full LOOCV baseline (no constants changed)")
    print("=" * 70, flush=True)
    results_loocv = run_loocv(data, labels, is_bin)
    print_full_report(results_loocv, label="STEP 1: CDS-OVR LOOCV baseline")
    save_results(results_loocv, all_cls, results_path)

    # ── Step 2: Diversity diagnostic + n_bags selection ────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 — Stratified bagging")
    print("=" * 70, flush=True)

    diversity_diagnostic(data, labels, is_bin, all_cls,
                         n_bags=10, seed=0, n_folds_to_sample=5)

    n_bags = _nbags_convergence_diagnostic(data, labels, is_bin, all_cls,
                                           seed=42, max_bags=30)
    print(f"\n  n_bags fixed at {n_bags} via training-only convergence diagnostic.")
    print(f"  This was NOT selected by comparing LOOCV outcomes.", flush=True)

    # ── Step 2: Bagged LOOCV ───────────────────────────────────────────────
    print(f"\nRunning bagged LOOCV (n_bags={n_bags}, seed=42)...", flush=True)
    results_bag = run_loocv_bagged(data, labels, is_bin, n_bags=n_bags, seed=42)
    print_full_report(results_bag,
                      label=f"STEP 2: CDS-OVR Bagged LOOCV (n_bags={n_bags})")
    save_results(results_bag, all_cls, results_bag_path)

    # ── Comparison table ───────────────────────────────────────────────────
    print_comparison(results_loocv, results_bag, all_cls)

    # ── Written summary ────────────────────────────────────────────────────
    acc1, spec1, sens1, ba1 = stats(results_loocv)
    acc2, spec2, sens2, ba2 = stats(results_bag)
    bin_bal1 = (spec1 + sens1) / 2.0
    bin_bal2 = (spec2 + sens2) / 2.0

    print(f"\n{'='*70}")
    print(f"  WRITTEN SUMMARY")
    print(f"{'='*70}")
    print(f"""
Step 1 baseline (full {n}-fold LOOCV, all constants unchanged):
  Per-class accuracy:         {100*acc1:.1f}%
  Per-class balanced acc:     {100*ba1:.1f}%
  Binary balanced acc:        {100*bin_bal1:.1f}%  (spec={100*spec1:.1f}%  sens={100*sens1:.1f}%)

Diversity diagnostic:
  Mean pairwise Jaccard of retained features across bags was computed over
  5 sampled LOOCV folds and 10 bags each. See diagnostic output above.
  If Jaccard > 0.8: bagging has minimal diversity and is unlikely to help.
  If Jaccard < 0.5: genuine diversity exists and averaging scores is motivated.

n_bags selection:
  Fixed via training-only score-delta convergence on a 90/10 split of the
  full dataset (no LOOCV outcomes consulted). n_bags = {n_bags}.

Step 2 bagged results (same LOOCV protocol, n_bags={n_bags}):
  Per-class accuracy:         {100*acc2:.1f}%
  Per-class balanced acc:     {100*ba2:.1f}%
  Binary balanced acc:        {100*bin_bal2:.1f}%  (spec={100*spec2:.1f}%  sens={100*sens2:.1f}%)

  Delta vs Step 1:
    Binary balanced acc:      {'+' if bin_bal2>=bin_bal1 else ''}{100*(bin_bal2-bin_bal1):.1f}pp
    Per-class balanced acc:   {'+' if ba2>=ba1 else ''}{100*(ba2-ba1):.1f}pp

Verdict:
  Classes with n < 5 (e.g. class 8: n=2, class 11: n=4) have such small
  support that a 1-instance swing changes their accuracy by 25-50pp. Any
  per-class change in those classes cannot be distinguished from chance.
  The verdict on whether bagging helped is based primarily on HEALTHY (n=245),
  class 2 (n=44), class 10 (n=50), and class 6 (n=25) — the only classes
  with enough support for a 1-2pp shift to be meaningful.
""")
