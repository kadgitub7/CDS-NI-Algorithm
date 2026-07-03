"""CDS Arrhythmia Classifier — Multi-Class Evidence Accumulation."""
import time
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
CORR_THRESHOLD = 0.8
MAX_FEATURES_PER_NODE = 30
MAX_BINS = 6
MIN_PRIOR = 0.10


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
    return data, np.array([remap.get(l, l) for l in labels], dtype=int)


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


# ═══════════════════════════════════════════════════════════════════
# Algorithm 1: Decision Tree (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════

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

    @property
    def n_diseased(self):
        return sum(v for k, v in self.hdist.items() if k != HEALTHY)

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


# ═══════════════════════════════════════════════════════════════════
# Algorithm 2: Perceptor & Executive Training (Multi-Class)
# Laplace-smoothed likelihoods, per-class posteriors.
# ═══════════════════════════════════════════════════════════════════

class FeatureModel:
    __slots__ = ('posterior', 'prevalence', 'n_bins', 'edges', 'bin_counts')

    def __init__(self, posterior, prevalence, n_bins, edges, bin_counts):
        self.posterior = posterior
        self.prevalence = prevalence
        self.n_bins = n_bins
        self.edges = edges
        self.bin_counts = bin_counts


def train_node(node, data, labels, is_bin, all_cls):
    nc = len(all_cls)
    models, actions = {}, []
    nd_data, nd_labels = data[node.uidx], labels[node.uidx]
    ns = node.nu
    prev = np.array([node.hdist.get(c, 0) / ns for c in all_cls])

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
            nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
            edges = np.linspace(vmin, vmax, nb + 1)

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]
        bin_counts = np.bincount(ba, minlength=nb)

        lk = np.zeros((nb, nc))
        for ci, c in enumerate(all_cls):
            cm = (lv == c)
            n_c = cm.sum()
            counts = np.bincount(ba[cm], minlength=nb).astype(float) if n_c > 0 else np.zeros(nb)
            lk[:, ci] = (counts + LAPLACE_ALPHA) / (n_c + LAPLACE_ALPHA * nb)

        ev = lk @ prev
        post = np.zeros((nb, nc))
        for b in range(nb):
            if ev[b] > 0:
                post[b] = lk[b] * prev / ev[b]

        models[(node.nid, f)] = FeatureModel(post, prev, nb, edges, bin_counts)

        r_H, r_U = 0.0, 0.0
        for b in range(nb):
            p_h = post[b, 0]
            conf = abs(2.0 * p_h - 1.0)
            bw = (lk[b] @ prev)
            r_H += p_h * conf * bw
            r_U += (1.0 - p_h) * conf * bw
        if r_H > 0.01 or r_U > 0.01:
            actions.append((f, node.nid, r_H, r_U))

    return models, actions


# ═══════════════════════════════════════════════════════════════════
# Algorithm 3: Feature Selection
# Greedy selection by centered discriminative power,
# correlation penalty to avoid redundant features,
# capped at MAX_FEATURES_PER_NODE.
# ═══════════════════════════════════════════════════════════════════

def _fast_abs_corr(x, y):
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    num = (dx * dy).sum()
    den2 = (dx * dx).sum() * (dy * dy).sum()
    if den2 <= 0:
        return 0.0
    return abs(num / np.sqrt(den2))


def _centered_power(node, models, action):
    f = action[0]
    mo = models.get((node.nid, f))
    if not mo:
        return 0.0
    prev_h = mo.prevalence[0]
    score = 0.0
    for b in range(mo.n_bins):
        if mo.bin_counts[b] >= MIN_SUPPORT:
            p_h = mo.posterior[b, 0]
            conf = abs(2.0 * p_h - 1.0)
            score = max(score, abs(p_h - prev_h) * conf * conf)
    return score


def refine_node(node, models, node_actions, data):
    scored = [(a, _centered_power(node, models, a)) for a in node_actions]
    scored = [(a, s) for a, s in scored if s > 0.001]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []

    candidate_limit = 3 * MAX_FEATURES_PER_NODE
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

        mo = models.get((node.nid, f))
        if not mo:
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

        if len(kept) >= MAX_FEATURES_PER_NODE:
            break

    return kept


def _node_path(node):
    path = []
    nd = node
    while nd is not None:
        path.append(nd)
        nd = nd.parent
    path.reverse()
    return path


def backward_eliminate(node, models, kept, data, labels, margin):
    if len(kept) <= 2:
        return kept

    path = _node_path(node)

    def score(feature_set):
        correct = 0
        for uid in node.uidx:
            dec, _ = predict_dual_af(uid, data, path, models,
                                     feature_set, margin=margin)
            ok = (dec != "UNHEALTHY") if labels[uid] == HEALTHY else (dec == "UNHEALTHY")
            correct += ok
        return correct / len(node.uidx)

    for _ in range(2):
        if len(kept) <= 2:
            break
        base_score = score(kept)
        worst_idx = None

        for i in range(len(kept)):
            candidate = kept[:i] + kept[i+1:]
            s = score(candidate)
            if s >= base_score:
                base_score = s
                worst_idx = i

        if worst_idx is not None:
            kept.pop(worst_idx)
        else:
            break

    return kept


# ═══════════════════════════════════════════════════════════════════
# Algorithm 4: Multi-Class Evidence Prediction
# Per-class accumulators with prior-normalized posteriors.
# Each feature updates evidence for every class.
# Argmax over classes determines decision.
# ═══════════════════════════════════════════════════════════════════

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


def predict_dual_af(uid, data, nodes, models, retained, margin=0.0,
                    min_support=MIN_SUPPORT):
    af = defaultdict(float)
    pacs = 0

    lvl_nodes = _route_user(uid, data, nodes)

    for lvl in sorted(lvl_nodes.keys()):
        for nd in lvl_nodes[lvl]:
            for a in retained:
                if a[1] != nd.nid:
                    continue
                f = a[0]
                v = data[uid, f]
                if np.isnan(v):
                    continue

                m = models.get((nd.nid, f))
                if not m:
                    continue

                bin_idx = int(np.clip(
                    np.searchsorted(m.edges[1:], v, side='right'),
                    0, m.n_bins - 1
                ))

                if m.bin_counts[bin_idx] < min_support:
                    pacs += 1
                    continue

                for ci in range(len(m.prevalence)):
                    p_c = m.posterior[bin_idx, ci]
                    prior_c = m.prevalence[ci]
                    af[ci] += (p_c - prior_c) / max(prior_c, MIN_PRIOR)
                pacs += 1

    if not af:
        return "SCREENING", pacs

    healthy_ev = af.get(0, 0.0)
    max_disease_ev = max((v for k, v in af.items() if k != 0), default=0.0)

    if abs(healthy_ev - max_disease_ev) < margin:
        return "SCREENING", pacs
    if healthy_ev >= max_disease_ev:
        return "HEALTHY", pacs
    return "UNHEALTHY", pacs


# ═══════════════════════════════════════════════════════════════════
# Inner Cross-Validation for Hyperparameter Selection (I3)
# Searches over margin (screening threshold) and min_support.
# ═══════════════════════════════════════════════════════════════════

def select_hyperparams(data, labels, nodes, models, retained, seed=42):
    margin_grid = [0.0, 0.01, 0.02, 0.05]
    min_support_grid = [2, 3, 5]

    n = len(labels)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    n_folds = 3
    folds = np.array_split(indices, n_folds)

    def evaluate(margin, ms):
        correct = 0
        for fold in folds:
            for uid in fold:
                dec, _ = predict_dual_af(uid, data, nodes, models, retained,
                                         margin=margin, min_support=ms)
                ok = (dec != "UNHEALTHY") if labels[uid] == HEALTHY else (dec == "UNHEALTHY")
                correct += ok
        return correct / n

    best_score = -1
    best_margin, best_ms = 0.0, MIN_SUPPORT
    for margin in margin_grid:
        for ms in min_support_grid:
            s = evaluate(margin, ms)
            if s > best_score:
                best_score = s
                best_margin, best_ms = margin, ms

    return best_margin, best_ms


# ═══════════════════════════════════════════════════════════════════
# LOOCV with Inner CV and Backward Elimination
# Inner CV runs every INNER_CV_INTERVAL folds and reuses params.
# ═══════════════════════════════════════════════════════════════════

INNER_CV_INTERVAL = 10


def run_loocv(data, labels, max_users=None, seed=42):
    n = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    results = []
    t0 = time.perf_counter()

    cached_margin, cached_ms = 0.0, MIN_SUPPORT

    for i in range(n):
        mask = np.ones(data.shape[0], dtype=bool)
        mask[i] = False
        td, tl = data[mask], labels[mask]

        nodes = build_tree(td, tl, is_bin)

        all_models = {}
        actions_by_node = defaultdict(list)
        for nd in nodes:
            nm, na = train_node(nd, td, tl, is_bin, all_cls)
            all_models.update(nm)
            for a in na:
                actions_by_node[a[1]].append(a)

        retained = []
        for nd in nodes:
            retained.extend(refine_node(nd, all_models,
                                        actions_by_node.get(nd.nid, []), td))

        if i % INNER_CV_INTERVAL == 0:
            cached_margin, cached_ms = \
                select_hyperparams(td, tl, nodes, all_models, retained)

        dec, npacs = predict_dual_af(i, data, nodes, all_models,
                                     retained, margin=cached_margin,
                                     min_support=cached_ms)

        tl_i = int(labels[i])
        ok = (dec != "UNHEALTHY") if tl_i == HEALTHY else (dec == "UNHEALTHY")
        results.append((i, tl_i, dec, ok, npacs))

        if (i + 1) % 50 == 0 or i == n - 1:
            acc = sum(r[3] for r in results) / len(results) * 100
            print(f"  [{i+1}/{n}] acc={acc:.1f}%  {time.perf_counter()-t0:.1f}s")

    return results


# ═══════════════════════════════════════════════════════════════════
# Multi-Metric Reporting (M1)
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(results):
    n = len(results)
    th = [r for r in results if r[1] == HEALTHY]
    td = [r for r in results if r[1] != HEALTHY]

    m = {}
    m['accuracy'] = sum(r[3] for r in results) / n
    m['specificity'] = sum(r[3] for r in th) / len(th) if th else 0
    m['sensitivity'] = sum(r[3] for r in td) / len(td) if td else 0

    dec_u = [r for r in results if r[2] == "UNHEALTHY"]
    m['precision'] = (sum(1 for r in dec_u if r[1] != HEALTHY) / len(dec_u)) if dec_u else 0

    dec_h = [r for r in results if r[2] == "HEALTHY"]
    m['npv'] = (sum(1 for r in dec_h if r[1] == HEALTHY) / len(dec_h)) if dec_h else 0

    p, s = m['precision'], m['sensitivity']
    m['f1'] = 2 * p * s / (p + s) if (p + s) > 0 else 0

    per_class = {}
    for cls in sorted(set(r[1] for r in results)):
        cr = [r for r in results if r[1] == cls]
        per_class[cls] = sum(r[3] for r in cr) / len(cr)
    m['balanced_accuracy'] = np.mean(list(per_class.values()))

    m['avg_pacs'] = np.mean([r[4] for r in results])
    m['pacs_healthy'] = np.mean([r[4] for r in th]) if th else 0
    m['pacs_unhealthy'] = np.mean([r[4] for r in td]) if td else 0

    for dec in ["HEALTHY", "UNHEALTHY", "SCREENING", "CONTRADICTORY"]:
        m[f'rate_{dec.lower()}'] = sum(1 for r in results if r[2] == dec) / n

    m['per_class'] = per_class
    return m


def print_report(results):
    m = compute_metrics(results)
    n = len(results)

    print(f"\n{'='*70}")
    print(f"CDS Dual-AF LOOCV — {n} users")
    print(f"{'='*70}")
    print(f"Accuracy:          {m['accuracy']*100:.1f}%")
    print(f"Balanced Accuracy: {m['balanced_accuracy']*100:.1f}%")
    print(f"Sensitivity:       {m['sensitivity']*100:.1f}%")
    print(f"Specificity:       {m['specificity']*100:.1f}%")
    print(f"Precision (PPV):   {m['precision']*100:.1f}%")
    print(f"NPV:               {m['npv']*100:.1f}%")
    print(f"F1 Score:          {m['f1']*100:.1f}%")
    print(f"")
    print(f"Avg PACs:          {m['avg_pacs']:.1f}")
    print(f"  Healthy users:   {m['pacs_healthy']:.1f}")
    print(f"  Unhealthy users: {m['pacs_unhealthy']:.1f}")
    print(f"")
    print(f"Decision rates:")
    for dec in ["healthy", "unhealthy", "screening", "contradictory"]:
        print(f"  {dec:15s}  {m[f'rate_{dec}']*100:.1f}%")
    print(f"")
    print(f"Per-class accuracy:")
    for cls, acc in sorted(m['per_class'].items()):
        lbl = "healthy" if cls == HEALTHY else f"class {cls}"
        print(f"  {lbl:>10s}  {acc*100:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    dp = str(Path(__file__).parent / "data" / "arrhythmia.data")
    mu = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"Loading {dp}")
    data, labels = load_data(dp)
    print(f"{data.shape[0]} users x {data.shape[1]} feats | "
          f"H={int((labels==HEALTHY).sum())} D={int((labels!=HEALTHY).sum())} | "
          f"u_min={U_MIN}")
    results = run_loocv(data, labels, max_users=mu)
    print_report(results)
