"""CDS Arrhythmia Classifier — Per-Class Evidence with Bin Decisiveness."""
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


class FeatureModel:
    __slots__ = ('posterior', 'prevalence', 'n_bins', 'edges', 'bin_counts',
                 'decisiveness')

    def __init__(self, posterior, prevalence, n_bins, edges, bin_counts,
                 decisiveness):
        self.posterior = posterior
        self.prevalence = prevalence
        self.n_bins = n_bins
        self.edges = edges
        self.bin_counts = bin_counts
        self.decisiveness = decisiveness


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

        # Per-bin decisiveness: how much does this bin shift ANY class from prior?
        # decisiveness[b] = max over classes of |P(c|bin) - prior_c| / max(prior_c, MIN_PRIOR)
        decisiveness = np.zeros(nb)
        for b in range(nb):
            if bin_counts[b] >= MIN_SUPPORT:
                shifts = np.abs(post[b] - prev) / np.maximum(prev, MIN_PRIOR)
                decisiveness[b] = shifts.max()

        models[(node.nid, f)] = FeatureModel(post, prev, nb, edges, bin_counts,
                                              decisiveness)

        # Feature passes action filter if ANY bin is decisive for ANY class
        max_dec = decisiveness.max() if nb > 0 else 0.0
        if max_dec > 0.05:
            actions.append((f, node.nid, max_dec))

    return models, actions


def _fast_abs_corr(x, y):
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    num = (dx * dy).sum()
    den2 = (dx * dx).sum() * (dy * dy).sum()
    if den2 <= 0:
        return 0.0
    return abs(num / np.sqrt(den2))


def _perclass_score(node, models, action):
    """Score a feature by its best per-class bin decisiveness."""
    f = action[0]
    mo = models.get((node.nid, f))
    if not mo:
        return 0.0
    return mo.decisiveness.max()


def refine_node(node, models, node_actions, data):
    scored = [(a, _perclass_score(node, models, a)) for a in node_actions]
    scored = [(a, s) for a, s in scored if s > 0.05]
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


def predict_perclass(uid, data, nodes, models, retained, all_cls,
                     min_support=MIN_SUPPORT, trace=None):
    """Per-class prediction: accumulate evidence weighted by bin decisiveness.
    Returns (predicted_class, af_dict, n_features_used)."""
    af = defaultdict(float)
    n_used = 0

    lvl_nodes = _route_user(uid, data, nodes)
    feat_traces = []

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

                if mo.bin_counts[bin_idx] < min_support:
                    if trace is not None:
                        feat_traces.append({
                            'node': nd.nid, 'feat': int(f), 'val': float(v),
                            'bin': int(bin_idx), 'n_bins': int(mo.n_bins),
                            'pop': int(mo.bin_counts[bin_idx]),
                            'skipped': 'low_support'
                        })
                    continue

                for ci in range(len(all_cls)):
                    p_c = mo.posterior[bin_idx, ci]
                    prior_c = mo.prevalence[ci]
                    shift = p_c - prior_c
                    # Only accumulate if this bin meaningfully shifts this class
                    # Skip near-zero shifts (noise from Laplace smoothing)
                    if abs(shift) < 0.005:
                        continue
                    af[ci] += shift

                n_used += 1

                if trace is not None:
                    posteriors = {int(all_cls[ci]): round(float(mo.posterior[bin_idx, ci]), 4)
                                  for ci in range(len(all_cls))}
                    priors = {int(all_cls[ci]): round(float(mo.prevalence[ci]), 4)
                              for ci in range(len(all_cls))}
                    feat_traces.append({
                        'node': nd.nid, 'feat': int(f), 'val': round(float(v), 4),
                        'bin': int(bin_idx), 'n_bins': int(mo.n_bins),
                        'pop': int(mo.bin_counts[bin_idx]),
                        'decisiveness': round(float(mo.decisiveness[bin_idx]), 4),
                        'posteriors': posteriors,
                        'priors': priors,
                        'evidence_contrib': {
                            int(all_cls[ci]): round(float(
                                mo.posterior[bin_idx, ci] - mo.prevalence[ci]
                            ), 4)
                            for ci in range(len(all_cls))
                        }
                    })

    if trace is not None:
        trace['features'] = feat_traces
        trace['af_final'] = {int(all_cls[ci]): round(float(af.get(ci, 0)), 4)
                             for ci in range(len(all_cls))}
        trace['n_features_used'] = n_used

    if not af:
        return HEALTHY, dict(af), n_used

    best_ci = max(af, key=af.get)
    predicted = all_cls[best_ci]
    return predicted, dict(af), n_used


INNER_CV_INTERVAL = 10


def run_loocv(data, labels, max_users=None, seed=42, trace_file=None):
    n = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    results = []
    t0 = time.perf_counter()

    all_traces = [] if trace_file else None

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

        trace = {} if trace_file else None

        pred, af_dict, n_used = predict_perclass(
            i, data, nodes, all_models, retained, all_cls,
            min_support=MIN_SUPPORT, trace=trace
        )

        true_cls = int(labels[i])
        ok = (pred == true_cls)

        if trace_file:
            trace['uid'] = int(i)
            trace['true_class'] = true_cls
            trace['predicted'] = int(pred)
            trace['correct'] = bool(ok)
            trace['retained_feats'] = [(int(a[0]), a[1]) for a in retained]
            trace['node_info'] = {
                nd.nid: {
                    'n_users': int(nd.nu),
                    'hdist': {int(k): int(v) for k, v in nd.hdist.items()}
                }
                for nd in nodes
            }
            all_traces.append(trace)

        results.append((i, true_cls, pred, ok, n_used))

        if (i + 1) % 50 == 0 or i == n - 1:
            acc = sum(r[3] for r in results) / len(results) * 100
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{n}] acc={acc:.1f}%  {elapsed:.1f}s")

    if trace_file and all_traces:
        with open(trace_file, 'w') as f:
            json.dump(all_traces, f)
        print(f"  Trace saved to {trace_file} ({len(all_traces)} users)")

    return results


def compute_metrics(results, all_cls=None):
    n = len(results)
    if all_cls is None:
        all_cls = sorted(set(r[1] for r in results))

    m = {}
    m['accuracy'] = sum(r[3] for r in results) / n

    th = [r for r in results if r[1] == HEALTHY]
    td = [r for r in results if r[1] != HEALTHY]
    m['specificity'] = sum(r[3] for r in th) / len(th) if th else 0
    m['sensitivity'] = sum(1 for r in td if r[2] != HEALTHY) / len(td) if td else 0

    per_class = {}
    for cls in all_cls:
        cr = [r for r in results if r[1] == cls]
        if cr:
            per_class[cls] = sum(r[3] for r in cr) / len(cr)
    m['balanced_accuracy'] = np.mean(list(per_class.values()))
    m['per_class'] = per_class

    # Confusion summary
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r[1]][r[2]] += 1
    m['confusion'] = dict(confusion)

    return m


def print_report(results):
    all_cls = sorted(set(r[1] for r in results) | set(r[2] for r in results))
    m = compute_metrics(results, all_cls)
    n = len(results)

    print(f"\n{'='*70}")
    print(f"CDS Per-Class LOOCV — {n} users")
    print(f"{'='*70}")
    print(f"Accuracy:          {m['accuracy']*100:.1f}%")
    print(f"Balanced Accuracy: {m['balanced_accuracy']*100:.1f}%")
    print(f"Sensitivity:       {m['sensitivity']*100:.1f}%")
    print(f"Specificity:       {m['specificity']*100:.1f}%")
    print()
    print(f"Per-class accuracy:")
    for cls in sorted(m['per_class'].keys()):
        n_cls = sum(1 for r in results if r[1] == cls)
        correct = sum(1 for r in results if r[1] == cls and r[3])
        lbl = "healthy" if cls == HEALTHY else f"class {cls}"
        # Show what wrong predictions were made as
        wrong = defaultdict(int)
        for r in results:
            if r[1] == cls and not r[3]:
                wrong[r[2]] += 1
        wrong_str = ""
        if wrong:
            wrong_parts = sorted(wrong.items(), key=lambda x: -x[1])[:3]
            wrong_str = "  misclassed as: " + ", ".join(
                f"{'H' if k==1 else f'c{k}'}({v})" for k, v in wrong_parts)
        print(f"  {lbl:>10s}  {correct:3d}/{n_cls:3d} = {100*correct/n_cls:5.1f}%{wrong_str}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    dp = str(Path(__file__).parent / "data" / "arrhythmia.data")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mu = int(args[0]) if args else None

    trace_path = None
    if "--trace" in sys.argv:
        trace_path = str(Path(__file__).parent / "output" / "loocv_trace.json")

    print(f"Loading {dp}")
    data, labels = load_data(dp)
    print(f"{data.shape[0]} users x {data.shape[1]} feats | "
          f"H={int((labels==HEALTHY).sum())} D={int((labels!=HEALTHY).sum())} | "
          f"u_min={U_MIN}")
    results = run_loocv(data, labels, max_users=mu, trace_file=trace_path)
    print_report(results)
