"""Variant: Binary feature selection + per-class prediction.
Uses the proven binary _centered_power for feature selection,
but predicts actual class via per-class evidence accumulation."""
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

# Import shared components
from cds_perclass import (load_data, classify_features, _hdist, Node,
                           build_tree, FeatureModel, _fast_abs_corr,
                           _route_user, compute_metrics, print_report)


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

        decisiveness = np.zeros(nb)
        for b in range(nb):
            if bin_counts[b] >= MIN_SUPPORT:
                shifts = np.abs(post[b] - prev) / np.maximum(prev, MIN_PRIOR)
                decisiveness[b] = shifts.max()

        models[(node.nid, f)] = FeatureModel(post, prev, nb, edges, bin_counts,
                                              decisiveness)

        # BINARY action filter (proven to work)
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


def _centered_power(node, models, action):
    """Binary scoring — proven to select good features."""
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


def predict_perclass(uid, data, nodes, models, retained, all_cls,
                     min_support=MIN_SUPPORT, trace=None):
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
                    continue

                for ci in range(len(all_cls)):
                    p_c = mo.posterior[bin_idx, ci]
                    prior_c = mo.prevalence[ci]
                    shift = p_c - prior_c
                    if abs(shift) < 0.005:
                        continue
                    af[ci] += shift

                n_used += 1

                if trace is not None:
                    feat_traces.append({
                        'node': nd.nid, 'feat': int(f), 'val': round(float(v), 4),
                        'bin': int(bin_idx), 'n_bins': int(mo.n_bins),
                        'pop': int(mo.bin_counts[bin_idx]),
                        'posteriors': {int(all_cls[ci]): round(float(mo.posterior[bin_idx, ci]), 4)
                                       for ci in range(len(all_cls))},
                        'priors': {int(all_cls[ci]): round(float(mo.prevalence[ci]), 4)
                                   for ci in range(len(all_cls))},
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


def run_loocv(data, labels, max_users=None, trace_file=None):
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
            all_traces.append(trace)

        results.append((i, true_cls, pred, ok, n_used))

        if (i + 1) % 50 == 0 or i == n - 1:
            acc = sum(r[3] for r in results) / len(results) * 100
            print(f"  [{i+1}/{n}] acc={acc:.1f}%  {time.perf_counter()-t0:.1f}s")

    if trace_file and all_traces:
        with open(trace_file, 'w') as f:
            json.dump(all_traces, f)
        print(f"  Trace saved to {trace_file}")

    return results


if __name__ == "__main__":
    import sys
    dp = str(Path(__file__).parent / "data" / "arrhythmia.data")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mu = int(args[0]) if args else None

    trace_path = None
    if "--trace" in sys.argv:
        trace_path = str(Path(__file__).parent / "output" / "loocv_trace_binarysel.json")

    print(f"Loading {dp}")
    data, labels = load_data(dp)
    print(f"{data.shape[0]} users x {data.shape[1]} feats | "
          f"H={int((labels==HEALTHY).sum())} D={int((labels!=HEALTHY).sum())}")
    results = run_loocv(data, labels, max_users=mu, trace_file=trace_path)
    print_report(results)
