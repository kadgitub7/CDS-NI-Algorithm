"""CDS Arrhythmia Classifier — Two-Stage: Binary then Per-Class.

Stage 1: Binary healthy/unhealthy (proven at 76.3%)
Stage 2: If unhealthy, which disease class? (per-class evidence among diseases only)

This separates the two decisions that have fundamentally different scales.
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
CORR_THRESHOLD = 0.8
MAX_FEATURES_PER_NODE = 30
MAX_BINS = 6
MIN_PRIOR = 0.10

from cds_perclass import (load_data, classify_features, _hdist, Node,
                           build_tree, FeatureModel, _fast_abs_corr,
                           _route_user, compute_metrics)


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

        # Binary action filter
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


def predict_twostage(uid, data, nodes, models, retained, all_cls,
                     margin=0.0, min_support=MIN_SUPPORT, trace=None):
    """Two-stage prediction:
    Stage 1: Binary healthy/unhealthy using centered evidence (proven formula)
    Stage 2: If unhealthy, argmax over disease classes using per-class posteriors
    """
    binary_af = 0.0  # positive = healthy, negative = unhealthy
    class_af = defaultdict(float)  # disease class accumulators
    n_used = 0

    lvl_nodes = _route_user(uid, data, nodes)
    feat_traces = []

    # Disease class indices (all except healthy which is index 0)
    disease_indices = list(range(1, len(all_cls)))

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

                n_used += 1

                # STAGE 1: Binary evidence (proven formula)
                p_h = mo.posterior[bin_idx, 0]
                prev_h = mo.prevalence[0]
                conf = abs(2.0 * p_h - 1.0)
                binary_af += (p_h - prev_h) * conf * conf

                # STAGE 2: Per-class evidence among diseases only
                # Compute disease-conditional priors and posteriors
                # P(class_d | unhealthy) = P(class_d) / P(unhealthy)
                p_unhealthy = 1.0 - prev_h
                if p_unhealthy > 0.01:
                    for di in disease_indices:
                        p_d = mo.posterior[bin_idx, di]
                        prior_d = mo.prevalence[di]
                        # Normalize within disease space
                        prior_d_given_U = prior_d / p_unhealthy
                        p_d_bin = mo.posterior[bin_idx, di]
                        p_all_disease_bin = 1.0 - mo.posterior[bin_idx, 0]
                        if p_all_disease_bin > 0.01:
                            # P(class_d | bin, unhealthy) = P(class_d | bin) / P(unhealthy | bin)
                            p_d_given_U_bin = p_d_bin / p_all_disease_bin
                        else:
                            p_d_given_U_bin = prior_d_given_U
                        shift = p_d_given_U_bin - prior_d_given_U
                        if abs(shift) > 0.005:
                            class_af[di] += shift

                if trace is not None:
                    posteriors = {int(all_cls[ci]): round(float(mo.posterior[bin_idx, ci]), 4)
                                  for ci in range(len(all_cls))}
                    priors = {int(all_cls[ci]): round(float(mo.prevalence[ci]), 4)
                              for ci in range(len(all_cls))}
                    feat_traces.append({
                        'node': nd.nid, 'feat': int(f), 'val': round(float(v), 4),
                        'bin': int(bin_idx), 'n_bins': int(mo.n_bins),
                        'pop': int(mo.bin_counts[bin_idx]),
                        'binary_ev': round(float((p_h - prev_h) * conf * conf), 4),
                        'posteriors': posteriors,
                        'priors': priors,
                        'disease_ev': {
                            int(all_cls[di]): round(float(class_af.get(di, 0)), 4)
                            for di in disease_indices
                        }
                    })

    if trace is not None:
        trace['features'] = feat_traces
        trace['binary_af'] = round(float(binary_af), 4)
        trace['class_af'] = {int(all_cls[di]): round(float(class_af.get(di, 0)), 4)
                             for di in disease_indices}
        trace['n_features_used'] = n_used

    # DECISION
    if n_used == 0:
        return HEALTHY, {'binary_af': binary_af, 'class_af': dict(class_af)}, n_used

    # Stage 1: healthy or unhealthy?
    if binary_af >= margin:
        return HEALTHY, {'binary_af': binary_af, 'class_af': dict(class_af)}, n_used

    # Stage 2: which disease class?
    if class_af:
        best_di = max(class_af, key=class_af.get)
        return all_cls[best_di], {'binary_af': binary_af, 'class_af': dict(class_af)}, n_used

    return HEALTHY, {'binary_af': binary_af, 'class_af': dict(class_af)}, n_used


INNER_CV_INTERVAL = 10


def select_hyperparams(data, labels, nodes, models, retained, all_cls, seed=42):
    margin_grid = [0.0, 0.01, 0.02, 0.05]
    min_support_grid = [2, 3, 5]

    n = len(labels)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, 3)

    def evaluate(margin, ms):
        correct = 0
        for fold in folds:
            for uid in fold:
                pred, _, _ = predict_twostage(uid, data, nodes, models, retained,
                                               all_cls, margin=margin, min_support=ms)
                if pred == labels[uid]:
                    correct += 1
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

        pred, af_info, n_used = predict_twostage(
            i, data, nodes, all_models, retained, all_cls,
            margin=0.0, min_support=MIN_SUPPORT, trace=trace
        )

        true_cls = int(labels[i])
        ok = (pred == true_cls)

        if trace_file:
            trace['uid'] = int(i)
            trace['true_class'] = true_cls
            trace['predicted'] = int(pred)
            trace['correct'] = bool(ok)
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


def print_report(results):
    all_cls = sorted(set(r[1] for r in results) | set(r[2] for r in results))
    m = compute_metrics(results, all_cls)
    n = len(results)

    print(f"\n{'='*70}")
    print(f"CDS Two-Stage LOOCV — {n} users")
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
        trace_path = str(Path(__file__).parent / "output" / "loocv_trace_twostage.json")

    print(f"Loading {dp}")
    data, labels = load_data(dp)
    print(f"{data.shape[0]} users x {data.shape[1]} feats | "
          f"H={int((labels==HEALTHY).sum())} D={int((labels!=HEALTHY).sum())}")
    results = run_loocv(data, labels, max_users=mu, trace_file=trace_path)
    print_report(results)
