"""CDS — Winner-Take-All per bin.

For each feature, only the class with the HIGHEST posterior in the user's
bin gets evidence. This prevents noise from spreading to all 13 classes.

Binary feature selection (proven). Per-class prediction with WTA evidence.
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

# Import training/refinement from binary-selection variant
from cds_perclass_binarysel import train_node, refine_node


def predict_wta(uid, data, nodes, models, retained, all_cls,
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

                n_used += 1

                # Winner-Take-All: find the class with highest posterior in this bin
                # and only give evidence to that class
                posteriors = mo.posterior[bin_idx]
                priors = mo.prevalence

                # Find class with biggest positive shift from prior
                shifts = posteriors - priors
                best_ci = int(np.argmax(shifts))
                best_shift = shifts[best_ci]

                if best_shift > 0.01:
                    # Give positive evidence only to the winner
                    af[best_ci] += best_shift

                    # Also penalize the class with the biggest NEGATIVE shift
                    worst_ci = int(np.argmin(shifts))
                    worst_shift = shifts[worst_ci]
                    if worst_shift < -0.01:
                        af[worst_ci] += worst_shift

                if trace is not None:
                    feat_traces.append({
                        'node': nd.nid, 'feat': int(f), 'val': round(float(v), 4),
                        'bin': int(bin_idx), 'n_bins': int(mo.n_bins),
                        'pop': int(mo.bin_counts[bin_idx]),
                        'winner': int(all_cls[best_ci]),
                        'winner_shift': round(float(best_shift), 4),
                        'posteriors': {int(all_cls[ci]): round(float(posteriors[ci]), 4)
                                       for ci in range(len(all_cls))},
                    })

    if trace is not None:
        trace['features'] = feat_traces
        trace['af_final'] = {int(all_cls[ci]): round(float(af.get(ci, 0)), 4)
                             for ci in range(len(all_cls))}
        trace['n_features_used'] = n_used

    if not af:
        return HEALTHY, dict(af), n_used

    best_ci = max(af, key=af.get)
    return all_cls[best_ci], dict(af), n_used


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

        pred, af_dict, n_used = predict_wta(
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
    print(f"CDS WTA LOOCV — {n} users")
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
        trace_path = str(Path(__file__).parent / "output" / "loocv_trace_wta.json")
    print(f"Loading {dp}")
    data, labels = load_data(dp)
    print(f"{data.shape[0]} users x {data.shape[1]} feats")
    results = run_loocv(data, labels, max_users=mu, trace_file=trace_path)
    print_report(results)
