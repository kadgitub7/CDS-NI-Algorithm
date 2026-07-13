"""Ablation test — test each structural change individually on 10-fold CV.
Self-contained: copies and modifies key functions rather than monkey-patching.
Uses 3 seeds for faster iteration.
"""
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

from cds_ovr_v3 import (
    load_data, classify_features, build_tree, _route_user,
    _fast_abs_corr, _compute_af, predict_ovr, print_report,
    Node, BinModel, _hdist,
    U_MIN, HEALTHY, N_FEAT, FORCE_SEX_BRANCHING, SEX_FEAT,
    LAPLACE_ALPHA, MIN_SUPPORT, CONF_SUPPORT, RATIO_EPS,
    CORR_THRESHOLD, MAX_BINS, HEALTHY_WEIGHT,
    SUSPICION_HCUT, SUSPICION_OFFSET, REMOVE_CLASSES,
    AGAINST_SCALE, HEALTHY_BAR_CAP, CLASS_THRESHOLDS,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


def train_ovr_node_configurable(node, data, labels, is_bin, target_class,
                                 use_quantile=False):
    """train_ovr_node with configurable binning."""
    models = {}
    actions = []
    nd_data, nd_labels = data[node.uidx], labels[node.uidx]
    ns = node.nu
    n_target = int((nd_labels == target_class).sum())
    if n_target < 1:
        return models, actions
    prior = n_target / ns

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
            if use_quantile:
                edges = np.quantile(vv, np.linspace(0, 1, nb + 1))
                edges[0] = vmin - 1e-10
                edges[-1] = vmax + 1e-10
                for i in range(1, len(edges)):
                    if edges[i] <= edges[i-1]:
                        edges[i] = edges[i-1] + 1e-10
            else:
                edges = np.linspace(vmin, vmax, nb + 1)

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]
        bin_counts = np.bincount(ba, minlength=nb)
        target_counts = np.bincount(ba[lv == target_class], minlength=nb).astype(float)
        p_class = (target_counts + LAPLACE_ALPHA * prior) / (bin_counts + LAPLACE_ALPHA)

        models[(node.nid, f)] = BinModel(
            nb, edges, bin_counts, target_counts, p_class, prior, CONF_SUPPORT)

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


def refine_ovr_node_configurable(node, models, node_actions, data,
                                  use_fisher_select=False,
                                  features_per_class=18):
    """refine_ovr_node with configurable ranking."""
    rank_idx = 3 if use_fisher_select else 2
    scored = [(a, a[rank_idx]) for a in node_actions]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []

    candidate_limit = 3 * features_per_class
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
        if len(kept) >= features_per_class:
            break
    return kept


def train_all_configurable(nodes, data, labels, is_bin, all_cls,
                            use_quantile=False, use_fisher_select=False,
                            features_per_class=18):
    class_models = {}
    class_retained = {}
    for cls in all_cls:
        cls_models = {}
        cls_actions_by_node = defaultdict(list)
        for nd in nodes:
            nm, na = train_ovr_node_configurable(
                nd, data, labels, is_bin, cls, use_quantile=use_quantile)
            cls_models.update(nm)
            for a in na:
                cls_actions_by_node[a[1]].append(a)
        cls_ret = []
        for nd in nodes:
            cls_ret.extend(
                refine_ovr_node_configurable(
                    nd, cls_models,
                    cls_actions_by_node.get(nd.nid, []), data,
                    use_fisher_select=use_fisher_select,
                    features_per_class=features_per_class))
        class_models[cls] = cls_models
        class_retained[cls] = cls_ret
    return class_models, class_retained


def run_10fold(data, labels, is_bin, seed,
               use_quantile=False, use_fisher_select=False,
               features_per_class=18):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 10)
    all_cls = sorted(set(labels))
    all_results = []

    for fi in range(10):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]

        nodes = build_tree(td, tl, is_bin)
        class_models, class_retained = train_all_configurable(
            nodes, td, tl, is_bin, all_cls,
            use_quantile=use_quantile,
            use_fisher_select=use_fisher_select,
            features_per_class=features_per_class)

        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict_ovr(
                uid, data, nodes, class_models, class_retained, all_cls)
            ok = (int(pred) == true_cls)
            all_results.append((int(uid), true_cls, int(pred), ok))

    return all_results


def stats(results):
    n = len(results)
    correct = sum(r[3] for r in results)
    acc = correct / n
    th = [r for r in results if r[1] == 1]
    td = [r for r in results if r[1] != 1]
    spec = sum(r[3] for r in th) / len(th) if th else 0
    sens = sum(1 for r in td if r[2] != 1) / len(td) if td else 0
    return acc, spec, sens


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats\n")

    seeds = [13, 20, 27]

    configs = [
        ("A) Baseline (18feat, equal-width)",   dict(use_quantile=False, use_fisher_select=False, features_per_class=18)),
        ("B) Quantile bins only",               dict(use_quantile=True,  use_fisher_select=False, features_per_class=18)),
        ("C) Fisher select only",               dict(use_quantile=False, use_fisher_select=True,  features_per_class=18)),
        ("D) 25 features only",                 dict(use_quantile=False, use_fisher_select=False, features_per_class=25)),
        ("E) 30 features only",                 dict(use_quantile=False, use_fisher_select=False, features_per_class=30)),
        ("F) Quantile + 25 features",           dict(use_quantile=True,  use_fisher_select=False, features_per_class=25)),
        ("G) Quantile + 30 features",           dict(use_quantile=True,  use_fisher_select=False, features_per_class=30)),
        ("H) Fisher + quantile",                dict(use_quantile=True,  use_fisher_select=True,  features_per_class=18)),
        ("I) Fisher + 25 features",             dict(use_quantile=False, use_fisher_select=True,  features_per_class=25)),
        ("J) All three combined",               dict(use_quantile=True,  use_fisher_select=True,  features_per_class=25)),
    ]

    print(f"{'Config':45s}  {'Acc':>6s}  {'Spec':>6s}  {'Sens':>6s}  {'Best':>6s}  {'Time':>6s}")
    print("-" * 90)

    for name, kwargs in configs:
        t0 = time.time()
        best_acc = 0
        accs = []
        for seed in seeds:
            results = run_10fold(data, labels, is_bin, seed, **kwargs)
            acc, spec, sens = stats(results)
            accs.append(acc)
            if acc > best_acc:
                best_acc = acc
                best_spec = spec
                best_sens = sens

        elapsed = time.time() - t0
        mean_acc = np.mean(accs)
        print(f"{name:45s}  {100*mean_acc:5.1f}%  {100*best_spec:5.1f}%  {100*best_sens:5.1f}%  {100*best_acc:5.1f}%  {elapsed:5.1f}s",
              flush=True)
