"""00 — Original CDS algorithm (binary classification) adapted for multiclass evaluation.

This is the unmodified base algorithm from cds.py, using:
- Bayesian likelihood/posterior model (FeatureModel)
- Sturges' uniform binning
- Set-cover refinement
- Binary AF prediction (HEALTHY/UNHEALTHY/SCREENING)

Adapted to output specific disease classes by tracking which class
contributed the strongest evidence for the UNHEALTHY decision.
"""
import numpy as np
from collections import defaultdict
from _common import (
    load_data, classify_features, build_tree, _route_user, _hdist,
    run_all_evaluations, HEALTHY, N_FEAT, SEEDS
)


class FeatureModel:
    __slots__ = ('likelihood', 'prevalence', 'evidence', 'posterior',
                 'h_min', 'h_max', 'n_bins', 'edges', 'min_hbin', 'max_hbin')
    def __init__(self, likelihood, prevalence, evidence, posterior,
                 h_min, h_max, n_bins, edges, min_hbin, max_hbin):
        self.likelihood = likelihood
        self.prevalence = prevalence
        self.evidence = evidence
        self.posterior = posterior
        self.h_min = h_min
        self.h_max = h_max
        self.n_bins = n_bins
        self.edges = edges
        self.min_hbin = min_hbin
        self.max_hbin = max_hbin


def train_node(node, data, labels, is_bin, all_cls):
    nc = len(all_cls)
    models, actions = {}, []
    nd_data, nd_labels = data[node.uidx], labels[node.uidx]
    ns = node.nu

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
            nb = max(2, int(np.ceil(1 + np.log2(nv))))
            edges = np.linspace(vmin, vmax, nb + 1)

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]

        lk = np.zeros((nb, nc))
        for ci, c in enumerate(all_cls):
            cm = (lv == c)
            n_c = cm.sum()
            if n_c > 0:
                lk[:, ci] = np.bincount(ba[cm], minlength=nb).astype(float) / n_c

        prev = np.array([node.hdist.get(c, 0) / ns for c in all_cls])
        ev = lk @ prev
        post = np.zeros((nb, nc))
        for b in range(nb):
            if ev[b] > 0:
                post[b] = lk[b] * prev / ev[b]

        hm = (lv == HEALTHY)
        has_healthy = hm.sum() > 0
        if has_healthy:
            hv = vv[hm]
            h_min, h_max = float(hv.min()), float(hv.max())
            hb = ba[hm]
            mhb, xhb = int(hb.min()), int(hb.max())
        else:
            h_min, h_max = np.inf, -np.inf
            mhb, xhb = nb, -1

        models[(node.nid, f)] = FeatureModel(
            lk, prev, ev, post, h_min, h_max, nb, edges, mhb, xhb)

        for ci, c in enumerate(all_cls):
            if c == HEALTHY or prev[ci] <= 0:
                continue
            if not has_healthy:
                r = 1.0
            else:
                r = ((lk[:mhb, ci].sum() if mhb > 0 else 0.0) +
                     (lk[xhb + 1:, ci].sum() if xhb < nb - 1 else 0.0))
            if r > 0:
                actions.append((f, node.nid, c, r))

    return models, actions


def refine_node(node, models, node_actions, data):
    na = sorted([a for a in node_actions if a[3] > 0],
                key=lambda a: a[3], reverse=True)
    if not na:
        return []
    kept = []
    newinf = np.zeros(node.nu, dtype=bool)
    buf = 0
    for h in sorted(set(a[2] for a in na)):
        for a in [x for x in na if x[2] == h]:
            m = models.get((node.nid, a[0]))
            if not m:
                continue
            raw = data[node.uidx, a[0]]
            vm = ~np.isnan(raw)
            newinf |= vm & ((raw < m.h_min) | (raw > m.h_max))
            s = int(newinf.sum())
            if s > buf:
                buf = s
                kept.append(a)
    return kept


def train_base(nodes, data, labels, is_bin, all_cls):
    all_models = {}
    actions_by_node = defaultdict(list)
    for nd in nodes:
        nm, na = train_node(nd, data, labels, is_bin, all_cls)
        all_models.update(nm)
        for a in na:
            actions_by_node[a[1]].append(a)
    retained = []
    for nd in nodes:
        retained.extend(refine_node(nd, all_models,
                                    actions_by_node.get(nd.nid, []), data))
    return all_models, retained


def predict_base(uid, data, nodes, all_cls, train_result):
    models, retained = train_result

    class_evidence = defaultdict(float)
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
                if v < m.h_min or v > m.h_max:
                    class_evidence[a[2]] += a[3]

    if class_evidence:
        best_cls = max(class_evidence, key=class_evidence.get)
        return int(best_cls), dict(class_evidence)
    return HEALTHY, {}


if __name__ == "__main__":
    data, labels = load_data()
    is_bin = classify_features(data)
    print(f"Loaded {data.shape[0]} patients, classes: {sorted(set(labels))}")
    run_all_evaluations(data, labels, is_bin, train_base, predict_base,
                        seeds=SEEDS, label="00: Original CDS Algorithm (Binary)")
