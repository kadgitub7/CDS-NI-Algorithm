"""Shared utilities for CDS incremental analysis.

Data loading, tree building, evaluation harness, and helper functions
used by all variant files.
"""
import numpy as np
from collections import defaultdict
from pathlib import Path
import time

DATA_PATH = str(Path(__file__).resolve().parent.parent / "data" / "arrhythmia.data")
U_MIN = 200
HEALTHY = 1
N_FEAT = 279
SEX_FEAT = 1
SEEDS = [13, 34, 55, 69, 76]


def load_data(path=None, remove_classes=None):
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
    remap = {1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,14:11,15:12,16:13}
    labels = np.array([remap.get(l, l) for l in labels], dtype=int)
    if remove_classes:
        keep = np.array([l not in remove_classes for l in labels])
        data, labels = data[keep], labels[keep]
    return data, labels


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


def _fast_abs_corr(x, y):
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    num = (dx * dy).sum()
    den2 = (dx * dx).sum() * (dy * dy).sum()
    return abs(num / np.sqrt(den2)) if den2 > 0 else 0.0


def stats(results):
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    correct = sum(r[3] for r in results)
    th = [r for r in results if r[1] == HEALTHY]
    td = [r for r in results if r[1] != HEALTHY]
    spec = sum(r[3] for r in th) / len(th) if th else 0
    sens = sum(1 for r in td if r[2] != HEALTHY) / len(td) if td else 0
    ba_cls = {}
    for cls in sorted(set(r[1] for r in results)):
        cr = [r for r in results if r[1] == cls]
        ba_cls[cls] = sum(r[3] for r in cr) / len(cr) if cr else 0
    ba = np.mean(list(ba_cls.values())) if ba_cls else 0.0
    return correct / n, spec, sens, ba


def binary_acc(results):
    if not results:
        return 0.0
    correct = sum(1 for _, t, p, _ in results
                  if (t == HEALTHY) == (p == HEALTHY))
    return correct / len(results)


def run_10fold(data, labels, is_bin, seed, train_fn, predict_fn):
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
            pred, _ = predict_fn(uid, data, nodes, all_cls, train_result)
            results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def run_split(data, labels, is_bin, seed, train_frac, train_fn, predict_fn):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]
    td, tl = data[train_idx], labels[train_idx]
    all_cls = sorted(set(labels))
    nodes = build_tree(td, tl, is_bin)
    train_result = train_fn(nodes, td, tl, is_bin, all_cls)
    results = []
    for uid in test_idx:
        true_cls = int(labels[uid])
        pred, _ = predict_fn(uid, data, nodes, all_cls, train_result)
        results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def run_all_evaluations(data, labels, is_bin, train_fn, predict_fn,
                        seeds=None, label=""):
    seeds = seeds or SEEDS
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    all_results = {
        '10fold': [], '9010_multi': [], '9010_binary': [], '6040': []
    }

    for seed in seeds:
        print(f"\n  Seed {seed}:", flush=True)

        t0 = time.time()
        r10 = run_10fold(data, labels, is_bin, seed, train_fn, predict_fn)
        acc10, spec10, sens10, ba10 = stats(r10)
        all_results['10fold'].append(acc10)
        print(f"    10-fold CV:       {100*acc10:.1f}%  (spec={100*spec10:.1f}%, sens={100*sens10:.1f}%)")

        r90 = run_split(data, labels, is_bin, seed, 0.9, train_fn, predict_fn)
        acc90, spec90, sens90, ba90 = stats(r90)
        all_results['9010_multi'].append(acc90)
        ba90_val = binary_acc(r90)
        all_results['9010_binary'].append(ba90_val)
        print(f"    90/10 multiclass: {100*acc90:.1f}%  binary: {100*ba90_val:.1f}%")

        r60 = run_split(data, labels, is_bin, seed, 0.6, train_fn, predict_fn)
        acc60, spec60, sens60, ba60 = stats(r60)
        all_results['6040'].append(acc60)
        print(f"    60/40 multiclass: {100*acc60:.1f}%")

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.1f}s")

    print(f"\n  {'SUMMARY':^60}")
    print(f"  {'-'*60}")
    for key, label_str in [('10fold', '10-fold CV'),
                           ('9010_multi', '90/10 multiclass'),
                           ('9010_binary', '90/10 binary'),
                           ('6040', '60/40 multiclass')]:
        vals = all_results[key]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"  {label_str:>20s}: {100*mean_val:.1f}% +/- {100*std_val:.1f}%  "
              f"(range: {100*min(vals):.1f}%-{100*max(vals):.1f}%)")
    print(f"{'='*70}\n")

    return all_results
