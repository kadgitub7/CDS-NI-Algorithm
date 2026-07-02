"""CDS Arrhythmia Classifier — Algorithms 1-4 with LOOCV."""
import math, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np

THRESHOLD = 0.025
U_MIN = math.ceil(5 / THRESHOLD)  # 200
HEALTHY = 1
N_FEAT = 279
FORCE_SEX_BRANCHING = True
SEX_FEAT = 1


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
# Algorithm 1: Decision Tree
# For each feature not in ancestry: expand into children.
# Inline U_MIN check (skip children < 200 users).
# Post-step: prune duplicate user sets at same level.
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

        # Prune duplicate user sets at this level
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
# Algorithm 2: Perceptor & Executive Training
# For each node and feature: compute likelihood P(B̂|h),
# prevalence P(h,f), evidence P(B̂), posterior P(h|B̂),
# healthy range, and executive action weights.
# When no healthy users exist: all bins are outside the
# (nonexistent) healthy range, so r_{o|h} = 1.0.
# ═══════════════════════════════════════════════════════════════════

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

        # Sturges' bins: K = ceil(1 + log2(n))
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

        # Likelihood: P(B̂|h) — (n_bins, n_classes)
        lk = np.zeros((nb, nc))
        for ci, c in enumerate(all_cls):
            cm = (lv == c)
            n_c = cm.sum()
            if n_c == 0:
                pass  # leave as zeros — no data, no contribution
            else:
                lk[:, ci] = np.bincount(ba[cm], minlength=nb).astype(float) / n_c

        # Prevalence: P(h, f) = count(h in node) / node_size — (n_classes,)
        prev = np.array([node.hdist.get(c, 0) / ns for c in all_cls])

        # Evidence: P(B̂) = Σ_h P(B̂|h) · P(h,f) — (n_bins,)
        ev = lk @ prev

        # Posterior: P(h|B̂) = P(B̂|h) · P(h,f) / P(B̂) — (n_bins, n_classes)
        post = np.zeros((nb, nc))
        for b in range(nb):
            if ev[b] > 0:
                post[b] = lk[b] * prev / ev[b]

        # Healthy range
        hm = (lv == HEALTHY)
        has_healthy = hm.sum() > 0
        if has_healthy:
            hv = vv[hm]
            h_min, h_max = float(hv.min()), float(hv.max())
            hb = ba[hm]
            mhb, xhb = int(hb.min()), int(hb.max())
        else:
            # No healthy users: any value is outside the (nonexistent) range
            h_min, h_max = np.inf, -np.inf
            mhb, xhb = nb, -1

        models[(node.nid, f)] = FeatureModel(
            lk, prev, ev, post, h_min, h_max, nb, edges, mhb, xhb)

        # Executive action library
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


# ═══════════════════════════════════════════════════════════════════
# Algorithm 3: Refinement (Greedy Set-Cover)
# Per-node: sort actions by r_o descending, remove 0-weight.
# Outer loop: disease class (reset newinf per h).
# Inner loop: features sorted by r_o — keep if new users caught.
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Algorithm 4: Prediction
# Initial random action from root → compute AF.
# BFS through focus levels: AF inherits from parent node,
# resets when moving to sibling. Features reset per node.
# First conclusive answer (HEALTHY/UNHEALTHY) wins.
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


def predict(uid, data, nodes, models, retained, rng):
    anh = defaultdict(list)
    for a in retained:
        anh[(a[1], a[2])].append(a)

    pacs = 0
    af_at = {}

    def phf(nd, h):
        return nd.hdist.get(h, 0) / max(nd.nu, 1)

    def phg(nd):
        return nd.n_diseased / max(nd.nu, 1)

    def afi(p, r, g):
        return (p * r / g) if g > 1e-12 else 0.0

    # Initial random action from root node only
    root = nodes[0]
    root_af = 0.0
    root_used = set()
    root_ok = [a for a in retained if a[1] == root.nid
               and not np.isnan(data[uid, a[0]])]
    if root_ok:
        ia = rng.choice(root_ok)
        m = models.get((root.nid, ia[0]))
        if m:
            root_af = min(1.0, root_af + afi(phf(root, ia[2]), ia[3], phg(root)))
            pacs += 1
            root_used.add(ia[0])
            v = data[uid, ia[0]]
            if v < m.h_min or v > m.h_max:
                return "UNHEALTHY", pacs

    # BFS through focus levels
    lvl_nodes = _route_user(uid, data, nodes)

    for lvl in sorted(lvl_nodes.keys()):
        for nd in lvl_nodes[lvl]:
            # Inherit AF from parent; features reset per node
            if nd.nid == root.nid:
                node_af = root_af
                node_used = set(root_used)
            else:
                parent_nid = nd.parent.nid if nd.parent else root.nid
                node_af = af_at.get(parent_nid, 0.0)
                node_used = set()

            for h in sorted(k for k in nd.hdist if k != HEALTHY and nd.hdist[k] > 0):
                cands = [a for a in anh.get((nd.nid, h), [])
                         if a[0] not in node_used
                         and not np.isnan(data[uid, a[0]])]
                cands.sort(key=lambda a: a[3], reverse=True)

                while cands:
                    best = min(cands,
                               key=lambda a: 1.0 - (node_af + afi(phf(nd, a[2]), a[3], phg(nd))))
                    cands = [a for a in cands if a is not best]
                    node_used.add(best[0])
                    m = models.get((nd.nid, best[0]))
                    if not m:
                        continue
                    pacs += 1
                    node_af = min(1.0, node_af + afi(phf(nd, best[2]), best[3], phg(nd)))
                    v = data[uid, best[0]]
                    if v < m.h_min or v > m.h_max:
                        return "UNHEALTHY", pacs
                    if 1.0 - node_af <= THRESHOLD:
                        return "HEALTHY", pacs

            af_at[nd.nid] = node_af

    return "SCREENING", pacs


# ═══════════════════════════════════════════════════════════════════
# LOOCV — per-fold retraining, train ALL nodes
# ═══════════════════════════════════════════════════════════════════

def run_loocv(data, labels, max_users=None, seed=42):
    n = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    rng = random.Random(seed)
    results = []
    t0 = time.perf_counter()

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

        dec, npacs = predict(i, data, nodes, all_models, retained, rng)
        tl_i = int(labels[i])
        ok = (dec != "UNHEALTHY") if tl_i == HEALTHY else (dec == "UNHEALTHY")
        results.append((i, tl_i, dec, ok, npacs))

        if (i + 1) % 50 == 0 or i == n - 1:
            acc = sum(r[3] for r in results) / len(results) * 100
            print(f"  [{i+1}/{n}] acc={acc:.1f}%  {time.perf_counter()-t0:.1f}s")

    return results


def print_report(results):
    n = len(results)
    nc = sum(r[3] for r in results)
    th = [r for r in results if r[1] == HEALTHY]
    td = [r for r in results if r[1] != HEALTHY]
    spec = sum(r[3] for r in th) / len(th) * 100 if th else 0
    sens = sum(r[3] for r in td) / len(td) * 100 if td else 0
    dc = defaultdict(int)
    for r in results:
        dc[r[2]] += 1

    print(f"\n{'='*60}")
    print(f"CDS LOOCV — {n} users")
    print(f"{'='*60}")
    print(f"Accuracy:    {nc/n*100:.1f}%  ({nc}/{n})")
    print(f"Specificity: {spec:.1f}%  ({len(th)} healthy)")
    print(f"Sensitivity: {sens:.1f}%  ({len(td)} diseased)")
    print(f"Decisions:   {dict(dc)}")
    print(f"\nPer-class:")
    print(f"  {'Class':>8s} {'N':>4s} {'OK':>4s} {'Acc':>5s}  Decisions")
    for cls in sorted(set(r[1] for r in results)):
        cr = [r for r in results if r[1] == cls]
        c = sum(r[3] for r in cr)
        cd = defaultdict(int)
        for r in cr:
            cd[r[2]] += 1
        lbl = "healthy" if cls == HEALTHY else f"h={cls}"
        print(f"  {lbl:>8s} {len(cr):4d} {c:4d} {c/len(cr)*100:5.1f}  {dict(cd)}")
    print(f"\nAvg PACs: {np.mean([r[4] for r in results]):.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    dp = str(Path(__file__).parent / "data" / "arrhythmia.data")
    mu = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"Loading {dp}")
    data, labels = load_data(dp)
    print(f"{data.shape[0]} users x {data.shape[1]} feats | "
          f"H={int((labels==HEALTHY).sum())} D={int((labels!=HEALTHY).sum())} | "
          f"threshold={THRESHOLD} u_min={U_MIN}")
    results = run_loocv(data, labels, max_users=mu)
    print_report(results)
