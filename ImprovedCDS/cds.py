"""
CDS Arrhythmia Classifier — Algorithms 1-4 with LOOCV.
Paper: "Brain-Inspired Intelligence for Real-Time Health Situation Understanding
        in Smart e-Health Home Applications" (IEEE Access, 2019)
"""

import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

THRESHOLD = 0.025
U_MIN = math.ceil(5 / THRESHOLD)  # 200
HEALTHY = 1
N_FEAT = 279
LAPLACE = 1e-6


# ─── Data ────────────────────────────────────────────────────────────────────

def load_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append([float('nan') if v.strip() == '?' else float(v.strip())
                         for v in line.split(",")])
    raw = np.array(rows, dtype=np.float64)
    data, labels = raw[:, :N_FEAT], raw[:, N_FEAT].astype(int)
    remap = {1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,14:11,15:12,16:13}
    return data, np.array([remap.get(l, l) for l in labels], dtype=int)


def classify_features(data):
    is_bin = np.zeros(N_FEAT, dtype=bool)
    for c in range(N_FEAT):
        v = data[:, c]; v = v[~np.isnan(v)]
        if len(v) > 0 and set(np.unique(v)).issubset({0.0, 1.0}):
            is_bin[c] = True
    return is_bin


# ─── Algorithm 1: Decision Tree ─────────────────────────────────────────────

class Node:
    __slots__ = ('nid','lvl','uidx','feats','hdist','bfeat','bvset','blo','bhi','bbin','children')
    def __init__(self, nid, lvl, uidx, feats, hdist):
        self.nid=nid; self.lvl=lvl; self.uidx=uidx; self.feats=feats; self.hdist=hdist
        self.bfeat=self.bvset=self.blo=self.bhi=None; self.bbin=False; self.children=[]
    @property
    def nu(self): return len(self.uidx)
    @property
    def nd(self): return sum(v for k,v in self.hdist.items() if k!=HEALTHY)
    def contains(self, row):
        if self.bfeat is None: return True
        v = row[self.bfeat]
        if np.isnan(v): return False
        if self.bbin: return v in self.bvset
        return self.blo <= v <= self.bhi


def _hdist(labels, idx):
    d = defaultdict(int)
    for u in idx: d[labels[u]] += 1
    return dict(d)


def build_tree(data, labels, is_bin):
    """Build decision tree. Returns list of all nodes."""
    n = data.shape[0]
    root = Node("root", 1, np.arange(n), set(range(N_FEAT)), _hdist(labels, np.arange(n)))
    nodes = [root]; nc = [0]

    def branch(parent):
        for f in sorted(parent.feats):
            col = data[parent.uidx, f]; vm = ~np.isnan(col); vv = col[vm]
            if len(vv) == 0: continue
            if is_bin[f]:
                uq = sorted(set(vv))
                if len(uq) < 2: continue
                parts = []
                for val in uq:
                    cu = parent.uidx[vm & (col == val)]
                    if len(cu) < U_MIN: parts = []; break
                    parts.append((cu, frozenset({val}), None, None, True))
                if len(parts) < 2: continue
            else:
                if float(vv.min()) == float(vv.max()): continue
                med = float(np.median(vv))
                lo_u, hi_u = parent.uidx[vm & (col<=med)], parent.uidx[vm & (col>med)]
                if len(lo_u) < U_MIN or len(hi_u) < U_MIN: continue
                parts = [(lo_u,None,-np.inf,med,False), (hi_u,None,med,np.inf,False)]
            for cu,vs,lo,hi,ib in parts:
                nc[0] += 1
                cf = set(parent.feats)
                if parent.lvl >= 2: cf.discard(f)
                nd = Node(f"L{parent.lvl+1}_f{f}_{nc[0]}", parent.lvl+1, cu, cf, _hdist(labels,cu))
                nd.bfeat,nd.bvset,nd.blo,nd.bhi,nd.bbin = f,vs,lo,hi,ib
                parent.children.append(nd); nodes.append(nd)

    branch(root)
    for nd in list(nodes):
        if nd.lvl == 2: branch(nd)
    return nodes


# ─── Algorithm 2: Perceptor & Executive Training ────────────────────────────
# model = (b_min_healthy, b_max_healthy, n_bins, p_bin_given_h, min_healthy_bin, max_healthy_bin)
# action = (feat_idx, node_id, disease_h, weight)

def train_node(node, data, labels, is_bin, all_cls, N_total):
    """Run Algorithm 2 for one node. Returns (models_dict, actions_list)."""
    nc = len(all_cls); ci_h = all_cls.index(HEALTHY)
    models = {}; actions = []
    nd_data = data[node.uidx]; nd_labels = labels[node.uidx]

    for f in sorted(node.feats):
        col = nd_data[:, f]; vm = ~np.isnan(col); vv = col[vm]; nv = len(vv)
        if nv == 0: continue
        bmin, bmax = float(vv.min()), float(vv.max())

        if is_bin[f]:
            if bmin == bmax: nb, edges = 1, np.array([bmin-.5, bmin+.5])
            else: nb, edges = 2, np.array([-.5, .5, 1.5])
        elif bmin == bmax: nb, edges = 1, np.array([bmin-.5, bmin+.5])
        else:
            nb = max(2, int(np.ceil(1 + np.log2(nv))))
            edges = np.linspace(bmin, bmax, nb+1)

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb-1)
        lv = nd_labels[vm]

        pbh = np.zeros((nb, nc))
        for ci, c in enumerate(all_cls):
            cm = (lv == c); n_c = cm.sum()
            if n_c == 0: pbh[:, ci] = 1.0/nb
            else:
                cnt = np.bincount(ba[cm], minlength=nb).astype(float) + LAPLACE
                pbh[:, ci] = cnt / cnt.sum()

        hm = (lv == HEALTHY)
        if hm.sum() > 0:
            hv = vv[hm]; h_min, h_max = float(hv.min()), float(hv.max())
            hb = ba[hm]; minhb, maxhb = int(hb.min()), int(hb.max())
        else:
            h_min, h_max, minhb, maxhb = bmin, bmax, 0, nb-1

        models[(node.nid, f)] = (h_min, h_max, nb, pbh, minhb, maxhb)

        if hm.sum() == 0: continue
        for ci, c in enumerate(all_cls):
            if c == HEALTHY or node.hdist.get(c, 0) <= 0: continue
            pb = pbh[:minhb, ci].sum() if minhb > 0 else 0.0
            pa = pbh[maxhb+1:, ci].sum() if maxhb < nb-1 else 0.0
            r = pb + pa
            if r > 0: actions.append((f, node.nid, c, r))

    return models, actions


# ─── Algorithm 3: Refinement ────────────────────────────────────────────────

def refine_node(node, models, actions, data):
    """Greedy set-cover for one node. Global buffer/newinf."""
    dcs = sorted(h for h in node.hdist if h != HEALTHY and node.hdist[h] > 0)
    if not dcs: return []
    newinf = np.zeros(node.nu, dtype=bool); buf = 0; kept = []
    acts_h = defaultdict(list)
    for a in actions:
        if a[1] == node.nid: acts_h[a[2]].append(a)

    for h in dcs:
        ha = sorted(acts_h.get(h, []), key=lambda a: a[3], reverse=True)
        for a in ha:
            m = models.get((node.nid, a[0]))
            if not m: continue
            raw = data[node.uidx, a[0]]
            valid = ~np.isnan(raw)
            newinf |= valid & ((raw < m[0]) | (raw > m[1]))
            s = int(newinf.sum())
            if s <= buf: continue
            buf = s; kept.append(a)
    return kept


# ─── Algorithm 4: Prediction ────────────────────────────────────────────────

def predict(uid, data, nodes, models, retained, rng):
    """Predict one user. Returns (decision, n_pacs)."""
    acts_nh = defaultdict(list)
    for a in retained: acts_nh[(a[1], a[2])].append(a)

    af, pacs, consumed = 0.0, 0, set()
    root = nodes[0]

    def phf(nd, h): return nd.hdist.get(h,0) / max(nd.nu,1)
    def phg(nd):
        d = nd.nd; return d/nd.nu if d > 0 and nd.nu > 0 else 1e-9
    def afi(p, r, g): return max(0.0, p*r/g) if g > 1e-12 else 0.0
    def rl(cands, nd, h, afc):
        if not cands: return None
        p, g = phf(nd,h), phg(nd)
        return min(cands, key=lambda a: 1.0-(afi(p,a[3],g)+afc))

    # Init action
    ra = [a for a in retained if a[1]==root.nid and not np.isnan(data[uid,a[0]])]
    h_init = j_init = None
    if ra:
        ia = rng.choice(ra); j_init, h_init = ia[0], ia[2]
        m = models.get((root.nid, j_init))
        if m:
            af = min(1.0, af + afi(phf(root,h_init), ia[3], phg(root)))
            pacs += 1; consumed.add((j_init, h_init))
            if data[uid,j_init] < m[0] or data[uid,j_init] > m[1]:
                return "UNHEALTHY", pacs

    max_lvl = max((nd.lvl for nd in nodes if nd.nid in {k[0] for k in models}), default=1)
    trained = {k[0] for k in models}

    for lvl in range(1, max_lvl+1):
        active = None
        if lvl == 1: active = root
        else:
            for nd in nodes:
                if nd.lvl==lvl and nd.nid in trained and nd.contains(data[uid]):
                    active = nd; break
        if not active: return "SCREENING", pacs

        dcs = sorted(h for h in active.hdist if h!=HEALTHY and active.hdist[h]>0)
        if not dcs: return "HEALTHY", pacs

        for h in dcs:
            cb = [a for a in acts_nh.get((active.nid,h),[]) if (a[0],h) not in consumed]
            if h_init is not None and h==h_init and lvl==1:
                cb = [a for a in cb if a[0]!=j_init]
            cb.sort(key=lambda a: a[3], reverse=True)
            while cb:
                sel = rl(cb, active, h, af)
                if not sel: break
                cb = [a for a in cb if a[0]!=sel[0]]
                v = data[uid, sel[0]]
                if np.isnan(v): continue
                pacs += 1; consumed.add((sel[0], h))
                m = models.get((active.nid, sel[0]))
                if not m: continue
                af = min(1.0, af + afi(phf(active,h), sel[3], phg(active)))
                if v < m[0] or v > m[1]: return "UNHEALTHY", pacs

        if 1.0 - af <= THRESHOLD: return "HEALTHY", pacs
    return "SCREENING", pacs


# ─── LOOCV ───────────────────────────────────────────────────────────────────

def _route_user(uid, data, nodes, trained_nids):
    """Find which nodes the test user would visit (one per level)."""
    visited = [nodes[0]]  # root
    for nd in nodes:
        if nd.lvl > 1 and nd.nid in trained_nids and nd.contains(data[uid]):
            visited.append(nd)
            break
    return visited


def run_loocv(data, labels, max_users=None, seed=42):
    n = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    rng = random.Random(seed)
    results = []; t0 = time.perf_counter()

    for i in range(n):
        mask = np.ones(data.shape[0], dtype=bool); mask[i] = False
        td, tl = data[mask], labels[mask]
        N_train = td.shape[0]

        # Alg 1: build tree
        nodes = build_tree(td, tl, is_bin)

        # Optimization: only train nodes the test user will visit
        # First pass: train root to get its models (needed for routing)
        root_models, root_actions = train_node(nodes[0], td, tl, is_bin, all_cls, N_train)

        # Find which level-2 node the test user belongs to
        all_models = dict(root_models)
        all_actions = list(root_actions)
        trained_nids = {nodes[0].nid}

        for nd in nodes:
            if nd.lvl == 2 and nd.contains(data[i]):
                nm, na = train_node(nd, td, tl, is_bin, all_cls, N_train)
                all_models.update(nm)
                all_actions.extend(na)
                trained_nids.add(nd.nid)
                break

        # Alg 3: refine actions per trained node
        retained = []
        for nd in nodes:
            if nd.nid in trained_nids:
                retained.extend(refine_node(nd, all_models, all_actions, td))

        # Alg 4: predict
        dec, npacs = predict(i, data, nodes, all_models, retained, rng)
        tl_i = int(labels[i])
        ok = (dec != "UNHEALTHY") if tl_i == HEALTHY else (dec == "UNHEALTHY")
        results.append((i, tl_i, dec, ok, npacs))

        if (i+1) % 50 == 0 or i == n-1:
            acc = sum(r[3] for r in results) / len(results) * 100
            print(f"  [{i+1}/{n}] acc={acc:.1f}%  {time.perf_counter()-t0:.1f}s")

    return results


def print_report(results):
    n = len(results); nc = sum(r[3] for r in results)
    th = [r for r in results if r[1]==HEALTHY]
    td = [r for r in results if r[1]!=HEALTHY]
    spec = sum(r[3] for r in th)/len(th)*100 if th else 0
    sens = sum(r[3] for r in td)/len(td)*100 if td else 0
    dc = defaultdict(int)
    for r in results: dc[r[2]] += 1

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
        cr = [r for r in results if r[1]==cls]
        c = sum(r[3] for r in cr); cd = defaultdict(int)
        for r in cr: cd[r[2]] += 1
        lbl = "healthy" if cls==HEALTHY else f"h={cls}"
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
