"""Find features where EW spreads target across many bins but supervised concentrates them.
Uses cds_dualAF.load_data for correct feature indexing.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency
import cds_dualAF as m

data, labels = m.load_data(str(Path(__file__).parent.parent / 'data' / 'arrhythmia.data'))

N_BINS = 6

def supervised_bins(fv, tgt, n_bins=6):
    sv = np.sort(np.unique(fv[~np.isnan(fv)]))
    cands = [(sv[i-1]+sv[i])/2 for i in range(1, len(sv))]
    if not cands:
        return np.linspace(np.nanmin(fv), np.nanmax(fv), n_bins+1)
    edges = [np.nanmin(fv)-0.001, np.nanmax(fv)+0.001]
    for _ in range(n_bins-1):
        best, bs = -1, None
        for c in cands:
            if c in edges: continue
            trial = sorted(edges+[c])
            b = np.digitize(fv, trial[1:-1])
            ct = pd.crosstab(b, tgt)
            if ct.shape[0]<2 or ct.shape[1]<2: continue
            if (ct.values==0).any(): continue
            chi2,_,_,_ = chi2_contingency(ct)
            if chi2>best: best, bs = chi2, c
        if bs is None: break
        edges.append(bs)
        edges.sort()
    return np.array(edges)

def count_target_bins(fv, tgt, edges):
    b = np.digitize(fv[tgt], edges[1:-1])
    return len(np.unique(b))

def max_purity(fv, tgt, edges):
    b = np.digitize(fv, edges[1:-1])
    best = 0
    for i in range(len(edges)-1):
        mask = b==i
        if mask.sum()==0: continue
        p = tgt[mask].sum()/mask.sum()
        if p > best: best = p
    return best

results = []
for cls in [2,3,4,5,6,9,10]:
    tgt = (labels==cls).astype(int)
    nt = tgt.sum()
    if nt < 5: continue
    for fi in range(m.N_FEAT):
        col = data[:,fi]
        v = col[~np.isnan(col)]
        if len(np.unique(v)) < 6: continue
        fr = v.max()-v.min()
        if fr < 1e-6: continue

        valid = ~np.isnan(col)
        fv = col[valid]
        tv = tgt[valid]

        ew = np.linspace(fv.min(), fv.max(), N_BINS+1)
        sv = supervised_bins(fv, tv, N_BINS)

        ew_spread = count_target_bins(fv, tv.astype(bool), ew)
        sv_spread = count_target_bins(fv, tv.astype(bool), sv)
        ew_pur = max_purity(fv, tv, ew)
        sv_pur = max_purity(fv, tv, sv)

        contrast = (ew_spread - sv_spread) + (sv_pur - ew_pur)*5
        if contrast <= 0: continue

        target_vals = fv[tv.astype(bool)]
        sep = abs(target_vals.mean() - fv[~tv.astype(bool)].mean()) / (fv.std() + 1e-9)

        results.append({
            'cls':cls, 'feat':fi, 'nt':nt,
            'ew_spr':ew_spread, 'sv_spr':sv_spread,
            'ew_pur':ew_pur, 'sv_pur':sv_pur,
            'sep':sep, 'contrast':contrast,
        })

results.sort(key=lambda x: x['contrast'], reverse=True)
print(f"{'Cls':>3} {'Feat':>5} {'n_t':>4} {'EW_s':>4} {'SV_s':>4} {'EW_p':>5} {'SV_p':>5} {'Sep':>5} {'Cont':>5}")
print("-"*50)
for r in results[:25]:
    print(f"{r['cls']:>3} {r['feat']:>5} {r['nt']:>4} "
          f"{r['ew_spr']:>4} {r['sv_spr']:>4} "
          f"{r['ew_pur']:>5.2f} {r['sv_pur']:>5.2f} "
          f"{r['sep']:>5.2f} {r['contrast']:>5.2f}")
