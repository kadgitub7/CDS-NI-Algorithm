"""Find features where supervised binning uses FEWER bins than EW max,
AND where the visual contrast is clear (target is distinct from rest)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path
import cds_dualAF as m

data, labels = m.load_data(str(Path(__file__).parent.parent / 'data' / 'arrhythmia.data'))

MAX_BINS = 6
RARE = {4, 5, 9}

def supervised_bin_edges(vv, is_target, max_bins, min_support):
    n = len(vv)
    vmin, vmax = float(vv.min()), float(vv.max())
    if vmin == vmax or n < 2 * min_support:
        return np.array([vmin - 0.5, vmax + 0.5])
    sort_idx = np.argsort(vv)
    sv, st = vv[sort_idx], is_target[sort_idx].astype(float)
    edges = [vmin, vmax]
    for _ in range(max_bins - 1):
        best_gain, best_split, best_seg = 0.0, None, None
        for seg_i in range(len(edges) - 1):
            lo, hi = edges[seg_i], edges[seg_i + 1]
            if seg_i == 0: mask = sv <= hi
            elif seg_i == len(edges) - 2: mask = sv > lo
            else: mask = (sv > lo) & (sv <= hi)
            seg_v, seg_t = sv[mask], st[mask]
            n_seg = len(seg_v)
            if n_seg < 2 * min_support: continue
            n_t, n_r = seg_t.sum(), n_seg - seg_t.sum()
            if n_t == 0 or n_r == 0: continue
            cands = np.where(seg_v[:-1] != seg_v[1:])[0]
            if len(cands) == 0: continue
            cum_t = np.cumsum(seg_t)
            for ci in cands:
                nl, nr = ci+1, n_seg-ci-1
                if nl < min_support or nr < min_support: continue
                tl, tr = cum_t[ci], n_t - cum_t[ci]
                rl, rr = nl - tl, nr - tr
                etl, etr = nl*n_t/n_seg, nr*n_t/n_seg
                erl, err = nl*n_r/n_seg, nr*n_r/n_seg
                chi2 = sum((o-e)**2/e for o,e in [(tl,etl),(tr,etr),(rl,erl),(rr,err)] if e>0)
                if chi2 > best_gain:
                    best_gain, best_split, best_seg = chi2, (seg_v[ci]+seg_v[ci+1])/2, seg_i
        if best_split is None or best_gain < 0.5:
            break
        edges.insert(best_seg + 1, best_split)
    edges[0] = vmin - 1e-10
    edges[-1] = vmax + 1e-10
    return np.array(sorted(set(edges)))

results = []
for cls in [2, 3, 4, 5, 6, 9, 10]:
    tgt = labels == cls
    nt = tgt.sum()
    if nt < 5: continue
    ms = 2 if cls in RARE else 3

    for fi in range(m.N_FEAT):
        col = data[:, fi]
        valid = ~np.isnan(col)
        fv, tv = col[valid], tgt[valid].astype(float)
        nv = valid.sum()
        if len(np.unique(fv)) < 6: continue
        fr = fv.max() - fv.min()
        if fr < 1e-6: continue

        max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
        sv_edges = supervised_bin_edges(fv, tv, max_nb, ms)
        sv_nbins = len(sv_edges) - 1

        if sv_nbins >= max_nb:
            continue

        ew_edges = np.linspace(fv.min(), fv.max(), max_nb + 1)

        # Compute max purity for each
        def max_pur(edges):
            b = np.digitize(fv, edges[1:-1])
            best = 0
            for i in range(len(edges)-1):
                mask = b == i
                if mask.sum() > 0:
                    p = tv[mask].sum() / mask.sum()
                    best = max(best, p)
            return best

        ew_pur = max_pur(ew_edges)
        sv_pur = max_pur(sv_edges)

        target_vals = fv[tv.astype(bool)]
        sep = abs(target_vals.mean() - fv[~tv.astype(bool)].mean()) / (fv.std() + 1e-9)

        results.append({
            'cls': cls, 'feat': fi, 'nt': nt,
            'ew_nb': max_nb, 'sv_nb': sv_nbins,
            'ew_pur': ew_pur, 'sv_pur': sv_pur,
            'sep': sep, 'purity_gain': sv_pur - ew_pur,
            'sv_edges': [f'{e:.1f}' for e in sv_edges],
        })

results.sort(key=lambda x: (x['ew_nb'] - x['sv_nb'], x['purity_gain'], x['sep']), reverse=True)

print(f"Features where supervised uses FEWER bins (sorted by bin reduction + purity gain):")
print(f"{'Cls':>3} {'Feat':>5} {'n_t':>4} {'EW_b':>4} {'SV_b':>4} {'EW_p':>5} {'SV_p':>5} {'Sep':>5}  SV edges")
print("-" * 85)
for r in results[:30]:
    print(f"{r['cls']:>3} {r['feat']:>5} {r['nt']:>4} "
          f"{r['ew_nb']:>4} {r['sv_nb']:>4} "
          f"{r['ew_pur']:>5.2f} {r['sv_pur']:>5.2f} "
          f"{r['sep']:>5.2f}  {r['sv_edges']}")
