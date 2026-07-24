"""Check specific features for Class 9 binning comparison."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from pathlib import Path
import cds_dualAF as m

data, labels = m.load_data(str(Path(__file__).parent.parent / 'data' / 'arrhythmia.data'))
MAX_BINS = 6
CLS = 9
MIN_SUPPORT = 2  # class 9 is rare

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
            n_targ = seg_t.sum()
            n_r = n_seg - n_targ
            if n_targ == 0 or n_r == 0: continue
            cands = np.where(seg_v[:-1] != seg_v[1:])[0]
            if len(cands) == 0: continue
            cum_t = np.cumsum(seg_t)
            for ci in cands:
                nl, nr = ci+1, n_seg-ci-1
                if nl < min_support or nr < min_support: continue
                tl, tr = cum_t[ci], n_targ - cum_t[ci]
                rl, rr = nl - tl, nr - tr
                etl, etr = nl*n_targ/n_seg, nr*n_targ/n_seg
                erl, err_ = nl*n_r/n_seg, nr*n_r/n_seg
                chi2 = sum((o-e)**2/e for o,e in [(tl,etl),(tr,etr),(rl,erl),(rr,err_)] if e>0)
                if chi2 > best_gain:
                    best_gain, best_split, best_seg = chi2, (seg_v[ci]+seg_v[ci+1])/2, seg_i
        if best_split is None or best_gain < 0.5:
            break
        edges.insert(best_seg + 1, best_split)
    edges[0] = vmin - 1e-10
    edges[-1] = vmax + 1e-10
    return np.array(sorted(set(edges)))

tgt = labels == CLS
nt = tgt.sum()
print(f"Class {CLS}: {nt} target samples\n")

for fi in [4, 148, 219, 227, 229, 113, 152]:
    col = data[:, fi]
    valid = ~np.isnan(col)
    fv, tv = col[valid], tgt[valid].astype(float)
    nv = valid.sum()
    n_unique = len(np.unique(fv))
    max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)

    target_vals = fv[tv.astype(bool)]
    rest_vals = fv[~tv.astype(bool)]
    sep = abs(target_vals.mean() - rest_vals.mean()) / (fv.std() + 1e-9) if fv.std() > 0 else 0

    print(f"--- Feature {fi} ---")
    print(f"  Valid: {nv}, Unique: {n_unique}, Sturges bins: {max_nb}")
    print(f"  Range: [{fv.min():.1f}, {fv.max():.1f}]")
    print(f"  Target mean: {target_vals.mean():.1f}, Rest mean: {rest_vals.mean():.1f}, Sep: {sep:.2f}")
    print(f"  Target values: {sorted(target_vals)}")

    if n_unique < 2:
        print(f"  SKIP: too few unique values\n")
        continue

    ew_edges = np.linspace(fv.min(), fv.max(), max_nb + 1)
    sv_edges = supervised_bin_edges(fv, tv, max_nb, MIN_SUPPORT)
    n_sv = len(sv_edges) - 1

    print(f"  EW edges ({max_nb} bins): {[f'{e:.1f}' for e in ew_edges]}")
    print(f"  SV edges ({n_sv} bins): {[f'{e:.1f}' for e in sv_edges]}")

    for label, edges in [("EW", ew_edges), ("SV", sv_edges)]:
        bins_arr = np.digitize(fv, edges[1:-1])
        best_pur = 0
        for b in range(len(edges)-1):
            mask = bins_arr == b
            n_t = int(tv[mask].sum())
            n_tot = mask.sum()
            if n_tot > 0:
                pur = n_t / n_tot
                best_pur = max(best_pur, pur)
                if n_t > 0:
                    print(f"    {label} bin [{edges[b]:.1f}, {edges[b+1]:.1f}]: "
                          f"{n_t}/{n_tot} target ({pur*100:.1f}%)")
        print(f"    {label} best purity: {best_pur:.2f}")

    # How many EW bins contain target?
    t_ew_bins = np.digitize(target_vals, ew_edges[1:-1])
    t_sv_bins = np.digitize(target_vals, sv_edges[1:-1])
    print(f"  Target spread: EW={len(np.unique(t_ew_bins))} bins, SV={len(np.unique(t_sv_bins))} bins")
    print()
