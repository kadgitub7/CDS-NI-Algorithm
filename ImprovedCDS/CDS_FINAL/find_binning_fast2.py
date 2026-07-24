"""Fast: find features where EW bins split the target cluster badly.
Uses cds_dualAF for correct feature indexing.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path
import cds_dualAF as m

data, labels = m.load_data(str(Path(__file__).parent.parent / 'data' / 'arrhythmia.data'))
N_BINS = 6

results = []
for cls in [2, 3, 4, 5, 6, 9, 10]:
    tgt = labels == cls
    nt = tgt.sum()
    if nt < 5:
        continue
    for fi in range(m.N_FEAT):
        col = data[:, fi]
        valid = ~np.isnan(col)
        fv = col[valid]
        tv = tgt[valid]
        if len(np.unique(fv)) < 6:
            continue
        fr = fv.max() - fv.min()
        if fr < 1e-6:
            continue

        target_vals = fv[tv]
        rest_vals = fv[~tv]

        # Target cluster tightness
        t_iqr = np.percentile(target_vals, 75) - np.percentile(target_vals, 25)
        cluster_ratio = t_iqr / fr

        # Separation: how distinct is target from rest
        sep = abs(target_vals.mean() - rest_vals.mean()) / (fv.std() + 1e-9)

        # EW bins
        ew_edges = np.linspace(fv.min(), fv.max(), N_BINS + 1)
        t_bins = np.digitize(target_vals, ew_edges[1:-1])
        ew_spread = len(np.unique(t_bins))

        # How many EW edges fall inside the target [p25, p75] range?
        p25, p75 = np.percentile(target_vals, 25), np.percentile(target_vals, 75)
        edges_in_iqr = sum(1 for e in ew_edges[1:-1] if p25 < e < p75)

        # Max EW purity
        ew_max_pur = 0
        for b in range(N_BINS):
            mask = np.digitize(fv, ew_edges[1:-1]) == b
            if mask.sum() > 0:
                p = tv[mask].sum() / mask.sum()
                ew_max_pur = max(ew_max_pur, p)

        # Could a supervised bin capture target better?
        # Simple check: bin [p10, p90] of target - what's its purity?
        p10, p90 = np.percentile(target_vals, 10), np.percentile(target_vals, 90)
        in_window = (fv >= p10) & (fv <= p90)
        window_purity = tv[in_window].sum() / in_window.sum() if in_window.sum() > 0 else 0
        window_coverage = tv[in_window].sum() / nt

        # Score: want tight cluster (low ratio) + good separation + EW edges cutting through
        # + high window purity (supervised would do well)
        score = (edges_in_iqr * 1.5 +
                 sep * 1.0 +
                 (1 - cluster_ratio) * 1.0 +
                 window_purity * 2.0 +
                 ew_spread * 0.3 -
                 ew_max_pur * 2.0)

        if score > 3 and sep > 0.3:
            results.append({
                'cls': cls, 'feat': fi, 'nt': nt,
                'ew_spr': ew_spread, 'edges_cut': edges_in_iqr,
                'ew_pur': ew_max_pur, 'win_pur': window_purity,
                'cluster': cluster_ratio, 'sep': sep,
                'score': score,
                't_p25': p25, 't_p75': p75,
            })

results.sort(key=lambda x: x['score'], reverse=True)
print(f"{'Cls':>3} {'Feat':>5} {'n_t':>4} {'EW_s':>4} {'Cut':>3} {'EW_p':>5} {'W_p':>5} {'Clst':>5} {'Sep':>5} {'Score':>5}  IQR")
print("-" * 75)
for r in results[:30]:
    print(f"{r['cls']:>3} {r['feat']:>5} {r['nt']:>4} "
          f"{r['ew_spr']:>4} {r['edges_cut']:>3} "
          f"{r['ew_pur']:>5.2f} {r['win_pur']:>5.2f} "
          f"{r['cluster']:>5.2f} {r['sep']:>5.2f} "
          f"{r['score']:>5.2f}  [{r['t_p25']:.0f}, {r['t_p75']:.0f}]")
