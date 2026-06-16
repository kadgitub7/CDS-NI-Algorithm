"""
Continuous Feature Multimodality & GMM Analysis
================================================

For every continuous feature this script:

  1. Runs Hartigan's dip test to detect multimodality.
  2. Plots the KDE so you can visually inspect the distribution.
  3. If a feature is multimodal (dip p-value < ALPHA), fits a GMM with
     k=2 and validates that both clusters have at least MIN_CLUSTER_SIZE
     members.  If either cluster is too small k is forced back to 1
     (unimodal / no split warranted).
  4. For GMM-split features, stores the fitted model parameters so that
     at inference time a test user can be assigned a cluster via a
     deterministic closed-form posterior probability — exactly the same
     maths that sklearn's GMM uses internally (no black-box needed).

The output is:
  • KDE plots  → kde_plots/  (one PNG per feature)
  • results CSV → continuous_feature_analysis.csv
      columns: feature_idx, feature_name, dip_stat, dip_pvalue,
               is_multimodal, gmm_used, gmm_k,
               cluster0_size, cluster1_size, gmm_params_json

Design notes
------------
* MIN_CLUSTER_SIZE = 200 is intentionally aggressive.  With N=452 healthy
  subjects, k=2 forces a minimum expected cluster of ~226 assuming a 50/50
  split.  Any feature that would produce a cluster < 200 members gets
  treated as unimodal.  This ensures the healthy-range logic downstream
  always has enough data to define a robust b_min / b_max within each
  cluster.

* GMM assignment rule (closed form, usable at inference without sklearn):
      For a 1-D GMM with k=2 components, component weights w0/w1,
      means mu0/mu1, and variances var0/var1:

          log_p0 = log(w0) - 0.5*log(2*pi*var0) - (x-mu0)**2 / (2*var0)
          log_p1 = log(w1) - 0.5*log(2*pi*var1) - (x-mu1)**2 / (2*var1)
          cluster = 0 if log_p0 >= log_p1 else 1

      This is identical to GMM.predict() but requires only the five stored
      scalars (w0, w1, mu0, mu1, var0, var1) — no sklearn at inference time.

      The gmm_params_json column stores these scalars so they can be loaded
      and used in BranchDef.contains() or any equivalent closed-form check.
"""

import json
import math
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
import diptest  # pip install diptest

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_PATH      = '/mnt/user-data/uploads/arrhythmia.data'
OUTPUT_CSV     = '/mnt/user-data/outputs/continuous_feature_analysis.csv'
KDE_DIR        = '/mnt/user-data/outputs/kde_plots'

ALPHA            = 0.05   # dip-test significance level for declaring multimodal
MIN_CLUSTER_SIZE = 200    # smallest acceptable GMM cluster; forces k=1 if violated
GMM_K            = 2      # candidate k when multimodality is detected
RANDOM_STATE     = 42

# ── Feature index maps (0-indexed, matching arrhythmia.data column order) ─────

def build_feature_map():
    """
    Returns:
        continuous_idx  : list of 0-indexed column positions for linear features
        feature_names   : dict {col_idx: human-readable name}
    """
    nominal_1indexed = set()
    nominal_1indexed.add(2)  # Sex (feature 2 in 1-indexed)
    for block_start in range(16, 160, 12):
        for offset in range(6, 12):
            nominal_1indexed.add(block_start + offset)

    nominal_0 = set(i - 1 for i in nominal_1indexed)
    continuous_0 = [i for i in range(279) if i not in nominal_0]

    # Build name dict for interpretable features; rest labelled by index
    names = {
        0:  'Age',
        2:  'Height',
        3:  'Weight',
        4:  'QRS_duration',
        5:  'PR_interval',
        6:  'QT_interval',
        7:  'T_interval',
        8:  'P_interval',
        9:  'QRS_angle',
        10: 'T_angle',
        11: 'P_angle',
        12: 'QRST_angle',
        13: 'J_angle',
        14: 'Heart_rate',
    }
    # Channel wave-width features (16-159) and amplitude features (160-278)
    channels = ['DI','DII','DIII','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
    width_labels = ['Q_width','R_width','S_width','Rprime_width','Sprime_width','n_deflections']
    amp_labels   = ['JJ_amp','Q_amp','R_amp','S_amp','Rprime_amp','Sprime_amp',
                    'P_amp','T_amp','QRSA','QRSTA']
    for ci, ch in enumerate(channels):
        block_start = 15 + ci * 12  # 0-indexed
        for li, lbl in enumerate(width_labels):
            idx = block_start + li
            if idx not in nominal_0:
                names[idx] = f'{ch}_{lbl}'
        amp_start = 159 + ci * 10  # 0-indexed
        for li, lbl in enumerate(amp_labels):
            names[amp_start + li] = f'{ch}_{lbl}'

    feature_names = {i: names.get(i, f'feat_{i:03d}') for i in continuous_0}
    return continuous_0, feature_names


# ── GMM closed-form assignment (no sklearn at inference) ──────────────────────

def gmm_assign(x: float, params: dict) -> int:
    """
    Deterministic cluster assignment using stored GMM parameters.
    Returns 0 or 1 (the higher-posterior component).

    params keys: w0, w1, mu0, mu1, var0, var1
    """
    def log_gauss(val, mu, var):
        return -0.5 * math.log(2 * math.pi * var) - (val - mu) ** 2 / (2 * var)

    log_p0 = math.log(params['w0']) + log_gauss(x, params['mu0'], params['var0'])
    log_p1 = math.log(params['w1']) + log_gauss(x, params['mu1'], params['var1'])
    return 0 if log_p0 >= log_p1 else 1


def fit_gmm_and_validate(values: np.ndarray):
    """
    Fits a 2-component GMM.  Returns (params_dict, labels, cluster_sizes)
    if both clusters meet MIN_CLUSTER_SIZE, otherwise returns (None, None, None).
    """
    clean = values[~np.isnan(values)].reshape(-1, 1)
    if len(clean) < MIN_CLUSTER_SIZE * GMM_K:
        return None, None, None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gmm = GaussianMixture(
            n_components=GMM_K,
            covariance_type='full',
            random_state=RANDOM_STATE,
            n_init=5,
        )
        gmm.fit(clean)

    labels = gmm.predict(clean)
    sizes = [int((labels == k).sum()) for k in range(GMM_K)]

    if min(sizes) < MIN_CLUSTER_SIZE:
        return None, None, None  # cluster too small — treat as unimodal

    params = {
        'w0':  float(gmm.weights_[0]),
        'w1':  float(gmm.weights_[1]),
        'mu0': float(gmm.means_[0, 0]),
        'mu1': float(gmm.means_[1, 0]),
        'var0': float(gmm.covariances_[0, 0, 0]),
        'var1': float(gmm.covariances_[1, 0, 0]),
    }
    return params, labels, sizes


# ── KDE plot ──────────────────────────────────────────────────────────────────

def plot_kde(values: np.ndarray, name: str, feat_idx: int,
             is_multimodal: bool, gmm_params: dict | None,
             dip_stat: float, dip_p: float, out_dir: str):
    clean = values[~np.isnan(values)]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(clean, ax=ax, fill=True, alpha=0.3, color='steelblue')

    if gmm_params is not None:
        # Overlay the two GMM component densities
        xs = np.linspace(clean.min(), clean.max(), 400)
        for k in range(GMM_K):
            w   = gmm_params[f'w{k}']
            mu  = gmm_params[f'mu{k}']
            var = gmm_params[f'var{k}']
            ys  = w / np.sqrt(2 * np.pi * var) * np.exp(-(xs - mu) ** 2 / (2 * var))
            ax.plot(xs, ys, lw=2, label=f'GMM component {k}')
        ax.legend(fontsize=8)

    title_flag = '★ MULTIMODAL' if is_multimodal else 'unimodal'
    ax.set_title(f'[{feat_idx}] {name}  |  {title_flag}\n'
                 f'dip={dip_stat:.4f}  p={dip_p:.4f}', fontsize=9)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    plt.tight_layout()
    safe_name = name.replace('/', '_')
    path = os.path.join(out_dir, f'{feat_idx:03d}_{safe_name}.png')
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(KDE_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, header=None, na_values='?')
    X  = df.iloc[:, :279].astype(float)
    y  = df.iloc[:, 279].astype(int)

    continuous_idx, feature_names = build_feature_map()

    records = []
    n_multimodal = 0
    n_gmm_used   = 0

    print(f"Analysing {len(continuous_idx)} continuous features over {len(df)} samples...\n")

    for feat_idx in continuous_idx:
        name   = feature_names[feat_idx]
        values = X.iloc[:, feat_idx].values
        clean  = values[~np.isnan(values)]

        if len(clean) < 20:
            # Too few observations — skip
            records.append(dict(
                feature_idx=feat_idx, feature_name=name,
                dip_stat=np.nan, dip_pvalue=np.nan,
                is_multimodal=False, gmm_used=False, gmm_k=1,
                cluster0_size=len(clean), cluster1_size=0,
                gmm_params_json='',
            ))
            continue

        # 1. Hartigan's dip test
        dip_stat, dip_p = diptest.diptest(clean)
        is_multimodal = dip_p < ALPHA

        if is_multimodal:
            n_multimodal += 1

        # 2. Attempt GMM if multimodal
        gmm_params = None
        cluster_sizes = [len(clean), 0]
        gmm_used = False
        gmm_k    = 1

        if is_multimodal:
            gmm_params, labels, sizes = fit_gmm_and_validate(values)
            if gmm_params is not None:
                gmm_used = True
                gmm_k    = GMM_K
                cluster_sizes = sizes
                n_gmm_used += 1
            else:
                # Multimodal but clusters too small — revert to unimodal
                print(f"  [{feat_idx}] {name}: multimodal but GMM cluster < "
                      f"{MIN_CLUSTER_SIZE} — reverting to unimodal")

        # 3. Plot KDE
        plot_kde(values, name, feat_idx, is_multimodal, gmm_params,
                 dip_stat, dip_p, KDE_DIR)

        records.append(dict(
            feature_idx    = feat_idx,
            feature_name   = name,
            dip_stat       = round(dip_stat, 6),
            dip_pvalue     = round(dip_p, 6),
            is_multimodal  = is_multimodal,
            gmm_used       = gmm_used,
            gmm_k          = gmm_k,
            cluster0_size  = cluster_sizes[0],
            cluster1_size  = cluster_sizes[1],
            gmm_params_json= json.dumps(gmm_params) if gmm_params else '',
        ))

    res = pd.DataFrame(records)
    res.to_csv(OUTPUT_CSV, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Continuous features analysed : {len(continuous_idx)}")
    print(f"Multimodal (dip p < {ALPHA})   : {n_multimodal}")
    print(f"GMM k=2 accepted (both clusters ≥ {MIN_CLUSTER_SIZE}) : {n_gmm_used}")
    print(f"Effectively unimodal (range split only)               : "
          f"{len(continuous_idx) - n_gmm_used}")
    print(f"\nResults CSV : {OUTPUT_CSV}")
    print(f"KDE plots   : {KDE_DIR}/")

    # Print multimodal features table
    multi = res[res['is_multimodal']].copy()
    if len(multi):
        print(f"\n{'─'*60}")
        print("Multimodal features:")
        print(multi[['feature_idx','feature_name','dip_pvalue',
                      'gmm_used','cluster0_size','cluster1_size']].to_string(index=False))

    # ── Inline self-test: verify closed-form assign matches sklearn predict ──
    print(f"\n{'─'*60}")
    print("Self-test: closed-form gmm_assign vs sklearn predict")
    mismatch_total = 0
    for row in res[res['gmm_used']].itertuples():
        params = json.loads(row.gmm_params_json)
        feat_idx = row.feature_idx
        clean = X.iloc[:, feat_idx].dropna().values

        gmm = GaussianMixture(n_components=2, covariance_type='full',
                              random_state=RANDOM_STATE, n_init=5)
        gmm.fit(clean.reshape(-1, 1))
        sklearn_labels = gmm.predict(clean.reshape(-1, 1))

        # Rebuild params from this fresh fit for comparison
        p2 = {
            'w0': float(gmm.weights_[0]), 'w1': float(gmm.weights_[1]),
            'mu0': float(gmm.means_[0,0]), 'mu1': float(gmm.means_[1,0]),
            'var0': float(gmm.covariances_[0,0,0]),
            'var1': float(gmm.covariances_[1,0,0]),
        }
        cf_labels = np.array([gmm_assign(v, p2) for v in clean])

        # Labels may be flipped (GMM component 0/1 is arbitrary) — check both
        match_direct = (cf_labels == sklearn_labels).mean()
        match_flip   = (1 - cf_labels == sklearn_labels).mean()
        match = max(match_direct, match_flip)
        status = 'OK' if match > 0.999 else f'MISMATCH ({match:.3f})'
        print(f"  [{feat_idx}] {row.feature_name}: {status}")
        if match <= 0.999:
            mismatch_total += 1

    if mismatch_total == 0:
        print("  All closed-form assignments agree with sklearn ✓")
    else:
        print(f"  {mismatch_total} features with assignment mismatch — check GMM convergence")


if __name__ == '__main__':
    main()