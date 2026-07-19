"""Complete ablation study + statistical significance tests + hyperparameter sensitivity.

Three goals:
  1. Statistical significance: paired Wilcoxon signed-rank tests on ablation results
  2. Full component ablation: remove each mechanism one-at-a-time
  3. Hyperparameter sensitivity sweep: CORR_THRESHOLD, MAX_BINS, FEATURES_PER_CLASS

All evaluations use 10-fold CV across 10 seeds (100 fold evaluations per config).
"""
import sys, os, io, json, time
import numpy as np
from scipy import stats as sp_stats
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

import cds_ovr
from cds_ovr import (
    load_data, classify_features, run_10fold, stats, binary_acc,
    RARE_CLASSES, AGAINST_SCALE_MAP, SEX_FEAT, CORR_THRESHOLD,
    FEATURES_PER_CLASS, MAX_BINS, HEALTHY_WEIGHT, SUSPICION_HCUT,
    SUSPICION_OFFSET, HEALTHY_BAR_CAP, CLASS_THRESHOLDS, N_FEAT,
    LAPLACE_ALPHA, _supervised_bin_edges, build_tree, train, predict,
    _train_ovr_node, _refine_ovr_node, _compute_af, _route_user,
    Node, BinModel, HEALTHY, MIN_SUPPORT_MAP, CONF_SUPPORT_MAP,
    RATIO_EPS
)
from collections import defaultdict

OUT_DIR = os.path.dirname(__file__)
SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]

report_lines = []
results_data = {}


def log(msg=""):
    report_lines.append(msg)
    print(msg, flush=True)


def section(title):
    log(f"\n{'='*80}")
    log(f"  {title}")
    log(f"{'='*80}\n")


def run_baseline(data, labels, is_bin):
    """Run baseline (current W11-01 settings) across all seeds."""
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    return accs


def paired_wilcoxon(baseline_accs, ablated_accs):
    """Paired Wilcoxon signed-rank test. Returns (stat, p_value, effect_size)."""
    diffs = np.array(baseline_accs) - np.array(ablated_accs)
    if np.all(diffs == 0):
        return 0.0, 1.0, 0.0
    try:
        stat, p = sp_stats.wilcoxon(diffs, alternative='two-sided')
    except ValueError:
        return 0.0, 1.0, 0.0
    # Effect size: r = Z / sqrt(N)
    n = len(diffs)
    z = sp_stats.norm.ppf(1 - p/2) if p > 0 else 0
    r = z / np.sqrt(n) if n > 0 else 0
    return float(stat), float(p), float(r)


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION: Equal-width binning (remove supervised chi-squared binning)
# ═══════════════════════════════════════════════════════════════════════════════

def _equal_width_bin_edges(vv, is_target, max_bins, min_support):
    """Sturges' rule equal-width binning (replaces supervised chi-squared)."""
    n = len(vv)
    vmin, vmax = float(vv.min()), float(vv.max())
    if vmin == vmax or n < 2 * min_support:
        return np.array([vmin - 0.5, vmax + 0.5])
    n_bins = min(max(2, int(np.ceil(1 + np.log2(n)))), max_bins)
    edges = np.linspace(vmin - 1e-10, vmax + 1e-10, n_bins + 1)
    return edges


def run_ablation_equal_width(data, labels, is_bin):
    """Replace supervised binning with equal-width."""
    orig_fn = cds_ovr._supervised_bin_edges
    cds_ovr._supervised_bin_edges = _equal_width_bin_edges
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr._supervised_bin_edges = orig_fn
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION: No correlation filtering (keep all top features)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_no_corr_filter(data, labels, is_bin):
    """Disable correlation filtering by setting threshold to 1.0."""
    orig = cds_ovr.CORR_THRESHOLD
    cds_ovr.CORR_THRESHOLD = 1.0  # effectively disables filtering
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr.CORR_THRESHOLD = orig
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION: No healthy bar (fixed thresholds only)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_no_healthy_bar(data, labels, is_bin):
    """Disable healthy bar by setting HEALTHY_WEIGHT to 0 and disabling suspicion."""
    orig_hw = cds_ovr.HEALTHY_WEIGHT
    orig_sh = cds_ovr.SUSPICION_HCUT
    orig_so = cds_ovr.SUSPICION_OFFSET
    orig_cap = cds_ovr.HEALTHY_BAR_CAP
    # Set healthy bar to always be 0 (never overrides thresholds)
    cds_ovr.HEALTHY_WEIGHT = 0.0
    cds_ovr.HEALTHY_BAR_CAP = 0.0
    # Disable suspicion (never lowers thresholds)
    cds_ovr.SUSPICION_HCUT = -1.0
    cds_ovr.SUSPICION_OFFSET = 0.0
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr.HEALTHY_WEIGHT = orig_hw
    cds_ovr.SUSPICION_HCUT = orig_sh
    cds_ovr.SUSPICION_OFFSET = orig_so
    cds_ovr.HEALTHY_BAR_CAP = orig_cap
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION: No Fisher weighting (uniform feature weights)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_no_fisher(data, labels, is_bin):
    """Replace Fisher weighting with uniform weights (fw=1.0 always).
    We monkey-patch _compute_af to ignore fisher_map."""
    orig_fn = cds_ovr._compute_af

    def _compute_af_no_fisher(uid, data, nodes, models, retained, against_scale):
        lvl_nodes = cds_ovr._route_user(uid, data, nodes)
        af_for, af_against = 0.0, 0.0
        n_for, n_against, n_used = 0, 0, 0
        max_for_contrib = 0.0

        for lvl in sorted(lvl_nodes.keys()):
            for nd in lvl_nodes[lvl]:
                for a in retained:
                    if a[1] != nd.nid:
                        continue
                    f = a[0]
                    v = data[uid, f]
                    if np.isnan(v):
                        continue
                    mo = models.get((nd.nid, f))
                    if not mo:
                        continue
                    bin_idx = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                         0, mo.n_bins - 1))
                    bc = mo.bin_counts[bin_idx]
                    if bc < 3:
                        continue
                    p_c = mo.p_class[bin_idx]
                    shift = p_c - mo.prior
                    confidence = min(1.0, bc / 10)
                    fw = 1.0  # uniform weight
                    weighted = abs(shift) * confidence * fw
                    if shift >= 0:
                        af_for += weighted
                        n_for += 1
                        if weighted > max_for_contrib:
                            max_for_contrib = weighted
                    else:
                        af_against += weighted * against_scale
                        n_against += 1
                    n_used += 1

        return af_for, af_against, n_used, n_for, n_against, max_for_contrib

    cds_ovr._compute_af = _compute_af_no_fisher
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr._compute_af = orig_fn
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION: No against_scale differentiation (uniform 0.8)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_uniform_against(data, labels, is_bin):
    """Use uniform against_scale=0.8 for all classes."""
    orig = dict(AGAINST_SCALE_MAP)
    for cls in RARE_CLASSES:
        cds_ovr.AGAINST_SCALE_MAP[cls] = 0.8
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    for cls in RARE_CLASSES:
        cds_ovr.AGAINST_SCALE_MAP[cls] = orig[cls]
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION: No sex branching
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_no_sex(data, labels, is_bin):
    """Disable sex branching."""
    orig = cds_ovr.SEX_FEAT
    cds_ovr.SEX_FEAT = -1
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr.SEX_FEAT = orig
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def run_hyperparam_sweep(data, labels, is_bin, param_name, values):
    """Sweep a single hyperparameter across values."""
    results = {}
    for val in values:
        orig = getattr(cds_ovr, param_name)
        setattr(cds_ovr, param_name, val)
        accs = []
        for seed in SEEDS:
            res = run_10fold(data, labels, is_bin, seed=seed)
            acc, _, _, _ = stats(res)
            accs.append(100 * acc)
        setattr(cds_ovr, param_name, orig)
        results[val] = {
            'mean': round(float(np.mean(accs)), 2),
            'std': round(float(np.std(accs)), 2),
            'min': round(float(np.min(accs)), 2),
            'max': round(float(np.max(accs)), 2),
            'accs': [round(a, 2) for a in accs]
        }
        log(f"  {param_name}={val}: mean={np.mean(accs):.2f}% (σ={np.std(accs):.2f}%)")
    return results


def main():
    log("Loading data...")
    data, labels = load_data()
    is_bin = classify_features(data)
    log(f"  {data.shape[0]} patients, {data.shape[1]} features, {len(set(labels))} classes")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 1: BASELINE
    # ═══════════════════════════════════════════════════════════════════════════
    section("PART 1: BASELINE (W11-01 current settings)")
    t0 = time.time()
    baseline = run_baseline(data, labels, is_bin)
    log(f"  Baseline mean: {np.mean(baseline):.2f}% (σ={np.std(baseline):.2f}%)")
    log(f"  Per-seed: {[f'{a:.1f}' for a in baseline]}")
    log(f"  Time: {time.time()-t0:.0f}s")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 2: FULL COMPONENT ABLATION
    # ═══════════════════════════════════════════════════════════════════════════
    section("PART 2: FULL COMPONENT ABLATION (remove one at a time)")

    ablations = {}

    # 2a. Supervised binning → equal-width
    log("--- 2a. Supervised Binning → Equal-Width (Sturges) ---")
    t0 = time.time()
    abl_binning = run_ablation_equal_width(data, labels, is_bin)
    log(f"  Equal-width mean: {np.mean(abl_binning):.2f}% (σ={np.std(abl_binning):.2f}%)")
    log(f"  Δ from baseline: {np.mean(baseline)-np.mean(abl_binning):+.2f} pp")
    log(f"  Time: {time.time()-t0:.0f}s")
    ablations['supervised_binning'] = abl_binning

    # 2b. Correlation filtering → disabled
    log("\n--- 2b. Correlation Filtering → Disabled ---")
    t0 = time.time()
    abl_corr = run_ablation_no_corr_filter(data, labels, is_bin)
    log(f"  No corr filter mean: {np.mean(abl_corr):.2f}% (σ={np.std(abl_corr):.2f}%)")
    log(f"  Δ from baseline: {np.mean(baseline)-np.mean(abl_corr):+.2f} pp")
    log(f"  Time: {time.time()-t0:.0f}s")
    ablations['correlation_filter'] = abl_corr

    # 2c. Healthy bar → disabled
    log("\n--- 2c. Healthy Bar → Disabled ---")
    t0 = time.time()
    abl_hbar = run_ablation_no_healthy_bar(data, labels, is_bin)
    log(f"  No healthy bar mean: {np.mean(abl_hbar):.2f}% (σ={np.std(abl_hbar):.2f}%)")
    log(f"  Δ from baseline: {np.mean(baseline)-np.mean(abl_hbar):+.2f} pp")
    log(f"  Time: {time.time()-t0:.0f}s")
    ablations['healthy_bar'] = abl_hbar

    # 2d. Fisher weighting → uniform
    log("\n--- 2d. Fisher Weighting → Uniform ---")
    t0 = time.time()
    abl_fisher = run_ablation_no_fisher(data, labels, is_bin)
    log(f"  No Fisher mean: {np.mean(abl_fisher):.2f}% (σ={np.std(abl_fisher):.2f}%)")
    log(f"  Δ from baseline: {np.mean(baseline)-np.mean(abl_fisher):+.2f} pp")
    log(f"  Time: {time.time()-t0:.0f}s")
    ablations['fisher_weighting'] = abl_fisher

    # 2e. against_scale → uniform 0.8
    log("\n--- 2e. Per-Class against_scale → Uniform 0.8 ---")
    t0 = time.time()
    abl_against = run_ablation_uniform_against(data, labels, is_bin)
    log(f"  Uniform against mean: {np.mean(abl_against):.2f}% (σ={np.std(abl_against):.2f}%)")
    log(f"  Δ from baseline: {np.mean(baseline)-np.mean(abl_against):+.2f} pp")
    log(f"  Time: {time.time()-t0:.0f}s")
    ablations['against_scale'] = abl_against

    # 2f. Sex branching → disabled
    log("\n--- 2f. Sex Branching → Disabled ---")
    t0 = time.time()
    abl_sex = run_ablation_no_sex(data, labels, is_bin)
    log(f"  No sex branch mean: {np.mean(abl_sex):.2f}% (σ={np.std(abl_sex):.2f}%)")
    log(f"  Δ from baseline: {np.mean(baseline)-np.mean(abl_sex):+.2f} pp")
    log(f"  Time: {time.time()-t0:.0f}s")
    ablations['sex_branching'] = abl_sex

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3: STATISTICAL SIGNIFICANCE TESTS
    # ═══════════════════════════════════════════════════════════════════════════
    section("PART 3: STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank, paired by seed)")

    log(f"  {'Component':<25} {'Baseline':>9} {'Ablated':>9} {'Δ pp':>7} {'W-stat':>7} {'p-value':>9} {'Significant':>12}")
    log(f"  {'-'*25} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*9} {'-'*12}")

    significance_results = {}
    for name, abl_accs in ablations.items():
        w_stat, p_val, effect_r = paired_wilcoxon(baseline, abl_accs)
        delta = np.mean(baseline) - np.mean(abl_accs)
        sig = "YES (p<0.05)" if p_val < 0.05 else "no"
        log(f"  {name:<25} {np.mean(baseline):>8.2f}% {np.mean(abl_accs):>8.2f}% {delta:>+6.2f} {w_stat:>7.1f} {p_val:>9.4f} {sig:>12}")
        significance_results[name] = {
            'baseline_mean': round(float(np.mean(baseline)), 2),
            'ablated_mean': round(float(np.mean(abl_accs)), 2),
            'delta_pp': round(float(delta), 2),
            'wilcoxon_W': round(float(w_stat), 2),
            'p_value': round(float(p_val), 4),
            'significant_005': p_val < 0.05,
            'effect_size_r': round(float(effect_r), 3),
            'baseline_accs': [round(a, 2) for a in baseline],
            'ablated_accs': [round(a, 2) for a in abl_accs]
        }

    results_data['ablation_significance'] = significance_results

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 4: HYPERPARAMETER SENSITIVITY
    # ═══════════════════════════════════════════════════════════════════════════
    section("PART 4: HYPERPARAMETER SENSITIVITY SWEEPS")

    # 4a. CORR_THRESHOLD
    log("--- 4a. CORR_THRESHOLD sweep ---")
    corr_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    corr_results = run_hyperparam_sweep(data, labels, is_bin, 'CORR_THRESHOLD', corr_values)
    results_data['sensitivity_CORR_THRESHOLD'] = corr_results

    # 4b. MAX_BINS
    log("\n--- 4b. MAX_BINS sweep ---")
    bins_values = [3, 4, 5, 6, 8, 10]
    bins_results = run_hyperparam_sweep(data, labels, is_bin, 'MAX_BINS', bins_values)
    results_data['sensitivity_MAX_BINS'] = bins_results

    # 4c. FEATURES_PER_CLASS
    log("\n--- 4c. FEATURES_PER_CLASS sweep ---")
    fpc_values = [10, 14, 18, 22, 26, 30]
    fpc_results = run_hyperparam_sweep(data, labels, is_bin, 'FEATURES_PER_CLASS', fpc_values)
    results_data['sensitivity_FEATURES_PER_CLASS'] = fpc_results

    # ═══════════════════════════════════════════════════════════════════════════
    # PART 5: SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    section("PART 5: SUMMARY")

    log("ABLATION RANKING (by accuracy drop when removed):")
    ranked = sorted(significance_results.items(), key=lambda x: x[1]['delta_pp'], reverse=True)
    for i, (name, r) in enumerate(ranked, 1):
        sig_mark = "*" if r['significant_005'] else " "
        log(f"  {i}. {name:<25} Δ = {r['delta_pp']:+.2f} pp  p = {r['p_value']:.4f} {sig_mark}")

    log("\nHYPERPARAMETER SENSITIVITY SUMMARY:")
    log(f"  CORR_THRESHOLD: best = {max(corr_results.items(), key=lambda x: x[1]['mean'])[0]} "
        f"({max(corr_results.items(), key=lambda x: x[1]['mean'])[1]['mean']}%), "
        f"range = {min(v['mean'] for v in corr_results.values()):.2f}–{max(v['mean'] for v in corr_results.values()):.2f}%")
    log(f"  MAX_BINS: best = {max(bins_results.items(), key=lambda x: x[1]['mean'])[0]} "
        f"({max(bins_results.items(), key=lambda x: x[1]['mean'])[1]['mean']}%), "
        f"range = {min(v['mean'] for v in bins_results.values()):.2f}–{max(v['mean'] for v in bins_results.values()):.2f}%")
    log(f"  FEATURES_PER_CLASS: best = {max(fpc_results.items(), key=lambda x: x[1]['mean'])[0]} "
        f"({max(fpc_results.items(), key=lambda x: x[1]['mean'])[1]['mean']}%), "
        f"range = {min(v['mean'] for v in fpc_results.values()):.2f}–{max(v['mean'] for v in fpc_results.values()):.2f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # Save outputs
    # ═══════════════════════════════════════════════════════════════════════════
    report_path = os.path.join(OUT_DIR, "ablation_full_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nReport saved to {report_path}")

    json_path = os.path.join(OUT_DIR, "ablation_full_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, default=str)
    log(f"Data saved to {json_path}")


if __name__ == "__main__":
    main()
