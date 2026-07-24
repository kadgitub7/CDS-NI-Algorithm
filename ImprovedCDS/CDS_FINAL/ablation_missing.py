"""Run ablation tests for components NOT covered by ablation_full.py.
Missing: Dual AF, Ratio Scoring, Laplace Smoothing, Per-Class Thresholds, Rare Class Params.

Uses same methodology: remove one component, run 10-fold CV across 10 seeds,
paired Wilcoxon signed-rank test.
"""
import sys, os, io, json, time
import numpy as np
from scipy import stats as sp_stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

import cds_ovr
from cds_ovr import (
    load_data, classify_features, run_10fold, stats,
    RATIO_EPS, HEALTHY, CLASS_THRESHOLDS,
    AGAINST_SCALE_MAP, MIN_SUPPORT_MAP, CONF_SUPPORT_MAP, RARE_CLASSES,
    LAPLACE_ALPHA, N_FEAT, HEALTHY_WEIGHT, HEALTHY_BAR_CAP,
    SUSPICION_HCUT, SUSPICION_OFFSET
)

SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]

def paired_wilcoxon(baseline_accs, ablated_accs):
    diffs = np.array(baseline_accs) - np.array(ablated_accs)
    if np.all(diffs == 0):
        return 0.0, 1.0, 0.0
    try:
        stat, p = sp_stats.wilcoxon(diffs, alternative='two-sided')
    except ValueError:
        return 0.0, 1.0, 0.0
    n = len(diffs)
    z = sp_stats.norm.ppf(1 - p/2) if p > 0 else 0
    r = z / np.sqrt(n) if n > 0 else 0
    return float(stat), float(p), float(r)


def run_baseline(data, labels, is_bin):
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION 1: No Dual AF (single accumulator - only af_for, no af_against)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_no_dual_af(data, labels, is_bin):
    """Replace dual AF with single accumulator (only positive evidence).
    The predict function uses ratio = (af_for + eps) / (af_against + eps).
    Without dual AF: ignore against evidence, score = af_for directly.
    """
    orig_fn = cds_ovr._compute_af

    def _compute_af_single(uid, data, nodes, models, retained, against_scale):
        lvl_nodes = cds_ovr._route_user(uid, data, nodes)
        af_for, af_against = 0.0, 0.0
        n_for, n_against, n_used = 0, 0, 0
        max_for_contrib = 0.0

        fisher_map = {}
        for a in retained:
            fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
        max_fisher = max(fisher_map.values()) if fisher_map else 1.0

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
                    fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                    weighted = abs(shift) * confidence * fw
                    if shift >= 0:
                        af_for += weighted
                        n_for += 1
                        if weighted > max_for_contrib:
                            max_for_contrib = weighted
                    # Key change: ignore negative evidence entirely
                    n_used += 1

        return af_for, 0.0, n_used, n_for, 0, max_for_contrib

    cds_ovr._compute_af = _compute_af_single
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr._compute_af = orig_fn
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION 2: No Ratio Scoring (use raw af_for instead of ratio)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_no_ratio(data, labels, is_bin):
    """Replace ratio scoring with raw af_for score.
    Monkey-patch predict to use af_for directly as the class score.
    """
    orig_predict = cds_ovr.predict

    def predict_no_ratio(uid, data, nodes, all_cls, train_result):
        class_models, class_retained = train_result
        class_scores = {}
        for cls in all_cls:
            ag = AGAINST_SCALE_MAP.get(cls, 0.8)
            af = cds_ovr._compute_af(uid, data, nodes, class_models[cls],
                                      class_retained[cls], ag)
            class_scores[cls] = af[0]  # raw af_for instead of ratio

        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY:
                continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t:
                continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    cds_ovr.predict = predict_no_ratio
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr.predict = orig_predict
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION 3: No Laplace Smoothing (alpha = 0)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_no_laplace(data, labels, is_bin):
    """Set LAPLACE_ALPHA to 0 (no smoothing)."""
    orig = cds_ovr.LAPLACE_ALPHA
    cds_ovr.LAPLACE_ALPHA = 0.0
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr.LAPLACE_ALPHA = orig
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION 4: Uniform Per-Class Thresholds (all = 3.5)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_uniform_thresholds(data, labels, is_bin):
    """Replace per-class thresholds with a uniform 3.5 for all."""
    orig = dict(CLASS_THRESHOLDS)
    for cls in CLASS_THRESHOLDS:
        cds_ovr.CLASS_THRESHOLDS[cls] = 3.5
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr.CLASS_THRESHOLDS.update(orig)
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION 5: No Rare Class Adaptation (uniform support/conf)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation_no_rare_params(data, labels, is_bin):
    """Use uniform min_support=3, conf_support=10 for all classes (no rare class adaptation)."""
    orig_ms = dict(MIN_SUPPORT_MAP)
    orig_cs = dict(CONF_SUPPORT_MAP)
    orig_ag = dict(AGAINST_SCALE_MAP)
    for cls in MIN_SUPPORT_MAP:
        cds_ovr.MIN_SUPPORT_MAP[cls] = 3
    for cls in CONF_SUPPORT_MAP:
        cds_ovr.CONF_SUPPORT_MAP[cls] = 10
    for cls in AGAINST_SCALE_MAP:
        cds_ovr.AGAINST_SCALE_MAP[cls] = 0.8
    accs = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, _, _, _ = stats(res)
        accs.append(100 * acc)
    cds_ovr.MIN_SUPPORT_MAP.update(orig_ms)
    cds_ovr.CONF_SUPPORT_MAP.update(orig_cs)
    cds_ovr.AGAINST_SCALE_MAP.update(orig_ag)
    return accs


ALL_ABLATIONS = {
    "dual_af": ("Dual AF (single accumulator)", run_ablation_no_dual_af),
    "ratio_scoring": ("Ratio Scoring (use raw af_for)", run_ablation_no_ratio),
    "laplace_smoothing": ("Laplace Smoothing (alpha=0)", run_ablation_no_laplace),
    "per_class_thresholds": ("Per-Class Thresholds (uniform 3.5)", run_ablation_uniform_thresholds),
    "rare_class_params": ("Rare Class Params (uniform support)", run_ablation_no_rare_params),
}


def run_single(key):
    """Run one ablation test and save result to its own JSON file."""
    t0 = time.perf_counter()
    name, fn = ALL_ABLATIONS[key]
    data, labels = load_data()
    is_bin = classify_features(data)

    print(f"[{key}] Running baseline...", flush=True)
    baseline = run_baseline(data, labels, is_bin)
    print(f"[{key}] Baseline: {np.mean(baseline):.2f} +/- {np.std(baseline):.2f}%", flush=True)

    print(f"[{key}] Running ablation: {name}...", flush=True)
    ablated = fn(data, labels, is_bin)
    delta = np.mean(baseline) - np.mean(ablated)
    w, p, r = paired_wilcoxon(baseline, ablated)

    result = {
        key: {
            "baseline_mean": round(float(np.mean(baseline)), 2),
            "ablated_mean": round(float(np.mean(ablated)), 2),
            "delta_pp": round(float(delta), 2),
            "wilcoxon_W": round(float(w), 1),
            "p_value": round(float(p), 4) if p >= 0.001 else float(p),
            "significant_005": bool(p < 0.05),
            "effect_size_r": round(float(r), 3),
            "baseline_accs": [round(a, 2) for a in baseline],
            "ablated_accs": [round(a, 2) for a in ablated],
        }
    }

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"[{key}] {name}: {np.mean(ablated):.2f}% (delta={delta:+.2f}pp, "
          f"W={w:.1f}, p={p:.4f}, r={r:.3f}) {sig}", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), f"ablation_{key}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    elapsed = time.perf_counter() - t0
    print(f"[{key}] Done in {elapsed:.1f}s. Saved to {out_path}", flush=True)


def merge_results():
    """Merge individual ablation results into ablation_missing_data.json."""
    merged = {}
    for key in ALL_ABLATIONS:
        path = os.path.join(os.path.dirname(__file__), f"ablation_{key}.json")
        if os.path.exists(path):
            with open(path) as f:
                merged.update(json.load(f))
    out_path = os.path.join(os.path.dirname(__file__), "ablation_missing_data.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(merged)} ablation results into {out_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "merge":
            merge_results()
        elif arg in ALL_ABLATIONS:
            run_single(arg)
        else:
            print(f"Unknown ablation: {arg}. Choose from: {list(ALL_ABLATIONS.keys())} or 'merge'")
            sys.exit(1)
    else:
        # Run all sequentially (legacy mode)
        for key in ALL_ABLATIONS:
            run_single(key)
        merge_results()
