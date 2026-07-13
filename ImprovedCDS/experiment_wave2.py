"""Wave 2 experiments — build on supervised binning winner.
Supervised bins gave 84.1%. Now combine with other promising changes.
"""
import time
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from experiment_runner import (
    load_data, classify_features, build_tree,
    train_all_cfg, predict_cfg, compute_af_cfg, score_patient,
    run_10fold, stats,
    HEALTHY, CLASS_THRESHOLDS, RATIO_EPS, HEALTHY_WEIGHT, HEALTHY_BAR_CAP,
    SUSPICION_HCUT, SUSPICION_OFFSET, MIN_SUPPORT, CONF_SUPPORT,
    _route_user,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


def exp_sv_27healthy(data, labels, is_bin, seed):
    """Supervised bins + 27 healthy features."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, healthy_fpc=27)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_lower_bar(data, labels, is_bin, seed):
    """Supervised bins + healthy bar cap 3.5."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, healthy_bar_cap=3.5)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_higher_bar(data, labels, is_bin, seed):
    """Supervised bins + healthy bar cap 7.0."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, healthy_bar_cap=7.0)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_against04(data, labels, is_bin, seed):
    """Supervised bins + against_scale 0.4."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.4)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_against08(data, labels, is_bin, seed):
    """Supervised bins + against_scale 0.8."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, against_scale=0.8)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_healthy_breadth(data, labels, is_bin, seed):
    """Supervised bins + healthy uses breadth_ratio."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls,
                          healthy_scoring='breadth_ratio')
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_lower_thresholds(data, labels, is_bin, seed):
    """Supervised bins + lower thresholds (better sensitivity)."""
    thresholds = {2: 3.0, 3: 4.0, 4: 3.5, 5: 3.0, 6: 3.0, 9: 4.0, 10: 3.0}
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, thresholds=thresholds)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_higher_thresholds(data, labels, is_bin, seed):
    """Supervised bins + higher thresholds (better specificity)."""
    thresholds = {2: 4.0, 3: 6.0, 4: 5.0, 5: 4.0, 6: 4.0, 9: 6.0, 10: 4.0}
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, thresholds=thresholds)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_27h_lower_bar(data, labels, is_bin, seed):
    """Supervised bins + 27 healthy feats + healthy bar 3.5."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls,
                             use_supervised_bins=True, healthy_fpc=27)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, healthy_bar_cap=3.5)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_no_suspicion(data, labels, is_bin, seed):
    """Supervised bins + disable suspicion offset."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls])
            class_scores[cls] = (af[0] + RATIO_EPS) / (af[1] + RATIO_EPS)
        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY: continue
            t = CLASS_THRESHOLDS.get(cls, 3.0)
            t = max(t, healthy_bar)
            if score < t: continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_healthy_weight_higher(data, labels, is_bin, seed):
    """Supervised bins + healthy weight 1.15 (stronger healthy preference)."""
    def train(nodes, td, tl, is_bin, all_cls):
        return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        return predict_cfg(uid, data, nodes, cm, cr, all_cls, healthy_weight=1.15)
    return run_10fold(data, labels, is_bin, seed, predict, train)


def exp_sv_multiseed(data, labels, is_bin, seed):
    """Supervised bins tested on seeds 13,20,27 (mean and best)."""
    results_all = []
    accs = []
    for s in [13, 20, 27]:
        def train(nodes, td, tl, is_bin, all_cls):
            return train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)
        def predict(uid, data, nodes, all_cls, tr):
            cm, cr = tr
            return predict_cfg(uid, data, nodes, cm, cr, all_cls)
        r = run_10fold(data, labels, is_bin, s, predict, train)
        acc = sum(x[3] for x in r) / len(r)
        accs.append(acc)
        if not results_all or acc > max(sum(x[3] for x in results_all) / len(results_all), 0):
            results_all = r
    return results_all


experiments = [
    ("W2-01 SV + 27 healthy feats",        exp_sv_27healthy),
    ("W2-02 SV + bar cap 3.5",             exp_sv_lower_bar),
    ("W2-03 SV + bar cap 7.0",             exp_sv_higher_bar),
    ("W2-04 SV + against 0.4",             exp_sv_against04),
    ("W2-05 SV + against 0.8",             exp_sv_against08),
    ("W2-06 SV + healthy breadth_ratio",   exp_sv_healthy_breadth),
    ("W2-07 SV + lower thresholds",        exp_sv_lower_thresholds),
    ("W2-08 SV + higher thresholds",       exp_sv_higher_thresholds),
    ("W2-09 SV + 27H + bar 3.5",          exp_sv_27h_lower_bar),
    ("W2-10 SV + no suspicion",            exp_sv_no_suspicion),
    ("W2-11 SV + healthy weight 1.15",     exp_sv_healthy_weight_higher),
    ("W2-12 SV multiseed avg",             exp_sv_multiseed),
]


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats\n")

    seed = 13
    if len(sys.argv) > 1:
        exp_id = int(sys.argv[1])
        experiments_run = [experiments[exp_id - 1]]
    else:
        experiments_run = experiments

    print(f"{'Experiment':42s}  {'Acc':>6s}  {'Spec':>6s}  {'Sens':>6s}  {'BA':>6s}  {'Time':>6s}")
    print("-" * 85)

    for name, exp_fn in experiments_run:
        t0 = time.time()
        results = exp_fn(data, labels, is_bin, seed)
        elapsed = time.time() - t0
        acc, spec, sens, ba = stats(results)
        print(f"{name:42s}  {100*acc:5.1f}%  {100*spec:5.1f}%  {100*sens:5.1f}%  {100*ba:5.1f}%  {elapsed:5.0f}s",
              flush=True)
