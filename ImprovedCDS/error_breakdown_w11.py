"""Error breakdown for W11-01 across all split types.
Generates per-class accuracy and misclassification patterns.
"""
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from experiment_runner import (
    load_data, classify_features, build_tree,
    train_ovr_node_cfg, refine_ovr_node_cfg,
    compute_af_cfg, score_patient,
    run_10fold, stats, HEALTHY, CLASS_THRESHOLDS,
    HEALTHY_WEIGHT, HEALTHY_BAR_CAP,
    SUSPICION_HCUT, SUSPICION_OFFSET,
    CORR_THRESHOLD,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")
RARE_CLASSES = {4, 5, 9}
SEED = 13

W1101_THRESHOLDS = dict(CLASS_THRESHOLDS)
W1101_THRESHOLDS[10] = 3.0

CLASS_NAMES = {
    1: "Normal", 2: "Coronary Artery Disease", 3: "Old Anterior MI",
    4: "Old Inferior MI", 5: "Sinus Tachycardia", 6: "Sinus Bradycardia",
    9: "Left Bundle Branch Block", 10: "Right Bundle Branch Block",
}


def _train_with_class_params(nodes, td, tl, is_bin, all_cls, min_sup_map, conf_sup_map):
    import experiment_runner as mod
    class_models, class_retained = {}, {}
    orig_ms, orig_cs = mod.MIN_SUPPORT, mod.CONF_SUPPORT
    for cls in all_cls:
        mod.MIN_SUPPORT = min_sup_map.get(cls, 3)
        mod.CONF_SUPPORT = conf_sup_map.get(cls, 10)
        cls_models = {}
        cls_actions = defaultdict(list)
        for nd in nodes:
            nm, na = train_ovr_node_cfg(nd, td, tl, is_bin, cls, use_supervised_bins=True)
            cls_models.update(nm)
            for a in na:
                cls_actions[a[1]].append(a)
        cls_ret = []
        for nd in nodes:
            cls_ret.extend(refine_ovr_node_cfg(
                nd, cls_models, cls_actions.get(nd.nid, []), td,
                rank_by='score', fpc=18))
        class_models[cls] = cls_models
        class_retained[cls] = cls_ret
    mod.MIN_SUPPORT = orig_ms
    mod.CONF_SUPPORT = orig_cs
    return class_models, class_retained


def make_w1101():
    ms = {cls: (2 if cls in RARE_CLASSES else 3) for cls in range(1, 14)}
    cs = {cls: (5 if cls in RARE_CLASSES else 10) for cls in range(1, 14)}

    def train(nodes, td, tl, is_bin, all_cls):
        return _train_with_class_params(nodes, td, tl, is_bin, all_cls, ms, cs)

    def predict(uid, data, nodes, all_cls, tr):
        cm, cr = tr
        class_scores = {}
        for cls in all_cls:
            ag = 0.5 if cls in RARE_CLASSES else 0.8
            af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=ag)
            class_scores[cls] = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
        h_score = class_scores.get(HEALTHY, 1.0)
        healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
        candidates = {}
        for cls, score in class_scores.items():
            if cls == HEALTHY:
                continue
            t = W1101_THRESHOLDS.get(cls, 3.0)
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t = max(t, healthy_bar)
            if score < t:
                continue
            candidates[cls] = (score - t) / max(t, 0.1)
        best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
        return best_cls, class_scores

    return train, predict


def run_split(data, labels, is_bin, seed, predict_fn, train_fn, train_frac):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]
    td, tl = data[train_idx], labels[train_idx]
    all_cls = sorted(set(labels))
    nodes = build_tree(td, tl, is_bin)
    train_result = train_fn(nodes, td, tl, is_bin, all_cls)
    results = []
    for uid in test_idx:
        true_cls = int(labels[uid])
        pred, scores = predict_fn(uid, data, nodes, all_cls, train_result)
        results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def error_breakdown(results, title):
    print(f"\n{'='*70}")
    print(f"  ERROR BREAKDOWN: {title}")
    print(f"{'='*70}")

    acc, spec, sens, ba = stats(results)
    correct_binary = sum(1 for _, t, p, _ in results
                         if (t == HEALTHY) == (p == HEALTHY))
    bin_acc = correct_binary / len(results)
    print(f"  Overall: Acc={100*acc:.1f}%, Binary={100*bin_acc:.1f}%, "
          f"Spec={100*spec:.1f}%, Sens={100*sens:.1f}%")

    by_class = defaultdict(list)
    for uid, true_cls, pred, correct in results:
        by_class[true_cls].append((uid, pred, correct))

    print(f"\n  Per-class detail:")
    print(f"  {'Class':<6s} {'Name':<28s} {'Correct':>7s} {'Total':>5s} {'Acc':>7s} {'Misclassed as':}")
    print(f"  {'-'*85}")

    total_fn = 0  # disease->healthy
    total_fp = 0  # healthy->disease
    total_wrong_disease = 0

    for cls in sorted(by_class.keys()):
        items = by_class[cls]
        total = len(items)
        correct = sum(1 for _, _, c in items if c)
        pct = 100 * correct / total if total > 0 else 0

        misclass = Counter()
        for _, pred, c in items:
            if not c:
                misclass[pred] += 1
                if cls == HEALTHY and pred != HEALTHY:
                    total_fp += 1
                elif cls != HEALTHY and pred == HEALTHY:
                    total_fn += 1
                elif cls != HEALTHY and pred != HEALTHY:
                    total_wrong_disease += 1

        name = CLASS_NAMES.get(cls, f"Class {cls}")
        mis_str = ", ".join(f"{'H' if p == 1 else f'c{p}'}({n})"
                           for p, n in misclass.most_common())
        print(f"  {cls:<6d} {name:<28s} {correct:>5d}/{total:<5d} {pct:5.1f}%  {mis_str}")

    total_errors = sum(1 for _, _, _, c in results if not c)
    print(f"\n  Error summary: {total_errors} total errors")
    print(f"    Disease->Healthy (FN):     {total_fn}")
    print(f"    Healthy->Disease (FP):      {total_fp}")
    print(f"    Disease->Wrong Disease:     {total_wrong_disease}")


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    print(f"{data.shape[0]} users x {data.shape[1]} feats")

    train_fn, predict_fn = make_w1101()

    # 10-fold CV (seed 13)
    results_cv = run_10fold(data, labels, is_bin, SEED, predict_fn, train_fn)
    error_breakdown(results_cv, f"10-Fold CV (seed={SEED})")

    # 90/10 split (seed 76 — best performer)
    results_90 = run_split(data, labels, is_bin, 76, predict_fn, train_fn, 0.9)
    error_breakdown(results_90, "90/10 Split (seed=76, best)")

    # 90/10 split (seed 13)
    results_90_13 = run_split(data, labels, is_bin, SEED, predict_fn, train_fn, 0.9)
    error_breakdown(results_90_13, f"90/10 Split (seed={SEED})")

    # 60/40 split (seed 76 — best performer)
    results_60 = run_split(data, labels, is_bin, 76, predict_fn, train_fn, 0.6)
    error_breakdown(results_60, "60/40 Split (seed=76, best)")

    # 60/40 split (seed 13)
    results_60_13 = run_split(data, labels, is_bin, SEED, predict_fn, train_fn, 0.6)
    error_breakdown(results_60_13, f"60/40 Split (seed={SEED})")
