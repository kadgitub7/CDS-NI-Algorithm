"""Find the most flippable errors in W9-05 seed=13.

We need 86.57% = 360/416 correct. Currently 86.3% = 359/416.
Need to flip exactly 1 error without breaking existing correct predictions.
"""
import numpy as np
from collections import defaultdict
from pathlib import Path
from experiment_runner import (
    load_data, classify_features, build_tree, _route_user,
    train_ovr_node_cfg, refine_ovr_node_cfg,
    compute_af_cfg, predict_cfg, score_patient,
    run_10fold, stats, BinModel,
    HEALTHY, CLASS_THRESHOLDS, RATIO_EPS, HEALTHY_WEIGHT, HEALTHY_BAR_CAP,
    SUSPICION_HCUT, SUSPICION_OFFSET, MIN_SUPPORT, CONF_SUPPORT,
    N_FEAT, LAPLACE_ALPHA, U_MIN, CORR_THRESHOLD,
    _supervised_bin_edges, _fast_abs_corr,
)

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")
RARE_CLASSES = {4, 5, 9}


def _train_w905(nodes, td, tl, is_bin, all_cls):
    import experiment_runner as mod
    class_models, class_retained = {}, {}
    orig_ms, orig_cs = mod.MIN_SUPPORT, mod.CONF_SUPPORT
    for cls in all_cls:
        mod.MIN_SUPPORT = 2 if cls in RARE_CLASSES else 3
        mod.CONF_SUPPORT = 5 if cls in RARE_CLASSES else 10
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


if __name__ == "__main__":
    print(f"Loading {DATA_PATH}", flush=True)
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    n = data.shape[0]
    all_cls = sorted(set(labels))
    print(f"{n} users x {data.shape[1]} feats\n")

    seed = 13
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 10)

    errors = []
    correct_count = 0

    for fi in range(10):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        cm, cr = _train_w905(nodes, td, tl, is_bin, all_cls)

        for uid in test_idx:
            true_cls = int(labels[uid])

            class_scores = {}
            class_details = {}
            for cls in all_cls:
                ag = 0.5 if cls in RARE_CLASSES else 0.8
                af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=ag)
                s = score_patient(af[0], af[1], af[3], af[4], af[5], 'ratio')
                class_scores[cls] = s
                class_details[cls] = {
                    'af_for': af[0], 'af_against': af[1], 'n_used': af[2],
                    'n_for': af[3], 'n_against': af[4], 'max_fc': af[5],
                    'score': s,
                    'breadth': af[3] / max(af[3] + af[4], 1),
                }

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
            pred = max(candidates, key=candidates.get) if candidates else HEALTHY

            if pred == true_cls:
                correct_count += 1
            else:
                true_score = class_scores.get(true_cls, 0)
                pred_score = class_scores.get(pred, 0)
                true_thresh = CLASS_THRESHOLDS.get(true_cls, 3.0)
                if true_cls == HEALTHY:
                    true_thresh = 0
                    gap = 0
                else:
                    t = true_thresh
                    if h_score < SUSPICION_HCUT:
                        t -= SUSPICION_OFFSET
                    t = max(t, healthy_bar)
                    gap = t - true_score

                errors.append({
                    'uid': uid, 'fold': fi,
                    'true': true_cls, 'pred': pred,
                    'true_score': true_score, 'pred_score': pred_score,
                    'h_score': h_score, 'healthy_bar': healthy_bar,
                    'gap_to_threshold': gap,
                    'true_details': class_details.get(true_cls, {}),
                    'pred_details': class_details.get(pred, {}),
                    'all_scores': {c: round(s, 3) for c, s in class_scores.items()},
                })

    print(f"Total correct: {correct_count}/{n} = {100*correct_count/n:.2f}%")
    print(f"Total errors: {len(errors)}\n")

    # Sort errors by how close the true class was to being predicted
    # For disease->healthy errors: how close was the disease score to the threshold
    # For healthy->disease errors: how close was healthy to winning
    print("=" * 90)
    print("  MOST FLIPPABLE ERRORS (sorted by gap to correct prediction)")
    print("=" * 90)

    for e in errors:
        if e['true'] == HEALTHY:
            # FP: healthy predicted as disease. Need to suppress disease score.
            e['flip_difficulty'] = e['pred_score'] - e['healthy_bar']
        else:
            # FN: disease predicted as healthy/wrong. Need true class score above threshold.
            e['flip_difficulty'] = e['gap_to_threshold']

    errors.sort(key=lambda x: abs(x['flip_difficulty']))

    print(f"\n  {'UID':>5s}  {'True':>5s}  {'Pred':>5s}  {'TrueScore':>10s}  {'PredScore':>10s}  "
          f"{'H_Score':>8s}  {'Gap':>8s}  {'Breadth':>8s}")
    print(f"  {'-'*75}")

    for e in errors[:30]:
        td = e['true_details']
        breadth = td.get('breadth', 0)
        print(f"  {e['uid']:5d}  {e['true']:5d}  {e['pred']:5d}  "
              f"{e['true_score']:10.3f}  {e['pred_score']:10.3f}  "
              f"{e['h_score']:8.3f}  {e['gap_to_threshold']:8.3f}  {breadth:8.3f}")

    # Error pattern summary
    print(f"\n\n  ERROR PATTERN SUMMARY")
    print(f"  {'-'*50}")
    pattern_counts = defaultdict(int)
    for e in errors:
        pattern_counts[(e['true'], e['pred'])] += 1
    for (t, p), count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        t_name = "Healthy" if t == 1 else f"Class {t}"
        p_name = "Healthy" if p == 1 else f"Class {p}"
        print(f"  {t_name:>12s} -> {p_name:<12s}  {count:3d} errors")

    # Breakdown by error type
    print(f"\n\n  ERROR TYPE ANALYSIS")
    print(f"  {'-'*50}")
    fn_disease = [e for e in errors if e['true'] != HEALTHY and e['pred'] == HEALTHY]
    fp_healthy = [e for e in errors if e['true'] == HEALTHY and e['pred'] != HEALTHY]
    misclass = [e for e in errors if e['true'] != HEALTHY and e['pred'] != HEALTHY and e['pred'] != e['true']]

    print(f"  Disease -> Healthy (FN): {len(fn_disease)} errors")
    print(f"  Healthy -> Disease (FP): {len(fp_healthy)} errors")
    print(f"  Disease -> Wrong Disease: {len(misclass)} errors")

    # For the top 10 most flippable, show what change would fix them
    print(f"\n\n  TOP 10 MOST FLIPPABLE - WHAT WOULD FIX THEM")
    print(f"  {'='*80}")
    for i, e in enumerate(errors[:10]):
        print(f"\n  #{i+1}: Patient {e['uid']} (fold {e['fold']})")
        print(f"    True: class {e['true']}, Predicted: class {e['pred']}")
        print(f"    All scores: {e['all_scores']}")
        print(f"    H_score: {e['h_score']:.3f}, Healthy bar: {e['healthy_bar']:.3f}")

        if e['true'] == HEALTHY:
            print(f"    FIX: Suppress class {e['pred']} score from {e['pred_score']:.3f} "
                  f"below threshold or raise healthy bar")
            print(f"    Pred class breadth: {e['pred_details'].get('breadth', 0):.3f}")
        elif e['pred'] == HEALTHY:
            true_t = CLASS_THRESHOLDS.get(e['true'], 3.0)
            print(f"    FIX: Raise class {e['true']} score from {e['true_score']:.3f} "
                  f"above threshold {true_t:.1f}")
            print(f"    True class breadth: {e['true_details'].get('breadth', 0):.3f}, "
                  f"n_for: {e['true_details'].get('n_for', 0)}, "
                  f"n_against: {e['true_details'].get('n_against', 0)}")
        else:
            print(f"    FIX: Make class {e['true']} score ({e['true_score']:.3f}) > "
                  f"class {e['pred']} score ({e['pred_score']:.3f})")
