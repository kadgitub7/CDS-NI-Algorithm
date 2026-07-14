"""Deep error analysis — understand exactly who is being misclassified and why.
Runs the best model (SV + against 0.8) and dissects every error.
"""
import numpy as np
from collections import defaultdict
from experiment_runner import (
    load_data, classify_features, build_tree, _route_user,
    train_all_cfg, compute_af_cfg, predict_cfg, score_patient,
    HEALTHY, CLASS_THRESHOLDS, HEALTHY_WEIGHT, HEALTHY_BAR_CAP,
    SUSPICION_HCUT, SUSPICION_OFFSET, MIN_SUPPORT, CONF_SUPPORT,
    N_FEAT, RATIO_EPS,
)
from pathlib import Path

DATA_PATH = str(Path(__file__).parent / "data" / "arrhythmia.data")


def analyze_errors(seed=13):
    print(f"Loading {DATA_PATH}")
    data, labels = load_data(DATA_PATH)
    is_bin = classify_features(data)
    n = data.shape[0]
    all_cls = sorted(set(labels))
    print(f"{n} patients, {len(all_cls)} classes: {all_cls}")
    print(f"Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}\n")

    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 10)

    errors = []
    corrects = []
    all_results = []

    for fi in range(10):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        cm, cr = train_all_cfg(nodes, td, tl, is_bin, all_cls, use_supervised_bins=True)

        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, class_scores = predict_cfg(uid, data, nodes, cm, cr, all_cls,
                                              against_scale=0.8)
            pred = int(pred)

            detail = {
                'uid': int(uid),
                'fold': fi,
                'true': true_cls,
                'pred': pred,
                'correct': pred == true_cls,
                'scores': {int(c): float(class_scores.get(c, 0)) for c in all_cls},
            }

            af_details = {}
            for cls in all_cls:
                af = compute_af_cfg(uid, data, nodes, cm[cls], cr[cls], against_scale=0.8)
                af_details[int(cls)] = {
                    'af_for': float(af[0]),
                    'af_against': float(af[1]),
                    'n_used': int(af[2]),
                    'n_for': int(af[3]),
                    'n_against': int(af[4]),
                    'max_fc': float(af[5]),
                    'ratio': float((af[0] + RATIO_EPS) / (af[1] + RATIO_EPS)),
                }
            detail['af'] = af_details

            h_score = class_scores.get(HEALTHY, 1.0)
            detail['h_score'] = float(h_score)
            detail['healthy_bar'] = float(min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP))

            n_features_used = {}
            for cls in all_cls:
                retained = cr[cls]
                lvl_nodes = _route_user(uid, data, nodes)
                n_feat_used = 0
                for lvl in sorted(lvl_nodes.keys()):
                    for nd in lvl_nodes[lvl]:
                        for a in retained:
                            if a[1] != nd.nid: continue
                            f = a[0]
                            if not np.isnan(data[uid, f]):
                                mo = cm[cls].get((nd.nid, f))
                                if mo:
                                    bi = int(np.clip(np.searchsorted(mo.edges[1:], data[uid,f], side='right'), 0, mo.n_bins-1))
                                    if mo.bin_counts[bi] >= MIN_SUPPORT:
                                        n_feat_used += 1
                n_features_used[int(cls)] = n_feat_used
            detail['n_features_used'] = n_features_used

            all_results.append(detail)
            if pred != true_cls:
                errors.append(detail)
            else:
                corrects.append(detail)

    print(f"=" * 80)
    print(f"OVERALL: {len(corrects)} correct, {len(errors)} errors ({100*len(corrects)/n:.1f}%)")
    print(f"=" * 80)

    # Error breakdown by true class
    print(f"\n{'='*80}")
    print("ERROR BREAKDOWN BY TRUE CLASS")
    print(f"{'='*80}")
    error_by_true = defaultdict(list)
    for e in errors:
        error_by_true[e['true']].append(e)

    total_by_class = defaultdict(int)
    for r in all_results:
        total_by_class[r['true']] += 1

    for cls in sorted(all_cls):
        errs = error_by_true.get(cls, [])
        total = total_by_class[cls]
        if not errs:
            print(f"\nClass {cls}: {total} patients, 0 errors (100.0% accuracy)")
            continue
        print(f"\nClass {cls}: {total} patients, {len(errs)} errors ({100*(total-len(errs))/total:.1f}% accuracy)")
        pred_counts = defaultdict(int)
        for e in errs:
            pred_counts[e['pred']] += 1
        print(f"  Misclassified as: {dict(pred_counts)}")

    # Confusion matrix
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX")
    print(f"{'='*80}")
    cm_matrix = defaultdict(lambda: defaultdict(int))
    for r in all_results:
        cm_matrix[r['true']][r['pred']] += 1

    header = "True\\Pred " + " ".join(f"{c:>5d}" for c in sorted(all_cls))
    print(header)
    for true_cls in sorted(all_cls):
        row = f"    {true_cls:>4d}  "
        for pred_cls in sorted(all_cls):
            count = cm_matrix[true_cls][pred_cls]
            row += f"{count:>5d} "
        print(row)

    # Detailed error analysis — what are the score patterns?
    print(f"\n{'='*80}")
    print("DETAILED ERROR PATTERNS")
    print(f"{'='*80}")

    error_patterns = defaultdict(list)
    for e in errors:
        error_patterns[(e['true'], e['pred'])].append(e)

    for (true_cls, pred_cls), errs in sorted(error_patterns.items(), key=lambda x: -len(x[1])):
        print(f"\n--- {len(errs)} patients: True={true_cls} -> Predicted={pred_cls} ---")
        for e in errs[:5]:  # Show first 5
            true_score = e['scores'][true_cls]
            pred_score = e['scores'][pred_cls]
            h_score = e['h_score']
            true_t = CLASS_THRESHOLDS.get(true_cls, 3.0)
            pred_t = CLASS_THRESHOLDS.get(pred_cls, 3.0)
            true_af = e['af'][true_cls]
            pred_af = e['af'].get(pred_cls, {})

            print(f"  Patient {e['uid']} (fold {e['fold']}):")
            print(f"    True class {true_cls}: score={true_score:.2f}, threshold={true_t:.1f}, "
                  f"AF_for={true_af['af_for']:.3f}, AF_ag={true_af['af_against']:.3f}, "
                  f"n_for={true_af['n_for']}, n_ag={true_af['n_against']}")
            if pred_cls != HEALTHY:
                print(f"    Pred class {pred_cls}: score={pred_score:.2f}, threshold={pred_t:.1f}, "
                      f"AF_for={pred_af.get('af_for',0):.3f}, AF_ag={pred_af.get('af_against',0):.3f}")
            print(f"    Healthy: score={h_score:.2f}, bar={e['healthy_bar']:.2f}")
            print(f"    Features used: true_cls={e['n_features_used'][true_cls]}, "
                  f"healthy={e['n_features_used'].get(1,0)}")

    # Score distribution analysis
    print(f"\n{'='*80}")
    print("SCORE DISTRIBUTIONS: CORRECT vs ERRORS")
    print(f"{'='*80}")
    for cls in sorted(all_cls):
        if cls == HEALTHY: continue
        correct_true = [r['scores'][cls] for r in corrects if r['true'] == cls]
        error_true = [r['scores'][cls] for r in errors if r['true'] == cls]
        correct_healthy_for_cls = [r['scores'][cls] for r in corrects if r['true'] == HEALTHY]
        if correct_true and error_true:
            print(f"\nClass {cls} (threshold={CLASS_THRESHOLDS.get(cls, 3.0)}):")
            print(f"  Correctly classified ({len(correct_true)}): "
                  f"mean={np.mean(correct_true):.2f}, min={np.min(correct_true):.2f}, max={np.max(correct_true):.2f}")
            print(f"  Misclassified ({len(error_true)}): "
                  f"mean={np.mean(error_true):.2f}, min={np.min(error_true):.2f}, max={np.max(error_true):.2f}")
            print(f"  Healthy scored as {cls} ({len(correct_healthy_for_cls)}): "
                  f"mean={np.mean(correct_healthy_for_cls):.2f}, "
                  f"P95={np.percentile(correct_healthy_for_cls, 95):.2f}")

    # Missing value analysis for errors
    print(f"\n{'='*80}")
    print("MISSING VALUE ANALYSIS")
    print(f"{'='*80}")
    correct_missing = [np.isnan(data[r['uid']]).sum() for r in corrects]
    error_missing = [np.isnan(data[e['uid']]).sum() for e in errors]
    print(f"Correct patients: mean missing={np.mean(correct_missing):.1f}, "
          f"median={np.median(correct_missing):.0f}")
    print(f"Error patients: mean missing={np.mean(error_missing):.1f}, "
          f"median={np.median(error_missing):.0f}")

    # AF breadth analysis
    print(f"\n{'='*80}")
    print("AF BREADTH ANALYSIS (n_for / (n_for + n_against))")
    print(f"{'='*80}")
    for cls in sorted(all_cls):
        if cls == HEALTHY: continue
        correct_breadth = []
        error_breadth = []
        for r in all_results:
            if r['true'] != cls: continue
            af = r['af'][cls]
            total = af['n_for'] + af['n_against']
            breadth = af['n_for'] / max(total, 1)
            if r['correct']:
                correct_breadth.append(breadth)
            else:
                error_breadth.append(breadth)
        if correct_breadth and error_breadth:
            print(f"Class {cls}: correct mean breadth={np.mean(correct_breadth):.3f}, "
                  f"error mean breadth={np.mean(error_breadth):.3f}")

    # Feature availability — how many of the selected features are usable?
    print(f"\n{'='*80}")
    print("FEATURE AVAILABILITY (how many features actually contribute)")
    print(f"{'='*80}")
    for cls in sorted(all_cls):
        if cls == HEALTHY: continue
        correct_nf = [r['n_features_used'][cls] for r in corrects if r['true'] == cls]
        error_nf = [r['n_features_used'][cls] for r in errors if r['true'] == cls]
        if correct_nf and error_nf:
            print(f"Class {cls}: correct uses {np.mean(correct_nf):.1f} features, "
                  f"errors use {np.mean(error_nf):.1f} features")

    return all_results, errors, corrects


if __name__ == "__main__":
    all_results, errors, corrects = analyze_errors()
