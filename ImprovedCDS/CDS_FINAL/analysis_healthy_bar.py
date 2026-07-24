"""Analyze healthy bar boundary misclassifications.
For each misclassified patient in 10-fold CV (seed 13), compute how close
their disease score was to the healthy bar threshold.
"""
import sys, os, io, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import (load_data, classify_features, build_tree, train, predict,
                     _compute_af, HEALTHY, CLASS_THRESHOLDS, HEALTHY_WEIGHT,
                     HEALTHY_BAR_CAP, SUSPICION_HCUT, SUSPICION_OFFSET,
                     AGAINST_SCALE_MAP, RATIO_EPS)

def run_detailed_cv(data, labels, is_bin, seed=13):
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 10)
    all_cls = sorted(set(labels))
    results = []

    for fi in range(10):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]
        nodes = build_tree(td, tl, is_bin)
        train_result = train(nodes, td, tl, is_bin, all_cls)
        class_models, class_retained = train_result

        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict(uid, data, nodes, all_cls, train_result)

            h_score = scores.get(HEALTHY, 1.0)
            healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)

            disease_details = {}
            for cls in all_cls:
                if cls == HEALTHY:
                    continue
                t = CLASS_THRESHOLDS.get(cls, 3.0)
                if h_score < SUSPICION_HCUT:
                    t -= SUSPICION_OFFSET
                effective_t = max(t, healthy_bar)
                margin = scores[cls] - effective_t
                disease_details[int(cls)] = {
                    "score": round(float(scores[cls]), 4),
                    "threshold": round(float(CLASS_THRESHOLDS.get(cls, 3.0)), 4),
                    "effective_threshold": round(float(effective_t), 4),
                    "margin": round(float(margin), 4),
                    "healthy_bar_active": bool(healthy_bar > CLASS_THRESHOLDS.get(cls, 3.0)),
                }

            results.append({
                "uid": int(uid),
                "true_class": true_cls,
                "predicted": int(pred),
                "correct": pred == true_cls,
                "healthy_score": round(float(h_score), 4),
                "healthy_bar": round(float(healthy_bar), 4),
                "suspicion": bool(h_score < SUSPICION_HCUT),
                "disease_details": disease_details,
            })

    return results

def main():
    data, labels = load_data()
    is_bin = classify_features(data)
    print("Running detailed 10-fold CV (seed 13)...", flush=True)
    results = run_detailed_cv(data, labels, is_bin, seed=13)

    misclassified = [r for r in results if not r["correct"]]
    correct = [r for r in results if r["correct"]]

    print(f"\nTotal: {len(results)}, Correct: {len(correct)}, Wrong: {len(misclassified)}")

    false_neg = [r for r in misclassified if r["true_class"] != HEALTHY and r["predicted"] == HEALTHY]
    false_pos = [r for r in misclassified if r["true_class"] == HEALTHY and r["predicted"] != HEALTHY]
    wrong_disease = [r for r in misclassified
                     if r["true_class"] != HEALTHY and r["predicted"] != HEALTHY
                     and r["true_class"] != r["predicted"]]

    print(f"\nFalse negatives (disease -> healthy): {len(false_neg)}")
    print(f"False positives (healthy -> disease): {len(false_pos)}")
    print(f"Wrong disease class: {len(wrong_disease)}")

    print("\n=== FALSE NEGATIVES (disease predicted as healthy) ===")
    print("These are patients where the healthy bar may have blocked disease detection:")
    hbar_blocked = 0
    near_threshold = 0
    for r in false_neg:
        true_cls = r["true_class"]
        dd = r["disease_details"].get(true_cls, {})
        score = dd.get("score", 0)
        eff_t = dd.get("effective_threshold", 0)
        margin = dd.get("margin", 0)
        hbar_active = dd.get("healthy_bar_active", False)

        if hbar_active:
            hbar_blocked += 1
        if -1.0 < margin < 0:
            near_threshold += 1

        print(f"  Patient {r['uid']}: true={true_cls}, h_score={r['healthy_score']:.3f}, "
              f"healthy_bar={r['healthy_bar']:.3f}, "
              f"disease_score={score:.3f}, eff_threshold={eff_t:.3f}, "
              f"margin={margin:.3f}, hbar_blocked={hbar_active}")

    print(f"\n  Healthy bar blocked true class: {hbar_blocked}/{len(false_neg)}")
    print(f"  Near threshold (margin > -1.0): {near_threshold}/{len(false_neg)}")

    print("\n=== FALSE POSITIVES (healthy predicted as disease) ===")
    for r in false_pos:
        pred = r["predicted"]
        dd = r["disease_details"].get(pred, {})
        print(f"  Patient {r['uid']}: pred={pred}, h_score={r['healthy_score']:.3f}, "
              f"healthy_bar={r['healthy_bar']:.3f}, "
              f"disease_score={dd.get('score',0):.3f}, margin={dd.get('margin',0):.3f}")

    without_hbar = 0
    for r in results:
        if r["true_class"] == HEALTHY:
            continue
        true_cls = r["true_class"]
        dd = r["disease_details"].get(true_cls, {})
        base_t = dd.get("threshold", 3.0)
        score = dd.get("score", 0)
        if score > base_t and not r["correct"]:
            without_hbar += 1

    print(f"\n=== HEALTHY BAR IMPACT SUMMARY ===")
    print(f"Disease patients whose true class score exceeded base threshold")
    print(f"but was blocked by healthy_bar: {hbar_blocked}")
    print(f"Disease patients within 1.0 of threshold: {near_threshold}")

    summary = {
        "total": len(results),
        "correct": len(correct),
        "misclassified": len(misclassified),
        "false_negatives": len(false_neg),
        "false_positives": len(false_pos),
        "wrong_disease": len(wrong_disease),
        "hbar_blocked_true_class": hbar_blocked,
        "near_threshold_count": near_threshold,
        "false_neg_details": [{
            "uid": r["uid"],
            "true_class": r["true_class"],
            "healthy_score": r["healthy_score"],
            "healthy_bar": r["healthy_bar"],
            "true_class_score": r["disease_details"].get(r["true_class"], {}).get("score", 0),
            "effective_threshold": r["disease_details"].get(r["true_class"], {}).get("effective_threshold", 0),
            "margin": r["disease_details"].get(r["true_class"], {}).get("margin", 0),
            "hbar_active": r["disease_details"].get(r["true_class"], {}).get("healthy_bar_active", False),
        } for r in false_neg],
        "false_pos_details": [{
            "uid": r["uid"],
            "predicted": r["predicted"],
            "healthy_score": r["healthy_score"],
            "healthy_bar": r["healthy_bar"],
            "pred_score": r["disease_details"].get(r["predicted"], {}).get("score", 0),
            "pred_margin": r["disease_details"].get(r["predicted"], {}).get("margin", 0),
        } for r in false_pos],
    }

    out_path = os.path.join(os.path.dirname(__file__), "analysis_healthy_bar.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
