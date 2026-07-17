"""Run CDS-OVR at every split ratio used by any benchmark paper.

Generates results for direct comparison with:
  - Sharma 2024:    70/30, 80/20, 90/10 (binary)
  - Mustaqeem 2018: 50/50, 60/40, 70/30, 80/20, 90/10 (multiclass)
  - Irfan 2022:     60/40 (multiclass)
  - Jadhav 2012:    80/20 (DS1), 75/25 (DS2), 70/30 (DS3), 85/15 (DS4), ~90/10 (DS5)
  - CDS-OVR:        10-fold CV

Each configuration is run with 10 seeds.
Results saved to  CDS_FINAL/results_all_splits.json
"""
import json, time, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from cds_ovr import load_data, classify_features, run_split, run_10fold, stats, binary_acc

SEEDS = [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]

SPLIT_FRACS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

def run_all():
    data, labels = load_data()
    is_bin = classify_features(data)
    n = data.shape[0]
    print(f"Loaded {n} patients, {data.shape[1]} features")
    print(f"Classes: {sorted(set(labels))}\n")

    all_results = {}

    for frac in SPLIT_FRACS:
        tag = f"{int(frac*100)}/{int((1-frac)*100)}"
        print(f"=== Split {tag} ===")
        seed_results = []
        for seed in SEEDS:
            res = run_split(data, labels, is_bin, seed=seed, train_frac=frac)
            acc, spec, sens, ba = stats(res)
            bacc = binary_acc(res)
            n_test = len(res)
            n_correct = sum(r[3] for r in res)

            per_class = {}
            for cls in sorted(set(r[1] for r in res)):
                cr = [r for r in res if r[1] == cls]
                per_class[str(cls)] = {
                    "n_test": len(cr),
                    "correct": sum(r[3] for r in cr),
                    "accuracy": round(sum(r[3] for r in cr) / len(cr), 4) if cr else 0
                }

            tp = sum(1 for r in res if r[1] != 1 and r[2] != 1)
            fn = sum(1 for r in res if r[1] != 1 and r[2] == 1)
            tn = sum(1 for r in res if r[1] == 1 and r[2] == 1)
            fp = sum(1 for r in res if r[1] == 1 and r[2] != 1)

            entry = {
                "seed": seed,
                "n_test": n_test,
                "multiclass_acc": round(100 * acc, 2),
                "binary_acc": round(100 * bacc, 2),
                "specificity": round(100 * spec, 2),
                "sensitivity": round(100 * sens, 2),
                "balanced_acc": round(100 * ba, 2),
                "TP": tp, "FN": fn, "TN": tn, "FP": fp,
                "per_class": per_class
            }
            seed_results.append(entry)
            print(f"  seed={seed:2d}  multi={100*acc:.1f}%  binary={100*bacc:.1f}%  "
                  f"spec={100*spec:.1f}%  sens={100*sens:.1f}%  n_test={n_test}")

        accs = [r["multiclass_acc"] for r in seed_results]
        baccs = [r["binary_acc"] for r in seed_results]
        import numpy as np
        summary = {
            "multi_mean": round(float(np.mean(accs)), 2),
            "multi_std": round(float(np.std(accs)), 2),
            "multi_best": round(float(np.max(accs)), 2),
            "multi_worst": round(float(np.min(accs)), 2),
            "binary_mean": round(float(np.mean(baccs)), 2),
            "binary_std": round(float(np.std(baccs)), 2),
            "binary_best": round(float(np.max(baccs)), 2),
            "binary_worst": round(float(np.min(baccs)), 2),
        }
        print(f"  SUMMARY: multi mean={summary['multi_mean']}% best={summary['multi_best']}% "
              f"std={summary['multi_std']}%")
        print(f"           binary mean={summary['binary_mean']}% best={summary['binary_best']}% "
              f"std={summary['binary_std']}%\n")

        all_results[tag] = {"summary": summary, "seeds": seed_results}

    # 10-fold CV
    print("=== 10-fold CV ===")
    cv_seeds = []
    for seed in SEEDS:
        res = run_10fold(data, labels, is_bin, seed=seed)
        acc, spec, sens, ba = stats(res)
        bacc = binary_acc(res)
        n_test = len(res)

        per_class = {}
        for cls in sorted(set(r[1] for r in res)):
            cr = [r for r in res if r[1] == cls]
            per_class[str(cls)] = {
                "n_test": len(cr),
                "correct": sum(r[3] for r in cr),
                "accuracy": round(sum(r[3] for r in cr) / len(cr), 4) if cr else 0
            }

        entry = {
            "seed": seed,
            "n_test": n_test,
            "multiclass_acc": round(100 * acc, 2),
            "binary_acc": round(100 * bacc, 2),
            "specificity": round(100 * spec, 2),
            "sensitivity": round(100 * sens, 2),
            "balanced_acc": round(100 * ba, 2),
            "per_class": per_class
        }
        cv_seeds.append(entry)
        print(f"  seed={seed:2d}  multi={100*acc:.1f}%  binary={100*bacc:.1f}%  "
              f"spec={100*spec:.1f}%  sens={100*sens:.1f}%")

    import numpy as np
    accs = [r["multiclass_acc"] for r in cv_seeds]
    baccs = [r["binary_acc"] for r in cv_seeds]
    summary = {
        "multi_mean": round(float(np.mean(accs)), 2),
        "multi_std": round(float(np.std(accs)), 2),
        "multi_best": round(float(np.max(accs)), 2),
        "multi_worst": round(float(np.min(accs)), 2),
        "binary_mean": round(float(np.mean(baccs)), 2),
        "binary_std": round(float(np.std(baccs)), 2),
        "binary_best": round(float(np.max(baccs)), 2),
        "binary_worst": round(float(np.min(baccs)), 2),
    }
    print(f"  SUMMARY: multi mean={summary['multi_mean']}% best={summary['multi_best']}% "
          f"std={summary['multi_std']}%\n")
    all_results["10-fold CV"] = {"summary": summary, "seeds": cv_seeds}

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    out_path = os.path.join(os.path.dirname(__file__), "results_all_splits.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    run_all()
