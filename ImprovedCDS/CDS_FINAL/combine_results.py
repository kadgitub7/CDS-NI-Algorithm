"""Combine parallel task outputs into comprehensive_results.json."""
import json, os, sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    combined = {}
    files = {
        "auc_roc": "results_auc.json",
        "60_40_stratified": "results_stratified.json",
        "computational_cost": "results_cost.json",
        "confusion_matrix_seed13": "results_confusion.json",
    }

    for key, fname in files.items():
        path = os.path.join(SRC_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                combined[key] = json.load(f)
            print(f"  Loaded {fname}")
        else:
            print(f"  MISSING: {fname}")

    out_path = os.path.join(SRC_DIR, "comprehensive_results.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved to {out_path}")
    print(f"Keys present: {list(combined.keys())}")


if __name__ == "__main__":
    main()
