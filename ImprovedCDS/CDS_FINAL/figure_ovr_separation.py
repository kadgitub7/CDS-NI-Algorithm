"""Generate data for OVR class separation figure.
For each class's top FDR feature, show value distributions colored by:
  - Target disease class (red)
  - Other disease classes (yellow)
  - Healthy (green)
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import load_data, classify_features, HEALTHY, N_FEAT

def fisher_ratio(vals_target, vals_rest):
    if len(vals_target) < 2 or len(vals_rest) < 2:
        return 0.0
    md2 = (vals_target.mean() - vals_rest.mean()) ** 2
    vs = vals_target.var() + vals_rest.var()
    return md2 / (vs + 1e-10)

def main():
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))

    results = []

    for target_cls in all_cls:
        if target_cls == HEALTHY:
            continue

        best_f, best_fdr = -1, -1
        for f in range(N_FEAT):
            if is_bin[f]:
                continue
            col = data[:, f]
            vm = ~np.isnan(col)
            vv = col[vm]
            lv = labels[vm]
            t_vals = vv[lv == target_cls]
            r_vals = vv[lv != target_cls]
            if len(t_vals) < 2 or len(r_vals) < 2:
                continue
            fdr = fisher_ratio(t_vals, r_vals)
            if fdr > best_fdr:
                best_fdr = fdr
                best_f = f

        if best_f < 0:
            continue

        col = data[:, best_f]
        vm = ~np.isnan(col)
        vv = col[vm]
        lv = labels[vm]

        target_vals = sorted(vv[lv == target_cls].tolist())
        healthy_vals = sorted(vv[lv == HEALTHY].tolist())
        other_disease_vals = sorted(vv[(lv != target_cls) & (lv != HEALTHY)].tolist())

        results.append({
            "target_class": int(target_cls),
            "feature_index": int(best_f),
            "fisher_ratio": round(float(best_fdr), 4),
            "target_values": [round(v, 4) for v in target_vals],
            "healthy_values": [round(v, 4) for v in healthy_vals],
            "other_disease_values": [round(v, 4) for v in other_disease_vals],
            "target_mean": round(float(np.mean(target_vals)), 4),
            "healthy_mean": round(float(np.mean(healthy_vals)), 4),
            "other_mean": round(float(np.mean(other_disease_vals)), 4) if other_disease_vals else 0,
            "target_count": len(target_vals),
            "healthy_count": len(healthy_vals),
            "other_count": len(other_disease_vals),
        })

    results.sort(key=lambda x: -x["fisher_ratio"])

    print("Top OVR separation features:")
    for r in results[:5]:
        print(f"  Class {r['target_class']}, Feature {r['feature_index']}: "
              f"FDR={r['fisher_ratio']:.2f}, "
              f"target_mean={r['target_mean']:.1f}, healthy_mean={r['healthy_mean']:.1f}, "
              f"other_mean={r['other_mean']:.1f}")

    out = {"features": results}
    out_path = os.path.join(os.path.dirname(__file__), "figure_data_ovr.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
