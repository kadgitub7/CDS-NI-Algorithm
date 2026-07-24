"""Generate data for supervised vs equal-width binning comparison figure.
Finds the best feature for illustrating the difference and outputs bin edges + distributions.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from cds_ovr import (load_data, classify_features, build_tree,
                     _supervised_bin_edges, MAX_BINS, MIN_SUPPORT_MAP,
                     HEALTHY, N_FEAT)

def equal_width_bins(vv, n_bins):
    vmin, vmax = float(vv.min()), float(vv.max())
    edges = np.linspace(vmin, vmax, n_bins + 1)
    edges[0] -= 1e-10
    edges[-1] += 1e-10
    return edges

def main():
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))

    best_feat = None
    best_score = -1
    best_info = None

    for target_cls in all_cls:
        if target_cls == HEALTHY:
            continue
        ms = MIN_SUPPORT_MAP.get(target_cls, 3)
        is_target = (labels == target_cls).astype(float)

        for f in range(N_FEAT):
            if is_bin[f]:
                continue
            col = data[:, f]
            vm = ~np.isnan(col)
            vv = col[vm]
            if len(vv) < 20:
                continue
            vmin, vmax = float(vv.min()), float(vv.max())
            if vmin == vmax:
                continue

            max_nb = min(max(2, int(np.ceil(1 + np.log2(len(vv))))), MAX_BINS)
            sup_edges = _supervised_bin_edges(vv, is_target[vm], max_nb, ms)
            ew_edges = equal_width_bins(vv, len(sup_edges) - 1)

            sup_bins = np.clip(np.searchsorted(sup_edges[1:], vv, side='right'), 0, len(sup_edges) - 2)
            ew_bins = np.clip(np.searchsorted(ew_edges[1:], vv, side='right'), 0, len(ew_edges) - 2)

            sup_purity = 0
            ew_purity = 0
            for b in range(len(sup_edges) - 1):
                mask = sup_bins == b
                if mask.sum() > 0:
                    t_frac = is_target[vm][mask].mean()
                    sup_purity += mask.sum() * max(t_frac, 1 - t_frac)
            for b in range(len(ew_edges) - 1):
                mask = ew_bins == b
                if mask.sum() > 0:
                    t_frac = is_target[vm][mask].mean()
                    ew_purity += mask.sum() * max(t_frac, 1 - t_frac)

            improvement = (sup_purity - ew_purity) / len(vv)

            if improvement > best_score and len(sup_edges) > 3:
                best_score = improvement
                target_vals = vv[is_target[vm] == 1]
                rest_vals = vv[is_target[vm] == 0]

                sup_bin_data = []
                for b in range(len(sup_edges) - 1):
                    mask = sup_bins == b
                    n_total = int(mask.sum())
                    n_target = int(is_target[vm][mask].sum())
                    sup_bin_data.append({
                        "lo": float(sup_edges[b]),
                        "hi": float(sup_edges[b + 1]),
                        "n_total": n_total,
                        "n_target": n_target,
                        "n_rest": n_total - n_target,
                        "target_frac": round(n_target / n_total, 4) if n_total > 0 else 0
                    })

                ew_bin_data = []
                for b in range(len(ew_edges) - 1):
                    mask = ew_bins == b
                    n_total = int(mask.sum())
                    n_target = int(is_target[vm][mask].sum())
                    ew_bin_data.append({
                        "lo": float(ew_edges[b]),
                        "hi": float(ew_edges[b + 1]),
                        "n_total": n_total,
                        "n_target": n_target,
                        "n_rest": n_total - n_target,
                        "target_frac": round(n_target / n_total, 4) if n_total > 0 else 0
                    })

                # Per-class distribution for multiclass view
                per_class_bins_sup = {}
                per_class_bins_ew = {}
                for cls in all_cls:
                    cls_mask = (labels[vm] == cls) if vm.sum() == len(labels) else None
                    cls_vals = vv[labels[vm] == cls] if vm.sum() == len(labels) else vv[(labels == cls)[vm]]

                per_class_dist = {}
                for cls in all_cls:
                    cls_mask_full = labels == cls
                    cls_col = data[cls_mask_full, f]
                    cls_valid = cls_col[~np.isnan(cls_col)]
                    per_class_dist[str(cls)] = {
                        "values": [round(float(v), 4) for v in sorted(cls_valid)],
                        "count": len(cls_valid)
                    }

                best_feat = f
                best_info = {
                    "feature_index": f,
                    "target_class": target_cls,
                    "n_samples": len(vv),
                    "n_target": int(is_target[vm].sum()),
                    "n_rest": int((1 - is_target[vm]).sum()),
                    "improvement_purity": round(best_score, 4),
                    "supervised_edges": [round(float(e), 4) for e in sup_edges],
                    "equal_width_edges": [round(float(e), 4) for e in ew_edges],
                    "supervised_bins": sup_bin_data,
                    "equal_width_bins": ew_bin_data,
                    "per_class_distribution": per_class_dist,
                    "target_values": [round(float(v), 4) for v in sorted(target_vals)],
                    "rest_values_sample": [round(float(v), 4) for v in sorted(rest_vals)[:50]],
                    "value_range": [round(float(vmin), 4), round(float(vmax), 4)]
                }

    print(f"Best feature: {best_info['feature_index']} for class {best_info['target_class']}")
    print(f"Purity improvement: {best_info['improvement_purity']}")
    print(f"Supervised edges: {best_info['supervised_edges']}")
    print(f"Equal-width edges: {best_info['equal_width_edges']}")
    print(f"\nSupervised bins:")
    for b in best_info['supervised_bins']:
        print(f"  [{b['lo']:.2f}, {b['hi']:.2f}]: {b['n_target']} target / {b['n_rest']} rest = {b['target_frac']:.3f}")
    print(f"\nEqual-width bins:")
    for b in best_info['equal_width_bins']:
        print(f"  [{b['lo']:.2f}, {b['hi']:.2f}]: {b['n_target']} target / {b['n_rest']} rest = {b['target_frac']:.3f}")

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    out_path = os.path.join(os.path.dirname(__file__), "figure_data_binning.json")
    with open(out_path, "w") as f_out:
        json.dump(best_info, f_out, indent=2, cls=NpEncoder)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
