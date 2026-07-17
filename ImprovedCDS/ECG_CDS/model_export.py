"""Export trained CDS-OVR model to JSON for edge deployment.

Trains on the full PhysioNet 2017 dataset, then serializes all model
artifacts (bin edges, class probabilities, retained features, tree
structure, thresholds) so the phone-side classifier can run predictions
without retraining.
"""
import json
import sys
import time
import numpy as np
from pathlib import Path

from cds_ovr_ecg import (
    load_data, classify_features, build_tree, train,
    N_FEAT, HEALTHY, ABNORMAL, U_MIN, LAPLACE_ALPHA, RATIO_EPS,
    CORR_THRESHOLD, FEATURES_PER_CLASS, MAX_BINS, HEALTHY_WEIGHT,
    HEALTHY_BAR_CAP, CLASS_THRESHOLDS, MIN_SUPPORT_MAP, CONF_SUPPORT_MAP,
    AGAINST_SCALE_MAP, TREE_SPLIT_FEAT,
)


def _serialize_node(node):
    """Convert a tree Node to a JSON-safe dict."""
    return {
        "nid": node.nid,
        "lvl": node.lvl,
        "nu": node.nu,
        "hdist": {str(k): v for k, v in node.hdist.items()},
        "bfeat": node.bfeat,
        "bbin": node.bbin,
        "bvset": list(node.bvset) if node.bvset else None,
        "blo": None if node.blo is None else (None if np.isinf(node.blo) and node.blo < 0 else float(node.blo)),
        "bhi": None if node.bhi is None else (None if np.isinf(node.bhi) and node.bhi > 0 else float(node.bhi)),
        "blo_is_neginf": node.blo is not None and np.isinf(node.blo) and node.blo < 0,
        "bhi_is_posinf": node.bhi is not None and np.isinf(node.bhi) and node.bhi > 0,
        "parent_nid": node.parent.nid if node.parent else None,
        "children_nids": [c.nid for c in node.children],
    }


def _serialize_bin_model(model):
    """Convert a BinModel to a JSON-safe dict."""
    return {
        "n_bins": model.n_bins,
        "edges": model.edges.tolist(),
        "bin_counts": model.bin_counts.tolist(),
        "target_counts": model.target_counts.tolist(),
        "p_class": model.p_class.tolist(),
        "prior": float(model.prior),
        "cls_conf_support": int(model.cls_conf_support),
    }


def export_model(output_path="cds_model.json"):
    print("Loading full PhysioNet 2017 dataset...")
    data, labels = load_data()
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    print(f"  {data.shape[0]} records x {data.shape[1]} features")
    print(f"  Normal={int((labels==1).sum())}, Abnormal={int((labels==2).sum())}")

    print("Building tree...")
    t0 = time.time()
    nodes = build_tree(data, labels, is_bin)
    print(f"  {len(nodes)} nodes ({time.time()-t0:.1f}s)")

    print("Training OVR models on full dataset...")
    t0 = time.time()
    class_models, class_retained = train(nodes, data, labels, is_bin, all_cls)
    elapsed = time.time() - t0
    print(f"  Training complete ({elapsed:.1f}s)")

    for cls in all_cls:
        n_models = sum(1 for k in class_models[cls] if class_models[cls][k] is not None)
        n_retained = len(class_retained[cls])
        print(f"  Class {cls}: {n_models} bin models, {n_retained} retained features")

    serialized_nodes = [_serialize_node(nd) for nd in nodes]

    serialized_models = {}
    for cls in all_cls:
        cls_key = str(cls)
        serialized_models[cls_key] = {}

        retained_keys = set()
        for action in class_retained[cls]:
            feat_idx, node_id = action[0], action[1]
            retained_keys.add((node_id, feat_idx))

        for (nid, feat_idx), model in class_models[cls].items():
            if (nid, feat_idx) not in retained_keys:
                continue
            key = f"{nid}|{feat_idx}"
            serialized_models[cls_key][key] = _serialize_bin_model(model)

    serialized_retained = {}
    for cls in all_cls:
        serialized_retained[str(cls)] = [
            {"feat": int(a[0]), "nid": a[1], "score": float(a[2]), "fisher": float(a[3])}
            for a in class_retained[cls]
        ]

    export = {
        "version": "1.0",
        "algorithm": "CDS-OVR",
        "dataset": "PhysioNet 2017 (188 features)",
        "n_features": N_FEAT,
        "n_records_trained": int(data.shape[0]),
        "classes": [int(c) for c in all_cls],
        "class_names": {str(HEALTHY): "Normal", str(ABNORMAL): "Abnormal"},
        "params": {
            "U_MIN": U_MIN,
            "LAPLACE_ALPHA": LAPLACE_ALPHA,
            "RATIO_EPS": RATIO_EPS,
            "CORR_THRESHOLD": CORR_THRESHOLD,
            "FEATURES_PER_CLASS": FEATURES_PER_CLASS,
            "MAX_BINS": MAX_BINS,
            "HEALTHY_WEIGHT": HEALTHY_WEIGHT,
            "HEALTHY_BAR_CAP": HEALTHY_BAR_CAP,
            "CLASS_THRESHOLDS": {str(k): v for k, v in CLASS_THRESHOLDS.items()},
            "AGAINST_SCALE_MAP": {str(k): v for k, v in AGAINST_SCALE_MAP.items()},
            "TREE_SPLIT_FEAT": TREE_SPLIT_FEAT,
        },
        "is_binary_features": [bool(x) for x in is_bin],
        "tree": serialized_nodes,
        "models": serialized_models,
        "retained": serialized_retained,
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out = Path(output_path)
    with open(out, "w") as f:
        json.dump(export, f, indent=1, cls=NumpyEncoder)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nExported model to {out} ({size_mb:.2f} MB)")
    print(f"  Tree nodes: {len(serialized_nodes)}")
    total_models = sum(len(v) for v in serialized_models.values())
    total_retained = sum(len(v) for v in serialized_retained.values())
    print(f"  Bin models (retained only): {total_models}")
    print(f"  Retained features: {total_retained}")
    return str(out)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cds_model.json"
    export_model(out)
