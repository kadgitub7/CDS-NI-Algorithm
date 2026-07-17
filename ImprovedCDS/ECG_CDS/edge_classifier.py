"""Edge CDS-OVR Classifier — runs predictions from an exported model JSON.

This is the inference-only engine. It loads the model exported by
model_export.py, takes a raw ECG signal, extracts 188 features, and
returns a Normal/Abnormal prediction with confidence scores.

No training code, no numpy-heavy operations beyond feature extraction.
Designed to run on a phone (via Kivy/Termux) or any Python environment.
"""
import json
import math
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

from physionet2017_feature_extraction_188 import extract_features_188


class CdsEdgeModel:
    """Lightweight CDS-OVR model loaded from exported JSON."""

    def __init__(self, model_path: str):
        with open(model_path, "r") as f:
            self._raw = json.load(f)

        self.n_features = self._raw["n_features"]
        self.classes = self._raw["classes"]
        self.class_names = self._raw["class_names"]
        self.params = self._raw["params"]
        self.is_binary = self._raw["is_binary_features"]

        self._nodes = self._raw["tree"]
        self._node_map = {nd["nid"]: nd for nd in self._nodes}
        self._models = self._raw["models"]
        self._retained = self._raw["retained"]

    def _branch_match(self, node: dict, features: np.ndarray) -> bool:
        """Check if a feature vector matches a node's branch condition."""
        bfeat = node["bfeat"]
        if bfeat is None:
            return True
        v = features[bfeat]
        if np.isnan(v):
            return False
        if node["bbin"]:
            return v in node["bvset"]

        if node["bhi_is_posinf"]:
            blo = node["blo"] if node["blo"] is not None else -math.inf
            return v > blo
        else:
            bhi = node["bhi"] if node["bhi"] is not None else math.inf
            return v <= bhi

    def _route(self, features: np.ndarray) -> Dict[int, list]:
        """Route a feature vector through the tree, returning active nodes per level."""
        by_lvl: Dict[int, list] = {}
        for nd in self._nodes:
            by_lvl.setdefault(nd["lvl"], []).append(nd)

        result = {}
        active_nids = set()

        for lvl in sorted(by_lvl.keys()):
            if lvl == 1:
                result[1] = [self._nodes[0]]
                active_nids = {self._nodes[0]["nid"]}
            else:
                matched = [
                    nd for nd in by_lvl[lvl]
                    if nd["parent_nid"] in active_nids
                    and self._branch_match(nd, features)
                ]
                if matched:
                    result[lvl] = matched
                    active_nids = {nd["nid"] for nd in matched}
                else:
                    break
        return result

    def _compute_af(self, features: np.ndarray, cls: int) -> Tuple[float, float, int]:
        """Compute affinity-for and affinity-against scores for a class."""
        cls_key = str(cls)
        retained = self._retained.get(cls_key, [])
        models = self._models.get(cls_key, {})
        against_scale = self.params["AGAINST_SCALE_MAP"].get(cls_key, 0.8)

        lvl_nodes = self._route(features)
        af_for, af_against = 0.0, 0.0
        n_used = 0

        fisher_map = {}
        for a in retained:
            f = a["feat"]
            fisher_map[f] = max(fisher_map.get(f, 0), a["fisher"])
        max_fisher = max(fisher_map.values()) if fisher_map else 1.0

        for lvl in sorted(lvl_nodes.keys()):
            for nd in lvl_nodes[lvl]:
                for a in retained:
                    if a["nid"] != nd["nid"]:
                        continue
                    f = a["feat"]
                    v = features[f]
                    if np.isnan(v):
                        continue

                    model_key = f"{nd['nid']}|{f}"
                    mo = models.get(model_key)
                    if mo is None:
                        continue

                    edges = mo["edges"]
                    n_bins = mo["n_bins"]
                    bin_idx = 0
                    for ei in range(1, len(edges)):
                        if v > edges[ei]:
                            bin_idx = min(ei, n_bins - 1)
                        else:
                            break

                    bc = mo["bin_counts"][bin_idx]
                    if bc < 3:
                        continue

                    p_c = mo["p_class"][bin_idx]
                    prior = mo["prior"]
                    shift = p_c - prior
                    confidence = min(1.0, bc / 10)
                    fw = max(math.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                    weighted = abs(shift) * confidence * fw

                    if shift >= 0:
                        af_for += weighted
                    else:
                        af_against += weighted * against_scale
                    n_used += 1

        return af_for, af_against, n_used

    def predict(self, features: np.ndarray) -> Tuple[int, str, Dict]:
        """Predict class from a 188-element feature vector.

        Returns:
            (predicted_class, class_name, details_dict)
        """
        ratio_eps = self.params["RATIO_EPS"]
        healthy_weight = self.params["HEALTHY_WEIGHT"]
        healthy_bar_cap = self.params["HEALTHY_BAR_CAP"]
        class_thresholds = self.params["CLASS_THRESHOLDS"]

        class_scores = {}
        class_details = {}

        for cls in self.classes:
            af_for, af_against, n_used = self._compute_af(features, cls)
            ratio = (af_for + ratio_eps) / (af_against + ratio_eps)
            class_scores[cls] = ratio
            class_details[cls] = {
                "af_for": round(af_for, 4),
                "af_against": round(af_against, 4),
                "ratio": round(ratio, 4),
                "n_features_used": n_used,
            }

        healthy_cls = 1
        abnormal_cls = 2
        h_score = class_scores.get(healthy_cls, 1.0)
        a_score = class_scores.get(abnormal_cls, 0.0)

        threshold = class_thresholds.get(str(abnormal_cls), 2.5)
        healthy_bar = min(healthy_weight * h_score, healthy_bar_cap)
        effective_threshold = max(threshold, healthy_bar)

        if a_score >= effective_threshold:
            pred = abnormal_cls
        else:
            pred = healthy_cls

        details = {
            "class_scores": {str(k): round(v, 4) for k, v in class_scores.items()},
            "class_details": {str(k): v for k, v in class_details.items()},
            "effective_threshold": round(effective_threshold, 4),
            "prediction": pred,
        }

        return pred, self.class_names[str(pred)], details

    def predict_from_signal(self, signal: np.ndarray, fs: int = 300) -> Tuple[int, str, Dict]:
        """Full pipeline: raw ECG signal -> feature extraction -> prediction."""
        features = extract_features_188(signal, fs)
        return self.predict(features)


def load_model(model_path: str = "cds_model.json") -> CdsEdgeModel:
    """Load an exported CDS model."""
    return CdsEdgeModel(model_path)


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "cds_model.json"
    p = Path(model_path)
    if not p.exists():
        print(f"Model file not found: {model_path}")
        print("Run model_export.py first to generate the model.")
        sys.exit(1)

    model = load_model(model_path)
    print(f"Loaded CDS model: {model.n_features} features, classes={model.classes}")
    print(f"Params: {json.dumps(model.params, indent=2)}")

    # Test with synthetic ECG
    print("\nGenerating synthetic ECG for testing...")
    rng = np.random.RandomState(42)
    fs = 300
    t = np.arange(0, 30, 1.0 / fs)
    ecg = np.zeros_like(t)
    for bt in np.arange(0.5, 29.5, 0.8):
        ecg += 1.5 * np.exp(-0.5 * ((t - bt) / 0.02) ** 2)
        ecg += 0.3 * np.exp(-0.5 * ((t - bt + 0.16) / 0.04) ** 2)
        ecg += 0.4 * np.exp(-0.5 * ((t - bt - 0.2) / 0.06) ** 2)
        ecg -= 0.2 * np.exp(-0.5 * ((t - bt - 0.04) / 0.015) ** 2)
    ecg += 0.05 * rng.randn(len(t))

    pred_cls, pred_name, details = model.predict_from_signal(ecg, fs)
    print(f"\nPrediction: {pred_name} (class {pred_cls})")
    print(f"Scores: {details['class_scores']}")
    print(f"Threshold: {details['effective_threshold']}")
    for cls_key, d in details["class_details"].items():
        name = model.class_names[cls_key]
        print(f"  {name}: for={d['af_for']}, against={d['af_against']}, "
              f"ratio={d['ratio']}, features_used={d['n_features_used']}")
