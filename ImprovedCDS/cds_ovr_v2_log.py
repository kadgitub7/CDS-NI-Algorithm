"""CDS OVR V2 — Fully instrumented version.

Every variable touched during training and prediction is logged to a
per-user trace dict, then written to a master JSON file by the runner.
"""
import time
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


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
        if isinstance(obj, frozenset):
            return list(obj)
        return super().default(obj)

U_MIN = 200
HEALTHY = 1
N_FEAT = 279
FORCE_SEX_BRANCHING = True
SEX_FEAT = 1
LAPLACE_ALPHA = 1.0
MIN_SUPPORT = 3
CONF_SUPPORT = 10
RATIO_EPS = 0.1
CORR_THRESHOLD = 0.8
FEATURES_PER_CLASS = 18
MAX_BINS = 6
DISEASE_THRESHOLD = 3.3
BASE_THRESHOLD = 2.5
LARGE_CLASS_N = 20
HEALTHY_WEIGHT = 1.05
SUSPICION_HCUT = 2.0
SUSPICION_OFFSET = 0.3
MERGE_CLASSES = {7, 8, 11, 12}
MERGED_LABEL = 99


def load_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float('nan') if v.strip() == '?' else float(v.strip())
                         for v in line.split(",")])
    raw = np.array(rows, dtype=np.float64)
    data, labels = raw[:, :N_FEAT], raw[:, N_FEAT].astype(int)
    remap = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 14:11, 15:12, 16:13}
    return data, np.array([remap.get(l, l) for l in labels], dtype=int)


def classify_features(data):
    is_bin = np.zeros(N_FEAT, dtype=bool)
    for c in range(N_FEAT):
        v = data[:, c]
        v = v[~np.isnan(v)]
        if len(v) > 0 and set(np.unique(v)).issubset({0.0, 1.0}):
            is_bin[c] = True
    return is_bin


def _hdist(labels, idx):
    d = defaultdict(int)
    for u in idx:
        d[labels[u]] += 1
    return dict(d)


class Node:
    __slots__ = ('nid', 'lvl', 'uidx', 'hdist', 'bfeat', 'bbin',
                 'bvset', 'blo', 'bhi', 'children', 'parent', 'ancestor_feats')

    def __init__(self, nid, lvl, uidx, hdist):
        self.nid = nid
        self.lvl = lvl
        self.uidx = uidx
        self.hdist = hdist
        self.bfeat = self.bvset = self.blo = self.bhi = None
        self.bbin = False
        self.children = []
        self.parent = None
        self.ancestor_feats = frozenset()

    @property
    def nu(self):
        return len(self.uidx)

    def branch_match(self, row):
        if self.bfeat is None:
            return True
        v = row[self.bfeat]
        if np.isnan(v):
            return False
        if self.bbin:
            return v in self.bvset
        if self.bhi == np.inf:
            return v > self.blo
        return v <= self.bhi


class BinModel:
    __slots__ = ('n_bins', 'edges', 'bin_counts', 'target_counts',
                 'p_class', 'prior', 'cls_conf_support')

    def __init__(self, n_bins, edges, bin_counts, target_counts,
                 p_class, prior, cls_conf_support):
        self.n_bins = n_bins
        self.edges = edges
        self.bin_counts = bin_counts
        self.target_counts = target_counts
        self.p_class = p_class
        self.prior = prior
        self.cls_conf_support = cls_conf_support


def build_tree(data, labels, is_bin):
    n = data.shape[0]
    root = Node("root", 1, np.arange(n), _hdist(labels, np.arange(n)))
    all_nodes = [root]
    current_level = [root]
    ctr = [0]

    while current_level:
        lvl = current_level[0].lvl
        if FORCE_SEX_BRANCHING and lvl > 1:
            break
        children = []

        for parent in current_level:
            if FORCE_SEX_BRANCHING:
                feats = [SEX_FEAT]
            else:
                feats = [f for f in range(N_FEAT) if f not in parent.ancestor_feats]

            for f in feats:
                col = data[parent.uidx, f]
                vm = ~np.isnan(col)
                vv = col[vm]
                if len(vv) == 0:
                    continue
                if is_bin[f]:
                    uq = sorted(set(vv))
                    if len(uq) < 2:
                        continue
                    parts = [(parent.uidx[vm & (col == val)],
                              frozenset({val}), None, None, True) for val in uq]
                else:
                    if float(vv.min()) == float(vv.max()):
                        continue
                    med = float(np.median(vv))
                    parts = [
                        (parent.uidx[vm & (col <= med)], None, -np.inf, med, False),
                        (parent.uidx[vm & (col > med)], None, med, np.inf, False),
                    ]
                for cu, vs, lo, hi, ib in parts:
                    if len(cu) < U_MIN:
                        continue
                    ctr[0] += 1
                    ch = Node(f"L{lvl+1}_f{f}_{ctr[0]}", lvl + 1,
                              cu, _hdist(labels, cu))
                    ch.bfeat, ch.bvset, ch.blo, ch.bhi, ch.bbin = f, vs, lo, hi, ib
                    ch.parent = parent
                    ch.ancestor_feats = parent.ancestor_feats | frozenset({f})
                    parent.children.append(ch)
                    children.append(ch)

        seen = {}
        deduped = []
        for ch in children:
            key = np.sort(ch.uidx).tobytes()
            if key not in seen:
                seen[key] = True
                deduped.append(ch)
            else:
                ch.parent.children.remove(ch)

        all_nodes.extend(deduped)
        current_level = deduped

    return all_nodes


def _fast_abs_corr(x, y):
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    num = (dx * dy).sum()
    den2 = (dx * dx).sum() * (dy * dy).sum()
    if den2 <= 0:
        return 0.0
    return abs(num / np.sqrt(den2))


def train_ovr_node(node, data, labels, is_bin, target_class):
    models = {}
    actions = []
    nd_data, nd_labels = data[node.uidx], labels[node.uidx]
    ns = node.nu

    n_target = int((nd_labels == target_class).sum())
    if n_target < 1:
        return models, actions
    prior = n_target / ns

    for f in range(N_FEAT):
        col = nd_data[:, f]
        vm = ~np.isnan(col)
        vv = col[vm]
        nv = len(vv)
        if nv == 0:
            continue

        vmin, vmax = float(vv.min()), float(vv.max())

        if is_bin[f]:
            nb = 1 if vmin == vmax else 2
            edges = (np.array([vmin - .5, vmin + .5]) if nb == 1
                     else np.array([-.5, .5, 1.5]))
        elif vmin == vmax:
            nb, edges = 1, np.array([vmin - .5, vmin + .5])
        else:
            nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
            edges = np.linspace(vmin, vmax, nb + 1)

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]
        bin_counts = np.bincount(ba, minlength=nb)
        target_counts = np.bincount(ba[lv == target_class], minlength=nb).astype(float)
        p_class = (target_counts + LAPLACE_ALPHA * prior) / (bin_counts + LAPLACE_ALPHA)

        models[(node.nid, f)] = BinModel(
            nb, edges, bin_counts, target_counts, p_class, prior, CONF_SUPPORT)

        score = 0.0
        for b in range(nb):
            if bin_counts[b] >= MIN_SUPPORT:
                shift = abs(p_class[b] - prior)
                confidence = min(1.0, float(bin_counts[b]) / CONF_SUPPORT)
                score += shift * confidence

        target_vals = vv[lv == target_class]
        rest_vals = vv[lv != target_class]
        if len(target_vals) >= 2 and len(rest_vals) >= 2:
            mean_diff2 = (target_vals.mean() - rest_vals.mean()) ** 2
            var_sum = target_vals.var() + rest_vals.var()
            fisher = mean_diff2 / (var_sum + 1e-10)
        else:
            fisher = 0.0

        if score > 0.001:
            actions.append((f, node.nid, score, fisher))

    return models, actions


def refine_ovr_node(node, models, node_actions, data):
    scored = [(a, a[2]) for a in node_actions]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []

    candidate_limit = 3 * FEATURES_PER_CLASS
    top_scored = scored[:candidate_limit]

    nd_data = data[node.uidx]
    top_feats = sorted(set(a[0] for a, _ in top_scored))
    correlations = {}
    for i, f1 in enumerate(top_feats):
        col1 = nd_data[:, f1]
        nan1 = np.isnan(col1)
        for f2 in top_feats[i+1:]:
            col2 = nd_data[:, f2]
            valid = ~(nan1 | np.isnan(col2))
            if valid.sum() > 10:
                c = _fast_abs_corr(col1[valid], col2[valid])
                if c > 0:
                    correlations[(f1, f2)] = c
                    correlations[(f2, f1)] = c

    kept = []
    kept_features = set()

    for a, s in top_scored:
        f = a[0]
        if f in kept_features:
            continue
        raw = data[node.uidx, f]
        if (~np.isnan(raw)).sum() == 0:
            continue
        if kept_features:
            max_corr = max(correlations.get((f, kf), 0.0) for kf in kept_features)
            if max_corr > CORR_THRESHOLD:
                continue
        kept.append(a)
        kept_features.add(f)
        if len(kept) >= FEATURES_PER_CLASS:
            break

    return kept


def _route_user(uid, data, nodes):
    by_lvl = defaultdict(list)
    for nd in nodes:
        by_lvl[nd.lvl].append(nd)
    result, active = {}, set()
    for lvl in sorted(by_lvl.keys()):
        if lvl == 1:
            result[1] = [nodes[0]]
            active = {nodes[0].nid}
        else:
            matched = [nd for nd in by_lvl[lvl]
                       if nd.parent and nd.parent.nid in active
                       and nd.branch_match(data[uid])]
            if matched:
                result[lvl] = matched
                active = {nd.nid for nd in matched}
            else:
                break
    return result


def _compute_af_logged(uid, data, nodes, models, retained):
    """Compute Fisher-weighted dual AFs with full per-feature logging."""
    lvl_nodes = _route_user(uid, data, nodes)
    af_for = 0.0
    af_against = 0.0
    n_used = 0

    fisher_map = {}
    for a in retained:
        fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
    max_fisher = max(fisher_map.values()) if fisher_map else 1.0

    route_log = {}
    for lvl in sorted(lvl_nodes.keys()):
        route_log[str(lvl)] = [nd.nid for nd in lvl_nodes[lvl]]

    feature_logs = []

    for lvl in sorted(lvl_nodes.keys()):
        for nd in lvl_nodes[lvl]:
            for a in retained:
                if a[1] != nd.nid:
                    continue
                f = a[0]
                v = data[uid, f]

                feat_entry = {
                    "feat": int(f),
                    "node": nd.nid,
                    "action_score": round(float(a[2]), 6),
                    "action_fisher": round(float(a[3]), 6),
                    "user_value": None if np.isnan(v) else round(float(v), 6),
                    "skipped": None,
                    "bin_idx": None,
                    "bin_count": None,
                    "target_count": None,
                    "n_bins": None,
                    "edges": None,
                    "p_class_bin": None,
                    "prior": None,
                    "shift": None,
                    "confidence": None,
                    "fisher_weight": None,
                    "weighted_contrib": None,
                    "direction": None,
                }

                if np.isnan(v):
                    feat_entry["skipped"] = "nan_value"
                    feature_logs.append(feat_entry)
                    continue

                mo = models.get((nd.nid, f))
                if not mo:
                    feat_entry["skipped"] = "no_model"
                    feature_logs.append(feat_entry)
                    continue

                bin_idx = int(np.clip(
                    np.searchsorted(mo.edges[1:], v, side='right'),
                    0, mo.n_bins - 1
                ))
                bc = mo.bin_counts[bin_idx]

                feat_entry["bin_idx"] = bin_idx
                feat_entry["bin_count"] = int(bc)
                feat_entry["target_count"] = int(mo.target_counts[bin_idx])
                feat_entry["n_bins"] = int(mo.n_bins)
                feat_entry["edges"] = [round(float(e), 6) for e in mo.edges]
                feat_entry["p_class_bin"] = round(float(mo.p_class[bin_idx]), 6)
                feat_entry["p_class_all_bins"] = [round(float(p), 6) for p in mo.p_class]
                feat_entry["bin_counts_all"] = [int(x) for x in mo.bin_counts]
                feat_entry["target_counts_all"] = [int(x) for x in mo.target_counts]
                feat_entry["prior"] = round(float(mo.prior), 6)

                if bc < MIN_SUPPORT:
                    feat_entry["skipped"] = f"low_support_{bc}"
                    feature_logs.append(feat_entry)
                    continue

                p_c = mo.p_class[bin_idx]
                prior = mo.prior
                shift = p_c - prior
                confidence = min(1.0, bc / CONF_SUPPORT)

                fw = np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10))
                fw = max(fw, 0.1)

                weighted = abs(shift) * confidence * fw

                feat_entry["shift"] = round(float(shift), 6)
                feat_entry["confidence"] = round(float(confidence), 6)
                feat_entry["fisher_weight"] = round(float(fw), 6)
                feat_entry["weighted_contrib"] = round(float(weighted), 6)
                feat_entry["direction"] = "FOR" if shift >= 0 else "AGAINST"

                if shift >= 0:
                    af_for += weighted
                else:
                    af_against += weighted
                n_used += 1

                feature_logs.append(feat_entry)

    return af_for, af_against, n_used, {
        "route": route_log,
        "max_fisher": round(float(max_fisher), 6),
        "fisher_map_top10": {str(k): round(float(v), 4)
                             for k, v in sorted(fisher_map.items(),
                                                key=lambda x: -x[1])[:10]},
        "n_retained": len(retained),
        "retained_features": [int(a[0]) for a in retained],
        "features": feature_logs,
        "af_for": round(float(af_for), 6),
        "af_against": round(float(af_against), 6),
        "n_used": n_used,
        "ratio": round(float((af_for + RATIO_EPS) / (af_against + RATIO_EPS)), 6),
    }


def size_based_thresholds(train_labels, all_cls):
    thresholds = {}
    for cls in all_cls:
        if cls == HEALTHY:
            continue
        n_cls = int((train_labels == cls).sum())
        thresholds[cls] = BASE_THRESHOLD if n_cls >= LARGE_CLASS_N else DISEASE_THRESHOLD
    return thresholds


def predict_ovr_logged(uid, data, nodes, class_models, class_retained, all_cls,
                       class_thresholds=None):
    """Three-layer clinical prediction with full logging."""
    class_scores = {}
    class_detail = {}
    class_logs = {}

    for cls in all_cls:
        af_for, af_against, n_used, af_log = _compute_af_logged(
            uid, data, nodes, class_models[cls], class_retained[cls])
        score = (af_for + RATIO_EPS) / (af_against + RATIO_EPS)
        class_scores[cls] = score
        class_detail[cls] = (af_for, af_against, n_used)
        class_logs[str(int(cls))] = af_log

    disease_scores = {c: s for c, s in class_scores.items() if c != HEALTHY}
    h_score = class_scores.get(HEALTHY, 1.0)
    healthy_bar = HEALTHY_WEIGHT * h_score

    decision_log = {
        "all_scores": {str(int(k)): round(float(v), 6) for k, v in class_scores.items()},
        "h_score": round(float(h_score), 6),
        "healthy_bar": round(float(healthy_bar), 6),
        "suspicion_active": h_score < SUSPICION_HCUT,
    }

    if class_thresholds:
        candidates = {}
        candidate_details = {}
        for cls, score in disease_scores.items():
            t = class_thresholds.get(cls, DISEASE_THRESHOLD)
            t_orig = t
            if h_score < SUSPICION_HCUT:
                t -= SUSPICION_OFFSET
            t_after_suspicion = t
            t = max(t, healthy_bar)
            t_final = t

            candidate_details[str(int(cls))] = {
                "score": round(float(score), 6),
                "threshold_base": t_orig,
                "threshold_after_suspicion": round(float(t_after_suspicion), 6),
                "threshold_final": round(float(t_final), 6),
                "margin": round(float(score - t_final), 6),
                "passed": score >= t_final,
            }

            if score >= t_final:
                candidates[cls] = (score - t_final) / max(t_final, 0.1)

        decision_log["candidate_details"] = candidate_details
        decision_log["n_candidates"] = len(candidates)

        if candidates:
            best_cls = max(candidates, key=candidates.get)
            decision_log["candidates"] = {str(int(k)): round(float(v), 6)
                                          for k, v in candidates.items()}
            decision_log["winner"] = int(best_cls)
            decision_log["winner_margin"] = round(float(candidates[best_cls]), 6)
        else:
            best_cls = HEALTHY
            decision_log["winner"] = HEALTHY
            decision_log["reason"] = "no_candidates_passed_threshold"
    else:
        best_disease = max(disease_scores, key=disease_scores.get)
        if disease_scores[best_disease] >= DISEASE_THRESHOLD:
            best_cls = best_disease
        else:
            best_cls = HEALTHY
        decision_log["winner"] = int(best_cls)

    return best_cls, class_scores, {
        "class_af_details": class_logs,
        "decision": decision_log,
        "prediction": int(best_cls),
    }


def _train_all_classes(nodes, data, labels, is_bin, all_cls):
    class_models = {}
    class_retained = {}
    for cls in all_cls:
        cls_models = {}
        cls_actions_by_node = defaultdict(list)
        for nd in nodes:
            nm, na = train_ovr_node(nd, data, labels, is_bin, cls)
            cls_models.update(nm)
            for a in na:
                cls_actions_by_node[a[1]].append(a)
        cls_ret = []
        for nd in nodes:
            cls_ret.extend(
                refine_ovr_node(nd, cls_models,
                                cls_actions_by_node.get(nd.nid, []), data))
        class_models[cls] = cls_models
        class_retained[cls] = cls_ret
    return class_models, class_retained


def log_training_summary(nodes, class_models, class_retained, train_labels, all_cls):
    """Log everything about the trained model."""
    tree_log = []
    for nd in nodes:
        entry = {
            "nid": nd.nid,
            "lvl": nd.lvl,
            "n_users": nd.nu,
            "hdist": {str(int(k)): v for k, v in nd.hdist.items()},
            "branch_feat": int(nd.bfeat) if nd.bfeat is not None else None,
            "branch_is_bin": nd.bbin,
        }
        if nd.bvset is not None:
            entry["branch_values"] = [float(x) for x in nd.bvset]
        if nd.blo is not None:
            entry["branch_lo"] = None if nd.blo == -np.inf else float(nd.blo)
            entry["branch_hi"] = None if nd.bhi == np.inf else float(nd.bhi)
        tree_log.append(entry)

    retained_log = {}
    for cls in all_cls:
        ret = class_retained.get(cls, [])
        retained_log[str(int(cls))] = {
            "n_features": len(ret),
            "features": [
                {
                    "feat": int(a[0]),
                    "node": a[1],
                    "score": round(float(a[2]), 6),
                    "fisher": round(float(a[3]), 6),
                }
                for a in ret
            ],
        }

    model_stats = {}
    for cls in all_cls:
        cls_m = {k: v for k, v in class_models.get(cls, {}).items()}
        n_models = len(cls_m)
        n_cls = int((train_labels == cls).sum())
        model_stats[str(int(cls))] = {
            "n_train_samples": n_cls,
            "n_bin_models": n_models,
            "prior_range": [],
        }
        priors = [m.prior for m in cls_m.values()]
        if priors:
            model_stats[str(int(cls))]["prior_range"] = [
                round(float(min(priors)), 6), round(float(max(priors)), 6)]

    thresholds = size_based_thresholds(train_labels, all_cls)
    threshold_log = {str(int(k)): v for k, v in thresholds.items()}

    return {
        "tree": tree_log,
        "retained_features": retained_log,
        "model_stats": model_stats,
        "thresholds": threshold_log,
        "constants": {
            "U_MIN": U_MIN, "FEATURES_PER_CLASS": FEATURES_PER_CLASS,
            "MAX_BINS": MAX_BINS, "DISEASE_THRESHOLD": DISEASE_THRESHOLD,
            "BASE_THRESHOLD": BASE_THRESHOLD, "LARGE_CLASS_N": LARGE_CLASS_N,
            "HEALTHY_WEIGHT": HEALTHY_WEIGHT, "SUSPICION_HCUT": SUSPICION_HCUT,
            "SUSPICION_OFFSET": SUSPICION_OFFSET, "RATIO_EPS": RATIO_EPS,
            "LAPLACE_ALPHA": LAPLACE_ALPHA, "MIN_SUPPORT": MIN_SUPPORT,
            "CONF_SUPPORT": CONF_SUPPORT, "CORR_THRESHOLD": CORR_THRESHOLD,
            "MERGE_CLASSES": sorted(MERGE_CLASSES), "MERGED_LABEL": MERGED_LABEL,
        },
    }


def print_report(results, label="OVR V2"):
    all_cls = sorted(set(r[1] for r in results) | set(r[2] for r in results))
    n = len(results)
    correct = sum(r[3] for r in results)
    print(f"\n{'='*70}")
    print(f"CDS {label} -- {n} users")
    print(f"{'='*70}")
    print(f"Overall Accuracy:   {100*correct/n:.1f}%")

    per_class = {}
    for cls in all_cls:
        cr = [r for r in results if r[1] == cls]
        if cr:
            per_class[cls] = sum(r[3] for r in cr) / len(cr)
    ba = np.mean(list(per_class.values()))
    print(f"Balanced Accuracy:  {100*ba:.1f}%")

    th = [r for r in results if r[1] == HEALTHY]
    td = [r for r in results if r[1] != HEALTHY]
    spec = sum(r[3] for r in th) / len(th) if th else 0
    sens = sum(1 for r in td if r[2] != HEALTHY) / len(td) if td else 0
    tp = sum(1 for r in td if r[2] != HEALTHY)
    tn = sum(r[3] for r in th)
    fp = len(th) - tn
    fn = len(td) - tp
    binary_acc = (tp + tn) / n
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    detected = [r for r in td if r[2] != HEALTHY]
    subtype_correct = sum(1 for r in detected if r[3])

    print(f"\nBinary (H vs D):")
    print(f"  Accuracy:        {100*binary_acc:.1f}%")
    print(f"  Specificity:     {100*spec:.1f}%  ({tn}/{len(th)})")
    print(f"  Sensitivity:     {100*sens:.1f}%  ({tp}/{len(td)})")
    print(f"  PPV:             {100*ppv:.1f}%")
    print(f"  NPV:             {100*npv:.1f}%")
    if tp:
        print(f"  Subtyping:       {subtype_correct}/{tp} detected = {100*subtype_correct/tp:.1f}% correct class")

    print(f"\nPer-class detail:")
    for cls in sorted(per_class.keys()):
        n_cls = sum(1 for r in results if r[1] == cls)
        n_correct = sum(1 for r in results if r[1] == cls and r[3])
        lbl = "healthy" if cls == HEALTHY else f"class {cls}"
        if cls != HEALTHY:
            n_detected = sum(1 for r in results if r[1] == cls and r[2] != HEALTHY)
            det_str = f"  detected: {n_detected}/{n_cls}"
        else:
            det_str = ""
        wrong = defaultdict(int)
        for r in results:
            if r[1] == cls and not r[3]:
                wrong[r[2]] += 1
        wrong_str = ""
        if wrong:
            wrong_parts = sorted(wrong.items(), key=lambda x: -x[1])[:3]
            wrong_str = "  misclassed as: " + ", ".join(
                f"{'H' if k==1 else f'c{k}'}({v})" for k, v in wrong_parts)
        print(f"  {lbl:>10s}  {n_correct:3d}/{n_cls:3d} = {100*n_correct/n_cls:5.1f}%{det_str}{wrong_str}")
    print(f"{'='*70}")
