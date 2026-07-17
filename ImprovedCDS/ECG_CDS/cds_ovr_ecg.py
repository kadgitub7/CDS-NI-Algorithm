"""CDS-OVR Arrhythmia Classifier -- PhysioNet 2017 ECG (188 features).

Adapted from the UCI-based CDS_FINAL/cds_ovr.py for real ECG waveform data.
Uses the 188-feature extraction from the PhysioNet Computing in Cardiology
Challenge 2017 dataset (single-lead ECG, 300 Hz, AliveCor KardiaMobile).

Binary classification:
  Class 1 = Normal sinus rhythm (N)
  Class 2 = Abnormal (Atrial Fibrillation + Other rhythms)

Key adaptations from UCI version:
  - N_FEAT = 188 (was 279)
  - Binary OVR (2 classes, was 13)
  - No demographic features (no sex-based tree split)
  - Tree splits on RR variability (feature 7: CVrr) instead of sex
  - Thresholds retuned for binary ECG classification
  - Data loaded from PhysioNet .mat files via 188-feature extraction
"""
import numpy as np
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

# ─── Constants ───

DATA_DIR = str(Path(__file__).parent.parent / "data" / "physioNetData2017")
CACHE_DIR = str(Path(__file__).parent.parent / "data" / "physioNetData2017_cache")

N_FEAT = 188
HEALTHY = 1
ABNORMAL = 2

U_MIN = 200
LAPLACE_ALPHA = 1.0
RATIO_EPS = 0.1
CORR_THRESHOLD = 0.8
FEATURES_PER_CLASS = 25
MAX_BINS = 6
HEALTHY_WEIGHT = 0.9
HEALTHY_BAR_CAP = 3.0

# Binary: only class 2 needs a threshold to exceed for "abnormal" prediction
CLASS_THRESHOLDS = {2: 1.8}

# Per-class tuning (binary is simpler)
MIN_SUPPORT_MAP = {1: 3, 2: 2}
CONF_SUPPORT_MAP = {1: 10, 2: 8}
AGAINST_SCALE_MAP = {1: 0.8, 2: 0.6}

# Feature index for tree split: CVrr (coefficient of variation of RR intervals)
# This is the best single feature for splitting Normal vs AFib populations
TREE_SPLIT_FEAT = 7  # index 7 = "CVrr" in the 188-feature set


# ─── Data loading ───

def load_data(data_dir=None, cache_dir=None, max_records=None):
    """Load PhysioNet 2017 data with 188 features (uses cache if available)."""
    data_dir = data_dir or DATA_DIR
    cache_dir = cache_dir or CACHE_DIR

    tag = f"n{max_records}" if max_records else "all"
    cache_file = Path(cache_dir) / f"features_188_{tag}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        data, labels = cached["data"], cached["labels"]
        return data, labels

    # If no cache, do extraction (requires scipy, the .mat files, etc.)
    from physionet2017_feature_extraction_188 import load_physionet2017_188
    data, labels = load_physionet2017_188(data_dir, max_records=max_records)
    return data, labels


def classify_features(data):
    """Identify binary features (0/1 only)."""
    is_bin = np.zeros(N_FEAT, dtype=bool)
    for c in range(N_FEAT):
        v = data[:, c]
        v = v[~np.isnan(v)]
        if len(v) > 0 and set(np.unique(v)).issubset({0.0, 1.0}):
            is_bin[c] = True
    return is_bin


# ─── Tree structure ───

def _hdist(labels, idx):
    d = defaultdict(int)
    for u in idx:
        d[labels[u]] += 1
    return dict(d)


class Node:
    __slots__ = ('nid', 'lvl', 'uidx', 'hdist', 'bfeat', 'bbin',
                 'bvset', 'blo', 'bhi', 'children', 'parent', 'ancestor_feats')
    def __init__(self, nid, lvl, uidx, hdist):
        self.nid, self.lvl, self.uidx, self.hdist = nid, lvl, uidx, hdist
        self.bfeat = self.bvset = self.blo = self.bhi = None
        self.bbin = False
        self.children, self.parent = [], None
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
        self.n_bins, self.edges = n_bins, edges
        self.bin_counts, self.target_counts = bin_counts, target_counts
        self.p_class, self.prior, self.cls_conf_support = p_class, prior, cls_conf_support


def build_tree(data, labels, is_bin):
    """Build CDS tree using CVrr as the split feature (replaces sex-based split)."""
    n = data.shape[0]
    root = Node("root", 1, np.arange(n), _hdist(labels, np.arange(n)))
    all_nodes = [root]
    current_level = [root]
    ctr = [0]

    while current_level:
        lvl = current_level[0].lvl
        if lvl > 1:
            break
        children = []
        for parent in current_level:
            # At level 1, split on CVrr (high vs low RR variability)
            # At level 2, split on other discriminative features
            if lvl == 1:
                feats = [TREE_SPLIT_FEAT]
            else:
                # Use a few top discriminative features for level 2
                feats = [0, 2, 7, 71, 104]  # AFEvidence, IrrEvidence, CVrr, freq, soa

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
                    parts = [(parent.uidx[vm & (col == val)], frozenset({val}), None, None, True)
                             for val in uq]
                else:
                    if float(vv.min()) == float(vv.max()):
                        continue
                    med = float(np.median(vv))
                    parts = [(parent.uidx[vm & (col <= med)], None, -np.inf, med, False),
                             (parent.uidx[vm & (col > med)], None, med, np.inf, False)]

                for cu, vs, lo, hi, ib in parts:
                    if len(cu) < U_MIN:
                        continue
                    ctr[0] += 1
                    ch = Node(f"L{lvl+1}_f{f}_{ctr[0]}", lvl+1, cu, _hdist(labels, cu))
                    ch.bfeat, ch.bvset, ch.blo, ch.bhi, ch.bbin = f, vs, lo, hi, ib
                    ch.parent = parent
                    ch.ancestor_feats = parent.ancestor_feats | frozenset({f})
                    parent.children.append(ch)
                    children.append(ch)

        seen, deduped = {}, []
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


# ─── Utilities ───

def _fast_abs_corr(x, y):
    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    num = (dx * dy).sum()
    den2 = (dx * dx).sum() * (dy * dy).sum()
    return abs(num / np.sqrt(den2)) if den2 > 0 else 0.0


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
                       if nd.parent and nd.parent.nid in active and nd.branch_match(data[uid])]
            if matched:
                result[lvl] = matched
                active = {nd.nid for nd in matched}
            else:
                break
    return result


# ─── Supervised binning ───

def _supervised_bin_edges(vv, is_target, max_bins, min_support):
    n = len(vv)
    vmin, vmax = float(vv.min()), float(vv.max())
    if vmin == vmax or n < 2 * min_support:
        return np.array([vmin - 0.5, vmax + 0.5])

    sort_idx = np.argsort(vv)
    sv = vv[sort_idx]
    st = is_target[sort_idx]
    edges = [vmin, vmax]

    for _ in range(max_bins - 1):
        best_gain, best_split, best_seg = 0.0, None, None
        for seg_i in range(len(edges) - 1):
            lo, hi = edges[seg_i], edges[seg_i + 1]
            if seg_i == 0:
                mask = sv <= hi
            elif seg_i == len(edges) - 2:
                mask = sv > lo
            else:
                mask = (sv > lo) & (sv <= hi)
            seg_vals, seg_targ = sv[mask], st[mask]
            n_seg = len(seg_vals)
            if n_seg < 2 * min_support:
                continue
            n_t, n_r = seg_targ.sum(), n_seg - seg_targ.sum()
            if n_t == 0 or n_r == 0:
                continue
            candidates = np.where(seg_vals[:-1] != seg_vals[1:])[0]
            if len(candidates) == 0:
                continue
            cum_t = np.cumsum(seg_targ)
            for ci in candidates:
                n_left, n_right = ci + 1, n_seg - ci - 1
                if n_left < min_support or n_right < min_support:
                    continue
                t_left = cum_t[ci]
                t_right = n_t - t_left
                r_left, r_right = n_left - t_left, n_right - t_right
                e_tl = n_left * n_t / n_seg
                e_tr = n_right * n_t / n_seg
                e_rl = n_left * n_r / n_seg
                e_rr = n_right * n_r / n_seg
                chi2 = sum((o - e)**2 / e for o, e in
                           [(t_left, e_tl), (t_right, e_tr),
                            (r_left, e_rl), (r_right, e_rr)] if e > 0)
                if chi2 > best_gain:
                    best_gain = chi2
                    best_split = (seg_vals[ci] + seg_vals[ci + 1]) / 2.0
                    best_seg = seg_i
        if best_split is None or best_gain < 0.5:
            break
        edges.insert(best_seg + 1, best_split)

    edges[0] = vmin - 1e-10
    edges[-1] = vmax + 1e-10
    return np.array(sorted(set(edges)))


# ─── Training ───

def _train_ovr_node(node, data, labels, is_bin, target_class, min_support, conf_support):
    models, actions = {}, []
    nd_data, nd_labels = data[node.uidx], labels[node.uidx]
    ns = node.nu
    n_target = int((nd_labels == target_class).sum())
    if n_target < 1:
        return models, actions
    prior = n_target / ns
    is_target = (nd_labels == target_class)

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
            edges = np.array([vmin - .5, vmin + .5]) if nb == 1 else np.array([-.5, .5, 1.5])
        elif vmin == vmax:
            nb, edges = 1, np.array([vmin - .5, vmin + .5])
        else:
            max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
            edges = _supervised_bin_edges(vv, is_target[vm].astype(float), max_nb, min_support)
            nb = len(edges) - 1

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]
        bin_counts = np.bincount(ba, minlength=nb)
        target_counts = np.bincount(ba[lv == target_class], minlength=nb).astype(float)
        p_class = (target_counts + LAPLACE_ALPHA * prior) / (bin_counts + LAPLACE_ALPHA)

        models[(node.nid, f)] = BinModel(nb, edges, bin_counts, target_counts,
                                         p_class, prior, conf_support)

        score = 0.0
        for b in range(nb):
            if bin_counts[b] >= min_support:
                shift = abs(p_class[b] - prior)
                confidence = min(1.0, float(bin_counts[b]) / conf_support)
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


def _refine_ovr_node(node, models, node_actions, data, fpc=FEATURES_PER_CLASS):
    scored = [(a, a[2]) for a in node_actions]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []

    top_scored = scored[:3 * fpc]
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
    kept, kept_features = [], set()
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
        if len(kept) >= fpc:
            break
    return kept


def train(nodes, data, labels, is_bin, all_cls):
    """Train OVR models for all classes."""
    class_models, class_retained = {}, {}

    for cls in all_cls:
        ms = MIN_SUPPORT_MAP.get(cls, 3)
        cs = CONF_SUPPORT_MAP.get(cls, 10)

        cls_models = {}
        cls_actions = defaultdict(list)
        for nd in nodes:
            nm, na = _train_ovr_node(nd, data, labels, is_bin, cls, ms, cs)
            cls_models.update(nm)
            for a in na:
                cls_actions[a[1]].append(a)

        cls_ret = []
        for nd in nodes:
            cls_ret.extend(_refine_ovr_node(
                nd, cls_models, cls_actions.get(nd.nid, []), data))

        class_models[cls] = cls_models
        class_retained[cls] = cls_ret

    return class_models, class_retained


# ─── Prediction ───

def _compute_af(uid, data, nodes, models, retained, against_scale):
    lvl_nodes = _route_user(uid, data, nodes)
    af_for, af_against = 0.0, 0.0
    n_for, n_against, n_used = 0, 0, 0
    max_for_contrib = 0.0

    fisher_map = {}
    for a in retained:
        fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
    max_fisher = max(fisher_map.values()) if fisher_map else 1.0

    for lvl in sorted(lvl_nodes.keys()):
        for nd in lvl_nodes[lvl]:
            for a in retained:
                if a[1] != nd.nid:
                    continue
                f = a[0]
                v = data[uid, f]
                if np.isnan(v):
                    continue
                mo = models.get((nd.nid, f))
                if not mo:
                    continue
                bin_idx = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                     0, mo.n_bins - 1))
                bc = mo.bin_counts[bin_idx]
                if bc < 3:
                    continue
                p_c = mo.p_class[bin_idx]
                shift = p_c - mo.prior
                confidence = min(1.0, bc / 10)
                fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
                weighted = abs(shift) * confidence * fw
                if shift >= 0:
                    af_for += weighted
                    n_for += 1
                    if weighted > max_for_contrib:
                        max_for_contrib = weighted
                else:
                    af_against += weighted * against_scale
                    n_against += 1
                n_used += 1

    return af_for, af_against, n_used, n_for, n_against, max_for_contrib


def predict(uid, data, nodes, all_cls, train_result):
    """Predict class for a single patient (binary: Normal vs Abnormal)."""
    class_models, class_retained = train_result
    class_scores = {}

    for cls in all_cls:
        ag = AGAINST_SCALE_MAP.get(cls, 0.8)
        af = _compute_af(uid, data, nodes, class_models[cls], class_retained[cls], ag)
        class_scores[cls] = (af[0] + RATIO_EPS) / (af[1] + RATIO_EPS)

    h_score = class_scores.get(HEALTHY, 1.0)
    abnormal_score = class_scores.get(ABNORMAL, 0.0)

    threshold = CLASS_THRESHOLDS.get(ABNORMAL, 2.5)
    healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
    effective_threshold = max(threshold, healthy_bar)

    if abnormal_score >= effective_threshold:
        return ABNORMAL, class_scores
    else:
        return HEALTHY, class_scores


# ─── Evaluation ───

def run_split(data, labels, is_bin, seed=13, train_frac=0.9):
    """Run a single train/test split."""
    n = data.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]
    td, tl = data[train_idx], labels[train_idx]
    all_cls = sorted(set(labels))
    nodes = build_tree(td, tl, is_bin)
    train_result = train(nodes, td, tl, is_bin, all_cls)

    results = []
    for uid in test_idx:
        true_cls = int(labels[uid])
        pred, scores = predict(uid, data, nodes, all_cls, train_result)
        results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def run_10fold(data, labels, is_bin, seed=13):
    """Run 10-fold cross-validation."""
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
        for uid in test_idx:
            true_cls = int(labels[uid])
            pred, scores = predict(uid, data, nodes, all_cls, train_result)
            results.append((int(uid), true_cls, int(pred), pred == true_cls))
    return results


def stats(results):
    """Compute accuracy, specificity, sensitivity, F1."""
    n = len(results)
    correct = sum(r[3] for r in results)
    acc = correct / n

    # Specificity: correctly identified normals
    normals = [r for r in results if r[1] == HEALTHY]
    spec = sum(r[3] for r in normals) / len(normals) if normals else 0

    # Sensitivity: correctly identified abnormals
    abnormals = [r for r in results if r[1] == ABNORMAL]
    sens = sum(r[3] for r in abnormals) / len(abnormals) if abnormals else 0

    # F1 for abnormal class
    tp = sum(1 for r in results if r[1] == ABNORMAL and r[2] == ABNORMAL)
    fp = sum(1 for r in results if r[1] == HEALTHY and r[2] == ABNORMAL)
    fn = sum(1 for r in results if r[1] == ABNORMAL and r[2] == HEALTHY)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sens
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return acc, spec, sens, f1


# ─── Main ───

if __name__ == "__main__":
    print("=" * 60)
    print("CDS-OVR for PhysioNet 2017 ECG (188 features)")
    print("=" * 60)

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 13

    print(f"\nLoading PhysioNet 2017 data (188 features)...")
    data, labels = load_data()
    is_bin = classify_features(data)
    print(f"  {data.shape[0]} records x {data.shape[1]} features")
    print(f"  Classes: {sorted(set(labels))}")
    print(f"  Distribution: Normal={int((labels==1).sum())}, Abnormal={int((labels==2).sum())}")
    print(f"  Class ratio: {(labels==1).sum()/(labels==2).sum():.2f} (Normal/Abnormal)")

    # 90/10 split
    print(f"\nRunning 90/10 split (seed={seed})...", flush=True)
    t0 = time.time()
    results_90 = run_split(data, labels, is_bin, seed, 0.9)
    elapsed = time.time() - t0
    acc, spec, sens, f1 = stats(results_90)
    print(f"  Accuracy:    {100*acc:.1f}%")
    print(f"  Specificity: {100*spec:.1f}% (Normal correct)")
    print(f"  Sensitivity: {100*sens:.1f}% (Abnormal correct)")
    print(f"  F1 (Abnorm): {100*f1:.1f}%")
    print(f"  Time:        {elapsed:.1f}s")

    print("=" * 60)
