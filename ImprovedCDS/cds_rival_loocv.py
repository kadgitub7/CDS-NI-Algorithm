"""
cds_rival_loocv.py

Phase 1: rival-aware feature retention.

Principle: when selecting which 18 features to retain per class, boost
features that specifically separate the target class from its empirically
observed top rival (derived from the baseline LOOCV confusion matrix),
rather than ranking purely on target-vs-rest discrimination.

GUARD: This change operates EXCLUSIVELY on _refine_ovr_node (feature
selection). It does NOT touch _compute_af, predict(), CLASS_THRESHOLDS,
or the decision rule. The rival_fisher value is used only to re-rank
which features enter class_retained before training completes. It never
enters the inference-time scoring path.

Implementation:
  - _train_ovr_node is called UNCHANGED.
  - After each node's actions are collected, rival Fisher scores are
    computed in a post-processing step and appended as a[4] to each
    action tuple: (f, node.nid, score, fisher, rival_fisher).
  - _compute_af accesses a[0]=f, a[1]=node.nid, a[3]=fisher -- unchanged.
  - _rival_refine_ovr_node re-ranks the candidate pool by:
        combined = score * (1 + RIVAL_ALPHA * rival_fisher / max_rival_fisher)
    then applies the same correlation filter as the baseline.

RIVAL_MAP (from loocv_baseline_results.json confusion matrix):
  c1 (HEALTHY): H->c2 12 errors           → rival = c2
  c2:           c2->H  6 errors            → rival = HEALTHY
  c3:           c3->c2 1 error (only)      → rival = c2
  c4:           c4->H  3 errors            → rival = HEALTHY
  c5:           c5->H  3 errors            → rival = HEALTHY
  c6:           tied H/c3/c4/c10 at 1 each → rival = HEALTHY (prevalence)
  c9:           no errors                  → rival = HEALTHY (default)
  c10:          c10->H 9 errors            → rival = HEALTHY
"""
import json
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

from cds_ovr_bagging import (
    load_data, classify_features, build_tree,
    _train_ovr_node, _refine_ovr_node,
    _compute_af, predict, stats, confusion_matrix,
    print_full_report, save_results,
    FEATURES_PER_CLASS, CORR_THRESHOLD, HEALTHY, N_FEAT,
    MIN_SUPPORT_MAP, CONF_SUPPORT_MAP,
    _fast_abs_corr, _NumpyEncoder,
    train as baseline_train,
)

_DIR = Path(__file__).parent

# ─── Rival map ─────────────────────────────────────────────────────────────
RIVAL_MAP = {
    1:  2,  # HEALTHY → c2 (12 H→c2 errors)
    2:  1,  # c2 → HEALTHY (6 c2→H errors)
    3:  2,  # c3 → c2 (1 error; only rival)
    4:  1,  # c4 → HEALTHY (3 errors)
    5:  1,  # c5 → HEALTHY (3 errors)
    6:  1,  # c6 → HEALTHY (tied; HEALTHY by prevalence)
    9:  1,  # c9 → HEALTHY (no errors; default)
    10: 1,  # c10 → HEALTHY (9 errors)
}

# Boost weight for rival-discrimination term.
# combined_score = score * (1 + RIVAL_ALPHA * rival_fisher / max_rival_fisher)
# alpha=0 reproduces baseline exactly; alpha=0.5 gives up to 50% boost.
RIVAL_ALPHA = 0.5


# ─── Feature name helper ────────────────────────────────────────────────────
_FEAT_NAMES = {
    0: "Age", 1: "Sex", 2: "Height", 3: "Weight",
    4: "QRS_dur", 5: "PR_int", 6: "QT_int", 7: "T_int",
    8: "P_int", 9: "QRS_mean", 10: "T_mean", 11: "P_mean",
    12: "QRST_mean", 13: "J_mean", 14: "HR",
}

def feat_name(f):
    return _FEAT_NAMES.get(f, f"f{f}")


# ─── Rival Fisher post-processing ──────────────────────────────────────────

def _compute_rival_fisher(node, data, labels, target_class, rival_class, feat_set):
    """
    Compute Fisher F-ratio for target_class vs. rival_class only.
    Called after _train_ovr_node; feat_set = features that made it into actions.
    Returns dict {f: rival_fisher}.
    """
    nd_data   = data[node.uidx]
    nd_labels = labels[node.uidx]
    out = {}
    rival_mask = (nd_labels == rival_class)
    target_mask = (nd_labels == target_class)
    for f in feat_set:
        col = nd_data[:, f]
        vm  = ~np.isnan(col)
        vv  = col[vm]
        lv  = nd_labels[vm]
        tv  = vv[lv == target_class]
        rv  = vv[lv == rival_class]
        if len(tv) >= 2 and len(rv) >= 2:
            md2 = (tv.mean() - rv.mean()) ** 2
            vs  = tv.var() + rv.var()
            out[f] = md2 / (vs + 1e-10)
        else:
            out[f] = 0.0
    return out


# ─── Modified _refine_ovr_node ─────────────────────────────────────────────

def _rival_refine_ovr_node(node, models, node_actions, data,
                           fpc=FEATURES_PER_CLASS):
    """
    Like _refine_ovr_node but re-ranks the candidate pool using rival_fisher.

    node_actions must have 5-element tuples:
        (f, node.nid, score, fisher, rival_fisher)

    combined_score = score * (1 + RIVAL_ALPHA * rival_fisher / max_rival_fisher)

    The correlation filter and fpc cap are identical to baseline.
    _compute_af is not called here; rival_fisher never reaches inference.
    """
    if not node_actions:
        return []

    # Sort by original score to pick the initial candidate pool (same as baseline)
    by_score = sorted(node_actions, key=lambda a: a[2], reverse=True)
    top_scored = by_score[:3 * fpc]

    # Re-rank within the pool using the rival-aware combined score
    max_rival = max(a[4] for a in top_scored) if top_scored else 0.0
    if max_rival > 0:
        combined = [
            (a, a[2] * (1.0 + RIVAL_ALPHA * a[4] / max_rival))
            for a in top_scored
        ]
    else:
        # No rival information available (e.g. rival class absent in node)
        combined = [(a, a[2]) for a in top_scored]
    combined.sort(key=lambda x: x[1], reverse=True)

    # Correlation filter — identical logic to _refine_ovr_node
    nd_data   = data[node.uidx]
    top_feats = sorted(set(a[0] for a, _ in combined))
    correlations = {}
    for i, f1 in enumerate(top_feats):
        col1 = nd_data[:, f1]
        nan1 = np.isnan(col1)
        for f2 in top_feats[i + 1:]:
            col2 = nd_data[:, f2]
            valid = ~(nan1 | np.isnan(col2))
            if valid.sum() > 10:
                c = _fast_abs_corr(col1[valid], col2[valid])
                if c > 0:
                    correlations[(f1, f2)] = c
                    correlations[(f2, f1)] = c

    kept, kept_features = [], set()
    for a, _ in combined:
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


# ─── Modified train ────────────────────────────────────────────────────────

def rival_train(nodes, data, labels, is_bin, all_cls):
    """
    Like train() but uses rival-aware feature selection.

    _train_ovr_node is called UNCHANGED to produce models and 4-element actions.
    A post-processing step adds rival_fisher as a[4].
    _rival_refine_ovr_node then re-ranks the candidate pool.
    class_models is identical to what baseline train() produces.
    Only class_retained differs.
    """
    class_models, class_retained = {}, {}
    for cls in all_cls:
        rival_cls = RIVAL_MAP.get(cls, HEALTHY)
        ms = MIN_SUPPORT_MAP.get(cls, 3)
        cs = CONF_SUPPORT_MAP.get(cls, 10)
        cls_models  = {}
        cls_actions = defaultdict(list)

        for nd in nodes:
            # Unchanged original training
            nm, na = _train_ovr_node(nd, data, labels, is_bin, cls, ms, cs)
            cls_models.update(nm)

            if na:
                # Post-processing: compute rival Fisher for action features
                feat_set = set(a[0] for a in na)
                rf = _compute_rival_fisher(nd, data, labels, cls, rival_cls, feat_set)
                # Extend action tuple: (f, nid, score, fisher) → (f, nid, score, fisher, rival_f)
                # _compute_af accesses a[0]=f, a[1]=nid, a[3]=fisher -- unchanged
                for a in na:
                    cls_actions[a[1]].append(
                        (a[0], a[1], a[2], a[3], rf.get(a[0], 0.0))
                    )

        cls_ret = []
        for nd in nodes:
            cls_ret.extend(_rival_refine_ovr_node(
                nd, cls_models, cls_actions.get(nd.nid, []), data))

        class_models[cls]   = cls_models
        class_retained[cls] = cls_ret

    return class_models, class_retained


# ─── LOOCV ─────────────────────────────────────────────────────────────────

def run_rival_loocv(data, labels, is_bin):
    """Full LOOCV using rival-aware feature retention. predict() unchanged."""
    n       = data.shape[0]
    all_cls = sorted(set(labels))
    results = []
    t0      = time.perf_counter()

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        td, tl = data[train_mask], labels[train_mask]

        nodes        = build_tree(td, tl, is_bin)
        train_result = rival_train(nodes, td, tl, is_bin, all_cls)

        # predict() imported unchanged from cds_ovr_bagging
        pred, _ = predict(i, data, nodes, all_cls, train_result)
        results.append((i, int(labels[i]), int(pred), pred == labels[i]))

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            done    = sum(r[3] for r in results)
            rate    = (i + 1) / elapsed
            eta     = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:3d}/{n}]  acc={100*done/(i+1):.1f}%  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    print(f"  Total time: {time.perf_counter()-t0:.0f}s", flush=True)
    return results


# ─── Feature comparison (single representative fold) ───────────────────────

def compare_features(data, labels, is_bin, fold_idx=0):
    """
    Train baseline and rival models on 415 patients (excluding fold_idx).
    Returns per-class dict: {baseline_feats, rival_feats, added, dropped}.
    """
    n       = data.shape[0]
    all_cls = sorted(set(labels))
    mask    = np.ones(n, dtype=bool)
    mask[fold_idx] = False
    td, tl  = data[mask], labels[mask]
    nodes   = build_tree(td, tl, is_bin)

    _, base_ret  = baseline_train(nodes, td, tl, is_bin, all_cls)
    _, rival_ret = rival_train(nodes, td, tl, is_bin, all_cls)

    out = {}
    for cls in all_cls:
        b_feats = set(a[0] for a in base_ret[cls])
        r_feats = set(a[0] for a in rival_ret[cls])
        out[cls] = {
            "baseline": sorted(b_feats),
            "rival":    sorted(r_feats),
            "added":    sorted(r_feats - b_feats),
            "dropped":  sorted(b_feats - r_feats),
        }
    return out


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE = {
        "binBA": 0.888733739109679,
        "spec":  0.9061224489795918,
        "sens":  0.8713450292397661,
        "cls":   {1: 0.9061, 2: 0.75, 3: 0.9333, 4: 0.80,
                  5: 0.7692, 6: 0.84, 9: 1.0, 10: 0.74},
    }

    # ── Step 1: Rival table ────────────────────────────────────────────────
    print("=" * 72)
    print("  STEP 1: RIVAL TABLE  (from loocv_baseline_results.json)")
    print("=" * 72)
    print()
    print(f"  {'Class':>7}  {'Rival':>7}  {'Err cnt':>7}  Note")
    print(f"  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*40}")
    _rival_info = {
        1:  (2,  12, "H->c2 errors; top rival by count"),
        2:  (1,   6, "c2->H errors"),
        3:  (2,   1, "c3->c2 only rival"),
        4:  (1,   3, "c4->H errors"),
        5:  (1,   3, "c5->H errors"),
        6:  (1,   1, "c6 tied H/c3/c4/c10 at 1; H by prevalence"),
        9:  (1,   0, "no errors; H as default"),
        10: (1,   9, "c10->H errors"),
    }
    for cls in [1, 2, 3, 4, 5, 6, 9, 10]:
        r, cnt, note = _rival_info[cls]
        clbl = "H(c1)" if cls == 1 else f"c{cls}"
        rlbl = "H(c1)" if r   == 1 else f"c{r}"
        print(f"  {clbl:>7}  {rlbl:>7}  {cnt:>7}  {note}")
    print()
    print(f"  RIVAL_ALPHA = {RIVAL_ALPHA}")
    print(f"  combined = score * (1 + {RIVAL_ALPHA} * rival_fisher / max_rival_fisher)")
    print()
    print("  GUARD: _compute_af imported unchanged from cds_ovr_bagging.")
    print("         predict() imported unchanged from cds_ovr_bagging.")
    print("         rival_fisher ONLY re-ranks the feature candidate pool.")
    print("         It never reaches the inference scoring path in _compute_af.")

    # ── Load data ──────────────────────────────────────────────────────────
    print()
    print("  Loading data...", flush=True)
    data, labels = load_data()
    is_bin       = classify_features(data)
    all_cls      = sorted(set(labels))
    n            = data.shape[0]
    print(f"  n={n} patients | {is_bin.sum()} binary features | "
          f"{(~is_bin).sum()} continuous features")

    # ── Step 2: Feature comparison ─────────────────────────────────────────
    print()
    print("=" * 72)
    print("  STEP 2: FEATURE COMPARISON  (fold 0: train on 415 patients)")
    print("=" * 72)
    print()
    print("  Training baseline + rival models...", flush=True)
    t_comp = time.perf_counter()
    comp   = compare_features(data, labels, is_bin, fold_idx=0)
    print(f"  Done in {time.perf_counter()-t_comp:.1f}s")
    print()

    total_added   = sum(len(c["added"])   for c in comp.values())
    total_dropped = sum(len(c["dropped"]) for c in comp.values())
    for cls in all_cls:
        c    = comp[cls]
        clbl = "H(c1)" if cls == 1 else f"c{cls}"
        rlbl = "H(c1)" if RIVAL_MAP.get(cls, 1) == 1 else f"c{RIVAL_MAP[cls]}"
        n_add  = len(c["added"])
        n_drop = len(c["dropped"])
        print(f"  {clbl} (rival={rlbl}): {n_add} added, {n_drop} dropped")
        if n_add > 0:
            print(f"    Added:   {', '.join(feat_name(f) for f in c['added'])}")
        if n_drop > 0:
            print(f"    Dropped: {', '.join(feat_name(f) for f in c['dropped'])}")
        if n_add == 0 and n_drop == 0:
            print(f"    (no change)")
    print()
    print(f"  Total changes: {total_added} added, {total_dropped} dropped "
          f"across {len(all_cls)} classes")

    # ── Step 3: Full LOOCV ─────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  STEP 3: FULL LOOCV  (rival-aware, n=416)")
    print("=" * 72)
    print()
    results = run_rival_loocv(data, labels, is_bin)

    # ── Print full report ──────────────────────────────────────────────────
    print_full_report(results, "RIVAL-AWARE FEATURE RETENTION  (Phase 1)")

    acc, spec, sens, ba = stats(results)
    binBA = (spec + sens) / 2.0
    per_cls = {}
    for cls in all_cls:
        cr = [r for r in results if r[1] == cls]
        per_cls[cls] = sum(r[3] for r in cr) / len(cr) if cr else 0.0

    # ── Comparison table ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  COMPARISON VS BASELINE")
    print("=" * 72)
    print()
    d_binBA = 100 * (binBA - BASE["binBA"])
    d_spec  = 100 * (spec  - BASE["spec"])
    d_sens  = 100 * (sens  - BASE["sens"])
    print(f"  {'Metric':<22}  {'Baseline':>9}  {'Rival':>9}  {'Delta':>8}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*8}")
    print(f"  {'binBA (H vs D) PRIMARY':<22}  "
          f"{100*BASE['binBA']:>8.4f}%  {100*binBA:>8.4f}%  {d_binBA:>+7.3f}pp")
    print(f"  {'Specificity':<22}  "
          f"{100*BASE['spec']:>8.4f}%  {100*spec:>8.4f}%  {d_spec:>+7.3f}pp")
    print(f"  {'Sensitivity':<22}  "
          f"{100*BASE['sens']:>8.4f}%  {100*sens:>8.4f}%  {d_sens:>+7.3f}pp")
    print()
    print(f"  Per-class accuracy:")
    print(f"  {'Class':>7}  {'N':>4}  {'Baseline':>9}  {'Rival':>9}  {'Delta':>8}")
    print(f"  {'-'*7}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*8}")
    collateral_ok = True
    for cls in all_cls:
        cr   = [r for r in results if r[1] == cls]
        clbl = "H(c1)" if cls == 1 else f"c{cls}"
        n_cls  = len(cr)
        base_v = BASE["cls"].get(cls, 0.0)
        riv_v  = per_cls.get(cls, 0.0)
        delta  = 100 * (riv_v - base_v)
        if cls not in (HEALTHY, 5, 6) and delta < -0.5:
            collateral_ok = False
        mark = "  <-- GAIN" if delta > 0.5 else ("  <-- LOSS" if delta < -0.5 else "")
        print(f"  {clbl:>7}  {n_cls:>4}  "
              f"{100*base_v:>8.2f}%  {100*riv_v:>8.2f}%  "
              f"{delta:>+7.2f}pp{mark}")

    # ── Decision rule assessment ───────────────────────────────────────────
    print()
    print("=" * 72)
    print("  DECISION RULE ASSESSMENT")
    print("=" * 72)
    print()
    coll_classes = [2, 3, 4, 9, 10]
    coll_deltas  = {cls: 100*(per_cls.get(cls, 0) - BASE["cls"].get(cls, 0))
                    for cls in coll_classes}
    has_coll = any(d < -0.5 for d in coll_deltas.values())

    for cls, d in sorted(coll_deltas.items()):
        flag = "  COLLATERAL" if d < -0.5 else ""
        print(f"    c{cls}: {d:>+.2f}pp{flag}")
    print()

    if d_binBA > 0.1 and not has_coll:
        verdict = ("POSITIVE: binBA gain with no collateral. "
                   "Principle transfers. Phase 2 worth considering.")
    elif d_binBA > 0.1 and has_coll:
        net = d_binBA + sum(min(0.0, d) for d in coll_deltas.values())
        verdict = (f"MIXED: gain present but collateral exists. "
                   f"Net estimate ~{net:+.2f}pp.")
    elif abs(d_binBA) <= 0.1 and not has_coll:
        verdict = ("NULL: no meaningful change. "
                   "Rival re-ranking at this alpha does not move the needle.")
    else:
        verdict = ("NEGATIVE: regression. "
                   "Rival re-ranking hurts; investigate which classes moved.")

    print(f"  binBA delta: {d_binBA:+.3f}pp")
    print(f"  Collateral:  {'none' if not has_coll else 'YES -- see above'}")
    print()
    print(f"  VERDICT: {verdict}")
    print()

    # ── Save results ───────────────────────────────────────────────────────
    out_path = _DIR / "loocv_rival_retention_results.json"
    save_results(results, all_cls, str(out_path))
