"""
cds_ovr_TRYOUT.py  --  Phase 2 milestone: sequential per-patient adaptive scoring.

SCOPE (this file only):
  Replaces the flat _compute_af pass with a step-by-step loop that, at each
  step, identifies the current leader and runner-up and selects the next
  feature specifically because it best discriminates THOSE TWO classes.

  Does NOT modify cds_ovr_bagging.py, cds_ovr.py, or cds_hr_bypass.py.
  Imports all model-building and data utilities from cds_ovr_bagging as-is.

Algorithm (_compute_af_seq):
  Prior: af_for = af_against = RATIO_EPS -> score 1.0 for all classes.
  At each step:
    1. Compute current scores; identify leader L (highest) and runner-up R.
       Tiebreak by class index (deterministic).
    2. Candidate features: union of L's and R's BinModel-contribution pools
       minus already-used features.  If that set is empty, fall back to any
       remaining feature.
    3. For each candidate f, compute L-vs-R Fisher from training data:
         (mean_L[f] - mean_R[f])^2 / (var_L[f] + var_R[f] + 1e-10)
       Select f* = argmax Fisher.
    4. Apply f*'s pre-computed BinModel contribution to every class that has
       f* in its pool; mark f* used.
  Repeat until all available features are exhausted.

  Final threshold/healthy_bar logic: identical to predict() in cds_ovr_bagging
  (no constants changed, same HEALTHY_BAR_CAP=3.5 now in effect).

Canary test: uid=171 (true=c4).
  The rival-aware Phase-1 run regressed this patient (c4->c2) because c4's
  evidence dropped while c2 stayed near-parity.  The sequential loop is
  expected to maintain c4's lead because once c2 emerges as runner-up it
  selects c4-vs-c2 discriminating features, reinforcing c4's margin.

  Using BASELINE training (non-rival), same as the validated baseline.

STOP CONDITION:
  After uid=171 trace + small-sample expansion, STOP.  No full LOOCV here.
"""

import sys, time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from cds_ovr_bagging import (
    load_data, classify_features, build_tree, train,
    predict as baseline_predict,
    _route_user,
    HEALTHY, RATIO_EPS,
    HEALTHY_WEIGHT, HEALTHY_BAR_CAP,
    SUSPICION_HCUT, SUSPICION_OFFSET,
    CLASS_THRESHOLDS, AGAINST_SCALE_MAP,
)

# ─── L-vs-R Fisher discriminant ──────────────────────────────────────────────

def _lv_fisher(f, td, tl, cls_L, cls_R):
    """Fisher discriminant of feature f between class L and class R
    in the fold-specific training data.  Same formula as in _train_ovr_node."""
    vL = td[tl == cls_L, f]
    vR = td[tl == cls_R, f]
    vL = vL[~np.isnan(vL)]
    vR = vR[~np.isnan(vR)]
    if len(vL) < 2 or len(vR) < 2:
        return 0.0
    return float((np.mean(vL) - np.mean(vR)) ** 2
                 / (np.var(vL) + np.var(vR) + 1e-10))


# ─── Pre-compute per-patient BinModel contributions ──────────────────────────

def _precompute_contribs(uid, data, nodes, train_result, all_cls):
    """
    For patient uid, pre-compute each class's per-feature evidence contribution.

    Returns:
      contribs[cls][f] = (for_contrib, against_contrib)
        where for_contrib / against_contrib are non-negative additive increments
        to af_for / af_against using the same BinModel logic as _compute_af.
      Only features with a non-zero contribution (from the patient's routed,
      non-NaN active nodes) appear in the dict.

    Fisher normalization matches _compute_af: max_fisher computed across ALL
    retained actions (including non-routed nodes).
    """
    class_models, class_retained = train_result

    lvl_nodes = _route_user(uid, data, nodes)
    routed_nids = {nd.nid for nds in lvl_nodes.values() for nd in nds}

    contribs = {}
    for cls in all_cls:
        against_scale = AGAINST_SCALE_MAP.get(cls, 0.8)
        retained = class_retained[cls]

        # Global fisher normalization (all retained, not just routed) -- matches _compute_af
        fisher_map = {}
        for a in retained:
            fisher_map[a[0]] = max(fisher_map.get(a[0], 0.0), a[3])
        max_fisher = max(fisher_map.values()) if fisher_map else 1.0

        cls_c = {}
        for a in retained:
            f, nid = a[0], a[1]
            if nid not in routed_nids:
                continue
            v = data[uid, f]
            if np.isnan(v):
                continue
            mo = class_models[cls].get((nid, f))
            if mo is None:
                continue
            bin_idx = int(np.clip(
                np.searchsorted(mo.edges[1:], v, side='right'),
                0, mo.n_bins - 1))
            bc = mo.bin_counts[bin_idx]
            if bc < 3:
                continue
            shift = mo.p_class[bin_idx] - mo.prior
            confidence = min(1.0, bc / 10.0)
            fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
            weighted = abs(shift) * confidence * fw

            pf, pa = cls_c.get(f, (0.0, 0.0))
            if shift >= 0:
                cls_c[f] = (pf + weighted, pa)
            else:
                cls_c[f] = (pf, pa + weighted * against_scale)

        contribs[cls] = cls_c

    return contribs


# ─── Sequential scoring loop ─────────────────────────────────────────────────

def _compute_af_seq(uid, data, nodes, train_result, td, tl, all_cls,
                    max_steps=None):
    """
    Replace _compute_af with a sequential adaptive loop.

    Returns:
      final_scores  dict {cls: float}
      trace         list of step dicts (step, feature, fisher_LR, leader,
                    runner_up, scores_after)
    """
    contribs = _precompute_contribs(uid, data, nodes, train_result, all_cls)

    # Per-class feature pools (only features with non-zero contribution)
    pool = {cls: set(contribs[cls].keys()) for cls in all_cls}

    # Which classes receive evidence when feature f is selected
    feat_owners = defaultdict(set)
    for cls, feats in pool.items():
        for f in feats:
            feat_owners[f].add(cls)

    all_avail = set(feat_owners.keys())
    steps = max_steps if max_steps is not None else len(all_avail)

    # Starting state: score = 1.0 for all classes (same as _compute_af with no evidence)
    af_for     = {cls: RATIO_EPS for cls in all_cls}
    af_against = {cls: RATIO_EPS for cls in all_cls}
    used       = set()
    trace      = []

    for step in range(steps):
        remaining = all_avail - used
        if not remaining:
            break

        # Current scores and ranking; tiebreak by class index
        scores     = {cls: af_for[cls] / af_against[cls] for cls in all_cls}
        ranked     = sorted(all_cls, key=lambda c: (-scores[c], c))
        leader     = ranked[0]
        runner_up  = ranked[1] if len(ranked) > 1 else None

        # Candidate features from leader's + runner-up's pools
        cands = (pool[leader] | (pool[runner_up] if runner_up else set())) & remaining
        if not cands:
            cands = remaining   # fallback: any remaining feature

        # Select f* = highest L-vs-R Fisher among candidates
        best_f, best_fish = None, -1.0
        for f in cands:
            fv = _lv_fisher(f, td, tl, leader, runner_up) if runner_up else 0.0
            if best_f is None or fv > best_fish:
                best_fish, best_f = fv, f

        # Apply f* to all classes that have it
        for cls in feat_owners[best_f]:
            pf, pa = contribs[cls].get(best_f, (0.0, 0.0))
            af_for[cls]     += pf
            af_against[cls] += pa
        used.add(best_f)

        scores_now = {cls: af_for[cls] / af_against[cls] for cls in all_cls}
        trace.append({
            'step':       step + 1,
            'feature':    int(best_f),
            'fisher_LR':  float(best_fish),
            'leader':     int(leader),
            'runner_up':  int(runner_up) if runner_up is not None else None,
            'scores':     {int(c): float(s) for c, s in scores_now.items()},
        })

    final_scores = {cls: af_for[cls] / af_against[cls] for cls in all_cls}
    return final_scores, trace


# ─── Sequential predict (same threshold logic as baseline predict()) ──────────

def predict_seq(uid, data, nodes, all_cls, train_result, td, tl, max_steps=None):
    """
    Sequential prediction.  Replaces _compute_af; threshold/healthy_bar logic
    is identical to predict() in cds_ovr_bagging (HEALTHY_BAR_CAP=3.5 now live).
    Returns (best_cls, final_scores, trace).
    """
    final_scores, trace = _compute_af_seq(
        uid, data, nodes, train_result, td, tl, all_cls, max_steps=max_steps)

    h_score    = final_scores.get(HEALTHY, 1.0)
    healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
    candidates  = {}
    for cls, score in final_scores.items():
        if cls == HEALTHY:
            continue
        t = CLASS_THRESHOLDS.get(cls, 3.0)
        if h_score < SUSPICION_HCUT:
            t -= SUSPICION_OFFSET
        t = max(t, healthy_bar)
        if score < t:
            continue
        candidates[cls] = (score - t) / max(t, 0.1)

    best_cls = max(candidates, key=candidates.get) if candidates else HEALTHY
    return best_cls, final_scores, trace


# ─── Fold runner (shared by canary and small-sample) ─────────────────────────

def _run_fold(uid, data, labels, is_bin, all_cls):
    """Train on all patients except uid; return baseline and seq predictions."""
    n    = data.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[uid] = False
    td, tl  = data[mask], labels[mask]
    nodes   = build_tree(td, tl, is_bin)
    tr      = train(nodes, td, tl, is_bin, all_cls)
    bp, bs  = baseline_predict(uid, data, nodes, all_cls, tr)
    sp, ss, trace = predict_seq(uid, data, nodes, all_cls, tr, td, tl)
    return bp, bs, sp, ss, trace, nodes, tr, td, tl


# ─── Formatting helpers ───────────────────────────────────────────────────────

def _cls_lbl(c):
    return "H" if c == HEALTHY else f"c{c}"


def _score_row(scores, all_cls, true_cls, pred_cls, width=7):
    parts = []
    for c in sorted(all_cls):
        s   = scores.get(c, 0.0)
        mrk = ("*T" if c == true_cls else (" P" if c == pred_cls else "  "))
        parts.append(f"{_cls_lbl(c)}={s:{width}.4f}{mrk}")
    return "  ".join(parts)


# ─── Main test ────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  cds_ovr_TRYOUT.py  --  Phase 2 milestone: sequential scoring")
    print("=" * 80)
    print()

    data, labels = load_data()
    is_bin  = classify_features(data)
    all_cls = sorted(set(labels))
    n       = data.shape[0]
    print(f"  n={n}  classes={all_cls}")
    print()

    # ── CANARY: uid=171 ───────────────────────────────────────────────────────
    UID = 171
    print("=" * 80)
    print(f"  CANARY TEST  uid={UID}  (true=c4)")
    print("=" * 80)
    print()
    print(f"  Building fold (train on {n-1} patients)...", flush=True)
    t0 = time.time()
    bp, bs, sp, ss, trace, nodes, tr, td, tl = _run_fold(UID, data, labels, is_bin, all_cls)
    print(f"  Done in {time.time()-t0:.1f}s")
    print()

    tc = int(labels[UID])
    print(f"  true={_cls_lbl(tc)}  baseline_pred={_cls_lbl(bp)}  seq_pred={_cls_lbl(sp)}")
    print()

    # Side-by-side score comparison
    print(f"  {'Class':>5}  {'Baseline':>10}  {'Sequential':>11}  {'threshold':>10}  notes")
    print(f"  {'-'*5}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*30}")
    h_b = bs.get(HEALTHY, 1.0)
    h_s = ss.get(HEALTHY, 1.0)
    hbar_b = min(HEALTHY_WEIGHT * h_b, HEALTHY_BAR_CAP)
    hbar_s = min(HEALTHY_WEIGHT * h_s, HEALTHY_BAR_CAP)
    for c in sorted(all_cls):
        b_sc = bs.get(c, 0.0)
        s_sc = ss.get(c, 0.0)
        if c == HEALTHY:
            thr_str = f"hbar={hbar_b:.3f}"
        else:
            raw_t = CLASS_THRESHOLDS.get(c, 3.0)
            eff_t = max(raw_t, hbar_b)
            thr_str = f">={eff_t:.2f}"
        notes = []
        if c == tc:    notes.append("TRUE")
        if c == bp:    notes.append("BASE_PRED")
        if c == sp:    notes.append("SEQ_PRED")
        print(f"  {_cls_lbl(c):>5}  {b_sc:>10.4f}  {s_sc:>11.4f}  {thr_str:>10}  "
              + "  ".join(notes))
    print()

    # Step-by-step trace
    print(f"  Step-by-step trace (all {len(trace)} steps):")
    print(f"  {'step':>4}  {'feat':>5}  {'LvR_fisher':>10}  "
          f"{'leader':>14}  {'runup':>14}  c4          c2")
    print(f"  {'-'*4}  {'-'*5}  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}")
    for t in trace:
        st    = t['step']
        f     = t['feature']
        fish  = t['fisher_LR']
        L     = t['leader']
        R     = t['runner_up']
        sc    = t['scores']
        L_lbl = f"{_cls_lbl(L)}({sc.get(L,0):.3f})"
        R_lbl = f"{_cls_lbl(R)}({sc.get(R,0):.3f})" if R is not None else "---"
        s_c4  = sc.get(4, 0.0)
        s_c2  = sc.get(2, 0.0)
        # Highlight when c4 or c2 is selected / leader changes
        note = ""
        if L == 4 and R == 2:   note = " << c4 vs c2"
        elif L == 2 and R == 4: note = " << c2 vs c4"
        print(f"  {st:>4}  f{f:<4}  {fish:>10.3f}  "
              f"{L_lbl:>14}  {R_lbl:>14}  {s_c4:>8.4f}  {s_c2:>8.4f}{note}")

    final_c4 = ss.get(4, 0.0)
    final_c2 = ss.get(2, 0.0)
    print()
    print(f"  Final: c4={final_c4:.4f}  c2={final_c2:.4f}  "
          f"c4_margin={final_c4-final_c2:+.4f}")
    print()
    c4_lead_all = all(t['scores'].get(4, 0) >= t['scores'].get(2, 0) for t in trace)
    c4_lead_first_c2_runup = None
    for t in trace:
        if t['runner_up'] == 2 or t['leader'] == 2:
            c4_ahead = t['scores'].get(4, 0) >= t['scores'].get(2, 0)
            if c4_lead_first_c2_runup is None:
                c4_lead_first_c2_runup = c4_ahead
    print(f"  c4 leads c2 at every step: {c4_lead_all}")
    print(f"  c4 ahead when c2 first enters top-2: {c4_lead_first_c2_runup}")
    print()
    outcome = "CORRECT (c4)" if sp == 4 else f"WRONG ({_cls_lbl(sp)})"
    print(f"  UID=171 OUTCOME: {outcome}  "
          f"(baseline={'CORRECT' if bp == tc else 'WRONG'}, "
          f"seq={'CORRECT' if sp == tc else 'WRONG'})")
    print()

    # ── SMALL SAMPLE EXPANSION ────────────────────────────────────────────────
    print("=" * 80)
    print("  SMALL SAMPLE EXPANSION")
    print("=" * 80)
    print()

    REGRESSION_UIDS = [51, 128, 258]
    SAMPLE_CLASSES  = [2, 4, 5, 6, 10]
    SAMPLE_N        = 10

    by_class = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_class[int(lbl)].append(i)

    rng = np.random.default_rng(42)
    eval_list = [(uid, f"regression(c{int(labels[uid])})", False)
                 for uid in REGRESSION_UIDS]

    exclude = {UID} | set(REGRESSION_UIDS)
    for cls in SAMPLE_CLASSES:
        avail = [i for i in by_class[cls] if i not in exclude]
        k     = min(SAMPLE_N, len(avail))
        for uid in rng.choice(avail, size=k, replace=False).tolist():
            eval_list.append((uid, f"random(c{cls})", False))
            exclude.add(uid)

    print(f"  {len(eval_list)} patients to evaluate "
          f"(3 regression + {len(eval_list)-3} random sampled).")
    print()
    print(f"  {'uid':>5}  {'true':>4}  {'base':>5}  {'seq':>5}  "
          f"{'base_ok':>7}  {'seq_ok':>6}  kind            status")
    print(f"  {'-'*5}  {'-'*4}  {'-'*5}  {'-'*5}  "
          f"{'-'*7}  {'-'*6}  {'-'*16}  {'-'*24}")

    results = []
    for idx, (uid, kind, _) in enumerate(eval_list):
        uid = int(uid)
        bp2, bs2, sp2, ss2, tr2, _, _, _, _ = _run_fold(uid, data, labels, is_bin, all_cls)
        tc2  = int(labels[uid])
        b_ok = (bp2 == tc2)
        s_ok = (sp2 == tc2)
        bp_l = _cls_lbl(bp2)
        sp_l = _cls_lbl(sp2)
        tc_l = _cls_lbl(tc2)

        status = ""
        if b_ok and not s_ok:
            status = "!! BASE_OK SEQ_BAD (regression)"
        elif not b_ok and s_ok:
            status = "++ BASE_BAD SEQ_OK (gain)"
        elif bp2 != sp2:
            status = "~~ both wrong, preds differ"

        print(f"  {uid:>5}  {tc_l:>4}  {bp_l:>5}  {sp_l:>5}  "
              f"{'Y' if b_ok else 'N':>7}  {'Y' if s_ok else 'N':>6}  "
              f"{kind:<16}  {status}")
        results.append((uid, tc2, bp2, sp2, b_ok, s_ok))

    # Summary
    n_b = sum(r[4] for r in results)
    n_s = sum(r[5] for r in results)
    reg = [r for r in results if r[4] and not r[5]]
    gain= [r for r in results if not r[4] and r[5]]
    diff= [r for r in results if r[2] != r[3]]

    print()
    print(f"  SMALL-SAMPLE SUMMARY  ({len(results)} patients):")
    print(f"    baseline correct:   {n_b}/{len(results)}")
    print(f"    sequential correct: {n_s}/{len(results)}")
    print(f"    regressions (base_ok, seq_bad):  {len(reg)}")
    print(f"    gains       (base_bad, seq_ok):  {len(gain)}")
    print(f"    prediction changed (any):        {len(diff)}")
    if reg:
        print(f"  Regressions:")
        for r in reg:
            print(f"    uid={r[0]}  true={_cls_lbl(r[1])}  "
                  f"base={_cls_lbl(r[2])}  seq={_cls_lbl(r[3])}")
    if gain:
        print(f"  Gains:")
        for r in gain:
            print(f"    uid={r[0]}  true={_cls_lbl(r[1])}  "
                  f"base={_cls_lbl(r[2])}  seq={_cls_lbl(r[3])}")

    # Include uid=171 in overall summary
    canary_ok = (sp == tc)
    print()
    print(f"  Including canary uid=171:")
    print(f"    baseline correct:   {n_b + (bp==tc)}/{len(results)+1}")
    print(f"    sequential correct: {n_s + canary_ok}/{len(results)+1}")

    print()
    print("=" * 80)
    print("  STOP CONDITION REACHED.")
    print("  Next step: full LOOCV only if milestone results warrant it.")
    print("=" * 80)


if __name__ == "__main__":
    main()
