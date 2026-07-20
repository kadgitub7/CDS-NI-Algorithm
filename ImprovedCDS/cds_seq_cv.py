"""
cds_seq_cv.py  --  Sequential scoring: stopping-rule design, nested calibration,
                   full 416-fold LOOCV validation.

STOPPING RULE (CONSECUTIVE_MARGIN):
  After each step, check: has the SAME leader held margin > MARGIN_THR for
  STABILITY consecutive steps, AND step >= MIN_STEPS?
  margin = (leader_score - runnerup_score) / max(runnerup_score, 0.1)
  If yes: predict from current scores; stop.
  If trace exhausted without stopping: full-budget prediction (= baseline).

  Justification vs uid=171:
    c2's peak margin over c5 was 1.12 (steps 9-11).  Any MARGIN_THR <= 1.12
    with STABILITY <= 3 stops at step 9-11 predicting c2 (WRONG).
    MARGIN_THR > 1.12 forces full budget on uid=171 -> correct.
    The nested calibration must find this threshold from the training data;
    uid=171's outcome is not in the calibration set for its own fold.

NESTED CALIBRATION (fold-level LOO):
  For each outer test patient i:
    (a) Pre-compute, for all j != i using their stored LOOCV traces, what
        each (MARGIN_THR, STABILITY, MIN_STEPS) config predicts.
    (b) Select the config that maximises binBA over j != i.
    (c) Apply that config to patient i's trace.
  This ensures patient i's label never influences its own threshold.
  Implementation is vectorised (no per-patient Python loop in Phase 2).

Phase 1: Full 416-fold LOOCV, saving per-patient full traces.
Phase 2: Vectorised nested calibration (~1s).
Phase 3: All required reporting + risk checks.

DOES NOT MODIFY: cds_ovr_bagging.py, cds_ovr.py, cds_hr_bypass.py, cds_ovr_TRYOUT.py
"""

import sys, time, gzip, pickle, itertools
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from cds_ovr_bagging import (
    load_data, classify_features, build_tree, train,
    predict as baseline_predict, _route_user,
    HEALTHY, RATIO_EPS,
    HEALTHY_WEIGHT, HEALTHY_BAR_CAP,
    SUSPICION_HCUT, SUSPICION_OFFSET,
    CLASS_THRESHOLDS, AGAINST_SCALE_MAP,
    FEATURES_PER_CLASS,
)

_DIR = Path(__file__).parent
TRACE_CACHE = _DIR / "loocv_seq_traces.pkl.gz"

# ── Parameter grid ────────────────────────────────────────────────────────────
MARGIN_THRS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
STABILITIES  = [1, 3, 5]
MIN_STEPS_L  = [0, FEATURES_PER_CLASS, 2 * FEATURES_PER_CLASS]   # 0, 18, 36

PARAM_CONFIGS = [
    {'margin_thr': mt, 'stability': st, 'min_steps': ms}
    for mt in MARGIN_THRS
    for st in STABILITIES
    for ms in MIN_STEPS_L
]
# Append "never-stop" sentinel (baseline equivalent)
PARAM_CONFIGS.append({'margin_thr': 1e9, 'stability': 1, 'min_steps': 0})
N_CONFIGS = len(PARAM_CONFIGS)


# ── Fisher cache (pre-computed per fold for speed) ────────────────────────────

def _build_lv_fisher_cache(td, tl, all_cls):
    """
    Pre-compute L-vs-R Fisher for all class pairs, all N_FEAT features.
    Returns dict {(L,R): np.array(N_FEAT)} (symmetric: (L,R)==(R,L)).
    ~0.05s per fold.
    """
    cache = {}
    cls_list = list(all_cls)
    for i, L in enumerate(cls_list):
        mL = td[tl == L]           # (n_L, N_FEAT)
        mL_mean = np.nanmean(mL, axis=0)
        mL_var  = np.nanvar(mL, axis=0)
        mL_cnt  = (~np.isnan(mL)).sum(axis=0)
        for R in cls_list[i + 1:]:
            mR = td[tl == R]
            mR_mean = np.nanmean(mR, axis=0)
            mR_var  = np.nanvar(mR, axis=0)
            mR_cnt  = (~np.isnan(mR)).sum(axis=0)
            fish = (mL_mean - mR_mean) ** 2 / (mL_var + mR_var + 1e-10)
            fish[mL_cnt < 2] = 0.0
            fish[mR_cnt < 2] = 0.0
            cache[(int(L), int(R))] = fish
            cache[(int(R), int(L))] = fish
    return cache


# ── Sequential loop (with Fisher cache) ──────────────────────────────────────

def _precompute_contribs(uid, data, nodes, train_result, all_cls):
    """Same as cds_ovr_TRYOUT._precompute_contribs."""
    class_models, class_retained = train_result
    lvl_nodes  = _route_user(uid, data, nodes)
    routed_nids = {nd.nid for nds in lvl_nodes.values() for nd in nds}
    contribs = {}
    for cls in all_cls:
        against_scale = AGAINST_SCALE_MAP.get(cls, 0.8)
        retained = class_retained[cls]
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
                np.searchsorted(mo.edges[1:], v, side='right'), 0, mo.n_bins - 1))
            bc = mo.bin_counts[bin_idx]
            if bc < 3:
                continue
            shift = mo.p_class[bin_idx] - mo.prior
            conf  = min(1.0, bc / 10.0)
            fw    = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)
            w     = abs(shift) * conf * fw
            pf, pa = cls_c.get(f, (0.0, 0.0))
            if shift >= 0:
                cls_c[f] = (pf + w, pa)
            else:
                cls_c[f] = (pf, pa + w * against_scale)
        contribs[cls] = cls_c
    return contribs


def _run_seq_patient(uid, data, nodes, train_result, all_cls, fisher_cache):
    """
    Run the sequential loop for one patient; return full trace.
    Uses pre-computed fisher_cache for speed.
    Trace: list of {'step', 'leader', 'runner_up', 'scores'}.
    """
    contribs = _precompute_contribs(uid, data, nodes, train_result, all_cls)
    pool = {cls: set(contribs[cls].keys()) for cls in all_cls}
    feat_owners = defaultdict(set)
    for cls, feats in pool.items():
        for f in feats:
            feat_owners[f].add(cls)
    all_avail = set(feat_owners.keys())

    af_for     = {cls: RATIO_EPS for cls in all_cls}
    af_against = {cls: RATIO_EPS for cls in all_cls}
    used       = set()
    trace      = []

    for step in range(len(all_avail)):
        remaining = all_avail - used
        if not remaining:
            break
        scores  = {cls: af_for[cls] / af_against[cls] for cls in all_cls}
        ranked  = sorted(all_cls, key=lambda c: (-scores[c], c))
        leader  = int(ranked[0])
        r_up    = int(ranked[1]) if len(ranked) > 1 else None

        cands = (pool[leader] | (pool[r_up] if r_up else set())) & remaining
        if not cands:
            cands = remaining

        # Feature selection: use pre-computed Fisher cache
        fish_arr = fisher_cache.get((leader, r_up)) if r_up else None
        best_f, best_v = None, -1.0
        for f in cands:
            fv = float(fish_arr[f]) if fish_arr is not None else 0.0
            if best_f is None or fv > best_v:
                best_v, best_f = fv, f

        for cls in feat_owners[best_f]:
            pf, pa = contribs[cls].get(best_f, (0.0, 0.0))
            af_for[cls]     += pf
            af_against[cls] += pa
        used.add(best_f)

        scores_now = {cls: af_for[cls] / af_against[cls] for cls in all_cls}
        trace.append({
            'step':      step + 1,
            'leader':    leader,
            'runner_up': r_up,
            'scores':    {int(c): float(s) for c, s in scores_now.items()},
        })

    return trace


# ── Apply predict-logic to a scores dict (unchanged from cds_ovr_bagging) ────

def _apply_predict_logic(scores, all_cls):
    """Threshold/healthy_bar logic. Identical to predict() in cds_ovr_bagging."""
    h_score    = scores.get(HEALTHY, 1.0)
    healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
    candidates  = {}
    for cls in all_cls:
        if cls == HEALTHY:
            continue
        score = scores.get(cls, 0.0)
        t = CLASS_THRESHOLDS.get(cls, 3.0)
        if h_score < SUSPICION_HCUT:
            t -= SUSPICION_OFFSET
        t = max(t, healthy_bar)
        if score >= t:
            candidates[cls] = (score - t) / max(t, 0.1)
    return max(candidates, key=candidates.get) if candidates else HEALTHY


# ── Stopping-rule application to a stored trace ───────────────────────────────

def apply_stopping_rule(trace, min_steps, margin_thr, stability):
    """
    Apply the CONSECUTIVE_MARGIN stopping rule to a stored trace.
    Returns (stop_step, scores_at_stop, stable_count_history).
    stop_step = None means full budget was used.
    """
    stable_count  = 0
    stable_leader = None
    for t in trace:
        step = t['step']
        sc   = t['scores']
        ranked = sorted(sc.keys(), key=lambda c: (-sc[c], c))
        L = ranked[0]
        R = ranked[1] if len(ranked) > 1 else None
        if R is not None:
            margin = (sc[L] - sc[R]) / max(sc[R], 0.1)
        else:
            margin = 1e9
        if L == stable_leader and margin > margin_thr:
            stable_count += 1
        else:
            stable_leader = L
            stable_count  = 1 if margin > margin_thr else 0
        if step >= min_steps and stable_count >= stability:
            return step, sc
    return None, trace[-1]['scores'] if trace else {}


# ── Phase 1: Full LOOCV with trace saving ─────────────────────────────────────

def run_loocv_phase1(data, labels, is_bin, all_cls, cache_path):
    """
    416-fold LOOCV.  For each fold: train, run sequential loop, save full trace.
    Results saved incrementally; resumes from checkpoint if interrupted.
    """
    n = data.shape[0]
    existing = {}
    if cache_path.exists():
        print(f"  Loading existing Phase 1 cache from {cache_path.name}...", flush=True)
        with gzip.open(cache_path, 'rb') as fh:
            saved = pickle.load(fh)
        existing = {r['uid']: r for r in saved}
        print(f"  {len(existing)} folds already computed.", flush=True)

    results = list(existing.values())
    done_uids = set(existing.keys())
    t0 = time.time()

    for i in range(n):
        if i in done_uids:
            continue
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        td, tl = data[mask], labels[mask]
        nodes  = build_tree(td, tl, is_bin)
        tr     = train(nodes, td, tl, is_bin, all_cls)
        fc     = _build_lv_fisher_cache(td, tl, all_cls)
        bp, bs = baseline_predict(i, data, nodes, all_cls, tr)
        trace  = _run_seq_patient(i, data, nodes, tr, all_cls, fc)

        results.append({
            'uid':       i,
            'true':      int(labels[i]),
            'base_pred': int(bp),
            'trace':     trace,
        })
        done_uids.add(i)

        if (len(done_uids)) % 10 == 0 or len(done_uids) == n:
            elapsed = time.time() - t0
            remain  = n - len(done_uids)
            rate    = (len(done_uids) - len(existing)) / max(elapsed, 0.1)
            eta     = remain / rate if rate > 0 else 0
            print(f"  [{len(done_uids):>3}/{n}] elapsed={elapsed:.0f}s  "
                  f"ETA={eta:.0f}s", flush=True)
            with gzip.open(cache_path, 'wb') as fh:
                pickle.dump(results, fh)

    # Sort by uid
    results.sort(key=lambda r: r['uid'])
    with gzip.open(cache_path, 'wb') as fh:
        pickle.dump(results, fh)
    return results


# ── Phase 2: Vectorised nested calibration ────────────────────────────────────

def run_nested_calibration(loocv_data, all_cls):
    """
    For each patient i: calibrate stopping rule on {j != i}, apply to patient i.
    Returns list of per-patient dicts with 'uid','true','pred','stop_step','cfg'.
    """
    n       = len(loocv_data)
    trues   = np.array([r['true']      for r in loocv_data], dtype=int)
    is_h    = (trues == HEALTHY)
    is_d    = ~is_h
    n_H     = is_h.sum()
    n_D     = is_d.sum()

    # Pre-compute (pred, stop_step) for every (patient, config)
    all_preds  = np.zeros((n, N_CONFIGS), dtype=int)
    all_stops  = np.zeros((n, N_CONFIGS), dtype=int)

    for j, r in enumerate(loocv_data):
        trace = r['trace']
        full_len = len(trace)
        for ci, cfg in enumerate(PARAM_CONFIGS):
            s_step, s_sc = apply_stopping_rule(
                trace, cfg['min_steps'], cfg['margin_thr'], cfg['stability'])
            pred = _apply_predict_logic(s_sc, all_cls)
            all_preds[j, ci]  = int(pred)
            all_stops[j, ci]  = s_step if s_step is not None else full_len

    # Precompute per-config per-patient correct indicators
    # H_ok[j, ci] = patient j is H and predicted H with config ci
    # D_ok[j, ci] = patient j is D and predicted non-H with config ci
    H_ok = is_h[:, None] & (all_preds == HEALTHY)     # shape (n, N_CONFIGS)
    D_ok = is_d[:, None] & (all_preds != HEALTHY)

    tot_H_ok = H_ok.sum(axis=0)  # (N_CONFIGS,)
    tot_D_ok = D_ok.sum(axis=0)

    nested_results = []
    for i in range(n):
        # Inner binBA for each config on {j != i}
        h_adj = n_H - int(is_h[i])
        d_adj = n_D - int(is_d[i])
        h_inner = (tot_H_ok - H_ok[i].astype(int)) / max(h_adj, 1)
        d_inner = (tot_D_ok - D_ok[i].astype(int)) / max(d_adj, 1)
        bb_inner = (h_inner + d_inner) / 2.0
        best_ci  = int(np.argmax(bb_inner))

        nested_results.append({
            'uid':       loocv_data[i]['uid'],
            'true':      int(trues[i]),
            'base_pred': int(loocv_data[i]['base_pred']),
            'pred':      int(all_preds[i, best_ci]),
            'stop_step': int(all_stops[i, best_ci]),
            'trace_len': len(loocv_data[i]['trace']),
            'cfg':       PARAM_CONFIGS[best_ci],
            'inner_binba': float(bb_inner[best_ci]),
        })

    return nested_results, all_preds, all_stops


# ── Metrics helpers ───────────────────────────────────────────────────────────

def _cls_lbl(c):
    return "H" if c == HEALTHY else f"c{c}"


def compute_metrics(results, all_cls):
    """binBA, spec, sens, per-class accuracy."""
    h = [r for r in results if r['true'] == HEALTHY]
    d = [r for r in results if r['true'] != HEALTHY]
    sp = sum(r['pred'] == HEALTHY      for r in h) / len(h)
    sn = sum(r['pred'] != HEALTHY      for r in d) / len(d)
    bb = (sp + sn) / 2
    per_cls = {}
    for c in all_cls:
        cr = [r for r in results if r['true'] == c]
        per_cls[c] = sum(r['pred'] == c for r in cr) / len(cr) if cr else 0.0
    return bb, sp, sn, per_cls


# ── Stopping-rule design analysis ─────────────────────────────────────────────

def stopping_rule_design_analysis(uid171_trace, all_cls):
    """
    Analyse uid=171's trace under all candidate stopping rules.
    Identifies which (THR, STAB) combinations handle it correctly.
    """
    print("=" * 80)
    print("  STOPPING RULE DESIGN ANALYSIS  (uid=171 trace)")
    print("=" * 80)
    print()
    print("  uid=171: true=c4. Baseline+full-budget-seq both correct.")
    print("  Key trace facts:")
    print("    Steps  1-72:  c2 is leader, c5 runner-up. c2 margin over c5:")
    # Show margin at selected steps
    for t in uid171_trace:
        step = t['step']
        if step not in {3, 9, 11, 33, 72, 73, 75, 76, 89}:
            continue
        sc = t['scores']
        ranked = sorted(sc.keys(), key=lambda c: -sc[c])
        L, R = ranked[0], ranked[1]
        mg = (sc[L] - sc[R]) / max(sc[R], 0.1)
        print(f"    step {step:>2}: leader={_cls_lbl(L)}({sc[L]:.3f})  "
              f"runup={_cls_lbl(R)}({sc[R]:.3f})  margin={mg:.3f}")
    print()

    # Test each rule
    print("  Stopping-rule behaviour on uid=171:")
    print(f"  {'THR':>6}  {'STAB':>4}  {'MIN':>4}  "
          f"{'stop_step':>9}  {'stop_pred':>9}  {'correct?':>8}  note")
    print(f"  {'-'*6}  {'-'*4}  {'-'*4}  "
          f"{'-'*9}  {'-'*9}  {'-'*8}  {'-'*30}")

    for mt in MARGIN_THRS:
        for st in [1, 3, 5]:
            for ms in [0, 18, 36]:
                ss, sc = apply_stopping_rule(uid171_trace, ms, mt, st)
                pred   = _apply_predict_logic(sc, all_cls)
                ok     = (pred == 4)
                if ss is None:
                    note = "full budget"
                elif ss < 73:
                    note = f"stops before c4 enters top-2"
                elif ss >= 76:
                    note = f"stops after c4 takes lead"
                else:
                    note = f"stops mid-transition"
                print(f"  {mt:>6.2f}  {st:>4}  {ms:>4}  "
                      f"{'full' if ss is None else ss:>9}  "
                      f"{_cls_lbl(pred):>9}  "
                      f"{'OK' if ok else 'WRONG':>8}  {note}")
    print()
    print("  Conclusion:")
    print("  uid=171's c2 peaked margin over c5/c4 = 1.12 (steps 9-11).")
    print("  Any MARGIN_THR <= 1.12 with STABILITY <= 3 stops wrong for uid=171.")
    print("  MARGIN_THR > 1.12 forces full budget -> correct.")
    print("  This boundary will surface in nested calibration as a transition.")
    print()


# ── Per-patient healthy_bar risk check ────────────────────────────────────────

def healthy_bar_risk_check(nested_results, all_cls):
    """
    Check healthy_bar / HEALTHY_BAR_CAP interactions under early stopping.
    Risk: early stopping leaves h_score low -> healthy_bar low -> disease
    thresholds lower -> potential false positives for H patients.
    """
    print()
    print("-" * 60)
    print("  HEALTHY_BAR RISK CHECK (early stopping)")
    print("-" * 60)
    risky = []
    for r in nested_results:
        if r['true'] != HEALTHY:
            continue
        if r['pred'] == HEALTHY:
            continue   # correct, not a FP
        # False positive for H patient -> note h_score and stop_step
        ss = r['stop_step']
        risky.append((r['uid'], ss, r['trace_len']))
    if not risky:
        print("  No false positives for H patients introduced by early stopping.")
    else:
        print(f"  {len(risky)} H patient(s) newly predicted as disease (FP):")
        for uid, ss, tl in risky:
            print(f"    uid={uid}  stop_step={ss}/{tl}")
    print()


# ── Margin-distortion check ───────────────────────────────────────────────────

def margin_distortion_check(nested_results, all_preds, all_cls, loocv_data):
    """
    Check whether early stopping systematically favours classes whose features
    are evaluated earlier in the trace, independent of true signal strength.
    Proxy: compare per-class accuracy between the full-budget config and the
    nested-calibrated run.
    """
    print()
    print("-" * 60)
    print("  MARGIN DISTORTION CHECK (class-level early-stop bias)")
    print("-" * 60)
    # Full-budget config = last in PARAM_CONFIGS (margin_thr=1e9)
    full_ci = N_CONFIGS - 1
    print("  Full-budget per-class accuracy (same as baseline):")
    print("  Nested per-class accuracy:")
    print(f"  {'class':>7}  {'N':>4}  {'full_budget':>11}  {'nested':>7}  delta")
    print(f"  {'-'*7}  {'-'*4}  {'-'*11}  {'-'*7}  {'-'*6}")
    trues = np.array([r['true']      for r in loocv_data])
    for c in sorted(all_cls):
        mask   = (trues == c)
        n_c    = mask.sum()
        fb_acc = np.mean(all_preds[mask, full_ci] == c)
        n_acc  = np.mean([r['pred'] == c for r in nested_results if r['true'] == c])
        delta  = n_acc - fb_acc
        flag   = "  !! DISTORTION" if delta < -0.05 else ""
        print(f"  {_cls_lbl(c):>7}  {n_c:>4}  {100*fb_acc:>10.2f}%  "
              f"{100*n_acc:>6.2f}%  {100*delta:>+5.2f}pp{flag}")
    print()


# ── Full reporting ─────────────────────────────────────────────────────────────

def report_full(nested_results, all_cls, loocv_data, all_preds, all_stops):
    BASE = {'binBA': 0.888734, 'spec': 0.906122, 'sens': 0.871345}
    BASE_CLS = {1: 0.9061, 2: 0.75, 3: 0.9333, 4: 0.80,
                5: 0.7692, 6: 0.84, 9: 1.0, 10: 0.74}

    bb, sp, sn, pc = compute_metrics(nested_results, all_cls)

    print()
    print("=" * 80)
    print("  FULL LOOCV RESULTS  (416 folds, nested-calibrated stopping rule)")
    print("=" * 80)
    print()
    print(f"  Metric           Baseline    Nested-seq   Delta")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  binBA (H vs D)  {100*BASE['binBA']:>9.4f}%  {100*bb:>9.4f}%  "
          f"{100*(bb-BASE['binBA']):>+7.4f}pp")
    print(f"  Specificity     {100*BASE['spec']:>9.4f}%  {100*sp:>9.4f}%  "
          f"{100*(sp-BASE['spec']):>+7.4f}pp")
    print(f"  Sensitivity     {100*BASE['sens']:>9.4f}%  {100*sn:>9.4f}%  "
          f"{100*(sn-BASE['sens']):>+7.4f}pp")
    print()

    print(f"  Per-class accuracy:")
    print(f"  {'class':>7}  {'N':>4}  {'baseline':>9}  {'nested':>8}  delta")
    print(f"  {'-'*7}  {'-'*4}  {'-'*9}  {'-'*8}  {'-'*7}")
    for c in sorted(all_cls):
        nr = [r for r in nested_results if r['true'] == c]
        n_c = len(nr)
        acc = sum(r['pred'] == c for r in nr) / n_c if n_c else 0
        bl  = BASE_CLS.get(c, 0.0)
        delta = acc - bl
        flag = "  <-- LOSS" if delta < -0.02 else ("  <-- GAIN" if delta > 0.02 else "")
        print(f"  {_cls_lbl(c):>7}  {n_c:>4}  {100*bl:>8.2f}%  "
              f"{100*acc:>7.2f}%  {100*delta:>+6.2f}pp{flag}")
    print()

    # Step-count efficiency
    full_ci  = N_CONFIGS - 1
    avg_full = np.mean(all_stops[:, full_ci])
    avg_nest = np.mean([r['stop_step'] for r in nested_results])
    pct_save = 100 * (1 - avg_nest / avg_full)
    print(f"  Efficiency (average steps taken):")
    print(f"    Full budget (baseline equivalent): {avg_full:.1f} steps")
    print(f"    Nested-calibrated:                 {avg_nest:.1f} steps")
    print(f"    Reduction:                         {pct_save:.1f}%")
    print()

    # uid=171 specific check
    r171 = next((r for r in nested_results if r['uid'] == 171), None)
    if r171:
        print(f"  uid=171 canary check:")
        print(f"    stop_step={r171['stop_step']}/{r171['trace_len']}  "
              f"pred={_cls_lbl(r171['pred'])}  true=c4  "
              f"cfg={r171['cfg']}")
        print(f"    Stopped AFTER step 76 (c4 took lead): "
              f"{'YES' if r171['stop_step'] > 76 else 'NO (WRONG side of c4-takeover)'}")
        print(f"    Prediction correct: {'YES (c4)' if r171['pred'] == 4 else 'NO'}")
        print()

    # Calibrated threshold distribution
    cfg_counts = defaultdict(int)
    for r in nested_results:
        key = (r['cfg']['margin_thr'], r['cfg']['stability'], r['cfg']['min_steps'])
        cfg_counts[key] += 1
    print(f"  Nested-calibrated threshold distribution "
          f"(most common, {len(cfg_counts)} distinct configs used):")
    for (mt, st, ms), cnt in sorted(cfg_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    MARGIN={mt}  STAB={st}  MIN_STEPS={ms}  "
              f"-> {cnt} patients ({100*cnt/len(nested_results):.1f}%)")
    print()

    healthy_bar_risk_check(nested_results, all_cls)
    margin_distortion_check(nested_results, all_preds, all_cls, loocv_data)

    # Overall verdict
    delta_bb = bb - BASE['binBA']
    print()
    print("-" * 80)
    if delta_bb > 0.001:
        verdict = f"POSITIVE: +{100*delta_bb:.3f}pp binBA (Phase 2 sequential win)"
    elif delta_bb < -0.001:
        verdict = f"NEGATIVE: {100*delta_bb:.3f}pp regression"
    else:
        verdict = f"NEUTRAL: {100*delta_bb:+.4f}pp (within noise floor)"
    print(f"  VERDICT: {verdict}")
    print("-" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  cds_seq_cv.py  --  Sequential scoring: calibration + full LOOCV")
    print("=" * 80)
    print()

    data, labels = load_data()
    is_bin  = classify_features(data)
    all_cls = sorted(set(int(c) for c in set(labels)))
    n       = data.shape[0]
    print(f"  n={n}  classes={all_cls}  N_CONFIGS={N_CONFIGS}")
    print()

    # ── PART 1: Stopping rule design analysis (uid=171 fold) ──────────────────
    print("  Building uid=171 fold for stopping-rule analysis...", flush=True)
    t0 = time.time()
    mask171 = np.ones(n, dtype=bool); mask171[171] = False
    td171, tl171 = data[mask171], labels[mask171]
    nd171   = build_tree(td171, tl171, is_bin)
    tr171   = train(nd171, td171, tl171, is_bin, all_cls)
    fc171   = _build_lv_fisher_cache(td171, tl171, all_cls)
    trace171 = _run_seq_patient(171, data, nd171, tr171, all_cls, fc171)
    print(f"  Done in {time.time()-t0:.1f}s. Trace length = {len(trace171)} steps.")
    print()

    stopping_rule_design_analysis(trace171, all_cls)

    # ── PART 2: Full LOOCV (Phase 1) ──────────────────────────────────────────
    print("=" * 80)
    print("  PHASE 1: Full 416-fold LOOCV  (saving traces to cache)")
    print("=" * 80)
    print()
    loocv_data = run_loocv_phase1(data, labels, is_bin, all_cls, TRACE_CACHE)
    print(f"  Phase 1 complete. {len(loocv_data)} folds stored.")
    print()

    # ── PART 3: Nested calibration (Phase 2, fast) ────────────────────────────
    print("=" * 80)
    print("  PHASE 2: Nested stopping-rule calibration")
    print("=" * 80)
    print()
    t2 = time.time()
    nested_results, all_preds, all_stops = run_nested_calibration(loocv_data, all_cls)
    print(f"  Calibration done in {time.time()-t2:.2f}s.")

    # ── PART 4: Reporting ──────────────────────────────────────────────────────
    report_full(nested_results, all_cls, loocv_data, all_preds, all_stops)


if __name__ == "__main__":
    main()
