"""Count multiplications in CDS-OVR V4 PAC algorithm.

Instruments prediction and training paths to count every multiplication
operation: * operator, / operator, np.sqrt, np.dot, np.multiply.
Each scalar operation counts as 1. Element-wise array ops count as len(array).
Division counts as 1 multiplication. np.sqrt counts as 1 per element.
Additions, comparisons, and indexing are NOT counted.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cds_ovr_v4_pac_twd as pac
from collections import defaultdict
import time


# ============================================================
# Global counter
# ============================================================

class MultCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.by_loc = defaultdict(int)

    def add(self, n, loc=""):
        self.count += n
        if loc:
            self.by_loc[loc] += n

mc = MultCounter()


# ============================================================
# Instrumented _compute_af  (prediction path)
# ============================================================

def counted_compute_af(uid, data, nodes, models, retained, against_scale):
    """Identical to pac._compute_af but counts multiplications."""
    lvl_nodes = pac._route_user(uid, data, nodes)
    af_for, af_against = 0.0, 0.0
    n_for, n_against, n_used = 0, 0, 0
    max_for_contrib = 0.0

    fisher_map = {}
    for a in retained:
        fisher_map[a[0]] = max(fisher_map.get(a[0], 0), a[3])
    max_fisher = max(fisher_map.values()) if fisher_map else 1.0

    # Phase 1: Collect evaluable features
    evaluable = []
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
                evaluable.append((f, mo, bin_idx, bc, a[3]))

    # Phase 2: Sort by Fisher (RL policy)
    evaluable.sort(key=lambda x: x[4], reverse=True)

    # Phase 3: Sequential perception-action cycle
    trajectory = []
    for f, mo, bin_idx, bc, fisher_raw in evaluable:
        p_c = mo.p_class[bin_idx]
        shift = p_c - mo.prior                              # subtraction, 0 mults

        # confidence = min(1.0, bc / 10)
        mc.add(1, "af:confidence_div")                      # bc / 10
        confidence = min(1.0, bc / 10)

        # fw = max(np.sqrt(fisher / (max_fisher + eps)), 0.1)
        mc.add(1, "af:fisher_div")                          # fisher / (max_fisher + eps)
        mc.add(1, "af:sqrt")                                # np.sqrt(...)
        fw = max(np.sqrt(fisher_map.get(f, 0.0) / (max_fisher + 1e-10)), 0.1)

        # weighted = abs(shift) * confidence * fw
        mc.add(2, "af:weighted_mult")                       # 2 multiplications
        weighted = abs(shift) * confidence * fw

        if shift >= 0:
            af_for += weighted
            n_for += 1
            if weighted > max_for_contrib:
                max_for_contrib = weighted
        else:
            # af_against += weighted * against_scale
            mc.add(1, "af:against_scale_mult")              # 1 multiplication
            af_against += weighted * against_scale
            n_against += 1

        n_used += 1

        # trajectory: (af_for + eps) / (af_against + eps)
        mc.add(1, "af:trajectory_div")                      # 1 division
        trajectory.append((af_for + pac.RATIO_EPS) / (af_against + pac.RATIO_EPS))

    return af_for, af_against, n_used, n_for, n_against, max_for_contrib, trajectory


# ============================================================
# Instrumented predict  (prediction path)
# ============================================================

def counted_predict(uid, data, nodes, all_cls, train_result):
    """Identical to pac.predict but counts multiplications."""
    class_models, class_retained = train_result
    disease_cls = [c for c in all_cls if c != pac.HEALTHY]

    # Phase 1: Healthy evaluation
    ag_h = pac.AGAINST_SCALE_MAP.get(pac.HEALTHY, 0.8)
    af_h = counted_compute_af(uid, data, nodes, class_models[pac.HEALTHY],
                              class_retained[pac.HEALTHY], ag_h)
    h_score = af_h[6][-1] if af_h[6] else 1.0

    # healthy_bar = min(HEALTHY_WEIGHT * h_score, HEALTHY_BAR_CAP)
    mc.add(1, "predict:healthy_weight_mult")                # HEALTHY_WEIGHT * h_score
    healthy_bar = min(pac.HEALTHY_WEIGHT * h_score, pac.HEALTHY_BAR_CAP)

    # Phase 2: Disease class trajectories
    class_trajs = {}
    class_thresholds = {}
    class_final_scores = {pac.HEALTHY: h_score}
    for cls in disease_cls:
        ag = pac.AGAINST_SCALE_MAP.get(cls, 0.8)
        af = counted_compute_af(uid, data, nodes, class_models[cls],
                                class_retained[cls], ag)
        traj = af[6]
        class_trajs[cls] = traj
        class_final_scores[cls] = traj[-1] if traj else 1.0
        t = pac.CLASS_THRESHOLDS.get(cls, 3.0)
        if h_score < pac.SUSPICION_HCUT:
            t -= pac.SUSPICION_OFFSET
        class_thresholds[cls] = max(t, healthy_bar)

    # Phase 3: PAC synchronized stepping
    max_steps = max((len(t) for t in class_trajs.values()), default=0)
    early_stopped = False
    early_step = -1
    for step in range(max_steps):
        early_candidates = {}
        for cls in disease_cls:
            traj = class_trajs[cls]
            if step >= len(traj):
                continue
            ratio = traj[step]
            t = class_thresholds[cls]
            if ratio < t:
                continue
            # excess = (ratio - t) / max(t, 0.1)
            mc.add(1, "predict:excess_div")                 # 1 division
            excess = (ratio - t) / max(t, 0.1)
            if excess >= pac.PAC_EXCESS_THRESHOLD:
                early_candidates[cls] = excess
        if early_candidates:
            early_stopped = True
            early_step = step
            return (max(early_candidates, key=early_candidates.get),
                    class_final_scores, early_stopped, early_step, max_steps)

    # Phase 4: Standard decision
    candidates = {}
    for cls in disease_cls:
        score = class_final_scores[cls]
        t = class_thresholds[cls]
        if score < t:
            continue
        # (score - t) / max(t, 0.1)
        mc.add(1, "predict:final_excess_div")               # 1 division
        candidates[cls] = (score - t) / max(t, 0.1)

    best_cls = max(candidates, key=candidates.get) if candidates else pac.HEALTHY
    return best_cls, class_final_scores, early_stopped, early_step, max_steps


# ============================================================
# Instrumented _supervised_bin_edges  (training path)
# ============================================================

def counted_supervised_bin_edges(vv, is_target, max_bins, min_support):
    """Identical to pac._supervised_bin_edges but counts multiplications."""
    n = len(vv)
    vmin, vmax = float(vv.min()), float(vv.max())
    if vmin == vmax or n < 2 * min_support:
        mc.add(1, "bin:2*min_support")                      # 2 * min_support
        return np.array([vmin - 0.5, vmax + 0.5])
    mc.add(1, "bin:2*min_support_check")

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
            mc.add(1, "bin:2*min_support_seg")              # 2 * min_support
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

                # e_tl = n_left * n_t / n_seg (and 3 more like it)
                mc.add(8, "bin:expected_vals")              # 4 * (1 mult + 1 div) = 8
                e_tl = n_left * n_t / n_seg
                e_tr = n_right * n_t / n_seg
                e_rl = n_left * n_r / n_seg
                e_rr = n_right * n_r / n_seg

                # chi2 = sum((o-e)**2 / e ...)
                terms = [(t_left, e_tl), (t_right, e_tr),
                         (r_left, e_rl), (r_right, e_rr)]
                chi2 = 0.0
                for o, e in terms:
                    if e > 0:
                        mc.add(2, "bin:chi2_term")          # (o-e)**2 = 1 mult, /e = 1 div
                        chi2 += (o - e)**2 / e

                if chi2 > best_gain:
                    best_gain = chi2
                    # (seg_vals[ci] + seg_vals[ci+1]) / 2.0
                    mc.add(1, "bin:split_div")              # 1 division
                    best_split = (seg_vals[ci] + seg_vals[ci + 1]) / 2.0
                    best_seg = seg_i
        if best_split is None or best_gain < 0.5:
            break
        edges.insert(best_seg + 1, best_split)

    edges[0] = vmin - 1e-10
    edges[-1] = vmax + 1e-10
    return np.array(sorted(set(edges)))


# ============================================================
# Instrumented _train_ovr_node  (training path)
# ============================================================

def counted_train_ovr_node(node, data, labels, is_bin, target_class, min_support, conf_support):
    """Identical to pac._train_ovr_node but counts multiplications."""
    models, actions = {}, []
    nd_data, nd_labels = data[node.uidx], labels[node.uidx]
    ns = node.nu
    n_target = int((nd_labels == target_class).sum())
    if n_target < 1:
        return models, actions
    # prior = n_target / ns
    mc.add(1, "train:prior_div")
    prior = n_target / ns
    is_target = (nd_labels == target_class)

    for f in range(pac.N_FEAT):
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
            # max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), MAX_BINS)
            # np.log2 not counted, np.ceil not counted
            max_nb = min(max(2, int(np.ceil(1 + np.log2(nv)))), pac.MAX_BINS)
            edges = counted_supervised_bin_edges(
                vv, is_target[vm].astype(float), max_nb, min_support)
            nb = len(edges) - 1

        ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)
        lv = nd_labels[vm]
        bin_counts = np.bincount(ba, minlength=nb)
        target_counts = np.bincount(ba[lv == target_class], minlength=nb).astype(float)

        # p_class = (target_counts + LAPLACE_ALPHA * prior) / (bin_counts + LAPLACE_ALPHA)
        mc.add(1, "train:laplace_alpha_mult")               # LAPLACE_ALPHA * prior (scalar)
        mc.add(nb, "train:p_class_div")                     # element-wise division over nb bins
        p_class = (target_counts + pac.LAPLACE_ALPHA * prior) / (bin_counts + pac.LAPLACE_ALPHA)

        models[(node.nid, f)] = pac.BinModel(nb, edges, bin_counts, target_counts,
                                             p_class, prior, conf_support)

        score = 0.0
        for b in range(nb):
            if bin_counts[b] >= min_support:
                shift = abs(p_class[b] - prior)             # subtraction + abs, 0 mults
                # confidence = min(1.0, float(bin_counts[b]) / conf_support)
                mc.add(1, "train:score_conf_div")           # 1 division
                confidence = min(1.0, float(bin_counts[b]) / conf_support)
                # score += shift * confidence
                mc.add(1, "train:score_mult")               # 1 multiplication
                score += shift * confidence

        target_vals = vv[lv == target_class]
        rest_vals = vv[lv != target_class]
        if len(target_vals) >= 2 and len(rest_vals) >= 2:
            # mean_diff2 = (target_vals.mean() - rest_vals.mean()) ** 2
            # .mean() = sum/n = 1 div each; **2 = 1 mult
            mc.add(2, "train:mean_div")                     # 2 .mean() calls
            mc.add(1, "train:mean_diff_sq")                 # **2
            mean_diff2 = (target_vals.mean() - rest_vals.mean()) ** 2

            # target_vals.var() = sum((x-m)^2)/n
            # n_t squarings + 1 div for target, n_r squarings + 1 div for rest
            nt = len(target_vals)
            nr = len(rest_vals)
            mc.add(nt, "train:var_target_sq")               # squarings
            mc.add(1, "train:var_target_div")               # /n
            mc.add(nr, "train:var_rest_sq")                 # squarings
            mc.add(1, "train:var_rest_div")                 # /n
            var_sum = target_vals.var() + rest_vals.var()

            # fisher = mean_diff2 / (var_sum + eps)
            mc.add(1, "train:fisher_div")                   # 1 division
            fisher = mean_diff2 / (var_sum + 1e-10)
        else:
            fisher = 0.0

        if score > 0.001:
            actions.append((f, node.nid, score, fisher))

    return models, actions


# ============================================================
# Instrumented _refine_ovr_node  (training path)
# ============================================================

def counted_refine_ovr_node(node, models, node_actions, data, fpc=pac.FEATURES_PER_CLASS):
    """Identical to pac._refine_ovr_node but counts multiplications."""
    scored = [(a, a[2]) for a in node_actions]
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []

    # top_scored = scored[:3 * fpc]
    mc.add(1, "refine:3*fpc")                               # 3 * fpc
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
                # _fast_abs_corr: mx=x.mean(1div), my=y.mean(1div),
                #   (dx*dy).sum() = n mults, (dx*dx).sum()=n, (dy*dy).sum()=n
                #   product of sums = 1 mult, sqrt = 1, final div = 1
                n_valid = int(valid.sum())
                mc.add(2, "refine:corr_mean_div")           # 2 .mean()
                mc.add(n_valid, "refine:corr_dx_dy")        # dx*dy
                mc.add(n_valid, "refine:corr_dx_dx")        # dx*dx
                mc.add(n_valid, "refine:corr_dy_dy")        # dy*dy
                mc.add(1, "refine:corr_den_mult")           # sum1 * sum2
                mc.add(1, "refine:corr_sqrt")               # np.sqrt
                mc.add(1, "refine:corr_final_div")          # num / sqrt(den2)
                c = pac._fast_abs_corr(col1[valid], col2[valid])
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
            # max_corr = max(correlations.get(...) for kf in kept_features)
            # no mults, just lookups
            max_corr = max(correlations.get((f, kf), 0.0) for kf in kept_features)
            if max_corr > pac.CORR_THRESHOLD:
                continue
        kept.append(a)
        kept_features.add(f)
        if len(kept) >= fpc:
            break
    return kept


# ============================================================
# Instrumented train
# ============================================================

def counted_train(nodes, data, labels, is_bin, all_cls):
    class_models, class_retained = {}, {}
    for cls in all_cls:
        ms = pac.MIN_SUPPORT_MAP.get(cls, 3)
        cs = pac.CONF_SUPPORT_MAP.get(cls, 10)
        cls_models = {}
        cls_actions = defaultdict(list)
        for nd in nodes:
            nm, na = counted_train_ovr_node(nd, data, labels, is_bin, cls, ms, cs)
            cls_models.update(nm)
            for a in na:
                cls_actions[a[1]].append(a)
        cls_ret = []
        for nd in nodes:
            cls_ret.extend(counted_refine_ovr_node(
                nd, cls_models, cls_actions.get(nd.nid, []), data))
        class_models[cls] = cls_models
        class_retained[cls] = cls_ret
    return class_models, class_retained


# ============================================================
# Main: Run 10-fold CV with instrumentation
# ============================================================

def main():
    print("Loading data...")
    data, labels = pac.load_data()
    is_bin = pac.classify_features(data)
    n = data.shape[0]
    all_cls = sorted(set(labels))
    print(f"  {n} patients, {data.shape[1]} features, {len(all_cls)} classes")
    print(f"  Classes: {all_cls}")
    print(f"  PAC_EXCESS_THRESHOLD = {pac.PAC_EXCESS_THRESHOLD}\n")

    seed = 13
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 10)

    # Accumulators
    pred_mults = []                # mults per patient prediction
    pred_details = []              # (uid, true_cls, pred_cls, mults, early_stopped, early_step, max_steps, breakdown)
    train_mults_per_fold = []      # total training mults per fold
    correct_count = 0
    total_count = 0

    t_start = time.time()

    for fi in range(10):
        print(f"Fold {fi+1}/10...", flush=True)
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(10) if j != fi])
        td, tl = data[train_idx], labels[train_idx]
        nodes = pac.build_tree(td, tl, is_bin)

        # === Count training multiplications ===
        mc.reset()
        train_result = counted_train(nodes, td, tl, is_bin, all_cls)
        train_mults = mc.count
        train_breakdown = dict(mc.by_loc)
        train_mults_per_fold.append(train_mults)

        # === Count prediction multiplications per patient ===
        for uid in test_idx:
            true_cls = int(labels[uid])
            mc.reset()
            pred_cls, scores, early_stopped, early_step, max_steps = \
                counted_predict(uid, data, nodes, all_cls, train_result)
            patient_mults = mc.count
            patient_breakdown = dict(mc.by_loc)
            pred_mults.append(patient_mults)
            pred_details.append((int(uid), true_cls, int(pred_cls), patient_mults,
                                early_stopped, early_step, max_steps, patient_breakdown))
            if pred_cls == true_cls:
                correct_count += 1
            total_count += 1

    elapsed = time.time() - t_start

    # ============================================================
    # Report
    # ============================================================

    print("\n" + "="*70)
    print("MULTIPLICATION COUNT RESULTS")
    print("="*70)

    # Verify accuracy matches
    acc = correct_count / total_count
    print(f"\nVerification: accuracy = {100*acc:.1f}% ({correct_count}/{total_count})")
    print(f"Total time: {elapsed:.1f}s\n")

    # --- Prediction statistics ---
    pred_arr = np.array(pred_mults)
    print("-"*50)
    print("PREDICTION (per patient)")
    print("-"*50)
    print(f"  Mean multiplications:   {pred_arr.mean():.1f}")
    print(f"  Median multiplications: {np.median(pred_arr):.1f}")
    print(f"  Min multiplications:    {pred_arr.min()}")
    print(f"  Max multiplications:    {pred_arr.max()}")
    print(f"  Std dev:                {pred_arr.std():.1f}")
    print(f"  Total (all {total_count} patients):  {pred_arr.sum()}")

    # Early stop analysis
    n_early = sum(1 for d in pred_details if d[4])
    n_full = total_count - n_early
    early_mults = [d[3] for d in pred_details if d[4]]
    full_mults = [d[3] for d in pred_details if not d[4]]
    print(f"\n  Early-stopped patients:  {n_early} ({100*n_early/total_count:.1f}%)")
    print(f"  Full-eval patients:      {n_full} ({100*n_full/total_count:.1f}%)")
    if early_mults:
        print(f"  Mean mults (early-stop): {np.mean(early_mults):.1f}")
    if full_mults:
        print(f"  Mean mults (full-eval):  {np.mean(full_mults):.1f}")

    # Estimate savings from early stop
    # For early-stopped patients, count how many features were skipped
    early_steps_saved = []
    for d in pred_details:
        if d[4]:  # early_stopped
            early_step = d[5]
            max_steps = d[6]
            steps_saved = max_steps - early_step - 1
            early_steps_saved.append(steps_saved)
    if early_steps_saved:
        print(f"  Mean features skipped per early-stop: {np.mean(early_steps_saved):.1f}")

    # Breakdown by operation type (aggregate across all patients)
    print(f"\n  Breakdown (summed across all {total_count} predictions):")
    agg_breakdown = defaultdict(int)
    for d in pred_details:
        for k, v in d[7].items():
            agg_breakdown[k] += v
    for k in sorted(agg_breakdown.keys()):
        print(f"    {k:40s} {agg_breakdown[k]:>10,}")

    # --- Training statistics ---
    train_arr = np.array(train_mults_per_fold)
    print(f"\n{'-'*50}")
    print("TRAINING (per fold)")
    print("-"*50)
    print(f"  Mean multiplications per fold:  {train_arr.mean():,.0f}")
    print(f"  Min:                            {train_arr.min():,}")
    print(f"  Max:                            {train_arr.max():,}")
    print(f"  Total (all 10 folds):           {train_arr.sum():,}")

    print(f"\n  Training breakdown (fold 0):")
    # Re-run fold 0 to get breakdown
    test_idx = folds[0]
    train_idx = np.concatenate([folds[j] for j in range(10) if j != 0])
    td, tl = data[train_idx], labels[train_idx]
    nodes = pac.build_tree(td, tl, is_bin)
    mc.reset()
    counted_train(nodes, td, tl, is_bin, all_cls)
    for k in sorted(mc.by_loc.keys()):
        print(f"    {k:40s} {mc.by_loc[k]:>12,}")

    # --- Summary table for paper ---
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (for paper)")
    print("="*70)
    print(f"  Algorithm: CDS-OVR V4 with PAC early-stop")
    print(f"  Dataset: {n} patients, {data.shape[1]} features, {len(all_cls)} classes")
    print(f"  Evaluation: 10-fold CV, seed={seed}")
    print(f"  Accuracy: {100*acc:.1f}%")
    print(f"")
    print(f"  --- Prediction Cost ---")
    print(f"  Multiplications per patient (mean): {pred_arr.mean():.1f}")
    print(f"  Multiplications per patient (min):  {pred_arr.min()}")
    print(f"  Multiplications per patient (max):  {pred_arr.max()}")
    print(f"  Patients early-stopped:             {n_early}/{total_count} ({100*n_early/total_count:.1f}%)")
    if early_mults and full_mults:
        savings_pct = (1 - np.mean(early_mults)/np.mean(full_mults)) * 100
        print(f"  Mult reduction from early-stop:     {savings_pct:.1f}%")
    print(f"")
    print(f"  --- Training Cost (per fold) ---")
    print(f"  Multiplications per fold (mean):    {train_arr.mean():,.0f}")
    print(f"")
    print(f"  --- Timing (from results_cost.json) ---")
    print(f"  10-fold CV time:                    260.67 s")
    print(f"  Training time:                      249.78 s")
    print(f"  Prediction time (all patients):     17.07 s")
    print(f"  Prediction time per patient:        41.03 ms")
    print(f"  Memory (peak):                      4.27 MB")


if __name__ == "__main__":
    main()
