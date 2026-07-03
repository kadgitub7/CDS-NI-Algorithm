"""Validation & diagnostic script for cds_dualAF.

Instruments every stage of the algorithm and outputs a structured report.
Run: python validate.py [N]   (N = number of LOOCV users, default 50)

Sections:
  1. Data overview — class distribution, feature stats, missing values
  2. Tree structure — node populations, prevalence per node
  3. Training diagnostics — likelihood quality, posterior spread, confidence
  4. Feature selection — retained features, discriminative power, correlation
  5. Evidence analysis — per-user accumulation, per-class separation
  6. Decision analysis — correct vs wrong, where errors concentrate
  7. Hyperparameter sensitivity — margin × min_support sweep
  8. Mini-LOOCV — quick generalization estimate
"""
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

import cds_dualAF as m


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def load():
    dp = str(Path(__file__).parent / "data" / "arrhythmia.data")
    data, labels = m.load_data(dp)
    is_bin = m.classify_features(data)
    all_cls = sorted(set(labels))
    return data, labels, is_bin, all_cls


def train_full(data, labels, is_bin, all_cls):
    nodes = m.build_tree(data, labels, is_bin)
    all_models = {}
    actions_by_node = defaultdict(list)
    for nd in nodes:
        nm, na = m.train_node(nd, data, labels, is_bin, all_cls)
        all_models.update(nm)
        for a in na:
            actions_by_node[a[1]].append(a)
    retained = []
    for nd in nodes:
        retained.extend(m.refine_node(nd, all_models,
                                      actions_by_node.get(nd.nid, []), data))
    return nodes, all_models, retained, actions_by_node


# ──────────────────────────────────────────────────────────────────
# Section 1: Data Overview
# ──────────────────────────────────────────────────────────────────

def report_data(data, labels, is_bin):
    section("1. DATA OVERVIEW")
    n, nf = data.shape
    n_h = int((labels == m.HEALTHY).sum())
    n_u = n - n_h
    print(f"Users: {n}  (healthy={n_h}, unhealthy={n_u}, prevalence={n_h/n:.3f})")
    print(f"Features: {nf}  (binary={int(is_bin.sum())}, continuous={nf - int(is_bin.sum())})")

    miss = np.isnan(data).sum(axis=0)
    miss_feats = (miss > 0).sum()
    print(f"Missing values: {int(np.isnan(data).sum())} total across {miss_feats} features")
    if miss_feats > 0:
        worst = np.argsort(miss)[-5:][::-1]
        for f in worst:
            if miss[f] > 0:
                print(f"  Feature {f}: {int(miss[f])} missing ({miss[f]/n*100:.1f}%)")

    print(f"\nClass distribution:")
    for cls in sorted(set(labels)):
        cnt = int((labels == cls).sum())
        lbl = "healthy" if cls == m.HEALTHY else f"class {cls}"
        print(f"  {lbl:>10s}: {cnt:4d} ({cnt/n*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────
# Section 2: Tree Structure
# ──────────────────────────────────────────────────────────────────

def report_tree(nodes, labels):
    section("2. TREE STRUCTURE")
    for nd in nodes:
        n_h = nd.hdist.get(m.HEALTHY, 0)
        n_u = nd.nu - n_h
        prev = m._node_prevalence(nd)
        split = ""
        if nd.bfeat is not None:
            if nd.bbin:
                split = f"  split: feat {nd.bfeat} in {nd.bvset}"
            else:
                split = f"  split: feat {nd.bfeat} {'<=' if nd.bhi != np.inf else '>'} {nd.bhi if nd.bhi != np.inf else nd.blo}"
        print(f"Node {nd.nid:>12s}: {nd.nu:4d} users ({n_h}H/{n_u}U, prev={prev:.3f}){split}")


# ──────────────────────────────────────────────────────────────────
# Section 3: Training Diagnostics
# ──────────────────────────────────────────────────────────────────

def report_training(nodes, all_models, retained, labels):
    section("3. TRAINING DIAGNOSTICS")
    healthy_idx = 0

    for nd in nodes:
        node_retained = [a for a in retained if a[1] == nd.nid]
        node_feats = [a[0] for a in node_retained]
        prev = m._node_prevalence(nd)

        print(f"\n--- Node {nd.nid} ({len(node_retained)} retained features, prevalence={prev:.3f}) ---")

        posteriors = []
        confidences = []
        centered_evidence_magnitudes = []
        for f in node_feats:
            mo = all_models.get((nd.nid, f))
            if not mo:
                continue
            for b in range(mo.n_bins):
                if mo.bin_counts[b] >= m.MIN_SUPPORT:
                    p_h = mo.posterior[b, healthy_idx]
                    posteriors.append(p_h)
                    confidences.append(mo.confidence[b])
                    centered_evidence_magnitudes.append(abs(p_h - prev) * mo.confidence[b]**2)

        if posteriors:
            p = np.array(posteriors)
            c = np.array(confidences)
            e = np.array(centered_evidence_magnitudes)
            print(f"  Posteriors P(H|bin):  mean={p.mean():.3f}, std={p.std():.3f}, min={p.min():.3f}, max={p.max():.3f}")
            print(f"  Confidence:           mean={c.mean():.3f}, std={c.std():.3f}, min={c.min():.3f}, max={c.max():.3f}")
            print(f"  Centered evidence |e|: mean={e.mean():.4f}, std={e.std():.4f}, max={e.max():.4f}")
            print(f"  Bins with P(H) > 0.7: {int((p > 0.7).sum())} / {len(p)}")
            print(f"  Bins with P(H) < 0.3: {int((p < 0.3).sum())} / {len(p)}")
            print(f"  Bins near prevalence (±0.05): {int((abs(p - prev) < 0.05).sum())} / {len(p)}")

        print(f"\n  Top 5 most discriminative retained features:")
        feat_scores = []
        for a in node_retained:
            f = a[0]
            mo = all_models.get((nd.nid, f))
            if not mo:
                continue
            max_dev = max(abs(mo.posterior[b, healthy_idx] - prev) * mo.confidence[b]**2
                         for b in range(mo.n_bins) if mo.bin_counts[b] >= m.MIN_SUPPORT)
            feat_scores.append((f, max_dev, a[2], a[3]))
        feat_scores.sort(key=lambda x: x[1], reverse=True)
        for f, score, rh, ru in feat_scores[:5]:
            print(f"    Feature {f:3d}: max|centered_ev|={score:.4f}, r_H={rh:.4f}, r_U={ru:.4f}")


# ──────────────────────────────────────────────────────────────────
# Section 4: Feature Selection
# ──────────────────────────────────────────────────────────────────

def report_features(nodes, all_models, retained, actions_by_node, data):
    section("4. FEATURE SELECTION")
    for nd in nodes:
        all_actions = actions_by_node.get(nd.nid, [])
        node_retained = [a for a in retained if a[1] == nd.nid]
        diffs = [abs(a[2] - a[3]) for a in all_actions if abs(a[2] - a[3]) > 0.01]
        ret_diffs = [abs(a[2] - a[3]) for a in node_retained]
        print(f"\nNode {nd.nid}:")
        print(f"  Candidate features (|r_H-r_U|>0.01): {len(diffs)}")
        print(f"  Retained: {len(node_retained)}")
        if diffs:
            print(f"  Candidate |r_H-r_U|: max={max(diffs):.4f}, mean={np.mean(diffs):.4f}")
        if ret_diffs:
            print(f"  Retained  |r_H-r_U|: max={max(ret_diffs):.4f}, mean={np.mean(ret_diffs):.4f}, min={min(ret_diffs):.4f}")

        overlap = set(a[0] for a in node_retained)
        for other_nd in nodes:
            if other_nd.nid == nd.nid:
                continue
            other_feats = set(a[0] for a in retained if a[1] == other_nd.nid)
            shared = overlap & other_feats
            if shared:
                print(f"  Shared with {other_nd.nid}: {len(shared)} features {sorted(shared)[:10]}{'...' if len(shared)>10 else ''}")


# ──────────────────────────────────────────────────────────────────
# Section 5: Evidence Analysis
# ──────────────────────────────────────────────────────────────────

def report_evidence(data, labels, nodes, all_models, retained):
    section("5. EVIDENCE ANALYSIS")
    healthy_idx = 0

    user_traces = []
    for uid in range(data.shape[0]):
        af_net = 0.0
        n_feats = 0
        evidence_list = []
        lvl_nodes = m._route_user(uid, data, nodes)
        for lvl in sorted(lvl_nodes.keys()):
            for nd in lvl_nodes[lvl]:
                prev = m._node_prevalence(nd)
                for a in retained:
                    if a[1] != nd.nid:
                        continue
                    f = a[0]
                    v = data[uid, f]
                    if np.isnan(v):
                        continue
                    mo = all_models.get((nd.nid, f))
                    if not mo:
                        continue
                    bi = int(np.clip(np.searchsorted(mo.edges[1:], v, side='right'),
                                     0, mo.n_bins - 1))
                    if mo.bin_counts[bi] < m.MIN_SUPPORT:
                        continue
                    p_h = mo.posterior[bi, healthy_idx]
                    conf = mo.confidence[bi]
                    ev = (p_h - prev) * conf * conf
                    af_net += ev
                    evidence_list.append(ev)
                    n_feats += 1

        user_traces.append({
            'uid': uid, 'label': labels[uid], 'af_net': af_net,
            'n_feats': n_feats, 'evidence': evidence_list,
            'pred': 'H' if af_net > 0 else 'U',
            'correct': (af_net > 0) == (labels[uid] == m.HEALTHY)
        })

    print(f"\n--- Per-class evidence summary ---")
    print(f"{'Class':>10s} {'N':>4s} {'AF_net':>8s} {'Std':>7s} {'|AF|':>7s} {'Feats':>5s} {'Acc%':>6s}")
    for cls in sorted(set(labels)):
        ct = [t for t in user_traces if t['label'] == cls]
        if not ct:
            continue
        lbl = "healthy" if cls == m.HEALTHY else f"cls {cls}"
        nets = [t['af_net'] for t in ct]
        acc = sum(1 for t in ct if t['correct']) / len(ct) * 100
        print(f"{lbl:>10s} {len(ct):4d} {np.mean(nets):8.4f} {np.std(nets):7.4f} "
              f"{np.mean(np.abs(nets)):7.4f} {np.mean([t['n_feats'] for t in ct]):5.1f} {acc:6.1f}")

    h_traces = [t for t in user_traces if t['label'] == m.HEALTHY]
    u_traces = [t for t in user_traces if t['label'] != m.HEALTHY]
    print(f"\nHealthy:   mean AF_net={np.mean([t['af_net'] for t in h_traces]):+.4f}, "
          f"median={np.median([t['af_net'] for t in h_traces]):+.4f}")
    print(f"Unhealthy: mean AF_net={np.mean([t['af_net'] for t in u_traces]):+.4f}, "
          f"median={np.median([t['af_net'] for t in u_traces]):+.4f}")

    h_nets = np.array([t['af_net'] for t in h_traces])
    u_nets = np.array([t['af_net'] for t in u_traces])
    overlap = (h_nets.min() < u_nets.max()) and (u_nets.min() < h_nets.max())
    if overlap:
        h_wrong = (h_nets < 0).sum()
        u_wrong = (u_nets > 0).sum()
        print(f"\nOverlap zone: {h_wrong} healthy misclassified ({h_wrong/len(h_traces)*100:.1f}%), "
              f"{u_wrong} unhealthy misclassified ({u_wrong/len(u_traces)*100:.1f}%)")

    print(f"\n--- Evidence magnitude per feature ---")
    all_ev = [abs(e) for t in user_traces for e in t['evidence']]
    if all_ev:
        all_ev = np.array(all_ev)
        print(f"  Mean |evidence| per feature: {all_ev.mean():.5f}")
        print(f"  Median: {np.median(all_ev):.5f}")
        print(f"  95th percentile: {np.percentile(all_ev, 95):.5f}")
        print(f"  Max: {all_ev.max():.5f}")
        pct_tiny = (all_ev < 0.001).sum() / len(all_ev) * 100
        print(f"  Fraction < 0.001 (near-zero evidence): {pct_tiny:.1f}%")

    return user_traces


# ──────────────────────────────────────────────────────────────────
# Section 6: Decision Analysis
# ──────────────────────────────────────────────────────────────────

def report_decisions(user_traces, labels):
    section("6. DECISION ANALYSIS")

    wrong = [t for t in user_traces if not t['correct']]
    right = [t for t in user_traces if t['correct']]

    print(f"Correct: {len(right)}/{len(user_traces)} ({len(right)/len(user_traces)*100:.1f}%)")
    print(f"Wrong: {len(wrong)}/{len(user_traces)} ({len(wrong)/len(user_traces)*100:.1f}%)")

    fn_users = [t for t in wrong if t['label'] != m.HEALTHY]
    fp_users = [t for t in wrong if t['label'] == m.HEALTHY]
    print(f"\nFalse negatives (unhealthy predicted healthy): {len(fn_users)}")
    print(f"False positives (healthy predicted unhealthy): {len(fp_users)}")

    if fn_users:
        print(f"\n--- False negative details ---")
        print(f"  AF_net: mean={np.mean([t['af_net'] for t in fn_users]):+.4f}, "
              f"range=[{min(t['af_net'] for t in fn_users):+.4f}, {max(t['af_net'] for t in fn_users):+.4f}]")
        cls_dist = defaultdict(int)
        for t in fn_users:
            cls_dist[t['label']] += 1
        print(f"  By class: {dict(sorted(cls_dist.items()))}")

        close_calls = [t for t in fn_users if abs(t['af_net']) < 0.05]
        print(f"  Close calls (|AF_net| < 0.05): {len(close_calls)} — these could flip with small changes")

    if fp_users:
        print(f"\n--- False positive details ---")
        print(f"  AF_net: mean={np.mean([t['af_net'] for t in fp_users]):+.4f}, "
              f"range=[{min(t['af_net'] for t in fp_users):+.4f}, {max(t['af_net'] for t in fp_users):+.4f}]")

    print(f"\n--- Confidence distribution (|AF_net|) ---")
    all_nets = np.array([abs(t['af_net']) for t in user_traces])
    for pct in [10, 25, 50, 75, 90]:
        print(f"  {pct}th percentile: {np.percentile(all_nets, pct):.4f}")


# ──────────────────────────────────────────────────────────────────
# Section 7: Hyperparameter Sensitivity
# ──────────────────────────────────────────────────────────────────

def report_hyperparam_sensitivity(data, labels, nodes, all_models, retained):
    section("7. HYPERPARAMETER SENSITIVITY (resubstitution)")
    print(f"{'margin':>7s} {'min_s':>5s} {'Acc':>6s} {'Sens':>6s} {'Spec':>6s} {'BAcc':>6s} {'Scr%':>5s}")

    for margin in [0.0, 0.01, 0.02, 0.05, 0.10]:
        for ms in [2, 3, 5]:
            tp = tn = fp = fn = scr = 0
            for uid in range(data.shape[0]):
                dec, _ = m.predict_dual_af(uid, data, nodes, all_models, retained,
                                           margin=margin, min_support=ms)
                is_h = labels[uid] == m.HEALTHY
                if dec == "SCREENING":
                    scr += 1
                    if not is_h: fn += 1
                elif dec == "HEALTHY":
                    if is_h: tn += 1
                    else: fn += 1
                else:
                    if not is_h: tp += 1
                    else: fp += 1

            n = len(labels)
            total_u = sum(1 for l in labels if l != m.HEALTHY)
            total_h = sum(1 for l in labels if l == m.HEALTHY)
            acc = (tp+tn)/n*100
            sens = tp/total_u*100
            spec = tn/total_h*100
            bacc = (sens+spec)/2
            print(f"{margin:7.2f} {ms:5d} {acc:6.1f} {sens:6.1f} {spec:6.1f} {bacc:6.1f} {scr/n*100:5.1f}")


# ──────────────────────────────────────────────────────────────────
# Section 8: Mini-LOOCV
# ──────────────────────────────────────────────────────────────────

def report_loocv(data, labels, is_bin, all_cls, n_users):
    full = (n_users is None or n_users >= data.shape[0])
    n_actual = data.shape[0] if full else n_users
    label = "FULL" if full else "MINI"
    section(f"8. {label}-LOOCV ({n_actual} users)")
    t0 = time.perf_counter()

    if full:
        test_idx = list(range(data.shape[0]))
    else:
        h_idx = [i for i in range(len(labels)) if labels[i] == m.HEALTHY]
        u_idx = [i for i in range(len(labels)) if labels[i] != m.HEALTHY]
        step_h = max(1, len(h_idx) // (n_users // 2))
        step_u = max(1, len(u_idx) // (n_users // 2))
        test_idx = h_idx[::step_h][:n_users//2] + u_idx[::step_u][:n_users//2]

    actual_h = sum(1 for i in test_idx if labels[i] == m.HEALTHY)
    actual_u = len(test_idx) - actual_h
    print(f"Testing {len(test_idx)} users ({actual_h}H, {actual_u}U)")

    results = []
    cached_margin, cached_ms = 0.0, m.MIN_SUPPORT
    segment_size = max(50, len(test_idx) // 8)

    for idx, uid in enumerate(test_idx):
        mask = np.ones(data.shape[0], dtype=bool)
        mask[uid] = False
        td, tl = data[mask], labels[mask]

        nodes = m.build_tree(td, tl, is_bin)
        all_models = {}
        actions_by_node = defaultdict(list)
        for nd in nodes:
            nm, na = m.train_node(nd, td, tl, is_bin, all_cls)
            all_models.update(nm)
            for a in na:
                actions_by_node[a[1]].append(a)
        retained = []
        for nd in nodes:
            retained.extend(m.refine_node(nd, all_models,
                                          actions_by_node.get(nd.nid, []), td))

        if idx % m.INNER_CV_INTERVAL == 0:
            cached_margin, cached_ms = \
                m.select_hyperparams(td, tl, nodes, all_models, retained)

        dec, npacs = m.predict_dual_af(uid, data, nodes, all_models, retained,
                                       margin=cached_margin, min_support=cached_ms)

        tl_i = int(labels[uid])
        ok = (dec != "UNHEALTHY") if tl_i == m.HEALTHY else (dec == "UNHEALTHY")
        results.append((uid, tl_i, dec, ok, npacs))

        if (idx + 1) % segment_size == 0 or idx == len(test_idx) - 1:
            seg_results = results[-segment_size:] if len(results) >= segment_size else results
            seg_acc = sum(r[3] for r in seg_results) / len(seg_results) * 100
            cum_acc = sum(r[3] for r in results) / len(results) * 100
            elapsed = time.perf_counter() - t0
            print(f"  [{idx+1}/{len(test_idx)}] segment_acc={seg_acc:.1f}%  cum_acc={cum_acc:.1f}%  "
                  f"params=(margin={cached_margin}, ms={cached_ms})  {elapsed:.0f}s")

    n = len(results)
    tp = sum(1 for r in results if r[2] == "UNHEALTHY" and r[1] != m.HEALTHY)
    tn = sum(1 for r in results if r[2] == "HEALTHY" and r[1] == m.HEALTHY)
    fp = sum(1 for r in results if r[2] == "UNHEALTHY" and r[1] == m.HEALTHY)
    fn = n - tp - tn - fp
    scr = sum(1 for r in results if r[2] == "SCREENING")

    acc = (tp+tn)/n*100
    sens = tp/actual_u*100 if actual_u else 0
    spec = tn/actual_h*100 if actual_h else 0
    bacc = (sens+spec)/2

    print(f"\nFinal Results:")
    print(f"  Accuracy:    {acc:.1f}%")
    print(f"  Sensitivity: {sens:.1f}%")
    print(f"  Specificity: {spec:.1f}%")
    print(f"  Balanced:    {bacc:.1f}%")
    print(f"  Screening:   {scr/n*100:.1f}%")
    print(f"  Time: {time.perf_counter()-t0:.1f}s")

    per_class = defaultdict(lambda: [0, 0])
    for r in results:
        per_class[r[1]][1] += 1
        if r[3]:
            per_class[r[1]][0] += 1
    print(f"\n  Per-class:")
    for cls in sorted(per_class):
        correct, total = per_class[cls]
        lbl = "healthy" if cls == m.HEALTHY else f"class {cls}"
        print(f"    {lbl:>10s}: {correct}/{total} ({correct/total*100:.0f}%)")

    print(f"\n  Decision distribution:")
    for dec in ["HEALTHY", "UNHEALTHY", "SCREENING"]:
        cnt = sum(1 for r in results if r[2] == dec)
        if cnt > 0:
            dec_correct = sum(1 for r in results if r[2] == dec and r[3])
            print(f"    {dec:>12s}: {cnt} ({cnt/n*100:.1f}%), accuracy within={dec_correct/cnt*100:.1f}%")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "50"
    n_loocv = None if arg.lower() == "full" else int(arg)
    t_total = time.perf_counter()

    data, labels, is_bin, all_cls = load()
    nodes, all_models, retained, actions_by_node = train_full(data, labels, is_bin, all_cls)

    report_data(data, labels, is_bin)
    report_tree(nodes, labels)
    report_training(nodes, all_models, retained, labels)
    report_features(nodes, all_models, retained, actions_by_node, data)
    traces = report_evidence(data, labels, nodes, all_models, retained)
    report_decisions(traces, labels)
    report_hyperparam_sensitivity(data, labels, nodes, all_models, retained)
    report_loocv(data, labels, is_bin, all_cls, n_loocv)

    print(f"\n{'='*70}")
    print(f"Total validation time: {time.perf_counter()-t_total:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
