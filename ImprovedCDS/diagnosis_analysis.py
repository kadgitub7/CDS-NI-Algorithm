"""Diagnostic analysis of CDS arrhythmia classifier LOOCV trace data.

Analyzes misclassification patterns, per-class discriminability,
feature utilization, score calibration, and FN recoverability.
"""
import json
import numpy as np
from collections import defaultdict

HEALTHY = 1
SUSPICION_HCUT = 1.5
SUSPICION_OFFSET = 0.5
HEALTHY_WEIGHT = 1.1

TRACE_PATH = "output/loocv_trace_ovr.json"


def load_trace():
    with open(TRACE_PATH) as f:
        return json.load(f)


def effective_threshold(t, class_thresholds, h_score):
    """Compute effective threshold for a disease class given h_score."""
    base_t = class_thresholds.get(str(t), 3.3)
    if h_score < SUSPICION_HCUT:
        base_t -= SUSPICION_OFFSET
    healthy_bar = HEALTHY_WEIGHT * h_score
    return max(base_t, healthy_bar)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def analyze_misclassifications(traces):
    section("1. MISCLASSIFICATION ANALYSIS")

    misclassed = [t for t in traces if not t['correct']]
    correct = [t for t in traces if t['correct']]
    print(f"\nTotal: {len(traces)} users, {len(correct)} correct, {len(misclassed)} misclassified")

    # Categorize
    fn = []  # diseased predicted healthy
    fp = []  # healthy predicted diseased
    wd = []  # wrong disease class

    for t in misclassed:
        tc = t['true_class']
        pr = t['predicted']
        if tc != HEALTHY and pr == HEALTHY:
            fn.append(t)
        elif tc == HEALTHY and pr != HEALTHY:
            fp.append(t)
        elif tc != HEALTHY and pr != HEALTHY:
            wd.append(t)

    print(f"  False Negatives (diseased->healthy): {len(fn)}")
    print(f"  False Positives (healthy->diseased): {len(fp)}")
    print(f"  Wrong Disease (detected, wrong class): {len(wd)}")

    # --- False Negatives ---
    print(f"\n--- FALSE NEGATIVES (diseased predicted healthy) ---")
    if fn:
        true_af_for = []
        true_af_against = []
        true_n_used = []
        true_scores = []
        h_scores = []
        h_af_for = []
        gaps = []

        for t in fn:
            tc = str(t['true_class'])
            d = t['class_detail'][tc]
            true_af_for.append(d['af_for'])
            true_af_against.append(d['af_against'])
            true_n_used.append(d['n_used'])
            true_scores.append(t['class_scores'][tc])

            hd = t['class_detail']['1']
            h_scores.append(t['class_scores']['1'])
            h_af_for.append(hd['af_for'])

            eff_t = effective_threshold(int(tc), t.get('class_thresholds', {}),
                                        t['class_scores']['1'])
            gaps.append(t['class_scores'][tc] - eff_t)

        print(f"  True-class af_for:     mean={np.mean(true_af_for):.3f}  median={np.median(true_af_for):.3f}  min={np.min(true_af_for):.3f}  max={np.max(true_af_for):.3f}")
        print(f"  True-class af_against: mean={np.mean(true_af_against):.3f}  median={np.median(true_af_against):.3f}")
        print(f"  True-class n_used:     mean={np.mean(true_n_used):.1f}  median={np.median(true_n_used):.0f}")
        print(f"  True-class score:      mean={np.mean(true_scores):.3f}  median={np.median(true_scores):.3f}")
        print(f"  Healthy score:         mean={np.mean(h_scores):.3f}  median={np.median(h_scores):.3f}")
        print(f"  Healthy af_for:        mean={np.mean(h_af_for):.3f}")
        print(f"  Gap (score - threshold): mean={np.mean(gaps):.3f}  median={np.median(gaps):.3f}")

        # Per true class breakdown
        fn_by_class = defaultdict(list)
        for t in fn:
            fn_by_class[t['true_class']].append(t)
        print(f"\n  FN breakdown by true class:")
        for cls in sorted(fn_by_class.keys()):
            items = fn_by_class[cls]
            scores = [t['class_scores'][str(cls)] for t in items]
            af_fors = [t['class_detail'][str(cls)]['af_for'] for t in items]
            n_total = sum(1 for t in traces if t['true_class'] == cls)
            print(f"    Class {cls:2d}: {len(items):3d} FNs / {n_total:3d} total | "
                  f"score mean={np.mean(scores):.3f} median={np.median(scores):.3f} | "
                  f"af_for mean={np.mean(af_fors):.3f}")

    # --- False Positives ---
    print(f"\n--- FALSE POSITIVES (healthy predicted diseased) ---")
    if fp:
        for t in fp:
            pr = t['predicted']
            pr_score = t['class_scores'][str(pr)]
            h_score = t['class_scores']['1']
            eff_t = effective_threshold(pr, t.get('class_thresholds', {}), h_score)
            margin = pr_score - eff_t
            print(f"  uid={t['uid']:3d}: predicted class {pr}, score={pr_score:.3f}, "
                  f"h_score={h_score:.3f}, eff_threshold={eff_t:.2f}, margin={margin:.3f}")
    else:
        print("  None!")

    # --- Wrong Disease ---
    print(f"\n--- WRONG DISEASE (detected, wrong class) ---")
    if wd:
        for t in wd:
            tc = t['true_class']
            pr = t['predicted']
            ts = t['class_scores'][str(tc)]
            ps = t['class_scores'][str(pr)]
            print(f"  uid={t['uid']:3d}: true={tc}, pred={pr}, "
                  f"true_score={ts:.3f}, pred_score={ps:.3f}, gap={ps-ts:.3f}")


def analyze_per_class(traces):
    section("2. PER-CLASS DISCRIMINABILITY (AUC & OVERLAP)")

    all_classes = sorted(set(t['true_class'] for t in traces))
    disease_classes = [c for c in all_classes if c != HEALTHY]

    healthy_traces = [t for t in traces if t['true_class'] == HEALTHY]

    print(f"\n  {'Class':>6s}  {'N':>4s}  {'AUC':>6s}  {'Overlap':>8s}  {'MeanScore_Pos':>14s}  {'MeanScore_Neg':>14s}  {'Verdict':>12s}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*12}")

    for cls in disease_classes:
        cls_key = str(cls)
        # Positive = members of this class, Negative = healthy members
        pos_scores = [t['class_scores'][cls_key] for t in traces if t['true_class'] == cls]
        neg_scores = [t['class_scores'][cls_key] for t in healthy_traces]

        if not pos_scores or not neg_scores:
            print(f"  {cls:6d}  {len(pos_scores):4d}  {'N/A':>6s}")
            continue

        # Manual AUC computation
        pos = np.array(pos_scores)
        neg = np.array(neg_scores)
        n_pos, n_neg = len(pos), len(neg)

        auc = 0.0
        for p in pos:
            auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        auc /= (n_pos * n_neg)

        # Overlap coefficient: histogram-based
        all_vals = np.concatenate([pos, neg])
        bins = np.linspace(all_vals.min(), all_vals.max(), 30)
        h_pos, _ = np.histogram(pos, bins=bins, density=True)
        h_neg, _ = np.histogram(neg, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        overlap = np.sum(np.minimum(h_pos, h_neg)) * bin_width

        mean_pos = np.mean(pos)
        mean_neg = np.mean(neg)

        if auc >= 0.75:
            verdict = "GOOD"
        elif auc >= 0.6:
            verdict = "WEAK"
        else:
            verdict = "NEAR-RANDOM"

        print(f"  {cls:6d}  {n_pos:4d}  {auc:6.3f}  {overlap:8.3f}  {mean_pos:14.3f}  {mean_neg:14.3f}  {verdict:>12s}")


def analyze_features(traces):
    section("3. FEATURE UTILIZATION ANALYSIS")

    all_classes = sorted(set(t['true_class'] for t in traces))

    # n_used analysis
    print(f"\n  n_used values across all class models:")
    all_n_used = []
    for t in traces:
        for cls_key, d in t['class_detail'].items():
            all_n_used.append(d['n_used'])
    all_n_used = np.array(all_n_used)
    print(f"    mean={np.mean(all_n_used):.1f}  median={np.median(all_n_used):.0f}  "
          f"min={np.min(all_n_used)}  max={np.max(all_n_used)}")
    unique, counts = np.unique(all_n_used, return_counts=True)
    print(f"    Distribution: ", end="")
    for u, c in zip(unique, counts):
        print(f"{u}:{c}  ", end="")
    print()

    # af_for for correct vs incorrect predictions per class
    print(f"\n  Average af_for: correct vs incorrect predictions per class")
    print(f"  {'Class':>6s}  {'N':>4s}  {'af_for(correct)':>16s}  {'af_for(incorrect)':>18s}  {'Ratio':>6s}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*16}  {'-'*18}  {'-'*6}")

    for cls in all_classes:
        cls_key = str(cls)
        cls_traces = [t for t in traces if t['true_class'] == cls]
        if not cls_traces:
            continue

        correct_af = [t['class_detail'][cls_key]['af_for']
                      for t in cls_traces if t['correct']]
        incorrect_af = [t['class_detail'][cls_key]['af_for']
                        for t in cls_traces if not t['correct']]

        c_mean = np.mean(correct_af) if correct_af else 0.0
        i_mean = np.mean(incorrect_af) if incorrect_af else 0.0
        ratio = c_mean / i_mean if i_mean > 0.001 else float('inf')

        print(f"  {cls:6d}  {len(cls_traces):4d}  {c_mean:16.4f}  {i_mean:18.4f}  {ratio:6.2f}")


def analyze_calibration(traces):
    section("4. SCORE CALIBRATION ANALYSIS")

    # For each disease class: TP scores vs FN scores
    all_classes = sorted(set(t['true_class'] for t in traces))
    disease_classes = [c for c in all_classes if c != HEALTHY]

    print(f"\n  Score distributions: True Positives vs False Negatives")
    print(f"  {'Class':>6s}  {'TP':>4s}  {'FN':>4s}  {'TP_score':>10s}  {'FN_score':>10s}  "
          f"{'TP_min':>8s}  {'FN_max':>8s}  {'Near_misses':>12s}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*12}")

    total_near_misses = 0

    for cls in disease_classes:
        cls_key = str(cls)
        cls_traces = [t for t in traces if t['true_class'] == cls]
        if not cls_traces:
            continue

        tp_traces = [t for t in cls_traces if t['predicted'] != HEALTHY]
        fn_traces = [t for t in cls_traces if t['predicted'] == HEALTHY]

        tp_scores = [t['class_scores'][cls_key] for t in tp_traces]
        fn_scores = [t['class_scores'][cls_key] for t in fn_traces]

        # Near misses: FN with score > 50% of threshold
        near_misses = 0
        for t in fn_traces:
            h_score = t['class_scores']['1']
            eff_t = effective_threshold(cls, t.get('class_thresholds', {}), h_score)
            score = t['class_scores'][cls_key]
            if score > 0.5 * eff_t:
                near_misses += 1

        total_near_misses += near_misses

        tp_mean = np.mean(tp_scores) if tp_scores else 0
        fn_mean = np.mean(fn_scores) if fn_scores else 0
        tp_min = np.min(tp_scores) if tp_scores else 0
        fn_max = np.max(fn_scores) if fn_scores else 0

        print(f"  {cls:6d}  {len(tp_traces):4d}  {len(fn_traces):4d}  {tp_mean:10.3f}  {fn_mean:10.3f}  "
              f"{tp_min:8.3f}  {fn_max:8.3f}  {near_misses:12d}")

    print(f"\n  Total near-misses (FN score > 50% of threshold): {total_near_misses}")


def analyze_recoverability(traces):
    section("5. FN RECOVERABILITY ANALYSIS")

    fn_traces = [t for t in traces if t['true_class'] != HEALTHY and t['predicted'] == HEALTHY]

    if not fn_traces:
        print("  No false negatives!")
        return

    # Categories by af_for of true class
    recoverable = []    # af_for > 1.0 but score below threshold
    marginal = []       # af_for between 0.5 and 1.0
    hopeless = []       # af_for < 0.5

    for t in fn_traces:
        tc = str(t['true_class'])
        af_for = t['class_detail'][tc]['af_for']
        af_against = t['class_detail'][tc]['af_against']
        score = t['class_scores'][tc]
        h_score = t['class_scores']['1']

        entry = {
            'uid': t['uid'],
            'true_class': t['true_class'],
            'af_for': af_for,
            'af_against': af_against,
            'score': score,
            'h_score': h_score,
            'n_used': t['class_detail'][tc]['n_used']
        }

        if af_for > 1.0:
            recoverable.append(entry)
        elif af_for >= 0.5:
            marginal.append(entry)
        else:
            hopeless.append(entry)

    print(f"\n  Total FNs: {len(fn_traces)}")
    print(f"  Recoverable (af_for > 1.0): {len(recoverable)} ({100*len(recoverable)/len(fn_traces):.1f}%)")
    print(f"  Marginal    (0.5 <= af_for <= 1.0): {len(marginal)} ({100*len(marginal)/len(fn_traces):.1f}%)")
    print(f"  Hopeless    (af_for < 0.5): {len(hopeless)} ({100*len(hopeless)/len(fn_traces):.1f}%)")

    if recoverable:
        print(f"\n  --- Recoverable FNs (evidence exists but below threshold) ---")
        print(f"  {'uid':>4s}  {'Class':>5s}  {'af_for':>7s}  {'af_agn':>7s}  {'score':>7s}  {'h_score':>7s}  {'n_used':>6s}")
        for e in sorted(recoverable, key=lambda x: -x['af_for']):
            print(f"  {e['uid']:4d}  {e['true_class']:5d}  {e['af_for']:7.3f}  {e['af_against']:7.3f}  "
                  f"{e['score']:7.3f}  {e['h_score']:7.3f}  {e['n_used']:6d}")

    if marginal:
        print(f"\n  --- Marginal FNs (some signal, needs stronger features) ---")
        print(f"  {'uid':>4s}  {'Class':>5s}  {'af_for':>7s}  {'af_agn':>7s}  {'score':>7s}  {'h_score':>7s}")
        for e in sorted(marginal, key=lambda x: -x['af_for']):
            print(f"  {e['uid']:4d}  {e['true_class']:5d}  {e['af_for']:7.3f}  {e['af_against']:7.3f}  "
                  f"{e['score']:7.3f}  {e['h_score']:7.3f}")

    # Diagnose WHY recoverable FNs fail
    if recoverable:
        print(f"\n  --- Why do recoverable FNs fail? ---")
        for e in recoverable:
            t = next(t for t in fn_traces if t['uid'] == e['uid'])
            tc = str(t['true_class'])
            h_score = t['class_scores']['1']
            eff_t = effective_threshold(int(tc), t.get('class_thresholds', {}), h_score)
            healthy_bar = HEALTHY_WEIGHT * h_score
            base_t = t.get('class_thresholds', {}).get(tc, 3.3)

            reason = []
            if e['score'] < base_t:
                reason.append(f"below base threshold ({e['score']:.2f} < {base_t})")
            if e['score'] < healthy_bar:
                reason.append(f"below healthy bar ({e['score']:.2f} < {healthy_bar:.2f})")
            if not reason:
                reason.append(f"score={e['score']:.2f} vs eff_t={eff_t:.2f}")

            print(f"    uid {e['uid']:3d} class {e['true_class']:2d}: {'; '.join(reason)}")

    # Summary: impact of recovering FNs
    print(f"\n  --- Impact Analysis ---")
    n_total = len(traces)
    n_correct = sum(1 for t in traces if t['correct'])
    print(f"  Current accuracy: {n_correct}/{n_total} = {100*n_correct/n_total:.1f}%")
    print(f"  If all recoverable FNs fixed: {n_correct+len(recoverable)}/{n_total} = "
          f"{100*(n_correct+len(recoverable))/n_total:.1f}%")
    print(f"  If all recoverable+marginal fixed: {n_correct+len(recoverable)+len(marginal)}/{n_total} = "
          f"{100*(n_correct+len(recoverable)+len(marginal))/n_total:.1f}%")
    print(f"  Theoretical max (fix all FNs): {n_correct+len(fn_traces)}/{n_total} = "
          f"{100*(n_correct+len(fn_traces))/n_total:.1f}%")


def analyze_threshold_sensitivity(traces):
    section("6. THRESHOLD SENSITIVITY (bonus)")

    # What if we lowered thresholds? Track accuracy change
    fn_traces = [t for t in traces if t['true_class'] != HEALTHY and t['predicted'] == HEALTHY]
    fp_risk = [t for t in traces if t['true_class'] == HEALTHY]

    print(f"\n  Simulating threshold reductions:")
    print(f"  {'Delta':>6s}  {'FNs_recovered':>14s}  {'New_FPs':>8s}  {'Net_gain':>9s}  {'New_Acc':>8s}")

    n_correct = sum(1 for t in traces if t['correct'])
    n_total = len(traces)

    for delta in [0.0, -0.2, -0.5, -0.8, -1.0, -1.5]:
        recovered = 0
        new_fps = 0

        for t in fn_traces:
            tc = str(t['true_class'])
            score = t['class_scores'][tc]
            h_score = t['class_scores']['1']
            eff_t = effective_threshold(int(tc), t.get('class_thresholds', {}), h_score)
            if score >= eff_t + delta:  # negative delta = lower threshold
                recovered += 1

        for t in fp_risk:
            # Check if any disease class would now exceed threshold
            was_healthy = t['predicted'] == HEALTHY
            if not was_healthy:
                continue
            for cls_key, score in t['class_scores'].items():
                if cls_key == '1':
                    continue
                h_score = t['class_scores']['1']
                eff_t = effective_threshold(int(cls_key), t.get('class_thresholds', {}), h_score)
                if score >= eff_t + delta and score < eff_t:
                    new_fps += 1
                    break

        net = recovered - new_fps
        new_acc = (n_correct + net) / n_total
        print(f"  {delta:6.1f}  {recovered:14d}  {new_fps:8d}  {net:9d}  {100*new_acc:7.1f}%")


def main():
    traces = load_trace()
    print(f"Loaded {len(traces)} trace records from {TRACE_PATH}")

    analyze_misclassifications(traces)
    analyze_per_class(traces)
    analyze_features(traces)
    analyze_calibration(traces)
    analyze_recoverability(traces)
    analyze_threshold_sensitivity(traces)

    section("SUMMARY")
    n_total = len(traces)
    n_correct = sum(1 for t in traces if t['correct'])
    fn = [t for t in traces if t['true_class'] != HEALTHY and t['predicted'] == HEALTHY]
    fp = [t for t in traces if t['true_class'] == HEALTHY and t['predicted'] != HEALTHY]
    wd = [t for t in traces if not t['correct'] and t['true_class'] != HEALTHY and t['predicted'] != HEALTHY]

    print(f"\n  Overall accuracy: {100*n_correct/n_total:.1f}%")
    print(f"  Error breakdown: {len(fn)} FN + {len(fp)} FP + {len(wd)} wrong-disease = {len(fn)+len(fp)+len(wd)} errors")
    print(f"  Primary loss channel: {'FN' if len(fn) >= len(fp) and len(fn) >= len(wd) else 'FP' if len(fp) >= len(wd) else 'WD'} ({max(len(fn),len(fp),len(wd))} cases)")
    print()


if __name__ == "__main__":
    main()
