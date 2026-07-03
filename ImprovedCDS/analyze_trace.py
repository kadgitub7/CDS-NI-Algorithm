"""Analyze LOOCV trace data — diagnose what works, what fails, and why."""
import json
import sys
from collections import defaultdict
from pathlib import Path

def load_trace(path):
    with open(path) as f:
        return json.load(f)

def analyze(traces):
    all_cls = sorted(set(t['true_class'] for t in traces))
    n = len(traces)

    # 1. Overall accuracy
    correct = sum(1 for t in traces if t['correct'])
    print(f"{'='*80}")
    print(f"OVERALL: {correct}/{n} = {100*correct/n:.1f}%")
    print(f"{'='*80}\n")

    # 2. Per-class breakdown with confusion
    print("PER-CLASS ACCURACY + CONFUSION:")
    print(f"{'Class':>8s} {'N':>4s} {'Correct':>8s} {'Acc%':>6s}  Misclassified as...")
    print("-" * 80)
    for cls in all_cls:
        cls_traces = [t for t in traces if t['true_class'] == cls]
        n_cls = len(cls_traces)
        n_correct = sum(1 for t in cls_traces if t['correct'])
        wrong = defaultdict(int)
        for t in cls_traces:
            if not t['correct']:
                wrong[t['predicted']] += 1
        wrong_str = ", ".join(f"c{k}({v})" for k, v in
                              sorted(wrong.items(), key=lambda x: -x[1])[:5])
        lbl = "healthy" if cls == 1 else f"class {cls}"
        print(f"{lbl:>8s} {n_cls:4d} {n_correct:8d} {100*n_correct/n_cls:5.1f}%  {wrong_str}")
    print()

    # 3. For each WRONG prediction, analyze WHY
    print("=" * 80)
    print("ERROR ANALYSIS — WHAT WENT WRONG FOR EACH MISCLASSIFIED USER")
    print("=" * 80)

    # Group errors by true class
    for cls in all_cls:
        errors = [t for t in traces if t['true_class'] == cls and not t['correct']]
        if not errors:
            continue

        lbl = "healthy" if cls == 1 else f"class {cls}"
        print(f"\n--- {lbl} errors ({len(errors)} users) ---")

        for t in errors[:5]:  # Show up to 5 per class
            uid = t['uid']
            pred = t['predicted']
            af = t['af_final']
            n_feats = t['n_features_used']

            true_af = af.get(str(cls), 0)
            pred_af = af.get(str(pred), 0)

            print(f"\n  User {uid}: true=c{cls}, predicted=c{pred}")
            print(f"    AF for true class (c{cls}): {true_af:+.3f}")
            print(f"    AF for pred class (c{pred}): {pred_af:+.3f}")
            print(f"    Margin: {pred_af - true_af:+.3f} ({n_feats} features used)")

            # Top 3 AF values
            sorted_af = sorted(af.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"    Top AF: {', '.join(f'c{k}={v:+.3f}' for k, v in sorted_af)}")

            # Which features pushed TOWARD wrong class and AWAY from true class?
            feats = t.get('features', [])
            if feats:
                pushers_wrong = []
                pushers_right = []
                for ft in feats:
                    if 'skipped' in ft:
                        continue
                    ev = ft.get('evidence_contrib', {})
                    ev_true = ev.get(str(cls), 0)
                    ev_pred = ev.get(str(pred), 0)
                    pushers_wrong.append((ft['feat'], ft['node'], ev_pred, ev_true,
                                          ft['bin'], ft['n_bins'], ft['pop'],
                                          ft.get('decisiveness', 0)))

                # Sort by how much they pushed toward wrong class
                pushers_wrong.sort(key=lambda x: x[2], reverse=True)
                print(f"    Top features pushing toward c{pred} (wrong):")
                for f, node, ev_pred, ev_true, bi, nb, pop, dec in pushers_wrong[:5]:
                    print(f"      F{f:3d} @{node} bin {bi}/{nb} pop={pop} "
                          f"dec={dec:.2f} ev_pred={ev_pred:+.4f} ev_true={ev_true:+.4f}")

                # Features that pushed away from true class
                pushers_away = sorted(pushers_wrong, key=lambda x: x[3])
                print(f"    Top features pushing away from c{cls} (true):")
                for f, node, ev_pred, ev_true, bi, nb, pop, dec in pushers_away[:5]:
                    print(f"      F{f:3d} @{node} bin {bi}/{nb} pop={pop} "
                          f"dec={dec:.2f} ev_pred={ev_pred:+.4f} ev_true={ev_true:+.4f}")

    # 4. Feature importance: which features appear most in correct vs incorrect
    print("\n" + "=" * 80)
    print("FEATURE CONTRIBUTION PATTERNS")
    print("=" * 80)

    feat_correct_push = defaultdict(float)
    feat_wrong_push = defaultdict(float)
    feat_count = defaultdict(int)

    for t in traces:
        cls = t['true_class']
        for ft in t.get('features', []):
            if 'skipped' in ft:
                continue
            ev = ft.get('evidence_contrib', {})
            ev_true = ev.get(str(cls), 0)
            fkey = ft['feat']
            feat_count[fkey] += 1
            if t['correct']:
                feat_correct_push[fkey] += ev_true
            else:
                feat_wrong_push[fkey] += ev_true

    print("\nFeatures that contribute MOST to correct predictions (high ev for true class):")
    top_helpful = sorted(feat_correct_push.items(), key=lambda x: x[1], reverse=True)[:20]
    for f, total_ev in top_helpful:
        wrong_ev = feat_wrong_push.get(f, 0)
        print(f"  F{f:3d}: correct_ev={total_ev:+.1f}  wrong_ev={wrong_ev:+.1f}  "
              f"used {feat_count[f]} times")

    print("\nFeatures that contribute MOST to wrong predictions (push away from true class):")
    top_harmful = sorted(feat_wrong_push.items(), key=lambda x: x[1])[:20]
    for f, total_ev in top_harmful:
        correct_ev = feat_correct_push.get(f, 0)
        print(f"  F{f:3d}: wrong_ev={total_ev:+.1f}  correct_ev={correct_ev:+.1f}  "
              f"used {feat_count[f]} times")

    # 5. Decisiveness distribution: are we weighting correctly?
    print("\n" + "=" * 80)
    print("DECISIVENESS DISTRIBUTION")
    print("=" * 80)

    dec_correct = []
    dec_wrong = []
    for t in traces:
        for ft in t.get('features', []):
            if 'skipped' in ft:
                continue
            d = ft.get('decisiveness', 0)
            if t['correct']:
                dec_correct.append(d)
            else:
                dec_wrong.append(d)

    if dec_correct:
        import numpy as np
        dc = np.array(dec_correct)
        dw = np.array(dec_wrong) if dec_wrong else np.array([0])
        print(f"  Correct predictions — decisiveness: "
              f"mean={dc.mean():.3f} median={np.median(dc):.3f} "
              f"p90={np.percentile(dc,90):.3f} max={dc.max():.3f}")
        print(f"  Wrong predictions   — decisiveness: "
              f"mean={dw.mean():.3f} median={np.median(dw):.3f} "
              f"p90={np.percentile(dw,90):.3f} max={dw.max():.3f}")

    # 6. The key diagnostic: for wrong predictions, how often did the
    #    true class have a STRONG signal that got drowned?
    print("\n" + "=" * 80)
    print("SIGNAL vs NOISE: Did correct class have strong features that got drowned?")
    print("=" * 80)

    drowned_count = 0
    no_signal_count = 0
    for t in traces:
        if t['correct']:
            continue
        cls = t['true_class']
        feats = t.get('features', [])
        if not feats:
            continue
        max_true_ev = max((ft.get('evidence_contrib', {}).get(str(cls), 0)
                           for ft in feats if 'skipped' not in ft), default=0)
        if max_true_ev > 0.5:
            drowned_count += 1
        else:
            no_signal_count += 1

    total_wrong = sum(1 for t in traces if not t['correct'])
    print(f"  Total wrong: {total_wrong}")
    print(f"  Had strong true-class signal (>0.5) but got drowned: {drowned_count}")
    print(f"  No strong true-class signal at all: {no_signal_count}")

    # 7. Healthy users misclassified as disease — what's going on?
    print("\n" + "=" * 80)
    print("HEALTHY USERS MISCLASSIFIED AS DISEASE")
    print("=" * 80)
    h_errors = [t for t in traces if t['true_class'] == 1 and not t['correct']]
    if h_errors:
        pred_dist = defaultdict(int)
        for t in h_errors:
            pred_dist[t['predicted']] += 1
        print(f"  {len(h_errors)} healthy users misclassified:")
        for k, v in sorted(pred_dist.items(), key=lambda x: -x[1]):
            print(f"    as class {k}: {v}")

        # Show a few examples
        for t in h_errors[:3]:
            uid = t['uid']
            pred = t['predicted']
            af = t['af_final']
            sorted_af = sorted(af.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n  User {uid}: predicted c{pred}")
            print(f"    AF: {', '.join(f'c{k}={v:+.3f}' for k, v in sorted_af)}")
            print(f"    AF(healthy): {af.get('1', 0):+.3f}")


if __name__ == "__main__":
    trace_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent / "output" / "loocv_trace.json")
    traces = load_trace(trace_path)
    analyze(traces)
