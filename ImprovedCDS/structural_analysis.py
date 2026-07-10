"""Deep structural analysis of V3 errors to identify radical changes."""
import json
import numpy as np
from collections import defaultdict

def load_log(path):
    with open(path) as f:
        return json.load(f)

def analyze(log):
    traces = log['user_traces']

    correct_by_class = defaultdict(list)
    wrong_by_class = defaultdict(list)
    healthy_fps = []
    disease_fns = []
    wrong_subtype = []
    all_records = []

    for uid_key, trace in traces.items():
        uid = trace['uid']
        true_cls = trace['true_class']
        pred_cls = trace['final_pred']
        is_correct = trace['correct']
        pl = trace['prediction_log']
        decision = pl['decision']
        cad = pl['class_af_details']

        true_key = str(true_cls)
        true_af = cad.get(true_key, {})
        if not true_af:
            continue

        features = true_af.get('features', [])
        n_for = sum(1 for f in features if f['direction'] == 'FOR')
        n_against = sum(1 for f in features if f['direction'] == 'AGAINST')
        n_skip = sum(1 for f in features if f.get('skipped') is not None)

        for_contribs = [f['weighted_contrib'] for f in features
                        if f['direction'] == 'FOR']
        against_contribs = [f['weighted_contrib'] for f in features
                           if f['direction'] == 'AGAINST']

        af_for = true_af['af_for']
        af_against = true_af['af_against']
        ratio = true_af['ratio']

        density_for = af_for / max(n_for, 1)
        total_feats = n_for + n_against
        breadth = n_for / total_feats if total_feats > 0 else 0
        net = af_for - af_against
        concentration = (max(for_contribs) / af_for
                        if for_contribs and af_for > 0 else 0)

        h_score = decision.get('h_score', 0)
        healthy_bar = decision.get('healthy_bar_capped',
                                   decision.get('healthy_bar', 0))

        # Collect ALL class scores for this patient
        all_class_ratios = {}
        all_class_nets = {}
        all_class_afs = {}
        for cls_key, cls_data in cad.items():
            c = int(cls_key)
            all_class_ratios[c] = cls_data['ratio']
            all_class_nets[c] = cls_data['af_for'] - cls_data['af_against']
            all_class_afs[c] = (cls_data['af_for'], cls_data['af_against'])

        # Rank of true class among all classes
        sorted_by_ratio = sorted(all_class_ratios.items(),
                                 key=lambda x: -x[1])
        true_rank_ratio = next(i for i, (c, _) in enumerate(sorted_by_ratio)
                               if c == true_cls) + 1
        sorted_by_net = sorted(all_class_nets.items(), key=lambda x: -x[1])
        true_rank_net = next(i for i, (c, _) in enumerate(sorted_by_net)
                             if c == true_cls) + 1

        record = {
            'uid': uid, 'true': true_cls, 'pred': pred_cls,
            'af_for': af_for, 'af_against': af_against, 'ratio': ratio,
            'n_for': n_for, 'n_against': n_against, 'n_skip': n_skip,
            'density_for': density_for, 'breadth': breadth,
            'net': net, 'concentration': concentration,
            'h_score': h_score, 'healthy_bar': healthy_bar,
            'true_rank_ratio': true_rank_ratio,
            'true_rank_net': true_rank_net,
            'all_class_ratios': all_class_ratios,
            'all_class_nets': all_class_nets,
            'all_class_afs': all_class_afs,
        }

        all_records.append(record)
        if is_correct:
            correct_by_class[true_cls].append(record)
        else:
            wrong_by_class[true_cls].append(record)
            if true_cls == 1:
                healthy_fps.append(record)
            elif pred_cls == 1:
                disease_fns.append(record)
            else:
                wrong_subtype.append(record)

    print("=" * 80)
    print("  STRUCTURAL ERROR ANALYSIS — LOOKING FOR RADICAL CHANGES")
    print("=" * 80)
    print(f"  Total: {len(all_records)} patients, "
          f"{sum(len(v) for v in correct_by_class.values())} correct, "
          f"{sum(len(v) for v in wrong_by_class.values())} wrong")
    print(f"  Errors: {len(healthy_fps)} healthy FPs, "
          f"{len(disease_fns)} disease->H FNs, "
          f"{len(wrong_subtype)} wrong subtype")

    disease_classes = sorted(c for c in
                             set(correct_by_class) | set(wrong_by_class)
                             if c != 1)

    # ===== 1. ALTERNATIVE METRICS =====
    print("\n" + "=" * 80)
    print("  1. ALTERNATIVE METRICS — PER-CLASS SEPARATION POWER")
    print("=" * 80)

    for cls in disease_classes:
        c = correct_by_class.get(cls, [])
        w = wrong_by_class.get(cls, [])
        if not c or not w:
            continue

        print(f"\n  --- Class {cls} ({len(c)} correct, {len(w)} wrong) ---")

        for name, fn in [
            ('ratio', lambda r: r['ratio']),
            ('net (for-ag)', lambda r: r['net']),
            ('af_for only', lambda r: r['af_for']),
            ('breadth', lambda r: r['breadth']),
            ('density_for', lambda r: r['density_for']),
            ('breadth*ratio', lambda r: r['breadth'] * r['ratio']),
            ('net*breadth', lambda r: r['net'] * r['breadth']),
        ]:
            c_vals = [fn(r) for r in c]
            w_vals = [fn(r) for r in w]
            c_mean = np.mean(c_vals)
            w_mean = np.mean(w_vals)
            sep = c_mean / max(abs(w_mean), 0.001)
            c_min = min(c_vals)
            rejected = sum(1 for v in w_vals if v < c_min)
            print(f"    {name:20s}  C={c_mean:8.3f}  W={w_mean:8.3f}  "
                  f"sep={sep:6.2f}x  gate@Cmin={rejected}/{len(w)}")

    # ===== 2. TRUE CLASS RANK ANALYSIS =====
    print("\n" + "=" * 80)
    print("  2. TRUE CLASS RANK — WHERE DOES THE TRUE CLASS SCORE?")
    print("=" * 80)
    print("  For wrong patients, what rank is the true class among all 8?")
    print("  rank=1 means true class has highest score (wrong pred is threshold issue)")
    print("  rank>1 means another class genuinely outscores true class")

    for cls in disease_classes:
        w = wrong_by_class.get(cls, [])
        if not w:
            continue
        ranks_ratio = [r['true_rank_ratio'] for r in w]
        ranks_net = [r['true_rank_net'] for r in w]
        print(f"\n  Class {cls} ({len(w)} errors):")
        print(f"    Ratio rank: {sorted(ranks_ratio)}  "
              f"mean={np.mean(ranks_ratio):.1f}")
        print(f"    Net rank:   {sorted(ranks_net)}  "
              f"mean={np.mean(ranks_net):.1f}")

        # How many would be fixed by "pick highest scoring class"?
        rank1_ratio = sum(1 for r in ranks_ratio if r == 1)
        rank1_net = sum(1 for r in ranks_net if r == 1)
        print(f"    True class is #1 by ratio: {rank1_ratio}/{len(w)}  "
              f"(these are threshold-blocked)")
        print(f"    True class is #1 by net: {rank1_net}/{len(w)}")

    # Also for healthy FPs
    h_ranks_ratio = [r['true_rank_ratio'] for r in healthy_fps]
    h_ranks_net = [r['true_rank_net'] for r in healthy_fps]
    print(f"\n  Healthy FPs ({len(healthy_fps)} errors):")
    print(f"    Healthy rank by ratio: mean={np.mean(h_ranks_ratio):.1f}  "
          f"rank1={sum(1 for r in h_ranks_ratio if r == 1)}/{len(healthy_fps)}")
    print(f"    Healthy rank by net:   mean={np.mean(h_ranks_net):.1f}  "
          f"rank1={sum(1 for r in h_ranks_net if r == 1)}/{len(healthy_fps)}")

    # ===== 3. ERROR MECHANISM BREAKDOWN =====
    print("\n" + "=" * 80)
    print("  3. ERROR MECHANISM BREAKDOWN")
    print("=" * 80)

    n_blocked_by_bar = 0
    n_below_thresh = 0
    n_outscored = 0

    for cls in disease_classes:
        for r in wrong_by_class.get(cls, []):
            if r['pred'] == 1:
                if r['healthy_bar'] > 3.5:
                    n_blocked_by_bar += 1
                else:
                    n_below_thresh += 1
            else:
                n_outscored += 1

    print(f"\n  Disease errors ({len(disease_fns) + len(wrong_subtype)} total):")
    print(f"    Blocked by healthy_bar (bar>3.5): {n_blocked_by_bar}")
    print(f"    Below class threshold:            {n_below_thresh}")
    print(f"    Outscored by wrong class:         {n_outscored}")
    print(f"  Healthy FPs:                        {len(healthy_fps)}")

    # ===== 4. EVIDENCE BREADTH PATTERNS =====
    print("\n" + "=" * 80)
    print("  4. EVIDENCE BREADTH (n_for / total features)")
    print("=" * 80)

    for cls in disease_classes:
        c = correct_by_class.get(cls, [])
        w = wrong_by_class.get(cls, [])
        if not c or not w:
            continue
        c_breadth = [r['breadth'] for r in c]
        w_breadth = [r['breadth'] for r in w]
        print(f"  Class {cls}: correct={np.mean(c_breadth):.3f}±{np.std(c_breadth):.3f}  "
              f"wrong={np.mean(w_breadth):.3f}±{np.std(w_breadth):.3f}  "
              f"overlap={'HIGH' if min(c_breadth) < max(w_breadth) else 'NONE'}")

    # ===== 5. CONCENTRATION =====
    print("\n" + "=" * 80)
    print("  5. EVIDENCE CONCENTRATION (top feature / total_for)")
    print("=" * 80)

    for cls in disease_classes:
        c = correct_by_class.get(cls, [])
        w = wrong_by_class.get(cls, [])
        if not c or not w:
            continue
        c_conc = [r['concentration'] for r in c]
        w_conc = [r['concentration'] for r in w]
        print(f"  Class {cls}: correct={np.mean(c_conc):.3f}  "
              f"wrong={np.mean(w_conc):.3f}  "
              f"{'WRONG MORE CONCENTRATED' if np.mean(w_conc) > np.mean(c_conc) else 'similar'}")

    # ===== 6. DISEASE FN DETAIL =====
    print("\n" + "=" * 80)
    print("  6. DISEASE->HEALTHY FN STRUCTURAL PATTERNS")
    print("=" * 80)

    fn_by_true = defaultdict(list)
    for r in disease_fns:
        fn_by_true[r['true']].append(r)

    for cls in sorted(fn_by_true):
        recs = fn_by_true[cls]
        correct = correct_by_class.get(cls, [])
        print(f"\n  --- Class {cls}: {len(recs)} FNs vs {len(correct)} correct ---")

        if correct:
            c_for = np.mean([r['af_for'] for r in correct])
            w_for = np.mean([r['af_for'] for r in recs])
            print(f"    Correct af_for={c_for:.3f}  FN af_for={w_for:.3f}  "
                  f"({100*w_for/c_for:.0f}% of correct)")

        for r in recs:
            # What would help this patient?
            true_af_for, true_af_ag = r['all_class_afs'][r['true']]
            h_af_for, h_af_ag = r['all_class_afs'][1]
            print(f"    uid={r['uid']}: ratio={r['ratio']:.2f} "
                  f"net={r['net']:.3f} breadth={r['breadth']:.2f} "
                  f"n_for={r['n_for']} n_ag={r['n_against']} "
                  f"h_score={r['h_score']:.2f} bar={r['healthy_bar']:.2f} "
                  f"rank_ratio={r['true_rank_ratio']} rank_net={r['true_rank_net']}")

    # ===== 7. GLOBAL METRIC COMPARISON =====
    print("\n" + "=" * 80)
    print("  7. GLOBAL SEPARATION: WHICH METRIC IS BEST?")
    print("=" * 80)

    for metric_name, fn in [
        ('ratio', lambda r: r['ratio']),
        ('net', lambda r: r['net']),
        ('af_for', lambda r: r['af_for']),
        ('breadth*ratio', lambda r: r['breadth'] * r['ratio']),
    ]:
        all_c = []
        all_w = []
        for cls in disease_classes:
            all_c.extend(fn(r) for r in correct_by_class.get(cls, []))
            all_w.extend(fn(r) for r in wrong_by_class.get(cls, []))
        if all_c and all_w:
            sep = np.mean(all_c) / max(abs(np.mean(all_w)), 0.001)
            print(f"  {metric_name:20s}: correct={np.mean(all_c):.3f}  "
                  f"wrong={np.mean(all_w):.3f}  sep={sep:.2f}x")

    # ===== 8. WHAT IF WE PICK HIGHEST NET/RATIO CLASS? =====
    print("\n" + "=" * 80)
    print("  8. SIMULATED: WHAT IF PREDICTION = HIGHEST SCORING CLASS?")
    print("=" * 80)

    for method_name, score_dict_key in [
        ('ratio (current)', 'all_class_ratios'),
        ('net (af_for - af_against)', 'all_class_nets'),
    ]:
        correct_count = 0
        for r in all_records:
            scores = r[score_dict_key]
            best_cls = max(scores, key=scores.get)
            if best_cls == r['true']:
                correct_count += 1
        print(f"  {method_name}: {correct_count}/{len(all_records)} = "
              f"{100*correct_count/len(all_records):.1f}%")

    # ===== 9. WHAT IF WE USE RATIO BUT WITH NET-BASED HEALTHY GATE? =====
    print("\n" + "=" * 80)
    print("  9. SIMULATED: RATIO SCORING + NET-BASED HEALTHY GATE")
    print("=" * 80)
    print("  If disease_net > healthy_net -> disease, else -> healthy")
    print("  Then among disease candidates, pick highest ratio")

    correct_count = 0
    for r in all_records:
        h_net = r['all_class_nets'][1]
        disease_nets = {c: n for c, n in r['all_class_nets'].items() if c != 1}
        max_disease_net = max(disease_nets.values()) if disease_nets else -999
        best_disease = max(disease_nets, key=disease_nets.get)

        if max_disease_net > h_net:
            pred = best_disease
        else:
            pred = 1

        if pred == r['true']:
            correct_count += 1
    print(f"  Net-gated: {correct_count}/{len(all_records)} = "
          f"{100*correct_count/len(all_records):.1f}%")

    # Try with ratio for disease selection
    correct_count = 0
    for r in all_records:
        h_net = r['all_class_nets'][1]
        disease_nets = {c: n for c, n in r['all_class_nets'].items() if c != 1}
        max_disease_net = max(disease_nets.values()) if disease_nets else -999

        if max_disease_net > h_net:
            disease_ratios = {c: s for c, s in r['all_class_ratios'].items()
                              if c != 1}
            pred = max(disease_ratios, key=disease_ratios.get)
        else:
            pred = 1

        if pred == r['true']:
            correct_count += 1
    print(f"  Net-gate + ratio-subtype: {correct_count}/{len(all_records)} = "
          f"{100*correct_count/len(all_records):.1f}%")

    # ===== 10. WHAT IF WE COMBINE NET + RATIO? =====
    print("\n" + "=" * 80)
    print("  10. SIMULATED: VARIOUS COMBINED SCORING APPROACHES")
    print("=" * 80)

    # Try: score = net * ratio
    for combo_name, score_fn in [
        ('net * ratio', lambda nets, ratios, c: nets[c] * ratios[c]),
        ('net * sqrt(ratio)', lambda nets, ratios, c: nets[c] * np.sqrt(ratios[c])),
        ('af_for / (1 + af_against)', lambda nets, ratios, c: None),
    ]:
        correct_count = 0
        for r in all_records:
            if combo_name.startswith('af_for'):
                scores = {}
                for c, (af_f, af_a) in r['all_class_afs'].items():
                    scores[c] = af_f / (1 + af_a)
            else:
                scores = {}
                for c in r['all_class_ratios']:
                    scores[c] = score_fn(r['all_class_nets'],
                                         r['all_class_ratios'], c)
            pred = max(scores, key=scores.get)
            if pred == r['true']:
                correct_count += 1
        print(f"  {combo_name}: {correct_count}/{len(all_records)} = "
              f"{100*correct_count/len(all_records):.1f}%")

    # ===== 11. WHAT ABOUT THE HEALTHY FP PROBLEM? =====
    print("\n" + "=" * 80)
    print("  11. HEALTHY FP ANALYSIS — WHAT MAKES THEM LOOK DISEASED?")
    print("=" * 80)

    fp_by_pred = defaultdict(list)
    for r in healthy_fps:
        fp_by_pred[r['pred']].append(r)

    for pred_cls in sorted(fp_by_pred, key=lambda c: -len(fp_by_pred[c])):
        fps = fp_by_pred[pred_cls]
        print(f"\n  Healthy -> class {pred_cls}: {len(fps)} FPs")
        # What are the disease class scores?
        for r in fps:
            d_af_for, d_af_ag = r['all_class_afs'][pred_cls]
            h_af_for, h_af_ag = r['all_class_afs'][1]
            d_ratio = r['all_class_ratios'][pred_cls]
            h_ratio = r['all_class_ratios'][1]
            d_net = r['all_class_nets'][pred_cls]
            h_net = r['all_class_nets'][1]
            print(f"    uid={r['uid']}: "
                  f"D(ratio={d_ratio:.1f},net={d_net:.3f},for={d_af_for:.3f}) "
                  f"H(ratio={h_ratio:.1f},net={h_net:.3f},for={h_af_for:.3f}) "
                  f"{'NET_WOULD_FIX' if h_net > d_net else 'NET_STILL_WRONG'}")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'output/logs/v3_log_10fold_seed13.json'
    log = load_log(path)
    analyze(log)
