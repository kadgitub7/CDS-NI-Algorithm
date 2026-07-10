"""Post-run error analysis for CDS master logs.

Usage:
    python analyze_logs.py output/logs/log_10fold_seed34.json
    python analyze_logs.py output/logs/log_90_10_seed48.json
    python analyze_logs.py output/logs/log_60_40_seed41.json

Reads the master JSON log and produces comprehensive error diagnostics.
"""
import sys
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HEALTHY = 1
MERGE_CLASSES = {7, 8, 11, 12}
MERGED_LABEL = 99


def load_log(path):
    with open(path) as f:
        return json.load(f)


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def subsection(title):
    print(f"\n  --- {title} ---")


def analyze(log):
    eval_type = log.get("eval_type", "unknown")
    traces = log["user_traces"]
    n = len(traces)

    correct = [t for t in traces.values() if t["correct"]]
    wrong = [t for t in traces.values() if not t["correct"]]

    print(f"\nEval: {eval_type}  |  Seed: {log.get('seed','')}  |  "
          f"N={n}  Correct={len(correct)}  Wrong={len(wrong)}  "
          f"Acc={100*len(correct)/n:.1f}%")

    by_true = defaultdict(list)
    for t in traces.values():
        by_true[t["true_class"]].append(t)

    # ================================================================
    # SECTION 1: Per-class score distributions
    # ================================================================
    section("1. PER-CLASS SCORE DISTRIBUTIONS (correct vs wrong)")

    for cls in sorted(by_true.keys()):
        entries = by_true[cls]
        cls_correct = [e for e in entries if e["correct"]]
        cls_wrong = [e for e in entries if not e["correct"]]
        cls_m = str(MERGED_LABEL if cls in MERGE_CLASSES else cls)

        lbl = "Healthy" if cls == HEALTHY else f"Class {cls}"
        print(f"\n  {lbl} (n={len(entries)}, correct={len(cls_correct)}, wrong={len(cls_wrong)})")

        def get_af(t, cls_key):
            d = t["prediction_log"]["class_af_details"].get(cls_key, {})
            return {
                "af_for": d.get("af_for", 0),
                "af_against": d.get("af_against", 0),
                "n_used": d.get("n_used", 0),
                "ratio": d.get("ratio", 0),
            }

        if cls_correct:
            afs = [get_af(e, cls_m) for e in cls_correct]
            print(f"    CORRECT: ratio={np.mean([a['ratio'] for a in afs]):.2f}"
                  f"+/-{np.std([a['ratio'] for a in afs]):.2f}"
                  f"  for={np.mean([a['af_for'] for a in afs]):.3f}"
                  f"  ag={np.mean([a['af_against'] for a in afs]):.3f}"
                  f"  n_used={np.mean([a['n_used'] for a in afs]):.1f}")

        if cls_wrong:
            afs = [get_af(e, cls_m) for e in cls_wrong]
            print(f"    WRONG:   ratio={np.mean([a['ratio'] for a in afs]):.2f}"
                  f"+/-{np.std([a['ratio'] for a in afs]):.2f}"
                  f"  for={np.mean([a['af_for'] for a in afs]):.3f}"
                  f"  ag={np.mean([a['af_against'] for a in afs]):.3f}"
                  f"  n_used={np.mean([a['n_used'] for a in afs]):.1f}")

            wrong_preds = defaultdict(int)
            for e in cls_wrong:
                wrong_preds[e["final_pred"]] += 1
            print(f"    Error targets: {dict(sorted(wrong_preds.items(), key=lambda x:-x[1]))}")

    # ================================================================
    # SECTION 2: Individual error traces
    # ================================================================
    section("2. DETAILED ERROR TRACES (per wrong user)")

    for cls in sorted(by_true.keys()):
        if cls == HEALTHY:
            continue

        entries = by_true[cls]
        cls_wrong = [e for e in entries if not e["correct"]]
        if not cls_wrong:
            continue

        cls_m = str(MERGED_LABEL if cls in MERGE_CLASSES else cls)
        lbl = f"Class {cls}"
        subsection(f"{lbl}: {len(cls_wrong)} errors")

        for e in cls_wrong:
            uid = e["uid"]
            pred = e["final_pred"]
            decision = e["prediction_log"]["decision"]
            true_af = e["prediction_log"]["class_af_details"].get(cls_m, {})

            pred_m = str(MERGED_LABEL if pred in MERGE_CLASSES else pred)
            pred_af = e["prediction_log"]["class_af_details"].get(pred_m, {})

            print(f"\n    uid={uid}  true=c{cls}  pred={'H' if pred==1 else f'c{pred}'}")
            print(f"      TRUE cls AF:  for={true_af.get('af_for',0):.4f}  "
                  f"ag={true_af.get('af_against',0):.4f}  "
                  f"ratio={true_af.get('ratio',0):.4f}  "
                  f"n_used={true_af.get('n_used',0)}")
            print(f"      PRED cls AF:  for={pred_af.get('af_for',0):.4f}  "
                  f"ag={pred_af.get('af_against',0):.4f}  "
                  f"ratio={pred_af.get('ratio',0):.4f}  "
                  f"n_used={pred_af.get('n_used',0)}")

            cand_details = decision.get("candidate_details", {})
            if cls_m in cand_details:
                cd = cand_details[cls_m]
                print(f"      Threshold: base={cd.get('threshold_base',0)}  "
                      f"final={cd.get('threshold_final',0):.2f}  "
                      f"margin={cd.get('margin',0):.4f}  "
                      f"passed={cd.get('passed',False)}")

            hbar = decision.get('healthy_bar', decision.get('healthy_bar_capped', 0))
            print(f"      h_score={decision.get('h_score',0):.4f}  "
                  f"healthy_bar={hbar:.4f}  "
                  f"suspicion={decision.get('suspicion_active',False)}")

            # Feature-level breakdown for the TRUE class
            feats = true_af.get("features", [])
            n_for = sum(1 for fe in feats if fe.get("direction") == "FOR")
            n_ag = sum(1 for fe in feats if fe.get("direction") == "AGAINST")
            n_skip = sum(1 for fe in feats if fe.get("skipped"))
            print(f"      Features: {n_for} FOR, {n_ag} AGAINST, {n_skip} skipped")

            # Show top contributing features (sorted by weighted_contrib)
            active_feats = [fe for fe in feats if fe.get("weighted_contrib") is not None]
            active_feats.sort(key=lambda x: x["weighted_contrib"], reverse=True)
            for fe in active_feats[:5]:
                print(f"        F{fe['feat']}: val={fe['user_value']}  "
                      f"bin={fe['bin_idx']}/{fe['n_bins']}  "
                      f"p={fe['p_class_bin']:.3f}  prior={fe['prior']:.3f}  "
                      f"shift={fe['shift']:+.4f}  "
                      f"fw={fe['fisher_weight']:.3f}  "
                      f"contrib={fe['weighted_contrib']:.4f}  {fe['direction']}")

    # ================================================================
    # SECTION 3: Healthy false positives deep dive
    # ================================================================
    section("3. HEALTHY FALSE POSITIVES")

    h_wrong = [e for e in by_true.get(HEALTHY, []) if not e["correct"]]
    print(f"\n  {len(h_wrong)} healthy patients misclassified as disease")

    fp_targets = defaultdict(int)
    for e in h_wrong:
        fp_targets[e["final_pred"]] += 1
    if fp_targets:
        print(f"  FP breakdown: {dict(sorted(fp_targets.items(), key=lambda x:-x[1]))}")

    for e in h_wrong[:10]:
        uid = e["uid"]
        pred = e["final_pred"]
        pred_m = str(MERGED_LABEL if pred in MERGE_CLASSES else pred)
        decision = e["prediction_log"]["decision"]
        pred_af = e["prediction_log"]["class_af_details"].get(pred_m, {})
        h_af = e["prediction_log"]["class_af_details"].get(str(HEALTHY), {})

        cand_details = decision.get("candidate_details", {})
        cd = cand_details.get(pred_m, {})

        print(f"\n    uid={uid}  pred=c{pred}")
        print(f"      Disease AF: for={pred_af.get('af_for',0):.4f}  "
              f"ag={pred_af.get('af_against',0):.4f}  "
              f"ratio={pred_af.get('ratio',0):.4f}")
        print(f"      Healthy AF: for={h_af.get('af_for',0):.4f}  "
              f"ag={h_af.get('af_against',0):.4f}  "
              f"ratio={h_af.get('ratio',0):.4f}")
        print(f"      Threshold: final={cd.get('threshold_final',0):.2f}  "
              f"margin={cd.get('margin',0):.4f}")

        # Which features drove the false positive?
        feats = pred_af.get("features", [])
        for_feats = [fe for fe in feats if fe.get("direction") == "FOR"
                     and fe.get("weighted_contrib")]
        for_feats.sort(key=lambda x: x["weighted_contrib"], reverse=True)
        if for_feats:
            print(f"      Top FOR features driving FP:")
            for fe in for_feats[:4]:
                print(f"        F{fe['feat']}: val={fe['user_value']}  "
                      f"p={fe['p_class_bin']:.3f} prior={fe['prior']:.3f}  "
                      f"shift={fe['shift']:+.4f}  contrib={fe['weighted_contrib']:.4f}")

    # ================================================================
    # SECTION 4: Feature failure analysis
    # ================================================================
    section("4. FEATURE FAILURE ANALYSIS")
    print("  (Which features give AGAINST evidence for true-class in wrong patients)")

    for cls in sorted(by_true.keys()):
        if cls == HEALTHY:
            continue
        cls_wrong = [e for e in by_true[cls] if not e["correct"]]
        if not cls_wrong:
            continue

        cls_m = str(MERGED_LABEL if cls in MERGE_CLASSES else cls)
        subsection(f"Class {cls}: {len(cls_wrong)} errors")

        feat_against_count = defaultdict(int)
        feat_for_count = defaultdict(int)
        feat_skip_count = defaultdict(int)
        feat_against_total_weight = defaultdict(float)
        feat_for_total_weight = defaultdict(float)

        for e in cls_wrong:
            feats = e["prediction_log"]["class_af_details"].get(cls_m, {}).get("features", [])
            for fe in feats:
                f = fe["feat"]
                if fe.get("skipped"):
                    feat_skip_count[f] += 1
                elif fe.get("direction") == "AGAINST":
                    feat_against_count[f] += 1
                    feat_against_total_weight[f] += fe.get("weighted_contrib", 0)
                elif fe.get("direction") == "FOR":
                    feat_for_count[f] += 1
                    feat_for_total_weight[f] += fe.get("weighted_contrib", 0)

        worst_feats = sorted(feat_against_count.items(), key=lambda x: -x[1])
        if worst_feats:
            print(f"    Features most often AGAINST for wrong patients:")
            for f, cnt in worst_feats[:8]:
                for_cnt = feat_for_count.get(f, 0)
                ag_w = feat_against_total_weight.get(f, 0)
                for_w = feat_for_total_weight.get(f, 0)
                skip = feat_skip_count.get(f, 0)
                print(f"      F{f}: AGAINST {cnt}/{len(cls_wrong)} patients "
                      f"(total_w={ag_w:.4f})  FOR {for_cnt} (total_w={for_w:.4f})  "
                      f"skip={skip}")

    # ================================================================
    # SECTION 5: Threshold optimization
    # ================================================================
    section("5. THRESHOLD OPTIMIZATION")

    for cls in sorted(by_true.keys()):
        if cls == HEALTHY:
            continue
        entries = by_true[cls]
        cls_m = str(MERGED_LABEL if cls in MERGE_CLASSES else cls)

        cls_correct = [e for e in entries if e["correct"]]
        cls_wrong_as_h = [e for e in entries if not e["correct"] and e["final_pred"] == HEALTHY]
        cls_wrong_other = [e for e in entries if not e["correct"] and e["final_pred"] != HEALTHY]

        if not entries:
            continue

        tp_ratios = []
        for e in cls_correct:
            d = e["prediction_log"]["class_af_details"].get(cls_m, {})
            tp_ratios.append(d.get("ratio", 0))

        fn_ratios = []
        for e in cls_wrong_as_h:
            d = e["prediction_log"]["class_af_details"].get(cls_m, {})
            fn_ratios.append(d.get("ratio", 0))

        # Get current threshold
        cand_details = entries[0]["prediction_log"]["decision"].get("candidate_details", {})
        cd = cand_details.get(cls_m, {})
        current_t = cd.get("threshold_base", 0)

        # FPs: healthy patients predicted as this class
        fp_ratios = []
        for e in by_true.get(HEALTHY, []):
            if not e["correct"] and e["final_pred"] == cls:
                d = e["prediction_log"]["class_af_details"].get(cls_m, {})
                fp_ratios.append(d.get("ratio", 0))

        print(f"\n  Class {cls} (n={len(entries)}, thresh={current_t}):")
        if tp_ratios:
            print(f"    TP ratios (n={len(tp_ratios)}): "
                  f"min={min(tp_ratios):.3f}  q25={np.percentile(tp_ratios,25):.3f}  "
                  f"med={np.median(tp_ratios):.3f}  mean={np.mean(tp_ratios):.3f}")
        if fn_ratios:
            print(f"    FN-as-H ratios (n={len(fn_ratios)}): "
                  f"max={max(fn_ratios):.3f}  mean={np.mean(fn_ratios):.3f}")
            for e in cls_wrong_as_h:
                d = e["prediction_log"]["class_af_details"].get(cls_m, {})
                dec = e["prediction_log"]["decision"]["candidate_details"].get(cls_m, {})
                print(f"      uid={e['uid']}: ratio={d.get('ratio',0):.3f}  "
                      f"thresh_final={dec.get('threshold_final',0):.3f}  "
                      f"gap={dec.get('margin',0):.3f}")
        if fp_ratios:
            print(f"    FP ratios (n={len(fp_ratios)}): "
                  f"min={min(fp_ratios):.3f}  max={max(fp_ratios):.3f}  "
                  f"mean={np.mean(fp_ratios):.3f}")

        # Find optimal threshold
        all_items = ([(r, 'TP') for r in tp_ratios] +
                     [(r, 'FN') for r in fn_ratios] +
                     [(r, 'FP') for r in fp_ratios])
        if len(all_items) > 1:
            all_items.sort()
            best_score = 0
            best_t = current_t
            for r, _ in all_items:
                tp_at = sum(1 for s, l in all_items if s >= r and l == 'TP')
                tn_at = sum(1 for s, l in all_items if s < r and l in ('FN', 'FP'))
                score = (tp_at + tn_at) / len(all_items)
                if score >= best_score:
                    best_score = score
                    best_t = r
            print(f"    Optimal threshold: {best_t:.3f} (acc={100*best_score:.1f}% "
                  f"vs current {current_t})")

    # ================================================================
    # SECTION 6: Confusion matrix
    # ================================================================
    section("6. CONFUSION MATRIX")

    confusion = defaultdict(lambda: defaultdict(int))
    for t in traces.values():
        confusion[t["true_class"]][t["final_pred"]] += 1

    all_cls = sorted(set(t["true_class"] for t in traces.values()) |
                     set(t["final_pred"] for t in traces.values()))

    header = f"{'True':>8s}"
    for c in all_cls:
        lbl = 'H' if c == HEALTHY else f'c{c}'
        header += f" {lbl:>4s}"
    print(f"\n  {header}  |   n   acc")

    for tc in all_cls:
        lbl = 'H' if tc == HEALTHY else f'c{tc}'
        row = f"  {lbl:>8s}"
        total = sum(confusion[tc].values())
        for pc in all_cls:
            cnt = confusion[tc][pc]
            if cnt > 0:
                row += f" {cnt:4d}"
            else:
                row += f"    ."
        acc = confusion[tc][tc] / total * 100 if total > 0 else 0
        row += f"  |{total:4d} {acc:5.1f}%"
        print(row)

    # ================================================================
    # SECTION 7: Cross-class interference
    # ================================================================
    section("7. CROSS-CLASS INTERFERENCE")
    print("  (When wrong, which other classes scored higher than true class?)")

    for cls in sorted(by_true.keys()):
        if cls == HEALTHY:
            continue
        cls_wrong = [e for e in by_true[cls] if not e["correct"]]
        if not cls_wrong:
            continue

        cls_m = str(MERGED_LABEL if cls in MERGE_CLASSES else cls)
        interferer_count = defaultdict(int)
        interferer_margin = defaultdict(list)

        for e in cls_wrong:
            all_scores = e["prediction_log"]["decision"].get("all_scores", {})
            true_score = all_scores.get(cls_m, 0)
            for k, v in all_scores.items():
                if k != cls_m and k != str(HEALTHY) and v > true_score:
                    interferer_count[k] += 1
                    interferer_margin[k].append(v - true_score)

        if interferer_count:
            print(f"\n  Class {cls} ({len(cls_wrong)} errors):")
            for k, cnt in sorted(interferer_count.items(), key=lambda x: -x[1]):
                avg_margin = np.mean(interferer_margin[k])
                print(f"    c{k} outscored true class in {cnt}/{len(cls_wrong)} cases "
                      f"(avg margin={avg_margin:.3f})")

    # ================================================================
    # SECTION 8: Feature usage statistics
    # ================================================================
    section("8. FEATURE USAGE ACROSS ALL USERS")

    feat_for_total = defaultdict(float)
    feat_against_total = defaultdict(float)
    feat_use_count = defaultdict(int)

    for t in traces.values():
        for cls_key, cls_log in t["prediction_log"]["class_af_details"].items():
            for fe in cls_log.get("features", []):
                if fe.get("direction") == "FOR":
                    feat_for_total[fe["feat"]] += fe.get("weighted_contrib", 0)
                    feat_use_count[fe["feat"]] += 1
                elif fe.get("direction") == "AGAINST":
                    feat_against_total[fe["feat"]] += fe.get("weighted_contrib", 0)
                    feat_use_count[fe["feat"]] += 1

    # Most impactful features
    feat_net = {}
    for f in feat_use_count:
        feat_net[f] = feat_for_total.get(f, 0) - feat_against_total.get(f, 0)

    print("\n  Top 15 features by total FOR weight:")
    for f, w in sorted(feat_for_total.items(), key=lambda x: -x[1])[:15]:
        ag = feat_against_total.get(f, 0)
        print(f"    F{f}: FOR={w:.3f}  AG={ag:.3f}  net={w-ag:+.3f}  "
              f"used={feat_use_count[f]}x")

    print("\n  Top 15 features by total AGAINST weight:")
    for f, w in sorted(feat_against_total.items(), key=lambda x: -x[1])[:15]:
        fo = feat_for_total.get(f, 0)
        print(f"    F{f}: AG={w:.3f}  FOR={fo:.3f}  net={fo-w:+.3f}  "
              f"used={feat_use_count[f]}x")

    # ================================================================
    # SECTION 9: Evidence balance per class
    # ================================================================
    section("9. EVIDENCE BALANCE (FOR vs AGAINST breakdown)")

    for cls in sorted(by_true.keys()):
        if cls == HEALTHY:
            continue
        entries = by_true[cls]
        cls_m = str(MERGED_LABEL if cls in MERGE_CLASSES else cls)
        cls_correct = [e for e in entries if e["correct"]]
        cls_wrong = [e for e in entries if not e["correct"]]

        print(f"\n  Class {cls}:")

        for label, subset in [("Correct", cls_correct), ("Wrong", cls_wrong)]:
            if not subset:
                continue
            af_for_vals = []
            af_ag_vals = []
            n_for_feats = []
            n_ag_feats = []
            n_skip_feats = []
            for e in subset:
                d = e["prediction_log"]["class_af_details"].get(cls_m, {})
                af_for_vals.append(d.get("af_for", 0))
                af_ag_vals.append(d.get("af_against", 0))
                feats = d.get("features", [])
                n_for_feats.append(sum(1 for fe in feats if fe.get("direction") == "FOR"))
                n_ag_feats.append(sum(1 for fe in feats if fe.get("direction") == "AGAINST"))
                n_skip_feats.append(sum(1 for fe in feats if fe.get("skipped")))

            net = [f - a for f, a in zip(af_for_vals, af_ag_vals)]
            print(f"    {label:>7s} (n={len(subset)}): "
                  f"af_for={np.mean(af_for_vals):.3f}  "
                  f"af_ag={np.mean(af_ag_vals):.3f}  "
                  f"net={np.mean(net):.3f}  "
                  f"n_for_feats={np.mean(n_for_feats):.1f}  "
                  f"n_ag_feats={np.mean(n_ag_feats):.1f}  "
                  f"n_skip={np.mean(n_skip_feats):.1f}")

    # ================================================================
    # SECTION 10: Training model summary
    # ================================================================
    section("10. TRAINING MODEL SUMMARY")

    if "training" in log:
        training = log["training"]
    elif "folds" in log:
        training = list(log["folds"].values())[0].get("training", {})
    else:
        training = {}

    if training:
        print(f"\n  Constants: {json.dumps(training.get('constants', {}), indent=4)}")

        thresholds = training.get("thresholds", {})
        print(f"\n  Thresholds: {thresholds}")

        tree = training.get("tree", [])
        print(f"\n  Tree: {len(tree)} nodes")
        for nd in tree:
            print(f"    {nd['nid']}: lvl={nd['lvl']}  n={nd['n_users']}  "
                  f"hdist={nd['hdist']}  feat={nd.get('branch_feat')}")

        retained = training.get("retained_features", {})
        for cls_key in sorted(retained.keys(), key=lambda x: int(x)):
            r = retained[cls_key]
            feats = r["features"]
            if feats:
                feat_ids = [f["feat"] for f in feats]
                scores = [f["score"] for f in feats]
                fishers = [f["fisher"] for f in feats]
                print(f"\n  Class {cls_key} ({r['n_features']} features):")
                print(f"    Features: {feat_ids}")
                print(f"    Scores:   [{', '.join(f'{s:.3f}' for s in scores)}]")
                print(f"    Fishers:  [{', '.join(f'{f:.3f}' for f in fishers)}]")

    print(f"\n{'='*80}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <log_file.json>")
        print("Example: python analyze_logs.py output/logs/log_10fold_seed34.json")
        sys.exit(1)

    log_path = sys.argv[1]
    print(f"Loading {log_path}...")
    log = load_log(log_path)
    analyze(log)
