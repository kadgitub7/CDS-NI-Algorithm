"""Post-hoc analysis of LOOCV trace: find optimal thresholds and detailed metrics."""
import json
import numpy as np
from collections import defaultdict

HEALTHY = 1

with open('output/loocv_trace_ovr.json') as f:
    traces = json.load(f)

n = len(traces)
print(f'Loaded {n} traces')
print()

def evaluate_scoring(traces, score_key, threshold):
    tp = tn = fp = fn = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    wrong_disease = 0

    for t in traces:
        true = t['true_class']
        scores = {int(k): v for k, v in t[score_key].items()}
        disease_scores = {c: s for c, s in scores.items() if c != HEALTHY}
        best_d = max(disease_scores, key=disease_scores.get)

        if disease_scores[best_d] >= threshold:
            pred = best_d
        else:
            pred = HEALTHY

        per_class_total[true] += 1
        true_b = 'H' if true == HEALTHY else 'D'
        pred_b = 'H' if pred == HEALTHY else 'D'

        if true_b == 'H' and pred_b == 'H': tn += 1; per_class_correct[true] += 1
        elif true_b == 'H' and pred_b == 'D': fp += 1
        elif true_b == 'D' and pred_b == 'D':
            tp += 1
            if pred == true: per_class_correct[true] += 1
            else: wrong_disease += 1
        else: fn += 1

    binary_acc = (tp + tn) / n
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ba = np.mean([per_class_correct[c]/per_class_total[c]
                  for c in per_class_total if per_class_total[c] > 0])
    per_cls_acc = sum(per_class_correct.values()) / n
    return {
        'binary_acc': binary_acc, 'spec': spec, 'sens': sens,
        'ppv': ppv, 'npv': npv, 'ba': ba, 'per_cls_acc': per_cls_acc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'wrong_disease': wrong_disease,
        'per_class_correct': dict(per_class_correct),
        'per_class_total': dict(per_class_total)
    }


# =====================================================================
# CONTRAST SCORING threshold sweep
# =====================================================================
print('='*70)
print('CONTRAST SCORING: (af_for - af_against) / (total + eps)')
print('='*70)
print(f'{"Thresh":>7s}  {"BinAcc":>7s}  {"Spec":>7s}  {"Sens":>7s}  {"PPV":>7s}  {"NPV":>7s}  {"FP":>4s}  {"FN":>4s}  {"WD":>4s}')

best_contrast_acc = 0; best_contrast_t = 0
for t_100 in range(-20, 96):
    t = t_100 / 100.0
    r = evaluate_scoring(traces, 'class_scores', t)
    if r['binary_acc'] > best_contrast_acc:
        best_contrast_acc = r['binary_acc']; best_contrast_t = t
    if t_100 % 5 == 0 or r['binary_acc'] >= best_contrast_acc:
        print(f'{t:7.2f}  {100*r["binary_acc"]:6.1f}%  {100*r["spec"]:6.1f}%  '
              f'{100*r["sens"]:6.1f}%  {100*r["ppv"]:6.1f}%  {100*r["npv"]:6.1f}%  '
              f'{r["fp"]:4d}  {r["fn"]:4d}  {r["wrong_disease"]:4d}'
              f'{"  <-- BEST" if r["binary_acc"] >= best_contrast_acc else ""}')

print()
print(f'Best contrast threshold: {best_contrast_t:.2f} -> {100*best_contrast_acc:.1f}% binary acc')
r_best_c = evaluate_scoring(traces, 'class_scores', best_contrast_t)
print(f'  Spec={100*r_best_c["spec"]:.1f}%  Sens={100*r_best_c["sens"]:.1f}%  '
      f'PPV={100*r_best_c["ppv"]:.1f}%  NPV={100*r_best_c["npv"]:.1f}%')
print(f'  Per-class acc={100*r_best_c["per_cls_acc"]:.1f}%  BA={100*r_best_c["ba"]:.1f}%')
print()

# =====================================================================
# RATIO SCORING threshold sweep (same evidence, different scoring)
# =====================================================================
print('='*70)
print('RATIO SCORING: (af_for + eps) / (af_against + eps)')
print('  (same evidence model with dynamic support + penalty)')
print('='*70)
print(f'{"Thresh":>7s}  {"BinAcc":>7s}  {"Spec":>7s}  {"Sens":>7s}  {"PPV":>7s}  {"NPV":>7s}  {"FP":>4s}  {"FN":>4s}  {"WD":>4s}')

best_ratio_acc = 0; best_ratio_t = 0
for t_10 in range(10, 80):
    t = t_10 / 10.0
    r = evaluate_scoring(traces, 'ratio_scores', t)
    if r['binary_acc'] > best_ratio_acc:
        best_ratio_acc = r['binary_acc']; best_ratio_t = t
    if t_10 % 5 == 0 or r['binary_acc'] >= best_ratio_acc:
        print(f'{t:7.1f}  {100*r["binary_acc"]:6.1f}%  {100*r["spec"]:6.1f}%  '
              f'{100*r["sens"]:6.1f}%  {100*r["ppv"]:6.1f}%  {100*r["npv"]:6.1f}%  '
              f'{r["fp"]:4d}  {r["fn"]:4d}  {r["wrong_disease"]:4d}'
              f'{"  <-- BEST" if r["binary_acc"] >= best_ratio_acc else ""}')

print()
print(f'Best ratio threshold: {best_ratio_t:.1f} -> {100*best_ratio_acc:.1f}% binary acc')
r_best_r = evaluate_scoring(traces, 'ratio_scores', best_ratio_t)
print(f'  Spec={100*r_best_r["spec"]:.1f}%  Sens={100*r_best_r["sens"]:.1f}%  '
      f'PPV={100*r_best_r["ppv"]:.1f}%  NPV={100*r_best_r["npv"]:.1f}%')
print()

# =====================================================================
# COMPARISON WITH PREVIOUS BASELINE
# =====================================================================
print('='*70)
print('COMPARISON')
print('='*70)
print(f'Previous LOOCV (ratio, no fixes):  80.3% binary, 87.8% spec, 71.5% sens')
print(f'New contrast (optimal threshold):  {100*best_contrast_acc:.1f}% binary, '
      f'{100*r_best_c["spec"]:.1f}% spec, {100*r_best_c["sens"]:.1f}% sens')
print(f'New ratio (optimal threshold):     {100*best_ratio_acc:.1f}% binary, '
      f'{100*r_best_r["spec"]:.1f}% spec, {100*r_best_r["sens"]:.1f}% sens')
print()

if best_contrast_acc >= best_ratio_acc:
    print(f'WINNER: Contrast scoring at t={best_contrast_t:.2f}')
    best_t = best_contrast_t; score_key = 'class_scores'
    r_win = r_best_c
else:
    print(f'WINNER: Ratio scoring at t={best_ratio_t:.1f}')
    best_t = best_ratio_t; score_key = 'ratio_scores'
    r_win = r_best_r
print()

# =====================================================================
# PER-CLASS DETAIL for the winner
# =====================================================================
print('='*70)
print('PER-CLASS DETAIL (best scoring method)')
print('='*70)

per_cls = defaultdict(lambda: {'total': 0, 'correct': 0, 'detected': 0, 'wrong': defaultdict(int)})
for t_entry in traces:
    true = t_entry['true_class']
    scores = {int(k): v for k, v in t_entry[score_key].items()}
    disease_scores = {c: s for c, s in scores.items() if c != HEALTHY}
    best_d = max(disease_scores, key=disease_scores.get)
    pred = best_d if disease_scores[best_d] >= best_t else HEALTHY

    per_cls[true]['total'] += 1
    if pred == true:
        per_cls[true]['correct'] += 1
    else:
        per_cls[true]['wrong'][pred] += 1
    if true != HEALTHY and pred != HEALTHY:
        per_cls[true]['detected'] += 1

for cls in sorted(per_cls.keys()):
    c = per_cls[cls]
    lbl = 'healthy' if cls == HEALTHY else f'class {cls}'
    det_str = f'  detected: {c["detected"]}/{c["total"]}' if cls != HEALTHY else ''
    wrong_parts = sorted(c['wrong'].items(), key=lambda x: -x[1])[:3]
    wrong_str = ''
    if wrong_parts:
        wrong_str = '  misclassed as: ' + ', '.join(
            f'{"H" if k==1 else f"c{k}"}({v})' for k, v in wrong_parts)
    print(f'  {lbl:>10s}  {c["correct"]:3d}/{c["total"]:3d} = {100*c["correct"]/c["total"]:5.1f}%{det_str}{wrong_str}')
