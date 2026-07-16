"""Run the final CDS-OVR with LOOCV and ALL disease classes (no removal)."""
import numpy as np
import time
from _common import load_data, classify_features, build_tree, stats, binary_acc, HEALTHY
from _ovr_engine import CDSConfig, train, predict

cfg = CDSConfig(
    binning='supervised',
    max_bins=6,
    corr_threshold=0.8,
    features_per_class=18,
    laplace_alpha=1.0,
    min_support=3,
    conf_support=10,
    against_scale=0.8,
    rare_classes={4, 5, 9},
    rare_min_support=2,
    rare_conf_support=5,
    rare_against_scale=0.5,
    af_mode='dual',
    fisher=True,
    scoring='ratio',
    ratio_eps=0.1,
    threshold_mode='healthy_bar',
    healthy_weight=1.05,
    suspicion_hcut=2.0,
    suspicion_offset=0.3,
    healthy_bar_cap=5.0,
    class_thresholds={2: 3.5, 3: 5.0, 4: 4.0, 5: 3.5, 6: 3.5, 9: 5.0, 10: 3.0},
    remove_classes=None,
)

data, labels = load_data(remove_classes=None)
is_bin = classify_features(data)
all_cls = sorted(set(labels))
n = data.shape[0]

print(f"Dataset: {n} patients, {len(all_cls)} classes: {all_cls}")
print(f"Class distribution:")
for c in all_cls:
    cnt = int(np.sum(labels == c))
    print(f"  Class {c:2d}: {cnt:4d} ({100*cnt/n:.1f}%)")

print(f"\nRunning LOOCV (Leave-One-Out Cross-Validation)...")
print(f"This will take a while — {n} iterations.")

t0 = time.time()
results = []
for i in range(n):
    train_idx = np.concatenate([np.arange(0, i), np.arange(i+1, n)])
    td, tl = data[train_idx], labels[train_idx]
    nodes = build_tree(td, tl, is_bin)
    train_result = train(cfg, nodes, td, tl, is_bin, all_cls)
    true_cls = int(labels[i])
    pred, _ = predict(cfg, i, data, nodes, all_cls, train_result)
    correct = pred == true_cls
    results.append((i, true_cls, int(pred), correct))
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        acc_so_far = sum(r[3] for r in results) / len(results)
        eta = elapsed / (i + 1) * (n - i - 1)
        print(f"  [{i+1}/{n}] acc so far: {100*acc_so_far:.1f}%  "
              f"elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s", flush=True)

total_time = time.time() - t0
acc, spec, sens, ba = stats(results)
bin_acc = binary_acc(results)

print(f"\n{'='*60}")
print(f"  LOOCV Results — Final CDS-OVR — ALL {len(all_cls)} Classes")
print(f"{'='*60}")
print(f"  Multiclass accuracy: {100*acc:.1f}%")
print(f"  Binary accuracy:     {100*bin_acc:.1f}%")
print(f"  Specificity:         {100*spec:.1f}%")
print(f"  Sensitivity:         {100*sens:.1f}%")
print(f"  Balanced accuracy:   {100*ba:.1f}%")
print(f"  Time: {total_time:.0f}s")
print(f"{'='*60}")

print(f"\nPer-class breakdown:")
from collections import defaultdict
by_class = defaultdict(lambda: {"total": 0, "correct": 0})
for _, true, pred, corr in results:
    by_class[true]["total"] += 1
    by_class[true]["correct"] += int(corr)

print(f"  {'Class':>6s} {'Count':>6s} {'Correct':>8s} {'Acc':>7s}")
for c in sorted(by_class.keys()):
    d = by_class[c]
    a = 100 * d["correct"] / d["total"] if d["total"] > 0 else 0
    print(f"  {c:6d} {d['total']:6d} {d['correct']:8d} {a:6.1f}%")
