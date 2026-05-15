"""
createAugmentedDataset.py
=========================
Generates arrhythmia_augmented.data:
  - Original 452 rows copied verbatim (including '?' missing values)
  - 500 synthetic rows appended using SMOTENC
  - Same comma-separated format, class label in final column

Run:
    python createAugmentedDataset.py
"""

import os
import warnings
import numpy as np
from collections import Counter
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.normpath(os.path.join(_HERE, ".."))
_WORKTREE = os.path.normpath(os.path.join(
    _HERE, "..", ".claude", "worktrees", "nifty-hoover-f39cac"
))

_CANDIDATES = [
    os.path.join(_WORKTREE, "arrhythmia.data"),
    os.path.join(_ROOT,     "arrhythmia.data"),
]
DATA_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("arrhythmia.data not found in expected locations.")

OUT_PATH = os.path.join(_HERE, "arrhythmia_augmented.data")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
HEALTHY_CLASS  = 1
N_NEW          = 500

# Nominal (categorical) column indices 0-indexed in the 279-feature space.
# Includes:
#   col 1   = Sex (0=male, 1=female) -- must be treated as categorical so
#             SMOTENC keeps it as 0 or 1, not interpolated to e.g. 0.43
#   cols 21-26, 33-38, ... = wave-shape flags (Ragged/Diphasic x R/P/T)
#   12 channels starting at: 21,33,45,57,69,81,93,105,117,129,141,153
SEX_COL = 1
_BINARY_STARTS = [21, 33, 45, 57, 69, 81, 93, 105, 117, 129, 141, 153]
BINARY_COLS = {SEX_COL}          # Sex is nominal -- include it here
for _s in _BINARY_STARTS:
    BINARY_COLS.update(range(_s, _s + 6))

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
print("=" * 70)
print("SMOTENC Augmentation  ->  arrhythmia_augmented.data")
print("=" * 70)

# Read original lines verbatim so we can write them back unchanged
with open(DATA_PATH, "r") as fh:
    original_lines = [ln.rstrip("\n") for ln in fh.readlines()]

# Numeric array for SMOTE (missing -> NaN)
raw    = np.genfromtxt(DATA_PATH, delimiter=",", missing_values="?",
                       filling_values=np.nan)
X_raw  = raw[:, :-1]          # (452, 279) features
y_raw  = raw[:, -1].astype(int)   # (452,)   labels
N, D   = X_raw.shape

counts = Counter(y_raw.tolist())
print(f"\n  Original dataset : {N} users x {D} features")
print(f"  Healthy (class 1): {counts[1]}")
print(f"  Disease classes  : {dict(sorted((c, v) for c, v in counts.items() if c != HEALTHY_CLASS))}")

# ---------------------------------------------------------------------------
# IMPUTE  (for SMOTE only -- original rows are written as-is)
# ---------------------------------------------------------------------------
imp   = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X_raw)     # (452, 279) NaN-free

# ---------------------------------------------------------------------------
# ALLOCATION: distribute N_NEW samples across disease classes only,
# weighted inversely by current class size (rarest classes get most new samples)
# ---------------------------------------------------------------------------
disease_classes = sorted(c for c in counts if c != HEALTHY_CLASS)
inv_w   = {c: 1.0 / counts[c] for c in disease_classes}
tot_w   = sum(inv_w.values())
alloc   = {c: int(round(N_NEW * inv_w[c] / tot_w)) for c in disease_classes}

# Fix integer rounding so total == N_NEW exactly
diff = N_NEW - sum(alloc.values())
for c in sorted(disease_classes, key=lambda c: counts[c]):
    if diff == 0:
        break
    alloc[c] += (1 if diff > 0 else -1)
    diff    -= (1 if diff > 0 else -1)

class_names = {
    2: "Coronary Artery Disease",   3: "Old Anterior MI",
    4: "Old Inferior MI",           5: "Sinus Tachycardia",
    6: "Sinus Bradycardia",         7: "PVC",
    8: "Supraventricular PC",       9: "Left Bundle Branch Block",
    10: "Right Bundle Branch Block", 14: "LV Hypertrophy",
    15: "Atrial Fibrillation",      16: "Others",
}

print(f"\n  New sample allocation ({N_NEW} total):")
print(f"  {'Class':>6}  {'Name':<30}  {'Current':>7}  {'+New':>5}  {'Final':>6}  Method")
print("  " + "-" * 70)
for c in sorted(disease_classes):
    n   = counts[c]
    add = alloc[c]
    if n < 3:
        method = "perturbation"
    elif n < 6:
        method = "SMOTENC k=2"
    else:
        method = "SMOTENC k=5"
    print(f"  {c:>6}  {class_names.get(c,''):30s}  {n:>7}  {add:>5}  "
          f"{n + add:>6}  {method}")
print(f"  {'1':>6}  {'Normal (not oversampled)':<30}  {counts[1]:>7}  {'0':>5}  "
      f"{counts[1]:>6}")
print(f"\n  Total: {N} + {sum(alloc.values())} = {N + sum(alloc.values())} rows")

# ---------------------------------------------------------------------------
# RUN SMOTENC  (groups by required k to handle tiny classes)
# ---------------------------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTENC
except ImportError:
    raise ImportError("Run: pip install imbalanced-learn")

binary_col_list = sorted(BINARY_COLS)
synthetic_X_all = []
synthetic_y_all = []

# Group 1: k=5  (classes with >= 6 real samples)
g5 = {c: counts[c] + alloc[c] for c in disease_classes
      if counts[c] >= 6 and alloc[c] > 0}
# Also include class 1 in sampling_strategy at its current count (no change)
# so SMOTENC doesn't try to resample it
g5[HEALTHY_CLASS] = counts[HEALTHY_CLASS]

if len(g5) > 1:  # need at least one minority class
    print("\n  Running SMOTENC k=5 for large classes ...")
    sm5 = SMOTENC(categorical_features=binary_col_list,
                  sampling_strategy={c: v for c, v in g5.items()
                                     if c != HEALTHY_CLASS},
                  k_neighbors=5, random_state=42)
    X5, y5 = sm5.fit_resample(X_imp, y_raw)
    new_mask = np.zeros(len(y5), dtype=bool)
    new_mask[N:] = True
    synthetic_X_all.append(X5[new_mask])
    synthetic_y_all.append(y5[new_mask])
    print(f"    Generated {new_mask.sum()} samples")

# Group 2: k=2  (classes with 3-5 real samples)
g2_classes = [c for c in disease_classes if 3 <= counts[c] <= 5 and alloc[c] > 0]
if g2_classes:
    # Build a subset with only these classes + the healthy class (needed as context)
    mask2 = np.isin(y_raw, g2_classes + [HEALTHY_CLASS])
    X_sub2 = X_imp[mask2]
    y_sub2 = y_raw[mask2]
    strat2 = {c: int((y_sub2 == c).sum()) + alloc[c] for c in g2_classes}
    print(f"\n  Running SMOTENC k=2 for sparse classes {g2_classes} ...")
    sm2 = SMOTENC(categorical_features=binary_col_list,
                  sampling_strategy=strat2,
                  k_neighbors=2, random_state=42)
    try:
        X2, y2 = sm2.fit_resample(X_sub2, y_sub2)
        n_orig2 = mask2.sum()
        new2_mask = np.zeros(len(y2), dtype=bool)
        new2_mask[n_orig2:] = True
        synthetic_X_all.append(X2[new2_mask])
        synthetic_y_all.append(y2[new2_mask])
        print(f"    Generated {new2_mask.sum()} samples")
    except Exception as e:
        print(f"    SMOTENC k=2 failed: {e} -- using perturbation instead")
        for c in g2_classes:
            cls_mask = (y_raw == c)
            Xc = X_imp[cls_mask].copy()
            stds = np.std(Xc, axis=0, ddof=0) + 1e-8
            rng  = np.random.default_rng(42)
            for _ in range(alloc[c]):
                base  = Xc[rng.integers(len(Xc))]
                noise = rng.normal(0, 0.05, size=D) * stds
                noise[binary_col_list] = 0.0
                synthetic_X_all.append((base + noise).reshape(1, -1))
                synthetic_y_all.append(np.array([c]))

# Group 3: perturbation for classes with < 3 samples (SMOTENC needs k+1)
g_perturb = [c for c in disease_classes if counts[c] < 3 and alloc[c] > 0]
if g_perturb:
    print(f"\n  Using perturbation for tiny classes {g_perturb} ...")
    rng = np.random.default_rng(42)
    for c in g_perturb:
        cls_mask = (y_raw == c)
        Xc   = X_imp[cls_mask].copy()
        stds = np.std(X_imp, axis=0, ddof=0) + 1e-8  # global stds for scale
        Xsyn = []
        for _ in range(alloc[c]):
            base  = Xc[rng.integers(len(Xc))]
            noise = rng.normal(0, 0.05, size=D) * stds
            noise[binary_col_list] = 0.0
            sample = base + noise
            # clip binary cols back to 0 or 1
            for bc in binary_col_list:
                sample[bc] = round(float(np.clip(base[bc], 0, 1)))
            Xsyn.append(sample)
        synthetic_X_all.append(np.array(Xsyn))
        synthetic_y_all.append(np.full(alloc[c], c, dtype=int))
        print(f"    Class {c}: generated {alloc[c]} samples via perturbation")

# Combine all synthetic samples
X_syn = np.vstack(synthetic_X_all)
y_syn = np.concatenate(synthetic_y_all).astype(int)

print(f"\n  Total synthetic samples generated: {len(y_syn)}")

# Trim or top-up to exactly N_NEW if rounding caused a small discrepancy
if len(y_syn) > N_NEW:
    X_syn = X_syn[:N_NEW]
    y_syn = y_syn[:N_NEW]
elif len(y_syn) < N_NEW:
    # Pad by duplicating last few rows with tiny noise
    shortage = N_NEW - len(y_syn)
    rng2 = np.random.default_rng(99)
    stds = np.std(X_imp, axis=0, ddof=0) + 1e-8
    extra_X, extra_y = [], []
    for i in range(shortage):
        idx   = i % len(X_syn)
        noise = rng2.normal(0, 0.02, size=D) * stds
        noise[binary_col_list] = 0.0
        extra_X.append(X_syn[idx] + noise)
        extra_y.append(int(y_syn[idx]))
    X_syn = np.vstack([X_syn, extra_X])
    y_syn = np.concatenate([y_syn, extra_y]).astype(int)

assert len(y_syn) == N_NEW, f"Expected {N_NEW} synthetic rows, got {len(y_syn)}"

# ---------------------------------------------------------------------------
# FORMAT SYNTHETIC ROWS
# Continuous values: 2 decimal places (matches original)
# Binary values: 0 or 1 as integers
# Class label: integer
# ---------------------------------------------------------------------------
def _format_synthetic_row(x_row, label):
    parts = []
    for col_idx in range(D):
        val = x_row[col_idx]
        if col_idx in BINARY_COLS:
            # snap to nearest 0 or 1
            parts.append(str(int(round(float(np.clip(val, 0, 1))))))
        else:
            # Two decimal places for continuous features
            parts.append(f"{val:.2f}")
    parts.append(str(int(label)))
    return ",".join(parts)

# ---------------------------------------------------------------------------
# WRITE OUTPUT FILE
# ---------------------------------------------------------------------------
print(f"\n  Writing {OUT_PATH} ...")

with open(OUT_PATH, "w", newline="\n") as fh:
    # Original 452 rows verbatim
    for line in original_lines:
        fh.write(line + "\n")

    # 500 synthetic rows
    for i in range(N_NEW):
        fh.write(_format_synthetic_row(X_syn[i], y_syn[i]) + "\n")

# ---------------------------------------------------------------------------
# VERIFY
# ---------------------------------------------------------------------------
verify = np.genfromtxt(OUT_PATH, delimiter=",", missing_values="?",
                       filling_values=np.nan)
verify_labels = verify[:, -1].astype(int)
v_counts      = Counter(verify_labels.tolist())

print(f"\n  VERIFICATION")
print(f"  Rows in output file : {len(verify_labels)}")
print(f"  Expected            : {N + N_NEW}")
assert len(verify_labels) == N + N_NEW, "Row count mismatch!"

print(f"\n  Class distribution in augmented dataset:")
print(f"  {'Class':>6}  {'Name':<30}  {'Original':>9}  {'Augmented':>9}  {'+Added':>7}")
print("  " + "-" * 64)
for c in sorted(v_counts):
    orig_n = counts.get(c, 0)
    aug_n  = v_counts[c]
    added  = aug_n - orig_n
    name   = "Normal" if c == HEALTHY_CLASS else class_names.get(c, "")
    print(f"  {c:>6}  {name:<30}  {orig_n:>9}  {aug_n:>9}  {added:>7}")
print(f"  {'TOTAL':>6}  {'':30}  {N:>9}  {N + N_NEW:>9}  {N_NEW:>7}")

print(f"\n  Done. File saved to:")
print(f"  {OUT_PATH}")
print("=" * 70)
