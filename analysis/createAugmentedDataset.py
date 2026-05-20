"""
createAugmentedDataset.py
=========================
Generates arrhythmia_augmented.data:
  - Original 452 rows copied verbatim (including '?' missing values)
  - 500 synthetic rows appended that match the original distribution exactly:
      * Same class proportions
      * Same male/female ratio overall and within every class
      * Same feature distributions (continuous features perturbed with small
        Gaussian noise sigma=5% of std; binary/wave-shape flags copied exactly)

Method: proportional stratified perturbation.
  For each (class, gender) cell:
    - Determine n_new proportional to how many real samples are in that cell
    - Pick random real donors from that cell
    - Perturb continuous features with N(0, 0.05 * feature_std) noise
    - Copy binary wave-shape flags exactly from the donor
    - Set Sex column to cell's gender value (never interpolated)

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
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

_CANDIDATES = [
    os.path.join(_REPO_ROOT, "data", "arrhythmia.data"),
]
DATA_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("arrhythmia.data not found. Place it in the data/ directory.")

OUT_PATH = os.path.join(_REPO_ROOT, "data", "arrhythmia_augmented.data")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
HEALTHY_CLASS = 1
SEX_COL       = 1        # 0=male, 1=female
MALE_CODE     = 0
FEMALE_CODE   = 1
N_NEW         = 500
SIGMA         = 0.05     # noise = sigma * per-feature std
RANDOM_SEED   = 42

# Nominal column indices (0-indexed): Sex + 72 binary wave-shape flags.
# Wave-shape flags: Ragged/Diphasic x R/P/T per ECG channel.
# 12 channels, each contributing 6 flags, starting at cols:
#   21,33,45,57,69,81,93,105,117,129,141,153
_BINARY_STARTS = [21, 33, 45, 57, 69, 81, 93, 105, 117, 129, 141, 153]
NOMINAL_COLS = set()
for _s in _BINARY_STARTS:
    NOMINAL_COLS.update(range(_s, _s + 6))
# Sex is also nominal -- must be copied, never interpolated
NOMINAL_COLS.add(SEX_COL)
NOMINAL_LIST = sorted(NOMINAL_COLS)

CONTINUOUS_COLS = [c for c in range(279) if c not in NOMINAL_COLS]

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
print("=" * 72)
print("Distribution-Preserving Augmentation  ->  arrhythmia_augmented.data")
print("=" * 72)

with open(DATA_PATH, "r") as fh:
    original_lines = [ln.rstrip("\n") for ln in fh.readlines()]

raw    = np.genfromtxt(DATA_PATH, delimiter=",", missing_values="?",
                       filling_values=np.nan)
X_raw  = raw[:, :-1]
y_raw  = raw[:, -1].astype(int)
N, D   = X_raw.shape

# Imputed copy for sampling (original rows still written with '?')
imp    = SimpleImputer(strategy="median")
X_imp  = imp.fit_transform(X_raw)

# Per-feature std from the whole dataset (used for noise scaling)
feat_std = np.std(X_imp, axis=0, ddof=0) + 1e-8

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# STEP 1: CLASS ALLOCATION  (proportional to original class frequencies)
# ---------------------------------------------------------------------------
counts  = Counter(y_raw.tolist())
classes = sorted(counts.keys())

# Target new samples per class, proportional to original frequency
raw_alloc = {c: N_NEW * counts[c] / N for c in classes}
alloc     = {c: int(round(v)) for c, v in raw_alloc.items()}

# Fix rounding to hit exactly N_NEW
diff = N_NEW - sum(alloc.values())
for c in sorted(classes, key=lambda c: abs(raw_alloc[c] - alloc[c]), reverse=True):
    if diff == 0:
        break
    alloc[c] += 1 if diff > 0 else -1
    diff      -= 1 if diff > 0 else -1

assert sum(alloc.values()) == N_NEW

# ---------------------------------------------------------------------------
# STEP 2: GENDER ALLOCATION within each class
# ---------------------------------------------------------------------------
# For each class, split proportionally by gender
cell_alloc = {}   # {(class, gender): n_new}
class_names = {
    1:  "Normal",
    2:  "Coronary Artery Disease",    3:  "Old Anterior MI",
    4:  "Old Inferior MI",            5:  "Sinus Tachycardia",
    6:  "Sinus Bradycardia",          7:  "PVC",
    8:  "Supraventricular PC",        9:  "Left Bundle Branch Block",
    10: "Right Bundle Branch Block",  14: "LV Hypertrophy",
    15: "Atrial Fibrillation",        16: "Others",
}

print(f"\n  {'Class':>6}  {'Name':<28}  "
      f"{'Orig':>5}  {'OrigM':>5}  {'OrigF':>5}  "
      f"{'New':>5}  {'NewM':>5}  {'NewF':>5}")
print("  " + "-" * 72)

for c in classes:
    mask_c  = (y_raw == c)
    n_c     = counts[c]
    n_m     = int((X_raw[mask_c, SEX_COL] == MALE_CODE).sum())
    n_f     = int((X_raw[mask_c, SEX_COL] == FEMALE_CODE).sum())
    n_new_c = alloc[c]

    # Gender split proportional to original gender ratio in this class
    if n_c > 0:
        n_new_m = int(round(n_new_c * n_m / n_c))
        n_new_f = n_new_c - n_new_m
    else:
        n_new_m = 0
        n_new_f = 0

    # If a gender has 0 real samples, we cannot generate any for it
    if n_m == 0:
        n_new_m = 0
        n_new_f = n_new_c
    if n_f == 0:
        n_new_f = 0
        n_new_m = n_new_c

    cell_alloc[(c, MALE_CODE)]   = n_new_m
    cell_alloc[(c, FEMALE_CODE)] = n_new_f

    print(f"  {c:>6}  {class_names.get(c,''):28s}  "
          f"{n_c:>5}  {n_m:>5}  {n_f:>5}  "
          f"{n_new_c:>5}  {n_new_m:>5}  {n_new_f:>5}")

total_new_m = sum(v for (c,g), v in cell_alloc.items() if g == MALE_CODE)
total_new_f = sum(v for (c,g), v in cell_alloc.items() if g == FEMALE_CODE)
print(f"\n  Original : {N} total  |  "
      f"M={int((X_raw[:,SEX_COL]==MALE_CODE).sum())}  "
      f"F={int((X_raw[:,SEX_COL]==FEMALE_CODE).sum())}  "
      f"({int((X_raw[:,SEX_COL]==FEMALE_CODE).sum())/N*100:.1f}% F)")
print(f"  New      : {N_NEW} total  |  "
      f"M={total_new_m}  F={total_new_f}  "
      f"({total_new_f/N_NEW*100:.1f}% F)")
print(f"  Augmented: {N+N_NEW} total  |  "
      f"M={int((X_raw[:,SEX_COL]==MALE_CODE).sum())+total_new_m}  "
      f"F={int((X_raw[:,SEX_COL]==FEMALE_CODE).sum())+total_new_f}  "
      f"({(int((X_raw[:,SEX_COL]==FEMALE_CODE).sum())+total_new_f)/(N+N_NEW)*100:.1f}% F)")

# ---------------------------------------------------------------------------
# STEP 3: GENERATE SYNTHETIC SAMPLES
# For each (class, gender) cell:
#   - Pick n_new random donors from that cell (with replacement)
#   - Perturb continuous features with N(0, sigma * feat_std) noise
#   - Copy binary/wave-shape flags exactly from donor
#   - Set Sex column explicitly to the cell's gender
# ---------------------------------------------------------------------------
print(f"\n  Generating {N_NEW} synthetic rows (sigma={SIGMA}) ...")

syn_X_list = []
syn_y_list = []

for c in classes:
    for g in [MALE_CODE, FEMALE_CODE]:
        n_gen = cell_alloc.get((c, g), 0)
        if n_gen == 0:
            continue

        # Find all real rows in this (class, gender) cell
        cell_mask = (y_raw == c) & (X_imp[:, SEX_COL] == g)
        X_cell    = X_imp[cell_mask]

        if len(X_cell) == 0:
            # No real samples to draw from -- skip (this shouldn't happen
            # because we set n_gen=0 for cells with 0 real samples above)
            continue

        # Draw random donors with replacement
        donor_idx = rng.integers(0, len(X_cell), size=n_gen)
        donors    = X_cell[donor_idx]           # (n_gen, 279)

        # Perturb continuous features
        noise = rng.normal(0.0, SIGMA, size=donors.shape) * feat_std
        noise[:, NOMINAL_LIST] = 0.0            # no noise on nominal cols

        synthetic = donors + noise

        # Clip continuous values to observed range + 10% margin
        col_min = np.nanmin(X_imp, axis=0)
        col_max = np.nanmax(X_imp, axis=0)
        margin  = 0.1 * (col_max - col_min)
        synthetic = np.clip(synthetic, col_min - margin, col_max + margin)

        # Restore nominal columns from donor (noise was zeroed but clip may drift)
        synthetic[:, NOMINAL_LIST] = donors[:, NOMINAL_LIST]

        # Snap binary wave-shape flags to 0 or 1 (should already be, but ensure)
        for bc in NOMINAL_LIST:
            if bc == SEX_COL:
                synthetic[:, bc] = float(g)     # force correct gender
            else:
                synthetic[:, bc] = np.round(np.clip(synthetic[:, bc], 0, 1))

        syn_X_list.append(synthetic)
        syn_y_list.append(np.full(n_gen, c, dtype=int))

X_syn = np.vstack(syn_X_list)
y_syn = np.concatenate(syn_y_list)

assert len(y_syn) == N_NEW, f"Expected {N_NEW}, got {len(y_syn)}"

# ---------------------------------------------------------------------------
# FORMAT SYNTHETIC ROWS
# Continuous: 2 decimal places  |  Nominal: integer 0 or 1
# ---------------------------------------------------------------------------
def _fmt_row(x, label):
    parts = []
    for col in range(D):
        v = x[col]
        if col in NOMINAL_COLS:
            parts.append(str(int(round(float(v)))))
        else:
            parts.append(f"{v:.2f}")
    parts.append(str(int(label)))
    return ",".join(parts)

# ---------------------------------------------------------------------------
# WRITE OUTPUT
# ---------------------------------------------------------------------------
print(f"\n  Writing {OUT_PATH} ...")
with open(OUT_PATH, "w", newline="\n") as fh:
    for line in original_lines:
        fh.write(line + "\n")
    for i in range(N_NEW):
        fh.write(_fmt_row(X_syn[i], y_syn[i]) + "\n")

# ---------------------------------------------------------------------------
# VERIFY
# ---------------------------------------------------------------------------
verify       = np.genfromtxt(OUT_PATH, delimiter=",", missing_values="?",
                              filling_values=np.nan)
verify_y     = verify[:, -1].astype(int)
verify_X     = verify[:, :-1]
v_counts     = Counter(verify_y.tolist())
verify_syn_X = verify_X[N:]
verify_syn_y = verify_y[N:]

print(f"\n  VERIFICATION")
print(f"  {'Check':<40}  {'Result'}")
print("  " + "-" * 56)

# Row count
ok = len(verify_y) == N + N_NEW
print(f"  {'Total rows':<40}  {len(verify_y)}  ({'PASS' if ok else 'FAIL'})")

# No missing in synthetic
n_missing = int(np.isnan(verify_syn_X).sum())
print(f"  {'Missing values in synthetic rows':<40}  {n_missing}  ({'PASS' if n_missing==0 else 'FAIL'})")

# All nominal cols strictly 0 or 1
bad_nom = sum(1 for bc in NOMINAL_LIST
              for v in verify_syn_X[:, bc]
              if v not in (0.0, 1.0))
print(f"  {'Non-0/1 values in nominal cols':<40}  {bad_nom}  ({'PASS' if bad_nom==0 else 'FAIL'})")

# Unique synthetic rows
n_unique = len(set(map(tuple, np.round(verify_syn_X, 2).tolist())))
print(f"  {'Unique synthetic rows':<40}  {n_unique}/500  ({'PASS' if n_unique==500 else 'CHECK'})")

# Gender ratio
orig_pct_f  = int((X_raw[:, SEX_COL] == FEMALE_CODE).sum()) / N * 100
aug_pct_f   = int((verify_X[:, SEX_COL] == FEMALE_CODE).sum()) / (N + N_NEW) * 100
syn_pct_f   = int((verify_syn_X[:, SEX_COL] == FEMALE_CODE).sum()) / N_NEW * 100
print(f"  {'Female % original':<40}  {orig_pct_f:.1f}%")
print(f"  {'Female % synthetic':<40}  {syn_pct_f:.1f}%")
print(f"  {'Female % augmented':<40}  {aug_pct_f:.1f}%")

# Class distribution comparison
print(f"\n  {'Class':>6}  {'Original':>10}  {'Orig %':>7}  {'Augmented':>10}  {'Aug %':>7}  {'Drift':>7}")
print("  " + "-" * 56)
for c in classes:
    o_n  = counts[c]
    a_n  = v_counts[c]
    o_p  = o_n / N * 100
    a_p  = a_n / (N + N_NEW) * 100
    drift = a_p - o_p
    flag  = " <-- drift" if abs(drift) > 0.3 else ""
    print(f"  {c:>6}  {o_n:>10}  {o_p:>6.2f}%  {a_n:>10}  {a_p:>6.2f}%  {drift:>+6.2f}%{flag}")

print(f"\n  Done. File: {OUT_PATH}")
print("=" * 72)
