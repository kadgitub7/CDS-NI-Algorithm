"""
dataAugmentation.py
===================
Simulate hypothetical CDS Focus Level 3 using data augmentation.

CONTEXT
-------
In the forced-sex CDS system (Algorithm1_forcedBranch.py), users are routed
to a sex-specific tree at Level 1.  Focus Level 2 then branches on all valid
ECG features within that tree.  A hypothetical Focus Level 3 would branch
again within Level 2 nodes -- but the female sub-population is too small
(249 diseased females, u_min=200) to support further branching without more
data.

The female error gap observed in the CDS algorithm has two root causes:
  A. Empty action tables for Classes 3/7/8 (all-male diseases, 0 female
     training examples -- no augmentation can fix this without domain input).
  B. Insufficient female data in Level 2 nodes for a stable Level 3 split.

This script isolates cause B by training a RandomForest classifier
(as a proxy for a Level-3 CDS node) on the female sub-population using
five augmentation strategies, then measuring whether the male/female
accuracy gap closes.

STRATEGIES
----------
  baseline         -- no augmentation
  random_oversample-- duplicate minority-class female samples (with replacement)
  perturbation     -- add small Gaussian noise to continuous features of
                      female samples (sigma = 5% of per-feature std)
  smotenc          -- SMOTE for mixed nominal/continuous features (Chawla 2002)
  cross_gender     -- synthesize female-like samples by perturbing male samples
                      and forcing Sex = female
  combined         -- random_oversample + perturbation

OUTPUTS
-------
  augmentation_results.csv   -- per-fold, per-strategy accuracy / gap
  augmentation_summary.csv   -- mean +/- std across folds per strategy

Run:
    python dataAugmentation.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

_DATA_CANDIDATES = [
    os.path.join(_REPO_ROOT, "data", "arrhythmia.data"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), None)
if DATA_PATH is None:
    raise FileNotFoundError("arrhythmia.data not found. Place it in the data/ directory.")

OUT_DIR = os.path.join(_REPO_ROOT, "output", "csv")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SEX_COL        = 1
MALE_CODE      = 0
FEMALE_CODE    = 1
HEALTHY_CLASS  = 1
N_FOLDS        = 5
RANDOM_SEED    = 42
PERTURB_SIGMA  = 0.05   # noise = sigma * per-feature std
PERTURB_COPIES = 3      # synthetic copies per original female sample
N_ESTIMATORS   = 200    # RandomForest trees

# Binary wave-shape flag columns (Ragged/Diphasic x R/P/T per 12 ECG channels)
_BINARY_STARTS = [21, 33, 45, 57, 69, 81, 93, 105, 117, 129, 141, 153]
BINARY_COLS = set()
for _s in _BINARY_STARTS:
    BINARY_COLS.update(range(_s, _s + 6))
# Sex is also binary but treated as the routing variable -- excluded from
# within-tree augmentation (all female samples have Sex=1 already)

CONTINUOUS_COLS = [c for c in range(279) if c not in BINARY_COLS and c != SEX_COL]

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
print("=" * 72)
print("CDS Focus Level 3 Simulation -- Data Augmentation Study")
print("=" * 72)
print(f"Data: {DATA_PATH}")

raw   = np.genfromtxt(DATA_PATH, delimiter=",", missing_values="?",
                      filling_values=np.nan)
X_raw = raw[:, :-1]          # (452, 279)
y_raw = raw[:, -1].astype(int)
N, D  = X_raw.shape

# Simple median imputation for classifier training
from sklearn.impute import SimpleImputer
imp   = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X_raw)

# Binary label: healthy (1) vs diseased (2-16)
y_bin = (y_raw != HEALTHY_CLASS).astype(int)   # 0=healthy, 1=diseased

sex    = X_imp[:, SEX_COL]
male_m = sex == MALE_CODE
fem_m  = sex == FEMALE_CODE

print(f"\nDataset: {N} users  |  "
      f"Male={int(male_m.sum())}  Female={int(fem_m.sum())}")
print(f"Class distribution: {dict(Counter(y_raw.tolist()))}")
print(f"Female diseased by class: "
      f"{dict(Counter(y_raw[fem_m & (y_raw != HEALTHY_CLASS)].tolist()))}")
print(f"Male diseased by class:   "
      f"{dict(Counter(y_raw[male_m & (y_raw != HEALTHY_CLASS)].tolist()))}")

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _gender_accs(y_true, y_pred, sex_arr):
    """Return (male_acc, female_acc, gap = female_acc - male_acc)."""
    m_mask = sex_arr == MALE_CODE
    f_mask = sex_arr == FEMALE_CODE
    m_acc  = float((y_pred[m_mask] == y_true[m_mask]).mean()) if m_mask.any() else float("nan")
    f_acc  = float((y_pred[f_mask] == y_true[f_mask]).mean()) if f_mask.any() else float("nan")
    gap    = f_acc - m_acc
    return m_acc, f_acc, gap


def _per_feature_std(X):
    """Column stds with ddof=0 (safe for n=1 samples)."""
    return np.std(X, axis=0, ddof=0) + 1e-8


# ---------------------------------------------------------------------------
# AUGMENTATION STRATEGIES
# ---------------------------------------------------------------------------

def augment_none(X_tr, y_tr, sex_tr, rng):
    """No augmentation -- return training data unchanged."""
    return X_tr.copy(), y_tr.copy(), sex_tr.copy()


def augment_random_oversample(X_tr, y_tr, sex_tr, rng):
    """
    Duplicate female minority-class samples (with replacement) until each
    female disease class has the same count as the female majority class.
    Only female samples are duplicated; male data is unchanged.
    """
    f_mask = sex_tr == FEMALE_CODE
    X_f, y_f = X_tr[f_mask], y_tr[f_mask]
    X_m, y_m = X_tr[~f_mask], y_tr[~f_mask]
    sex_f     = sex_tr[f_mask]
    sex_m     = sex_tr[~f_mask]

    class_counts = Counter(y_f.tolist())
    if len(class_counts) == 0:
        return X_tr.copy(), y_tr.copy(), sex_tr.copy()

    max_count = max(class_counts.values())

    X_syn_list, y_syn_list, s_syn_list = [], [], []
    for cls, cnt in class_counts.items():
        n_needed = max_count - cnt
        if n_needed <= 0:
            continue
        idx_cls = np.where(y_f == cls)[0]
        chosen  = rng.choice(idx_cls, size=n_needed, replace=True)
        X_syn_list.append(X_f[chosen])
        y_syn_list.append(np.full(n_needed, cls, dtype=int))
        s_syn_list.append(np.full(n_needed, FEMALE_CODE))

    if X_syn_list:
        X_syn  = np.vstack(X_syn_list)
        y_syn  = np.concatenate(y_syn_list)
        s_syn  = np.concatenate(s_syn_list)
        X_aug  = np.vstack([X_m, X_f, X_syn])
        y_aug  = np.concatenate([y_m, y_f, y_syn])
        s_aug  = np.concatenate([sex_m, sex_f, s_syn])
    else:
        X_aug, y_aug, s_aug = X_tr.copy(), y_tr.copy(), sex_tr.copy()

    return X_aug, y_aug, s_aug


def augment_perturbation(X_tr, y_tr, sex_tr, rng, n_copies=PERTURB_COPIES):
    """
    Generate n_copies synthetic female samples per real female sample by
    adding N(0, sigma * feature_std) noise to continuous features only.
    Binary wave-shape flags are copied exactly from the donor.
    """
    f_mask = sex_tr == FEMALE_CODE
    X_f, y_f = X_tr[f_mask], y_tr[f_mask]
    sex_f     = sex_tr[f_mask]
    stds      = _per_feature_std(X_f)

    X_syn_list, y_syn_list, s_syn_list = [], [], []

    for _ in range(n_copies):
        noise = rng.normal(0.0, PERTURB_SIGMA, size=X_f.shape) * stds
        # Zero noise on binary cols and Sex col
        for bc in BINARY_COLS:
            noise[:, bc] = 0.0
        noise[:, SEX_COL] = 0.0

        X_syn = X_f + noise
        # Snap binary cols back to 0/1 (noise was zeroed, but be safe)
        for bc in BINARY_COLS:
            X_syn[:, bc] = np.clip(np.round(X_f[:, bc]), 0, 1)
        # Keep Sex = female
        X_syn[:, SEX_COL] = FEMALE_CODE

        X_syn_list.append(X_syn)
        y_syn_list.append(y_f.copy())
        s_syn_list.append(sex_f.copy())

    X_aug  = np.vstack([X_tr] + X_syn_list)
    y_aug  = np.concatenate([y_tr] + y_syn_list)
    s_aug  = np.concatenate([sex_tr] + s_syn_list)
    return X_aug, y_aug, s_aug


def augment_smotenc(X_tr, y_tr, sex_tr, rng):
    """
    SMOTE-NC (Chawla 2002) on the female sub-population.
    Handles mixed nominal (binary wave-shape flags) and continuous features.
    Sex is excluded from SMOTENC because it is constant (all female).
    """
    try:
        from imblearn.over_sampling import SMOTENC
    except ImportError:
        print("  [smotenc] imbalanced-learn not installed -- skipping.")
        return X_tr.copy(), y_tr.copy(), sex_tr.copy()

    f_mask = sex_tr == FEMALE_CODE
    X_f, y_f = X_tr[f_mask], y_tr[f_mask]
    sex_f     = sex_tr[f_mask]

    # Need at least 2 samples per class for SMOTENC
    cls_counts = Counter(y_f.tolist())
    valid_cls  = [c for c, cnt in cls_counts.items() if cnt >= 2]
    if len(valid_cls) < 2:
        return X_tr.copy(), y_tr.copy(), sex_tr.copy()

    # Restrict to classes with >=2 samples
    keep_mask = np.isin(y_f, valid_cls)
    X_f_v, y_f_v = X_f[keep_mask], y_f[keep_mask]

    # Build categorical feature indices for SMOTENC (relative to X_f, Sex excluded)
    # Columns are 0-indexed into the 279-feature array; Sex col is present but
    # constant = 1 so we list binary cols only.
    cat_cols = sorted(BINARY_COLS)   # Sex excluded -- constant within female group

    # SMOTENC requires at least 2 samples in minority class
    try:
        sm = SMOTENC(
            categorical_features = cat_cols,
            random_state         = rng.integers(0, 2**31),
            k_neighbors          = min(5, min(Counter(y_f_v.tolist()).values()) - 1),
        )
        X_res, y_res = sm.fit_resample(X_f_v, y_f_v)
    except Exception as e:
        print(f"  [smotenc] failed: {e} -- falling back to baseline.")
        return X_tr.copy(), y_tr.copy(), sex_tr.copy()

    n_new     = len(y_res) - len(y_f_v)
    X_new     = X_res[len(X_f_v):]
    y_new     = y_res[len(y_f_v):]
    s_new     = np.full(n_new, FEMALE_CODE)

    # Force Sex = female (SMOTENC might interpolate it)
    X_new[:, SEX_COL] = FEMALE_CODE
    # Snap binary cols to 0/1
    for bc in BINARY_COLS:
        X_new[:, bc] = np.clip(np.round(X_new[:, bc]), 0, 1)

    X_aug = np.vstack([X_tr, X_new])
    y_aug = np.concatenate([y_tr, y_new])
    s_aug = np.concatenate([sex_tr, s_new])
    return X_aug, y_aug, s_aug


def augment_cross_gender(X_tr, y_tr, sex_tr, rng, verbose=False):
    """
    For each disease class, generate female-like synthetic samples by
    perturbing male samples from the same class and relabelling them female.

    This is the only strategy that can create female samples for all-male
    disease classes (3, 7, 8). However, those samples are not biologically
    grounded -- they test whether having ANY female training data for those
    classes reduces the error gap.
    """
    f_mask = sex_tr == FEMALE_CODE
    m_mask = sex_tr == MALE_CODE

    X_aug  = X_tr.copy()
    y_aug  = y_tr.copy()
    s_aug  = sex_tr.copy()

    stds = _per_feature_std(X_tr)

    cls_in_females = set(y_tr[f_mask].tolist())
    cls_in_males   = set(y_tr[m_mask].tolist())
    # Only synthesize for disease classes where females have <= males
    target_classes = [
        c for c in cls_in_males
        if c != HEALTHY_CLASS and
        sum(1 for i in range(len(y_tr)) if y_tr[i] == c and f_mask[i]) <
        sum(1 for i in range(len(y_tr)) if y_tr[i] == c and m_mask[i])
    ]

    X_syn_list, y_syn_list, s_syn_list = [], [], []

    for cls in target_classes:
        m_cls_idx  = np.where((y_tr == cls) & m_mask)[0]
        f_cls_count = int((y_tr[f_mask] == cls).sum())
        m_cls_count = len(m_cls_idx)
        n_needed    = m_cls_count - f_cls_count

        if n_needed <= 0 or len(m_cls_idx) == 0:
            continue

        donors = m_cls_idx[rng.integers(0, len(m_cls_idx), size=n_needed)]
        X_d    = X_tr[donors].copy()

        # Use ddof=0 to handle single-sample donor groups safely
        donor_stds = np.std(X_d, axis=0, ddof=0) + 1e-8
        noise      = rng.normal(0.0, PERTURB_SIGMA, size=X_d.shape) * donor_stds
        for bc in BINARY_COLS:
            noise[:, bc] = 0.0
        noise[:, SEX_COL] = 0.0

        X_syn = X_d + noise
        for bc in BINARY_COLS:
            X_syn[:, bc] = np.clip(np.round(X_d[:, bc]), 0, 1)
        X_syn[:, SEX_COL] = FEMALE_CODE   # relabel as female

        X_syn_list.append(X_syn)
        y_syn_list.append(np.full(n_needed, cls, dtype=int))
        s_syn_list.append(np.full(n_needed, FEMALE_CODE))

        if verbose:
            print(f"    class {cls}: donated {n_needed} male->female samples")

    if X_syn_list:
        X_aug = np.vstack([X_aug] + X_syn_list)
        y_aug = np.concatenate([y_aug] + y_syn_list)
        s_aug = np.concatenate([s_aug] + s_syn_list)

    return X_aug, y_aug, s_aug


def augment_combined(X_tr, y_tr, sex_tr, rng):
    """random_oversample then perturbation -- stacks both effects."""
    X1, y1, s1 = augment_random_oversample(X_tr, y_tr, sex_tr, rng)
    X2, y2, s2 = augment_perturbation(X1, y1, s1, rng, n_copies=1)
    return X2, y2, s2


# ---------------------------------------------------------------------------
# 5-FOLD CROSS VALIDATION
# ---------------------------------------------------------------------------
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

STRATEGIES = {
    "baseline"         : augment_none,
    "random_oversample": augment_random_oversample,
    "perturbation"     : augment_perturbation,
    "smotenc"          : augment_smotenc,
    "cross_gender"     : augment_cross_gender,
    "combined"         : augment_combined,
}

rng = np.random.default_rng(RANDOM_SEED)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                      random_state=int(rng.integers(0, 2**31)))

print(f"\nRunning {N_FOLDS}-fold CV for {len(STRATEGIES)} strategies ...")
print(f"Binary task: healthy (class 1) vs diseased (classes 2-16)")
print("-" * 72)

all_results = []

# Stratify on binary label (healthy vs diseased) to keep fold sizes stable
for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_imp, y_bin)):
    X_tr_full, X_te = X_imp[tr_idx], X_imp[te_idx]
    y_tr_full, y_te = y_bin[tr_idx], y_bin[te_idx]
    s_tr_full, s_te = sex[tr_idx],   sex[te_idx]

    # Per-fold std for perturbation noise (computed on training set only)
    fold_stds = _per_feature_std(X_tr_full)

    print(f"\nFold {fold_idx + 1}/{N_FOLDS}  "
          f"(train={len(tr_idx)}, test={len(te_idx)}, "
          f"female_test={int((s_te == FEMALE_CODE).sum())})")

    for strat_name, strat_fn in STRATEGIES.items():
        # Augment training data
        X_aug, y_aug, s_aug = strat_fn(X_tr_full, y_tr_full, s_tr_full, rng)

        # Train RandomForest
        clf = RandomForestClassifier(
            n_estimators = N_ESTIMATORS,
            random_state = int(rng.integers(0, 2**31)),
            n_jobs       = -1,
        )
        clf.fit(X_aug, y_aug)

        # Predict on held-out test fold (unaugmented)
        y_pred = clf.predict(X_te)

        m_acc, f_acc, gap = _gender_accs(y_te, y_pred, s_te)
        overall_acc = float((y_pred == y_te).mean())

        all_results.append({
            "fold"       : fold_idx + 1,
            "strategy"   : strat_name,
            "n_train"    : len(y_aug),
            "n_train_f"  : int((s_aug == FEMALE_CODE).sum()),
            "overall_acc": overall_acc,
            "male_acc"   : m_acc,
            "female_acc" : f_acc,
            "gap_f_minus_m": gap,
        })

        print(f"  {strat_name:<20} | overall={overall_acc:.3f} | "
              f"male={m_acc:.3f} | female={f_acc:.3f} | "
              f"gap(F-M)={gap:+.3f}")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(all_results)

summary_rows = []
for strat in STRATEGIES:
    sub = results_df[results_df["strategy"] == strat]
    summary_rows.append({
        "strategy"          : strat,
        "overall_acc_mean"  : sub["overall_acc"].mean(),
        "overall_acc_std"   : sub["overall_acc"].std(ddof=0),
        "male_acc_mean"     : sub["male_acc"].mean(),
        "male_acc_std"      : sub["male_acc"].std(ddof=0),
        "female_acc_mean"   : sub["female_acc"].mean(),
        "female_acc_std"    : sub["female_acc"].std(ddof=0),
        "gap_mean"          : sub["gap_f_minus_m"].mean(),
        "gap_std"           : sub["gap_f_minus_m"].std(ddof=0),
    })

summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 72)
print("SUMMARY  (mean +/- std across folds)")
print("=" * 72)
print(f"{'Strategy':<22} {'Overall':>8} {'Male':>8} {'Female':>8} {'Gap(F-M)':>10}")
print("-" * 60)
for _, row in summary_df.iterrows():
    print(f"{row['strategy']:<22} "
          f"{row['overall_acc_mean']:>6.3f}+-{row['overall_acc_std']:.3f}  "
          f"{row['male_acc_mean']:>6.3f}+-{row['male_acc_std']:.3f}  "
          f"{row['female_acc_mean']:>6.3f}+-{row['female_acc_std']:.3f}  "
          f"{row['gap_mean']:>+8.3f}+-{row['gap_std']:.3f}")

# ---------------------------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------------------------
baseline_gap = summary_df.loc[summary_df["strategy"] == "baseline", "gap_mean"].values[0]
print("\n" + "=" * 72)
print("INTERPRETATION")
print("=" * 72)
print(f"Baseline gender gap (F - M accuracy): {baseline_gap:+.3f}")
print()
for _, row in summary_df.iterrows():
    if row["strategy"] == "baseline":
        continue
    delta = row["gap_mean"] - baseline_gap
    direction = "closes" if delta > 0 else "widens"
    print(f"  {row['strategy']:<22}: gap {direction} by {abs(delta):.3f} "
          f"(gap = {row['gap_mean']:+.3f})")

print()
print("Gap > 0  : females more accurate than males (gap favours females)")
print("Gap < 0  : males more accurate than females (gap favours males)")
print()
print("NOTE: A negative baseline gap at the RF level means the CDS female")
print("error gap is NOT a raw classification difficulty -- it is specific")
print("to the CDS algorithm's empty action tables for Classes 3/7/8")
print("(all-male disease classes with zero female training examples).")
print("Augmentation cannot fully fix cause A (no female donors for those")
print("classes); it can only address cause B (sparse Level-2 statistics).")

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
results_path = os.path.join(OUT_DIR, "augmentation_results.csv")
summary_path = os.path.join(OUT_DIR, "augmentation_summary.csv")
results_df.to_csv(results_path, index=False)
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved: {results_path}")
print(f"Saved: {summary_path}")
print("=" * 72)
