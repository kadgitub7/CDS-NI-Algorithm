"""Preprocessing utilities for CDS variants.

Strategies:
1. Outlier clipping (1st-99th percentile from training, non-binary only)
   - Prevents outliers from wasting bin budget in equal-width binning
2. Test-time imputation (fill test NaN with training median/mode)
   - Recovers evidence votes without polluting training statistics
3. Minority class oversampling (simple duplication)
"""
import numpy as np


def clip_outliers(data, train_mask, is_bin, clip_lo=1, clip_hi=99):
    """Clip non-binary features to training percentile bounds.

    Returns a preprocessed COPY. NaN values pass through unchanged.
    """
    pp = data.copy()
    n_feat = data.shape[1]
    train_data = data[train_mask]

    for f in range(n_feat):
        if is_bin[f]:
            continue
        train_col = train_data[:, f]
        valid = train_col[~np.isnan(train_col)]
        if len(valid) < 5:
            continue
        p_lo, p_hi = np.percentile(valid, [clip_lo, clip_hi])
        if p_lo < p_hi:
            pp[:, f] = np.clip(pp[:, f], p_lo, p_hi)

    return pp


def compute_fill_values(train_data, is_bin):
    """Compute per-feature fill values from training data.

    Binary features: mode (rounded median). Continuous: median.
    """
    n_feat = train_data.shape[1]
    medians = np.nanmedian(train_data, axis=0)
    fill = np.zeros(n_feat)
    for f in range(n_feat):
        if np.isnan(medians[f]):
            fill[f] = 0.0
        elif is_bin[f]:
            fill[f] = round(medians[f])
        else:
            fill[f] = medians[f]
    return fill


def impute_test_rows(pp_data, test_indices, fill_values):
    """Impute NaN only in test rows (training rows keep NaN for clean statistics)."""
    for uid in test_indices:
        row = pp_data[uid]
        nan_mask = np.isnan(row)
        if nan_mask.any():
            pp_data[uid, nan_mask] = fill_values[nan_mask]


def impute_all(data, train_mask, is_bin):
    """Impute all NaN values using training medians. Returns a copy."""
    pp = data.copy()
    fill = compute_fill_values(data[train_mask], is_bin)
    for f in range(data.shape[1]):
        nan_mask = np.isnan(pp[:, f])
        if nan_mask.any():
            pp[nan_mask, f] = fill[f]
    return pp


def oversample_minority(td, tl, min_count=10):
    """Duplicate minority class samples so each class has at least min_count."""
    classes, counts = np.unique(tl, return_counts=True)
    extra_data = []
    extra_labels = []

    for cls, cnt in zip(classes, counts):
        if cnt >= min_count:
            continue
        cls_idx = np.where(tl == cls)[0]
        n_needed = min_count - cnt
        repeats = np.tile(cls_idx, (n_needed // cnt) + 1)[:n_needed]
        extra_data.append(td[repeats])
        extra_labels.append(np.full(n_needed, cls, dtype=tl.dtype))

    if extra_data:
        return np.vstack([td] + extra_data), np.concatenate([tl] + extra_labels)
    return td, tl
