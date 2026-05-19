"""
augmentation_strategies.py
==========================
Importable augmentation functions for the CDS LOOCV pipeline.

These mirror the strategies in genderPlottingAlgorithms/dataAugmentation.py
but are structured as pure functions (no top-level script execution) so they
can be imported by Algorithm4.run_loocv without side effects.

All functions have the signature:
    (X_train, y_train, rng) -> (X_augmented, y_augmented)

where X includes the sex column (col 1) and y is the full class label (1-16).
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Tuple

import numpy as np

SEX_COL = 1
MALE_CODE = 0
FEMALE_CODE = 1
HEALTHY_CLASS = 1
PERTURB_SIGMA = 0.05
PERTURB_COPIES = 3

_BINARY_STARTS = [21, 33, 45, 57, 69, 81, 93, 105, 117, 129, 141, 153]
BINARY_COLS = set()
for _s in _BINARY_STARTS:
    BINARY_COLS.update(range(_s, _s + 6))

CONTINUOUS_COLS = [c for c in range(279) if c not in BINARY_COLS and c != SEX_COL]


def _per_feature_std(X: np.ndarray) -> np.ndarray:
    return np.std(X, axis=0, ddof=0) + 1e-8


def _snap_binary(X_syn: np.ndarray, X_donor: np.ndarray) -> None:
    for bc in BINARY_COLS:
        X_syn[:, bc] = np.clip(np.round(X_syn[:, bc]), 0, 1)
    X_syn[:, SEX_COL] = FEMALE_CODE


def augment_none(
    X_tr: np.ndarray, y_tr: np.ndarray, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    return X_tr.copy(), y_tr.copy()


def augment_random_oversample(
    X_tr: np.ndarray, y_tr: np.ndarray, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    sex_tr = X_tr[:, SEX_COL]
    f_mask = sex_tr == FEMALE_CODE

    X_f, y_f = X_tr[f_mask], y_tr[f_mask]
    cls_counts = Counter(y_f.tolist())
    if len(cls_counts) == 0:
        return X_tr.copy(), y_tr.copy()

    max_count = max(cls_counts.values())
    X_syn_list, y_syn_list = [], []

    for cls, cnt in cls_counts.items():
        n_needed = max_count - cnt
        if n_needed <= 0:
            continue
        idx_cls = np.where(y_f == cls)[0]
        chosen = rng.choice(idx_cls, size=n_needed, replace=True)
        X_syn_list.append(X_f[chosen])
        y_syn_list.append(np.full(n_needed, cls, dtype=int))

    if X_syn_list:
        X_aug = np.vstack([X_tr] + X_syn_list)
        y_aug = np.concatenate([y_tr] + y_syn_list)
    else:
        X_aug, y_aug = X_tr.copy(), y_tr.copy()

    return X_aug, y_aug


def augment_perturbation(
    X_tr: np.ndarray, y_tr: np.ndarray, rng: np.random.Generator,
    n_copies: int = PERTURB_COPIES,
) -> Tuple[np.ndarray, np.ndarray]:
    sex_tr = X_tr[:, SEX_COL]
    f_mask = sex_tr == FEMALE_CODE
    X_f, y_f = X_tr[f_mask], y_tr[f_mask]
    stds = _per_feature_std(X_f)

    X_syn_list, y_syn_list = [], []
    for _ in range(n_copies):
        noise = rng.normal(0.0, PERTURB_SIGMA, size=X_f.shape) * stds
        for bc in BINARY_COLS:
            noise[:, bc] = 0.0
        noise[:, SEX_COL] = 0.0

        X_syn = X_f + noise
        _snap_binary(X_syn, X_f)
        X_syn_list.append(X_syn)
        y_syn_list.append(y_f.copy())

    X_aug = np.vstack([X_tr] + X_syn_list)
    y_aug = np.concatenate([y_tr] + y_syn_list)
    return X_aug, y_aug


def augment_smotenc(
    X_tr: np.ndarray, y_tr: np.ndarray, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTENC
    except ImportError:
        return X_tr.copy(), y_tr.copy()

    sex_tr = X_tr[:, SEX_COL]
    f_mask = sex_tr == FEMALE_CODE
    X_f, y_f = X_tr[f_mask], y_tr[f_mask]

    cls_counts = Counter(y_f.tolist())
    valid_cls = [c for c, cnt in cls_counts.items() if cnt >= 2]
    if len(valid_cls) < 2:
        return X_tr.copy(), y_tr.copy()

    keep_mask = np.isin(y_f, valid_cls)
    X_f_v, y_f_v = X_f[keep_mask], y_f[keep_mask]
    cat_cols = sorted(BINARY_COLS)

    try:
        sm = SMOTENC(
            categorical_features=cat_cols,
            random_state=int(rng.integers(0, 2**31)),
            k_neighbors=min(5, min(Counter(y_f_v.tolist()).values()) - 1),
        )
        X_res, y_res = sm.fit_resample(X_f_v, y_f_v)
    except Exception:
        return X_tr.copy(), y_tr.copy()

    X_new = X_res[len(X_f_v):]
    y_new = y_res[len(y_f_v):]
    X_new[:, SEX_COL] = FEMALE_CODE
    for bc in BINARY_COLS:
        X_new[:, bc] = np.clip(np.round(X_new[:, bc]), 0, 1)

    X_aug = np.vstack([X_tr, X_new])
    y_aug = np.concatenate([y_tr, y_new])
    return X_aug, y_aug


def augment_cross_gender(
    X_tr: np.ndarray, y_tr: np.ndarray, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    sex_tr = X_tr[:, SEX_COL]
    f_mask = sex_tr == FEMALE_CODE
    m_mask = sex_tr == MALE_CODE
    stds = _per_feature_std(X_tr)

    cls_in_males = set(y_tr[m_mask].tolist())
    target_classes = [
        c for c in cls_in_males
        if c != HEALTHY_CLASS
        and (y_tr[f_mask] == c).sum() < (y_tr[m_mask] == c).sum()
    ]

    X_syn_list, y_syn_list = [], []
    for cls in target_classes:
        m_cls_idx = np.where((y_tr == cls) & m_mask)[0]
        f_cls_count = int((y_tr[f_mask] == cls).sum())
        n_needed = len(m_cls_idx) - f_cls_count
        if n_needed <= 0 or len(m_cls_idx) == 0:
            continue

        donors = m_cls_idx[rng.integers(0, len(m_cls_idx), size=n_needed)]
        X_d = X_tr[donors].copy()
        donor_stds = np.std(X_d, axis=0, ddof=0) + 1e-8
        noise = rng.normal(0.0, PERTURB_SIGMA, size=X_d.shape) * donor_stds
        for bc in BINARY_COLS:
            noise[:, bc] = 0.0
        noise[:, SEX_COL] = 0.0

        X_syn = X_d + noise
        _snap_binary(X_syn, X_d)
        X_syn_list.append(X_syn)
        y_syn_list.append(np.full(n_needed, cls, dtype=int))

    if X_syn_list:
        X_aug = np.vstack([X_tr] + X_syn_list)
        y_aug = np.concatenate([y_tr] + y_syn_list)
    else:
        X_aug, y_aug = X_tr.copy(), y_tr.copy()

    return X_aug, y_aug


def augment_combined(
    X_tr: np.ndarray, y_tr: np.ndarray, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    X1, y1 = augment_random_oversample(X_tr, y_tr, rng)
    X2, y2 = augment_perturbation(X1, y1, rng, n_copies=1)
    return X2, y2


STRATEGIES: Dict[str, Callable] = {
    "none": augment_none,
    "random_oversample": augment_random_oversample,
    "perturbation": augment_perturbation,
    "smotenc": augment_smotenc,
    "cross_gender": augment_cross_gender,
    "combined": augment_combined,
}


def apply_augmentation(
    strategy_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a named augmentation strategy to training data.

    Parameters
    ----------
    strategy_name : key in STRATEGIES dict.
    X_train       : (N, 279) feature matrix.
    y_train       : (N,) class labels.
    rng_seed      : random seed.

    Returns
    -------
    (X_augmented, y_augmented)
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown augmentation strategy '{strategy_name}'. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )
    rng = np.random.default_rng(rng_seed)
    return STRATEGIES[strategy_name](X_train, y_train, rng)
