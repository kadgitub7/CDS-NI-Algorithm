"""
================================================================================
Algorithm 1: Creating the CDS Decision Tree
REVISED FOR:
  • UCI Arrhythmia tabular dataset
  • WFDB-based datasets via tabularized feature extraction
================================================================================

REVISION GOALS
--------------
This revision preserves the full original Algorithm 1 structure while adding:
  1. A dataset abstraction layer (DatasetSchema / DatasetBundle)
  2. Generic loading for:
       • UCI Arrhythmia CSV
       • WFDB record collections
  3. Removal of hard-coded assumptions:
       • 452 users
       • 279 features
       • label column fixed at 279
       • Sex codes fixed to male=0, female=1
  4. Dynamic feature naming and metadata support
  5. Compatibility with downstream Algorithms 2–4 without removing any of the
     original core functions/classes.

IMPORTANT WFDB NOTE
-------------------
WFDB databases store waveform records, not fixed-size row/column feature tables.
To run Algorithms 1–4, we MUST convert each WFDB record into a fixed-length
feature vector. This file therefore adds helper functions that:
  • read WFDB signals / comments / annotations
  • extract summary features per record
  • assign labels through a user-provided mapping or metadata strategy
  • produce a DatasetBundle compatible with the CDS pipeline

The original Algorithm 1 decision-tree logic is preserved.
================================================================================
"""

from __future__ import annotations

import logging
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd

# Optional WFDB import
try:
    import wfdb
except ImportError:
    wfdb = None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 – LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger(name: str = "CDS.Alg1") -> logging.Logger:
    """
    Two handlers:
      • STDOUT – INFO and above
      • STDOUT – DEBUG and above when logger level is raised

    [ENGR] Matches the original structure and keeps verbose tracing available.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)-7s | %(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG)
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger


log = _build_logger()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – GLOBAL CDS CONSTANTS (generalized)
# ─────────────────────────────────────────────────────────────────────────────

# [PAPER] Eq. 3: u_min = 5 / Threshold
DIAGNOSTIC_THRESHOLD: float = 0.025
U_MIN: int = math.ceil(5 / DIAGNOSTIC_THRESHOLD)

# [PAPER Eq. 9] Complexity threshold placeholder
COMPLEXITY_THRESHOLD: int = 16

# [PAPER default] healthy class is 1 in the arrhythmia formulation.
# [ENGR] For generalized datasets, this value is stored in DatasetSchema and
#        propagated into TreeNode / DecisionTree.  This constant remains as a
#        default fallback for backward compatibility.
HEALTHY_CLASS: int = 1

# [LEGACY BACKWARD COMPATIBILITY]
LABEL_COL_IDX: int = 279
N_FEATURES: int = 279


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – DATASET ABSTRACTION LAYER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSchema:
    """
    Generalized schema for any dataset consumed by Algorithms 1–4.

    Attributes
    ----------
    dataset_name       : human-readable dataset identifier.
    source_format      : e.g. 'uci_arrhythmia', 'wfdb'.
    n_features         : number of feature columns in the final tabular matrix.
    healthy_class      : label value corresponding to healthy users.
    label_values       : sorted list of all labels present.
    feature_names      : index -> feature name.
    sex_feature_index  : column index of sex feature, if available.
    male_code          : numeric code for male, if sex is encoded.
    female_code        : numeric code for female, if sex is encoded.
    metadata           : free-form additional dataset metadata.
    """
    dataset_name: str
    source_format: str
    n_features: int
    healthy_class: int
    label_values: List[int]
    feature_names: Dict[int, str]
    sex_feature_index: Optional[int] = None
    male_code: Optional[float] = None
    female_code: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def feature_name(self, idx: int) -> str:
        return self.feature_names.get(idx, f"feat_{idx}")


@dataclass
class DatasetBundle:
    """
    Standardized dataset object consumed by the CDS pipeline.

    Attributes
    ----------
    data   : (N, F) float matrix with NaN for missing values
    labels : (N,) integer class labels
    schema : DatasetSchema
    """
    data: np.ndarray
    labels: np.ndarray
    schema: DatasetSchema


def _build_default_feature_names(n_features: int) -> Dict[int, str]:
    """Create generic names feat_0 ... feat_{F-1}."""
    return {i: f"feat_{i}" for i in range(n_features)}


def _build_ecg_channel_names_arrhythmia(n_features: int) -> Dict[int, str]:
    """
    Original Arrhythmia naming logic preserved, but generalized to n_features.
    """
    names: Dict[int, str] = _build_default_feature_names(n_features)

    base = {
        0: "Age",       1: "Sex",       2: "Height",    3: "Weight",
        4: "QRS_dur",   5: "PR_int",    6: "QT_int",    7: "T_int",
        8: "P_int",
        9:  "QRS_angle", 10: "T_angle",  11: "P_angle",
        12: "QRST_angle", 13: "J_angle", 14: "Heart_rate",
    }
    for k, v in base.items():
        if k < n_features:
            names[k] = v

    channels = ["DI","DII","DIII","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]
    wave_labels = [
        "Q_wid","R_wid","S_wid","Rp_wid","Sp_wid","N_defl",
        "Rag_R","Diph_R","Rag_P","Diph_P","Rag_T","Diph_T",
    ]
    amp_labels = [
        "JJ_amp","Q_amp","R_amp","S_amp","Rp_amp","Sp_amp",
        "P_amp","T_amp","QRSA","QRSTA",
    ]

    for i, ch in enumerate(channels):
        base_w = 15 + i * 12
        for j, lbl in enumerate(wave_labels):
            idx = base_w + j
            if idx < n_features:
                names[idx] = f"{ch}_{lbl}"

        base_a = 159 + i * 10
        for j, lbl in enumerate(amp_labels):
            idx = base_a + j
            if idx < n_features:
                names[idx] = f"{ch}_{lbl}"

    return names


# Preserve original global name object for backward compatibility.
FEATURE_NAMES: Dict[int, str] = _build_ecg_channel_names_arrhythmia(N_FEATURES)

# [LEGACY] Original missing-value-heavy columns in the UCI Arrhythmia dataset.
MISSING_VALUE_COLS: FrozenSet[int] = frozenset({10, 11, 12, 13, 14})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – UCI ARRHYTHMIA LOADING (preserved + generalized)
# ─────────────────────────────────────────────────────────────────────────────

def load_arrhythmia_csv_dataset(path: str) -> DatasetBundle:
    """
    Load the UCI Arrhythmia CSV and return DatasetBundle.

    Returns
    -------
    DatasetBundle for the original UCI dataset.
    """
    log.info(f"Loading UCI Arrhythmia dataset from: {path}")
    df = pd.read_csv(path, header=None, na_values="?")

    if df.shape[1] != 280:
        raise ValueError(
            f"Expected 280 columns for arrhythmia.data but got {df.shape[1]}"
        )

    data   = df.iloc[:, :-1].to_numpy(dtype=float)
    labels = df.iloc[:,  -1].to_numpy(dtype=int)

    schema = DatasetSchema(
        dataset_name      = "UCI Arrhythmia",
        source_format     = "uci_arrhythmia",
        n_features        = data.shape[1],
        healthy_class     = 1,
        label_values      = sorted(np.unique(labels).tolist()),
        feature_names     = _build_ecg_channel_names_arrhythmia(data.shape[1]),
        sex_feature_index = 1,
        male_code         = 0.0,
        female_code       = 1.0,
        metadata          = {"expected_shape": (452, 280)},
    )

    log.info(f"  Loaded {data.shape[0]} users × {data.shape[1]} features.")
    log.info(f"  Label distribution: "
             f"{ {c: int((labels==c).sum()) for c in sorted(set(labels))} }")
    missing_per_col = np.isnan(data).sum(axis=0)
    nonempty = [(i, int(missing_per_col[i])) for i in range(data.shape[1])
                if missing_per_col[i] > 0]
    log.info(f"  Columns with missing values: "
             f"{ {i: n for i,n in nonempty} }")

    return DatasetBundle(data=data, labels=labels, schema=schema)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – WFDB SUPPORT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _require_wfdb() -> None:
    """Raise a friendly error if wfdb is unavailable."""
    if wfdb is None:
        raise ImportError(
            "WFDB support requires the 'wfdb' package. "
            "Install it with: pip install wfdb"
        )


def _parse_age_from_comments(comments: List[str]) -> float:
    """
    Parse age from WFDB header comments if possible.

    Examples:
      'male, age 69'
      'female, age 75'
      'Age: 63'
    """
    for c in comments:
        m = re.search(r"\bage[:=\s,]+(\d+)\b", c, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return np.nan


def _parse_sex_from_comments(comments: List[str],
                             male_code: float = 0.0,
                             female_code: float = 1.0) -> float:
    """
    Parse sex from WFDB comments.

    Returns
    -------
    male_code / female_code / NaN.
    """
    for c in comments:
        low = c.lower()
        if "female" in low or re.search(r"\bsex[:=\s]+f\b", low):
            return female_code
        if "male" in low or re.search(r"\bsex[:=\s]+m\b", low):
            return male_code
    return np.nan


def _safe_nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if x.size else np.nan


def _safe_nanstd(x: np.ndarray) -> float:
    return float(np.nanstd(x)) if x.size else np.nan


def _safe_nanmin(x: np.ndarray) -> float:
    return float(np.nanmin(x)) if x.size else np.nan


def _safe_nanmax(x: np.ndarray) -> float:
    return float(np.nanmax(x)) if x.size else np.nan


def _safe_nanmedian(x: np.ndarray) -> float:
    return float(np.nanmedian(x)) if x.size else np.nan


def _safe_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.square(x)))) if x.size else np.nan


def _extract_wfdb_signal_summary_features(
    signal_matrix: np.ndarray,
    signal_names: List[str],
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Convert one WFDB record's multi-channel signal into a fixed-length vector.

    [ENGR] This preserves compatibility with Algorithms 1–4, which require a
           row-per-user tabular matrix.

    For each channel we compute:
      • mean
      • std
      • min
      • max
      • median
      • rms

    Returns
    -------
    (feature_vector, feature_name_map)
    """
    feats: List[float] = []
    names: Dict[int, str] = {}
    idx = 0

    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix[:, None]

    for ch in range(signal_matrix.shape[1]):
        x = signal_matrix[:, ch].astype(float)
        ch_name = signal_names[ch] if ch < len(signal_names) else f"sig{ch}"

        channel_features = {
            f"{ch_name}_mean":   _safe_nanmean(x),
            f"{ch_name}_std":    _safe_nanstd(x),
            f"{ch_name}_min":    _safe_nanmin(x),
            f"{ch_name}_max":    _safe_nanmax(x),
            f"{ch_name}_median": _safe_nanmedian(x),
            f"{ch_name}_rms":    _safe_rms(x),
        }

        for name, val in channel_features.items():
            names[idx] = name
            feats.append(val)
            idx += 1

    return np.asarray(feats, dtype=float), names


def _extract_wfdb_annotation_summary_features(
    record_name: str,
    annotation_extension: Optional[str] = "atr",
    pn_dir: Optional[str] = None,
    local_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Read a WFDB annotation file if available and summarize it.

    [ENGR] Keeps features fixed-length by extracting counts and simple rhythm
           statistics, if annotations exist.

    Features extracted:
      • ann_count
      • unique_symbol_count

    If no annotation is available, returns empty arrays.
    """
    _require_wfdb()

    try:
        if pn_dir is not None:
            ann = wfdb.rdann(record_name, annotation_extension, pn_dir=pn_dir)
        else:
            full = str(local_dir / record_name) if local_dir is not None else record_name
            ann = wfdb.rdann(full, annotation_extension)
    except Exception:
        return np.asarray([], dtype=float), {}

    syms = ann.symbol if hasattr(ann, "symbol") and ann.symbol is not None else []
    ann_count = float(len(syms))
    unique_count = float(len(set(syms))) if syms else 0.0

    feats = np.asarray([ann_count, unique_count], dtype=float)
    names = {
        0: "ann_count",
        1: "ann_unique_symbol_count",
    }
    return feats, names


def _concat_feature_blocks(
    blocks: List[np.ndarray],
    name_blocks: List[Dict[int, str]],
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Concatenate multiple feature vectors and reindex feature names.
    """
    feats: List[float] = []
    names: Dict[int, str] = {}
    idx = 0

    for block, block_names in zip(blocks, name_blocks):
        for local_i, val in enumerate(block.tolist()):
            feats.append(float(val))
            names[idx] = block_names.get(local_i, f"feat_{idx}")
            idx += 1

    return np.asarray(feats, dtype=float), names


def load_wfdb_dataset(
    db_dir: Optional[str] = None,
    record_names: Optional[List[str]] = None,
    label_map: Optional[Dict[str, int]] = None,
    pn_dir: Optional[str] = None,
    healthy_class: int = 1,
    include_age: bool = True,
    include_sex: bool = True,
    include_annotations: bool = True,
    annotation_extension: str = "atr",
    male_code: float = 0.0,
    female_code: float = 1.0,
) -> DatasetBundle:
    """
    Load a WFDB database and convert it into a fixed-length tabular dataset.

    Parameters
    ----------
    db_dir       : local directory holding record files (e.g. 'data/mitdb').
    record_names : record basenames, e.g. ['100','101',...]
    label_map    : mapping record_name -> class label (required)
    pn_dir       : optional PhysioNet database name for direct remote loading,
                   e.g. 'mitdb'
    healthy_class: label value for healthy class
    include_age  : add age feature if parseable from header/comments
    include_sex  : add sex feature if parseable from header/comments
    include_annotations : whether to read annotations and summarize them
    annotation_extension: WFDB annotation extension, e.g. 'atr'

    Returns
    -------
    DatasetBundle usable by Algorithms 1–4.

    IMPORTANT
    ---------
    The CDS algorithms require one fixed-length feature vector per user/record.
    WFDB waveforms are therefore summarized into scalar features here.
    """
    _require_wfdb()

    if label_map is None:
        raise ValueError("load_wfdb_dataset requires label_map: {record_name -> class_label}")

    if record_names is None:
        record_names = sorted(label_map.keys())

    local_dir = Path(db_dir) if db_dir is not None else None

    rows: List[np.ndarray] = []
    labels: List[int] = []
    feature_names_global: Optional[Dict[int, str]] = None
    sex_feature_index: Optional[int] = None

    log.info("Loading WFDB dataset...")
    if pn_dir is not None:
        log.info(f"  Source: PhysioNet remote database {pn_dir!r}")
    if db_dir is not None:
        log.info(f"  Source: local WFDB directory {db_dir!r}")
    log.info(f"  Records requested: {len(record_names)}")

    for rec_name in record_names:
        if rec_name not in label_map:
            raise ValueError(f"Missing class label for WFDB record {rec_name!r}")

        # Read record
        if pn_dir is not None:
            rec = wfdb.rdrecord(rec_name, pn_dir=pn_dir)
        else:
            if local_dir is None:
                raise ValueError("Either db_dir or pn_dir must be provided for WFDB loading.")
            rec = wfdb.rdrecord(str(local_dir / rec_name))

        # Extract signal matrix
        if hasattr(rec, "p_signal") and rec.p_signal is not None:
            sig = rec.p_signal
        elif hasattr(rec, "d_signal") and rec.d_signal is not None:
            sig = rec.d_signal.astype(float)
        else:
            raise ValueError(f"WFDB record {rec_name!r} has no readable signal matrix.")

        if sig.ndim == 1:
            sig = sig[:, None]

        sig_names = rec.sig_name if getattr(rec, "sig_name", None) else [f"sig{i}" for i in range(sig.shape[1])]
        comments = getattr(rec, "comments", []) or []

        # Block 1: waveform summary
        sig_feats, sig_names_map = _extract_wfdb_signal_summary_features(sig, sig_names)
        feature_blocks = [sig_feats]
        name_blocks = [sig_names_map]

        # Block 2: annotations summary
        if include_annotations:
            ann_feats, ann_names = _extract_wfdb_annotation_summary_features(
                record_name=rec_name,
                annotation_extension=annotation_extension,
                pn_dir=pn_dir,
                local_dir=local_dir,
            )
            if ann_feats.size > 0:
                feature_blocks.append(ann_feats)
                name_blocks.append(ann_names)

        # Block 3: metadata features
        meta_feats: List[float] = []
        meta_names: Dict[int, str] = {}
        meta_idx = 0

        if include_age:
            meta_names[meta_idx] = "Age"
            meta_feats.append(_parse_age_from_comments(comments))
            meta_idx += 1

        if include_sex:
            meta_names[meta_idx] = "Sex"
            meta_feats.append(_parse_sex_from_comments(comments, male_code=male_code, female_code=female_code))
            meta_idx += 1

        meta_names[meta_idx] = "SamplingFreq"
        meta_feats.append(float(getattr(rec, "fs", np.nan)))
        meta_idx += 1

        meta_names[meta_idx] = "NumSignals"
        meta_feats.append(float(sig.shape[1]))
        meta_idx += 1

        feature_blocks.append(np.asarray(meta_feats, dtype=float))
        name_blocks.append(meta_names)

        feat_vec, feat_names = _concat_feature_blocks(feature_blocks, name_blocks)

        rows.append(feat_vec)
        labels.append(int(label_map[rec_name]))

        if feature_names_global is None:
            feature_names_global = feat_names.copy()
            if include_sex:
                # Determine sex feature index dynamically
                for idx, name in feature_names_global.items():
                    if name == "Sex":
                        sex_feature_index = idx
                        break

    data = np.vstack(rows).astype(float)
    label_arr = np.asarray(labels, dtype=int)

    schema = DatasetSchema(
        dataset_name      = "WFDB Tabularized Dataset",
        source_format     = "wfdb",
        n_features        = data.shape[1],
        healthy_class     = healthy_class,
        label_values      = sorted(np.unique(label_arr).tolist()),
        feature_names     = feature_names_global or _build_default_feature_names(data.shape[1]),
        sex_feature_index = sex_feature_index,
        male_code         = male_code if include_sex else None,
        female_code       = female_code if include_sex else None,
        metadata          = {
            "record_count": len(record_names),
            "pn_dir": pn_dir,
            "db_dir": db_dir,
            "annotation_extension": annotation_extension,
        },
    )

    log.info(f"  WFDB tabularization complete: {data.shape[0]} records × {data.shape[1]} features")
    log.info(f"  Label distribution: "
             f"{ {c: int((label_arr==c).sum()) for c in sorted(set(label_arr))} }")

    return DatasetBundle(data=data, labels=label_arr, schema=schema)


def load_dataset(
    path: Optional[str] = None,
    dataset_type: str = "arrhythmia",
    *,
    wfdb_db_dir: Optional[str] = None,
    wfdb_record_names: Optional[List[str]] = None,
    wfdb_label_map: Optional[Dict[str, int]] = None,
    wfdb_pn_dir: Optional[str] = None,
    wfdb_healthy_class: int = 1,
    wfdb_include_age: bool = True,
    wfdb_include_sex: bool = True,
    wfdb_include_annotations: bool = True,
    wfdb_annotation_extension: str = "atr",
    wfdb_male_code: float = 0.0,
    wfdb_female_code: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible public loader.

    RETURNS
    -------
    (data, labels)

    NOTE
    ----
    This preserves the original signature style while internally supporting both
    Arrhythmia and WFDB.  The full schema can be recovered from load_dataset_bundle().
    """
    bundle = load_dataset_bundle(
        path=path,
        dataset_type=dataset_type,
        wfdb_db_dir=wfdb_db_dir,
        wfdb_record_names=wfdb_record_names,
        wfdb_label_map=wfdb_label_map,
        wfdb_pn_dir=wfdb_pn_dir,
        wfdb_healthy_class=wfdb_healthy_class,
        wfdb_include_age=wfdb_include_age,
        wfdb_include_sex=wfdb_include_sex,
        wfdb_include_annotations=wfdb_include_annotations,
        wfdb_annotation_extension=wfdb_annotation_extension,
        wfdb_male_code=wfdb_male_code,
        wfdb_female_code=wfdb_female_code,
    )
    return bundle.data, bundle.labels


def load_dataset_bundle(
    path: Optional[str] = None,
    dataset_type: str = "arrhythmia",
    *,
    wfdb_db_dir: Optional[str] = None,
    wfdb_record_names: Optional[List[str]] = None,
    wfdb_label_map: Optional[Dict[str, int]] = None,
    wfdb_pn_dir: Optional[str] = None,
    wfdb_healthy_class: int = 1,
    wfdb_include_age: bool = True,
    wfdb_include_sex: bool = True,
    wfdb_include_annotations: bool = True,
    wfdb_annotation_extension: str = "atr",
    wfdb_male_code: float = 0.0,
    wfdb_female_code: float = 1.0,
) -> DatasetBundle:
    """
    Generalized loader returning DatasetBundle.
    """
    dataset_type = dataset_type.lower()

    if dataset_type in {"arrhythmia", "uci", "uci_arrhythmia"}:
        if path is None:
            raise ValueError("path is required for dataset_type='arrhythmia'")
        return load_arrhythmia_csv_dataset(path)

    if dataset_type in {"wfdb", "physionet"}:
        return load_wfdb_dataset(
            db_dir=wfdb_db_dir,
            record_names=wfdb_record_names,
            label_map=wfdb_label_map,
            pn_dir=wfdb_pn_dir,
            healthy_class=wfdb_healthy_class,
            include_age=wfdb_include_age,
            include_sex=wfdb_include_sex,
            include_annotations=wfdb_include_annotations,
            annotation_extension=wfdb_annotation_extension,
            male_code=wfdb_male_code,
            female_code=wfdb_female_code,
        )

    raise ValueError(f"Unsupported dataset_type={dataset_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – FEATURE-TYPE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class FeatureKind(Enum):
    """
    [PAPER] Algorithm 1 lines 3-4 distinguish:
      BINARY     – nominal / binary features
      CONTINUOUS – numerical features

    [INFER] Detection is data-driven.
    """
    BINARY     = auto()
    CONTINUOUS = auto()


def classify_features(data: np.ndarray) -> Dict[int, FeatureKind]:
    """
    Classify each feature as BINARY or CONTINUOUS.

    [REVISED] Now uses data.shape[1] instead of hard-coded 279.
    """
    kinds: Dict[int, FeatureKind] = {}
    n_features = data.shape[1]

    for col in range(n_features):
        col_data = data[:, col]
        valid    = col_data[~np.isnan(col_data)]
        if len(valid) == 0:
            kinds[col] = FeatureKind.CONTINUOUS
            continue
        unique_vals = set(valid.tolist())
        if unique_vals.issubset({0.0, 1.0}):
            kinds[col] = FeatureKind.BINARY
        else:
            kinds[col] = FeatureKind.CONTINUOUS
    return kinds


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – BRANCH DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BranchDef:
    """
    Defines membership in a single branch for a given feature.
    """
    feature_idx: int
    branch_idx:  int
    kind:        FeatureKind
    value_set:   Optional[FrozenSet[float]] = None
    lo:          Optional[float]            = None
    hi:          Optional[float]            = None
    label:       str                        = ""

    def contains(self, value: float) -> bool:
        if np.isnan(value):
            return False
        if self.kind == FeatureKind.BINARY:
            return value in self.value_set        # type: ignore[operator]
        else:
            return self.lo <= value < self.hi     # type: ignore[operator]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – FSD / BRANCH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _feature_name(feature_idx: int, schema: Optional[DatasetSchema] = None) -> str:
    """
    Unified feature-name lookup.
    """
    if schema is not None:
        return schema.feature_name(feature_idx)
    return FEATURE_NAMES.get(feature_idx, f"feat_{feature_idx}")


def _sex_labels_for_value(
    value: float,
    feature_idx: int,
    schema: Optional[DatasetSchema] = None,
) -> Optional[str]:
    """
    Return 'male' or 'female' when feature_idx corresponds to sex and the value
    matches the schema-coded sex labels.
    """
    if schema is None or schema.sex_feature_index is None:
        return None
    if feature_idx != schema.sex_feature_index:
        return None

    if schema.male_code is not None and value == schema.male_code:
        return "male"
    if schema.female_code is not None and value == schema.female_code:
        return "female"
    return None


def _fsd_binary(feature_idx: int, user_indices: np.ndarray,
                data: np.ndarray,
                schema: Optional[DatasetSchema] = None) -> List[BranchDef]:
    """
    Create 2 branch definitions for a BINARY feature.
    """
    col   = data[user_indices, feature_idx]
    valid = col[~np.isnan(col)]
    values_present = sorted(set(valid.tolist()))
    if len(values_present) < 2:
        return []

    fname = _feature_name(feature_idx, schema)
    branches = []

    for bidx, v in enumerate(values_present, start=1):
        sex_lbl = _sex_labels_for_value(v, feature_idx, schema)
        if sex_lbl is not None:
            lbl = sex_lbl
        else:
            if float(v).is_integer():
                lbl = f"{fname}={int(v)}"
            else:
                lbl = f"{fname}={v}"
        branches.append(BranchDef(
            feature_idx = feature_idx,
            branch_idx  = bidx,
            kind        = FeatureKind.BINARY,
            value_set   = frozenset({float(v)}),
            label       = lbl,
        ))
    return branches


def _fsd_continuous_median(feature_idx: int, user_indices: np.ndarray,
                           data: np.ndarray,
                           schema: Optional[DatasetSchema] = None) -> List[BranchDef]:
    """
    Create 2 branch definitions for a CONTINUOUS feature using a median split.
    """
    col_vals = data[user_indices, feature_idx]
    valid    = col_vals[~np.isnan(col_vals)]
    if len(valid) == 0:
        return []
    median_val = float(np.median(valid))
    if float(valid.min()) == float(valid.max()):
        return []

    fname = _feature_name(feature_idx, schema)
    b1 = BranchDef(
        feature_idx = feature_idx,
        branch_idx  = 1,
        kind        = FeatureKind.CONTINUOUS,
        lo          = -np.inf,
        hi          = median_val,
        label       = f"{fname}<{median_val}",
    )
    b2 = BranchDef(
        feature_idx = feature_idx,
        branch_idx  = 2,
        kind        = FeatureKind.CONTINUOUS,
        lo          = median_val,
        hi          = np.inf,
        label       = f"{fname}>={median_val}",
    )
    return [b1, b2]


def compute_fsd_branches(feature_idx: int,
                         user_indices: np.ndarray,
                         data: np.ndarray,
                         kind: FeatureKind,
                         schema: Optional[DatasetSchema] = None) -> List[BranchDef]:
    """
    Dispatch to binary or continuous discretisation strategy.
    """
    if kind == FeatureKind.BINARY:
        return _fsd_binary(feature_idx, user_indices, data, schema=schema)
    else:
        return _fsd_continuous_median(feature_idx, user_indices, data, schema=schema)


def filter_users_by_branch(user_indices: np.ndarray,
                            branch: BranchDef,
                            data: np.ndarray) -> np.ndarray:
    """
    Return subset of user_indices in the branch.
    """
    col_vals = data[user_indices, branch.feature_idx]

    if branch.kind == FeatureKind.BINARY:
        val = next(iter(branch.value_set))
        mask = (col_vals == val)
    else:
        mask = (col_vals >= branch.lo) & (col_vals < branch.hi)

    mask = mask & ~np.isnan(col_vals)
    return user_indices[mask]


def health_distribution(user_indices: np.ndarray,
                         labels: np.ndarray) -> Dict[int, int]:
    """
    Count class labels among users.
    """
    classes = labels[user_indices]
    dist = defaultdict(int)
    for c in classes:
        dist[int(c)] += 1
    return dict(sorted(dist.items()))


def compute_branch_probability(branch_user_count: int,
                                parent_user_count: int) -> float:
    """
    P_f = branch_user_count / parent_user_count
    """
    if parent_user_count == 0:
        return 0.0
    return branch_user_count / parent_user_count


def check_branching_condition(branch_user_count: int,
                               parent_user_count: int,
                               threshold: float = DIAGNOSTIC_THRESHOLD) -> Tuple[bool, int, int]:
    """
    Eq. 2 check: branch_user_count >= u_min
    """
    u_min_val = math.ceil(5 / threshold)
    lhs = math.ceil(parent_user_count *
                    compute_branch_probability(branch_user_count, parent_user_count))
    return (lhs >= u_min_val), lhs, u_min_val


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – FEATURE EXCLUSION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def compute_child_feature_set(parent_feature_indices: List[int],
                               branching_feature_k: int,
                               focus_level_m: int) -> List[int]:
    """
    Line 9 pruning rule.
    """
    if focus_level_m <= 2:
        return list(parent_feature_indices)
    return [f for f in parent_feature_indices if f > branching_feature_k]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – TREE NODE DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    """
    One node in the CDS decision tree.
    """
    node_id:          str
    focus_level:      int
    branching_feat_k: int
    branch_f:         int
    branch_def:       Optional[BranchDef]
    user_indices:     np.ndarray
    feature_indices:  List[int]
    branch_prob:      float
    health_dist:      Dict[int, int]
    healthy_class:    int = HEALTHY_CLASS
    children_by_k:    Dict[int, List["TreeNode"]] = field(default_factory=dict)
    is_leaf:          bool                   = False
    prune_reason:     str                    = ""

    @property
    def n_users(self) -> int:
        return len(self.user_indices)

    @property
    def n_features(self) -> int:
        return len(self.feature_indices)

    @property
    def n_healthy(self) -> int:
        return self.health_dist.get(self.healthy_class, 0)

    @property
    def n_diseased(self) -> int:
        return self.n_users - self.n_healthy

    @property
    def all_children(self) -> List["TreeNode"]:
        out = []
        for lst in self.children_by_k.values():
            out.extend(lst)
        return out

    def add_child(self, child: "TreeNode") -> None:
        k = child.branching_feat_k
        if k not in self.children_by_k:
            self.children_by_k[k] = []
        self.children_by_k[k].append(child)

    def __repr__(self) -> str:
        return (f"TreeNode({self.node_id!r}, m={self.focus_level}, "
                f"k={self.branching_feat_k}, f={self.branch_f}, "
                f"n_users={self.n_users}, n_feat={self.n_features})")


@dataclass
class DecisionTree:
    """
    Container for full CDS decision tree.
    """
    root:           TreeNode
    schema:         Optional[DatasetSchema] = None
    nodes_by_level: Dict[int, List[TreeNode]] = field(default_factory=dict)
    all_nodes:      Dict[str, TreeNode]        = field(default_factory=dict)
    valid_branches: Dict[int, int]             = field(default_factory=dict)
    pruned_branches:Dict[int, int]             = field(default_factory=dict)
    feature_kinds:  Dict[int, FeatureKind]     = field(default_factory=dict)
    threshold:      float                      = DIAGNOSTIC_THRESHOLD
    u_min:          int                        = U_MIN

    def register(self, node: TreeNode) -> None:
        m = node.focus_level
        if m not in self.nodes_by_level:
            self.nodes_by_level[m] = []
        self.nodes_by_level[m].append(node)
        self.all_nodes[node.node_id] = node

    def depth(self) -> int:
        return max(self.nodes_by_level.keys()) if self.nodes_by_level else 1

    def count_nodes(self) -> int:
        return len(self.all_nodes)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – CORE TREE BUILDING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _try_split_node(parent:          TreeNode,
                    feature_k:       int,
                    kind:            FeatureKind,
                    data:            np.ndarray,
                    labels:          np.ndarray,
                    threshold:       float,
                    schema:          Optional[DatasetSchema] = None,
                    ) -> Tuple[List[TreeNode], int]:
    """
    Try splitting a parent node on one feature.
    """
    m_child = parent.focus_level + 1

    branch_defs = compute_fsd_branches(
        feature_idx  = feature_k,
        user_indices = parent.user_indices,
        data         = data,
        kind         = kind,
        schema       = schema,
    )
    if len(branch_defs) < 2:
        return [], 0

    branch_user_sets: List[Tuple[BranchDef, np.ndarray]] = []
    for bdef in branch_defs:
        branch_users = filter_users_by_branch(parent.user_indices, bdef, data)
        branch_user_sets.append((bdef, branch_users))

    created: List[TreeNode] = []
    n_pruned = 0
    for bdef, busers in branch_user_sets:
        passes, lhs, u_min_val = check_branching_condition(
            branch_user_count=len(busers),
            parent_user_count=parent.n_users,
            threshold=threshold,
        )
        if not passes:
            n_pruned += 1
            continue

        p_f = compute_branch_probability(len(busers), parent.n_users)
        feats = compute_child_feature_set(
            parent_feature_indices=parent.feature_indices,
            branching_feature_k=feature_k,
            focus_level_m=m_child,
        )
        child_id = f"{parent.node_id}|k{feature_k}_f{bdef.branch_idx}"
        child = TreeNode(
            node_id=child_id,
            focus_level=m_child,
            branching_feat_k=feature_k,
            branch_f=bdef.branch_idx,
            branch_def=bdef,
            user_indices=busers,
            feature_indices=feats,
            branch_prob=p_f,
            health_dist=health_distribution(busers, labels),
            healthy_class=parent.healthy_class,
        )
        created.append(child)

    if len(created) == 1:
        created[0].is_leaf = True
        created[0].prune_reason = "Single-branch split; further branching disabled."

    return created, n_pruned


def _expand_node(parent:      TreeNode,
                 data:        np.ndarray,
                 labels:      np.ndarray,
                 kinds:       Dict[int, FeatureKind],
                 threshold:   float,
                 tree:        DecisionTree) -> int:
    """
    Try every available feature as a branching candidate.
    """
    m_child     = parent.focus_level + 1
    n_created   = 0
    n_pruned    = 0
    valid_ks    = []

    log.debug(f"  Expanding {parent.node_id!r} -> m={m_child}, "
              f"trying {len(parent.feature_indices)} features …")

    for k in sorted(parent.feature_indices):
        kind = kinds[k]
        children, pruned = _try_split_node(
            parent    = parent,
            feature_k = k,
            kind      = kind,
            data      = data,
            labels    = labels,
            threshold = threshold,
            schema    = tree.schema,
        )
        n_pruned += pruned
        if children:
            valid_ks.append(k)
            n_created += len(children)
            for child in children:
                parent.add_child(child)
                tree.register(child)

    if m_child not in tree.valid_branches:
        tree.valid_branches[m_child]  = 0
        tree.pruned_branches[m_child] = 0
    tree.valid_branches[m_child]  += n_created
    tree.pruned_branches[m_child] += n_pruned

    log.info(
        f"  m={parent.focus_level}->{m_child} | node {parent.node_id!r} | "
        f"valid splits: {len(valid_ks)} features -> {n_created} children | "
        f"pruned branches: {n_pruned}"
    )

    if not valid_ks:
        parent.is_leaf    = True
        parent.prune_reason = f"No feature split passes Eq.2 at m={m_child}"
    return n_created


def build_decision_tree(data:       np.ndarray,
                         labels:    np.ndarray,
                         threshold: float = DIAGNOSTIC_THRESHOLD,
                         max_m:     int   = 2,
                         schema:    Optional[DatasetSchema] = None) -> DecisionTree:
    """
    Algorithm 1: Creating a Decision Tree

    [REVISED] Accepts DatasetSchema so that feature count, names, sex coding,
    and healthy class are dataset-specific rather than Arrhythmia-specific.
    """
    if schema is None:
        # Backward-compatible fallback for legacy callers.
        schema = DatasetSchema(
            dataset_name="LegacyDataset",
            source_format="unknown",
            n_features=data.shape[1],
            healthy_class=HEALTHY_CLASS,
            label_values=sorted(np.unique(labels).tolist()),
            feature_names=_build_default_feature_names(data.shape[1]),
            sex_feature_index=1 if data.shape[1] > 1 else None,
            male_code=0.0,
            female_code=1.0,
        )

    log.info("=" * 72)
    log.info("Algorithm 1: Building CDS Decision Tree")
    log.info(f"  dataset    = {schema.dataset_name} ({schema.source_format})")
    log.info(f"  threshold  = {threshold},  u_min = {math.ceil(5/threshold)}")
    log.info(f"  N_users    = {len(labels)},  N_features = {data.shape[1]}")
    log.info("=" * 72)

    kinds = classify_features(data)
    n_binary = sum(1 for k in kinds.values() if k == FeatureKind.BINARY)
    n_cont   = sum(1 for k in kinds.values() if k == FeatureKind.CONTINUOUS)
    log.info(f"Feature classification: {n_binary} binary, {n_cont} continuous.")

    all_user_indices    = np.arange(len(labels), dtype=int)
    all_feature_indices = list(range(data.shape[1]))

    root = TreeNode(
        node_id          = "root",
        focus_level      = 1,
        branching_feat_k = -1,
        branch_f         = 0,
        branch_def       = None,
        user_indices     = all_user_indices,
        feature_indices  = all_feature_indices,
        branch_prob      = 1.0,
        health_dist      = health_distribution(all_user_indices, labels),
        healthy_class    = schema.healthy_class,
    )

    tree = DecisionTree(
        root          = root,
        schema        = schema,
        feature_kinds = kinds,
        threshold     = threshold,
        u_min         = math.ceil(5 / threshold),
    )
    tree.register(root)
    tree.nodes_by_level[1] = [root]

    log.info(f"Root node: {root.n_users} users, {root.n_features} features.")
    log.info(f"  Health distribution: {root.health_dist}")

    current_level_nodes = [root]

    for m in range(2, max_m + 1):
        log.info(f"\n{'─'*60}")
        log.info(f"FOCUS LEVEL m = {m}: "
                 f"expanding {len(current_level_nodes)} parent node(s) …")

        next_level_nodes = []
        total_new = 0

        for parent_node in current_level_nodes:
            n_new = _expand_node(
                parent    = parent_node,
                data      = data,
                labels    = labels,
                kinds     = kinds,
                threshold = threshold,
                tree      = tree,
            )
            total_new += n_new
            next_level_nodes.extend(parent_node.all_children)

        log.info(f"  Total new nodes at m={m}: {total_new}")

        if total_new == 0:
            log.info(f"  No valid branches at m={m}.  Tree construction complete.")
            break

        current_level_nodes = next_level_nodes

    log.info(f"\n{'='*72}")
    log.info(f"Algorithm 1 complete.  Tree depth: {tree.depth()}  |  "
             f"Total nodes: {tree.count_nodes()}")
    return tree


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 – REPORTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def print_tree_summary(tree: DecisionTree) -> None:
    """
    Print a compact tree summary table.
    """
    print("\n" + "=" * 90)
    print(f"{'CDS DECISION TREE SUMMARY':^90}")
    print(f"  dataset={tree.schema.dataset_name if tree.schema else 'unknown'}, "
          f"threshold={tree.threshold}, u_min={tree.u_min}, depth={tree.depth()}, "
          f"total_nodes={tree.count_nodes()}")
    print("=" * 90)
    print(f"{'m':>3}  {'k':>4}  {'f':>3}  {'|U|':>5}  {'%H':>6}  "
          f"{'P_f':>6}  {'#feat':>5}  label")
    print("-" * 90)
    for m in sorted(tree.nodes_by_level.keys()):
        for node in tree.nodes_by_level[m]:
            pct_h = 100 * node.n_healthy / node.n_users if node.n_users else 0.0
            lbl   = node.branch_def.label if node.branch_def else "—root—"
            print(f"{m:>3}  {node.branching_feat_k:>4}  {node.branch_f:>3}  "
                  f"{node.n_users:>5}  {pct_h:>5.1f}%  {node.branch_prob:>6.4f}  "
                  f"{node.n_features:>5}  {lbl}")
        if m < max(tree.nodes_by_level.keys()):
            print()
    print("=" * 90)


def print_node_details(node: TreeNode,
                       feature_kinds: Dict[int, FeatureKind],
                       schema: Optional[DatasetSchema] = None) -> None:
    """
    Print full details for a single TreeNode.
    """
    fname = lambda k: _feature_name(k, schema)

    print(f"\n{'─'*60}")
    print(f"NODE: {node.node_id}")
    print(f"  Focus level m     : {node.focus_level}")
    print(f"  Branching feat k  : {node.branching_feat_k} "
          f"({fname(node.branching_feat_k) if node.branching_feat_k >= 0 else 'root'})")
    print(f"  Branch f          : {node.branch_f}")
    print(f"  Branch definition : {node.branch_def.label if node.branch_def else 'root'}")
    print(f"  Users (|U|)       : {node.n_users}")
    print(f"  Healthy/Diseased  : {node.n_healthy} / {node.n_diseased}")
    print(f"  Branch probability: {node.branch_prob:.6f}")
    print(f"  Available features: {node.n_features}")
    print(f"  Health distribution:")
    for cls, cnt in sorted(node.health_dist.items()):
        tag = " ← HEALTHY" if cls == node.healthy_class else ""
        print(f"    Class {cls:2d}: {cnt:3d}{tag}")
    print(f"  Children (by branching feature):")
    if node.children_by_k:
        for k, children in sorted(node.children_by_k.items()):
            kname = fname(k)
            labels_str = ", ".join(
                f"f={c.branch_f}({c.branch_def.label if c.branch_def else '?'})"
                f"->{c.n_users}u" for c in children
            )
            print(f"    k={k} ({kname}): {labels_str}")
    else:
        print(f"    (leaf node – {node.prune_reason or 'no further splits'})")


def print_level_statistics(tree: DecisionTree) -> None:
    """
    Per-level statistics table.
    """
    print("\n" + "=" * 60)
    print(f"{'LEVEL STATISTICS':^60}")
    print("=" * 60)
    print(f"{'m':>3}  {'nodes':>6}  {'valid_br':>9}  {'pruned_feat':>11}  "
          f"{'avg_|U|':>8}")
    print("-" * 60)
    for m in sorted(tree.nodes_by_level.keys()):
        nodes   = tree.nodes_by_level[m]
        avg_u   = np.mean([n.n_users for n in nodes]) if nodes else 0
        valid   = tree.valid_branches.get(m, 0)
        pruned  = tree.pruned_branches.get(m, 0)
        print(f"{m:>3}  {len(nodes):>6}  {valid:>9}  {pruned:>11}  {avg_u:>8.1f}")
    print("=" * 60)


def print_branching_features_at_level(tree: DecisionTree, m: int) -> None:
    """
    List all features producing valid branches at focus level m.
    """
    nodes_at_m = tree.nodes_by_level.get(m, [])
    if not nodes_at_m:
        print(f"\nNo nodes at focus level m={m}.")
        return

    by_parent_k: Dict[str, Dict[int, List[TreeNode]]] = defaultdict(lambda: defaultdict(list))
    for node in nodes_at_m:
        parent_id = node.node_id.rsplit("|", 1)[0]
        by_parent_k[parent_id][node.branching_feat_k].append(node)

    print(f"\n{'='*72}")
    print(f"VALID BRANCHING FEATURES AT m={m}  (u_min={tree.u_min})")
    print(f"{'='*72}")
    print(f"{'k':>4}  {'feature':20}  {'kind':12}  {'branches':>8}  branch_sizes")
    print("-" * 72)

    seen_ks: Set[int] = set()
    for parent_id, k_map in sorted(by_parent_k.items()):
        for k, children in sorted(k_map.items()):
            if k in seen_ks:
                continue
            seen_ks.add(k)
            fname = _feature_name(k, tree.schema)
            kind  = tree.feature_kinds.get(k, FeatureKind.CONTINUOUS)
            sizes = ", ".join(
                f"{c.branch_def.label}:{c.n_users}" for c in children
            )
            print(f"{k:>4}  {fname:20}  {kind.name:12}  {len(children):>8}  {sizes}")
    print(f"{'='*72}")
    print(f"Total unique valid branching features at m={m}: {len(seen_ks)}")


def print_sex_branch_details(tree: DecisionTree) -> None:
    """
    Print information for the sex-branching feature if the dataset has one.
    """
    schema = tree.schema
    if schema is None or schema.sex_feature_index is None:
        print("\n" + "=" * 60)
        print(f"{'SEX-BRANCHED NODES':^60}")
        print("=" * 60)
        print("  No sex feature defined in dataset schema.")
        return

    SEX_K = schema.sex_feature_index
    nodes_at_2 = tree.nodes_by_level.get(2, [])
    sex_nodes  = [n for n in nodes_at_2 if n.branching_feat_k == SEX_K]

    print("\n" + "=" * 60)
    print(f"{'SEX-BRANCHED NODES':^60}")
    print("=" * 60)
    if not sex_nodes:
        print(f"  Sex (k={SEX_K}) did not produce valid branches!")
        return

    for node in sex_nodes:
        lbl = node.branch_def.label if node.branch_def else "?"
        print(f"  Branch f={node.branch_f}  ({lbl})")
        print(f"    Users       : {node.n_users}")
        print(f"    Branch prob : {node.branch_prob:.4f}")
        print(f"    Healthy     : {node.n_healthy}  "
              f"({100*node.n_healthy/node.n_users:.1f}%)")
        print(f"    Diseased    : {node.n_diseased}  "
              f"({100*node.n_diseased/node.n_users:.1f}%)")
        print(f"    Health dist : {node.health_dist}")
        passes_eq2, lhs, u_min_val = check_branching_condition(
            node.n_users, tree.root.n_users, tree.threshold
        )
        print(f"    Eq.2 check  : ceil({tree.root.n_users}×{node.branch_prob:.4f})"
              f" = {lhs} >= {u_min_val}? {passes_eq2}")
        print(f"    Features    : {node.n_features}")
        print()


def explain_level3_impossibility(tree: DecisionTree) -> None:
    """
    Explain whether focus level 3 is possible under Eq.2 constraints.
    """
    print("\n" + "=" * 72)
    print("WHY FOCUS LEVEL 3 MAY OR MAY NOT BE POSSIBLE (Eq. 2 proof)")
    print("=" * 72)
    u_min = tree.u_min
    nodes_at_2 = tree.nodes_by_level.get(2, [])
    if not nodes_at_2:
        print("  No level-2 nodes to analyse.")
        return

    for parent in nodes_at_2:
        best_possible = parent.n_users // 2
        print(f"\n  Parent node: {parent.node_id!r}  |  {parent.n_users} users")
        print(f"    Best-case branch size (50/50 split): {best_possible}")
        print(f"    u_min required                     : {u_min}")
        print(f"    Best-case >= u_min?                : {best_possible >= u_min}")
        print(f"    -> LEVEL 3 {'POSSIBLE' if best_possible >= u_min else 'IMPOSSIBLE'} "
              f"for this parent node.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 – VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_user_partition(tree: DecisionTree) -> None:
    """
    Verify user-set partition invariants.
    """
    print("\n" + "=" * 60)
    print("USER-PARTITION VALIDATION")
    print("=" * 60)
    all_ok = True

    for m in sorted(tree.nodes_by_level.keys()):
        if m == 1:
            continue

        groups: Dict[Tuple[str, int], List[TreeNode]] = defaultdict(list)
        for node in tree.nodes_by_level[m]:
            parent_id = node.node_id.rsplit("|", 1)[0]
            groups[(parent_id, node.branching_feat_k)].append(node)

        for (parent_id, k), siblings in groups.items():
            parent = tree.all_nodes.get(parent_id)
            if parent is None:
                continue

            union_sets = [set(s.user_indices.tolist()) for s in siblings]
            sibling_union = set().union(*union_sets)

            ok_disjoint = True
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    inter = union_sets[i] & union_sets[j]
                    if inter:
                        ok_disjoint = False
                        print(f"  FAIL disjoint: m={m} k={k} "
                              f"f={siblings[i].branch_f}∩f={siblings[j].branch_f} "
                              f"has {len(inter)} common users!")

            ok_subset = sibling_union.issubset(set(parent.user_indices.tolist()))
            status = "PASS" if (ok_disjoint and ok_subset) else "FAIL"
            if not (ok_disjoint and ok_subset):
                all_ok = False

            print(f"  m={m} | k={k} ({_feature_name(k, tree.schema)}) | "
                  f"parent={parent.n_users}u | "
                  f"children={[s.n_users for s in siblings]} | "
                  f"union={len(sibling_union)} | disjoint={ok_disjoint} | "
                  f"subset={ok_subset} -> {status}")

    print(f"\n  Overall validation: {'ALL PASS ✓' if all_ok else 'FAILURES DETECTED ✗'}")


def validate_eq2(tree: DecisionTree) -> None:
    """
    Verify every non-root node satisfies Eq. 2.
    """
    print("\n" + "=" * 60)
    print("EQ. 2 VALIDATION (all non-root nodes)")
    print("=" * 60)
    all_ok = True
    for m in sorted(tree.nodes_by_level.keys()):
        if m == 1:
            continue
        for node in tree.nodes_by_level[m]:
            parent_id = node.node_id.rsplit("|", 1)[0]
            parent = tree.all_nodes.get(parent_id)
            if parent is None:
                continue
            passes, lhs, u_min_val = check_branching_condition(
                node.n_users, parent.n_users, tree.threshold
            )
            if not passes:
                all_ok = False
                print(f"  FAIL: {node.node_id!r} | lhs={lhs} < u_min={u_min_val}")
            else:
                print(f"  PASS: {node.node_id!r} | {lhs} >= {u_min_val}")
    if all_ok:
        print("  All nodes satisfy Eq. 2 ✓")


def validate_feature_exclusion(tree: DecisionTree) -> None:
    """
    Verify m>2 nodes satisfy line-9 feature exclusion rule.
    """
    print("\n" + "=" * 60)
    print("FEATURE EXCLUSION VALIDATION (line 9, m>2 nodes only)")
    print("=" * 60)
    violations = 0
    for m in sorted(tree.nodes_by_level.keys()):
        if m <= 2:
            print(f"  m={m}: exclusion rule does NOT apply (line 8 condition: m>2)")
            continue
        for node in tree.nodes_by_level[m]:
            k = node.branching_feat_k
            bad = [f for f in node.feature_indices if f <= k]
            if bad:
                violations += 1
                print(f"  FAIL: {node.node_id!r} | k={k} but features <= k present: {bad}")
            else:
                print(f"  PASS: {node.node_id!r} | k={k} | "
                      f"features all > k ✓ ({node.n_features} remaining)")
    if violations == 0:
        print("  All m>2 nodes satisfy feature exclusion rule ✓")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 – MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main(
    data_path: Optional[str] = None,
    dataset_type: str = "arrhythmia",
    wfdb_db_dir: Optional[str] = None,
    wfdb_record_names: Optional[List[str]] = None,
    wfdb_label_map: Optional[Dict[str, int]] = None,
    wfdb_pn_dir: Optional[str] = None,
) -> DecisionTree:
    """
    End-to-end execution of Algorithm 1.

    Supports:
      • dataset_type='arrhythmia'
      • dataset_type='wfdb'
    """
    bundle = load_dataset_bundle(
        path=data_path,
        dataset_type=dataset_type,
        wfdb_db_dir=wfdb_db_dir,
        wfdb_record_names=wfdb_record_names,
        wfdb_label_map=wfdb_label_map,
        wfdb_pn_dir=wfdb_pn_dir,
    )
    data, labels, schema = bundle.data, bundle.labels, bundle.schema

    tree = build_decision_tree(data, labels, schema=schema)

    print_tree_summary(tree)
    print_level_statistics(tree)
    print_branching_features_at_level(tree, m=2)
    print_sex_branch_details(tree)
    explain_level3_impossibility(tree)
    validate_eq2(tree)
    validate_feature_exclusion(tree)

    if schema.sex_feature_index is not None:
        for node in tree.nodes_by_level.get(2, []):
            if node.branching_feat_k == schema.sex_feature_index:
                print_node_details(node, tree.feature_kinds, schema=tree.schema)

    return tree


def get_arrhythmia_path(filename: str = "arrhythmia.data") -> Path:
    """
    Find the arrhythmia dataset path in a cross-platform way.
    """
    if len(sys.argv) > 1:
        p = Path(sys.argv[1]).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Dataset not found: {p}")

    cwd_path = Path.cwd() / filename
    if cwd_path.exists():
        return cwd_path.resolve()

    script_dir = Path(__file__).parent
    script_path = script_dir / filename
    if script_path.exists():
        return script_path.resolve()

    data_path = script_dir / "data" / filename
    if data_path.exists():
        return data_path.resolve()

    raise FileNotFoundError(
        f"Could not locate {filename}. "
        "Place it beside the script, inside ./data/, "
        "or pass the path as a command-line argument."
    )


if __name__ == "__main__":
    # Default: original Arrhythmia dataset behavior
    path = str(Path(__file__).parent / "data" / "arrhythmia.data")
    tree = main(data_path=path, dataset_type="arrhythmia")