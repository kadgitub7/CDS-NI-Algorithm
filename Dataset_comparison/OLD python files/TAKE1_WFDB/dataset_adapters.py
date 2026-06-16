from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    wfdb = None


@dataclass
class DatasetSchema:
    dataset_name: str
    n_features: int
    healthy_class: int
    label_values: List[int]
    feature_names: Dict[int, str]
    sex_feature_index: Optional[int] = None
    male_code: Optional[float] = None
    female_code: Optional[float] = None
    source_format: str = "tabular"


@dataclass
class DatasetBundle:
    data: np.ndarray
    labels: np.ndarray
    schema: DatasetSchema


def build_default_feature_names(n_features: int) -> Dict[int, str]:
    return {i: f"feat_{i}" for i in range(n_features)}


def build_arrhythmia_feature_names(n_features: int) -> Dict[int, str]:
    names = build_default_feature_names(n_features)
    base = {
        0: "Age", 1: "Sex", 2: "Height", 3: "Weight",
        4: "QRS_dur", 5: "PR_int", 6: "QT_int", 7: "T_int",
        8: "P_int", 9: "QRS_angle", 10: "T_angle", 11: "P_angle",
        12: "QRST_angle", 13: "J_angle", 14: "Heart_rate",
    }
    for k, v in base.items():
        if k < n_features:
            names[k] = v
    return names


def load_uci_arrhythmia_dataset(path: str) -> DatasetBundle:
    df = pd.read_csv(path, header=None, na_values="?")
    if df.shape[1] != 280:
        raise ValueError(f"Expected 280 columns for UCI Arrhythmia, got {df.shape[1]}")
    data = df.iloc[:, :-1].to_numpy(dtype=float)
    labels = df.iloc[:, -1].to_numpy(dtype=int)

    schema = DatasetSchema(
        dataset_name="uci_arrhythmia",
        n_features=data.shape[1],
        healthy_class=1,
        label_values=sorted(np.unique(labels).tolist()),
        feature_names=build_arrhythmia_feature_names(data.shape[1]),
        sex_feature_index=1,
        male_code=0.0,
        female_code=1.0,
        source_format="tabular",
    )
    return DatasetBundle(data=data, labels=labels, schema=schema)


def _safe_comment_value(comments: List[str], key: str) -> Optional[str]:
    pat = re.compile(rf"{re.escape(key)}\s*[:=]\s*(.+)", re.IGNORECASE)
    for c in comments:
        m = pat.search(c)
        if m:
            return m.group(1).strip()
    return None


def _parse_age(comments: List[str]) -> float:
    for c in comments:
        m = re.search(r'age[:=\s]+(\d+)', c, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return np.nan


def _parse_sex(comments: List[str]) -> float:
    for c in comments:
        low = c.lower()
        if "female" in low or re.search(r'\bsex[:=\s]+f\b', low):
            return 1.0
        if "male" in low or re.search(r'\bsex[:=\s]+m\b', low):
            return 0.0
    return np.nan


def _extract_signal_summary_features(sig: np.ndarray, sig_names: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
    feats: List[float] = []
    names: Dict[int, str] = {}
    idx = 0
    for ch in range(sig.shape[1]):
        x = sig[:, ch].astype(float)
        ch_name = sig_names[ch] if ch < len(sig_names) else f"ch{ch}"
        vals = {
            f"{ch_name}_mean": np.nanmean(x),
            f"{ch_name}_std": np.nanstd(x),
            f"{ch_name}_min": np.nanmin(x),
            f"{ch_name}_max": np.nanmax(x),
            f"{ch_name}_median": np.nanmedian(x),
            f"{ch_name}_rms": np.sqrt(np.nanmean(x ** 2)),
        }
        for n, v in vals.items():
            feats.append(float(v))
            names[idx] = n
            idx += 1
    return np.asarray(feats, dtype=float), names


def load_wfdb_dataset(
    db_dir: str,
    record_names: List[str],
    label_map: Dict[str, int],
    healthy_class: int = 1,
    include_age: bool = True,
    include_sex: bool = True,
) -> DatasetBundle:
    if wfdb is None:
        raise ImportError("wfdb is not installed. Install with: pip install wfdb")

    rows: List[np.ndarray] = []
    labels: List[int] = []
    feature_names: Optional[Dict[int, str]] = None
    sex_feature_index: Optional[int] = None

    for rec_name in record_names:
        rec = wfdb.rdrecord(str(Path(db_dir) / rec_name))
        sig = rec.p_signal if rec.p_signal is not None else rec.d_signal.astype(float)
        if sig.ndim == 1:
            sig = sig[:, None]

        sig_names = rec.sig_name or [f"ch{i}" for i in range(sig.shape[1])]
        feat_vec, feat_names = _extract_signal_summary_features(sig, sig_names)

        extras = []
        extra_names: Dict[int, str] = {}
        comments = getattr(rec, "comments", []) or []

        base_idx = len(feat_vec)

        if include_age:
            extra_names[base_idx + len(extras)] = "Age"
            extras.append(_parse_age(comments))

        if include_sex:
            sex_feature_index = base_idx + len(extras)
            extra_names[sex_feature_index] = "Sex"
            extras.append(_parse_sex(comments))

        extra_names[base_idx + len(extras)] = "SamplingFreq"
        extras.append(float(getattr(rec, "fs", np.nan)))

        extra_names[base_idx + len(extras)] = "NumSignals"
        extras.append(float(sig.shape[1]))

        row = np.concatenate([feat_vec, np.asarray(extras, dtype=float)])
        rows.append(row)

        if rec_name not in label_map:
            raise ValueError(f"Missing label for WFDB record {rec_name}")
        labels.append(int(label_map[rec_name]))

        if feature_names is None:
            feature_names = {}
            feature_names.update(feat_names)
            feature_names.update(extra_names)

    X = np.vstack(rows).astype(float)
    y = np.asarray(labels, dtype=int)

    schema = DatasetSchema(
        dataset_name="wfdb_tabularized",
        n_features=X.shape[1],
        healthy_class=healthy_class,
        label_values=sorted(np.unique(y).tolist()),
        feature_names=feature_names or build_default_feature_names(X.shape[1]),
        sex_feature_index=sex_feature_index,
        male_code=0.0,
        female_code=1.0,
        source_format="wfdb",
    )
    return DatasetBundle(data=X, labels=y, schema=schema)


def load_dataset_generic(dataset_type: str, **kwargs) -> DatasetBundle:
    if dataset_type == "uci_arrhythmia":
        return load_uci_arrhythmia_dataset(kwargs["path"])
    if dataset_type == "wfdb":
        return load_wfdb_dataset(
            db_dir=kwargs["db_dir"],
            record_names=kwargs["record_names"],
            label_map=kwargs["label_map"],
            healthy_class=kwargs.get("healthy_class", 1),
            include_age=kwargs.get("include_age", True),
            include_sex=kwargs.get("include_sex", True),
        )
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")