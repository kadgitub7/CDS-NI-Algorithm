"""
Dataset Configuration for Multi-Database CDS Pipeline
=====================================================

Provides dataset-specific constants and loaders for:
  - UCI Arrhythmia (original paper dataset)
  - PhysioNet 2017 AF Classification Challenge

Usage:
    from dataset_config import set_dataset, get_config

    set_dataset("uci")        # or "physionet"
    cfg = get_config()
    data, labels = cfg.load_fn(cfg.data_path)
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("CDS.DatasetConfig")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ─────────────────────────────────────────────────────────────────────────────
# UCI Arrhythmia feature names (279 features)
# ─────────────────────────────────────────────────────────────────────────────

def _build_uci_feature_names() -> Dict[int, str]:
    names: Dict[int, str] = {
        0: "Age", 1: "Sex", 2: "Height", 3: "Weight",
        4: "QRS_dur", 5: "PR_int", 6: "QT_int", 7: "T_int",
        8: "P_int",
        9: "QRS_angle", 10: "T_angle", 11: "P_angle",
        12: "QRST_angle", 13: "J_angle", 14: "Heart_rate",
    }
    channels = ["DI", "DII", "DIII", "AVR", "AVL", "AVF",
                "V1", "V2", "V3", "V4", "V5", "V6"]
    wave_labels = [
        "Q_wid", "R_wid", "S_wid", "Rp_wid", "Sp_wid", "N_defl",
        "Rag_R", "Diph_R", "Rag_P", "Diph_P", "Rag_T", "Diph_T",
    ]
    amp_labels = [
        "JJ_amp", "Q_amp", "R_amp", "S_amp", "Rp_amp", "Sp_amp",
        "P_amp", "T_amp", "QRSA", "QRSTA",
    ]
    for i, ch in enumerate(channels):
        base_w = 15 + i * 12
        for j, lbl in enumerate(wave_labels):
            names[base_w + j] = f"{ch}_{lbl}"
        base_a = 159 + i * 10
        for j, lbl in enumerate(amp_labels):
            names[base_a + j] = f"{ch}_{lbl}"
    return names


# ─────────────────────────────────────────────────────────────────────────────
# PhysioNet 2017 feature names (40 extracted features)
# ─────────────────────────────────────────────────────────────────────────────

PHYSIONET_FEATURE_NAMES: Dict[int, str] = {
    0: "Mean", 1: "Std", 2: "Skewness", 3: "Kurtosis",
    4: "Min", 5: "Max", 6: "Range", 7: "Median",
    8: "IQR", 9: "MAD", 10: "RMS",
    11: "ZeroCrossRate", 12: "MeanAbsDiff",
    13: "Duration_s", 14: "NumSamples",
    15: "MeanRR", 16: "StdRR", 17: "RMSSD", 18: "MedianRR",
    19: "MeanHR", 20: "StdHR", 21: "NumBeats",
    22: "pNN50", 23: "pNN20", 24: "RR_Range",
    25: "DominantFreq", 26: "SpectralCentroid", 27: "SpectralBandwidth",
    28: "SpectralRolloff", 29: "SpectralEntropy",
    30: "Power_VLF", 31: "Power_LF", 32: "Power_HF", 33: "Power_20_50Hz",
    34: "LF_HF_Ratio",
    35: "MeanR_Amp", 36: "StdR_Amp",
    37: "MeanQRS_Width", 38: "WaveformComplexity",
    39: "AutocorrLag1",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name: str
    n_features: int
    feature_names: Dict[int, str]
    healthy_class: int
    disease_classes: Tuple[int, ...]
    all_classes: Tuple[int, ...]
    label_col_idx: int
    missing_value_cols: FrozenSet[int]
    data_path: str
    has_sex_feature: bool
    sex_feature_idx: Optional[int]
    male_value: Optional[float]
    female_value: Optional[float]
    diagnostic_threshold: float
    healthy_range_percentile: float
    load_fn: Optional[Callable] = None


# ─────────────────────────────────────────────────────────────────────────────
# UCI Arrhythmia loader
# ─────────────────────────────────────────────────────────────────────────────

def load_uci_arrhythmia(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load UCI Arrhythmia CSV (452 x 280 or augmented)."""
    df = pd.read_csv(path, header=None, na_values="?")
    n_rows, n_cols = df.shape
    if n_cols != 280:
        raise ValueError(f"Expected 280 columns but got {n_cols}.")
    data = df.iloc[:, :279].to_numpy(dtype=float)
    labels = df.iloc[:, 279].to_numpy(dtype=int)
    log.info(f"UCI Arrhythmia: {n_rows} users x 279 features")
    log.info(f"  Labels: {dict(sorted({int(c): int((labels==c).sum()) for c in set(labels)}.items()))}")
    return data, labels


# ─────────────────────────────────────────────────────────────────────────────
# PhysioNet 2017 loader (with feature extraction and caching)
# ─────────────────────────────────────────────────────────────────────────────

PHYSIONET_SAMPLING_RATE = 300
PHYSIONET_LABEL_MAP = {"N": 1, "A": 2, "O": 2}
PHYSIONET_EXCLUDED_LABELS = {"~"}


def _load_ecg(mat_path: str) -> np.ndarray:
    import scipy.io
    d = scipy.io.loadmat(mat_path)
    return d["val"].flatten().astype(float)


def _extract_features_single(signal: np.ndarray, fs: int = PHYSIONET_SAMPLING_RATE) -> np.ndarray:
    """Extract 40 features from a single ECG waveform."""
    import scipy.signal
    import scipy.stats

    n = len(signal)
    duration = n / fs
    features = np.full(40, np.nan)

    features[0] = np.mean(signal)
    features[1] = np.std(signal)
    features[2] = float(scipy.stats.skew(signal))
    features[3] = float(scipy.stats.kurtosis(signal))
    features[4] = np.min(signal)
    features[5] = np.max(signal)
    features[6] = features[5] - features[4]
    features[7] = np.median(signal)
    q75, q25 = np.percentile(signal, [75, 25])
    features[8] = q75 - q25
    features[9] = np.mean(np.abs(signal - np.mean(signal)))
    features[10] = np.sqrt(np.mean(signal ** 2))
    zc = np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0)
    features[11] = zc / duration
    features[12] = np.mean(np.abs(np.diff(signal)))
    features[13] = duration
    features[14] = float(n)

    try:
        sos_bp = scipy.signal.butter(4, [5, 15], btype="band", fs=fs, output="sos")
        filtered = scipy.signal.sosfilt(sos_bp, signal)
        squared = filtered ** 2
        win_size = max(int(0.15 * fs), 1)
        kernel = np.ones(win_size) / win_size
        smoothed = np.convolve(squared, kernel, mode="same")
        threshold = 0.4 * np.max(smoothed)
        min_distance = int(0.3 * fs)
        peaks, props = scipy.signal.find_peaks(
            smoothed, height=threshold, distance=min_distance
        )
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / fs
            features[15] = np.mean(rr_intervals)
            features[16] = np.std(rr_intervals)
            features[17] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 1 else 0.0
            features[18] = np.median(rr_intervals)
            hr = 60.0 / rr_intervals
            features[19] = np.mean(hr)
            features[20] = np.std(hr)
            features[21] = float(len(peaks))
            rr_diff = np.abs(np.diff(rr_intervals))
            features[22] = np.sum(rr_diff > 0.05) / len(rr_diff) if len(rr_diff) > 0 else 0.0
            features[23] = np.sum(rr_diff > 0.02) / len(rr_diff) if len(rr_diff) > 0 else 0.0
            features[24] = np.max(rr_intervals) - np.min(rr_intervals)
            r_amps = signal[peaks]
            features[35] = np.mean(r_amps)
            features[36] = np.std(r_amps)
            widths = scipy.signal.peak_widths(smoothed, peaks, rel_height=0.5)[0] / fs
            features[37] = np.mean(widths)
        else:
            features[21] = float(len(peaks))
    except Exception:
        pass

    try:
        freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=min(n, 1024))
        total_power = np.sum(psd)
        if total_power > 0:
            features[25] = freqs[np.argmax(psd)]
            features[26] = np.sum(freqs * psd) / total_power
            features[27] = np.sqrt(np.sum(((freqs - features[26]) ** 2) * psd) / total_power)
            cumsum = np.cumsum(psd) / total_power
            rolloff_idx = np.searchsorted(cumsum, 0.85)
            features[28] = freqs[min(rolloff_idx, len(freqs) - 1)]
            psd_norm = psd / total_power
            psd_norm = psd_norm[psd_norm > 0]
            features[29] = -np.sum(psd_norm * np.log2(psd_norm))
            features[30] = np.sum(psd[(freqs >= 0) & (freqs < 1)])
            features[31] = np.sum(psd[(freqs >= 1) & (freqs < 8)])
            features[32] = np.sum(psd[(freqs >= 8) & (freqs < 20)])
            features[33] = np.sum(psd[(freqs >= 20) & (freqs < 50)])
            if features[32] > 0:
                features[34] = features[31] / features[32]
    except Exception:
        pass

    try:
        local_max = scipy.signal.argrelextrema(signal, np.greater, order=int(0.02 * fs))[0]
        local_min = scipy.signal.argrelextrema(signal, np.less, order=int(0.02 * fs))[0]
        features[38] = (len(local_max) + len(local_min)) / duration
    except Exception:
        pass
    try:
        acf = np.correlate(signal[:min(n, fs * 5)], signal[:min(n, fs * 5)], mode="full")
        acf = acf[len(acf) // 2:]
        if acf[0] != 0:
            features[39] = acf[1] / acf[0]
    except Exception:
        pass

    return features


def load_physionet2017(
    data_dir: str,
    max_records: Optional[int] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PhysioNet 2017 data, extract features, and return (data, labels).

    Binary classification: Normal (N) -> 1, Abnormal (A+O) -> 2.
    Noisy (~) recordings are excluded.
    Features are cached to disk after first extraction.
    """
    cache_dir = Path(data_dir).parent / "physioNetData2017_cache"
    cache_file = cache_dir / f"features_n{max_records or 'all'}.pkl"

    if use_cache and cache_file.exists():
        log.info(f"Loading cached PhysioNet features from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["data"], cached["labels"]

    log.info(f"Extracting features from {data_dir}...")
    ref_path = os.path.join(data_dir, "REFERENCE-original.csv")
    ref_df = pd.read_csv(ref_path, header=None, names=["record_id", "label"])
    if max_records is not None:
        ref_df = ref_df.head(max_records)

    records = []
    labels_list = []
    skipped = 0
    excluded_noisy = 0

    for _, row in ref_df.iterrows():
        rid = row["record_id"]
        label_str = row["label"]
        mat_path = os.path.join(data_dir, f"{rid}.mat")
        if not os.path.exists(mat_path):
            skipped += 1
            continue
        if label_str in PHYSIONET_EXCLUDED_LABELS:
            excluded_noisy += 1
            continue
        if label_str not in PHYSIONET_LABEL_MAP:
            skipped += 1
            continue
        try:
            sig = _load_ecg(mat_path)
            feats = _extract_features_single(sig)
            records.append(feats)
            labels_list.append(PHYSIONET_LABEL_MAP[label_str])
        except Exception as e:
            log.warning(f"Error processing {rid}: {e}")
            skipped += 1

    data = np.array(records, dtype=float)
    labels = np.array(labels_list, dtype=int)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)
        log.info(f"Cached features to {cache_file}")

    log.info(f"PhysioNet 2017: {len(records)} records x 40 features "
             f"({skipped} skipped, {excluded_noisy} noisy excluded)")
    log.info(f"  Labels: {dict(sorted({int(c): int((labels==c).sum()) for c in set(labels)}.items()))}")
    return data, labels


# ─────────────────────────────────────────────────────────────────────────────
# Preset configurations
# ─────────────────────────────────────────────────────────────────────────────

def _make_uci_config() -> DatasetConfig:
    uci_disease = (2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16)
    return DatasetConfig(
        name="uci",
        n_features=279,
        feature_names=_build_uci_feature_names(),
        healthy_class=1,
        disease_classes=uci_disease,
        all_classes=(1,) + uci_disease,
        label_col_idx=279,
        missing_value_cols=frozenset({10, 11, 12, 13, 14}),
        data_path=str(PROJECT_ROOT / "data" / "arrhythmia.data"),
        has_sex_feature=True,
        sex_feature_idx=1,
        male_value=0.0,
        female_value=1.0,
        diagnostic_threshold=0.025,
        healthy_range_percentile=100.0,
        load_fn=load_uci_arrhythmia,
    )


def _make_physionet_config() -> DatasetConfig:
    # Uses the 188-feature extraction matching the paper's reference [59]
    # (Shreyasi Datta PhysioNet 2017 challenge submission).
    # HEALTHY_RANGE_PERCENTILE: Paper Eq.5 explicitly requires 100% — all
    # healthy user values must fall within [b_min, b_max]. This guarantees
    # FA=0 (zero false alarms on training data).
    from physionet2017_feature_extraction_188 import (
        load_physionet2017_188,
        N_FEATURES as PN188_N_FEATURES,
        FEATURE_NAMES as PN188_FEATURE_NAMES,
    )
    return DatasetConfig(
        name="physionet",
        n_features=PN188_N_FEATURES,
        feature_names=PN188_FEATURE_NAMES,
        healthy_class=1,
        disease_classes=(2,),
        all_classes=(1, 2),
        label_col_idx=PN188_N_FEATURES,
        missing_value_cols=frozenset(),
        data_path=str(PROJECT_ROOT / "data" / "physioNetData2017"),
        has_sex_feature=False,
        sex_feature_idx=None,
        male_value=None,
        female_value=None,
        diagnostic_threshold=0.025,
        healthy_range_percentile=100.0,
        load_fn=load_physionet2017_188,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Global state and switching
# ─────────────────────────────────────────────────────────────────────────────

_CONFIGS = {
    "uci": _make_uci_config,
    "physionet": _make_physionet_config,
}

_active_config: Optional[DatasetConfig] = None


def set_dataset(name: str) -> DatasetConfig:
    """
    Set the active dataset configuration.

    Parameters
    ----------
    name : "uci" or "physionet"

    Returns
    -------
    The active DatasetConfig.
    """
    global _active_config
    name = name.lower().strip()
    if name not in _CONFIGS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(_CONFIGS.keys())}")
    _active_config = _CONFIGS[name]()
    log.info(f"Dataset set to: {_active_config.name} "
             f"({_active_config.n_features} features, "
             f"healthy_class={_active_config.healthy_class})")
    return _active_config


def get_config() -> DatasetConfig:
    """Return the active dataset config. Defaults to UCI if not set."""
    global _active_config
    if _active_config is None:
        _active_config = _make_uci_config()
    return _active_config
