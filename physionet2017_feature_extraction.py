"""
Feature extraction from PhysioNet 2017 AF Classification Challenge data.

Converts raw single-lead ECG waveforms (.mat files) into a tabular feature
matrix compatible with the CDS pipeline (same interface as UCI Arrhythmia).

The PhysioNet 2017 dataset:
- Single-lead ECG (300 Hz sampling rate)
- Labels: N (Normal), A (AFib), O (Other rhythm), ~ (Noisy/unclassifiable)
- Each record is a .mat file with key 'val', shape (1, N_samples)
- Record lengths vary (9s to 60s+)

We extract time-domain and frequency-domain features from each ECG waveform
to create a fixed-width feature vector per record.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import scipy.stats

log = logging.getLogger("CDS.PhysioNet2017")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    log.addHandler(h)
    log.propagate = False

SAMPLING_RATE = 300  # Hz

LABEL_MAP = {"N": 1, "A": 2, "O": 3, "~": 4}
LABEL_NAMES = {1: "Normal", 2: "AFib", 3: "Other", 4: "Noisy"}
HEALTHY_CLASS = 1
DISEASE_CLASSES = (2, 3, 4)


def _load_ecg(mat_path: str) -> np.ndarray:
    """Load a single ECG waveform from a .mat file, return 1D float array."""
    d = scipy.io.loadmat(mat_path)
    sig = d["val"].flatten().astype(float)
    return sig


def _extract_features_single(signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Extract features from a single ECG waveform.

    Features (40 total):
    Time-domain (0-14):
      0: mean, 1: std, 2: skewness, 3: kurtosis,
      4: min, 5: max, 6: range, 7: median,
      8: IQR, 9: MAD, 10: RMS,
      11: zero_crossing_rate, 12: mean_abs_diff,
      13: signal_length_seconds, 14: num_samples

    R-peak / HRV features (15-24):
      15: mean_RR, 16: std_RR, 17: RMSSD, 18: median_RR,
      19: mean_HR, 20: std_HR, 21: num_beats,
      22: pNN50, 23: pNN20, 24: RR_range

    Frequency-domain (25-34):
      25: dominant_freq, 26: spectral_centroid, 27: spectral_bandwidth,
      28: spectral_rolloff, 29: spectral_entropy,
      30: power_0_1Hz (VLF), 31: power_1_8Hz (LF),
      32: power_8_20Hz (HF), 33: power_20_50Hz,
      34: LF_HF_ratio

    Morphological (35-39):
      35: mean_R_amplitude, 36: std_R_amplitude,
      37: mean_QRS_width, 38: waveform_complexity (num_extrema/sec),
      39: autocorrelation_lag1
    """
    n = len(signal)
    duration = n / fs
    features = np.full(40, np.nan)

    # --- Time-domain ---
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

    # --- R-peak detection (simple pan-tompkins-like) ---
    try:
        # Bandpass filter 5-15 Hz to isolate QRS
        sos_bp = scipy.signal.butter(4, [5, 15], btype="band", fs=fs, output="sos")
        filtered = scipy.signal.sosfilt(sos_bp, signal)
        squared = filtered ** 2
        # Moving average (150ms window)
        win_size = int(0.15 * fs)
        if win_size < 1:
            win_size = 1
        kernel = np.ones(win_size) / win_size
        smoothed = np.convolve(squared, kernel, mode="same")
        # Adaptive threshold
        threshold = 0.4 * np.max(smoothed)
        min_distance = int(0.3 * fs)  # min 300ms between beats
        peaks, props = scipy.signal.find_peaks(
            smoothed, height=threshold, distance=min_distance
        )

        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / fs  # in seconds
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

            # Morphological features from R-peaks
            r_amps = signal[peaks]
            features[35] = np.mean(r_amps)
            features[36] = np.std(r_amps)
            # Estimate QRS width: width at half-prominence
            widths = scipy.signal.peak_widths(smoothed, peaks, rel_height=0.5)[0] / fs
            features[37] = np.mean(widths)
        else:
            features[21] = float(len(peaks))
    except Exception:
        pass

    # --- Frequency-domain ---
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
            # Band powers
            features[30] = np.sum(psd[(freqs >= 0) & (freqs < 1)])
            features[31] = np.sum(psd[(freqs >= 1) & (freqs < 8)])
            features[32] = np.sum(psd[(freqs >= 8) & (freqs < 20)])
            features[33] = np.sum(psd[(freqs >= 20) & (freqs < 50)])
            if features[32] > 0:
                features[34] = features[31] / features[32]
    except Exception:
        pass

    # --- Additional morphological ---
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


N_FEATURES = 40

FEATURE_NAMES: Dict[int, str] = {
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


def load_physionet2017(
    data_dir: str,
    reference_csv: str = "REFERENCE.csv",
    max_records: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load PhysioNet 2017 data and extract features.

    Parameters
    ----------
    data_dir : path to directory containing .mat, .hea, and REFERENCE.csv
    reference_csv : name of the reference/label file
    max_records : limit number of records (for testing)

    Returns
    -------
    data   : np.ndarray, shape (N, 40), float64
    labels : np.ndarray, shape (N,), int (1=Normal, 2=AFib, 3=Other, 4=Noisy)
    record_ids : list of record IDs
    """
    ref_path = os.path.join(data_dir, reference_csv)
    ref_df = pd.read_csv(ref_path, header=None, names=["record_id", "label"])
    if max_records is not None:
        ref_df = ref_df.head(max_records)

    records = []
    labels_list = []
    record_ids = []
    skipped = 0

    for _, row in ref_df.iterrows():
        rid = row["record_id"]
        label_str = row["label"]
        mat_path = os.path.join(data_dir, f"{rid}.mat")
        if not os.path.exists(mat_path):
            skipped += 1
            continue
        if label_str not in LABEL_MAP:
            skipped += 1
            continue

        try:
            sig = _load_ecg(mat_path)
            feats = _extract_features_single(sig)
            records.append(feats)
            labels_list.append(LABEL_MAP[label_str])
            record_ids.append(rid)
        except Exception as e:
            log.warning(f"Error processing {rid}: {e}")
            skipped += 1

    data = np.array(records, dtype=float)
    labels = np.array(labels_list, dtype=int)
    log.info(f"Loaded {len(records)} records ({skipped} skipped)")
    log.info(f"  Shape: {data.shape}")
    log.info(f"  Labels: { {LABEL_NAMES[c]: int((labels==c).sum()) for c in sorted(set(labels))} }")
    return data, labels, record_ids


def load_dataset_physionet2017(
    data_dir: str,
    max_records: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for CDS_Paper_Algorithms.load_dataset().
    Returns (data, labels) only (no record_ids).
    """
    data, labels, _ = load_physionet2017(data_dir, max_records=max_records)
    return data, labels


if __name__ == "__main__":
    import time
    data_dir = str(Path(__file__).parent / "data" / "physioNetData2017")
    max_rec = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    log.info(f"Extracting features from PhysioNet 2017 (max_records={max_rec})")
    t0 = time.time()
    data, labels, rids = load_physionet2017(data_dir, max_records=max_rec)
    elapsed = time.time() - t0
    log.info(f"Feature extraction took {elapsed:.1f}s ({elapsed/len(rids)*1000:.0f}ms/record)")
    log.info(f"Feature matrix shape: {data.shape}")
    nan_counts = np.isnan(data).sum(axis=0)
    log.info(f"NaN counts per feature: {dict(enumerate(nan_counts))}")
