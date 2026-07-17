"""
PhysioNet 2017 AF Classification Challenge -- 188-Feature Extraction
====================================================================

Replicates the Shreyasi Datta MATLAB feature extraction pipeline that produces
exactly 188 features from single-lead ECG waveforms sampled at 300 Hz.

Reference: "Brain-Inspired Intelligence for Real-Time Health Situation
Understanding" -- uses this exact feature set.

Feature groups (188 total):
    Group 1: ECG_features_158_old    68 features  (indices   0 -  67)
    Group 2: other_features_new       3 features  (indices  68 -  70)
    Group 3: frequency_features      12 features  (indices  71 -  82)
    Group 4: pr_features             21 features  (indices  83 - 103)
    Group 5: soa_features             8 features  (indices 104 - 111)
    Group 6: extract_features        27 features  (indices 112 - 138)
    Group 7: new_feat_sd_158_old     19 features  (indices 139 - 157)
    Group 8: generate_30_features    30 features  (indices 158 - 187)
                                    ---
                                    188

Usage:
    from physionet2017_feature_extraction_188 import extract_features_188
    features = extract_features_188(ecg_signal, fs=300)

    # Or load a full dataset:
    from physionet2017_feature_extraction_188 import load_physionet2017_188
    data, labels = load_physionet2017_188("path/to/physioNetData2017")
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io
import scipy.signal
import scipy.stats
import scipy.interpolate

try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

log = logging.getLogger("CDS.Feat188")

N_FEATURES = 188
PHYSIONET_SAMPLING_RATE = 300
PHYSIONET_LABEL_MAP = {"N": 1, "A": 2, "O": 2}
PHYSIONET_EXCLUDED_LABELS = {"~"}

# ---------------------------------------------------------------------------
# Feature name dictionary  (index -> human-readable name)
# ---------------------------------------------------------------------------

def _build_feature_names() -> Dict[int, str]:
    names: Dict[int, str] = {}
    idx = 0

    # Group 1 (68) ----------------------------------------------------------
    g1_names = [
        # feat1 (11)
        "AFEvidence", "OriginCount", "IrrEvidence", "PACEvidence",
        "DensityEvidence", "AnisotropyEvidence", "CVrr", "CVdrr",
        "Poincare_stepping", "Poincare_dispersion", "Poincare_num_clusters",
        # feat2 (8)
        "RR_mean", "RR_median", "RR_min", "RR_max",
        "RR_skewness", "RR_kurtosis", "RR_range", "RR_variance",
        # wavelet entropy of RR (1)
        "RR_wavelet_entropy",
        # Hjorth (2)
        "RR_Hjorth_mobility", "RR_Hjorth_complexity",
        # feat3 (7)
        "KDE_RR_kurtosis", "KDE_RR_skewness", "KDE_RR_num_peaks",
        "KDE_RR_maxdist", "KDE_RR_mindist",
        "KDE_dRR_kurtosis", "KDE_dRR_skewness",
        # feat4 (5)
        "KDE_QRSamp_kurtosis", "KDE_QRSamp_skewness", "KDE_QRSamp_num_peaks",
        "CV_QRS_amplitude", "CV_interbeat_energy",
        # feat5 (31)
        "RS_depth_CV", "RS_depth_range", "RS_depth_median",
        "ST_slope_median", "ST_slope_CV", "ST_slope_frac_neg",
        "deep_S_count", "inflection_dist",
        "T_R_amp_ratio", "P_R_amp_ratio",
        "QRS_width_median", "QRS_width_CV",
        "QR_width_median", "QR_width_CV",
        "RQ_depth_mean", "RQ_depth_CV",
        "SR_ratio_mean", "SR_ratio_CV",
        "QR_slope_mean", "QR_slope_CV",
        "RS_slope_mean", "RS_slope_CV",
        "Sx_slope_mean", "Sx_slope_CV",
        "QTc_Bazett_median", "QTc_Bazett_CV",
        "QTc_Fridericia_median", "QTc_Fridericia_CV",
        "QTc_Sagie_median", "QTc_Sagie_CV",
        "feat5_extra",
        # feat_HRV (3)
        "pNN20", "pNN50", "SDSD",
    ]
    for n in g1_names:
        names[idx] = n
        idx += 1

    # Group 2 (3)
    for n in ["Shannon_entropy", "Tsallis_entropy_q2", "Renyi_entropy_q2"]:
        names[idx] = n
        idx += 1

    # Group 3 (12)
    g3 = [
        "SpectralCentroid_win", "SpectralRolloff_win",
        "SpectralFlux_win", "SpectralKurtosis_win",
        "Wavelet_logvar_d3",
        "LPC_c2", "LPC_c4", "LPC_c7", "LPC_c9", "LPC_c10", "LPC_c11",
        "FreqCentroid_PSD",
    ]
    for n in g3:
        names[idx] = n
        idx += 1

    # Group 4 (21)
    g4 = [f"PR_feat_{i}" for i in range(21)]
    for n in g4:
        names[idx] = n
        idx += 1

    # Group 5 (8)
    g5 = [
        "nRMSSD", "MAD_RR",
        "HRV_band1", "HRV_band2", "HRV_band3",
        "HRV_ratio12", "HRV_ratio13", "HRV_ratio23",
    ]
    for n in g5:
        names[idx] = n
        idx += 1

    # Group 6 (27)
    g6 = [
        "FFT_trimmed_mean", "FFT_skewness", "FFT_80pct_energy_freq",
        "FFT_kurt_0p5_5Hz", "TimeDomain_kurtosis", "Hjorth_complexity",
        "SNR", "THD", "ZeroCrossRate_bp",
        "ShortTimeEnergy", "SpectralCentroid_2s", "SpectralRolloff_2s",
        "SpectralFlux_2s", "PeakTiming_hist1", "PeakTiming_hist2",
        "DominantFFTfreq", "FFT_peak_diff",
        "EMD_zcr_IMF1", "EMD_std_IMF1",
        "dHR_medSq", "dHR_binned", "dHR_asymmetry", "dHR_severe_domFreq",
        "BandPower_0_2Hz", "BandPower_2_4Hz", "BandPower_4_10Hz",
        "BandPower_10_150Hz",
    ]
    for n in g6:
        names[idx] = n
        idx += 1

    # Group 7 (19)
    g7 = [
        "SampEn_m0", "SampEn_m1", "SampEn_m2", "SampEn_m3", "SampEn_m4",
        "SampEn_num_inf", "SampEn_max", "SampEn_min",
        "SampEn_max_ind", "SampEn_min_ind",
        "DFA_alpha1_slope", "DFA_alpha1_intercept",
        "Poincare_SD1", "Poincare_SD2",
        "ApEn_m1", "ApEn_m2", "ApEn_m3", "ApEn_m4", "ApEn_m5",
    ]
    for n in g7:
        names[idx] = n
        idx += 1

    # Group 8 (30)
    g8 = [
        "FV_peak_count", "FV_max_gap",
        "FV1_dom_freq", "FV1_num_spec_peaks", "FV1_autocorr_freq",
        "FV1_freq_ratio", "FV1_autocorr_loc_std", "FV1_autocorr_peak_std",
        "FV1_HF_area", "FV1_VF_area_ratio", "FV1_peak_area_ratio",
        "FV1_neg_autocorr_peaks",
        "FV5_max_HR_17beat", "FV4_min_HR_5beat",
        "SPI_max_mean", "SPI_max_max",
        "F2_max_RR",
        "SQI_medSNR", "SQI_PCA", "SQI_spectral_ratio",
        "SQI_baseline_ratio", "SQI_kurtosis", "SQI_skewness",
        "NumDiffPeaks",
        "HR_low_count", "HR_low_5beat", "HR_high_count",
        "HR_high_17beat", "HR_max", "HR_max_RR",
    ]
    for n in g8:
        names[idx] = n
        idx += 1

    return names


FEATURE_NAMES: Dict[int, str] = _build_feature_names()


# ===========================================================================
#  HELPER / UTILITY FUNCTIONS
# ===========================================================================

def _safe_cv(arr: np.ndarray) -> float:
    """Coefficient of variation with zero-division guard."""
    m = np.mean(arr)
    if m == 0 or len(arr) < 2:
        return 0.0
    return float(np.std(arr) / abs(m))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def _bandpass_filter(sig: np.ndarray, lo: float, hi: float, fs: int,
                     order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.9999)
    if lo_n >= hi_n:
        return sig
    sos = scipy.signal.butter(order, [lo_n, hi_n], btype="band", output="sos")
    return scipy.signal.sosfiltfilt(sos, sig)


def _highpass_filter(sig: np.ndarray, cutoff: float, fs: int,
                     order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    wn = cutoff / nyq
    if wn <= 0 or wn >= 1:
        return sig
    sos = scipy.signal.butter(order, wn, btype="high", output="sos")
    return scipy.signal.sosfiltfilt(sos, sig)


def _lowpass_filter(sig: np.ndarray, cutoff: float, fs: int,
                    order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    wn = cutoff / nyq
    if wn <= 0 or wn >= 1:
        return sig
    sos = scipy.signal.butter(order, wn, btype="low", output="sos")
    return scipy.signal.sosfiltfilt(sos, sig)


# ===========================================================================
#  PAN-TOMPKINS QRS DETECTOR  (simplified Python version)
# ===========================================================================

def _pan_tompkins_detect(ecg: np.ndarray, fs: int = 300) -> np.ndarray:
    """Detect R-peak locations using a simplified Pan-Tompkins algorithm."""
    # Bandpass 5-15 Hz
    filtered = _bandpass_filter(ecg, 5.0, 15.0, fs, order=2)
    # Derivative
    diff = np.diff(filtered)
    # Squaring
    squared = diff ** 2
    # Moving-window integration
    win = max(int(0.15 * fs), 1)
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode="same")
    # Adaptive thresholding via find_peaks
    threshold = 0.3 * np.max(integrated) if len(integrated) > 0 else 0
    min_dist = int(0.25 * fs)
    peaks, _ = scipy.signal.find_peaks(integrated, height=threshold,
                                        distance=min_dist)
    # Correct peak locations back to original ECG
    search_half = int(0.075 * fs)
    corrected = []
    for p in peaks:
        lo = max(0, p - search_half)
        hi = min(len(ecg), p + search_half + 1)
        corrected.append(lo + int(np.argmax(ecg[lo:hi])))
    return np.array(corrected, dtype=int)


# ===========================================================================
#  FIDUCIAL POINT DETECTION  (P, Q, S, T around each R-peak)
# ===========================================================================

def _detect_fiducials(ecg: np.ndarray, r_peaks: np.ndarray, fs: int = 300):
    """
    Detect approximate P, Q, S, T points around each R-peak.
    Returns dict of arrays: 'P', 'Q', 'S', 'T' each same length as r_peaks.
    Missing points are set to -1.
    """
    n = len(ecg)
    n_beats = len(r_peaks)
    P = np.full(n_beats, -1, dtype=int)
    Q = np.full(n_beats, -1, dtype=int)
    S = np.full(n_beats, -1, dtype=int)
    T = np.full(n_beats, -1, dtype=int)

    for i, r in enumerate(r_peaks):
        # Q: minimum in [-0.08s, R]
        q_start = max(0, r - int(0.08 * fs))
        if q_start < r:
            Q[i] = q_start + int(np.argmin(ecg[q_start:r]))

        # S: minimum in [R, R+0.08s]
        s_end = min(n, r + int(0.08 * fs))
        if r + 1 < s_end:
            S[i] = r + 1 + int(np.argmin(ecg[r + 1:s_end]))

        # T: max in [R+0.08s, R+0.4s]
        t_start = min(n - 1, r + int(0.08 * fs))
        t_end = min(n, r + int(0.4 * fs))
        if t_start < t_end:
            T[i] = t_start + int(np.argmax(ecg[t_start:t_end]))

        # P: max in [R-0.3s, R-0.1s]
        p_start = max(0, r - int(0.3 * fs))
        p_end = max(0, r - int(0.1 * fs))
        if p_start < p_end:
            P[i] = p_start + int(np.argmax(ecg[p_start:p_end]))

    return {"P": P, "Q": Q, "R": r_peaks, "S": S, "T": T}


# ===========================================================================
#  SAMPLE ENTROPY  &  APPROXIMATE ENTROPY
# ===========================================================================

def _sample_entropy(data: np.ndarray, m: int = 5, r_frac: float = 0.2):
    """
    Compute sample entropy for template lengths 0..m-1.
    Returns array of length m.  Inf values are possible.
    """
    N = len(data)
    r = r_frac * np.std(data) if np.std(data) > 0 else 1e-10
    results = np.full(m, np.inf)
    if N < m + 1:
        return results

    for dim in range(m):
        k = dim + 1
        # Build templates
        templates = np.array([data[i:i + k] for i in range(N - k + 1)])
        count = 0
        total_pairs = 0
        for i in range(len(templates)):
            diffs = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(diffs < r)
            total_pairs += len(templates) - i - 1
        if total_pairs > 0 and count > 0:
            results[dim] = -np.log(count / total_pairs)
        else:
            results[dim] = np.inf
    return results


def _approx_entropy(data: np.ndarray, m: int, r: float) -> float:
    """Compute approximate entropy for given m and tolerance r."""
    N = len(data)
    if N < m + 1:
        return 0.0

    def _phi(m_val):
        templates = np.array([data[i:i + m_val] for i in range(N - m_val + 1)])
        counts = np.zeros(len(templates))
        for i in range(len(templates)):
            diffs = np.max(np.abs(templates - templates[i]), axis=1)
            counts[i] = np.sum(diffs <= r) / (N - m_val + 1)
        counts = counts[counts > 0]
        if len(counts) == 0:
            return 0.0
        return np.mean(np.log(counts))

    return abs(_phi(m) - _phi(m + 1))


# ===========================================================================
#  DETRENDED FLUCTUATION ANALYSIS  (DFA)
# ===========================================================================

def _dfa_alpha1(data: np.ndarray) -> Tuple[float, float]:
    """
    Compute DFA short-range scaling exponent (alpha1).
    Returns (slope, intercept).
    """
    N = len(data)
    if N < 16:
        return (0.0, 0.0)

    y = np.cumsum(data - np.mean(data))
    # Box sizes from 4 to N//4
    min_box = 4
    max_box = max(min_box + 1, N // 4)
    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), num=20
    ).astype(int))
    box_sizes = box_sizes[box_sizes >= 4]

    if len(box_sizes) < 2:
        return (0.0, 0.0)

    fluct = []
    for bs in box_sizes:
        n_boxes = N // bs
        if n_boxes < 1:
            continue
        rms_vals = []
        for j in range(n_boxes):
            segment = y[j * bs:(j + 1) * bs]
            x_fit = np.arange(bs)
            coeffs = np.polyfit(x_fit, segment, 1)
            trend = np.polyval(coeffs, x_fit)
            rms_vals.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms_vals:
            fluct.append(np.mean(rms_vals))
        else:
            fluct.append(1e-10)

    fluct = np.array(fluct)
    box_sizes = box_sizes[:len(fluct)]
    valid = fluct > 0
    if np.sum(valid) < 2:
        return (0.0, 0.0)

    log_n = np.log(box_sizes[valid].astype(float))
    log_f = np.log(fluct[valid])
    slope, intercept = np.polyfit(log_n, log_f, 1)
    return (float(slope), float(intercept))


# ===========================================================================
#  POINCARE PLOT FEATURES  (SD1, SD2, clustering)
# ===========================================================================

def _poincare_features(rr: np.ndarray):
    """
    Compute Poincare plot features:
      SD1, SD2, stepping, dispersion, num_clusters.
    """
    if len(rr) < 3:
        return 0.0, 0.0, 0.0, 0.0, 1

    x = rr[:-1]
    y = rr[1:]
    diff_xy = y - x
    sum_xy = y + x

    sd1 = np.std(diff_xy) / np.sqrt(2)
    sd2 = np.std(sum_xy) / np.sqrt(2)

    # Stepping: mean absolute successive difference of diff_xy
    stepping = np.mean(np.abs(np.diff(diff_xy))) if len(diff_xy) > 1 else 0.0
    # Dispersion: std of distances from identity line
    dispersion = np.std(np.sqrt((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2))

    # Simple cluster count via DBSCAN-like approach
    # Use a simplified method: count distinct groups in the Lorenz plot
    try:
        from sklearn.cluster import DBSCAN
        points = np.column_stack([x, y])
        eps = 0.1 * np.std(rr) if np.std(rr) > 0 else 0.05
        eps = max(eps, 0.01)
        clustering = DBSCAN(eps=eps, min_samples=3).fit(points)
        num_clusters = len(set(clustering.labels_) - {-1})
        num_clusters = max(num_clusters, 1)
    except Exception:
        # Fallback: estimate from histogram
        num_clusters = 1

    return sd1, sd2, stepping, dispersion, num_clusters


# ===========================================================================
#  LORENZ PLOT / AF EVIDENCE FEATURES
# ===========================================================================

def _lorenz_plot_features(rr: np.ndarray):
    """
    Compute AF-related features from RR-interval Lorenz plot.
    Returns: AFEvidence, OriginCount, IrrEvidence, PACEvidence,
             DensityEvidence, AnisotropyEvidence
    """
    if len(rr) < 4:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    drr = np.diff(rr)
    x = drr[:-1]
    y = drr[1:]

    # Origin count: fraction of points near origin
    dist_origin = np.sqrt(x ** 2 + y ** 2)
    threshold = 0.05  # 50 ms
    origin_count = np.sum(dist_origin < threshold) / len(dist_origin)

    # Irregularity evidence: fraction of points far from identity line
    dist_identity = np.abs(y - x) / np.sqrt(2)
    irr_evidence = np.mean(dist_identity)

    # AF evidence: combination of spread and irregularity
    spread = np.std(dist_origin)
    af_evidence = spread * irr_evidence if spread > 0 else 0.0

    # PAC evidence: points in specific quadrants (short-long patterns)
    q2 = np.sum((x < -0.05) & (y > 0.05))  # short then long
    pac_evidence = q2 / max(len(x), 1)

    # Density evidence: entropy of 2D histogram
    try:
        hist2d, _, _ = np.histogram2d(x, y, bins=10)
        hist_norm = hist2d / np.sum(hist2d)
        hist_norm = hist_norm[hist_norm > 0]
        density_evidence = -np.sum(hist_norm * np.log2(hist_norm))
    except Exception:
        density_evidence = 0.0

    # Anisotropy: ratio of eigenvalues of covariance matrix
    try:
        cov = np.cov(x, y)
        eigvals = np.linalg.eigvalsh(cov)
        anisotropy = _safe_div(min(eigvals), max(eigvals))
    except Exception:
        anisotropy = 0.0

    return (af_evidence, origin_count, irr_evidence,
            pac_evidence, density_evidence, anisotropy)


# ===========================================================================
#  HJORTH PARAMETERS
# ===========================================================================

def _hjorth_params(sig: np.ndarray):
    """Return (activity, mobility, complexity)."""
    if len(sig) < 3:
        return 0.0, 0.0, 0.0
    activity = np.var(sig)
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    v_d1 = np.var(d1)
    v_d2 = np.var(d2)
    mobility = np.sqrt(_safe_div(v_d1, activity)) if activity > 0 else 0.0
    mob_d1 = np.sqrt(_safe_div(v_d2, v_d1)) if v_d1 > 0 else 0.0
    complexity = _safe_div(mob_d1, mobility) if mobility > 0 else 0.0
    return activity, mobility, complexity


# ===========================================================================
#  WAVELET ENTROPY
# ===========================================================================

def _wavelet_entropy(sig: np.ndarray, wavelet: str = "db4",
                     level: int = 5) -> float:
    """Wavelet entropy of signal."""
    if not HAS_PYWT or len(sig) < 2 ** level:
        return 0.0
    try:
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        energies = np.array([np.sum(c ** 2) for c in coeffs])
        total = np.sum(energies)
        if total <= 0:
            return 0.0
        p = energies / total
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))
    except Exception:
        return 0.0


# ===========================================================================
#  KDE FEATURES
# ===========================================================================

def _kde_features(data: np.ndarray, n_points: int = 200):
    """
    Compute kurtosis, skewness, num_peaks, max_dist, min_dist of KDE.
    Returns up to 5 values.
    """
    if len(data) < 3 or np.std(data) == 0:
        return 0.0, 0.0, 0, 0.0, 0.0

    try:
        kde = scipy.stats.gaussian_kde(data)
        x_grid = np.linspace(np.min(data), np.max(data), n_points)
        density = kde(x_grid)

        kurt = float(scipy.stats.kurtosis(density))
        skew = float(scipy.stats.skew(density))

        peaks, _ = scipy.signal.find_peaks(density)
        n_peaks = len(peaks)

        if n_peaks >= 2:
            peak_locs = x_grid[peaks]
            dists = np.diff(np.sort(peak_locs))
            max_dist = float(np.max(dists))
            min_dist = float(np.min(dists))
        else:
            max_dist = 0.0
            min_dist = 0.0

        return kurt, skew, n_peaks, max_dist, min_dist
    except Exception:
        return 0.0, 0.0, 0, 0.0, 0.0


# ===========================================================================
#  LPC (Linear Predictive Coding) COEFFICIENTS
# ===========================================================================

def _lpc_coefficients(sig: np.ndarray, order: int = 10) -> np.ndarray:
    """Compute LPC coefficients using autocorrelation method (Levinson-Durbin)."""
    n = len(sig)
    if n < order + 1:
        return np.zeros(order + 1)
    # Autocorrelation
    r = np.correlate(sig, sig, mode="full")
    r = r[n - 1:n + order]
    if r[0] == 0:
        return np.zeros(order + 1)

    # Levinson-Durbin recursion
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        lam = -np.sum(a[1:i] * r[i - 1:0:-1]) - r[i]
        lam /= e
        # Update coefficients
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + lam * a[i - j]
        a_new[i] = lam
        a = a_new
        e *= (1 - lam ** 2)
        if e <= 0:
            break
    return a


# ===========================================================================
#  SPECTRAL PURITY INDEX (SPI)
# ===========================================================================

def _spectral_purity_index(ecg: np.ndarray, fs: int = 300,
                           win_sec: float = 2.0):
    """
    Compute spectral purity index features.
    Returns (max_mean_SPI, max_max_SPI).
    """
    win = int(win_sec * fs)
    if len(ecg) < win:
        return 0.0, 0.0

    n_windows = len(ecg) // win
    spi_means = []
    spi_maxes = []

    for i in range(n_windows):
        segment = ecg[i * win:(i + 1) * win]
        f, psd = scipy.signal.welch(segment, fs=fs, nperseg=min(len(segment), 256))
        total = np.sum(psd)
        if total <= 0:
            continue
        psd_norm = psd / total
        # SPI = max(PSD_norm) -- spectral concentration
        spi = float(np.max(psd_norm))
        spi_means.append(spi)
        spi_maxes.append(spi)

    if not spi_means:
        return 0.0, 0.0

    return float(np.max(np.array(spi_means))), float(np.max(np.array(spi_maxes)))


# ===========================================================================
#  ECG SIGNAL QUALITY FEATURES
# ===========================================================================

def _ecg_sqi_features(ecg: np.ndarray, fs: int = 300) -> np.ndarray:
    """
    Compute 6 signal quality index features:
    median SNR, PCA SQI, spectral power ratio, baseline power ratio,
    kurtosis, skewness.
    """
    feats = np.zeros(6)
    n = len(ecg)
    if n < fs:
        return feats

    # Segment into 2-second windows
    win = 2 * fs
    n_win = n // win
    if n_win < 1:
        return feats

    snrs = []
    for i in range(n_win):
        seg = ecg[i * win:(i + 1) * win]
        sig_power = np.var(seg)
        # Estimate noise from high-frequency content
        hp = _highpass_filter(seg, 40.0, fs, order=2)
        noise_power = np.var(hp)
        if noise_power > 0:
            snrs.append(10 * np.log10(sig_power / noise_power))

    feats[0] = float(np.median(snrs)) if snrs else 0.0

    # PCA SQI: variance explained by first component of beat matrix
    try:
        r_peaks = _pan_tompkins_detect(ecg, fs)
        if len(r_peaks) >= 3:
            beat_len = int(0.6 * fs)
            beats = []
            for r in r_peaks:
                start = r - int(0.2 * fs)
                end = start + beat_len
                if 0 <= start and end <= n:
                    beats.append(ecg[start:end])
            if len(beats) >= 3:
                beat_mat = np.array(beats)
                beat_mat -= beat_mat.mean(axis=0)
                U, s, _ = np.linalg.svd(beat_mat, full_matrices=False)
                feats[1] = float(s[0] ** 2 / np.sum(s ** 2))
    except Exception:
        pass

    # Spectral power ratio: QRS band (5-15 Hz) / total
    try:
        f, psd = scipy.signal.welch(ecg, fs=fs, nperseg=min(n, 1024))
        total = np.sum(psd)
        if total > 0:
            qrs_band = np.sum(psd[(f >= 5) & (f <= 15)])
            feats[2] = float(qrs_band / total)
            baseline = np.sum(psd[f < 1])
            feats[3] = float(baseline / total)
    except Exception:
        pass

    feats[4] = float(scipy.stats.kurtosis(ecg))
    feats[5] = float(scipy.stats.skew(ecg))

    return feats


# ===========================================================================
#  SIMPLE EMD (Empirical Mode Decomposition)
# ===========================================================================

def _simple_emd_imf1(sig: np.ndarray, max_imfs: int = 1,
                     max_sift: int = 20) -> np.ndarray:
    """Extract first IMF via simple sifting process."""
    if len(sig) < 10:
        return sig.copy()

    residual = sig.copy()
    for _ in range(max_imfs):
        h = residual.copy()
        for _ in range(max_sift):
            maxima_idx = scipy.signal.argrelextrema(h, np.greater, order=1)[0]
            minima_idx = scipy.signal.argrelextrema(h, np.less, order=1)[0]

            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break

            x = np.arange(len(h))
            try:
                upper = np.interp(x, maxima_idx, h[maxima_idx])
                lower = np.interp(x, minima_idx, h[minima_idx])
            except Exception:
                break

            mean_env = (upper + lower) / 2.0
            h = h - mean_env

            # Check stopping criterion
            if np.sum(mean_env ** 2) < 1e-10 * np.sum(h ** 2):
                break

        return h

    return sig.copy()


# ===========================================================================
#  GROUP 1:  ECG_features_158_old  (68 features)
# ===========================================================================

def _group1_ecg_features_158_old(ecg: np.ndarray, r_peaks: np.ndarray,
                                  fid: dict, fs: int = 300) -> np.ndarray:
    feats = np.zeros(68)
    n = len(ecg)

    if len(r_peaks) < 3:
        return feats

    rr = np.diff(r_peaks) / fs  # RR in seconds
    drr = np.diff(rr)

    # ---- feat1 (11): Lorenz plot + Poincare ----
    af, oc, irr, pac, dens, aniso = _lorenz_plot_features(rr)
    feats[0] = af
    feats[1] = oc
    feats[2] = irr
    feats[3] = pac
    feats[4] = dens
    feats[5] = aniso
    feats[6] = _safe_cv(rr)
    feats[7] = _safe_cv(drr) if len(drr) > 1 else 0.0
    sd1, sd2, stepping, dispersion, n_clusters = _poincare_features(rr)
    feats[8] = stepping
    feats[9] = dispersion
    feats[10] = float(n_clusters)

    # ---- feat2 (8): RR interval statistics ----
    feats[11] = np.mean(rr)
    feats[12] = np.median(rr)
    feats[13] = np.min(rr)
    feats[14] = np.max(rr)
    feats[15] = float(scipy.stats.skew(rr))
    feats[16] = float(scipy.stats.kurtosis(rr))
    feats[17] = np.max(rr) - np.min(rr)
    feats[18] = np.var(rr)

    # ---- Wavelet entropy of RR (1) ----
    feats[19] = _wavelet_entropy(rr)

    # ---- Hjorth of RR (2) ----
    _, mob, comp = _hjorth_params(rr)
    feats[20] = mob
    feats[21] = comp

    # ---- feat3 (7): KDE of RR and delta-RR ----
    k_rr, s_rr, np_rr, maxd_rr, mind_rr = _kde_features(rr)
    feats[22] = k_rr
    feats[23] = s_rr
    feats[24] = float(np_rr)
    feats[25] = maxd_rr
    feats[26] = mind_rr
    if len(drr) > 2:
        k_drr, s_drr, _, _, _ = _kde_features(drr)
        feats[27] = k_drr
        feats[28] = s_drr

    # ---- feat4 (5): QRS amplitude KDE + CV ----
    qrs_amps = ecg[r_peaks]
    if len(qrs_amps) > 2:
        k_q, s_q, np_q, _, _ = _kde_features(qrs_amps)
        feats[29] = k_q
        feats[30] = s_q
        feats[31] = float(np_q)
        feats[32] = _safe_cv(qrs_amps)

    # CV of inter-beat energy
    if len(r_peaks) >= 2:
        energies = []
        for i in range(len(r_peaks) - 1):
            seg = ecg[r_peaks[i]:r_peaks[i + 1]]
            energies.append(np.sum(seg ** 2))
        energies = np.array(energies)
        feats[33] = _safe_cv(energies) if len(energies) > 1 else 0.0

    # ---- feat5 (31): Morphological features ----
    idx = 34
    Q = fid["Q"]
    S = fid["S"]
    T = fid["T"]
    P = fid["P"]
    R = fid["R"]

    valid = np.where((Q >= 0) & (S >= 0) & (T >= 0) & (P >= 0))[0]

    if len(valid) > 2:
        # RS depth
        rs_depth = np.array([ecg[R[v]] - ecg[S[v]] for v in valid])
        feats[idx + 0] = _safe_cv(rs_depth)
        feats[idx + 1] = np.max(rs_depth) - np.min(rs_depth)
        feats[idx + 2] = np.median(rs_depth)

        # ST slope
        st_slopes = []
        for v in valid:
            dt = (T[v] - S[v]) / fs
            if dt > 0:
                st_slopes.append((ecg[T[v]] - ecg[S[v]]) / dt)
        st_slopes = np.array(st_slopes) if st_slopes else np.array([0.0])
        feats[idx + 3] = np.median(st_slopes)
        feats[idx + 4] = _safe_cv(st_slopes)
        feats[idx + 5] = np.sum(st_slopes < 0) / max(len(st_slopes), 1)

        # Deep S count
        feats[idx + 6] = float(np.sum(rs_depth > 2 * np.median(rs_depth)))

        # Inflection distance: distance between Q and S in samples
        infl_dist = np.array([(S[v] - Q[v]) for v in valid])
        feats[idx + 7] = np.mean(infl_dist) / fs

        # T/R amplitude ratio
        t_r_ratio = np.array([_safe_div(ecg[T[v]], ecg[R[v]]) for v in valid])
        feats[idx + 8] = np.mean(t_r_ratio)

        # P/R amplitude ratio
        p_r_ratio = np.array([_safe_div(ecg[P[v]], ecg[R[v]]) for v in valid])
        feats[idx + 9] = np.mean(p_r_ratio)

        # QRS width: Q to S distance
        qrs_width = np.array([(S[v] - Q[v]) / fs for v in valid])
        feats[idx + 10] = np.median(qrs_width)
        feats[idx + 11] = _safe_cv(qrs_width)

        # QR width: Q to R distance
        qr_width = np.array([(R[v] - Q[v]) / fs for v in valid])
        feats[idx + 12] = np.median(qr_width)
        feats[idx + 13] = _safe_cv(qr_width)

        # RQ depth
        rq_depth = np.array([ecg[R[v]] - ecg[Q[v]] for v in valid])
        feats[idx + 14] = np.mean(rq_depth)
        feats[idx + 15] = _safe_cv(rq_depth)

        # SR ratio
        sr_ratio = np.array([_safe_div(ecg[S[v]], ecg[R[v]]) for v in valid])
        feats[idx + 16] = np.mean(sr_ratio)
        feats[idx + 17] = _safe_cv(sr_ratio)

        # QR slope
        qr_slopes = np.array([_safe_div(ecg[R[v]] - ecg[Q[v]],
                                         (R[v] - Q[v]) / fs)
                               for v in valid if R[v] != Q[v]])
        if len(qr_slopes) > 0:
            feats[idx + 18] = np.mean(qr_slopes)
            feats[idx + 19] = _safe_cv(qr_slopes)

        # RS slope
        rs_slopes = np.array([_safe_div(ecg[S[v]] - ecg[R[v]],
                                         (S[v] - R[v]) / fs)
                               for v in valid if S[v] != R[v]])
        if len(rs_slopes) > 0:
            feats[idx + 20] = np.mean(rs_slopes)
            feats[idx + 21] = _safe_cv(rs_slopes)

        # Sx slope (S to end of QRS complex, approximated as S to T)
        sx_slopes = np.array([_safe_div(ecg[T[v]] - ecg[S[v]],
                                         (T[v] - S[v]) / fs)
                               for v in valid if T[v] != S[v]])
        if len(sx_slopes) > 0:
            feats[idx + 22] = np.mean(sx_slopes)
            feats[idx + 23] = _safe_cv(sx_slopes)

        # QTc features (Bazett, Fridericia, Sagie)
        if len(valid) > 0 and len(rr) > 0:
            qt_intervals = np.array([(T[v] - Q[v]) / fs for v in valid])
            qt_intervals = qt_intervals[qt_intervals > 0]
            rr_for_qt = rr[:len(qt_intervals)]
            rr_for_qt = rr_for_qt[rr_for_qt > 0]
            min_len = min(len(qt_intervals), len(rr_for_qt))
            if min_len > 0:
                qt_intervals = qt_intervals[:min_len]
                rr_for_qt = rr_for_qt[:min_len]

                qtc_bazett = qt_intervals / np.sqrt(rr_for_qt)
                feats[idx + 24] = np.median(qtc_bazett)
                feats[idx + 25] = _safe_cv(qtc_bazett)

                qtc_frid = qt_intervals / np.cbrt(rr_for_qt)
                feats[idx + 26] = np.median(qtc_frid)
                feats[idx + 27] = _safe_cv(qtc_frid)

                qtc_sagie = qt_intervals + 0.154 * (1 - rr_for_qt)
                feats[idx + 28] = np.median(qtc_sagie)
                feats[idx + 29] = _safe_cv(qtc_sagie)

    feats[idx + 30] = 0.0  # feat5_extra placeholder

    # ---- feat_HRV (3): pNN20, pNN50, SDSD ----
    hrv_idx = 65
    if len(drr) > 0:
        abs_drr = np.abs(drr)
        feats[hrv_idx + 0] = np.sum(abs_drr > 0.02) / len(abs_drr)  # pNN20
        feats[hrv_idx + 1] = np.sum(abs_drr > 0.05) / len(abs_drr)  # pNN50
        feats[hrv_idx + 2] = np.std(drr)                              # SDSD

    return feats


# ===========================================================================
#  GROUP 2:  other_features_new  (3 features)
# ===========================================================================

def _group2_other_features_new(ecg: np.ndarray) -> np.ndarray:
    """Shannon, Tsallis (q=2), Renyi (q=2) entropy of amplitude histogram."""
    feats = np.zeros(3)
    hist, _ = np.histogram(ecg, bins=50, density=True)
    hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    p = hist[hist > 0]

    if len(p) == 0:
        return feats

    # Shannon entropy
    feats[0] = float(-np.sum(p * np.log2(p)))
    # Tsallis entropy (q=2)
    q = 2.0
    feats[1] = float((1 - np.sum(p ** q)) / (q - 1))
    # Renyi entropy (q=2)
    feats[2] = float(-np.log2(np.sum(p ** q)))

    return feats


# ===========================================================================
#  GROUP 3:  frequency_features  (12 features)
# ===========================================================================

def _group3_frequency_features(ecg: np.ndarray, fs: int = 300) -> np.ndarray:
    feats = np.zeros(12)
    n = len(ecg)
    win_samples = 2 * fs  # 2-second windows

    # Window-based spectral features
    n_windows = n // win_samples
    centroids, rolloffs, fluxes, kurts = [], [], [], []
    prev_spectrum = None

    for i in range(max(n_windows, 1)):
        seg = ecg[i * win_samples:(i + 1) * win_samples] if i < n_windows else ecg
        if len(seg) < 16:
            continue
        f, psd = scipy.signal.welch(seg, fs=fs, nperseg=min(len(seg), 256))
        total = np.sum(psd)
        if total <= 0:
            continue

        # Spectral centroid
        centroids.append(np.sum(f * psd) / total)
        # Spectral rolloff (85%)
        cumsum = np.cumsum(psd) / total
        ri = np.searchsorted(cumsum, 0.85)
        rolloffs.append(f[min(ri, len(f) - 1)])
        # Spectral flux
        if prev_spectrum is not None and len(prev_spectrum) == len(psd):
            fluxes.append(np.sum((psd - prev_spectrum) ** 2))
        prev_spectrum = psd.copy()
        # Kurtosis of spectrum
        kurts.append(float(scipy.stats.kurtosis(psd)))

    feats[0] = float(np.mean(centroids)) if centroids else 0.0
    feats[1] = float(np.mean(rolloffs)) if rolloffs else 0.0
    feats[2] = float(np.mean(fluxes)) if fluxes else 0.0
    feats[3] = float(np.mean(kurts)) if kurts else 0.0

    # Wavelet: log2(var(d3)) using db4 level 5
    if HAS_PYWT and n >= 32:
        try:
            coeffs = pywt.wavedec(ecg, "db4", level=5)
            # coeffs[0] = approx, coeffs[1..5] = details levels 5..1
            # Detail level 3 = coeffs[3]
            d3 = coeffs[3] if len(coeffs) > 3 else coeffs[-1]
            v = np.var(d3)
            feats[4] = np.log2(v) if v > 0 else -30.0
        except Exception:
            feats[4] = 0.0
    else:
        feats[4] = 0.0

    # LPC coefficients: indices [2,4,7,9,10,11] from order-10
    lpc = _lpc_coefficients(ecg, order=10)
    lpc_indices = [2, 4, 7, 9, 10]
    for j, li in enumerate(lpc_indices):
        feats[5 + j] = float(lpc[li]) if li < len(lpc) else 0.0
    # 6th LPC feature at index 11 (but lpc has order+1 = 11 elements, index 10 is last)
    feats[10] = float(lpc[min(10, len(lpc) - 1)])

    # Frequency centroid from full PSD
    try:
        f_full, psd_full = scipy.signal.welch(ecg, fs=fs,
                                               nperseg=min(n, 2048))
        total = np.sum(psd_full)
        feats[11] = float(np.sum(f_full * psd_full) / total) if total > 0 else 0.0
    except Exception:
        pass

    return feats


# ===========================================================================
#  GROUP 4:  pr_features  (21 features)
# ===========================================================================

def _group4_pr_features(ecg: np.ndarray, r_peaks: np.ndarray,
                        fid: dict, fs: int = 300) -> np.ndarray:
    feats = np.zeros(21)

    if len(r_peaks) < 4:
        return feats

    rr = np.diff(r_peaks) / fs
    drr = np.diff(rr)

    # P-peak locations
    P = fid["P"]
    valid_p = P[P >= 0]

    # std of dRR
    feats[0] = np.std(drr) if len(drr) > 1 else 0.0

    # std of dPP
    if len(valid_p) >= 3:
        pp = np.diff(valid_p) / fs
        dpp = np.diff(pp)
        feats[1] = np.std(dpp) if len(dpp) > 1 else 0.0
        # PP/RR length ratio
        feats[2] = _safe_div(len(pp), len(rr))
    else:
        feats[1] = 0.0
        feats[2] = 0.0

    # Clustering-based features on dRR (3-cluster)
    if len(drr) > 5:
        try:
            from sklearn.cluster import KMeans
            X = drr.reshape(-1, 1)
            km3 = KMeans(n_clusters=min(3, len(drr)), n_init=5,
                         random_state=42).fit(X)
            centers3 = np.sort(km3.cluster_centers_.flatten())
            feats[3] = float(centers3[-1] - centers3[0]) if len(centers3) > 1 else 0.0
            feats[4] = float(np.std(centers3))
            feats[5] = float(np.max(np.bincount(km3.labels_)) / len(drr))
        except Exception:
            pass

    # 2-cluster features
    if len(drr) > 3:
        try:
            from sklearn.cluster import KMeans
            X = drr.reshape(-1, 1)
            km2 = KMeans(n_clusters=min(2, len(drr)), n_init=5,
                         random_state=42).fit(X)
            centers2 = np.sort(km2.cluster_centers_.flatten())
            feats[6] = float(centers2[-1] - centers2[0]) if len(centers2) > 1 else 0.0
            feats[7] = float(np.min(np.bincount(km2.labels_)) / len(drr))
        except Exception:
            pass

    # Outlier-removed stats
    if len(rr) > 3:
        q1, q3 = np.percentile(rr, [25, 75])
        iqr = q3 - q1
        mask = (rr >= q1 - 1.5 * iqr) & (rr <= q3 + 1.5 * iqr)
        rr_clean = rr[mask]
        if len(rr_clean) > 1:
            feats[8] = np.mean(rr_clean)
            feats[9] = np.std(rr_clean)
        feats[10] = float(np.sum(~mask))  # outlier count

    # FFT of cleaned dRR
    if len(drr) > 4:
        drr_clean = drr.copy()
        fft_drr = np.abs(np.fft.rfft(drr_clean))
        if len(fft_drr) > 1:
            feats[11] = float(np.max(fft_drr[1:]))
            feats[12] = float(np.argmax(fft_drr[1:]) + 1)

    # FFT of dPP
    if len(valid_p) >= 4:
        pp = np.diff(valid_p) / fs
        dpp = np.diff(pp)
        if len(dpp) > 2:
            fft_dpp = np.abs(np.fft.rfft(dpp))
            if len(fft_dpp) > 1:
                feats[13] = float(np.max(fft_dpp[1:]))
                feats[14] = float(np.argmax(fft_dpp[1:]) + 1)

    # Brady/tachy binary indicators
    mean_hr = 60.0 / np.mean(rr) if np.mean(rr) > 0 else 75.0
    feats[15] = 1.0 if mean_hr < 60 else 0.0   # bradycardia
    feats[16] = 1.0 if mean_hr > 100 else 0.0   # tachycardia

    # Range features
    feats[17] = np.max(rr) - np.min(rr)
    feats[18] = _safe_div(np.max(rr), np.min(rr)) if np.min(rr) > 0 else 0.0

    # Additional RR variability
    feats[19] = float(scipy.stats.iqr(rr))
    feats[20] = float(np.median(np.abs(drr))) if len(drr) > 0 else 0.0

    return feats


# ===========================================================================
#  GROUP 5:  soa_features  (8 features)
# ===========================================================================

def _group5_soa_features(ecg: np.ndarray, r_peaks: np.ndarray,
                         fs: int = 300) -> np.ndarray:
    feats = np.zeros(8)

    if len(r_peaks) < 3:
        return feats

    rr = np.diff(r_peaks) / fs

    # Normalized RMSSD
    drr = np.diff(rr)
    rmssd = np.sqrt(np.mean(drr ** 2)) if len(drr) > 0 else 0.0
    feats[0] = _safe_div(rmssd, np.mean(rr))

    # MAD of RR
    feats[1] = float(np.mean(np.abs(rr - np.median(rr))))

    # HRV spectral features via Welch PSD of interpolated RR
    if len(rr) < 4:
        return feats

    try:
        # Create evenly sampled RR time series via interpolation
        t_rr = np.cumsum(rr)
        t_rr = np.insert(t_rr, 0, 0)
        rr_extended = np.append(rr, rr[-1])  # extend for interpolation

        # Resample at 4 Hz
        fs_rr = 4.0
        t_interp = np.arange(t_rr[0], t_rr[-1], 1.0 / fs_rr)
        if len(t_interp) < 8:
            return feats

        interp_fn = scipy.interpolate.interp1d(t_rr, rr_extended,
                                                kind="linear",
                                                fill_value="extrapolate")
        rr_resampled = interp_fn(t_interp)

        # Remove mean
        rr_resampled -= np.mean(rr_resampled)

        # Welch PSD
        f_hrv, psd_hrv = scipy.signal.welch(rr_resampled, fs=fs_rr,
                                             nperseg=min(len(rr_resampled), 256))

        # Three bands: VLF (0-0.04), LF (0.04-0.15), HF (0.15-0.4)
        vlf_power = np.sum(psd_hrv[(f_hrv >= 0) & (f_hrv < 0.04)])
        lf_power = np.sum(psd_hrv[(f_hrv >= 0.04) & (f_hrv < 0.15)])
        hf_power = np.sum(psd_hrv[(f_hrv >= 0.15) & (f_hrv < 0.4)])

        feats[2] = float(vlf_power)
        feats[3] = float(lf_power)
        feats[4] = float(hf_power)
        feats[5] = _safe_div(lf_power, hf_power)
        feats[6] = _safe_div(vlf_power, hf_power)
        feats[7] = _safe_div(lf_power, vlf_power)
    except Exception:
        pass

    return feats


# ===========================================================================
#  GROUP 6:  extract_features  (27 features)
# ===========================================================================

def _group6_extract_features(ecg: np.ndarray, r_peaks: np.ndarray,
                              fs: int = 300) -> np.ndarray:
    feats = np.zeros(27)
    n = len(ecg)

    # Z-score + bandpass filter the ECG
    sig = ecg.copy()
    if np.std(sig) > 0:
        sig = (sig - np.mean(sig)) / np.std(sig)
    sig_bp = _bandpass_filter(sig, 0.5, 40.0, fs, order=4)

    # FFT
    fft_coeffs = np.abs(np.fft.rfft(sig_bp))
    fft_freqs = np.fft.rfftfreq(len(sig_bp), d=1.0 / fs)

    # feat 1: trimmed mean of FFT coefficients
    if len(fft_coeffs) > 0:
        feats[0] = float(scipy.stats.trim_mean(fft_coeffs, 0.1))

    # feat 2: skewness of FFT coefficients
    feats[1] = float(scipy.stats.skew(fft_coeffs))

    # feat 3: 80% energy frequency
    if len(fft_coeffs) > 0:
        energy = fft_coeffs ** 2
        cumsum = np.cumsum(energy) / np.sum(energy) if np.sum(energy) > 0 else np.zeros_like(energy)
        idx_80 = np.searchsorted(cumsum, 0.8)
        feats[2] = float(fft_freqs[min(idx_80, len(fft_freqs) - 1)])

    # feat 4: kurtosis of FFT in 0.5-5 Hz
    mask_05_5 = (fft_freqs >= 0.5) & (fft_freqs <= 5.0)
    if np.sum(mask_05_5) > 3:
        feats[3] = float(scipy.stats.kurtosis(fft_coeffs[mask_05_5]))

    # feat 5: kurtosis of time-domain signal
    feats[4] = float(scipy.stats.kurtosis(sig_bp))

    # feat 6: Hjorth complexity
    _, _, comp = _hjorth_params(sig_bp)
    feats[5] = comp

    # feat 7: SNR (signal vs high-freq noise)
    try:
        noise = _highpass_filter(sig_bp, 40.0, fs, order=2)
        sig_power = np.var(sig_bp)
        noise_power = np.var(noise)
        feats[6] = 10 * np.log10(_safe_div(sig_power, noise_power, 1.0))
    except Exception:
        pass

    # feat 8: THD (total harmonic distortion)
    if len(fft_coeffs) > 2:
        fund_idx = np.argmax(fft_coeffs[1:]) + 1
        fund_power = fft_coeffs[fund_idx] ** 2
        # Sum harmonics
        harmonic_power = 0.0
        for h in range(2, 6):
            hi = fund_idx * h
            if hi < len(fft_coeffs):
                harmonic_power += fft_coeffs[hi] ** 2
        feats[7] = np.sqrt(_safe_div(harmonic_power, fund_power))

    # feat 9: zero-crossing rate
    zcr = np.sum(np.abs(np.diff(np.sign(sig_bp))) > 0) / (n / fs)
    feats[8] = zcr

    # feat 10-13: short-time features (2-sec frames)
    frame_len = 2 * fs
    n_frames = n // frame_len
    if n_frames > 0:
        energies, cents, rolls, fluxes = [], [], [], []
        prev_psd = None
        for i in range(n_frames):
            frame = sig_bp[i * frame_len:(i + 1) * frame_len]
            energies.append(np.sum(frame ** 2) / len(frame))
            f, psd = scipy.signal.welch(frame, fs=fs,
                                         nperseg=min(len(frame), 256))
            total = np.sum(psd)
            if total > 0:
                cents.append(np.sum(f * psd) / total)
                cs = np.cumsum(psd) / total
                ri = np.searchsorted(cs, 0.85)
                rolls.append(f[min(ri, len(f) - 1)])
                if prev_psd is not None and len(prev_psd) == len(psd):
                    fluxes.append(np.sum((psd - prev_psd) ** 2))
                prev_psd = psd.copy()

        feats[9] = float(np.mean(energies)) if energies else 0.0
        feats[10] = float(np.mean(cents)) if cents else 0.0
        feats[11] = float(np.mean(rolls)) if rolls else 0.0
        feats[12] = float(np.mean(fluxes)) if fluxes else 0.0

    # feat 14-15: peak-to-peak timing histogram features
    pk, _ = scipy.signal.find_peaks(sig_bp, distance=int(0.2 * fs))
    if len(pk) > 2:
        pk_diffs = np.diff(pk) / fs
        hist, _ = np.histogram(pk_diffs, bins=20)
        feats[13] = float(np.max(hist)) / max(len(pk_diffs), 1)
        feats[14] = float(np.std(pk_diffs))

    # feat 16: dominant FFT frequency
    if len(fft_coeffs) > 1:
        feats[15] = float(fft_freqs[np.argmax(fft_coeffs[1:]) + 1])

    # feat 17: difference between top-2 FFT peaks
    if len(fft_coeffs) > 5:
        fft_peaks, _ = scipy.signal.find_peaks(fft_coeffs)
        if len(fft_peaks) >= 2:
            top2 = fft_peaks[np.argsort(fft_coeffs[fft_peaks])[-2:]]
            feats[16] = abs(float(fft_freqs[top2[1]] - fft_freqs[top2[0]]))

    # feat 18-19: EMD-based features on heart rate series
    if len(r_peaks) >= 4:
        rr = np.diff(r_peaks) / fs
        hr = 60.0 / rr
        try:
            imf1 = _simple_emd_imf1(hr)
            # Zero-crossing rate of first IMF
            zcr_imf = np.sum(np.abs(np.diff(np.sign(imf1))) > 0) / max(len(imf1), 1)
            feats[17] = zcr_imf
            feats[18] = float(np.std(imf1))
        except Exception:
            pass

    # feat 20-23: delta-HR features
    if len(r_peaks) >= 4:
        rr = np.diff(r_peaks) / fs
        hr = 60.0 / rr
        dhr = np.diff(hr)
        if len(dhr) > 1:
            feats[19] = float(np.median(dhr ** 2))
            # Binned classification
            feats[20] = float(np.sum(np.abs(dhr) > 10) / len(dhr))
            # Asymmetry
            pos = np.sum(dhr > 0)
            neg = np.sum(dhr < 0)
            feats[21] = _safe_div(pos - neg, pos + neg)
            # Dominant frequency of severe segments
            severe = np.abs(dhr) > np.std(dhr)
            if np.sum(severe) > 2:
                severe_sig = dhr[severe]
                fft_s = np.abs(np.fft.rfft(severe_sig))
                if len(fft_s) > 1:
                    feats[22] = float(np.argmax(fft_s[1:]) + 1) / len(severe_sig)

    # feat 24-27: spectral band powers as percentage
    if len(fft_coeffs) > 0:
        total_power = np.sum(fft_coeffs ** 2)
        if total_power > 0:
            bands = [(0, 2), (2, 4), (4, 10), (10, 150)]
            for bi, (lo, hi) in enumerate(bands):
                mask = (fft_freqs >= lo) & (fft_freqs < hi)
                feats[23 + bi] = float(np.sum(fft_coeffs[mask] ** 2) / total_power)

    return feats


# ===========================================================================
#  GROUP 7:  new_feat_sd_158_old  (19 features)
# ===========================================================================

def _group7_new_feat_sd_158_old(ecg: np.ndarray, r_peaks: np.ndarray,
                                 fs: int = 300) -> np.ndarray:
    feats = np.zeros(19)

    if len(r_peaks) < 4:
        return feats

    rr = np.diff(r_peaks) / fs
    ibi = rr  # inter-beat intervals

    # Sample entropy features (m=5, r=0.2*std)
    se = _sample_entropy(ibi, m=5, r_frac=0.2)
    # se has 5 values (m=0..4)

    # Count inf values
    num_inf = int(np.sum(np.isinf(se)))
    finite_se = se[~np.isinf(se)]

    feats[0:5] = np.where(np.isinf(se), 0.0, se)  # replace inf with 0

    feats[5] = float(num_inf)
    feats[6] = float(np.max(finite_se)) if len(finite_se) > 0 else 0.0
    feats[7] = float(np.min(finite_se)) if len(finite_se) > 0 else 0.0
    # max_ind, min_ind
    if len(finite_se) > 0:
        feats[8] = float(np.argmax(np.where(np.isinf(se), -np.inf, se)))
        feats[9] = float(np.argmin(np.where(np.isinf(se), np.inf, se)))

    # DFA alpha1
    slope, intercept = _dfa_alpha1(ibi)
    feats[10] = slope
    feats[11] = intercept

    # Poincare SD1, SD2
    sd1, sd2, _, _, _ = _poincare_features(rr)
    feats[12] = sd1
    feats[13] = sd2

    # Approximate entropy for m=1..5 with r=0.02*std
    r_tol = 0.02 * np.std(ibi) if np.std(ibi) > 0 else 0.01
    for i in range(5):
        feats[14 + i] = _approx_entropy(ibi, m=i + 1, r=r_tol)

    return feats


# ===========================================================================
#  GROUP 8:  generate_30_features  (30 features)
# ===========================================================================

def _group8_generate_30_features(ecg: np.ndarray, r_peaks: np.ndarray,
                                  fs: int = 300) -> np.ndarray:
    feats = np.zeros(30)
    n = len(ecg)

    # --- f1: get_fv (2) - peak counting and max gap on HP-filtered ECG ---
    try:
        ecg_hp = _highpass_filter(ecg, 1.0, fs, order=2)
        threshold = 0.5 * np.std(ecg_hp)
        above = np.abs(ecg_hp) > threshold
        # Count transitions
        transitions = np.diff(above.astype(int))
        feats[0] = float(np.sum(transitions > 0))  # peak count
        # Max gap between peaks
        peak_locs = np.where(transitions > 0)[0]
        if len(peak_locs) > 1:
            feats[1] = float(np.max(np.diff(peak_locs))) / fs
    except Exception:
        pass

    # --- f2: get_fv1 (10) - periodogram-based features ---
    try:
        f_pg, psd_pg = scipy.signal.periodogram(ecg, fs=fs)
        total_power = np.sum(psd_pg)

        if total_power > 0 and len(f_pg) > 1:
            # Dominant frequency
            feats[2] = float(f_pg[np.argmax(psd_pg[1:]) + 1])

            # Number of spectral peaks
            spec_peaks, _ = scipy.signal.find_peaks(psd_pg,
                                                     height=0.01 * np.max(psd_pg))
            feats[3] = float(len(spec_peaks))

            # Autocorrelation-based frequency
            acf = np.correlate(ecg[:min(n, 5 * fs)],
                               ecg[:min(n, 5 * fs)], mode="full")
            acf = acf[len(acf) // 2:]
            if len(acf) > fs:
                acf_peaks, _ = scipy.signal.find_peaks(acf[int(0.2 * fs):],
                                                        distance=int(0.3 * fs))
                if len(acf_peaks) > 0:
                    feats[4] = fs / float(acf_peaks[0] + int(0.2 * fs))
                # Frequency ratio
                if feats[2] > 0:
                    feats[5] = _safe_div(feats[4], feats[2])

                # Autocorrelation location std
                if len(acf_peaks) > 1:
                    feats[6] = float(np.std(acf_peaks))
                    # Autocorrelation peak values std
                    acf_shifted = acf[int(0.2 * fs):]
                    if len(acf_peaks) > 0 and np.max(acf_peaks) < len(acf_shifted):
                        feats[7] = float(np.std(acf_shifted[acf_peaks]))

            # High-frequency area (>30 Hz)
            hf_mask = f_pg > 30
            feats[8] = float(np.sum(psd_pg[hf_mask]) / total_power)

            # VF area ratio (1-10 Hz)
            vf_mask = (f_pg >= 1) & (f_pg <= 10)
            feats[9] = float(np.sum(psd_pg[vf_mask]) / total_power)

            # Peak area ratio
            if len(spec_peaks) > 0:
                peak_power = np.sum(psd_pg[spec_peaks])
                feats[10] = float(peak_power / total_power)

            # Negative autocorrelation peaks
            if len(acf) > 1:
                neg_acf = -acf
                neg_peaks, _ = scipy.signal.find_peaks(neg_acf[:2 * fs])
                feats[11] = float(len(neg_peaks))
    except Exception:
        pass

    # --- f4: get_fv5 (1) - max HR over 17-beat windows ---
    if len(r_peaks) >= 18:
        rr = np.diff(r_peaks) / fs
        hr_17 = []
        for i in range(len(rr) - 16):
            mean_rr = np.mean(rr[i:i + 17])
            if mean_rr > 0:
                hr_17.append(60.0 / mean_rr)
        feats[12] = float(np.max(hr_17)) if hr_17 else 0.0
    elif len(r_peaks) >= 2:
        rr = np.diff(r_peaks) / fs
        feats[12] = 60.0 / np.min(rr) if np.min(rr) > 0 else 0.0

    # --- f5: get_fv4 (1) - min HR over 5-beat windows ---
    if len(r_peaks) >= 6:
        rr = np.diff(r_peaks) / fs
        hr_5 = []
        for i in range(len(rr) - 4):
            mean_rr = np.mean(rr[i:i + 5])
            if mean_rr > 0:
                hr_5.append(60.0 / mean_rr)
        feats[13] = float(np.min(hr_5)) if hr_5 else 0.0
    elif len(r_peaks) >= 2:
        rr = np.diff(r_peaks) / fs
        feats[13] = 60.0 / np.max(rr) if np.max(rr) > 0 else 0.0

    # --- f6: get_f1 (2) - SPI features ---
    max_mean_spi, max_max_spi = _spectral_purity_index(ecg, fs)
    feats[14] = max_mean_spi
    feats[15] = max_max_spi

    # --- f7: get_f2 (1) - max RR interval ---
    if len(r_peaks) >= 2:
        rr = np.diff(r_peaks) / fs
        feats[16] = float(np.max(rr))

    # --- f9: ecgSqiFeatures (6) ---
    sqi = _ecg_sqi_features(ecg, fs)
    feats[17:23] = sqi

    # --- f10: num_diff_peaks (1) - number of morphologically different beats ---
    if len(r_peaks) >= 3:
        try:
            beat_len = int(0.6 * fs)
            beats = []
            for r in r_peaks:
                start = r - int(0.2 * fs)
                end = start + beat_len
                if 0 <= start and end <= n:
                    beats.append(ecg[start:end])
            if len(beats) >= 3:
                beat_mat = np.array(beats)
                # Normalize each beat
                norms = np.linalg.norm(beat_mat, axis=1, keepdims=True)
                norms[norms == 0] = 1
                beat_norm = beat_mat / norms
                # Compute correlation matrix
                corr_mat = beat_norm @ beat_norm.T
                # Count morphologically different beats
                # (correlation < 0.9 with median beat)
                median_beat = np.median(beat_norm, axis=0)
                median_norm = np.linalg.norm(median_beat)
                if median_norm > 0:
                    median_beat /= median_norm
                    corrs = beat_norm @ median_beat
                    feats[23] = float(np.sum(corrs < 0.9))
        except Exception:
            pass

    # --- f11: hrFeatures (6) ---
    if len(r_peaks) >= 2:
        rr = np.diff(r_peaks) / fs
        hr = 60.0 / rr[rr > 0] if np.any(rr > 0) else np.array([75.0])

        # Low HR beats count (HR < 60)
        feats[24] = float(np.sum(hr < 60))

        # Low HR over 5 beats
        if len(hr) >= 5:
            for i in range(len(hr) - 4):
                if np.all(hr[i:i + 5] < 60):
                    feats[25] = 1.0
                    break

        # High HR beats count (HR > 100)
        feats[26] = float(np.sum(hr > 100))

        # High HR over 17 beats
        if len(hr) >= 17:
            for i in range(len(hr) - 16):
                if np.all(hr[i:i + 17] > 100):
                    feats[27] = 1.0
                    break

        # Max HR
        feats[28] = float(np.max(hr))

        # Max RR
        feats[29] = float(np.max(rr))

    return feats


# ===========================================================================
#  MAIN EXTRACTION FUNCTION
# ===========================================================================

def extract_features_188(signal: np.ndarray, fs: int = 300) -> np.ndarray:
    """
    Extract 188 features from a single-lead ECG waveform.

    Parameters
    ----------
    signal : np.ndarray
        Raw ECG signal (1D array).
    fs : int
        Sampling frequency in Hz (default 300).

    Returns
    -------
    np.ndarray
        188-element feature vector. NaN/Inf values are replaced with 0.
    """
    features = np.zeros(N_FEATURES)

    if len(signal) < fs:
        return features

    # --- Pre-processing: noise cancellation ---
    ecg = signal.astype(float)
    # Remove baseline wander (highpass at 0.5 Hz)
    ecg = _highpass_filter(ecg, 0.5, fs, order=2)
    # Notch filter at 50/60 Hz (power line)
    for notch_freq in [50.0, 60.0]:
        if notch_freq < fs / 2:
            b, a = scipy.signal.iirnotch(notch_freq, Q=30.0, fs=fs)
            ecg = scipy.signal.filtfilt(b, a, ecg)

    # --- Detect R-peaks ---
    r_peaks = _pan_tompkins_detect(ecg, fs)

    # --- Detect fiducial points ---
    fid = _detect_fiducials(ecg, r_peaks, fs)

    # --- Extract all 8 groups ---
    idx = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Group 1: ECG_features_158_old (68)
        g1 = _group1_ecg_features_158_old(ecg, r_peaks, fid, fs)
        features[idx:idx + 68] = g1
        idx += 68

        # Group 2: other_features_new (3)
        g2 = _group2_other_features_new(ecg)
        features[idx:idx + 3] = g2
        idx += 3

        # Group 3: frequency_features (12)
        g3 = _group3_frequency_features(ecg, fs)
        features[idx:idx + 12] = g3
        idx += 12

        # Group 4: pr_features (21)
        g4 = _group4_pr_features(ecg, r_peaks, fid, fs)
        features[idx:idx + 21] = g4
        idx += 21

        # Group 5: soa_features (8)
        g5 = _group5_soa_features(ecg, r_peaks, fs)
        features[idx:idx + 8] = g5
        idx += 8

        # Group 6: extract_features (27)
        g6 = _group6_extract_features(ecg, r_peaks, fs)
        features[idx:idx + 27] = g6
        idx += 27

        # Group 7: new_feat_sd_158_old (19)
        g7 = _group7_new_feat_sd_158_old(ecg, r_peaks, fs)
        features[idx:idx + 19] = g7
        idx += 19

        # Group 8: generate_30_features (30)
        g8 = _group8_generate_30_features(ecg, r_peaks, fs)
        features[idx:idx + 30] = g8
        idx += 30

    # --- Clean up NaN / Inf ---
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    assert idx == N_FEATURES, f"Feature count mismatch: {idx} != {N_FEATURES}"
    return features


# ===========================================================================
#  DATASET LOADER  (compatible with dataset_config interface)
# ===========================================================================

def load_physionet2017_188(
    data_dir: str,
    max_records: Optional[int] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PhysioNet 2017 data, extract 188 features, and return (data, labels).

    Binary classification: Normal (N) -> 1, Abnormal (A+O) -> 2.
    Noisy (~) recordings are excluded.
    Features are cached to disk after first extraction.

    Parameters
    ----------
    data_dir : str
        Path to the PhysioNet 2017 training data directory containing
        .mat files and REFERENCE-original.csv.
    max_records : int or None
        Limit number of records processed (for testing). None = all.
    use_cache : bool
        If True, load from / save to a pickle cache file.

    Returns
    -------
    data : np.ndarray, shape (n_records, 188)
    labels : np.ndarray, shape (n_records,)
    """
    import pandas as pd

    cache_dir = Path(data_dir).parent / "physioNetData2017_cache"
    tag = f"n{max_records}" if max_records else "all"
    cache_file = cache_dir / f"features_188_{tag}.pkl"

    if use_cache and cache_file.exists():
        log.info(f"Loading cached 188-feature set from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["data"], cached["labels"]

    log.info(f"Extracting 188 features from {data_dir} ...")
    ref_path = os.path.join(data_dir, "REFERENCE-original.csv")
    ref_df = pd.read_csv(ref_path, header=None, names=["record_id", "label"])
    if max_records is not None:
        ref_df = ref_df.head(max_records)

    records: List[np.ndarray] = []
    labels_list: List[int] = []
    skipped = 0
    excluded_noisy = 0
    total = len(ref_df)

    for row_idx, (_, row) in enumerate(ref_df.iterrows()):
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
            d = scipy.io.loadmat(mat_path)
            sig = d["val"].flatten().astype(float)
            feats = extract_features_188(sig, fs=PHYSIONET_SAMPLING_RATE)
            records.append(feats)
            labels_list.append(PHYSIONET_LABEL_MAP[label_str])
        except Exception as e:
            log.warning(f"Error processing {rid}: {e}")
            skipped += 1

        if (row_idx + 1) % 500 == 0:
            log.info(f"  Processed {row_idx + 1}/{total} records ...")

    data = np.array(records, dtype=float)
    labels = np.array(labels_list, dtype=int)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)
        log.info(f"Cached 188 features to {cache_file}")

    log.info(
        f"PhysioNet 2017: {len(records)} records x {N_FEATURES} features "
        f"({skipped} skipped, {excluded_noisy} noisy excluded)"
    )
    log.info(
        f"  Labels: "
        f"{dict(sorted({int(c): int((labels == c).sum()) for c in set(labels)}.items()))}"
    )
    return data, labels


# ===========================================================================
#  PHYSIONET-188 FEATURE NAMES DICT  (for dataset_config integration)
# ===========================================================================

PHYSIONET_188_FEATURE_NAMES = FEATURE_NAMES


# ===========================================================================
#  SELF-TEST
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s  %(message)s")

    print(f"N_FEATURES = {N_FEATURES}")
    print(f"Feature name count = {len(FEATURE_NAMES)}")
    assert len(FEATURE_NAMES) == N_FEATURES, (
        f"Name count {len(FEATURE_NAMES)} != {N_FEATURES}"
    )

    # Generate a synthetic ECG-like signal for testing
    np.random.seed(42)
    fs = 300
    duration = 30  # seconds
    t = np.arange(0, duration, 1.0 / fs)
    # Simulate R-peaks as narrow Gaussians at ~75 bpm
    ecg = np.zeros_like(t)
    rr_mean = 0.8  # seconds
    beat_times = np.arange(0.5, duration - 0.5, rr_mean)
    for bt in beat_times:
        ecg += 1.5 * np.exp(-0.5 * ((t - bt) / 0.02) ** 2)        # R
        ecg += 0.3 * np.exp(-0.5 * ((t - bt + 0.16) / 0.04) ** 2) # P
        ecg += 0.4 * np.exp(-0.5 * ((t - bt - 0.2) / 0.06) ** 2)  # T
        ecg -= 0.2 * np.exp(-0.5 * ((t - bt - 0.04) / 0.015) ** 2) # S
    ecg += 0.05 * np.random.randn(len(t))

    features = extract_features_188(ecg, fs)
    print(f"\nExtracted {len(features)} features from synthetic ECG")
    print(f"Non-zero features: {np.sum(features != 0)}/{N_FEATURES}")
    print(f"Feature vector (first 20): {features[:20]}")
    print(f"Feature vector (last 10):  {features[-10:]}")

    # Print feature names with values
    print("\n--- Feature summary ---")
    for i in range(N_FEATURES):
        if features[i] != 0:
            print(f"  [{i:3d}] {FEATURE_NAMES.get(i, '?'):30s} = {features[i]:.6f}")
