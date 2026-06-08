"""
PhysioNet 2017 ECG Feature Distribution Analyzer
=================================================

Loads .mat/.hea ECG files, extracts 188 features, and plots the
distribution of each feature split by label:
  - Label N  (Normal)    -> "Healthy"
  - Label A / O          -> "Unhealthy"
  - Noisy (~) records are skipped

Usage
-----
    python analyze_feature_distributions.py --data_dir /path/to/physioNetData2017

The directory must contain:
  - REFERENCE-original.csv  (columns: record_id, label)
  - <record_id>.mat files   (each with a 'val' key containing the ECG waveform)

Optional flags:
  --max_records  N     Limit processing to the first N records (useful for a
                       quick test run, default: all)
  --output_dir   DIR   Where to save the PDF/PNG plots (default: ./plots)
  --batch_size   N     Number of features per plot page (default: 9)
  --format       ext   Output image format: pdf | png (default: pdf)
  --no_cache          Do not use cached features even if available

Example quick test with just a handful of records:
    python analyze_feature_distributions.py --data_dir ./data --max_records 50
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe on all platforms
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DistributionAnalyzer")

# ---------------------------------------------------------------------------
# Import the feature extractor (must be on PYTHONPATH or in the same dir)
# ---------------------------------------------------------------------------
try:
    from physionet2017_feature_extraction_188 import (
        extract_features_188,
        FEATURE_NAMES,
        N_FEATURES,
        PHYSIONET_SAMPLING_RATE,
        PHYSIONET_LABEL_MAP,
        PHYSIONET_EXCLUDED_LABELS,
    )
except ImportError as exc:
    sys.exit(
        f"[ERROR] Cannot import physionet2017_feature_extraction_188.\n"
        f"Make sure the script is in the same folder or on PYTHONPATH.\n"
        f"Original error: {exc}"
    )

# ---------------------------------------------------------------------------
# Colour palette — colourblind-friendly
# ---------------------------------------------------------------------------
COLOR_HEALTHY   = "#2196F3"   # blue
COLOR_UNHEALTHY = "#F44336"   # red
ALPHA = 0.45

# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def load_records(
    data_dir: str,
    max_records: Optional[int] = None,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
):
    """
    Walk the data directory, extract 188 features for each record, and
    return (features, labels) arrays.

    labels:  1 = Healthy (N),  2 = Unhealthy (A / O)
    """
    import pandas as pd

    if use_cache and cache_path and cache_path.exists():
        log.info(f"Loading cached features from {cache_path}")
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        key = "features" if "features" in cached else "data"
        return cached[key], cached["labels"]

    ref_csv = Path(data_dir) / "REFERENCE-original.csv"
    if not ref_csv.exists():
        sys.exit(f"[ERROR] REFERENCE-original.csv not found in {data_dir}")

    ref_df = pd.read_csv(ref_csv, header=None, names=["record_id", "label"])
    if max_records is not None:
        ref_df = ref_df.head(max_records)

    feature_rows, label_rows = [], []
    skipped = excluded = 0
    total = len(ref_df)

    for i, (_, row) in enumerate(ref_df.iterrows(), start=1):
        rid        = str(row["record_id"]).strip()
        label_str  = str(row["label"]).strip()
        mat_path   = Path(data_dir) / f"{rid}.mat"

        if label_str in PHYSIONET_EXCLUDED_LABELS:
            excluded += 1
            continue
        if label_str not in PHYSIONET_LABEL_MAP:
            log.debug(f"  Unknown label '{label_str}' for {rid} — skipping")
            skipped += 1
            continue
        if not mat_path.exists():
            log.debug(f"  .mat not found: {mat_path} — skipping")
            skipped += 1
            continue

        try:
            mat   = scipy.io.loadmat(str(mat_path))
            sig   = mat["val"].flatten().astype(float)
            feats = extract_features_188(sig, fs=PHYSIONET_SAMPLING_RATE)
            feature_rows.append(feats)
            label_rows.append(PHYSIONET_LABEL_MAP[label_str])
        except Exception as exc:
            log.warning(f"  Error processing {rid}: {exc}")
            skipped += 1

        if i % 200 == 0:
            log.info(f"  Processed {i}/{total} …")

    if not feature_rows:
        sys.exit("[ERROR] No records were successfully processed.")

    features = np.array(feature_rows, dtype=float)
    labels   = np.array(label_rows,   dtype=int)

    log.info(
        f"Loaded {len(labels)} records  "
        f"(healthy={np.sum(labels==1)}, unhealthy={np.sum(labels==2)}, "
        f"skipped={skipped}, noisy-excluded={excluded})"
    )

    if use_cache and cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump({"features": features, "labels": labels}, fh)
        log.info(f"Cached features to {cache_path}")

    return features, labels


# ---------------------------------------------------------------------------
# Per-feature summary statistics
# ---------------------------------------------------------------------------

def feature_summary(healthy: np.ndarray, unhealthy: np.ndarray) -> dict:
    """Return a dict of summary stats and a KS-test p-value."""
    def _stats(arr):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return {"n": 0, "mean": 0, "std": 0, "median": 0, "q1": 0, "q3": 0}
        return {
            "n":      len(arr),
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "median": float(np.median(arr)),
            "q1":     float(np.percentile(arr, 25)),
            "q3":     float(np.percentile(arr, 75)),
        }

    h_clean = healthy  [np.isfinite(healthy)]
    u_clean = unhealthy[np.isfinite(unhealthy)]

    ks_stat, ks_p = (0.0, 1.0)
    if len(h_clean) >= 3 and len(u_clean) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ks_stat, ks_p = scipy.stats.ks_2samp(h_clean, u_clean)

    return {
        "healthy":   _stats(h_clean),
        "unhealthy": _stats(u_clean),
        "ks_stat":   ks_stat,
        "ks_p":      ks_p,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _hist_or_kde(ax, data: np.ndarray, color: str, label: str, n_bins: int = 40):
    """Draw a semi-transparent histogram with an optional KDE overlay."""
    clean = data[np.isfinite(data)]
    if len(clean) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    # Clip extreme outliers for display (keep 1st–99th percentile range)
    lo, hi = np.percentile(clean, [0.5, 99.5])
    if lo == hi:
        lo -= 1e-9
        hi += 1e-9
    clipped = np.clip(clean, lo, hi)

    ax.hist(
        clipped,
        bins=n_bins,
        color=color,
        alpha=ALPHA,
        density=True,
        label=f"{label} (n={len(clean)})",
    )

    # KDE overlay when enough samples
    if len(clipped) >= 5 and np.std(clipped) > 0:
        try:
            kde = scipy.stats.gaussian_kde(clipped)
            xs  = np.linspace(lo, hi, 300)
            ax.plot(xs, kde(xs), color=color, linewidth=1.6)
        except Exception:
            pass


def plot_feature_page(
    ax,
    feat_idx: int,
    healthy: np.ndarray,
    unhealthy: np.ndarray,
    summary: dict,
):
    """Populate a single Axes with the distribution plot for one feature."""
    feat_name = FEATURE_NAMES.get(feat_idx, f"feature_{feat_idx}")

    _hist_or_kde(ax, healthy,   COLOR_HEALTHY,   "Healthy")
    _hist_or_kde(ax, unhealthy, COLOR_UNHEALTHY,  "Unhealthy")

    # Annotate with KS p-value
    ks_p = summary["ks_p"]
    sig_str = (
        "***" if ks_p < 0.001 else
        "**"  if ks_p < 0.01  else
        "*"   if ks_p < 0.05  else
        "n.s."
    )
    ax.set_title(
        f"[{feat_idx}] {feat_name}\nKS p={ks_p:.2e}  {sig_str}",
        fontsize=7.5,
        pad=3,
    )
    ax.set_xlabel("Value", fontsize=6.5)
    ax.set_ylabel("Density", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, framealpha=0.6)


# ---------------------------------------------------------------------------
# Main plot loop
# ---------------------------------------------------------------------------

def plot_all_features(
    features: np.ndarray,
    labels:   np.ndarray,
    output_dir: str,
    batch_size: int = 9,
    fmt: str = "pdf",
):
    """
    Generate one (PDF) or many (PNG) output file(s) with distribution plots
    for all 188 features arranged in a grid of `batch_size` panels.

    For PDF: all pages are merged into a single multi-page PDF.
    For PNG: one PNG file per page.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    healthy   = features[labels == 1]   # Normal
    unhealthy = features[labels == 2]   # Abnormal

    n_cols = 3
    n_rows = (batch_size + n_cols - 1) // n_cols   # ceil(batch_size / n_cols)

    n_pages = (N_FEATURES + batch_size - 1) // batch_size

    # Pre-compute summaries (fast — no plotting yet)
    log.info("Computing per-feature statistics …")
    summaries = [
        feature_summary(healthy[:, i], unhealthy[:, i])
        for i in range(N_FEATURES)
    ]

    # Sort by KS p-value so most discriminating features appear first
    # (comment out the sort if you prefer index order)
    sorted_indices = sorted(range(N_FEATURES), key=lambda i: summaries[i]["ks_p"])

    log.info(f"Generating plots ({n_pages} page(s), format={fmt}) …")

    if fmt == "pdf":
        pdf_file = out_path / "feature_distributions.pdf"
        with PdfPages(str(pdf_file)) as pdf:
            for page in range(n_pages):
                feat_slice = sorted_indices[page * batch_size:(page + 1) * batch_size]
                _render_page(
                    feat_slice, healthy, unhealthy, summaries,
                    n_rows, n_cols, page, n_pages, pdf_obj=pdf, out_path=None,
                )
                if (page + 1) % 10 == 0:
                    log.info(f"  Page {page+1}/{n_pages} …")
        log.info(f"Saved  →  {pdf_file}")

    else:  # PNG (one file per page)
        for page in range(n_pages):
            feat_slice = sorted_indices[page * batch_size:(page + 1) * batch_size]
            png_file = out_path / f"feature_distributions_page{page+1:03d}.png"
            _render_page(
                feat_slice, healthy, unhealthy, summaries,
                n_rows, n_cols, page, n_pages,
                pdf_obj=None, out_path=png_file,
            )
            if (page + 1) % 10 == 0:
                log.info(f"  Page {page+1}/{n_pages} …")
        log.info(f"Saved {n_pages} PNG files to {out_path}")

    # -----------------------------------------------------------------------
    # Summary table: top-20 most discriminating features
    # -----------------------------------------------------------------------
    _save_summary_table(summaries, sorted_indices, out_path)


def _render_page(
    feat_slice, healthy, unhealthy, summaries,
    n_rows, n_cols, page, n_pages,
    pdf_obj, out_path,
):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_i, feat_idx in enumerate(feat_slice):
        plot_feature_page(
            axes_flat[ax_i],
            feat_idx,
            healthy  [:, feat_idx],
            unhealthy[:, feat_idx],
            summaries[feat_idx],
        )

    # Hide unused panels
    for ax_i in range(len(feat_slice), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    fig.suptitle(
        f"Feature Distributions — Healthy vs Unhealthy  "
        f"(page {page+1}/{n_pages}, sorted by KS discriminability)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()

    if pdf_obj is not None:
        pdf_obj.savefig(fig, bbox_inches="tight")
    else:
        fig.savefig(str(out_path), dpi=130, bbox_inches="tight")

    plt.close(fig)


def _save_summary_table(summaries, sorted_indices, out_path: Path):
    """Write a CSV with per-feature stats, sorted by KS p-value."""
    csv_path = out_path / "feature_summary.csv"
    rows = []
    for i in sorted_indices:
        s   = summaries[i]
        h   = s["healthy"]
        u   = s["unhealthy"]
        rows.append(
            f"{i},"
            f"{FEATURE_NAMES.get(i, f'feature_{i}')},"
            f"{s['ks_p']:.4e},"
            f"{s['ks_stat']:.4f},"
            f"{h['n']},{h['mean']:.4f},{h['std']:.4f},{h['median']:.4f},"
            f"{u['n']},{u['mean']:.4f},{u['std']:.4f},{u['median']:.4f}"
        )
    header = (
        "feat_idx,feat_name,ks_p,ks_stat,"
        "h_n,h_mean,h_std,h_median,"
        "u_n,u_mean,u_std,u_median"
    )
    csv_path.write_text(header + "\n" + "\n".join(rows))
    log.info(f"Summary CSV  →  {csv_path}")

    # Print top-20 to console
    log.info("\nTop 20 most discriminating features (by KS p-value):")
    log.info(f"  {'Rank':>4}  {'Idx':>5}  {'Feature':<35}  {'KS-p':>10}  "
             f"{'H-mean':>9}  {'U-mean':>9}")
    for rank, i in enumerate(sorted_indices[:20], start=1):
        s = summaries[i]
        log.info(
            f"  {rank:>4}  {i:>5}  {FEATURE_NAMES.get(i,'?'):<35}  "
            f"{s['ks_p']:10.3e}  "
            f"{s['healthy']['mean']:9.4f}  {s['unhealthy']['mean']:9.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot per-feature distributions for PhysioNet 2017 ECG data "
                    "(healthy vs unhealthy), sorted by discriminability."
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Directory containing .mat files + REFERENCE-original.csv",
    )
    p.add_argument(
        "--output_dir", default="./plots",
        help="Where to save output plots and summary CSV (default: ./plots)",
    )
    p.add_argument(
        "--max_records", type=int, default=None,
        help="Limit the number of records processed (useful for quick tests)",
    )
    p.add_argument(
        "--batch_size", type=int, default=9,
        help="Number of feature panels per page (default: 9 — gives a 3×3 grid)",
    )
    p.add_argument(
        "--format", choices=["pdf", "png"], default="pdf",
        help="Output format: 'pdf' (single multi-page file) or 'png' (one per page)",
    )
    p.add_argument(
        "--no_cache", action="store_true",
        help="Ignore cached features and re-extract from raw files",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tag        = f"n{args.max_records}" if args.max_records else "all"
    cache_dir  = Path(args.data_dir).parent / "physioNetData2017_cache"
    cache_path = cache_dir / f"features_188_{tag}.pkl"

    features, labels = load_records(
        data_dir    = args.data_dir,
        max_records = args.max_records,
        cache_path  = cache_path,
        use_cache   = not args.no_cache,
    )

    plot_all_features(
        features    = features,
        labels      = labels,
        output_dir  = args.output_dir,
        batch_size  = args.batch_size,
        fmt         = args.format,
    )

    log.info("Done.")