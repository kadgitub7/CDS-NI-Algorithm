"""
================================================================================
CDS Pipeline for PhysioNet 2017 AF Classification Challenge
================================================================================

Runs the full CDS Algorithm 1-4 pipeline on PhysioNet 2017 ECG data.

The PhysioNet 2017 dataset has single-lead ECG waveforms labeled:
  N = Normal sinus rhythm     -> class 1 (healthy)
  A = Atrial fibrillation     -> class 2
  O = Other rhythm            -> class 3
  ~ = Noisy / unclassifiable  -> class 4

This script:
  1. Extracts 40 features from raw ECG waveforms (time, frequency, HRV)
  2. Patches the CDS constants to match PhysioNet 2017 structure
  3. Runs Algorithms 1-4 with LOOCV evaluation
  4. Reports accuracy, sensitivity, specificity

Since PhysioNet 2017 has no demographic features (age, sex, etc.),
fairness analysis is disabled.
================================================================================
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Setup paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# ── Import feature extraction ───────────────────────────────────────────────
from physionet2017_feature_extraction import (
    load_physionet2017,
    load_dataset_physionet2017,
    N_FEATURES as PN_N_FEATURES,
    FEATURE_NAMES as PN_FEATURE_NAMES,
    HEALTHY_CLASS as PN_HEALTHY_CLASS,
    DISEASE_CLASSES as PN_DISEASE_CLASSES,
    LABEL_NAMES,
)

# ── Patch CDS_Paper_Algorithms constants BEFORE importing ───────────────────
import CDS_Paper_Algorithms as cds

# Save originals for reference
_ORIG_N_FEATURES = cds.N_FEATURES
_ORIG_HEALTHY_CLASS = cds.HEALTHY_CLASS
_ORIG_LABEL_COL_IDX = cds.LABEL_COL_IDX

# Patch for PhysioNet 2017
cds.N_FEATURES = PN_N_FEATURES
cds.HEALTHY_CLASS = PN_HEALTHY_CLASS
cds.LABEL_COL_IDX = PN_N_FEATURES  # label is after all features
cds.FEATURE_NAMES = PN_FEATURE_NAMES
cds.MISSING_VALUE_COLS = frozenset()  # no pre-known missing columns

# Adjust u_min for larger dataset
# With 8528 records, u_min=200 is fine (allows deep trees)
# But for subsets, we may need to adjust
cds.U_MIN = 200

# Patch classify_features to work with variable N_FEATURES
_orig_classify = cds.classify_features
def _classify_features_flexible(data: np.ndarray) -> Dict[int, cds.FeatureKind]:
    kinds = {}
    n_feat = data.shape[1]
    for col in range(n_feat):
        col_data = data[:, col]
        valid = col_data[~np.isnan(col_data)]
        if len(valid) == 0:
            kinds[col] = cds.FeatureKind.CONTINUOUS
            continue
        unique_vals = set(valid)
        if unique_vals.issubset({0.0, 1.0}):
            kinds[col] = cds.FeatureKind.BINARY
        else:
            kinds[col] = cds.FeatureKind.CONTINUOUS
    return kinds

cds.classify_features = _classify_features_flexible

# ── Import Algorithm modules (they read constants from cds at import time) ──
import Algorithm2 as alg2_mod
import Algorithm3 as alg3_mod

# Patch Algorithm 4 disease classes
import Algorithm4 as alg4_mod
alg4_mod.ALL_DISEASE_CLASSES = PN_DISEASE_CLASSES
alg4_mod.HEALTHY_CLASS_ALG4 = PN_HEALTHY_CLASS

# Disable fairness (no sex feature in PhysioNet 2017)
import fairness_config as _fc
_fc.ENABLE_FAIRNESS_RL = False
_fc.ENABLE_FORCED_SEX_BRANCHING = False
_fc.ENABLE_EQUALIZED_ODDS = False
_fc.ENABLE_DATA_AUGMENTATION = False

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger("CDS.PhysioNet2017.Pipeline")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    log.addHandler(h)
    log.propagate = False


# ── Feature cache ────────────────────────────────────────────────────────────
CACHE_DIR = SCRIPT_DIR / "data" / "physioNetData2017_cache"


def load_or_extract_features(
    data_dir: str,
    max_records: Optional[int] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load features from cache or extract from raw ECG."""
    cache_file = CACHE_DIR / f"features_n{max_records or 'all'}.pkl"

    if use_cache and cache_file.exists():
        log.info(f"Loading cached features from {cache_file}")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        return cached["data"], cached["labels"], cached["record_ids"]

    log.info(f"Extracting features from {data_dir}...")
    t0 = time.time()
    data, labels, record_ids = load_physionet2017(data_dir, max_records=max_records)
    elapsed = time.time() - t0
    log.info(f"Feature extraction: {elapsed:.1f}s ({elapsed/len(record_ids)*1000:.1f}ms/record)")

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"data": data, "labels": labels, "record_ids": record_ids}, f)
        log.info(f"Cached to {cache_file}")

    return data, labels, record_ids


def run_cds_pipeline(
    data: np.ndarray,
    labels: np.ndarray,
    max_users: Optional[int] = None,
    rng_seed: int = 42,
    n_bins: int = 20,
) -> alg4_mod.Algorithm4Output:
    """
    Run the full CDS Algorithm 1-4 pipeline with LOOCV.

    Parameters
    ----------
    data   : (N, 40) feature matrix from PhysioNet 2017
    labels : (N,) class labels (1=Normal, 2=AFib, 3=Other, 4=Noisy)
    max_users : limit LOOCV to first N users
    """
    import logging as _logging

    # Suppress verbose sub-algorithm logs
    for name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3"):
        _logging.getLogger(name).setLevel(_logging.WARNING)

    n = data.shape[0]
    n_feat = data.shape[1]
    log.info("=" * 65)
    log.info("CDS ALGORITHM 4: PhysioNet 2017 AF Classification")
    log.info("=" * 65)
    log.info(f"Dataset: {n} records × {n_feat} features")
    log.info(f"Labels: { {LABEL_NAMES.get(c, c): int((labels==c).sum()) for c in sorted(set(labels))} }")
    log.info(f"Healthy class: {PN_HEALTHY_CLASS} ({LABEL_NAMES[PN_HEALTHY_CLASS]})")
    log.info(f"Disease classes: {PN_DISEASE_CLASSES}")

    # Build tree on full dataset for inspection
    log.info("\nStep 1: Building decision tree (full dataset)...")
    tree = cds.build_decision_tree(data, labels)
    log.info(f"  Tree nodes: {len(tree.all_nodes)}")
    for nid, node in tree.all_nodes.items():
        log.info(f"    {nid}: {node.n_users} users, "
                 f"health_dist={dict(sorted(node.health_dist.items()))}")

    # Run LOOCV
    log.info(f"\nStep 2: LOOCV (max_users={max_users or n})...")
    output = alg4_mod.run_loocv(
        data=data,
        labels=labels,
        max_users=max_users,
        rng_seed=rng_seed,
        verbose=False,
        n_bins=n_bins,
    )

    # Print summary
    log.info("\n" + "=" * 65)
    log.info("RESULTS SUMMARY")
    log.info("=" * 65)
    log.info(f"Overall accuracy:  {output.overall_accuracy * 100:.1f}%")
    log.info(f"Sensitivity:       {output.sensitivity * 100:.1f}%  "
             f"(diseased correctly identified)")
    log.info(f"Specificity:       {output.specificity * 100:.1f}%  "
             f"(healthy correctly identified)")
    log.info(f"False alarm rate:  {output.false_alarm_rate * 100:.1f}%")
    log.info(f"Screening:         {output.n_screening}")
    log.info(f"Healthy: {output.n_healthy_correct}/{output.n_healthy_total} correct")
    log.info(f"Diseased: {output.n_diseased_correct}/{output.n_diseased_total} correct")

    # Per-class breakdown
    if output.records:
        log.info("\nPer-class breakdown:")
        class_correct = {}
        class_total = {}
        for rec in output.records:
            c = rec.true_label
            class_total[c] = class_total.get(c, 0) + 1
            if rec.is_correct:
                class_correct[c] = class_correct.get(c, 0) + 1
        for c in sorted(class_total.keys()):
            correct = class_correct.get(c, 0)
            total = class_total[c]
            name = LABEL_NAMES.get(c, f"Class {c}")
            log.info(f"  {name}: {correct}/{total} = {correct/total*100:.1f}%")

    return output


def main():
    data_dir = str(SCRIPT_DIR / "data" / "physioNetData2017")

    # Parse arguments
    max_records = None
    max_users = None
    if len(sys.argv) > 1:
        max_records = int(sys.argv[1])
    if len(sys.argv) > 2:
        max_users = int(sys.argv[2])

    # Load / extract features
    data, labels, record_ids = load_or_extract_features(
        data_dir, max_records=max_records
    )

    # Run CDS pipeline
    output = run_cds_pipeline(
        data=data,
        labels=labels,
        max_users=max_users,
        rng_seed=42,
    )

    return output


if __name__ == "__main__":
    main()
