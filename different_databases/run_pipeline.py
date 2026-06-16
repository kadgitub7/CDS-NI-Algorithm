"""
Multi-Database CDS Pipeline Runner
===================================

Runs the full CDS Algorithm 1-4 pipeline on either:
  - UCI Arrhythmia database (original paper dataset)
  - PhysioNet 2017 AF Classification Challenge

Usage:
    python run_pipeline.py --dataset uci [--max-users N]
    python run_pipeline.py --dataset physionet [--max-users N] [--max-records N]

The --dataset flag switches all constants, feature definitions, data loaders,
and fairness settings to match the selected database.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure this directory is on the path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Run CDS Algorithms 1-4 on UCI or PhysioNet dataset"
    )
    parser.add_argument(
        "--dataset", choices=["uci", "physionet"], default="uci",
        help="Which dataset to use (default: uci)"
    )
    parser.add_argument(
        "--max-users", type=int, default=None,
        help="Limit LOOCV to first N users (for quick testing)"
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="PhysioNet only: limit number of ECG records to load"
    )
    parser.add_argument(
        "--no-loocv", action="store_true",
        help="Skip LOOCV (just build tree and train models)"
    )
    args = parser.parse_args()

    # ── Step 1: Set dataset configuration ────────────────────────────────────
    from dataset_config import set_dataset, get_config
    cfg = set_dataset(args.dataset)

    # ── Step 2: Disable fairness for datasets without sex feature ────────────
    #    Must happen BEFORE importing Algorithm2/4, since they read
    #    fairness_config at import time.
    if not cfg.has_sex_feature:
        import fairness_config as fc
        fc.ENABLE_FAIRNESS_RL = False
        fc.ENABLE_FORCED_SEX_BRANCHING = False
        fc.ENABLE_EQUALIZED_ODDS = False
        fc.ENABLE_DATA_AUGMENTATION = False
        fc.ENABLE_ADVERSARIAL_DEBIASING = False
        fc.ENABLE_REWEIGHING = False

    # ── Step 3: Reload algorithm constants from config ───────────────────────
    from CDS_Paper_Algorithms import reload_config
    reload_config()
    from Algorithm2 import reload_config_alg2
    reload_config_alg2()
    from Algorithm4 import reload_config_alg4
    reload_config_alg4()

    # ── Step 4: Configure logging ────────────────────────────────────────────
    log = logging.getLogger("CDS.Pipeline")
    if not log.handlers:
        log.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
        log.addHandler(h)
        log.propagate = False

    # Suppress verbose sub-algorithm logs
    for name in ("CDS.Alg1", "CDS.Alg2", "CDS.Alg3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    log.info("=" * 65)
    log.info(f"CDS MULTI-DATABASE PIPELINE: {args.dataset.upper()}")
    log.info("=" * 65)
    log.info(f"Dataset:    {cfg.name}")
    log.info(f"Features:   {cfg.n_features}")
    log.info(f"Threshold:  {cfg.diagnostic_threshold}")
    log.info(f"Has sex:    {cfg.has_sex_feature}")
    log.info(f"Classes:    healthy={cfg.healthy_class}, disease={cfg.disease_classes}")

    # ── Step 5: Load data ────────────────────────────────────────────────────
    log.info("\nStep 1: Loading dataset...")
    t0 = time.time()

    if args.dataset == "physionet" and args.max_records is not None:
        data, labels = cfg.load_fn(cfg.data_path, max_records=args.max_records)
    else:
        data, labels = cfg.load_fn(cfg.data_path)

    log.info(f"  Loaded: {data.shape[0]} records x {data.shape[1]} features "
             f"({time.time()-t0:.1f}s)")
    log.info(f"  Labels: {dict(sorted({int(c): int((labels==c).sum()) for c in set(labels)}.items()))}")

    # ── Step 6: Build decision tree (Algorithm 1) ────────────────────────────
    log.info("\nStep 2: Building decision tree (Algorithm 1)...")
    from CDS_Paper_Algorithms import build_decision_tree
    tree = build_decision_tree(data, labels)
    log.info(f"  Tree depth: {tree.depth()}, nodes: {tree.count_nodes()}")
    for nid, node in tree.all_nodes.items():
        log.info(f"    {nid}: {node.n_users} users")

    if args.no_loocv:
        log.info("\nSkipping LOOCV (--no-loocv flag set)")

        # Still run Algorithm 2 and 3 for demonstration
        from Algorithm2 import run_algorithm2
        from Algorithm3 import run_algorithm3

        log.info("\nStep 3: Algorithm 2 (Perceptor + Executive Training)...")
        alg2 = run_algorithm2(tree=tree, data=data, labels=labels)
        log.info(f"  Perceptor entries: {alg2.n_perceptor_entries}")
        log.info(f"  Executive entries: {alg2.n_executive_entries}")

        log.info("\nStep 4: Algorithm 3 (Action Refinement)...")
        alg3 = run_algorithm3(alg2_output=alg2, tree=tree, data=data, labels=labels)
        log.info(f"  Retained actions: {len(alg3.refined_actions)}")
        log.info(f"  Removed actions:  {len(alg3.removed_actions)}")

        log.info("\n" + "=" * 65)
        log.info("Pipeline complete (no LOOCV). Tree and models built successfully.")
        log.info("=" * 65)
        return

    # ── Step 7: Run LOOCV (Algorithm 4) ──────────────────────────────────────
    log.info(f"\nStep 3: LOOCV with per-fold retraining "
             f"(max_users={args.max_users or data.shape[0]})...")

    from Algorithm4 import run_loocv
    t0 = time.time()
    output = run_loocv(
        data=data,
        labels=labels,
        max_users=args.max_users,
        rng_seed=42,
        verbose=False,
    )
    elapsed = time.time() - t0

    # ── Step 8: Results ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info(f"RESULTS ({args.dataset.upper()}) — {elapsed:.1f}s elapsed")
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
            log.info(f"  Class {c}: {correct}/{total} = {correct/total*100:.1f}%")

    log.info("=" * 65)
    return output


if __name__ == "__main__":
    main()
