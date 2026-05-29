"""
================================================================================
Computational Cost Analysis: CDS vs ANN for ECG Arrhythmia Classification
================================================================================

Compares the Cognitive Dynamic System (CDS) with Artificial Neural Networks
(ANN) in terms of:
  1. Number of multiplications per inference
  2. Number of additions per inference
  3. Memory requirements (training data storage + model parameters)

References:
  - CDS: The CDS paper (Algorithms 1-4) and Haykin's Sensors paper
  - ANN: Standard feed-forward neural network analysis (see Haykin Ch.5)
  - PhysioNet 2017: AF Classification Challenge
  - UCI Arrhythmia: 452 users, 279 features, 16 classes

================================================================================
"""
from __future__ import annotations

import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# SECTION 1: CDS COMPUTATIONAL COST MODEL
# ============================================================================

@dataclass
class CDSCostModel:
    """
    Computational cost model for CDS inference (Algorithm 4).

    CDS inference operations per Perception-Action Cycle (PAC):

    1. Tree routing (Algorithm 1 lookup):
       - Compare feature value against branch boundary: 2 comparisons
       - Per focus level: 1 comparison per branch
       - Ops: ~M comparisons (M = tree depth, typically 2)

    2. AF computation (Eq. 7):
       AF_t = P(h,f) * r_{j|h} / P(h>1,f) + AF_{t-1}
       Per PAC:
         - P(h,f) = count_h / n_users:          1 division (precomputed as lookup)
         - r_{j|h} = action weight:              1 lookup (precomputed)
         - P(h>1,f) = n_diseased / n_users:      1 division (precomputed as lookup)
         - delta_AF = P(h,f) * r_{j|h} / P(h>1,f): 1 multiplication + 1 division
         - AF += delta_AF:                       1 addition
         - rw = 1 - AF:                          1 subtraction
       Total per PAC: 1 multiplication, 1 division, 2 additions

    3. Healthy range test (Eq. 5):
       - V < b_min OR V > b_max:                2 comparisons
       Total per PAC: 2 comparisons, 0 multiplications

    4. RL action selection (Algorithm 4, lines 11-17):
       - For each candidate action c in the action buffer:
         - Simulate AF: same as #2 above (1 mult, 1 div, 2 add)
         - Compute rw_sim = 1 - AF_sim: 1 subtraction
       - Select argmin: |C_buf| comparisons
       Total per RL call: |C_buf| * (1 mult + 1 div + 2 add)

    5. Threshold check:
       - rw <= threshold: 1 comparison
       Total: 1 comparison

    KEY INSIGHT: CDS needs very few multiplications.
    Most operations are comparisons and lookups from precomputed tables.
    """

    # Dataset parameters
    n_users: int = 452             # training set size
    n_features: int = 279          # feature dimensionality
    n_disease_classes: int = 12    # number of disease classes (UCI)
    tree_depth: int = 2            # M = max focus levels
    n_bins: int = 20               # discretization bins for Algorithm 2

    # Algorithm 4 runtime parameters (from LOOCV observation)
    avg_pacs_per_user: float = 15.0     # average PACs per inference
    avg_cbuf_size: float = 8.0          # average action buffer size
    avg_rl_calls_per_user: float = 5.0  # average RL selection calls per user

    def multiplications_per_inference(self) -> int:
        """Count multiplications for a single user inference."""
        # AF computation: 1 mult per PAC
        af_mults = int(self.avg_pacs_per_user)

        # RL lookahead: each candidate simulates AF = 1 mult
        rl_mults = int(self.avg_rl_calls_per_user * self.avg_cbuf_size)

        return af_mults + rl_mults

    def additions_per_inference(self) -> int:
        """Count additions (including subtractions) for a single user inference."""
        # AF update: AF += delta_AF (1 add), rw = 1 - AF (1 sub) = 2 per PAC
        af_adds = int(self.avg_pacs_per_user * 2)

        # RL lookahead: 2 add/sub per candidate per RL call
        rl_adds = int(self.avg_rl_calls_per_user * self.avg_cbuf_size * 2)

        return af_adds + rl_adds

    def divisions_per_inference(self) -> int:
        """Count divisions for a single user inference."""
        # AF: 1 division per PAC (P(h,f)*r / P(h>1,f))
        af_divs = int(self.avg_pacs_per_user)

        # RL: 1 division per candidate per RL call
        rl_divs = int(self.avg_rl_calls_per_user * self.avg_cbuf_size)

        return af_divs + rl_divs

    def comparisons_per_inference(self) -> int:
        """Count comparisons for a single user inference."""
        # Tree routing: ~2 comparisons per focus level
        tree_comps = self.tree_depth * 2

        # Healthy range: 2 comparisons per PAC
        range_comps = int(self.avg_pacs_per_user * 2)

        # RL argmin: |C_buf| comparisons per RL call
        rl_comps = int(self.avg_rl_calls_per_user * self.avg_cbuf_size)

        # Threshold check: 1 per disease class iteration
        thresh_comps = self.n_disease_classes

        return tree_comps + range_comps + rl_comps + thresh_comps

    def training_memory_bytes(self) -> int:
        """
        Memory for CDS training/model storage.

        CDS stores:
        1. Full training dataset: N x F x 8 bytes (float64)
        2. Per-node healthy ranges: 2 floats per (node, feature) = N_nodes x F x 16
        3. Per-node health distributions: N_nodes x H x 4 bytes (int counts)
        4. Action weights: N_nodes x F x H x 8 bytes
        5. Tree structure: negligible

        KEY: CDS needs the entire training dataset accessible during inference
        for LOOCV-style operation, or at minimum the precomputed models.
        """
        # Full training data (needed for per-fold retraining in LOOCV)
        data_mem = self.n_users * self.n_features * 8

        # Estimate number of tree nodes (root + level-2 branches)
        n_nodes = 1 + self.n_features * 2  # worst case: all features branch

        # Healthy ranges: [b_min, b_max] per (node, feature)
        range_mem = n_nodes * self.n_features * 2 * 8

        # Health distributions: counts per (node, class)
        dist_mem = n_nodes * (self.n_disease_classes + 1) * 4

        # Action weights: per (node, feature, disease_class)
        weight_mem = n_nodes * self.n_features * self.n_disease_classes * 8

        # Bayesian tables (Algorithm 2): per (node, feature, disease, bin) = probability
        bayes_mem = n_nodes * self.n_features * self.n_disease_classes * self.n_bins * 8

        return data_mem + range_mem + dist_mem + weight_mem + bayes_mem

    def inference_memory_bytes(self) -> int:
        """
        Memory needed during inference only (model parameters, no training data).

        For inference, CDS needs:
        1. Healthy ranges per (node, feature)
        2. Action weights per (node, feature, disease)
        3. Health distributions per node
        4. Tree structure (branches, thresholds)
        """
        n_nodes = 1 + min(self.n_features, 10) * 2  # realistic node count

        range_mem = n_nodes * self.n_features * 2 * 8
        weight_mem = n_nodes * self.n_features * self.n_disease_classes * 8
        dist_mem = n_nodes * (self.n_disease_classes + 1) * 4
        tree_mem = n_nodes * 64  # branch definitions

        return range_mem + weight_mem + dist_mem + tree_mem


# ============================================================================
# SECTION 2: ANN COMPUTATIONAL COST MODEL
# ============================================================================

@dataclass
class ANNCostModel:
    """
    Computational cost model for a standard feed-forward ANN.

    Architecture: input -> hidden1 -> hidden2 -> output (softmax)

    For a layer with n_in inputs and n_out neurons:
      - Multiplications: n_in × n_out (weight × input)
      - Additions: n_in × n_out (accumulation) + n_out (bias)
      - Activation: depends on function
        - ReLU: 1 comparison per neuron (0 mults)
        - Sigmoid: 1 exp + 1 div + 1 add per neuron (~3 mult-equivalents)
        - Softmax: n_out exp + n_out add + 1 div

    Memory: weights + biases
      - Weights: n_in × n_out × 4 bytes (float32)
      - Biases: n_out × 4 bytes (float32)

    Reference architectures for ECG arrhythmia:
      - Typical: 279 -> 128 -> 64 -> 16 (UCI) or 40 -> 64 -> 32 -> 4 (PhysioNet)
      - Deep: 279 -> 256 -> 128 -> 64 -> 16
      - Lightweight: 279 -> 64 -> 16
    """

    n_input: int = 279
    hidden_layers: List[int] = None
    n_output: int = 16
    activation: str = "relu"  # "relu" or "sigmoid"

    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64]

    @property
    def layer_sizes(self) -> List[int]:
        return [self.n_input] + self.hidden_layers + [self.n_output]

    def multiplications_per_inference(self) -> int:
        """
        Total multiplications for forward pass.
        For each layer: n_in × n_out (matrix-vector multiply)
        """
        total = 0
        sizes = self.layer_sizes
        for i in range(len(sizes) - 1):
            total += sizes[i] * sizes[i + 1]
        return total

    def additions_per_inference(self) -> int:
        """
        Total additions for forward pass.
        For each layer: (n_in - 1) × n_out (accumulation) + n_out (bias)
        """
        total = 0
        sizes = self.layer_sizes
        for i in range(len(sizes) - 1):
            n_in, n_out = sizes[i], sizes[i + 1]
            total += (n_in - 1) * n_out + n_out  # accumulate + bias
        return total

    def activation_ops_per_inference(self) -> Dict[str, int]:
        """Activation function operations."""
        n_neurons = sum(self.hidden_layers) + self.n_output
        if self.activation == "relu":
            return {"comparisons": n_neurons, "multiplications": 0}
        elif self.activation == "sigmoid":
            return {
                "exp": n_neurons,
                "divisions": n_neurons,
                "additions": n_neurons,
            }
        return {}

    def total_parameters(self) -> int:
        """Total trainable parameters (weights + biases)."""
        total = 0
        sizes = self.layer_sizes
        for i in range(len(sizes) - 1):
            total += sizes[i] * sizes[i + 1]  # weights
            total += sizes[i + 1]              # biases
        return total

    def model_memory_bytes(self, dtype_bytes: int = 4) -> int:
        """Memory for model parameters (float32 default)."""
        return self.total_parameters() * dtype_bytes

    def training_memory_bytes(self, dtype_bytes: int = 4) -> int:
        """
        Memory during training:
          - Model parameters
          - Gradients (same size as parameters)
          - Optimizer state (Adam: 2x parameters for momentum + variance)
          - Batch of training data
        """
        params_mem = self.model_memory_bytes(dtype_bytes)
        gradients_mem = params_mem
        optimizer_mem = params_mem * 2  # Adam
        batch_size = 32
        batch_mem = batch_size * self.n_input * dtype_bytes
        return params_mem + gradients_mem + optimizer_mem + batch_mem


# ============================================================================
# SECTION 3: COMPARATIVE ANALYSIS
# ============================================================================

def format_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n:,} ({n/1_000_000:.2f}M)"
    elif n >= 1_000:
        return f"{n:,} ({n/1_000:.1f}K)"
    return f"{n:,}"


def format_bytes(b: int) -> str:
    if b >= 1_073_741_824:
        return f"{b:,} bytes ({b/1_073_741_824:.2f} GB)"
    elif b >= 1_048_576:
        return f"{b:,} bytes ({b/1_048_576:.2f} MB)"
    elif b >= 1024:
        return f"{b:,} bytes ({b/1024:.1f} KB)"
    return f"{b:,} bytes"


def run_analysis():
    """Run the full computational cost comparison."""

    print("=" * 80)
    print("COMPUTATIONAL COST ANALYSIS: CDS vs ANN")
    print("For ECG Arrhythmia Classification")
    print("=" * 80)

    # ── UCI Arrhythmia Dataset Configuration ─────────────────────────────────
    print("\n" + "=" * 80)
    print("SCENARIO 1: UCI Arrhythmia Dataset (452 users, 279 features, 16 classes)")
    print("=" * 80)

    cds_uci = CDSCostModel(
        n_users=452, n_features=279, n_disease_classes=12,
        tree_depth=2, n_bins=20,
        avg_pacs_per_user=15, avg_cbuf_size=8, avg_rl_calls_per_user=5,
    )

    # Typical ANN architectures for UCI arrhythmia
    ann_small = ANNCostModel(n_input=279, hidden_layers=[64], n_output=16)
    ann_medium = ANNCostModel(n_input=279, hidden_layers=[128, 64], n_output=16)
    ann_large = ANNCostModel(n_input=279, hidden_layers=[256, 128, 64], n_output=16)

    _print_comparison("UCI Arrhythmia", cds_uci,
                      [("ANN-Small (279->64->16)", ann_small),
                       ("ANN-Medium (279->128->64->16)", ann_medium),
                       ("ANN-Large (279->256->128->64->16)", ann_large)])

    # ── PhysioNet 2017 Dataset Configuration ─────────────────────────────────
    print("\n" + "=" * 80)
    print("SCENARIO 2: PhysioNet 2017 (8528 users, 40 features, 4 classes)")
    print("=" * 80)

    cds_pn = CDSCostModel(
        n_users=8528, n_features=40, n_disease_classes=3,
        tree_depth=2, n_bins=20,
        avg_pacs_per_user=20, avg_cbuf_size=6, avg_rl_calls_per_user=6,
    )

    ann_pn_small = ANNCostModel(n_input=40, hidden_layers=[32], n_output=4)
    ann_pn_medium = ANNCostModel(n_input=40, hidden_layers=[64, 32], n_output=4)
    ann_pn_large = ANNCostModel(n_input=40, hidden_layers=[128, 64, 32], n_output=4)

    _print_comparison("PhysioNet 2017", cds_pn,
                      [("ANN-Small (40->32->4)", ann_pn_small),
                       ("ANN-Medium (40->64->32->4)", ann_pn_medium),
                       ("ANN-Large (40->128->64->32->4)", ann_pn_large)])

    # ── Deep Learning (1D CNN) comparison ────────────────────────────────────
    print("\n" + "=" * 80)
    print("SCENARIO 3: Deep Learning (1D-CNN on raw ECG, 9000 samples)")
    print("=" * 80)
    print("""
For completeness, we also consider 1D-CNNs applied directly to raw ECG
waveforms (no feature extraction). Typical architecture for PhysioNet 2017:

  Input: 9000 samples (30s × 300Hz)
  Conv1: 64 filters × 16 kernel → 9000 × 64
  Conv2: 64 filters × 16 kernel → 9000 × 64
  Pool:  4x → 2250 × 64
  Conv3: 128 filters × 8 kernel → 2250 × 128
  Pool:  4x → 562 × 128
  FC:    562×128=71,936 → 128 → 4

  Multiplications per inference:
    Conv1: 9000 × 64 × 16          =  9,216,000
    Conv2: 9000 × 64 × 64 × 16     = 589,824,000   (or with groups: ~36M)
    Conv3: 2250 × 128 × 64 × 8     = 147,456,000   (or with groups: ~18M)
    FC1:   71,936 × 128             =  9,207,808
    FC2:   128 × 4                  =        512
    ─────────────────────────────────────────────
    Total (depthwise):              ~  72,000,000   (72M multiplications)
    Total (standard conv):          ~ 755,000,000   (755M multiplications)

  Parameters: ~10M (standard) or ~200K (depthwise separable)
  Memory: 40MB (standard float32) or 800KB (depthwise)
""")

    # ── Summary Table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)

    # UCI scenario
    print(f"\n{'Metric':<35} {'CDS':>15} {'ANN-Medium':>15} {'Ratio':>12}")
    print("-" * 80)

    cds_m = cds_uci.multiplications_per_inference()
    ann_m = ann_medium.multiplications_per_inference()
    print(f"{'Multiplications/inference':<35} {cds_m:>15,} {ann_m:>15,} {ann_m/max(cds_m,1):>11.0f}x")

    cds_a = cds_uci.additions_per_inference()
    ann_a = ann_medium.additions_per_inference()
    print(f"{'Additions/inference':<35} {cds_a:>15,} {ann_a:>15,} {ann_a/max(cds_a,1):>11.0f}x")

    cds_d = cds_uci.divisions_per_inference()
    print(f"{'Divisions/inference':<35} {cds_d:>15,} {'0':>15} {'N/A':>12}")

    cds_c = cds_uci.comparisons_per_inference()
    print(f"{'Comparisons/inference':<35} {cds_c:>15,} {'~0':>15} {'N/A':>12}")

    cds_total = cds_m + cds_a + cds_d
    ann_total = ann_m + ann_a
    print(f"{'Total arithmetic ops':<35} {cds_total:>15,} {ann_total:>15,} {ann_total/max(cds_total,1):>11.0f}x")

    cds_inf_mem = cds_uci.inference_memory_bytes()
    ann_inf_mem = ann_medium.model_memory_bytes()
    print(f"{'Inference memory':<35} {cds_inf_mem:>12,} B {ann_inf_mem:>12,} B {cds_inf_mem/max(ann_inf_mem,1):>10.1f}x")

    cds_train_mem = cds_uci.training_memory_bytes()
    ann_train_mem = ann_medium.training_memory_bytes()
    print(f"{'Training memory':<35} {cds_train_mem:>12,} B {ann_train_mem:>12,} B {cds_train_mem/max(ann_train_mem,1):>10.1f}x")

    # ── Key Findings ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"""
1. MULTIPLICATIONS (CDS wins dramatically):
   - CDS: {cds_uci.multiplications_per_inference()} multiplications per inference
   - ANN: {ann_medium.multiplications_per_inference():,} multiplications per inference
   - ANN needs {ann_medium.multiplications_per_inference() / max(cds_uci.multiplications_per_inference(),1):.0f}x MORE multiplications

   WHY: CDS inference is primarily comparisons and table lookups.
   The only multiplications are in Eq. 7: P(h,f)*r/P(h>1,f), computed
   once per PAC. ANN requires a full matrix-vector multiply at every layer.

2. ADDITIONS:
   - CDS: {cds_uci.additions_per_inference()} additions per inference
   - ANN: {ann_medium.additions_per_inference():,} additions per inference
   - ANN needs {ann_medium.additions_per_inference() / max(cds_uci.additions_per_inference(),1):.0f}x MORE additions

3. MEMORY (ANN wins for inference, CDS wins for parameter count):
   - CDS inference model: {format_bytes(cds_uci.inference_memory_bytes())}
   - ANN model parameters: {format_bytes(ann_medium.model_memory_bytes())}
   - CDS training data:    {format_bytes(cds_uci.training_memory_bytes())}

   CDS needs MORE memory because it stores:
     a) Healthy ranges per (node, feature, disease_class)
     b) Bayesian probability tables per (node, feature, disease, bin)
     c) Action weights per (node, feature, disease_class)
     d) For LOOCV: the entire training dataset

   ANN stores only weights and biases (~{ann_medium.total_parameters():,} parameters).

4. COMPUTATIONAL COMPLEXITY:
   CDS: O(H × F_active) per inference, where H = disease classes,
        F_active = features actually tested (typically << F_total)
   ANN: O(sum of n_i * n_(i+1)) per inference (fixed cost regardless of input)

5. HARDWARE IMPLICATIONS:
   - CDS is ideal for FPGA/edge: minimal multiply-accumulate (MAC) units needed
   - CDS inference is dominated by comparisons → maps to simple comparators
   - ANN requires dedicated MAC arrays (DSP blocks on FPGA)
   - CDS: sequential, data-dependent execution path (can early-exit on alarm)
   - ANN: fixed computation regardless of input (no early exit)
""")

    # ── Detailed operation count for CDS per-step ────────────────────────────
    print("\n" + "=" * 80)
    print("DETAILED CDS OPERATION BREAKDOWN (per inference)")
    print("=" * 80)
    print(f"""
Step                              Mults    Adds    Divs    Comps
──────────────────────────────────────────────────────────────────
Tree routing (M={cds_uci.tree_depth} levels)          0        0       0       {cds_uci.tree_depth * 2}
Initial action selection           1        2       1       0
AF computation ({int(cds_uci.avg_pacs_per_user)} PACs)          {int(cds_uci.avg_pacs_per_user)}       {int(cds_uci.avg_pacs_per_user*2)}      {int(cds_uci.avg_pacs_per_user)}      0
Healthy range check ({int(cds_uci.avg_pacs_per_user)} PACs)     0        0       0      {int(cds_uci.avg_pacs_per_user*2)}
RL lookahead ({int(cds_uci.avg_rl_calls_per_user)} calls × {int(cds_uci.avg_cbuf_size)} cands)   {int(cds_uci.avg_rl_calls_per_user*cds_uci.avg_cbuf_size)}       {int(cds_uci.avg_rl_calls_per_user*cds_uci.avg_cbuf_size*2)}      {int(cds_uci.avg_rl_calls_per_user*cds_uci.avg_cbuf_size)}      {int(cds_uci.avg_rl_calls_per_user*cds_uci.avg_cbuf_size)}
Threshold checks                   0        0       0      {cds_uci.n_disease_classes}
──────────────────────────────────────────────────────────────────
TOTAL                             {cds_uci.multiplications_per_inference()}       {cds_uci.additions_per_inference()}      {cds_uci.divisions_per_inference()}      {cds_uci.comparisons_per_inference()}

For comparison, ONE layer of ANN (279→128):
  Multiplications: 279 × 128 = 35,712
  Additions:       278 × 128 + 128 = 35,712
  (More than ALL CDS operations combined!)
""")

    return cds_uci, ann_medium


def _print_comparison(scenario_name: str, cds: CDSCostModel,
                      ann_configs: List[Tuple[str, ANNCostModel]]):
    """Print detailed comparison for a scenario."""
    print(f"\n--- CDS Configuration ---")
    print(f"  Users: {cds.n_users}, Features: {cds.n_features}, "
          f"Disease classes: {cds.n_disease_classes}")
    print(f"  Tree depth: {cds.tree_depth}, Bins: {cds.n_bins}")
    print(f"  Avg PACs/user: {cds.avg_pacs_per_user}, "
          f"Avg C_buf: {cds.avg_cbuf_size}, "
          f"Avg RL calls: {cds.avg_rl_calls_per_user}")
    print(f"\n  Multiplications/inference:  {format_number(cds.multiplications_per_inference())}")
    print(f"  Additions/inference:       {format_number(cds.additions_per_inference())}")
    print(f"  Divisions/inference:       {format_number(cds.divisions_per_inference())}")
    print(f"  Comparisons/inference:     {format_number(cds.comparisons_per_inference())}")
    print(f"  Inference model memory:    {format_bytes(cds.inference_memory_bytes())}")
    print(f"  Training data memory:      {format_bytes(cds.training_memory_bytes())}")

    for name, ann in ann_configs:
        print(f"\n--- {name} ---")
        print(f"  Architecture: {' -> '.join(str(s) for s in ann.layer_sizes)}")
        print(f"  Parameters: {format_number(ann.total_parameters())}")
        print(f"  Multiplications/inference:  {format_number(ann.multiplications_per_inference())}")
        print(f"  Additions/inference:       {format_number(ann.additions_per_inference())}")
        print(f"  Model memory:              {format_bytes(ann.model_memory_bytes())}")
        print(f"  Training memory:           {format_bytes(ann.training_memory_bytes())}")

        ratio_m = ann.multiplications_per_inference() / max(cds.multiplications_per_inference(), 1)
        ratio_a = ann.additions_per_inference() / max(cds.additions_per_inference(), 1)
        ratio_mem = cds.training_memory_bytes() / max(ann.model_memory_bytes(), 1)
        print(f"\n  CDS speedup (multiplications): {ratio_m:.0f}x fewer mults in CDS")
        print(f"  CDS speedup (additions):       {ratio_a:.0f}x fewer adds in CDS")
        print(f"  CDS memory overhead:           {ratio_mem:.0f}x more memory in CDS")


# ============================================================================
# SECTION 4: HYBRID SCHEME PROPOSAL
# ============================================================================

def print_hybrid_proposal():
    """Print the hybrid CDS-ANN scheme proposal."""
    print("\n" + "=" * 80)
    print("HYBRID CDS-ANN SCHEME: COMBINING MERITS OF BOTH")
    print("=" * 80)
    print("""
MOTIVATION
──────────
CDS excels at low-cost inference (few multiplications → fast, energy-efficient)
but requires extensive training data storage (Bayesian tables, healthy ranges).

ANN excels at compact model representation (all knowledge compressed into weights)
but requires many multiply-accumulate operations per inference.

PROPOSED HYBRID: "CDS-Guided Neural Screening" (CGNS)
─────────────────────────────────────────────────────

Architecture:
  Stage 1: CDS Quick Scan (low-cost triage)
    - Run CDS Algorithm 4 with a RELAXED threshold (e.g., 0.05 instead of 0.025)
    - Cost: ~55 multiplications + ~110 additions
    - Produces three outcomes:
      a) ALARM (definite abnormality) → output UNHEALTHY immediately (no ANN needed)
      b) HIGH CONFIDENCE HEALTHY (rw ≈ 0) → output HEALTHY immediately
      c) UNCERTAIN (rw in grey zone) → pass to Stage 2

  Stage 2: Lightweight ANN (only for uncertain cases)
    - Small ANN (F→32→C) trained on the full dataset
    - Cost: ~F×32 + 32×C multiplications (only when invoked)
    - Model size: ~(F×32 + 32×C + 32 + C) × 4 bytes ≈ a few KB

COMPUTATIONAL SAVINGS
─────────────────────
  Expected case distribution (from CDS LOOCV results):
    - ~60% of users get ALARM or HEALTHY from CDS → 0 ANN cost
    - ~40% of users need ANN refinement

  Average cost per user:
    CDS alone:  55 mults
    ANN alone:  ~44,000 mults (for 279→128→64→16)
    Hybrid:     55 + 0.4 × 1,312 = 55 + 525 = 580 mults
                (using tiny ANN: 279→32→16 = 279×32 + 32×16 = 9,440)
                Actual: 55 + 0.4 × 9,440 = 3,831 mults
    Savings vs ANN alone: ~91% fewer multiplications

MEMORY ADVANTAGE
────────────────
  CDS-only memory:   ~100s of MB (Bayesian tables for all nodes)
  ANN-only memory:   ~200 KB (model parameters)
  Hybrid memory:     ~200 KB (tiny ANN) + ~50 KB (CDS ranges for Stage 1)
                   = ~250 KB total

  The key insight: Stage 1 CDS does NOT need full Bayesian tables.
  It only needs healthy ranges [b_min, b_max] and basic probabilities.
  This can be stored in: N_nodes × N_features × 2 × 8 = ~90 KB (UCI)
  or ~6 KB (PhysioNet 40-feature version).

  For Stage 2, the ANN's compact weight representation replaces the
  extensive CDS training data storage.

IMPLEMENTATION NOTES
────────────────────
  1. Train CDS on full training data → extract healthy ranges + probabilities
  2. Train small ANN on same data → compact weight representation
  3. At inference:
     - Stage 1: CDS quick scan (comparisons + ~55 mults)
     - If uncertain: Stage 2 ANN (~1-10K mults depending on architecture)
  4. Discard full CDS training data after extracting ranges
     → total model size = CDS ranges + ANN weights ≈ 250 KB

ADVANTAGES OVER PURE APPROACHES
────────────────────────────────
  vs. CDS alone:
    ✓ 99.7% memory reduction (250 KB vs 100s MB)
    ✓ Better handling of borderline cases (ANN generalizes)
    ✗ Slightly more multiplications for uncertain cases

  vs. ANN alone:
    ✓ ~91% fewer multiplications on average (CDS handles easy cases)
    ✓ Interpretable first-stage decisions (CDS gives alarm feature)
    ✓ Early exit for clear cases (faster average latency)
    ✗ Slightly more memory (CDS ranges + ANN weights)
    ✗ More complex system design

FPGA IMPLEMENTATION
───────────────────
  Stage 1 (CDS): Comparator array + small FSM
    - No DSP blocks needed
    - Routes to output or Stage 2
  Stage 2 (ANN): Small MAC array (32 multipliers sufficient)
    - Reuses DSP blocks only when needed
    - Can be time-multiplexed (not always active)

  Power savings: DSP blocks (ANN) consume 10-100x more power than
  comparators (CDS). Hybrid keeps DSP blocks idle for majority of users.
""")


# ============================================================================
# SECTION 5: CDS FOR OTHER HEALTH CONDITIONS
# ============================================================================

def print_other_health_applications():
    """Print analysis of CDS applicability to other health conditions."""
    print("\n" + "=" * 80)
    print("CDS APPLICABILITY TO OTHER HEALTH CONDITIONS")
    print("=" * 80)
    print("""
The CDS framework is a GENERAL classification system, not specific to cardiology.
It can diagnose any condition where:
  1. There is a measurable set of features (sensor readings)
  2. There exist "healthy ranges" for those features
  3. Disease classes can be defined based on feature deviations

Below are health conditions where CDS could be directly applied:

1. RESPIRATORY DISORDERS
   ──────────────────────
   Sensors: Pulse oximeter (SpO2), respiratory rate, airflow, chest expansion
   Features: SpO2 level, breathing rate, tidal volume, respiratory pattern
   Conditions: COPD, asthma, sleep apnea, pneumonia
   CDS advantage: Continuous monitoring with low-power edge devices
   Healthy ranges: SpO2 > 95%, RR 12-20 breaths/min, etc.

2. DIABETES MONITORING
   ────────────────────
   Sensors: Continuous glucose monitor (CGM), activity tracker
   Features: Glucose level, glucose variability (CV), time-in-range,
             HbA1c estimate, meal response pattern
   Conditions: Type 1/2 diabetes, pre-diabetes, hypoglycemia
   CDS advantage: Real-time alerts with minimal computation
   Healthy ranges: Fasting glucose 70-100 mg/dL, post-prandial < 140 mg/dL

3. NEUROLOGICAL DISORDERS (EEG-based)
   ────────────────────────────────────
   Sensors: EEG electrodes (similar to ECG but brain signals)
   Features: Band powers (delta, theta, alpha, beta, gamma),
             coherence, entropy, spike rate
   Conditions: Epilepsy (seizure detection), Alzheimer's (cognitive decline),
               Parkinson's (tremor patterns), depression
   CDS advantage: Same signal processing pipeline as ECG arrhythmia
   NOTE: EEG is structurally similar to ECG — CDS feature extraction
         methods transfer directly.

4. MUSCULOSKELETAL / MOVEMENT DISORDERS
   ──────────────────────────────────────
   Sensors: Accelerometer, gyroscope (IMU), EMG
   Features: Gait symmetry, stride length, tremor frequency/amplitude,
             muscle activation patterns
   Conditions: Parkinson's (tremor), fall risk assessment,
               rehabilitation progress, osteoarthritis
   CDS advantage: Wearable deployment (smartwatch/phone)

5. CARDIOVASCULAR (beyond arrhythmia)
   ────────────────────────────────────
   Sensors: Blood pressure cuff, pulse wave sensor, ECG
   Features: Systolic/diastolic BP, pulse wave velocity, heart rate
             variability, QT interval
   Conditions: Hypertension, heart failure (HF), coronary artery disease
   CDS advantage: Multi-sensor fusion with existing ECG infrastructure

6. MENTAL HEALTH / STRESS MONITORING
   ───────────────────────────────────
   Sensors: Wearable (HR, HRV, skin conductance, temperature)
   Features: HRV metrics (SDNN, RMSSD), electrodermal activity,
             skin temperature, sleep quality scores
   Conditions: Chronic stress, anxiety, burnout, PTSD screening
   CDS advantage: Privacy-preserving (all processing on-device)

7. LIVER / KIDNEY FUNCTION
   ─────────────────────────
   Sensors: Blood chemistry panel (lab results as input)
   Features: ALT, AST, creatinine, BUN, GFR, bilirubin, albumin
   Conditions: Liver disease (hepatitis, cirrhosis), chronic kidney disease
   CDS advantage: Well-defined healthy ranges exist in clinical practice
   NOTE: This is perhaps the most natural fit — lab values already have
         established reference ranges, directly mapping to CDS's
         [b_min, b_max] healthy boundaries.

GENERAL REQUIREMENTS FOR CDS APPLICABILITY
───────────────────────────────────────────
  ✓ Measurable features with defined healthy ranges
  ✓ Disease classes distinguishable by feature deviations
  ✓ Training data available for Bayesian model construction
  ✓ Sequential/incremental testing is meaningful (not all-at-once)

LIMITATIONS
───────────
  ✗ Imaging-based diagnosis (X-ray, MRI) — needs spatial feature extraction
    that CDS doesn't natively provide (would need CNN front-end → hybrid)
  ✗ Highly nonlinear class boundaries — CDS uses range-based thresholds,
    which are inherently linear/rectangular decision boundaries
  ✗ Very high-dimensional data without clear "healthy ranges"
    (e.g., genomics, proteomics) — CDS assumes interpretable features
""")


if __name__ == "__main__":
    run_analysis()
    print_hybrid_proposal()
    print_other_health_applications()
