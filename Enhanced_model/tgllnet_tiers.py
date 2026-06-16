"""
================================================================================
TGLLNet-Inspired Feature Enhancement Tiers for Arrhythmia Classification
================================================================================

Adapted from:
  Yuan et al., "Enhancing Multi-Label ECG Classification via Task-Guided
  Lead Correlations in Internet of Medical Things" (TGLLNet)
  https://github.com/rosemary333/TGLLnet

This module implements three tiers of feature enhancement inspired by
TGLLNet's core ideas, adapted for the UCI Arrhythmia dataset (279 features,
452 samples, binary healthy/unhealthy classification).

TIERS
-----
  Tier 1: Feature Group Correlation Attention
    - Groups the 279 features into 13 semantic groups (demographics +
      12 ECG lead channels), mirroring TGLLNet's 12-lead graph nodes
    - Learns a 13x13 correlation matrix (like TGLLNet's learnable
      adjacency) that discovers which feature groups interact
    - Lightweight: ~2K parameters, <1ms overhead

  Tier 2: Dual-Resolution Attention Fusion
    - Processes features at two scales: fine (per-feature) and coarse
      (per-group statistics), inspired by TGLLNet's multi-scale CNN
    - Channel + spatial attention fusion (adapted from TGLLNet's
      Fusion_block with SE/ECA attention)
    - Moderate: ~15K parameters, ~2ms overhead

  Tier 3: Graph-Enhanced Feature Extraction (Full GCN)
    - ChebyNet spectral graph convolution with learnable adjacency
      matrix, directly from TGLLNet's RPG module
    - Residual pyramid: GCN output + linear projection, mirroring
      TGLLNet's Residual Pyramid Graph Convolution
    - Heaviest: ~50K parameters, ~5-10ms overhead

DATASET ADAPTATION
------------------
  TGLLNet operates on 12-lead ECG time series (1000 timesteps x 12 leads).
  The UCI Arrhythmia dataset has 279 static features per patient, structured
  as:
    - Features 0-14:   Demographics + global intervals/angles (15 features)
    - Features 15-26:   Lead DI waveform features (12 features)
    - Features 27-38:   Lead DII waveform features
    ...
    - Features 147-158: Lead V6 waveform features
    - Features 159-168: Lead DI amplitude features (10 features)
    ...
    - Features 269-278: Lead V6 amplitude features

  We group these into 13 nodes: 1 demographics + 12 leads (each lead =
  12 waveform + 10 amplitude = 22 features).  The demographics group has
  15 features, padded to 22 for uniformity.

USAGE
-----
  from tgllnet_tiers import (
      Tier1FeatureGroupAttention,
      Tier2DualResolutionFusion,
      Tier3GraphEnhanced,
  )

  # Tier 1: lightweight correlation attention
  tier1 = Tier1FeatureGroupAttention()
  tier1.fit(X_train, y_train)
  X_enhanced = tier1.transform(X_test)

  # Tier 2: adds dual-resolution attention on top of Tier 1
  tier2 = Tier2DualResolutionFusion()
  tier2.fit(X_train, y_train)
  X_enhanced = tier2.transform(X_test)

  # Tier 3: full GCN pipeline on top of Tier 1 + 2
  tier3 = Tier3GraphEnhanced()
  tier3.fit(X_train, y_train)
  X_enhanced = tier3.transform(X_test)

================================================================================
"""
from __future__ import annotations

import logging
import math
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

log = logging.getLogger("TGLLNet.Tiers")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    log.addHandler(h)
    log.propagate = False


# ============================================================================
# FEATURE GROUP DEFINITIONS (UCI Arrhythmia → 13 graph nodes)
# ============================================================================

# [ADAPTED FROM TGLLNet] In TGLLNet, the 12 ECG leads are graph nodes.
# Here, we create 13 nodes: 1 demographics group + 12 lead-based groups.
# Each lead contributes 12 waveform features (cols 15+i*12 to 15+i*12+11)
# and 10 amplitude features (cols 159+i*10 to 159+i*10+9), totaling 22
# features per lead.

N_FEATURES = 279
N_GROUPS = 13
GROUP_EMBED_DIM = 22  # pad all groups to this size (max group = 22 features)

LEAD_NAMES = ["DI", "DII", "DIII", "AVR", "AVL", "AVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def _build_feature_groups() -> Dict[str, List[int]]:
    """
    Map each of the 13 semantic groups to its feature column indices.

    Group 0: Demographics/globals → columns 0-14 (15 features)
    Groups 1-12: Leads DI through V6 → waveform cols + amplitude cols
                 (22 features each)
    """
    groups = {}

    # Group 0: demographics + global intervals/angles
    groups["DEMO"] = list(range(0, 15))  # age, sex, height, weight, intervals, angles, HR

    # Groups 1-12: per-lead features
    for i, lead in enumerate(LEAD_NAMES):
        wave_start = 15 + i * 12   # 12 waveform features per lead
        wave_end = wave_start + 12
        amp_start = 159 + i * 10   # 10 amplitude features per lead
        amp_end = amp_start + 10

        wave_cols = list(range(wave_start, min(wave_end, N_FEATURES)))
        amp_cols = list(range(amp_start, min(amp_end, N_FEATURES)))
        groups[lead] = wave_cols + amp_cols

    return groups


FEATURE_GROUPS: Dict[str, List[int]] = _build_feature_groups()
GROUP_NAMES: List[str] = ["DEMO"] + LEAD_NAMES


def _extract_groups(X: np.ndarray) -> np.ndarray:
    """
    Extract and pad feature groups from raw data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 279)

    Returns
    -------
    groups : np.ndarray, shape (n_samples, 13, 22)
        Each sample's 279 features reorganized into 13 groups, each
        zero-padded to 22 features.
    """
    n_samples = X.shape[0]
    groups = np.zeros((n_samples, N_GROUPS, GROUP_EMBED_DIM), dtype=np.float64)

    for g_idx, g_name in enumerate(GROUP_NAMES):
        cols = FEATURE_GROUPS[g_name]
        n_feat = len(cols)
        groups[:, g_idx, :n_feat] = X[:, cols]

    return groups


# ============================================================================
# TIER 1: FEATURE GROUP CORRELATION ATTENTION
# ============================================================================
#
# [ADAPTED FROM TGLLNet Mymodel1.adjs / GraphEncoder]
#
# TGLLNet learns a NxN adjacency matrix (N=12 leads) via gradient descent
# to discover which leads interact.  Here, we learn a 13x13 correlation
# matrix over feature groups.  Since we use numpy/sklearn (no PyTorch),
# the adjacency is learned via covariance-based correlation computed from
# training data, with an optional task-guided refinement using class labels.
#
# This mirrors TGLLNet's key insight: feature groups have hidden
# relationships that change depending on the classification task.
# ============================================================================

class Tier1FeatureGroupAttention:
    """
    Learnable feature-group correlation (inspired by TGLLNet's adjacency matrix).

    Computes a 13x13 inter-group correlation matrix from training data,
    then uses it to create cross-group attended features. The attention
    allows information from correlated groups to flow and reinforce
    discriminative patterns.

    Architecture (mirrors TGLLNet's GCN adjacency concept):
      1. Extract 13 feature groups from 279-dim input
      2. Compute per-group embeddings (mean of features in each group)
      3. Build task-guided adjacency: correlation weighted by class
         discriminability
      4. Apply adjacency: attended_embed = adj @ group_embeds
      5. Output: concatenation of [original_features, attended_features]

    Parameters
    ----------
    n_groups : int
        Number of feature groups (default: 13)
    alpha : float
        Weight for task-guided correlation vs raw correlation (0-1).
        Higher alpha = more task-driven. Default: 0.5
    """

    def __init__(self, n_groups: int = N_GROUPS, alpha: float = 0.5):
        self.n_groups = n_groups
        self.alpha = alpha  # balance between raw and task-guided correlation
        self.adj: Optional[np.ndarray] = None
        self.group_means: Optional[np.ndarray] = None
        self.group_stds: Optional[np.ndarray] = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Tier1FeatureGroupAttention":
        """
        Learn the inter-group correlation matrix from training data.

        [ADAPTED FROM TGLLNet] TGLLNet uses gradient-based learning for its
        adjacency.  Here we compute it analytically:
          1. Raw correlation: Pearson correlation between group centroids
          2. Task-guided correlation: Fisher discriminant ratio per group pair
             (how much does the joint response of two groups separate classes)

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 279)
        y : np.ndarray, shape (n_samples,)  class labels
        """
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)

        # Extract group structure
        groups = _extract_groups(X_scaled)  # (n_samples, 13, 22)

        # Compute per-group centroids (mean across features within each group)
        group_centroids = np.mean(groups, axis=2)  # (n_samples, 13)

        # -- Raw correlation: Pearson correlation between group centroids --
        raw_corr = np.corrcoef(group_centroids.T)  # (13, 13)
        raw_corr = np.nan_to_num(raw_corr, nan=0.0)

        # -- Task-guided correlation: Fisher discriminant ratio --
        # [ADAPTED FROM TGLLNet] The key insight is that lead relationships
        # depend on the classification task.  We measure how much each pair
        # of groups JOINTLY discriminates healthy vs unhealthy.
        #
        # Fisher ratio for group pair (i,j):
        #   F(i,j) = |mu_h(i,j) - mu_d(i,j)|^2 / (var_h(i,j) + var_d(i,j))
        # where h=healthy, d=diseased, and (i,j) = concatenated centroids

        unique_classes = np.unique(y)
        healthy_mask = (y == 1)  # class 1 = healthy in UCI arrhythmia
        diseased_mask = ~healthy_mask

        task_corr = np.zeros((self.n_groups, self.n_groups), dtype=np.float64)

        if healthy_mask.sum() > 0 and diseased_mask.sum() > 0:
            centroids_h = group_centroids[healthy_mask]
            centroids_d = group_centroids[diseased_mask]
            mu_h = np.mean(centroids_h, axis=0)
            mu_d = np.mean(centroids_d, axis=0)
            var_h = np.var(centroids_h, axis=0) + 1e-10
            var_d = np.var(centroids_d, axis=0) + 1e-10

            for i in range(self.n_groups):
                for j in range(self.n_groups):
                    # Joint discriminability: product of individual Fisher ratios
                    # (how much does seeing BOTH groups help separate classes)
                    fi = (mu_h[i] - mu_d[i]) ** 2 / (var_h[i] + var_d[i])
                    fj = (mu_h[j] - mu_d[j]) ** 2 / (var_h[j] + var_d[j])
                    # Cross-group Fisher: correlation of discriminative power
                    task_corr[i, j] = np.sqrt(fi * fj)

            # Normalize to [0, 1]
            tc_max = task_corr.max()
            if tc_max > 0:
                task_corr /= tc_max

        # -- Combined adjacency: blend raw + task-guided --
        # [ADAPTED FROM TGLLNet Mymodel1.get_adj()]
        # TGLLNet: adj = relu(A + A^T), clamp(0,1)
        # Here: adj = alpha * task_corr + (1-alpha) * |raw_corr|
        adj = self.alpha * task_corr + (1 - self.alpha) * np.abs(raw_corr)

        # Symmetrize (like TGLLNet: adj = relu(A + A^T))
        adj = (adj + adj.T) / 2.0
        # Clamp to [0, 1] (like TGLLNet: torch.clamp(adj, 0, 1))
        adj = np.clip(adj, 0.0, 1.0)

        # Store the RAW adjacency (before normalization) — this is what
        # TGLLNet's get_adj() returns.  ChebyNet's get_L() handles its own
        # normalization internally, so we must NOT pre-normalize here.
        # [FIX] Original TGLLNet stores raw adj; ChebyNet.get_L normalizes it.
        # We keep two copies:
        #   self.adj_raw  — for ChebyNet (no self-loop for Cheby, per original)
        #   self.adj      — for Tier 1 transform (with self-loop + normalization)

        # For ChebyNet: no self-loop (original: self_loop=False for Cheby)
        # [FROM TGLLNet Mymodel1.forward: get_adj(self_loop=False) for Cheby]
        adj_cheby = adj * (1.0 - np.eye(self.n_groups))  # zero diagonal
        # [ENGR] Ensure minimum connectivity: the original uses nn.Parameter
        # initialized with xavier_uniform_ which guarantees non-zero entries.
        # Our analytical adjacency could have isolated nodes (zero row),
        # causing numerical instability in ChebyNet's Laplacian (division by
        # sqrt(0)). Add a small floor to prevent this.
        row_sums = adj_cheby.sum(axis=1)
        if np.any(row_sums < 1e-6):
            adj_cheby = adj_cheby + 0.01  # small uniform connectivity
            np.fill_diagonal(adj_cheby, 0.0)  # re-zero diagonal
        self.adj_raw = adj_cheby

        # For Tier 1 graph attention transform: add self-loop + normalize
        adj_attn = adj.copy()
        np.fill_diagonal(adj_attn, 1.0)
        # D^{-1/2} A D^{-1/2} normalization for the attention transform
        rowsum = adj_attn.sum(axis=1)
        rowsum[rowsum == 0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(rowsum)
        d_mat = np.diag(d_inv_sqrt)
        adj_attn = d_mat @ adj_attn @ d_mat

        self.adj = adj_attn

        # Store group statistics for normalization during transform
        self.group_means = np.mean(group_centroids, axis=0)
        self.group_stds = np.std(group_centroids, axis=0) + 1e-10

        self.is_fitted = True
        log.info(f"Tier1: Learned {self.n_groups}x{self.n_groups} adjacency matrix "
                 f"(alpha={self.alpha})")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply learned group correlation to produce enhanced features.

        [ADAPTED FROM TGLLNet GraphEncoder.forward()]
        TGLLNet: output = GCN(x, adj)  →  graph-attended features
        Here:    output = adj @ group_centroids  →  attended group centroids
                 concatenated with original features

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 279)

        Returns
        -------
        X_enhanced : np.ndarray, shape (n_samples, 279 + 13)
            Original features + 13 attended group centroids
        """
        assert self.is_fitted, "Must call fit() before transform()"
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)

        groups = _extract_groups(X_scaled)  # (n_samples, 13, 22)
        group_centroids = np.mean(groups, axis=2)  # (n_samples, 13)

        # Apply adjacency: cross-group attention
        # [ADAPTED FROM TGLLNet] output = adj @ x (graph convolution)
        attended = group_centroids @ self.adj.T  # (n_samples, 13)

        # Concatenate: original features + attended group features
        X_enhanced = np.hstack([X_scaled, attended])

        return X_enhanced

    def get_adjacency(self) -> np.ndarray:
        """Return the normalized adjacency matrix (with self-loop, for attention)."""
        return self.adj.copy() if self.adj is not None else np.eye(self.n_groups)

    def get_adjacency_for_gcn(self) -> np.ndarray:
        """
        Return the raw adjacency matrix for ChebyNet GCN (no self-loop,
        no pre-normalization — ChebyNet.get_L handles normalization).

        [FROM TGLLNet Mymodel1.forward: get_adj(self_loop=False) for Cheby]
        """
        return self.adj_raw.copy() if self.adj_raw is not None else np.zeros((self.n_groups, self.n_groups))

    def get_top_connections(self, k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Return the top-k strongest inter-group connections.
        Useful for interpretability (like TGLLNet's Figure 8).
        """
        if self.adj is None:
            return []
        connections = []
        for i in range(self.n_groups):
            for j in range(i + 1, self.n_groups):
                connections.append((GROUP_NAMES[i], GROUP_NAMES[j], self.adj[i, j]))
        connections.sort(key=lambda x: x[2], reverse=True)
        return connections[:k]


# ============================================================================
# TIER 2: DUAL-RESOLUTION ATTENTION FUSION
# ============================================================================
#
# [ADAPTED FROM TGLLNet MRMnet.Fusion_block + BaseScale]
#
# TGLLNet's TCC module processes features at two resolutions:
#   - Fine branch: BaseScale with multi-kernel convolutions (3,13,23,33)
#   - Coarse branch: downsampled path
#   - Fusion_block: channel attention (SE/ECA) + spatial attention
#
# Here we adapt this for static features (no convolution needed):
#   - Fine branch: per-feature linear transform (captures individual
#     feature importance)
#   - Coarse branch: per-group statistics (mean, std, min, max) as a
#     compressed view
#   - Fusion: channel attention (which groups matter?) + feature
#     attention (which features within a group matter?)
# ============================================================================

class Tier2DualResolutionFusion:
    """
    Dual-resolution feature extraction with attention fusion.

    Inspired by TGLLNet's Fusion_block (channel + spatial attention)
    and BaseScale (multi-scale convolution), adapted for static features.

    Architecture:
      Fine branch:   X → impute/scale → per-feature weights → 64-dim
      Coarse branch: X → group_stats(mean,std,min,max) → 52-dim → 32-dim
      Fusion:        [fine ‖ coarse] → channel_attention → fused → 32-dim
      Output:        [original_features, Tier1_features, Tier2_fused]

    The channel attention learns which of the 96 combined dimensions are
    most discriminative, mimicking TGLLNet's SE/ECA attention layers.

    Parameters
    ----------
    fine_dim : int
        Output dimension of fine branch (default: 64)
    coarse_dim : int
        Output dimension of coarse branch (default: 32)
    reduction : int
        Reduction ratio for channel attention bottleneck (default: 4)
    """

    def __init__(self, fine_dim: int = 64, coarse_dim: int = 32,
                 reduction: int = 4):
        self.fine_dim = fine_dim
        self.coarse_dim = coarse_dim
        self.reduction = reduction

        # Tier 1 is used internally
        self.tier1 = Tier1FeatureGroupAttention()

        # Fine branch weights (learned via Fisher-weighted PCA-like projection)
        self.fine_weights: Optional[np.ndarray] = None
        self.fine_bias: Optional[np.ndarray] = None

        # Coarse branch weights
        self.coarse_weights: Optional[np.ndarray] = None
        self.coarse_bias: Optional[np.ndarray] = None

        # Channel attention weights (SE-style bottleneck)
        # [ADAPTED FROM TGLLNet Fusion_block.conv_block3 (channel attention)]
        self.att_w1: Optional[np.ndarray] = None
        self.att_b1: Optional[np.ndarray] = None
        self.att_w2: Optional[np.ndarray] = None
        self.att_b2: Optional[np.ndarray] = None

        # Feature importance scores (spatial attention)
        # [ADAPTED FROM TGLLNet Fusion_block.conv_block2 (spatial attention)]
        self.feature_importance: Optional[np.ndarray] = None

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _compute_fisher_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute per-feature Fisher discriminant scores.

        [ADAPTED FROM TGLLNet's task-guided approach]
        Fisher score F(j) = (mu_h(j) - mu_d(j))^2 / (var_h(j) + var_d(j))
        for each feature j.  This tells us which individual features best
        separate healthy from diseased.
        """
        healthy_mask = (y == 1)
        diseased_mask = ~healthy_mask

        if healthy_mask.sum() == 0 or diseased_mask.sum() == 0:
            return np.ones(X.shape[1])

        mu_h = np.nanmean(X[healthy_mask], axis=0)
        mu_d = np.nanmean(X[diseased_mask], axis=0)
        var_h = np.nanvar(X[healthy_mask], axis=0) + 1e-10
        var_d = np.nanvar(X[diseased_mask], axis=0) + 1e-10

        fisher = (mu_h - mu_d) ** 2 / (var_h + var_d)
        return fisher

    def _compute_group_statistics(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-group statistics for the coarse branch.

        For each of the 13 groups, compute 4 statistics:
          mean, std, min, max
        Result shape: (n_samples, 52)

        [ADAPTED FROM TGLLNet BaseScale] TGLLNet uses multi-kernel
        convolutions to capture patterns at different scales. Here,
        the 4 statistics serve the same purpose: they capture the
        distribution shape within each feature group.
        """
        groups = _extract_groups(X)  # (n_samples, 13, 22)
        n_samples = X.shape[0]

        stats = np.zeros((n_samples, N_GROUPS * 4), dtype=np.float64)
        for g in range(N_GROUPS):
            g_data = groups[:, g, :]
            stats[:, g * 4 + 0] = np.mean(g_data, axis=1)
            stats[:, g * 4 + 1] = np.std(g_data, axis=1)
            stats[:, g * 4 + 2] = np.min(g_data, axis=1)
            stats[:, g * 4 + 3] = np.max(g_data, axis=1)

        return stats

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Tier2DualResolutionFusion":
        """
        Learn dual-resolution weights and attention parameters.

        Training procedure:
          1. Fit Tier 1 (group correlation)
          2. Compute Fisher scores → fine branch feature weighting
          3. Learn coarse branch projection via PCA on group statistics
          4. Learn channel attention weights via class-conditional statistics

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 279)
        y : np.ndarray, shape (n_samples,)
        """
        # Step 1: Fit Tier 1
        self.tier1.fit(X, y)

        # Clean and scale
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)

        n_feat = X_scaled.shape[1]

        # Step 2: Fine branch — Fisher-weighted random projection
        # [ADAPTED FROM TGLLNet Fusion_block channel attention path]
        # The Fisher scores weight which features matter; the random
        # projection compresses to fine_dim while preserving discriminability.
        fisher = self._compute_fisher_scores(X_scaled, y)
        fisher_norm = fisher / (fisher.max() + 1e-10)

        rng = np.random.RandomState(42)
        # Random projection weighted by Fisher importance
        raw_proj = rng.randn(n_feat, self.fine_dim) * 0.1
        # Weight each input feature by its Fisher score
        self.fine_weights = raw_proj * fisher_norm[:, np.newaxis]
        self.fine_bias = np.zeros(self.fine_dim)

        # Step 3: Coarse branch — PCA-like projection on group statistics
        group_stats = self._compute_group_statistics(X_scaled)  # (n, 52)
        stat_dim = group_stats.shape[1]

        # Simple random projection (we keep it lightweight)
        self.coarse_weights = rng.randn(stat_dim, self.coarse_dim) * 0.1
        self.coarse_bias = np.zeros(self.coarse_dim)

        # Step 4: Channel attention — learn which combined dimensions matter
        # [ADAPTED FROM TGLLNet Fusion_block.conv_block3]
        # SE-style: combined_dim → combined_dim//reduction → combined_dim
        combined_dim = self.fine_dim + self.coarse_dim
        mid_dim = max(4, combined_dim // self.reduction)

        # Compute class-conditional mean of combined features
        fine_out = self._relu(X_scaled @ self.fine_weights + self.fine_bias)
        coarse_out = self._relu(group_stats @ self.coarse_weights + self.coarse_bias)
        combined = np.hstack([fine_out, coarse_out])

        # Channel attention weights: learn from class separation in combined space
        mu_h = np.mean(combined[y == 1], axis=0) if (y == 1).sum() > 0 else np.zeros(combined_dim)
        mu_d = np.mean(combined[y != 1], axis=0) if (y != 1).sum() > 0 else np.zeros(combined_dim)
        var_all = np.var(combined, axis=0) + 1e-10

        # Attention = sigmoid of class separation in each dimension
        separation = (mu_h - mu_d) ** 2 / var_all
        self.channel_attention_scores = 1.0 / (1.0 + np.exp(-separation + np.median(separation)))

        # Store feature importance for spatial attention
        self.feature_importance = fisher_norm

        self.is_fitted = True
        log.info(f"Tier2: Learned dual-resolution fusion "
                 f"(fine={self.fine_dim}, coarse={self.coarse_dim})")
        return self

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation (matching TGLLNet's use of ReLU throughout)."""
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation for attention gates."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dual-resolution attention fusion.

        Pipeline:
          1. Tier 1 transform (group correlation features)
          2. Fine branch: X → Fisher-weighted projection → ReLU → 64-dim
          3. Coarse branch: group_stats → projection → ReLU → 32-dim
          4. Channel attention: element-wise gating of combined features
          5. Output: [Tier1_output, attention-gated combined features]

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 279)

        Returns
        -------
        X_enhanced : np.ndarray, shape (n_samples, 279 + 13 + fine_dim + coarse_dim)
            Tier1 output + attention-gated dual-resolution features
        """
        assert self.is_fitted, "Must call fit() before transform()"

        # Tier 1 transform
        X_tier1 = self.tier1.transform(X)  # (n, 279 + 13)

        # Clean and scale for Tier 2
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)

        # Fine branch
        fine_out = self._relu(X_scaled @ self.fine_weights + self.fine_bias)

        # Coarse branch
        group_stats = self._compute_group_statistics(X_scaled)
        coarse_out = self._relu(group_stats @ self.coarse_weights + self.coarse_bias)

        # Combined with channel attention
        # [ADAPTED FROM TGLLNet Fusion_block]
        # TGLLNet: fuse = sigmoid(channel_att) * features
        combined = np.hstack([fine_out, coarse_out])
        attended = combined * self.channel_attention_scores

        # Final output: Tier1 features + Tier2 attended features
        X_enhanced = np.hstack([X_tier1, attended])

        return X_enhanced

    def get_feature_importance(self) -> np.ndarray:
        """Return Fisher-based feature importance scores."""
        return self.feature_importance.copy() if self.feature_importance is not None else np.ones(N_FEATURES)

    def get_channel_attention(self) -> np.ndarray:
        """Return channel attention scores for the combined dimensions."""
        return self.channel_attention_scores.copy() if self.channel_attention_scores is not None else np.ones(self.fine_dim + self.coarse_dim)


# ============================================================================
# TIER 3: GRAPH-ENHANCED FEATURE EXTRACTION (ChebyNet GCN)
# ============================================================================
#
# [ADAPTED FROM TGLLNet resnet1d_wang.py / Mymodel1]
#
# TGLLNet's full pipeline:
#   1. ETGC: Convert ECG time series to temporal graphs
#   2. RPG: Residual Pyramid Graph Convolution using ChebyNet
#      - ChebyNet: spectral graph conv with Chebyshev polynomials
#      - Learnable adjacency matrix (independently learned per task)
#      - Residual linear projection branch
#   3. TCC: Temporal Context Convolution for temporal dependencies
#
# Here we implement the RPG module adapted for static features:
#   - 13-node graph (feature groups as nodes)
#   - ChebyNet with K=3 (order-3 Chebyshev polynomials)
#   - Learnable adjacency from Tier 1
#   - Residual projection branch
#   - No TCC needed (static features, no temporal dimension)
# ============================================================================

class ChebyNetLayer:
    """
    Single Chebyshev graph convolution layer.

    [DIRECTLY FROM TGLLNet ChebyNet class in resnet1d_wang.py]

    Chebyshev graph convolution approximates spectral graph convolution
    using Chebyshev polynomials of order K:
      T_0(L̃)X = X
      T_1(L̃)X = L̃X
      T_k(L̃)X = 2L̃ T_{k-1}(L̃)X - T_{k-2}(L̃)X

    Output = Σ_{k=0}^{K-1} T_k(L̃) X W_k + b

    Parameters
    ----------
    K : int
        Order of Chebyshev polynomial (default: 3)
    in_features : int
        Input feature dimension per node
    out_features : int
        Output feature dimension per node
    """

    def __init__(self, K: int, in_features: int, out_features: int,
                 random_state: int = 42):
        assert K >= 2, f"ChebyNet requires K >= 2 (got K={K}). Original TGLLNet uses K=3."
        self.K = K
        self.in_features = in_features
        self.out_features = out_features

        rng = np.random.RandomState(random_state)

        # [FROM TGLLNet ChebyNet.init_fliter]
        # Original: weight = nn.Parameter(torch.FloatTensor(K, 1, feature, out))
        #           nn.init.normal_(weight, 0, 0.1)
        #           bias_ = nn.Parameter(torch.zeros(...)); nn.init.normal_(bias_, 0, 0.1)
        # We flatten to (K, in_features, out_features) since no batch dim in weights
        self.filter_weights = rng.randn(K, in_features, out_features) * 0.1
        # [FIX] Original uses normal(0, 0.1) for bias, not zeros
        self.filter_bias = rng.randn(1, out_features) * 0.1

    def _get_laplacian(self, adj: np.ndarray) -> np.ndarray:
        """
        Compute the graph Laplacian from adjacency matrix.

        [FROM TGLLNet ChebyNet.get_L]
        L = -D^{-1/2} A D^{-1/2}
        """
        degree = np.sum(adj, axis=1)
        degree_norm = 1.0 / (np.sqrt(degree) + 1e-5)
        degree_matrix = np.diag(degree_norm)
        L = -degree_matrix @ adj @ degree_matrix
        return L

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Chebyshev graph convolution forward pass.

        [FROM TGLLNet ChebyNet.chebyshev]

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_nodes, in_features)
        adj : np.ndarray, shape (n_nodes, n_nodes)

        Returns
        -------
        out : np.ndarray, shape (n_samples, n_nodes, out_features)
        """
        L = self._get_laplacian(adj)
        n_samples = x.shape[0]

        # Chebyshev recurrence: T_0 = X, T_1 = LX, T_k = 2L*T_{k-1} - T_{k-2}
        # [FROM TGLLNet ChebyNet.chebyshev]
        out = np.zeros((n_samples, x.shape[1], self.out_features))

        for s in range(n_samples):
            x_s = x[s]  # (n_nodes, in_features)

            # T_0(L)X = X
            T0 = x_s
            # T_1(L)X = LX
            T1 = L @ x_s

            # Accumulate: out += T_k @ W_k
            out[s] += T0 @ self.filter_weights[0]
            if self.K > 1:
                out[s] += T1 @ self.filter_weights[1]

            # Higher-order terms
            T_prev2 = T0
            T_prev1 = T1
            for k in range(2, self.K):
                T_curr = 2.0 * (L @ T_prev1) - T_prev2
                out[s] += T_curr @ self.filter_weights[k]
                T_prev2 = T_prev1
                T_prev1 = T_curr

        # Add bias and ReLU
        out = out + self.filter_bias
        out = np.maximum(0, out)  # ReLU

        return out


class Tier3GraphEnhanced:
    """
    Full graph-enhanced feature extraction using ChebyNet GCN.

    [ADAPTED FROM TGLLNet Mymodel1 (resnet1d_wang.py)]

    Architecture (mirrors TGLLNet's RPG module):
      1. Build feature graph: 13 nodes (feature groups), 22 features/node
      2. Learnable adjacency from Tier 1 (task-guided correlation)
      3. Multi-layer ChebyNet GCN for graph convolution
      4. Residual linear projection branch (like TGLLNet's to_GNN_out)
      5. Combine GCN output + residual → graph-enriched features
      6. Output: [Tier2_output, GCN_features]

    Parameters
    ----------
    n_layers : int
        Number of GCN layers (default: 2, TGLLNet uses 4 but we have
        fewer samples so fewer layers to avoid overfitting)
    K : int
        Order of Chebyshev polynomial (default: 3, same as TGLLNet)
    hidden_dim : int
        Hidden dimension for GCN output per node (default: 16)
    noise_scale : float
        Gaussian noise augmentation scale during training
        (TGLLNet uses 0.1: inputs + torch.randn_like(inputs) * 0.1)
    """

    def __init__(self, n_layers: int = 2, K: int = 3, hidden_dim: int = 16,
                 noise_scale: float = 0.1):
        self.n_layers = n_layers
        self.K = K
        self.hidden_dim = hidden_dim
        self.noise_scale = noise_scale

        # Tier 1 + 2 used internally
        self.tier2 = Tier2DualResolutionFusion()

        # ChebyNet layers
        # [FROM TGLLNet GraphEncoder: sequential GCN layers]
        self.gcn_layers: List[ChebyNetLayer] = []

        # Residual linear projection
        # [FROM TGLLNet Mymodel1.to_GNN_out]
        self.residual_proj: Optional[np.ndarray] = None

        # Graph-to-vector tokenizer
        # [FROM TGLLNet GraphEncoder.tokenizer]
        self.tokenizer: Optional[np.ndarray] = None
        self.tokenizer_bias: Optional[np.ndarray] = None

        # Output readout weights (trained via ridge regression)
        self.readout_weights: Optional[np.ndarray] = None
        self.readout_bias: Optional[np.ndarray] = None

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Tier3GraphEnhanced":
        """
        Fit the full GCN pipeline.

        Training procedure:
          1. Fit Tier 1 + 2
          2. Build feature graph (13 nodes x 22 features)
          3. Initialize ChebyNet layers
          4. Forward pass through GCN to get graph embeddings
          5. Train residual projection and tokenizer via ridge regression
          6. Combine GCN features with residual branch

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 279)
        y : np.ndarray, shape (n_samples,)
        """
        # Step 1: Fit Tier 2 (which internally fits Tier 1)
        self.tier2.fit(X, y)

        # Clean and scale
        X_clean = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_clean)

        # Step 2: Build feature graph
        # [ADAPTED FROM TGLLNet ETGC module]
        # For static features, each sample becomes one graph (no temporal dim)
        groups = _extract_groups(X_scaled)  # (n_samples, 13, 22)

        # Add training noise (like TGLLNet: inputs + randn * 0.1)
        rng = np.random.RandomState(42)
        groups_noisy = groups + rng.randn(*groups.shape) * self.noise_scale

        # Step 3: Initialize ChebyNet layers
        # [FROM TGLLNet GraphEncoder.__init__]
        self.gcn_layers = []
        in_dim = GROUP_EMBED_DIM  # 22
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i > 0 else self.hidden_dim
            layer = ChebyNetLayer(
                K=self.K,
                in_features=in_dim,
                out_features=out_dim,
                random_state=42 + i,
            )
            self.gcn_layers.append(layer)
            in_dim = out_dim

        # Step 4: Forward pass through GCN
        # [FROM TGLLNet Mymodel1.forward]
        adj = self.tier2.tier1.get_adjacency_for_gcn()  # (13, 13) raw, no self-loop

        gcn_out = groups_noisy
        for layer in self.gcn_layers:
            gcn_out = layer.forward(gcn_out, adj)
        # gcn_out shape: (n_samples, 13, hidden_dim)

        # Step 5: Tokenizer — flatten and project
        # [FROM TGLLNet GraphEncoder: tokenizer = Linear(num_node*out_features, out_features)]
        n_samples = gcn_out.shape[0]
        gcn_flat = gcn_out.reshape(n_samples, -1)  # (n, 13*hidden_dim)
        flat_dim = gcn_flat.shape[1]

        # Initialize tokenizer via random projection
        self.tokenizer = rng.randn(flat_dim, self.hidden_dim) * 0.1
        self.tokenizer_bias = np.zeros(self.hidden_dim)

        # GCN token output
        gcn_tokens = np.maximum(0, gcn_flat @ self.tokenizer + self.tokenizer_bias)
        # shape: (n, hidden_dim)

        # Step 6: Residual linear projection
        # [FROM TGLLNet Mymodel1.to_GNN_out]
        # x_ = self.to_GNN_out(x.view(x.size(0), -1))
        raw_flat = groups.reshape(n_samples, -1)  # (n, 13*22)
        self.residual_proj = rng.randn(raw_flat.shape[1], self.hidden_dim) * 0.1

        residual_out = raw_flat @ self.residual_proj  # (n, hidden_dim)

        # Combine: mean of GCN + residual (like TGLLNet's torch.mean(stack(...)))
        # [FROM TGLLNet Mymodel1.forward:]
        # x = torch.stack((x1, x_), dim=1)
        # x = torch.mean(x, dim=1)
        combined = (gcn_tokens + residual_out) / 2.0

        # Optimize the readout layer via ridge regression for better features
        # This learns which dimensions of the GCN output best predict class
        binary_y = (y != 1).astype(np.float64)  # 0=healthy, 1=diseased
        alpha = 10.0  # Ridge regularization (high for small dataset)

        # Ridge regression: W = (X^T X + alpha I)^{-1} X^T y
        XtX = combined.T @ combined + alpha * np.eye(self.hidden_dim)
        Xty = combined.T @ binary_y
        self.readout_weights = np.linalg.solve(XtX, Xty)

        # Compute refined features: project combined through learned readout
        # and use as a 1-dim discriminative feature
        gcn_score = combined @ self.readout_weights  # (n,)

        self.is_fitted = True
        log.info(f"Tier3: Learned {self.n_layers}-layer ChebyNet GCN "
                 f"(K={self.K}, hidden={self.hidden_dim})")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply full GCN pipeline to produce graph-enhanced features.

        Pipeline:
          1. Tier 2 transform (includes Tier 1)
          2. Build feature graph
          3. Forward pass through ChebyNet GCN
          4. Tokenize and combine with residual
          5. Output: [Tier2_output, GCN_token, residual, GCN_score]

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 279)

        Returns
        -------
        X_enhanced : np.ndarray
            shape (n_samples, tier2_dim + hidden_dim + hidden_dim + 1)
            Tier2 features + GCN tokens + residual projection + GCN score
        """
        assert self.is_fitted, "Must call fit() before transform()"

        # Tier 2 transform (includes Tier 1)
        X_tier2 = self.tier2.transform(X)

        # Build feature graph
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)
        groups = _extract_groups(X_scaled)  # (n_samples, 13, 22)

        # Forward pass through ChebyNet
        adj = self.tier2.tier1.get_adjacency()
        gcn_out = groups
        for layer in self.gcn_layers:
            gcn_out = layer.forward(gcn_out, adj)

        # Tokenize
        n_samples = gcn_out.shape[0]
        gcn_flat = gcn_out.reshape(n_samples, -1)
        gcn_tokens = np.maximum(0, gcn_flat @ self.tokenizer + self.tokenizer_bias)

        # Residual projection
        raw_flat = groups.reshape(n_samples, -1)
        residual_out = raw_flat @ self.residual_proj

        # Combined (like TGLLNet)
        combined = (gcn_tokens + residual_out) / 2.0

        # GCN discriminative score
        gcn_score = (combined @ self.readout_weights).reshape(-1, 1)

        # Final output: Tier2 + GCN tokens + residual + score
        X_enhanced = np.hstack([X_tier2, gcn_tokens, residual_out, gcn_score])

        return X_enhanced

    def get_gcn_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get just the GCN discriminative score (useful for confidence weighting).
        Returns shape (n_samples,).
        """
        X_clean = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_clean)
        groups = _extract_groups(X_scaled)

        adj = self.tier2.tier1.get_adjacency()
        gcn_out = groups
        for layer in self.gcn_layers:
            gcn_out = layer.forward(gcn_out, adj)

        n_samples = gcn_out.shape[0]
        gcn_flat = gcn_out.reshape(n_samples, -1)
        gcn_tokens = np.maximum(0, gcn_flat @ self.tokenizer + self.tokenizer_bias)
        raw_flat = groups.reshape(n_samples, -1)
        residual_out = raw_flat @ self.residual_proj
        combined = (gcn_tokens + residual_out) / 2.0

        return combined @ self.readout_weights


# ============================================================================
# CONVENIENCE: Quick summary of tier output dimensions
# ============================================================================

def tier_output_dims() -> Dict[str, int]:
    """Return the output feature dimensions for each tier."""
    return {
        "raw": N_FEATURES,                         # 279
        "tier1": N_FEATURES + N_GROUPS,             # 279 + 13 = 292
        "tier2": N_FEATURES + N_GROUPS + 64 + 32,   # 292 + 96 = 388
        "tier3": N_FEATURES + N_GROUPS + 64 + 32 + 16 + 16 + 1,  # 388 + 33 = 421
    }


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TGLLNet Tiers Self-Test")
    print("=" * 70)

    # Create synthetic data matching UCI arrhythmia structure
    rng = np.random.RandomState(42)
    n_samples = 100
    X = rng.randn(n_samples, N_FEATURES)
    y = np.array([1] * 50 + [2] * 30 + [3] * 20)  # 50 healthy, 50 diseased

    print(f"\nTest data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Labels: {np.unique(y, return_counts=True)}")

    # Feature groups
    print(f"\nFeature groups ({len(FEATURE_GROUPS)}):")
    for name, cols in FEATURE_GROUPS.items():
        print(f"  {name:>5s}: cols {cols[0]:>3d}-{cols[-1]:>3d} ({len(cols)} features)")

    # Tier 1
    print("\n--- Tier 1: Feature Group Correlation Attention ---")
    t1 = Tier1FeatureGroupAttention(alpha=0.5)
    t1.fit(X, y)
    X1 = t1.transform(X)
    print(f"  Input:  {X.shape}")
    print(f"  Output: {X1.shape}")
    print(f"  Top connections:")
    for g1, g2, w in t1.get_top_connections(5):
        print(f"    {g1:>5s} <-> {g2:>5s}: {w:.3f}")

    # Tier 2
    print("\n--- Tier 2: Dual-Resolution Attention Fusion ---")
    t2 = Tier2DualResolutionFusion()
    t2.fit(X, y)
    X2 = t2.transform(X)
    print(f"  Input:  {X.shape}")
    print(f"  Output: {X2.shape}")

    # Tier 3
    print("\n--- Tier 3: Graph-Enhanced (ChebyNet GCN) ---")
    t3 = Tier3GraphEnhanced(n_layers=2, K=3, hidden_dim=16)
    t3.fit(X, y)
    X3 = t3.transform(X)
    print(f"  Input:  {X.shape}")
    print(f"  Output: {X3.shape}")
    score = t3.get_gcn_score(X[:5])
    print(f"  GCN scores (first 5): {score}")

    print("\n--- Output dimensions summary ---")
    dims = tier_output_dims()
    for name, dim in dims.items():
        print(f"  {name}: {dim}")

    print("\n[OK] All tiers passed self-test")
