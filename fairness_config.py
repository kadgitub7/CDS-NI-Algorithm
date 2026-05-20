"""
fairness_config.py
==================
Centralized configuration for all bias mitigation toggles in the CDS pipeline.

Set flags here to control which fairness interventions are active.  Every
combination of True/False across these flags is valid, allowing you to
compare baselines against individual or combined mitigation strategies.

USAGE
-----
  # Baseline (original paper algorithm, no modifications):
  #   Set ALL flags to False.
  #
  # Individual strategies:
  #   ENABLE_REWEIGHING  = True   (pre-processing only)
  #   ENABLE_FAIRNESS_RL = True   (in-processing only)
  #   ENABLE_ADVERSARIAL_DEBIASING = True  (post-processing only)
  #
  # Combined strategies:
  #   ENABLE_REWEIGHING + ENABLE_FAIRNESS_RL = True  (pre + in)
  #   All three = True  (full mitigation stack)

After changing flags, re-run the desired script (e.g. reproducibilityReport.py).
Algorithm2 and Algorithm4 import these values at startup.
"""

# ─────────────────────────────────────────────────────────────────────────────
# PRE-PROCESSING: Reweighing (Kamiran & Calders, 2012)
# ─────────────────────────────────────────────────────────────────────────────
# Assigns instance weights W(S=s, Y=y) = P(S)*P(Y) / P(S,Y) to training
# examples so that the weighted joint distribution is independent of the
# protected attribute.  Applied in Algorithm 2's action weight computation.
ENABLE_REWEIGHING: bool = True

# ─────────────────────────────────────────────────────────────────────────────
# IN-PROCESSING: Fairness-constrained RL reward (modified Algorithm 4)
# ─────────────────────────────────────────────────────────────────────────────
# Modifies the RL action-selection reward:
#   rw_modified = rw_original + lambda * |AF_male_contrib - AF_female_contrib|
# Higher lambda = stronger fairness constraint at cost of accuracy.
ENABLE_FAIRNESS_RL: bool = False

# Lambda: fairness penalty weight.  Grid search range: 0.01 to 1.0.
FAIRNESS_LAMBDA: float = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING: Adversarial Debiasing (Zhang et al., 2018 adapted)
# ─────────────────────────────────────────────────────────────────────────────
# Trains a lightweight adversary network that tries to predict the protected
# attribute (sex) from the CDS prediction outputs.  The adversary's loss is
# used to adjust per-group decision thresholds so that the final predictions
# are less predictive of gender.
#
# Architecture (adapted for CDS):
#   - Predictor:   CDS Algorithm 4 (frozen — not retrained)
#   - Adversary:   small MLP (AF_real, rw_real, n_actions) -> P(sex)
#   - Debiasing:   adjust per-group DIAGNOSTIC_THRESHOLD so adversary
#                  accuracy approaches chance (50%)
#
# This is a TRUE adversarial debiasing implementation, not just the RL
# fairness penalty.  It operates as a post-processing calibration step
# after Algorithm 4 predictions are collected.
ENABLE_ADVERSARIAL_DEBIASING: bool = False

# Adversarial debiasing hyperparameters
ADVERSARIAL_HIDDEN_DIM: int = 16
ADVERSARIAL_EPOCHS: int = 100
ADVERSARIAL_LR: float = 0.01
ADVERSARIAL_THRESHOLD_SEARCH_STEPS: int = 50

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL: Forced sex branching (Algorithm 1 variant)
# ─────────────────────────────────────────────────────────────────────────────
# When True, uses Algorithm1_forcedBranch.py which routes users to
# sex-specific sub-trees BEFORE any other branching.  This gives each
# gender its own CDS decision tree with gender-specific healthy ranges.
ENABLE_FORCED_SEX_BRANCHING: bool = False

# ─────────────────────────────────────────────────────────────────────────────
# DATA: Augmentation strategy for female sub-population
# ─────────────────────────────────────────────────────────────────────────────
# When enabled, augments the female training data before building the CDS
# tree.  Only meaningful when ENABLE_FORCED_SEX_BRANCHING is also True
# (augmentation targets the female sub-tree).
ENABLE_DATA_AUGMENTATION: bool = False

# Strategy: one of "none", "random_oversample", "perturbation",
#           "smotenc", "cross_gender", "combined"
AUGMENTATION_STRATEGY: str = "smotenc"

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS (do not change unless the dataset changes)
# ─────────────────────────────────────────────────────────────────────────────
SEX_FEATURE_INDEX: int = 1
MALE_CODE: int = 0
FEMALE_CODE: int = 1


def summary() -> str:
    """Return a one-line summary of active flags for logging."""
    flags = []
    if ENABLE_REWEIGHING:
        flags.append("Reweigh")
    if ENABLE_FAIRNESS_RL:
        flags.append(f"FairnessRL(lambda={FAIRNESS_LAMBDA})")
    if ENABLE_ADVERSARIAL_DEBIASING:
        flags.append("AdvDebiasing")
    if ENABLE_FORCED_SEX_BRANCHING:
        flags.append("ForcedSexBranch")
    if ENABLE_DATA_AUGMENTATION:
        flags.append(f"Augment({AUGMENTATION_STRATEGY})")
    return " + ".join(flags) if flags else "Baseline (no modifications)"
