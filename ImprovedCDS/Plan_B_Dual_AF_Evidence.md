# Plan B: Dual-AF Evidence Accumulation

## Philosophy
Keep the dual assurance factor architecture (AF_H and AF_U as separate
accumulators) but dramatically improve the quality of evidence that feeds them.
This is the plan closest to the user's original vision: replace healthy ranges
with probabilistic bins, accumulate evidence in both directions, and use a
gap-based decision rule. The innovation is in *how evidence is generated*,
not in *how it is accumulated*.

This plan is more interpretable than Plan A (SPRT) because you can examine
AF_H and AF_U separately to understand why a decision was made. It also
preserves the paper's philosophical framework more closely.

## Ideas Used

| ID | Name | Section Modified |
|----|------|-----------------|
| A8+A5 | Probabilistic bins (eliminate healthy range) | Algorithm 2 |
| B1 | Posterior probability table | Algorithm 2 |
| B3 | Confidence-weighted evidence | Algorithm 2 + 4 |
| B4 | Posterior smoothing (Laplace) | Algorithm 2 |
| C1 | Dual weights r_H and r_U | Algorithm 2 + 3 |
| I8 | Minimum support requirement | Algorithm 2 + 4 |
| I4 | Feature correlation penalty | Algorithm 3 |
| L6 | Backward feature elimination | Algorithm 3 |
| D1 | Additive accumulation (keep) | Algorithm 4 |
| E1 | Symmetric thresholds | Algorithm 4 |
| E4 | Gap-based decision | Algorithm 4 |
| E9 | Minimum feature requirement | Algorithm 4 |
| I1 | Capped AF contributions | Algorithm 4 |
| F1 | Most discriminative first | Algorithm 4 |
| F8 | Deterministic first action | Algorithm 4 |
| G1 | AF inheritance across tree levels | Algorithm 4 |
| K2 | Contradictory evidence detector | Algorithm 4 |
| I3 | Inner cross-validated threshold selection | LOOCV |
| M1 | Multi-metric evaluation | LOOCV reporting |
| H5 | Extreme outliers to nearest bin | Algorithm 4 |

## Ideas NOT Used (and Why)

| ID | Name | Why Excluded |
|----|------|-------------|
| K8/N3/N4 | SPRT / Bayes factor | This plan uses two separate AFs, not a single LLR |
| B6 | Density-based evidence | B3 (confidence weighting) provides similar benefit more simply |
| B8 | Graduated confidence | Replaced by I8 (simpler hard cutoff) |
| B10 | Adaptive binning | Deferred — validate with Sturges first |
| C7 | Distance-from-range-center | No healthy range exists |
| D2 | Log-odds accumulation | Sticking with additive (D1) for simplicity |
| H1 | Missing data as evidence | Deferred — adds complexity, validate core first |
| O5 | Negative evidence amplification | B3 confidence weighting serves same purpose |
| K1 | Two-phase prediction | Implicit in F1 ordering |

## Key Difference from Plan A

Plan A collapses healthy and unhealthy evidence into a single number
(cumulative LLR). Plan B keeps them separate:

```
Plan A:  S = sum(log(P(x|U)/P(x|H)))     one number
Plan B:  AF_H = sum(E_H(x))              two numbers
         AF_U = sum(E_U(x))
```

The advantage of Plan B: you can detect contradictions naturally
(both AF_H and AF_U are high) and the decision rule (E1+E4) is
more flexible than SPRT boundaries.

The disadvantage: no theoretical optimality guarantees on stopping.

---

## Detailed Changes by Algorithm

---

### Algorithm 1: Decision Tree (NO CHANGES)

Identical to current implementation. No modifications needed.

---

### Algorithm 2: Perceptor & Executive Training (MAJOR CHANGES)

#### Change 2.1: Laplace Smoothing (B4)

Identical to Plan A. After computing the raw likelihood matrix, add
pseudocounts:

```python
LAPLACE_ALPHA = 1.0
lk_smooth = np.zeros((nb, nc))
for ci, c in enumerate(all_cls):
    cm = (lv == c)
    n_c = cm.sum()
    counts = np.bincount(ba[cm], minlength=nb).astype(float) if n_c > 0 else np.zeros(nb)
    lk_smooth[:, ci] = (counts + LAPLACE_ALPHA) / (n_c + LAPLACE_ALPHA * nb)
lk = lk_smooth
```

#### Change 2.2: Store Bin Counts (I8)

Identical to Plan A. Add `bin_counts` to FeatureModel.

#### Change 2.3: Compute Confidence Weights Per Bin (B3)

**What changes:** Each bin gets a confidence weight based on how strongly it
discriminates between healthy and unhealthy.

**Why:** A bin where P(H|bin)=0.52 and P(U|bin)=0.48 is nearly useless — it
provides almost no information. A bin where P(H|bin)=0.95 and P(U|bin)=0.05
is highly informative. The confidence weight scales the contribution so that
uninformative bins contribute little to either AF.

**Implementation:**

After computing posteriors in `train_node`, compute per-bin confidence:

```python
# Confidence = |P(H|bin) - P(U|bin)|
# where P(U|bin) = 1 - P(H|bin) for binary case
healthy_idx = all_cls.index(HEALTHY)
confidence = np.zeros(nb)
for b in range(nb):
    p_h = post[b, healthy_idx]
    p_u = 1.0 - p_h
    confidence[b] = abs(p_h - p_u)  # ranges from 0 (useless) to 1 (perfect)
```

Add `confidence` to the FeatureModel class (new slot).

#### Change 2.4: Compute Dual Action Weights (C1)

**What changes:** Compute r_H (expected healthy evidence) and r_U (expected
unhealthy evidence) for each feature.

**Implementation:**

```python
healthy_idx = all_cls.index(HEALTHY)

r_H = 0.0
r_U = 0.0
for b in range(nb):
    p_h = post[b, healthy_idx]
    p_u = 1.0 - p_h
    bin_weight = ev[b]  # P(bin) = fraction of users in this bin

    r_H += p_h * confidence[b] * bin_weight
    r_U += p_u * confidence[b] * bin_weight

# The action tuple becomes (feature_idx, node_id, r_H, r_U)
if r_H > 0.01 or r_U > 0.01:
    actions.append((f, node.nid, r_H, r_U))
```

**Note:** The evidence function during prediction uses P(H|bin) * confidence,
not r_H directly. The r_H and r_U values are used for feature ordering
(deciding which feature to evaluate next).

#### Change 2.5: Remove Healthy Range Dependency

Same as Plan A: h_min, h_max, min_hbin, max_hbin are kept in the model but
not used for decision-making.

---

### Algorithm 3: Refinement / Feature Selection (MODERATE CHANGES)

#### Change 3.1: Greedy Selection by Discriminative Power

Same logic as Plan A: sort by |r_H - r_U| descending.

#### Change 3.2: Correlation Penalty (I4)

Identical to Plan A. Skip features with correlation > 0.8 to
already-retained features.

#### Change 3.3: Backward Elimination (L6)

Same concept as Plan A, but the `score` function uses the dual-AF
prediction instead of SPRT:

```python
def score(feature_set):
    correct = 0
    for uid in node.uidx:
        dec = predict_dual_af(uid, node, models, feature_set,
                              data, T, delta)
        true_label = labels[uid]
        ok = (dec != "UNHEALTHY") if true_label == HEALTHY else (dec == "UNHEALTHY")
        correct += ok
    return correct / len(node.uidx)
```

---

### Algorithm 4: Prediction (MAJOR CHANGES)

#### Change 4.1: Dual-AF with Evidence Functions

**What changes:** Replace the single AF with two accumulators. Each feature
contributes to both AF_H and AF_U based on the posterior probability and
confidence weight of the user's bin.

**The evidence function:**

```
E_H(x) = P(H | bin(x)) * confidence(bin(x)) * r_H
E_U(x) = P(U | bin(x)) * confidence(bin(x)) * r_U
```

Where:
- P(H|bin(x)): posterior probability of healthy given the user's bin
- confidence(bin(x)): discriminative strength of that bin (B3)
- r_H, r_U: feature-level action weights (C1)

**Why this formula works:**
- P(H|bin) handles the direction: which class does this bin favor?
- confidence handles the strength: how strongly does it favor that class?
- r_H/r_U handles the feature quality: is this feature generally useful?

A bin with P(H)=0.52, confidence=0.04, r_H=0.3 contributes:
E_H = 0.52 * 0.04 * 0.3 = 0.006 (negligible — good, this bin tells us nothing)

A bin with P(H)=0.95, confidence=0.90, r_H=0.8 contributes:
E_H = 0.95 * 0.90 * 0.8 = 0.684 (strong healthy evidence)

**Implementation:**

```python
def predict_dual_af(uid, data, nodes, models, retained, T, delta,
                    cap=0.5, min_support=5, k_min=2):
    """
    Dual-AF prediction with gap-based decision.

    Args:
        uid: user index
        data: full data matrix
        nodes: list of tree nodes
        models: dict of (node_id, feature) -> FeatureModel
        retained: list of (feature, node_id, r_H, r_U) tuples
        T: threshold for winning AF (E1)
        delta: minimum gap between AFs (E4)
        cap: maximum evidence contribution per feature (I1)
        min_support: minimum bin count to trust posterior (I8)
        k_min: minimum features before decision (E9)
    """
    af_h = 0.0  # healthy assurance factor
    af_u = 0.0  # unhealthy assurance factor
    pacs = 0

    # Tracking for contradiction detection (K2)
    features_favoring_h = 0
    features_favoring_u = 0

    # Route user through tree
    lvl_nodes = _route_user(uid, data, nodes)
    healthy_idx = 0  # index of HEALTHY in all_cls

    # AF inherits across tree levels (G1)
    # Since af_h and af_u are simple running sums, inheritance is automatic

    for lvl in sorted(lvl_nodes.keys()):
        for nd in lvl_nodes[lvl]:
            # Get retained features for this node
            node_features = [(a[0], a[2], a[3]) for a in retained
                             if a[1] == nd.nid]

            # Sort by max(r_H, r_U) * confidence — most discriminative first (F1)
            # We need to look up confidence from the model
            def sort_key(item):
                f, rh, ru = item
                m = models.get((nd.nid, f))
                if m:
                    avg_conf = np.mean(m.confidence)
                    return max(rh, ru) * avg_conf
                return max(rh, ru)

            node_features.sort(key=sort_key, reverse=True)

            for f, r_h, r_u in node_features:
                v = data[uid, f]
                if np.isnan(v):
                    continue  # skip missing (Plan B doesn't use H1)

                m = models.get((nd.nid, f))
                if not m:
                    continue

                # Find bin
                bin_idx = int(np.clip(
                    np.searchsorted(m.edges[1:], v, side='right'),
                    0, m.n_bins - 1
                ))

                # Minimum support check (I8)
                if m.bin_counts[bin_idx] < min_support:
                    pacs += 1
                    continue

                # Compute evidence
                p_h = m.posterior[bin_idx, healthy_idx]
                p_u = 1.0 - p_h
                conf = m.confidence[bin_idx]

                e_h = p_h * conf * r_h
                e_u = p_u * conf * r_u

                # Cap contributions (I1)
                e_h = min(e_h, cap)
                e_u = min(e_u, cap)

                af_h += e_h
                af_u += e_u
                pacs += 1

                # Track evidence direction for K2
                if e_h > e_u + 0.01:
                    features_favoring_h += 1
                elif e_u > e_h + 0.01:
                    features_favoring_u += 1

                # Decision check (only after k_min features) (E9)
                if pacs >= k_min:
                    gap = abs(af_h - af_u)

                    # Healthy decision (E1 + E4)
                    if af_h >= T and gap >= delta:
                        return "HEALTHY", pacs

                    # Unhealthy decision (E1 + E4)
                    if af_u >= T and gap >= delta:
                        return "UNHEALTHY", pacs

    # No decision reached — classify why (K2)
    if features_favoring_h >= 3 and features_favoring_u >= 3:
        balance = min(features_favoring_h, features_favoring_u) / \
                  max(features_favoring_h, features_favoring_u)
        if balance > 0.4:
            return "CONTRADICTORY", pacs

    return "SCREENING", pacs
```

---

### LOOCV Changes

#### Change 5.1: Inner CV for Thresholds (I3)

Similar to Plan A, but the hyperparameters are different:

```python
def select_hyperparams_inner_cv(data, labels, nodes, models, retained,
                                 n_folds=5, seed=42):
    T_grid     = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    delta_grid = [0.0, 0.3, 0.5, 1.0, 1.5]
    cap_grid   = [0.3, 0.5, 0.8, 1.0]
    k_min_grid = [1, 2, 3]

    # Coarse-to-fine:
    # 1. Fix cap=0.5, k_min=2. Sweep T x delta.
    # 2. Fix best T/delta. Sweep cap.
    # 3. Fix best T/delta/cap. Sweep k_min.

    indices = np.arange(len(labels))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    def evaluate(T, delta, cap, k_min):
        correct = 0
        total = 0
        for fold_idx in range(n_folds):
            for uid in folds[fold_idx]:
                dec, _ = predict_dual_af(uid, data, nodes, models, retained,
                                          T, delta, cap=cap, k_min=k_min)
                true_label = labels[uid]
                ok = (dec != "UNHEALTHY") if true_label == HEALTHY else (dec == "UNHEALTHY")
                correct += ok
                total += 1
        return correct / total

    # Stage 1
    best_score = -1
    best_T, best_delta = 1.5, 0.5
    for t in T_grid:
        for d in delta_grid:
            s = evaluate(t, d, 0.5, 2)
            if s > best_score:
                best_score = s
                best_T, best_delta = t, d

    # Stage 2
    best_cap = 0.5
    for c in cap_grid:
        s = evaluate(best_T, best_delta, c, 2)
        if s > best_score:
            best_score = s
            best_cap = c

    # Stage 3
    best_kmin = 2
    for k in k_min_grid:
        s = evaluate(best_T, best_delta, best_cap, k)
        if s > best_score:
            best_score = s
            best_kmin = k

    return best_T, best_delta, best_cap, best_kmin
```

#### Change 5.2: Multi-Metric Reporting (M1)

Identical to Plan A. Same `compute_metrics` and `print_report_extended`
functions.

---

## Summary of Information Flow

```
Algorithm 1 (unchanged)
    |
    v
Algorithm 2 (CHANGED)
    - Posterior tables with Laplace smoothing (B1+B4)
    - Per-bin confidence weights (B3)
    - Bin counts stored (I8)
    - Dual weights r_H, r_U computed (C1)
    - Healthy range fields kept but unused
    |
    v
Algorithm 3 (CHANGED)
    - Greedy selection by |r_H - r_U|
    - Correlation penalty (I4): skip features corr > 0.8
    - Backward elimination (L6)
    |
    v
Algorithm 4 (REPLACED)
    - Dual AF accumulators: AF_H and AF_U
    - Evidence: E_H = P(H|bin) * confidence * r_H  (B1+B3+C1)
    - Evidence: E_U = P(U|bin) * confidence * r_U
    - Capped contributions (I1)
    - Deterministic most-discriminative-first ordering (F1+F8)
    - Minimum support check per bin (I8)
    - Minimum feature count (E9)
    - Gap-based decision: AF >= T AND gap >= delta (E1+E4)
    - Contradiction detection (K2)
    - AFs inherit across tree levels (G1)
    - Outliers clipped to nearest bin (H5)
    |
    v
LOOCV (CHANGED)
    - Inner 5-fold CV for (T, delta, cap, k_min) (I3)
    - Multi-metric reporting (M1)
```

## Hyperparameters to Tune (via Inner CV)

| Parameter | Description | Search Range |
|-----------|-------------|-------------|
| T | Threshold for winning AF | [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] |
| delta | Minimum gap between AFs | [0.0, 0.3, 0.5, 1.0, 1.5] |
| cap | Max evidence per feature | [0.3, 0.5, 0.8, 1.0] |
| k_min | Min features before decision | [1, 2, 3] |

Fixed parameters:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LAPLACE_ALPHA | 1.0 | Standard Laplace smoothing |
| min_support | 5 | Standard minimum bin count |
| CORR_THRESHOLD | 0.8 | Standard correlation cutoff |

## How Plan B Compares to Plan A

| Aspect | Plan A (SPRT) | Plan B (Dual-AF) |
|--------|---------------|-------------------|
| **Evidence representation** | Single cumulative LLR | Two separate AF_H, AF_U |
| **Decision rule** | Cross boundary A or B | AF >= T and gap >= delta |
| **Theoretical guarantees** | Yes (error rate bounds) | No (empirical tuning) |
| **Interpretability** | Moderate (one number) | High (can inspect both AFs) |
| **Contradiction detection** | Post-hoc analysis | Natural (both AFs visible) |
| **Complexity** | Simpler math | More parameters |
| **Closest to paper** | Moderate departure | Close to original vision |

## Expected Outcomes

- **Accuracy**: Should improve similarly to Plan A — both eliminate the
  brittle healthy-range binary test.
- **PACs**: May be slightly higher than Plan A (no optimal stopping guarantee)
  but still lower than current system due to F1 ordering.
- **Interpretability**: Better than Plan A — you can say "AF_H = 3.2, AF_U = 0.8,
  gap = 2.4 > delta, so HEALTHY." This is more intuitive than "LLR = -2.4 < B."
- **SCREENING rate**: Tunable via delta. Higher delta = more SCREENING.
  Target: 5-15%.
