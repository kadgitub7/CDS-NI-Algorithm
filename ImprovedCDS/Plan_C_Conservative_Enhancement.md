# Plan C: Conservative Enhancement (Minimal Departure from Paper)

## Philosophy
Keep the healthy-range concept but improve it. Add a second assurance factor
(AF_U) alongside the existing one (AF_H). Make the smallest number of changes
that meaningfully improve accuracy. This plan is ideal if you want a low-risk
baseline improvement that you can compare Plans A and B against, or if the
paper reviewers prefer staying close to the original algorithm.

The core insight: the original algorithm only accumulates evidence toward
"healthy" and immediately declares "unhealthy" if a user is outside the range.
Plan C adds a symmetric unhealthy AF so that being outside the range
accumulates evidence rather than triggering an immediate decision.

## Ideas Used

| ID | Name | Section Modified |
|----|------|-----------------|
| Idea 2 | Adaptive healthy range per feature | Algorithm 2 |
| B4 | Posterior smoothing (Laplace) | Algorithm 2 |
| C1 | Dual weights r_H and r_U | Algorithm 2 |
| D1 | Additive accumulation (keep) | Algorithm 4 |
| E1 | Symmetric thresholds | Algorithm 4 |
| E4 | Gap-based decision | Algorithm 4 |
| F8 | Deterministic first action | Algorithm 4 |
| G1 | AF inheritance across tree levels | Algorithm 4 |
| I1 | Capped AF contributions | Algorithm 4 |
| I3 | Inner cross-validated threshold selection | LOOCV |
| M1 | Multi-metric evaluation | LOOCV reporting |
| H5 | Extreme outliers to nearest bin | Algorithm 4 |

## Ideas NOT Used (and Why)

| ID | Name | Why Excluded |
|----|------|-------------|
| A8+A5 | Probabilistic bins | This plan keeps healthy ranges |
| B1/B3/B6/B8 | Posterior table improvements | Not needed — ranges handle evidence |
| K8/N3/N4 | SPRT / Bayes factor | Too large a departure from paper |
| I4 | Correlation penalty | Keeping it simple |
| L6 | Backward elimination | Keeping it simple |
| I8 | Minimum support | Not needed without probabilistic bins |
| K2 | Contradictory evidence | Deferred for simplicity |
| H1 | Missing data as evidence | Deferred |
| F1 | Most discriminative first (full) | Using F8 deterministic only |

---

## Detailed Changes by Algorithm

---

### Algorithm 1: Decision Tree (NO CHANGES)

Identical to current implementation.

---

### Algorithm 2: Perceptor & Executive Training (MODERATE CHANGES)

#### Change 2.1: Adaptive Healthy Range Per Feature (Idea 2)

**What changes:** Instead of using [min(healthy), max(healthy)] as the healthy
range for every feature, select the range construction method per feature based
on how well it separates healthy and unhealthy users.

**Options per feature:**
1. **Full range**: [min(healthy), max(healthy)] — current method
2. **Percentile shrinkage**: [P_alpha, P_(100-alpha)] where alpha is chosen
   to exclude healthy outliers. E.g., [P5, P95] excludes the most extreme
   5% of healthy users on each side.
3. **Density-based**: Find the contiguous interval containing the highest
   density of healthy users (the mode region).

**Selection criterion:** For each feature, try all three methods and pick the
one that maximizes the separation score:

```
separation = (fraction of unhealthy users outside range) -
             (fraction of healthy users outside range)
```

A good range has many unhealthy users outside and few healthy users outside.

**Implementation:**

```python
def compute_adaptive_healthy_range(vv_healthy, vv_unhealthy, is_binary):
    """
    Try multiple range methods, return the best one.

    Args:
        vv_healthy: feature values for healthy users (no NaN)
        vv_unhealthy: feature values for unhealthy users (no NaN)
        is_binary: whether the feature is binary

    Returns:
        (h_min, h_max, method_name)
    """
    if is_binary or len(vv_healthy) < 5:
        # Binary features or very few samples: use full range
        return float(vv_healthy.min()), float(vv_healthy.max()), "full"

    candidates = []

    # Method 1: Full range (current)
    h_min, h_max = float(vv_healthy.min()), float(vv_healthy.max())
    candidates.append((h_min, h_max, "full"))

    # Method 2: Percentile shrinkage at various levels
    for alpha in [2, 5, 10]:
        plo = float(np.percentile(vv_healthy, alpha))
        phi = float(np.percentile(vv_healthy, 100 - alpha))
        if plo < phi:
            candidates.append((plo, phi, f"percentile_{alpha}"))

    # Method 3: IQR-based (interquartile range)
    q1 = float(np.percentile(vv_healthy, 25))
    q3 = float(np.percentile(vv_healthy, 75))
    iqr = q3 - q1
    if iqr > 0:
        iqr_min = q1 - 1.5 * iqr
        iqr_max = q3 + 1.5 * iqr
        candidates.append((iqr_min, iqr_max, "iqr"))

    # Score each candidate
    best_score = -float('inf')
    best = candidates[0]

    for h_min, h_max, name in candidates:
        # Fraction of unhealthy outside range (want high)
        if len(vv_unhealthy) > 0:
            frac_u_outside = np.mean((vv_unhealthy < h_min) | (vv_unhealthy > h_max))
        else:
            frac_u_outside = 0.0

        # Fraction of healthy outside range (want low)
        frac_h_outside = np.mean((vv_healthy < h_min) | (vv_healthy > h_max))

        # Separation score
        score = frac_u_outside - 2.0 * frac_h_outside  # penalize healthy exclusion

        if score > best_score:
            best_score = score
            best = (h_min, h_max, name)

    return best
```

**Integration into train_node:**

Replace the current healthy range computation (lines 238-248 of cds.py):

```python
# Current:
if has_healthy:
    hv = vv[hm]
    h_min, h_max = float(hv.min()), float(hv.max())

# New:
if has_healthy:
    hv = vv[hm]
    uv = vv[~hm]  # unhealthy values
    h_min, h_max, range_method = compute_adaptive_healthy_range(
        hv, uv, is_bin[f]
    )
```

#### Change 2.2: Laplace Smoothing (B4)

Same as Plans A and B. Apply to likelihood table.

#### Change 2.3: Compute Dual Action Weights (C1)

Compute r_H and r_U using the healthy range:

```python
if has_healthy:
    # r_U: fraction of disease-class likelihood OUTSIDE the healthy range
    # (this is the existing 'r' computation)
    for ci, c in enumerate(all_cls):
        if c == HEALTHY or prev[ci] <= 0:
            continue
        r_u = ((lk[:mhb, ci].sum() if mhb > 0 else 0.0) +
               (lk[xhb + 1:, ci].sum() if xhb < nb - 1 else 0.0))

        # r_H: fraction of healthy likelihood INSIDE the healthy range
        r_h = lk[mhb:xhb+1, healthy_idx].sum() if mhb <= xhb else 0.0

        if r_h > 0 or r_u > 0:
            actions.append((f, node.nid, c, r_h, r_u))
```

**Note:** Action tuple changes from `(f, node_id, class, r)` to
`(f, node_id, class, r_h, r_u)`.

---

### Algorithm 3: Refinement (MINOR CHANGES)

#### Change 3.1: Sort by max(r_H, r_U) Instead of r Alone

```python
def refine_node(node, models, node_actions, data):
    # Sort by max(r_h, r_u) descending
    na = sorted([a for a in node_actions if max(a[3], a[4]) > 0],
                key=lambda a: max(a[3], a[4]), reverse=True)
```

The rest of the greedy set-cover logic remains the same: check if a feature
catches new users outside the healthy range. The healthy range still exists
in this plan, so the existing newinf logic works unchanged.

---

### Algorithm 4: Prediction (MODERATE CHANGES)

#### Change 4.1: Add Unhealthy AF

**What changes:** Instead of accumulating only AF (toward healthy) and
immediately declaring UNHEALTHY when outside the range, accumulate both
AF_H (healthy evidence) and AF_U (unhealthy evidence).

- Inside healthy range: increment AF_H
- Outside healthy range: increment AF_U
- Decision based on which AF wins with sufficient gap

**The key behavioral change:** Being outside the healthy range no longer
causes an immediate UNHEALTHY decision. Instead, it adds to AF_U. The user
might still end up HEALTHY if enough other features confirm health.

**Implementation:**

```python
def predict_dual(uid, data, nodes, models, retained, T, delta, cap, rng):
    """
    Dual-AF prediction with adaptive healthy ranges.

    Replaces the current predict() function.
    """
    af_h = 0.0
    af_u = 0.0
    pacs = 0

    def phf(nd, h):
        return nd.hdist.get(h, 0) / max(nd.nu, 1)

    def phg(nd):
        return nd.n_diseased / max(nd.nu, 1)

    def compute_af_increment(p, r, g):
        """Same formula as original afi(), but used for both directions."""
        return (p * r / g) if g > 1e-12 else 0.0

    # Route user through tree
    lvl_nodes = _route_user(uid, data, nodes)
    anh = defaultdict(list)
    for a in retained:
        anh[(a[1], a[2])].append(a)

    for lvl in sorted(lvl_nodes.keys()):
        for nd in lvl_nodes[lvl]:
            node_used = set()

            for h in sorted(k for k in nd.hdist if k != HEALTHY and nd.hdist[k] > 0):
                cands = [a for a in anh.get((nd.nid, h), [])
                         if a[0] not in node_used
                         and not np.isnan(data[uid, a[0]])]

                # Sort by max(r_h, r_u) descending — deterministic (F8)
                cands.sort(key=lambda a: max(a[3], a[4]), reverse=True)

                for action in cands:
                    f = action[0]
                    r_h = action[3]
                    r_u = action[4]
                    node_used.add(f)

                    m = models.get((nd.nid, f))
                    if not m:
                        continue

                    v = data[uid, f]
                    pacs += 1

                    # Inside or outside healthy range?
                    if m.h_min <= v <= m.h_max:
                        # INSIDE: healthy evidence
                        inc = compute_af_increment(phf(nd, h), r_h, phg(nd))
                        inc = min(inc, cap)  # I1 capping
                        af_h += inc
                    else:
                        # OUTSIDE: unhealthy evidence
                        # (instead of immediate UNHEALTHY return)
                        inc = compute_af_increment(phf(nd, h), r_u, phg(nd))
                        inc = min(inc, cap)  # I1 capping
                        af_u += inc

                    # Decision check (E1 + E4)
                    gap = abs(af_h - af_u)
                    if af_h >= T and gap >= delta:
                        return "HEALTHY", pacs
                    if af_u >= T and gap >= delta:
                        return "UNHEALTHY", pacs

    return "SCREENING", pacs
```

#### Key Difference from Current Algorithm

Current behavior at cds.py lines 360-361 and 394-395:

```python
# CURRENT: outside range = immediate UNHEALTHY
if v < m.h_min or v > m.h_max:
    return "UNHEALTHY", pacs
```

New behavior:

```python
# NEW: outside range = increment AF_U, continue evaluating
if v < m.h_min or v > m.h_max:
    af_u += inc
    # Only decide UNHEALTHY if AF_U >= T and gap >= delta
```

This single change is the most impactful modification. It transforms the
algorithm from "one strike and you're out" to "accumulate evidence and decide
when confident."

---

### LOOCV Changes

#### Change 5.1: Inner CV for Thresholds (I3)

Hyperparameters to tune:

| Parameter | Description | Search Range |
|-----------|-------------|-------------|
| T | Threshold for winning AF | [0.5, 0.8, 1.0, 1.5, 2.0] |
| delta | Minimum gap between AFs | [0.0, 0.2, 0.5, 0.8, 1.0] |
| cap | Max AF increment per feature | [0.3, 0.5, 1.0, 2.0] |

Implementation is the same coarse-to-fine approach as Plans A and B.

#### Change 5.2: Multi-Metric Reporting (M1)

Identical to Plans A and B.

---

## Summary of Information Flow

```
Algorithm 1 (unchanged)
    |
    v
Algorithm 2 (MODERATE CHANGES)
    - Adaptive healthy range per feature (Idea 2)
    - Laplace smoothing on likelihoods (B4)
    - Dual weights r_H, r_U (C1)
    |
    v
Algorithm 3 (MINOR CHANGES)
    - Sort by max(r_H, r_U) instead of r
    - Greedy set-cover otherwise unchanged
    |
    v
Algorithm 4 (MODERATE CHANGES)
    - Two accumulators: AF_H and AF_U
    - Inside range -> AF_H += increment
    - Outside range -> AF_U += increment (NOT immediate UNHEALTHY)
    - Capped contributions (I1)
    - Deterministic ordering by max(r_H, r_U) (F8)
    - Gap-based decision (E1+E4)
    - AF inherits across tree levels (G1)
    - Outliers to nearest bin (H5)
    |
    v
LOOCV (CHANGED)
    - Inner CV for (T, delta, cap) (I3)
    - Multi-metric reporting (M1)
```

## Why Plan C Exists

Plan C serves three purposes:

1. **Baseline comparison**: You can compare Plans A and B against Plan C to
   measure how much value the probabilistic-bin approach adds over simply
   improving the healthy range + adding a second AF.

2. **Low-risk fallback**: If Plans A/B prove too complex to implement or
   debug, Plan C gives you a meaningful improvement with minimal code changes.
   The core change is literally removing one `return "UNHEALTHY"` statement
   and adding `af_u += inc` instead.

3. **Ablation study**: By comparing Plan C (better ranges + dual AF) against
   Plan B (probabilistic bins + dual AF), you isolate the effect of
   eliminating healthy ranges. This is useful for the paper's evaluation
   section.

## Expected Outcomes

- **Accuracy**: Should improve over the current algorithm primarily because
  the "immediate UNHEALTHY for any outside-range value" brittleness is removed.
  Improvement may be 3-8 percentage points.
- **PACs**: May increase slightly (more features evaluated before deciding)
  because the system no longer short-circuits on the first outside-range value.
  But F8 (deterministic ordering) partially compensates.
- **Specificity**: Should improve significantly — healthy users who happen
  to have one outlier feature value will no longer be immediately misclassified.
- **Sensitivity**: Should remain similar or slightly improve — unhealthy users
  with multiple abnormal features will accumulate AF_U above the threshold.
- **SCREENING rate**: Will depend on T and delta. May be higher than current
  (which has 0% screening for UNHEALTHY-decided users). Target: 5-10%.
