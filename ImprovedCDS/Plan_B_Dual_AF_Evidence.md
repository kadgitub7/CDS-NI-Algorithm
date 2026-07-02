# Plan B: Dual-AF Evidence Accumulation

## Philosophy

Keep the dual assurance factor architecture (AF_H and AF_U as separate
accumulators) but dramatically improve the quality of evidence that feeds them.
This is the plan closest to the user's original vision: replace healthy ranges
with probabilistic bins, accumulate evidence in both directions, and use a
gap-based decision rule. The innovation is in *how evidence is generated*,
not in *how it is accumulated*.

This plan is more interpretable than a single-number approach because you can
examine AF_H and AF_U separately to understand why a decision was made. It
also preserves the paper's philosophical framework more closely.

### Why Dual-AF Instead of a Single Number

A single-number approach (like a cumulative log-likelihood ratio) collapses
healthy and unhealthy evidence into one value. Plan B keeps them separate:

```
Single number:   S = sum(log(P(x|U)/P(x|H)))        one number
Plan B:          AF_H = sum(E_H(x))                  two numbers
                 AF_U = sum(E_U(x))
```

The advantage of Plan B: you can detect contradictions naturally
(both AF_H and AF_U are high) and the decision rule is more flexible.
The disadvantage: no theoretical optimality guarantees on stopping.

### The Core Problem This Plan Solves

The current algorithm (cds.py lines 360-361 and 394-395) has a critical flaw:

```python
# Current behavior: ONE outside-range value = instant UNHEALTHY
if v < m.h_min or v > m.h_max:
    return "UNHEALTHY", pacs
```

This means a perfectly healthy user who has ONE feature value slightly outside
the healthy range is immediately classified as UNHEALTHY, regardless of how
many other features strongly confirm health. Plan B replaces this binary
test with probabilistic evidence that accumulates gradually in both directions.

---

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
| C7 | Distance-from-range-center | No healthy range exists in this plan |
| D2 | Log-odds accumulation | Sticking with additive (D1) for simplicity |
| H1 | Missing data as evidence | Deferred — adds complexity, validate core first |
| O5 | Negative evidence amplification | B3 confidence weighting serves same purpose |
| K1 | Two-phase prediction | Implicit in F1 ordering |

---

## Detailed Changes by Algorithm

---

### Algorithm 1: Decision Tree (NO CHANGES)

The tree construction remains identical to the current implementation:
- Load data from UCI arrhythmia dataset, replace '?' with NaN
- Classify features as binary (values in {0, 1}) or continuous
- Build tree with forced sex branching at level 1 (SEX_FEAT = feature 1)
- Use median split for continuous features, value-based split for binary
- Prune nodes with fewer than U_MIN (200) users
- Deduplicate child nodes with identical user sets at the same level

No modifications are needed to Algorithm 1 for Plan B.

---

### Algorithm 2: Perceptor & Executive Training (MAJOR CHANGES)

The current Algorithm 2 computes, for each node and feature:
1. Likelihood table P(bin|class) -- **KEEP but smooth**
2. Prevalence P(class) per node -- **KEEP**
3. Evidence P(bin) -- **KEEP**
4. Posterior P(class|bin) via Bayes' theorem -- **KEEP but smooth**
5. Healthy range [h_min, h_max] -- **KEEP but stop using for decisions**
6. Action weights based on r_outside -- **REPLACE with dual weights**

#### Change 2.1: Laplace Smoothing on Posteriors (B4)

**What it is:** A standard statistical technique that adds a small pseudocount
(typically 1.0) to every cell of the likelihood table before normalizing.
This ensures no probability is ever exactly zero.

**Why it helps:** With only 452 users split across ~10 bins and up to 13
disease classes, many bins will have zero users from certain classes. In the
current algorithm, this doesn't cause errors because the healthy-range test
is binary (inside/outside). But in Plan B, the posterior P(H|bin) directly
drives the evidence function. If P(bin|H) = 0 because no healthy users
happened to land in bin 4 during training, then P(H|bin 4) = 0, which means
a user in bin 4 gets zero healthy evidence -- even if the zero count is just
a sampling artifact from having very few data points. Laplace smoothing
prevents this by ensuring every bin has a nonzero probability for every class.

**How it works mathematically:**

Without smoothing (current code, cds.py lines 216-223):
```
P(bin | class) = count(users of class c in bin b) / count(users of class c)
```

With Laplace smoothing:
```
P(bin | class) = (count(users of class c in bin b) + alpha) /
                 (count(users of class c) + alpha * num_bins)
```

Where alpha = 1.0 (standard Laplace). This adds 1 imaginary observation to
every bin for every class, then re-normalizes.

**Example:** A feature with 10 bins and a disease class that has 20 users:
- Without smoothing: a bin with 0 users gives P(bin|class) = 0/20 = 0
- With smoothing: that bin gives P(bin|class) = (0+1)/(20+10) = 0.033
- A bin with 5 users gives P(bin|class) = (5+1)/(20+10) = 0.2 (was 5/20 = 0.25)
- The smoothing effect is proportional to how sparse the data is

**Implementation:**

In the `train_node` function, replace the likelihood computation at
cds.py lines 216-223:

```python
# CURRENT code:
lk = np.zeros((nb, nc))
for ci, c in enumerate(all_cls):
    cm = (lv == c)
    n_c = cm.sum()
    if n_c == 0:
        pass
    else:
        lk[:, ci] = np.bincount(ba[cm], minlength=nb).astype(float) / n_c

# NEW code with Laplace smoothing:
LAPLACE_ALPHA = 1.0  # pseudocount (1.0 = standard Laplace)
lk = np.zeros((nb, nc))
for ci, c in enumerate(all_cls):
    cm = (lv == c)
    n_c = cm.sum()
    counts = np.bincount(ba[cm], minlength=nb).astype(float) if n_c > 0 else np.zeros(nb)
    lk[:, ci] = (counts + LAPLACE_ALPHA) / (n_c + LAPLACE_ALPHA * nb)
```

**Effect on downstream computations:** The prevalence P(class) and evidence
P(bin) computations remain unchanged -- they use the smoothed likelihood
automatically since they are computed from `lk`. The posteriors P(class|bin)
also benefit from smoothing since they are derived from `lk`.

**Integration with Plan B:** Laplace smoothing is essential for Plan B because
the evidence functions E_H and E_U multiply by the posterior P(H|bin). Without
smoothing, bins with P(H|bin) = 0 would contribute zero healthy evidence even
when the zero is just a sampling artifact. With smoothing, every bin contributes
proportional evidence, making the dual-AF accumulation more robust.

---

#### Change 2.2: Store Bin Counts for Minimum Support (I8)

**What it is:** Record how many total training users fall into each bin, and
store this count array alongside the existing FeatureModel fields.

**Why it helps:** In Plan B, every bin's posterior probability drives the
evidence function. But a bin with only 1 or 2 users in it has an extremely
unreliable posterior estimate. A bin with 1 healthy user and 0 unhealthy users
gives P(H|bin) = 1.0 (or close to 1.0 with smoothing), which would contribute
very strong healthy evidence -- but it could easily be a sampling artifact.
By storing bin counts, we can check during prediction whether a bin has enough
training data to trust its posterior. If not, we skip that feature for this user
rather than allowing unreliable evidence to influence the decision.

**Implementation:**

Step 1: Add `bin_counts` to the FeatureModel class (cds.py line 168):

```python
class FeatureModel:
    __slots__ = ('likelihood', 'prevalence', 'evidence', 'posterior',
                 'h_min', 'h_max', 'n_bins', 'edges', 'min_hbin', 'max_hbin',
                 'bin_counts', 'confidence')  # NEW: bin_counts and confidence

    def __init__(self, likelihood, prevalence, evidence, posterior,
                 h_min, h_max, n_bins, edges, min_hbin, max_hbin,
                 bin_counts, confidence):
        self.likelihood = likelihood
        self.prevalence = prevalence
        self.evidence = evidence
        self.posterior = posterior
        self.h_min = h_min
        self.h_max = h_max
        self.n_bins = n_bins
        self.edges = edges
        self.min_hbin = min_hbin
        self.max_hbin = max_hbin
        self.bin_counts = bin_counts    # NEW
        self.confidence = confidence    # NEW (see Change 2.3)
```

Step 2: In `train_node`, after computing bin assignments `ba` (cds.py line 212),
compute the bin count array:

```python
ba = np.clip(np.searchsorted(edges[1:], vv, side='right'), 0, nb - 1)

# NEW: count total users per bin (across all classes)
bin_counts = np.bincount(ba, minlength=nb)
```

Step 3: Pass `bin_counts` to the FeatureModel constructor (cds.py line 250):

```python
models[(node.nid, f)] = FeatureModel(
    lk, prev, ev, post, h_min, h_max, nb, edges, mhb, xhb,
    bin_counts, confidence)  # NEW arguments
```

**How it integrates with prediction (Algorithm 4):** During prediction, before
computing evidence from a bin, check if `m.bin_counts[bin_idx] >= min_support`.
If not, skip this feature entirely for this user. The default `min_support = 5`
means a bin needs at least 5 training users before its posterior is trusted.

---

#### Change 2.3: Compute Confidence Weights Per Bin (B3)

**What it is:** Each bin gets a confidence weight between 0 and 1 that measures
how strongly the bin discriminates between healthy and unhealthy classes.

**Why it helps:** Not all bins are equally informative. Consider two scenarios:
- Bin A: P(H|bin) = 0.52, P(U|bin) = 0.48 -- this bin is nearly useless; the
  posterior is close to 50/50 so landing in this bin tells us almost nothing
  about whether the user is healthy or unhealthy.
- Bin B: P(H|bin) = 0.95, P(U|bin) = 0.05 -- this bin is highly informative;
  a user in this bin is very likely healthy.

Without confidence weighting, both bins would contribute evidence proportional
to their posterior. But Bin A's evidence is almost noise -- it's barely better
than a coin flip. The confidence weight scales the contribution so that
uninformative bins contribute little to either AF, while highly discriminative
bins contribute substantially. This reduces the noise in the AF accumulators
and makes decisions more reliable.

**How it works mathematically:**

```
confidence(bin) = |P(H|bin) - P(U|bin)|
```

Where P(U|bin) = 1 - P(H|bin) for the binary healthy vs. unhealthy case.

- If P(H|bin) = 0.5: confidence = |0.5 - 0.5| = 0.0 (completely uninformative)
- If P(H|bin) = 0.7: confidence = |0.7 - 0.3| = 0.4 (moderately informative)
- If P(H|bin) = 0.95: confidence = |0.95 - 0.05| = 0.9 (highly informative)
- If P(H|bin) = 0.1: confidence = |0.1 - 0.9| = 0.8 (strongly favors unhealthy)

The confidence ranges from 0 (useless bin) to 1 (perfectly discriminative bin).

**Implementation:**

In `train_node`, after computing posteriors `post` (cds.py lines 232-235),
compute per-bin confidence:

```python
# After the posterior computation:
post = np.zeros((nb, nc))
for b in range(nb):
    if ev[b] > 0:
        post[b] = lk[b] * prev / ev[b]

# NEW: Compute per-bin confidence weight
healthy_idx = all_cls.index(HEALTHY)
confidence = np.zeros(nb)
for b in range(nb):
    p_h = post[b, healthy_idx]
    p_u = 1.0 - p_h
    confidence[b] = abs(p_h - p_u)  # ranges from 0 (useless) to 1 (perfect)
```

Store `confidence` as a field in the FeatureModel (see Change 2.2 for the
updated class definition).

**Integration with Plan B:** The confidence weight appears in the evidence
function used during prediction:

```
E_H(x) = P(H|bin(x)) * confidence(bin(x)) * r_H
E_U(x) = P(U|bin(x)) * confidence(bin(x)) * r_U
```

This means that a bin with P(H|bin) = 0.52 and confidence = 0.04 contributes
nearly nothing (0.52 * 0.04 * r_H ~ 0.02 * r_H), while a bin with
P(H|bin) = 0.95 and confidence = 0.90 contributes strongly
(0.95 * 0.90 * r_H ~ 0.855 * r_H). This is exactly the behavior we want:
strong posteriors generate strong evidence, weak posteriors generate weak
evidence.

---

#### Change 2.4: Compute Dual Action Weights (C1)

**What it is:** Instead of computing a single action weight `r` (fraction of
disease likelihood outside the healthy range), compute two weights:
- `r_H`: expected healthy evidence from this feature
- `r_U`: expected unhealthy evidence from this feature

**Why it helps:** The current algorithm computes a single weight `r` that only
measures how much of a disease class's likelihood falls outside the healthy
range (cds.py lines 254-263). This is an "abnormality detector" -- it only
measures how good a feature is at catching unhealthy users. But in Plan B, we
accumulate evidence in BOTH directions (healthy and unhealthy), so we need to
know how much a feature contributes to each direction. A feature might be
excellent at confirming health (high r_H) but mediocre at detecting disease
(low r_U), or vice versa. Dual weights let us properly prioritize features
and scale evidence in both directions.

**How it works:** For each feature, r_H measures how much healthy evidence we
expect from it on average, and r_U measures how much unhealthy evidence we
expect. These are computed by weighting each bin's posterior by how likely
users of each class are to fall into that bin.

**Implementation:**

Replace the current action weight computation in `train_node` (cds.py lines
253-263):

```python
# CURRENT code:
for ci, c in enumerate(all_cls):
    if c == HEALTHY or prev[ci] <= 0:
        continue
    if not has_healthy:
        r = 1.0
    else:
        r = ((lk[:mhb, ci].sum() if mhb > 0 else 0.0) +
             (lk[xhb + 1:, ci].sum() if xhb < nb - 1 else 0.0))
    if r > 0:
        actions.append((f, node.nid, c, r))

# NEW code: compute dual weights r_H and r_U
healthy_idx = all_cls.index(HEALTHY)

r_H = 0.0
r_U = 0.0
for b in range(nb):
    p_h = post[b, healthy_idx]
    p_u = 1.0 - p_h
    bin_weight = ev[b]  # P(bin) = fraction of all users in this bin

    r_H += p_h * confidence[b] * bin_weight
    r_U += p_u * confidence[b] * bin_weight

# Store as action: (feature_idx, node_id, r_H, r_U)
# NOTE: action tuple format changes from (f, node_id, class, r)
#       to (f, node_id, r_H, r_U) -- no per-disease-class actions
if r_H > 0.01 or r_U > 0.01:
    actions.append((f, node.nid, r_H, r_U))
```

**Key difference from current code:** The current algorithm creates separate
actions for each disease class (the `for ci, c in enumerate(all_cls)` loop).
Plan B collapses this into a single action per feature per node, with
separate healthy and unhealthy weights. This is possible because Plan B treats
the prediction as binary (healthy vs. not-healthy) rather than per-disease-class.

**The action tuple format changes from:**
```
(feature_index, node_id, disease_class, r)       # current: 4-tuple
```
**to:**
```
(feature_index, node_id, r_H, r_U)               # Plan B: 4-tuple
```

This change propagates to Algorithms 3 and 4.

**How r_H and r_U are used:**
- **Feature ordering (F1):** Features are sorted by `max(r_H, r_U)` or
  `|r_H - r_U|` to evaluate the most discriminative features first.
- **Evidence scaling:** During prediction, the evidence function multiplies
  by r_H or r_U to scale contributions by feature quality.
- **Feature selection (Algorithm 3):** Features are selected based on
  discriminative power measured by `|r_H - r_U|`.

---

#### Change 2.5: Stop Using Healthy Range for Decision-Making (A8+A5)

**What it is:** The fields `h_min`, `h_max`, `min_hbin`, `max_hbin` in
FeatureModel are still computed and stored (for diagnostic/debugging purposes)
but are **no longer used for prediction decisions**. The concept of
"inside/outside healthy range" is replaced entirely by: "which bin does the
user fall into, and what are the posteriors and confidence for that bin?"

**Why it helps:** The healthy range is the root cause of the algorithm's
brittleness. A user whose heart rate is 101 bpm when the healthy range is
[55, 100] gets an immediate UNHEALTHY decision, even though 101 is barely
outside the range and most other features confirm health. With probabilistic
bins, a user at 101 bpm falls into some bin (say bin 7), which has a posterior
like P(H|bin 7) = 0.35. This contributes moderate unhealthy evidence
(P(U|bin 7) = 0.65) rather than an instant UNHEALTHY. The evidence accumulates
across features, and the final decision reflects the totality of evidence.

**What changes in the code:**
- `h_min`, `h_max`, `mhb`, `xhb` are still computed in `train_node` (for
  backward compatibility and debugging) but are NOT referenced in the
  prediction function (Algorithm 4).
- The prediction function no longer checks `if v < m.h_min or v > m.h_max`.
- Instead, it looks up the user's bin and uses the posterior P(H|bin) and
  confidence to compute evidence.

**No code changes needed in Algorithm 2 itself** -- the healthy range is still
computed. The change is in Algorithm 4, where the healthy range is simply
ignored in favor of bin-based posteriors.

---

### Algorithm 3: Refinement / Feature Selection (MODERATE CHANGES)

The current Algorithm 3 (cds.py lines 275-297) does a greedy set-cover:
sort actions by `r` (action weight) descending, then keep features that catch
new unhealthy users outside the healthy range.

Plan B modifies this in three ways:

1. Change the sorting criterion (no healthy range)
2. Add a correlation penalty to avoid redundant features
3. Add backward elimination to remove features that don't help accuracy

#### Change 3.1: Greedy Selection by Discriminative Power

**What it is:** Since there is no healthy range in Plan B, we can't check
"is this user outside the range?" to decide if a feature is useful. Instead,
we sort features by their discriminative power: `|r_H - r_U|`. A feature where
r_H and r_U are nearly equal provides weak evidence in either direction. A
feature where one dominates the other provides strong directional evidence.

**Why it helps:** The current algorithm's greedy selection relies on the healthy
range to determine if a feature "catches" abnormal users. Since Plan B
eliminates the healthy range, we need a new criterion. The absolute difference
`|r_H - r_U|` directly measures how much a feature's evidence tips the balance
toward one class. Features with high `|r_H - r_U|` are the most useful for
making decisions, so they should be retained first.

**How it works:** Sort all candidate features by `|r_H - r_U|` descending.
Keep features that have discriminative power above a minimum threshold (0.01).
Features where r_H ~ r_U (i.e., the feature doesn't favor either class) are
excluded because they would add noise to the AF accumulators without helping
to separate classes.

**Implementation:**

Replace the current `refine_node` function (cds.py lines 275-297):

```python
def refine_node(node, models, node_actions, data):
    """
    Greedy feature selection by discriminative power.

    Replaces the current set-cover approach which relies on healthy ranges.
    Now selects features based on |r_H - r_U| — how strongly the feature
    differentiates between healthy and unhealthy users.

    Args:
        node: tree Node object
        models: dict of (node_id, feature) -> FeatureModel
        node_actions: list of (feature, node_id, r_H, r_U) tuples
        data: full data matrix

    Returns:
        list of retained (feature, node_id, r_H, r_U) actions
    """
    # Sort by discriminative power: |r_H - r_U| descending
    # Filter out features with negligible discriminative power
    na = sorted([a for a in node_actions if abs(a[2] - a[3]) > 0.01],
                key=lambda a: abs(a[2] - a[3]), reverse=True)

    if not na:
        return []

    kept = []
    kept_features = set()

    for a in na:
        f = a[0]
        if f in kept_features:
            continue

        m = models.get((node.nid, f))
        if not m:
            continue

        # Verify feature has valid data for users in this node
        raw = data[node.uidx, f]
        vm = ~np.isnan(raw)
        if vm.sum() == 0:
            continue

        # --- Correlation check (I4) inserted here ---
        # (See Change 3.2 below for full implementation)

        kept.append(a)
        kept_features.add(f)

    return kept
```

---

#### Change 3.2: Correlation Penalty (I4)

**What it is:** Before adding a feature to the retained set, check its
Pearson correlation with all already-retained features. If the maximum
correlation exceeds 0.8, skip the candidate feature.

**Why it helps:** Highly correlated features carry overlapping information.
For example, features 27 and 28 in the UCI arrhythmia dataset might both
measure similar ECG intervals. If both are retained, they would contribute
nearly identical evidence to AF_H and AF_U, effectively double-counting
the same biological signal. This inflates the AFs artificially and can cause
premature or incorrect decisions.

By skipping features that are highly correlated with already-retained ones,
we ensure each retained feature contributes genuinely new information. This
makes the dual-AF accumulation more reliable because each evidence increment
represents an independent observation about the user's health.

**How it works:** Before the greedy loop, precompute pairwise Pearson
correlations between all candidate features using the users in this node.
During the greedy loop, when considering adding feature f, check if
`max_correlation(f, already_kept_features) > 0.8`. If so, skip f and move
to the next candidate.

**Implementation:**

Integrate this into the `refine_node` function from Change 3.1:

```python
def refine_node(node, models, node_actions, data):
    na = sorted([a for a in node_actions if abs(a[2] - a[3]) > 0.01],
                key=lambda a: abs(a[2] - a[3]), reverse=True)
    if not na:
        return []

    # Precompute pairwise correlations between candidate features
    nd_data = data[node.uidx]
    feature_ids = sorted(set(a[0] for a in na))
    correlations = {}
    for i, f1 in enumerate(feature_ids):
        for f2 in feature_ids[i+1:]:
            col1 = nd_data[:, f1]
            col2 = nd_data[:, f2]
            valid = ~(np.isnan(col1) | np.isnan(col2))
            if valid.sum() > 10:  # need enough data points for meaningful correlation
                c = abs(np.corrcoef(col1[valid], col2[valid])[0, 1])
                if not np.isnan(c):
                    correlations[(f1, f2)] = c
                    correlations[(f2, f1)] = c

    CORR_THRESHOLD = 0.8

    kept = []
    kept_features = set()

    for a in na:
        f = a[0]
        if f in kept_features:
            continue

        m = models.get((node.nid, f))
        if not m:
            continue

        # Verify feature has valid data
        raw = data[node.uidx, f]
        vm = ~np.isnan(raw)
        if vm.sum() == 0:
            continue

        # Correlation check (I4): skip if too correlated with any kept feature
        if kept_features:
            max_corr = max(correlations.get((f, kf), 0.0) for kf in kept_features)
            if max_corr > CORR_THRESHOLD:
                continue  # skip this feature — too similar to one already retained

        kept.append(a)
        kept_features.add(f)

    return kept
```

**Performance note:** The correlation precomputation is O(F^2 * N) where F is
the number of candidate features and N is the number of users in the node. For
the UCI arrhythmia dataset (~279 features, ~200-450 users per node), this is
feasible but adds measurable time per LOOCV fold. If performance is an issue,
consider precomputing correlations once across all nodes.

---

#### Change 3.3: Backward Elimination (L6)

**What it is:** After the greedy forward selection pass (Changes 3.1 + 3.2),
run a backward elimination pass: try removing each retained feature one at a
time. If removing a feature improves accuracy (or doesn't hurt it) on the
training users in this node, remove it permanently. Repeat until no feature
can be removed without hurting accuracy.

**Why it helps:** Greedy forward selection can retain suboptimal features
because earlier selections constrain later ones. For example, feature A might
be selected first because it has the highest `|r_H - r_U|`. Then feature B is
selected because it's the best remaining. But together, A and B might perform
worse than B alone (if A introduces noise that counteracts B's signal). Backward
elimination catches these cases by testing whether each feature still helps
after all features are selected.

This is a standard technique in feature selection called "sequential floating
backward selection." It typically retains fewer, higher-quality features, which
improves both accuracy and interpretability.

**Implementation:**

Add this function and call it after `refine_node`:

```python
def backward_eliminate(node, models, kept, data, labels, T, delta, cap, k_min):
    """
    Remove features that don't help (or hurt) prediction accuracy.

    After greedy forward selection, tries removing each feature one at a time.
    If accuracy on the node's training users doesn't decrease, the feature
    is removed permanently. Repeats until stable.

    Args:
        node: tree Node object
        models: dict of (node_id, feature) -> FeatureModel
        kept: list of (feature, node_id, r_H, r_U) retained actions
        data: full data matrix
        labels: label array
        T: threshold for decision (from inner CV or default)
        delta: minimum gap between AFs
        cap: maximum evidence contribution per feature
        k_min: minimum features before decision

    Returns:
        pruned list of retained actions
    """
    if len(kept) <= 2:
        return kept  # need at least 2 features to make meaningful decisions

    def score(feature_set):
        """Compute accuracy on this node's training users using dual-AF prediction."""
        correct = 0
        for uid in node.uidx:
            dec = predict_dual_af(uid, data, [node], models, feature_set,
                                  T, delta, cap=cap, k_min=k_min)
            true_label = labels[uid]
            ok = (dec[0] != "UNHEALTHY") if true_label == HEALTHY else (dec[0] == "UNHEALTHY")
            correct += ok
        return correct / len(node.uidx)

    improved = True
    while improved and len(kept) > 2:
        improved = False
        base_score = score(kept)
        worst_idx = None

        for i in range(len(kept)):
            # Try removing feature i
            candidate = kept[:i] + kept[i+1:]
            s = score(candidate)
            if s >= base_score:  # >= means "at least as good" — remove for simplicity
                base_score = s
                worst_idx = i
                improved = True

        if worst_idx is not None:
            kept.pop(worst_idx)

    return kept
```

**Integration into the LOOCV loop:** After calling `refine_node`, call
`backward_eliminate`:

```python
# In run_loocv, after the refine step:
retained = []
for nd in nodes:
    node_kept = refine_node(nd, all_models,
                            actions_by_node.get(nd.nid, []), td)
    node_kept = backward_eliminate(nd, all_models, node_kept,
                                    td, tl, T, delta, cap, k_min)
    retained.extend(node_kept)
```

**Performance consideration:** Backward elimination evaluates all training users
in the node for each candidate removal. With ~200 users per node and ~20
retained features, this is ~4000 prediction calls per node per iteration. For
452 LOOCV folds, this adds significant time. To mitigate:
- Only run backward elimination if the greedy pass retained > 5 features
- Limit to 2-3 elimination rounds instead of iterating to convergence
- Cache intermediate predictions where possible

---

### Algorithm 4: Prediction (MAJOR CHANGES)

This is the most significant change in Plan B. The entire prediction loop is
replaced. Instead of a single AF with an immediate UNHEALTHY return on
outside-range values, we use two accumulators (AF_H and AF_U) with a gap-based
decision rule.

#### Change 4.1: Dual-AF with Evidence Functions

**What it is:** Replace the single AF with two accumulators. For each feature
the user is evaluated on, compute evidence for BOTH the healthy and unhealthy
hypotheses and add them to their respective accumulators.

**The evidence functions:**

```
E_H(x) = P(H | bin(x)) * confidence(bin(x)) * r_H
E_U(x) = P(U | bin(x)) * confidence(bin(x)) * r_U
```

Where:
- **P(H|bin(x))**: the posterior probability that a user is healthy given they
  fall into this bin. Computed during training (Algorithm 2) with Laplace
  smoothing. This handles the DIRECTION of evidence -- which class does this
  bin favor?

- **confidence(bin(x))**: the discriminative strength of the bin, computed as
  |P(H|bin) - P(U|bin)|. This handles the STRENGTH of evidence -- how strongly
  does the bin favor that class? A bin with P(H) = 0.52 has confidence 0.04
  (nearly useless), while P(H) = 0.95 has confidence 0.90 (very strong).

- **r_H, r_U**: the feature-level action weights from Algorithm 2. These
  handle the FEATURE QUALITY -- is this feature generally useful for this
  direction of evidence? A feature that is irrelevant to health status will
  have low r_H and r_U regardless of bin posteriors.

**Why this three-part formula works (with worked example):**

Consider a user being evaluated on heart rate (feature f):
- The user's heart rate is 95 bpm
- 95 bpm falls into bin 5 (trained on the node's users)
- Bin 5's posterior: P(H|bin 5) = 0.82, P(U|bin 5) = 0.18
- Bin 5's confidence: |0.82 - 0.18| = 0.64
- Heart rate's weights: r_H = 0.7, r_U = 0.5

Evidence for this feature:
- E_H = 0.82 * 0.64 * 0.7 = **0.367** (substantial healthy evidence)
- E_U = 0.18 * 0.64 * 0.5 = **0.058** (minimal unhealthy evidence)

Now consider a different user with heart rate 140 bpm in bin 9:
- Bin 9's posterior: P(H|bin 9) = 0.10, P(U|bin 9) = 0.90
- Bin 9's confidence: |0.10 - 0.90| = 0.80
- Same feature weights: r_H = 0.7, r_U = 0.5

Evidence:
- E_H = 0.10 * 0.80 * 0.7 = **0.056** (minimal healthy evidence)
- E_U = 0.90 * 0.80 * 0.5 = **0.360** (substantial unhealthy evidence)

A useless feature (P(H) = 0.52, confidence = 0.04, r_H = 0.3):
- E_H = 0.52 * 0.04 * 0.3 = **0.006** (negligible -- good, this tells us nothing)

**The decision rule (E1 + E4):**

After each feature is evaluated, check two conditions:
1. **Threshold (E1):** Has the winning AF reached the threshold T?
   - `af_h >= T` (for healthy) or `af_u >= T` (for unhealthy)
2. **Gap (E4):** Is the gap between the two AFs large enough?
   - `|af_h - af_u| >= delta`

Both conditions must be met. The threshold ensures enough total evidence has
accumulated. The gap ensures the evidence clearly favors one class over the
other. This prevents premature decisions when both AFs are low, and prevents
ambiguous decisions when both AFs are similar.

**Minimum feature requirement (E9):** No decision is made until at least
`k_min` features have been evaluated. This prevents snap decisions based on
one or two features, which might be unreliable due to measurement noise.

**Outlier handling (H5):** If a user's value is outside the range of training
data for this feature, it gets clipped to the nearest bin (first or last bin).
This is already handled by the `np.clip` in the bin lookup.

**Implementation:**

```python
def predict_dual_af(uid, data, nodes, models, retained, T, delta,
                    cap=0.5, min_support=5, k_min=2):
    """
    Dual-AF prediction with gap-based decision.

    Replaces the current predict() function entirely. Instead of a single AF
    with immediate UNHEALTHY on outside-range values, accumulates evidence in
    two directions and decides based on threshold + gap.

    Args:
        uid: user index in the original data matrix
        data: full data matrix (n_users x n_features)
        nodes: list of tree Node objects from Algorithm 1
        models: dict of (node_id, feature) -> FeatureModel from Algorithm 2
        retained: list of (feature, node_id, r_H, r_U) tuples from Algorithm 3
        T: threshold for the winning AF (E1) -- tuned via inner CV
        delta: minimum gap between AFs for a decision (E4) -- tuned via inner CV
        cap: maximum evidence contribution per feature (I1) -- tuned via inner CV
        min_support: minimum bin count to trust a posterior (I8) -- fixed at 5
        k_min: minimum features before allowing a decision (E9) -- tuned via inner CV

    Returns:
        tuple: (decision_string, number_of_PACs)
        decision_string is one of: "HEALTHY", "UNHEALTHY", "CONTRADICTORY", "SCREENING"
    """
    af_h = 0.0  # healthy assurance factor accumulator
    af_u = 0.0  # unhealthy assurance factor accumulator
    pacs = 0    # number of physical actions (features evaluated)

    # Tracking for contradiction detection (K2)
    features_favoring_h = 0
    features_favoring_u = 0

    # Route user through tree (same routing logic as current algorithm)
    lvl_nodes = _route_user(uid, data, nodes)
    healthy_idx = 0  # index of HEALTHY in all_cls (HEALTHY = 1, sorted first)

    # AF inherits across tree levels (G1):
    # Since af_h and af_u are simple running sums that are never reset,
    # evidence from the root node automatically carries into child nodes.
    # This is simpler than the current system which tracks af_at per node.

    for lvl in sorted(lvl_nodes.keys()):
        for nd in lvl_nodes[lvl]:
            # Get retained features for this node
            # Each action is (feature_idx, node_id, r_H, r_U)
            node_features = [(a[0], a[2], a[3]) for a in retained
                             if a[1] == nd.nid]

            # Sort by discriminative power -- most informative first (F1 + F8)
            # Use max(r_H, r_U) * average_confidence as the sort key
            # This is deterministic (F8) -- no randomness in feature ordering
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

                # Skip missing values -- Plan B does not use H1
                # (missing data does not contribute evidence)
                if np.isnan(v):
                    continue

                m = models.get((nd.nid, f))
                if not m:
                    continue

                # Find which bin the user's value falls into
                # np.clip handles outliers (H5): values outside training range
                # are assigned to the first or last bin automatically
                bin_idx = int(np.clip(
                    np.searchsorted(m.edges[1:], v, side='right'),
                    0, m.n_bins - 1
                ))

                # Minimum support check (I8):
                # If this bin had fewer than min_support training users,
                # its posterior is unreliable. Skip this feature for this user.
                if m.bin_counts[bin_idx] < min_support:
                    pacs += 1
                    continue

                # Compute evidence from the posterior, confidence, and weight
                p_h = m.posterior[bin_idx, healthy_idx]
                p_u = 1.0 - p_h
                conf = m.confidence[bin_idx]

                e_h = p_h * conf * r_h
                e_u = p_u * conf * r_u

                # Cap contributions (I1):
                # Prevents any single feature from dominating the AFs.
                # Without capping, a feature with perfect posterior (P(H)=1.0)
                # and high r_H could single-handedly push AF_H past the threshold.
                e_h = min(e_h, cap)
                e_u = min(e_u, cap)

                # Accumulate evidence (D1: additive accumulation)
                af_h += e_h
                af_u += e_u
                pacs += 1

                # Track which direction each feature's evidence points (K2)
                if e_h > e_u + 0.01:
                    features_favoring_h += 1
                elif e_u > e_h + 0.01:
                    features_favoring_u += 1

                # Decision check (only after k_min features) (E9)
                if pacs >= k_min:
                    gap = abs(af_h - af_u)

                    # Healthy decision: AF_H reached threshold AND gap is sufficient
                    if af_h >= T and gap >= delta:
                        return "HEALTHY", pacs

                    # Unhealthy decision: AF_U reached threshold AND gap is sufficient
                    if af_u >= T and gap >= delta:
                        return "UNHEALTHY", pacs

    # All features exhausted without reaching a decision.
    # Classify WHY no decision was reached (K2):

    # Contradiction detection: if many features point in both directions,
    # the evidence is genuinely conflicting (not just insufficient)
    if features_favoring_h >= 3 and features_favoring_u >= 3:
        balance = min(features_favoring_h, features_favoring_u) / \
                  max(features_favoring_h, features_favoring_u)
        if balance > 0.4:
            # Nearly equal numbers of features pointing each way
            return "CONTRADICTORY", pacs

    # Insufficient evidence: not enough features evaluated, or evidence
    # was too weak/ambiguous to reach the threshold with sufficient gap
    return "SCREENING", pacs
```

**Why each sub-change matters:**

- **F1 (most discriminative first):** By evaluating the most informative
  features first, the algorithm reaches decisions faster (fewer PACs). Strong
  features push AF_H or AF_U above the threshold quickly, so ambiguous features
  at the end may never need to be evaluated.

- **F8 (deterministic ordering):** The current algorithm uses a random first
  action (`rng.choice(root_ok)` at cds.py line 353). This makes results
  non-reproducible and wastes the first evaluation on a potentially weak feature.
  Plan B sorts deterministically by discriminative power, ensuring the best
  feature is always evaluated first.

- **G1 (AF inheritance):** The current algorithm tracks `af_at` per node and
  resets features per node (cds.py lines 369-375). Plan B simplifies this:
  af_h and af_u are running sums that are never reset, so evidence from the
  root node automatically carries forward into child nodes. This is the correct
  behavior because evidence from the root applies to all users regardless of
  which child node they route to.

- **I1 (capped contributions):** Without capping, a single feature with
  P(H|bin) = 0.99, confidence = 0.98, r_H = 0.9 would contribute
  0.99 * 0.98 * 0.9 = 0.87 to AF_H -- potentially enough to cross the threshold
  on its own. Capping at `cap` (e.g., 0.5) prevents any single feature from
  being decisive, forcing the algorithm to consider multiple features.

- **K2 (contradiction detection):** When both AF_H and AF_U are high but
  neither wins decisively, something interesting is happening: the features
  genuinely disagree about this user's health status. Rather than forcing an
  arbitrary decision, we label this "CONTRADICTORY" to flag it for clinical
  review. This is more informative than "SCREENING" (which means insufficient
  evidence).

- **H5 (outlier handling):** A user with a feature value far outside the
  training range (e.g., heart rate = 300 bpm when training max was 200) is
  clipped to the last bin. This is handled automatically by `np.clip(..., 0,
  m.n_bins - 1)` in the bin lookup. The last bin's posterior will typically
  favor unhealthy (since extreme values are rare in healthy users), so the
  outlier contributes appropriate evidence without causing an error.

---

### LOOCV Changes

#### Change 5.1: Inner Cross-Validation for Threshold Selection (I3)

**What it is:** Instead of hard-coding the hyperparameters T (threshold),
delta (gap), cap (evidence cap), and k_min (minimum features), use a 5-fold
cross-validation within each LOOCV fold to select the best combination.

**Why it helps:** The current algorithm hard-codes THRESHOLD = 0.025 (cds.py
line 7). This value was chosen once and applies to all nodes and users. But
the optimal threshold depends on the evidence model, which we've completely
changed. The old threshold was tuned for the original AF formula
(`afi(p, r, g) = p * r / g`); it's meaningless for the new dual-AF evidence
functions. We need to find new values empirically.

Hard-coding thresholds is also fragile: a value that works well on one dataset
may fail on another. Inner cross-validation finds the best values for each
LOOCV fold, adapting to the actual distribution of the training data.

**How it works:**

For each LOOCV fold (where user i is held out):
1. Train the tree, models, and feature selection on the remaining 451 users
2. Split the 451 training users into 5 inner folds
3. For each combination of hyperparameters, predict on each inner fold using
   models trained on the other 4 inner folds (or, for speed, reuse the outer
   fold's models as an approximation)
4. Select the combination with the highest accuracy
5. Use those hyperparameters to predict the held-out user i

**Coarse-to-fine search:** To avoid evaluating all 6*5*4*3 = 360 combinations,
use a staged approach:
1. Fix cap=0.5, k_min=2. Sweep T x delta (30 combinations)
2. Fix best T and delta. Sweep cap (4 combinations)
3. Fix best T, delta, cap. Sweep k_min (3 combinations)
Total: ~37 evaluations instead of 360.

**Implementation:**

```python
def select_hyperparams_inner_cv(data, labels, nodes, models, retained,
                                 n_folds=5, seed=42):
    """
    Use 5-fold CV on the training data to select the best hyperparameters
    for the dual-AF prediction.

    This function is called once per LOOCV fold (452 times total). It reuses
    the outer fold's models (trees and feature models trained on 451 users)
    rather than retraining per inner fold, which is a fast approximation.

    Args:
        data: training data matrix (451 users x 279 features)
        labels: training labels (451 values)
        nodes: tree nodes from Algorithm 1
        models: feature models from Algorithm 2
        retained: retained features from Algorithm 3
        n_folds: number of inner CV folds (default 5)
        seed: random seed for fold assignment

    Returns:
        tuple: (best_T, best_delta, best_cap, best_kmin)
    """
    # Hyperparameter grids
    T_grid     = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    delta_grid = [0.0, 0.3, 0.5, 1.0, 1.5]
    cap_grid   = [0.3, 0.5, 0.8, 1.0]
    k_min_grid = [1, 2, 3]

    # Create inner folds
    indices = np.arange(len(labels))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    def evaluate(T, delta, cap, k_min):
        """Evaluate a hyperparameter combination on all inner folds."""
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

    # Stage 1: Sweep T x delta (fix cap=0.5, k_min=2)
    best_score = -1
    best_T, best_delta = 1.5, 0.5  # defaults if all scores are equal
    for t in T_grid:
        for d in delta_grid:
            s = evaluate(t, d, 0.5, 2)
            if s > best_score:
                best_score = s
                best_T, best_delta = t, d

    # Stage 2: Sweep cap (fix best T, best delta, k_min=2)
    best_cap = 0.5
    for c in cap_grid:
        s = evaluate(best_T, best_delta, c, 2)
        if s > best_score:
            best_score = s
            best_cap = c

    # Stage 3: Sweep k_min (fix best T, best delta, best cap)
    best_kmin = 2
    for k in k_min_grid:
        s = evaluate(best_T, best_delta, best_cap, k)
        if s > best_score:
            best_score = s
            best_kmin = k

    return best_T, best_delta, best_cap, best_kmin
```

**Computational cost note:** Each `evaluate()` call predicts ~451 users. The
coarse-to-fine search evaluates ~37 combinations. That's ~16,700 prediction
calls per LOOCV fold. With 452 LOOCV folds, the total is ~7.5 million
predictions. Each prediction iterates over retained features (typically 10-30),
so total computation is substantial but feasible. If too slow:
- Reduce grid sizes (e.g., T_grid = [1.0, 2.0, 3.0])
- Reduce inner folds to 3 instead of 5
- Run inner CV only every 10th LOOCV fold and reuse the hyperparameters

**Integration into run_loocv:**

```python
def run_loocv(data, labels, max_users=None, seed=42):
    n = data.shape[0] if max_users is None else min(max_users, data.shape[0])
    is_bin = classify_features(data)
    all_cls = sorted(set(labels))
    results = []
    t0 = time.perf_counter()

    for i in range(n):
        mask = np.ones(data.shape[0], dtype=bool)
        mask[i] = False
        td, tl = data[mask], labels[mask]

        nodes = build_tree(td, tl, is_bin)

        all_models = {}
        actions_by_node = defaultdict(list)
        for nd in nodes:
            nm, na = train_node(nd, td, tl, is_bin, all_cls)
            all_models.update(nm)
            for a in na:
                actions_by_node[a[1]].append(a)

        retained = []
        for nd in nodes:
            node_kept = refine_node(nd, all_models,
                                    actions_by_node.get(nd.nid, []), td)
            retained.extend(node_kept)

        # Inner CV for hyperparameter selection (I3)
        T, delta, cap, k_min = select_hyperparams_inner_cv(
            td, tl, nodes, all_models, retained)

        # Backward elimination with tuned hyperparameters (L6)
        # (optional: can be done before or after inner CV)
        retained_final = []
        for nd in nodes:
            node_kept = [a for a in retained if a[1] == nd.nid]
            node_kept = backward_eliminate(nd, all_models, node_kept,
                                            td, tl, T, delta, cap, k_min)
            retained_final.extend(node_kept)

        # Predict the held-out user with tuned hyperparameters
        dec, npacs = predict_dual_af(i, data, nodes, all_models,
                                      retained_final, T, delta,
                                      cap=cap, k_min=k_min)

        tl_i = int(labels[i])
        ok = (dec != "UNHEALTHY") if tl_i == HEALTHY else (dec == "UNHEALTHY")
        results.append((i, tl_i, dec, ok, npacs))

        if (i + 1) % 50 == 0 or i == n - 1:
            acc = sum(r[3] for r in results) / len(results) * 100
            print(f"  [{i+1}/{n}] acc={acc:.1f}%  {time.perf_counter()-t0:.1f}s")

    return results
```

---

#### Change 5.2: Multi-Metric Reporting (M1)

**What it is:** Extend the current `print_report()` function (cds.py lines
448-477) to report additional metrics beyond accuracy, specificity, and
sensitivity. The new report includes precision (PPV), negative predictive value
(NPV), F1 score, balanced accuracy, average PACs broken down by class, and
decision rates for all four possible outcomes (HEALTHY, UNHEALTHY, SCREENING,
CONTRADICTORY).

**Why it helps:** The current report only shows accuracy, specificity,
sensitivity, and a decision count summary. This is insufficient for properly
evaluating Plan B because:

1. **Precision/PPV** tells you: of users the algorithm declared UNHEALTHY,
   what fraction were actually unhealthy? Low precision means many false alarms.

2. **NPV** tells you: of users the algorithm declared HEALTHY, what fraction
   were actually healthy? Low NPV means unhealthy users are being missed.

3. **F1 score** is the harmonic mean of precision and sensitivity, providing
   a single number that balances false positives and false negatives.

4. **Balanced accuracy** averages per-class accuracy, preventing the majority
   class (healthy users = 245/452 = 54%) from dominating the overall accuracy.

5. **SCREENING and CONTRADICTORY rates** are new decision outcomes in Plan B
   that need to be tracked. A high SCREENING rate (>15%) suggests the
   thresholds are too aggressive; a high CONTRADICTORY rate suggests feature
   selection isn't working well.

6. **PACs by class** reveals whether the algorithm is efficient for both
   healthy and unhealthy users, or if one group requires many more features.

**Implementation:**

```python
def compute_metrics(results):
    """
    Compute comprehensive evaluation metrics from LOOCV results.

    Args:
        results: list of (user_id, true_label, decision, correct, n_pacs) tuples

    Returns:
        dict with all computed metrics
    """
    n = len(results)
    metrics = {}

    # Split by true class
    th = [r for r in results if r[1] == HEALTHY]     # truly healthy users
    td = [r for r in results if r[1] != HEALTHY]     # truly diseased users

    # Basic accuracy
    metrics['accuracy'] = sum(r[3] for r in results) / n

    # Specificity: fraction of healthy users correctly classified as not-UNHEALTHY
    metrics['specificity'] = sum(r[3] for r in th) / len(th) if th else 0

    # Sensitivity: fraction of unhealthy users correctly classified as UNHEALTHY
    metrics['sensitivity'] = sum(r[3] for r in td) / len(td) if td else 0

    # Precision (PPV): of those declared UNHEALTHY, how many are truly unhealthy?
    dec_u = [r for r in results if r[2] == "UNHEALTHY"]
    metrics['precision'] = (sum(1 for r in dec_u if r[1] != HEALTHY)
                            / len(dec_u)) if dec_u else 0

    # NPV: of those declared HEALTHY, how many are truly healthy?
    dec_h = [r for r in results if r[2] == "HEALTHY"]
    metrics['npv'] = (sum(1 for r in dec_h if r[1] == HEALTHY)
                      / len(dec_h)) if dec_h else 0

    # F1 Score: harmonic mean of precision and sensitivity
    p, r = metrics['precision'], metrics['sensitivity']
    metrics['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0

    # Balanced accuracy: average per-class accuracy
    # Prevents the majority class from inflating overall accuracy
    per_class = {}
    for cls in sorted(set(r[1] for r in results)):
        cr = [r for r in results if r[1] == cls]
        per_class[cls] = sum(r[3] for r in cr) / len(cr)
    metrics['balanced_accuracy'] = np.mean(list(per_class.values()))

    # PACs (Physical Actions Count) -- number of features evaluated
    metrics['avg_pacs'] = np.mean([r[4] for r in results])
    metrics['pacs_healthy'] = np.mean([r[4] for r in th]) if th else 0
    metrics['pacs_unhealthy'] = np.mean([r[4] for r in td]) if td else 0

    # Decision rate breakdown
    for dec in ["HEALTHY", "UNHEALTHY", "SCREENING", "CONTRADICTORY"]:
        metrics[f'rate_{dec.lower()}'] = sum(1 for r in results if r[2] == dec) / n

    metrics['per_class'] = per_class

    return metrics


def print_report_extended(results):
    """
    Print a comprehensive evaluation report with all metrics.

    Replaces the current print_report() function.
    """
    m = compute_metrics(results)
    n = len(results)

    print(f"\n{'='*70}")
    print(f"CDS Dual-AF LOOCV — {n} users")
    print(f"{'='*70}")
    print(f"Accuracy:          {m['accuracy']*100:.1f}%")
    print(f"Balanced Accuracy: {m['balanced_accuracy']*100:.1f}%")
    print(f"Sensitivity:       {m['sensitivity']*100:.1f}%")
    print(f"Specificity:       {m['specificity']*100:.1f}%")
    print(f"Precision (PPV):   {m['precision']*100:.1f}%")
    print(f"NPV:               {m['npv']*100:.1f}%")
    print(f"F1 Score:          {m['f1']*100:.1f}%")
    print(f"")
    print(f"Avg PACs:          {m['avg_pacs']:.1f}")
    print(f"  Healthy users:   {m['pacs_healthy']:.1f}")
    print(f"  Unhealthy users: {m['pacs_unhealthy']:.1f}")
    print(f"")
    print(f"Decision rates:")
    for dec in ["healthy", "unhealthy", "screening", "contradictory"]:
        print(f"  {dec:15s}  {m[f'rate_{dec}']*100:.1f}%")
    print(f"")
    print(f"Per-class accuracy:")
    for cls, acc in sorted(m['per_class'].items()):
        lbl = "healthy" if cls == HEALTHY else f"class {cls}"
        print(f"  {lbl:>10s}  {acc*100:.1f}%")
    print(f"{'='*70}")
```

---

## Summary of Information Flow

```
Algorithm 1 (UNCHANGED)
    - Build tree with forced sex branching, U_MIN=200
    - Output: list of Node objects with user partitions
    |
    v
Algorithm 2 (CHANGED)
    - For each node and feature:
      1. Compute likelihood P(bin|class) with Laplace smoothing (B4)
         WHY: prevents zero probabilities that would break evidence functions
      2. Compute prevalence P(class), evidence P(bin), posterior P(class|bin)
         (unchanged formulas, but now use smoothed likelihoods)
      3. Store bin_counts per bin (I8)
         WHY: enables minimum support filtering during prediction
      4. Compute per-bin confidence = |P(H|bin) - P(U|bin)| (B3)
         WHY: scales evidence by discriminative strength of each bin
      5. Compute dual weights r_H, r_U per feature (C1)
         WHY: measures feature's expected contribution to each direction
      6. Keep healthy range fields but they are NOT used for decisions
    - Output: FeatureModel objects + action tuples (f, node_id, r_H, r_U)
    |
    v
Algorithm 3 (CHANGED)
    - Greedy selection by |r_H - r_U| descending
      WHY: replaces healthy-range-based set-cover with discriminative criterion
    - Correlation penalty (I4): skip features with correlation > 0.8 to
      already-retained features
      WHY: prevents double-counting evidence from correlated features
    - Backward elimination (L6): remove features that don't help accuracy
      WHY: prunes features that passed greedy selection but are net-negative
    - Output: pruned list of retained feature actions
    |
    v
Algorithm 4 (REPLACED)
    - Dual AF accumulators: AF_H and AF_U (both start at 0)
    - For each retained feature, compute evidence:
      E_H = P(H|bin) * confidence * r_H  (combines B1+B3+C1)
      E_U = P(U|bin) * confidence * r_U
    - Capped contributions: e_h = min(e_h, cap) (I1)
      WHY: no single feature can dominate the decision
    - Deterministic most-discriminative-first ordering (F1+F8)
      WHY: faster decisions, reproducible results
    - Minimum support check per bin (I8)
      WHY: skip unreliable posteriors from sparse bins
    - Minimum feature count before decision (E9)
      WHY: prevent snap decisions from 1-2 features
    - Gap-based decision: AF >= T AND gap >= delta (E1+E4)
      WHY: requires both sufficient evidence AND clear direction
    - Contradiction detection (K2)
      WHY: distinguish conflicting evidence from insufficient evidence
    - AFs inherit across tree levels automatically (G1)
      WHY: evidence from root applies to all users regardless of branch
    - Outliers clipped to nearest bin (H5)
      WHY: graceful handling of extreme values without errors
    - Output: decision (HEALTHY/UNHEALTHY/SCREENING/CONTRADICTORY) + PACs
    |
    v
LOOCV (CHANGED)
    - Inner 5-fold CV for (T, delta, cap, k_min) (I3)
      WHY: empirically tunes thresholds instead of hard-coding
    - Multi-metric reporting (M1)
      WHY: comprehensive evaluation beyond accuracy/specificity/sensitivity
```

## Hyperparameters to Tune (via Inner CV)

| Parameter | Description | Search Range | Effect of Higher Values |
|-----------|-------------|-------------|------------------------|
| T | Threshold for winning AF | [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] | More evidence needed -> fewer decisions, more SCREENING |
| delta | Minimum gap between AFs | [0.0, 0.3, 0.5, 1.0, 1.5] | Larger gap needed -> more confident decisions, more SCREENING |
| cap | Max evidence per feature | [0.3, 0.5, 0.8, 1.0] | Less capping -> individual features matter more |
| k_min | Min features before decision | [1, 2, 3] | More features needed -> slower but more reliable decisions |

Fixed parameters (not tuned):
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LAPLACE_ALPHA | 1.0 | Standard Laplace smoothing; widely used default |
| min_support | 5 | Standard minimum for reliable frequency estimates |
| CORR_THRESHOLD | 0.8 | Standard cutoff for "highly correlated" features |

## Expected Outcomes

- **Accuracy**: Should improve over the current algorithm because evidence is
  properly accumulated in both directions, preventing the brittleness of binary
  healthy-range decisions. The current algorithm misclassifies healthy users
  whose single feature is slightly outside range; Plan B would correctly
  accumulate predominantly healthy evidence for these users.

- **PACs**: May be slightly higher than the current algorithm for unhealthy
  users (because we no longer short-circuit on the first outside-range value)
  but should decrease for users that are clear cases, due to F1 ordering
  evaluating the most informative features first.

- **Sensitivity**: Should improve because unhealthy users who are borderline
  (inside the healthy range on some features but unhealthy on posterior
  probability) now contribute probabilistic unhealthy evidence instead of being
  counted as healthy.

- **Specificity**: Should improve significantly because healthy users near the
  boundary of the healthy range no longer trigger an immediate UNHEALTHY
  decision. Instead, their borderline features contribute weak unhealthy
  evidence that is outweighed by their many normal features.

- **SCREENING rate**: Tunable via T and delta. Higher T = more SCREENING.
  Target: 5-15%. A small SCREENING rate is acceptable and even desirable --
  these are genuinely ambiguous cases that warrant additional testing.

- **CONTRADICTORY rate**: Should be low (<5%). If high, it suggests the feature
  selection (Algorithm 3) is retaining features that genuinely disagree, which
  might indicate that the correlation penalty threshold should be tightened.

- **Interpretability**: Better than a single-number approach. You can inspect
  AF_H and AF_U to understand why a decision was made:
  - "AF_H = 3.2, AF_U = 0.8, gap = 2.4 > delta, so HEALTHY" is intuitive.
  - You can examine which features contributed to each AF to understand the
    clinical reasoning behind the decision.

## Implementation Order

For a developer implementing Plan B, the recommended order is:

1. **Add Laplace smoothing (Change 2.1)** -- minimal code change, foundational
2. **Add bin_counts and confidence to FeatureModel (Changes 2.2, 2.3)** --
   extends the data model
3. **Compute dual weights r_H, r_U (Change 2.4)** -- changes action tuple format
4. **Update Algorithm 3 (Changes 3.1, 3.2)** -- adapt feature selection to new
   action format (skip backward elimination for now)
5. **Implement predict_dual_af (Change 4.1)** -- the core prediction change
6. **Update run_loocv to use predict_dual_af** -- with hard-coded defaults
   (T=1.5, delta=0.5, cap=0.5, k_min=2) initially
7. **Run LOOCV and verify** -- ensure accuracy improves vs. current algorithm
8. **Add inner CV (Change 5.1)** -- replace hard-coded defaults with tuned values
9. **Add backward elimination (Change 3.3)** -- refine feature selection
10. **Add multi-metric reporting (Change 5.2)** -- better evaluation
