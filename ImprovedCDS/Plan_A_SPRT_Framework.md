# Plan A: SPRT Framework (Recommended Best Approach)

## Philosophy
Replace the single assurance factor and binary healthy-range test with a formal
Sequential Probability Ratio Test (SPRT). This is the most principled approach
because it provides theoretical guarantees on error rates and optimal stopping
(minimum features needed per user). The current algorithm is already an informal
SPRT — this plan formalizes it.

## Ideas Used

| ID | Name | Section Modified |
|----|------|-----------------|
| A8+A5 | Probabilistic bins (eliminate healthy range) | Algorithm 2 |
| B1 | Posterior probability table | Algorithm 2 |
| B4 | Posterior smoothing (Laplace) | Algorithm 2 |
| C1 | Dual weights r_H and r_U | Algorithm 2 + 3 |
| I8 | Minimum support requirement | Algorithm 2 + 4 |
| I4 | Feature correlation penalty | Algorithm 3 |
| L6 | Backward feature elimination | Algorithm 3 |
| K8 | SPRT framework | Algorithm 4 |
| N3/N4 | Bayes factor / posterior odds | Algorithm 4 (same as K8) |
| I1 | Capped LLR contributions | Algorithm 4 |
| F1 | Most discriminative first | Algorithm 4 |
| F8 | Deterministic first action | Algorithm 4 |
| G1 | AF inheritance across tree levels | Algorithm 4 |
| K2 | Contradictory evidence detector | Algorithm 4 |
| E9 | Minimum feature requirement | Algorithm 4 |
| I3 | Inner cross-validated threshold selection | LOOCV |
| M1 | Multi-metric evaluation | LOOCV reporting |
| H1 | Missing data as evidence | Algorithm 2 + 4 |
| H5 | Extreme outliers to nearest bin | Algorithm 4 |

## Ideas NOT Used (and Why)

| ID | Name | Why Excluded |
|----|------|-------------|
| Idea 2 | Adaptive healthy ranges | Incompatible — we eliminate healthy ranges entirely |
| D1 as separate dual-AF | Two separate accumulators | SPRT uses one cumulative LLR instead |
| B3 | Confidence weighting multiplier | Redundant — I1 (capping) + I8 (min support) achieve the same goal more simply |
| B6 | Density-based evidence | Added complexity; posteriors from B1 already encode density information |
| B8 | Graduated confidence reduction | Replaced by simpler I8 (hard cutoff) |
| B10 | Adaptive binning | Deferred — validate with Sturges first, optimize binning later |
| C7 | Distance-from-range-center | No healthy range exists in this plan |
| D2 | Log-odds accumulation | SPRT already works in log space |
| E1/E4 | Symmetric + gap thresholds | Replaced by SPRT boundaries (alpha/beta) |
| K1 | Two-phase prediction | Emerges naturally from SPRT + F1 ordering |
| O5 | Negative evidence amplification | Redundant with proper posteriors (B1) |

---

## Detailed Changes by Algorithm

---

### Algorithm 1: Decision Tree (MINIMAL CHANGES)

No structural changes. The tree construction remains the same:
- Load data, replace '?' with NaN
- Classify features as binary or continuous
- Build tree with forced sex branching at level 1
- Prune nodes with < U_MIN users
- Deduplicate nodes with identical user sets

The only change is that Algorithm 1 no longer needs to pass information
about healthy ranges downstream — it just partitions users into nodes.

---

### Algorithm 2: Perceptor & Executive Training (MAJOR CHANGES)

#### Current Behavior
For each node and feature, Algorithm 2 computes:
1. Likelihood table P(bin|class) — KEEP
2. Prevalence P(class) per node — KEEP
3. Evidence P(bin) — KEEP
4. Posterior P(class|bin) — KEEP
5. Healthy range [h_min, h_max] — REMOVE
6. Action weights based on r_outside — REPLACE

#### Change 2.1: Add Laplace Smoothing to Posteriors (B4)

**What changes:** After computing the likelihood table, add a small pseudocount
to prevent zero probabilities.

**Why:** With 452 users split across ~10 bins and 13 classes, many bins will
have zero users from certain classes. Zero likelihoods cause log(0) = -infinity
in the SPRT. Laplace smoothing prevents this.

**Implementation:**

In the `train_node` function, after computing the likelihood matrix `lk`:

```python
# Current code (line ~217-223 of cds.py):
lk = np.zeros((nb, nc))
for ci, c in enumerate(all_cls):
    cm = (lv == c)
    n_c = cm.sum()
    if n_c == 0:
        pass
    else:
        lk[:, ci] = np.bincount(ba[cm], minlength=nb).astype(float) / n_c

# NEW: Add Laplace smoothing
LAPLACE_ALPHA = 1.0  # pseudocount (1.0 = standard Laplace)
lk_smooth = np.zeros((nb, nc))
for ci, c in enumerate(all_cls):
    cm = (lv == c)
    n_c = cm.sum()
    counts = np.bincount(ba[cm], minlength=nb).astype(float) if n_c > 0 else np.zeros(nb)
    lk_smooth[:, ci] = (counts + LAPLACE_ALPHA) / (n_c + LAPLACE_ALPHA * nb)

lk = lk_smooth
```

**Effect:** A bin with 0 healthy users and 2 unhealthy users no longer gives
P(bin|H) = 0. Instead it gives P(bin|H) = 1/(n_H + nb), a very small but
nonzero value. This prevents infinite LLRs.

#### Change 2.2: Store Bin Counts for Minimum Support (I8)

**What changes:** Record how many total training users fall into each bin.

**Implementation:**

Add `bin_counts` to the FeatureModel class:

```python
class FeatureModel:
    __slots__ = ('likelihood', 'prevalence', 'evidence', 'posterior',
                 'h_min', 'h_max', 'n_bins', 'edges', 'min_hbin', 'max_hbin',
                 'bin_counts')  # NEW

    def __init__(self, likelihood, prevalence, evidence, posterior,
                 h_min, h_max, n_bins, edges, min_hbin, max_hbin, bin_counts):
        # ... existing assignments ...
        self.bin_counts = bin_counts  # NEW
```

In `train_node`, after computing bin assignments `ba`:

```python
bin_counts = np.bincount(ba, minlength=nb)  # total users per bin
```

Pass `bin_counts` to the FeatureModel constructor.

#### Change 2.3: Compute Dual Action Weights r_H and r_U (C1)

**What changes:** Instead of computing a single action weight `r` (fraction of
disease likelihood outside the healthy range), compute two weights:
- `r_H`: expected healthy evidence from this feature
- `r_U`: expected unhealthy evidence from this feature

**Why:** The current `r` only measures how much a feature reveals abnormality.
With dual weights, we also know how much a feature confirms health. This
enables proper feature ordering (F1) and feeds the SPRT.

**Implementation:**

Replace the current action weight computation (lines 253-263 of cds.py):

```python
# REPLACE the entire action-weight block with:

# Compute dual weights for this feature
# r_H = expected P(H|bin) across all bins, weighted by how many healthy
#        users fall in each bin
# r_U = expected P(U|bin) across all bins, weighted by how many unhealthy
#        users fall in each bin
healthy_idx = all_cls.index(HEALTHY)

# Weight by bin population for each class
r_H = 0.0
r_U = 0.0
n_healthy_total = sum(1 for l in lv if l == HEALTHY)
n_unhealthy_total = sum(1 for l in lv if l != HEALTHY)

for b in range(nb):
    p_h_given_bin = post[b, healthy_idx]
    p_u_given_bin = 1.0 - p_h_given_bin

    # How many healthy users are in this bin?
    n_h_bin = sum(1 for j in range(len(ba)) if ba[j] == b and lv[j] == HEALTHY)
    n_u_bin = sum(1 for j in range(len(ba)) if ba[j] == b and lv[j] != HEALTHY)

    if n_healthy_total > 0:
        r_H += p_h_given_bin * (n_h_bin / n_healthy_total)
    if n_unhealthy_total > 0:
        r_U += p_u_given_bin * (n_u_bin / n_unhealthy_total)

# Store as action: (feature_idx, node_id, r_H, r_U)
if r_H > 0 or r_U > 0:
    actions.append((f, node.nid, r_H, r_U))
```

**Note:** The action tuple format changes from `(f, node_id, class_h, r)` to
`(f, node_id, r_H, r_U)`. This affects Algorithm 3 and 4.

#### Change 2.4: Compute Missing-Data Evidence (H1)

**What changes:** For each feature in each node, compute P(missing|H) and
P(missing|U) — the probability that the feature is missing given the user's
class.

**Implementation:**

```python
# After the main feature loop in train_node:
n_missing_h = sum(1 for u in node.uidx
                  if np.isnan(data[u, f]) and labels[u] == HEALTHY)
n_missing_u = sum(1 for u in node.uidx
                  if np.isnan(data[u, f]) and labels[u] != HEALTHY)
n_h = node.hdist.get(HEALTHY, 0)
n_u = node.nu - n_h

p_missing_h = n_missing_h / max(n_h, 1)
p_missing_u = n_missing_u / max(n_u, 1)
```

Store `p_missing_h` and `p_missing_u` in the FeatureModel (add two new slots).

**Only use this if** the missing rates differ substantially between classes
(e.g., |p_missing_h - p_missing_u| > 0.05). Otherwise, missing data is
uninformative and should contribute LLR = 0.

#### Change 2.5: Remove Healthy Range (A8+A5)

**What changes:** The fields `h_min`, `h_max`, `min_hbin`, `max_hbin` in
FeatureModel are no longer used for decision-making. They can be kept for
diagnostic purposes but play no role in prediction.

The concept of "inside/outside healthy range" is replaced by:
"which bin does the user fall into, and what are the posteriors for that bin?"

---

### Algorithm 3: Refinement / Feature Selection (MODERATE CHANGES)

#### Current Behavior
Greedy set-cover: sort actions by `r` (action weight), keep features that
catch new unhealthy users outside the healthy range.

#### Change 3.1: Update Greedy Criterion for Probabilistic Framework

**What changes:** Since there's no healthy range, we can't check "is this user
outside the range?" Instead, the greedy pass retains features that provide
high discriminative evidence.

**New criterion:** A feature is useful if it has bins where the posterior
strongly favors one class. Measure this by `|r_H - r_U|` — the absolute
difference between the feature's healthy and unhealthy evidence weights.

**Implementation:**

```python
def refine_node(node, models, node_actions, data):
    # Sort by discriminative power: |r_H - r_U| descending
    na = sorted([a for a in node_actions if abs(a[2] - a[3]) > 0.01],
                key=lambda a: abs(a[2] - a[3]), reverse=True)
    if not na:
        return []

    # Greedy selection: keep features that improve classification
    # on the training users in this node
    kept = []
    kept_features = set()

    for a in na:
        f = a[0]
        if f in kept_features:
            continue

        m = models.get((node.nid, f))
        if not m:
            continue

        # Check: does this feature provide new information?
        # Use the posteriors to see if this feature would change
        # any user's cumulative evidence direction
        raw = data[node.uidx, f]
        vm = ~np.isnan(raw)
        if vm.sum() == 0:
            continue

        # Feature provides value if it has strong posteriors
        # (already filtered by |r_H - r_U| > 0.01 above)
        kept.append(a)
        kept_features.add(f)

    return kept
```

#### Change 3.2: Add Correlation Penalty (I4)

**What changes:** Before adding a feature to the retained set, check its
correlation with already-retained features. Skip if correlation > 0.8.

**Implementation:**

```python
def refine_node(node, models, node_actions, data):
    na = sorted([a for a in node_actions if abs(a[2] - a[3]) > 0.01],
                key=lambda a: abs(a[2] - a[3]), reverse=True)
    if not na:
        return []

    # Precompute correlations between candidate features
    nd_data = data[node.uidx]
    feature_ids = sorted(set(a[0] for a in na))
    correlations = {}
    for i, f1 in enumerate(feature_ids):
        for f2 in feature_ids[i+1:]:
            col1 = nd_data[:, f1]
            col2 = nd_data[:, f2]
            valid = ~(np.isnan(col1) | np.isnan(col2))
            if valid.sum() > 10:
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

        # Correlation check
        if kept_features:
            max_corr = max(correlations.get((f, kf), 0.0) for kf in kept_features)
            if max_corr > CORR_THRESHOLD:
                continue

        kept.append(a)
        kept_features.add(f)

    return kept
```

#### Change 3.3: Backward Elimination (L6)

**What changes:** After the greedy pass, try removing each feature one at a
time. If removing it improves (or doesn't hurt) accuracy on training users,
remove it permanently.

**Implementation:**

```python
def backward_eliminate(node, models, kept, data, labels, alpha, beta, cap):
    """Remove features that hurt accuracy."""
    if len(kept) <= 2:
        return kept

    def score(feature_set):
        correct = 0
        for uid in node.uidx:
            dec = predict_single_user_sprt(uid, node, models, feature_set,
                                           data, alpha, beta, cap)
            true_label = labels[uid]
            if (dec != "UNHEALTHY" and true_label == HEALTHY) or \
               (dec == "UNHEALTHY" and true_label != HEALTHY):
                correct += 1
        return correct / len(node.uidx)

    improved = True
    while improved and len(kept) > 2:
        improved = False
        base_score = score(kept)
        worst_idx = None

        for i in range(len(kept)):
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

---

### Algorithm 4: Prediction (MAJOR CHANGES)

This is where the SPRT replaces the single-AF mechanism.

#### Change 4.1: SPRT Core (K8 + N3/N4)

**What changes:** Replace the entire prediction loop. Instead of accumulating
a single AF and checking against THRESHOLD, accumulate a log-likelihood ratio
and check against SPRT boundaries.

**Implementation:**

```python
def predict_sprt(uid, data, nodes, models, retained, alpha, beta, cap,
                 min_support=5, k_min=2):
    """
    SPRT-based prediction.

    Args:
        uid: user index in original data array
        data: full data matrix
        nodes: list of tree nodes
        models: dict of (node_id, feature) -> FeatureModel
        retained: list of (feature, node_id, r_H, r_U) tuples
        alpha: desired false positive rate (healthy called unhealthy)
        beta: desired false negative rate (unhealthy called healthy)
        cap: maximum |LLR| contribution per feature
        min_support: minimum bin count to trust posterior
        k_min: minimum features before allowing a decision
    """
    # SPRT boundaries
    A = math.log((1 - beta) / max(alpha, 1e-10))   # upper (unhealthy)
    B = math.log(max(beta, 1e-10) / (1 - alpha))    # lower (healthy)

    cumulative_llr = 0.0
    pacs = 0

    # Tracking for contradiction detection (K2)
    total_pos_evidence = 0.0  # sum of positive LLRs
    total_neg_evidence = 0.0  # sum of |negative LLRs|
    n_pos = 0
    n_neg = 0

    # Route user through tree
    lvl_nodes = _route_user(uid, data, nodes)

    for lvl in sorted(lvl_nodes.keys()):
        for nd in lvl_nodes[lvl]:
            # Get retained features for this node, sorted by
            # discriminative power (F1)
            node_features = [(a[0], a[2], a[3]) for a in retained
                             if a[1] == nd.nid]

            # Sort by max(r_H, r_U) descending — most informative first (F1)
            node_features.sort(key=lambda x: max(x[1], x[2]), reverse=True)

            # Deterministic ordering — no randomness (F8)
            for f, r_h, r_u in node_features:
                v = data[uid, f]

                # Handle missing data (H1)
                if np.isnan(v):
                    m = models.get((nd.nid, f))
                    if m and hasattr(m, 'p_missing_h') and hasattr(m, 'p_missing_u'):
                        if abs(m.p_missing_h - m.p_missing_u) > 0.05:
                            p_mh = max(m.p_missing_h, 1e-10)
                            p_mu = max(m.p_missing_u, 1e-10)
                            llr = math.log(p_mu / p_mh)
                            llr = max(-cap, min(cap, llr))
                            cumulative_llr += llr
                            pacs += 1
                    continue

                m = models.get((nd.nid, f))
                if not m:
                    continue

                # Find which bin the user's value falls into
                bin_idx = int(np.clip(
                    np.searchsorted(m.edges[1:], v, side='right'),
                    0, m.n_bins - 1
                ))

                # Handle extreme outliers (H5): clip to nearest bin
                # (already handled by clip above)

                # Minimum support check (I8)
                if m.bin_counts[bin_idx] < min_support:
                    pacs += 1  # count as evaluated but contribute no evidence
                    continue

                # Compute log-likelihood ratio
                healthy_idx = 0  # assuming HEALTHY is first in all_cls
                p_h = m.likelihood[bin_idx, healthy_idx]
                p_u = sum(m.likelihood[bin_idx, ci]
                          for ci in range(m.likelihood.shape[1])
                          if ci != healthy_idx)

                # Both guaranteed > 0 due to Laplace smoothing (B4)
                if p_h > 0 and p_u > 0:
                    raw_llr = math.log(p_u / p_h)
                else:
                    raw_llr = 0.0

                # Cap the contribution (I1)
                llr = max(-cap, min(cap, raw_llr))

                cumulative_llr += llr
                pacs += 1

                # Track for contradiction detection (K2)
                if llr > 0.1:
                    total_pos_evidence += llr
                    n_pos += 1
                elif llr < -0.1:
                    total_neg_evidence += abs(llr)
                    n_neg += 1

                # SPRT decision (only after k_min features evaluated)
                if pacs >= k_min:
                    if cumulative_llr >= A:
                        return "UNHEALTHY", pacs
                    elif cumulative_llr <= B:
                        return "HEALTHY", pacs

    # Neither boundary crossed — classify the outcome (K2)
    total_evidence = total_pos_evidence + total_neg_evidence

    if total_evidence > 3.0:
        balance = min(total_pos_evidence, total_neg_evidence) / \
                  max(total_pos_evidence, total_neg_evidence, 1e-12)
        if balance > 0.4:
            return "CONTRADICTORY", pacs

    return "SCREENING", pacs
```

#### Change 4.2: AF Inheritance Across Tree Levels (G1)

**What changes:** The cumulative LLR carries forward from parent to child
nodes automatically — it's a single running sum. When the user routes from
root to a child node, the LLR accumulated at the root is already included.

This is simpler than the current system where `af_at` tracks per-node AF
values. In the SPRT, there's just one number: `cumulative_llr`.

**Implementation:** Already handled in Change 4.1 above. The cumulative_llr
is not reset when moving between nodes — it persists across the entire
prediction.

---

### LOOCV Changes

#### Change 5.1: Inner Cross-Validation for Thresholds (I3)

**What changes:** Instead of hard-coding alpha and beta, use an inner 5-fold
CV within each LOOCV fold to select the best (alpha, beta, cap, k_min)
combination.

**Implementation:**

```python
def select_hyperparams_inner_cv(data, labels, nodes, models, retained,
                                 n_folds=5, seed=42):
    """
    Use 5-fold CV on the training data to select best hyperparameters.
    Reuses the outer fold's models (fast shortcut).
    """
    alpha_grid = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
    beta_grid  = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
    cap_grid   = [0.5, 1.0, 1.5, 2.0, 3.0]
    k_min_grid = [1, 2, 3, 5]

    # To keep computation tractable, use a coarse-to-fine approach:
    # 1. Fix cap=1.5, k_min=2. Sweep alpha x beta.
    # 2. Fix best alpha/beta. Sweep cap.
    # 3. Fix best alpha/beta/cap. Sweep k_min.

    indices = np.arange(len(labels))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    def evaluate(alpha, beta, cap, k_min):
        correct = 0
        total = 0
        for fold_idx in range(n_folds):
            for uid in folds[fold_idx]:
                dec, _ = predict_sprt(uid, data, nodes, models, retained,
                                       alpha, beta, cap, k_min=k_min)
                true_label = labels[uid]
                ok = (dec != "UNHEALTHY") if true_label == HEALTHY else (dec == "UNHEALTHY")
                correct += ok
                total += 1
        return correct / total

    # Stage 1: Sweep alpha x beta
    best_score = -1
    best_alpha, best_beta = 0.05, 0.10
    for a in alpha_grid:
        for b in beta_grid:
            s = evaluate(a, b, 1.5, 2)
            if s > best_score:
                best_score = s
                best_alpha, best_beta = a, b

    # Stage 2: Sweep cap
    best_cap = 1.5
    for c in cap_grid:
        s = evaluate(best_alpha, best_beta, c, 2)
        if s > best_score:
            best_score = s
            best_cap = c

    # Stage 3: Sweep k_min
    best_kmin = 2
    for k in k_min_grid:
        s = evaluate(best_alpha, best_beta, best_cap, k)
        if s > best_score:
            best_score = s
            best_kmin = k

    return best_alpha, best_beta, best_cap, best_kmin
```

**Note on computational cost:** The coarse-to-fine approach evaluates
~6*6 + 5 + 4 = 45 configurations instead of 6*6*5*4 = 720. Each evaluation
predicts ~451 users. Total: ~20,000 predictions per LOOCV fold. With 452
folds, that's ~9 million predictions. This is feasible but slow. Consider
caching predictions or reducing the grid.

#### Change 5.2: Multi-Metric Reporting (M1)

**What changes:** Extend `print_report()` to track precision, NPV, F1,
balanced accuracy, PACs by class, and SCREENING/CONTRADICTORY rates.

**Implementation:**

```python
def compute_metrics(results):
    n = len(results)
    metrics = {}

    # Basic counts
    th = [r for r in results if r[1] == HEALTHY]
    td = [r for r in results if r[1] != HEALTHY]

    metrics['accuracy'] = sum(r[3] for r in results) / n
    metrics['specificity'] = sum(r[3] for r in th) / len(th) if th else 0
    metrics['sensitivity'] = sum(r[3] for r in td) / len(td) if td else 0

    # Precision (PPV)
    dec_u = [r for r in results if r[2] == "UNHEALTHY"]
    metrics['precision'] = (sum(1 for r in dec_u if r[1] != HEALTHY)
                            / len(dec_u)) if dec_u else 0

    # NPV
    dec_h = [r for r in results if r[2] == "HEALTHY"]
    metrics['npv'] = (sum(1 for r in dec_h if r[1] == HEALTHY)
                      / len(dec_h)) if dec_h else 0

    # F1
    p, r = metrics['precision'], metrics['sensitivity']
    metrics['f1'] = 2*p*r/(p+r) if (p+r) > 0 else 0

    # Balanced accuracy
    per_class = {}
    for cls in sorted(set(r[1] for r in results)):
        cr = [r for r in results if r[1] == cls]
        per_class[cls] = sum(r[3] for r in cr) / len(cr)
    metrics['balanced_accuracy'] = np.mean(list(per_class.values()))

    # PACs
    metrics['avg_pacs'] = np.mean([r[4] for r in results])
    metrics['pacs_healthy'] = np.mean([r[4] for r in th]) if th else 0
    metrics['pacs_unhealthy'] = np.mean([r[4] for r in td]) if td else 0

    # Decision rates
    for dec in ["HEALTHY", "UNHEALTHY", "SCREENING", "CONTRADICTORY"]:
        metrics[f'rate_{dec.lower()}'] = sum(1 for r in results if r[2] == dec) / n

    metrics['per_class'] = per_class

    return metrics


def print_report_extended(results):
    m = compute_metrics(results)
    n = len(results)

    print(f"\n{'='*70}")
    print(f"CDS SPRT LOOCV - {n} users")
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
Algorithm 1 (unchanged)
    |
    v
Algorithm 2 (CHANGED)
    - Posterior tables with Laplace smoothing (B1+B4)
    - Bin counts stored (I8)
    - Dual weights r_H, r_U computed (C1)
    - Missing-data evidence computed (H1)
    - Healthy range fields kept but unused
    |
    v
Algorithm 3 (CHANGED)
    - Greedy selection by |r_H - r_U| instead of r_outside
    - Correlation penalty (I4): skip features corr > 0.8
    - Backward elimination (L6): remove features that hurt accuracy
    |
    v
Algorithm 4 (REPLACED)
    - SPRT with cumulative log-likelihood ratio (K8/N3/N4)
    - Capped LLR per feature (I1)
    - Deterministic most-discriminative-first ordering (F1+F8)
    - Minimum support check per bin (I8)
    - Minimum feature count before decision (E9)
    - Missing data contributes evidence (H1)
    - Outliers clipped to nearest bin (H5)
    - Contradiction detection on inconclusive cases (K2)
    - LLR inherits across tree levels (G1)
    |
    v
LOOCV (CHANGED)
    - Inner 5-fold CV for (alpha, beta, cap, k_min) (I3)
    - Multi-metric reporting (M1)
```

## Hyperparameters to Tune (via Inner CV)

| Parameter | Description | Search Range |
|-----------|-------------|-------------|
| alpha | False positive rate | [0.01, 0.03, 0.05, 0.10, 0.15, 0.20] |
| beta | False negative rate | [0.01, 0.03, 0.05, 0.10, 0.15, 0.20] |
| cap | Max |LLR| per feature | [0.5, 1.0, 1.5, 2.0, 3.0] |
| k_min | Min features before decision | [1, 2, 3, 5] |

Fixed parameters (not tuned):
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LAPLACE_ALPHA | 1.0 | Standard Laplace smoothing |
| min_support | 5 | Standard minimum bin count |
| CORR_THRESHOLD | 0.8 | Standard correlation cutoff |
| LLR_MEANINGFUL | 0.1 | Threshold for contradiction tracking |

## Expected Outcomes

- **Accuracy**: Should improve because evidence is properly accumulated in
  both directions, preventing the brittleness of binary healthy-range decisions.
- **PACs**: Should decrease because SPRT has optimal stopping — clear cases
  decided in 2-5 features instead of exhausting the action space.
- **Sensitivity**: Should improve because unhealthy users who are borderline
  (inside healthy range on some features) now contribute probabilistic evidence
  instead of being ignored.
- **Specificity**: Should improve because healthy users near the boundary now
  contribute healthy evidence instead of triggering an immediate UNHEALTHY.
- **SCREENING rate**: May decrease (more decisive), but CONTRADICTORY cases
  may emerge. Target: combined SCREENING + CONTRADICTORY < 15%.
