"""
================================================================================
FAIRNESS_IMPLEMENTATION_GUIDE.md – Equalized Odds Post-Processing for CDS
================================================================================

## Overview

This guide documents the implementation of equalized odds post-processing 
following **Hardt et al. (2016): "Equality of Opportunity in Supervised Learning"** 
(arXiv:1610.02413) integrated into the CDS Algorithm pipeline.

### Goal
Ensure equal **True Positive Rate (TPR)** across male and female users while 
minimizing loss in overall accuracy.

---

## 1. MATHEMATICAL FRAMEWORK

### 1.1 Equalized Odds Constraint

For two demographic groups A (male) and B (female), equalized odds requires:

```
TPR_A(t_A) = TPR_B(t_B)
```

where:
- `TPR_A(t_A)` = true positive rate for group A using threshold t_A
- `TPR_B(t_B)` = true positive rate for group B using threshold t_B
- A true positive = correctly predicting "unhealthy" for a user who is actually diseased

### 1.2 CDS Context

In the CDS algorithm:
- **Prediction score**: `rw` (remaining work) = `1 - AF` (assurance factor)
- **Decision rule**: User is classified as unhealthy if `rw ≤ threshold`
  - Lower rw → higher confidence in healthy
  - Higher rw → lower confidence in healthy

- **Ground truth labels**:
  - 1 = healthy (negative class)
  - 2-16 = various arrhythmia types (positive class / diseased)

### 1.3 Metrics

For each threshold and group:

```
TPR = TP / (TP + FN)      [True Positive Rate / Sensitivity]
FPR = FP / (FP + TN)      [False Positive Rate]
FNR = FN / (FN + TP)      [False Negative Rate]
```

---

## 2. IMPLEMENTATION ARCHITECTURE

### 2.1 Module Structure

```
CDS_Paper_Algorithms.py          [Algorithm 1 – Decision Tree]
    ↓
Algorithm2.py                     [Algorithm 2 – Perceptor & Executive]
    ↓
Algorithm3.py                     [Algorithm 3 – Action Refinement]
    ↓
Algorithm4.py                     [Algorithm 4 – Prediction / Inference]
    ↓
Algorithm4_FairnessIntegration.py [Extract scores, apply fairness]
    ↓
Fairness_EqualizedOdds.py         [Core fairness post-processing]
    ↓
Optimal gender-specific thresholds (output)
```

### 2.2 Core Classes and Functions

#### **Fairness_EqualizedOdds.py**

**ROCPoint**
- Represents one operating point on an ROC curve
- Stores: threshold, TPR, FPR, and confusion matrix entries (TP, FP, TN, FN)

**ROCCurve**
- Full ROC curve for one demographic group
- Contains list of ROCPoint objects computed across threshold range

**ConvexHullResult**
- Represents the Pareto frontier of achievable (TPR, FPR) pairs
- Filters to only points satisfying equalized TPR constraint

**EqualizedOddsResult**
- Final output: gender-specific thresholds and fairness metrics
- Used to replace DIAGNOSTIC_THRESHOLD in deployment

#### **Algorithm4_FairnessIntegration.py**

**FairnessMetrics**
- Aggregates TPR, FPR, accuracy for baseline vs. fair predictions
- Reports fairness gaps (how much difference between genders)

**FairnessPostProcessingOutput**
- Complete integration result
- Contains both original and fair predictions for comparison

---

## 3. ALGORITHM WORKFLOW

### Step 1: Compute ROC Curves by Gender

```python
roc_male = compute_roc_curve(scores[male_mask], labels[male_mask], "Male")
roc_female = compute_roc_curve(scores[female_mask], labels[female_mask], "Female")
```

For each unique threshold, compute:
- Predictions: `ŷ = 1 if rw ≤ threshold else 0`
- TPR, FPR using confusion matrix

**Output**: 100+ points per ROC curve

### Step 2: Find Convex Hull (Pareto Frontier)

```python
hull = compute_convex_hull(roc_male, roc_female)
```

Filter to threshold pairs `(t_A, t_B)` where:
- `TPR_A(t_A) = TPR_B(t_B)` ← **Equalized odds constraint**
- Different pairs → different accuracy-fairness tradeoffs

**Output**: List of feasible operating points

### Step 3: Optimize Loss

```python
loss = compute_loss(hull, baseline_threshold=0.025)
```

Search convex hull for thresholds that minimize:
```
L(t_A, t_B) = (FPR_A + FPR_B + FNR_A + FNR_B) / 4
```

subject to equal TPR.

**Output**: Optimal threshold pair and utility loss

### Step 4: Apply Fair Thresholds

```python
predictions_fair = apply_equalized_odds_thresholds(rw_scores, data, fairness_result)
```

Reclassify using gender-specific thresholds:
- Males: use `threshold_male`
- Females: use `threshold_female`

---

## 4. DESIGN DECISIONS & DILEMMAS

### 4.1 ROC Point Selection [RESOLUTION]

**Dilemma**: How many thresholds to evaluate?
- Too few: miss optimal point
- Too many: computational cost

**Resolution**: Hybrid approach
- Include extreme thresholds: `-∞` (predict all healthy) and `+∞` (predict all unhealthy)
- Sample unique score values + uniform linspace
- Default: ~100 points per group

**Code**:
```python
thresholds = np.concatenate([
    np.array([-np.inf, np.inf]),
    unique_scores,
    np.linspace(scores.min(), scores.max(), n_points),
])
```

### 4.2 Equalized Odds vs. Demographic Parity [CLARIFICATION]

**Hardt et al. propose two fairness notions**:

1. **Demographic Parity**: P(ŷ=1 | group=A) = P(ŷ=1 | group=B)
   - Equal prediction rates (not group-aware)

2. **Equalized Odds** ← *This implementation*
   - Equal TPR AND equal FPR across groups
   - More nuanced: requires different treatment for error types

**Why Equalized Odds?**
- Better for healthcare: TPR (sensitivity) is critical
- Equal sensitivity means diagnostic reliability is same for both groups
- Prevents group-specific diagnostic disparities

**Implementation note**: We enforce equal TPR; FPR may differ slightly.

### 4.3 Loss Function Specification [RESOLUTION]

**Dilemma**: What loss to minimize on the convex hull?
- Equal cost for all error types?
- Weighted by group size?
- Weighted by misclassification severity?

**Resolution**: Balanced error rate (default)
```python
loss = (FPR + FNR) / 2
```

**Rationale**:
- FPR = false alarm rate (healthy → unhealthy)
- FNR = missed disease rate (unhealthy → healthy)
- Both equally important in healthcare

**Custom cost function**: Users can specify `cost_fn` parameter in `compute_loss()`
```python
def custom_cost(t_male, t_female, point):
    fpr, tpr, fnr = point
    # Custom weighting: e.g., heavily penalize FNR
    return 0.3 * fpr + 0.7 * fnr
```

### 4.4 Handling Empty Convex Hull [RESOLUTION]

**Dilemma**: What if no threshold pair satisfies equalized TPR?

**Resolution**: Fallback to baseline
```python
if not hull.hull_points:
    log.warning("No feasible equalized odds solution")
    return _fallback_loss_result(hull, baseline_threshold)
```

Returns baseline threshold for both groups with warning.

**Why this occurs**:
- Very imbalanced group sizes
- Very different baseline performances
- Insufficient data resolution

**Mitigation**: Use finer ROC point granularity

### 4.5 Handling NaN and Missing Values [RESOLUTION]

**Dilemma**: Should NaN rw scores be treated specially?

**Resolution**: No special treatment
- NaN comparisons (e.g., `NaN ≤ threshold`) return False
- NaN users are classified as negative (healthy)

**Reasoning**:
- Algorithm 4 should not produce NaN rw values
- If present, likely due to incomplete prediction trace
- Conservative: don't flag as unhealthy

### 4.6 Gender Encoding [SPECIFICATION]

**Encoding**:
- Column 1 (0-indexed) in arrhythmia dataset
- 0.0 = male
- 1.0 = female

**Code**:
```python
SEX_FEATURE_IDX = 1
MALE_VALUE = 0.0
FEMALE_VALUE = 1.0
```

**Extensibility**: To add more protected attributes (e.g., age groups):
- Replicate ROC curve computation per group
- Adapt convex hull to 3+ dimensional case (more complex optimization)

### 4.7 Threshold Comparison Direction [CLARIFICATION]

**Important**: Decision rule is `rw ≤ threshold`

Low rw = high confidence in healthy → don't alarm
High rw = low confidence in healthy → alarm

Example:
- User A: rw = 0.01 (confident healthy)
- User B: rw = 0.10 (less confident)
- If threshold = 0.025:
  - A: rw ≤ 0.025 ✓ → HEALTHY
  - B: rw > 0.025 ✗ → SCREENING/UNHEALTHY

---

## 5. INTEGRATION WITH ALGORITHM 4

### 5.1 Score Extraction

Extract `rw` (remaining work) from Algorithm4Output:
```python
rw_scores = extract_rw_scores_from_algorithm4(alg4_output)
# Uses final rw from af_trace[-1].rw_real
```

### 5.2 Prediction Reclassification

Original predictions from Algorithm 4:
```
HealthDecision.UNHEALTHY → binary 1
HealthDecision.HEALTHY   → binary 0
HealthDecision.SCREENING → binary 0 (conservative)
```

Fair predictions using gender-specific thresholds:
```python
predictions_fair = apply_equalized_odds_thresholds(
    rw_scores, data, fairness_result
)
```

### 5.3 Comparison Metrics

**Before fairness**:
- TPR_male might be 0.95, TPR_female = 0.87 → gap of 0.08

**After fairness**:
- TPR_male = TPR_female = 0.90 → gap of 0.00 ✓

**Cost**: Overall accuracy drops slightly (utility loss)

---

## 6. DEPLOYMENT CONSIDERATIONS

### 6.1 Using Fair Thresholds in Production

After running fairness post-processing:

```python
fairness_result = compute_bayes_optimal_predictor(
    rw_scores, true_labels, data,
    baseline_threshold=0.025
)

# Extract thresholds
threshold_male = fairness_result.threshold_male
threshold_female = fairness_result.threshold_female

# For new prediction on unknown user:
def predict_user(rw_score, gender):
    if gender == "male":
        return "unhealthy" if rw_score <= threshold_male else "healthy"
    else:
        return "unhealthy" if rw_score <= threshold_female else "healthy"
```

### 6.2 Monitoring and Validation

After deployment, continuously monitor:
```python
# Real-world TPR by gender
tpr_male_live = (TP_male) / (TP_male + FN_male)
tpr_female_live = (TP_female) / (TP_female + FN_female)

# Should remain approximately equal (tpr_gap ≈ 0)
```

---

## 7. LIMITATIONS & FUTURE WORK

### 7.1 Current Limitations

1. **Binary gender**: Implementation assumes only M/F. 
   - Extension: Create separate ROC curves per gender category

2. **Single protected attribute**: Only considers gender
   - Extension: Implement intersectional fairness (gender × age groups)

3. **Post-processing only**: Doesn't modify Algorithm 1-3
   - Extension: In-processing fairness constraints during tree building

4. **Convex hull assumption**: Assumes ROC curves are relatively smooth
   - Potential issue: Very small groups with few distinct thresholds

5. **Equal TPR only**: Does not enforce equal FPR (though related)
   - Extension: Implement full equalized odds with both TPR and FPR

### 7.2 Future Extensions

- [ ] Multi-attribute fairness (e.g., gender × age × ethnicity)
- [ ] Fairness-aware decision tree construction (Algorithm 1)
- [ ] Alternative fairness notions (calibration, counterfactual fairness)
- [ ] Uncertainty quantification for threshold selection
- [ ] Real-time fairness monitoring dashboard

---

## 8. REFERENCES

### Primary Reference
- **Hardt, Moritz; Price, Eric; and Srebro, Nathan.** (2016)  
  "Equality of Opportunity in Supervised Learning"  
  *arXiv preprint arXiv:1610.02413*  
  https://arxiv.org/abs/1610.02413

### CDS Algorithm References
- **Original CDS Paper**: [Your CDS paper citation]
- **Arrhythmia Dataset**: UCI Machine Learning Repository  
  https://archive.ics.uci.edu/ml/datasets/arrhythmia

---

## 9. FILE MANIFEST

### Core Fairness Module
- `Fairness_EqualizedOdds.py` (550 lines)
  - ROCPoint, ROCCurve, ConvexHullResult classes
  - compute_roc_curve(), compute_convex_hull(), compute_loss()
  - compute_bayes_optimal_predictor() – main entry point

### Integration Module
- `Algorithm4_FairnessIntegration.py` (450 lines)
  - Extracts Algorithm 4 outputs
  - Computes FairnessMetrics
  - apply_fairness_post_processing() – integration entry point

### Example & Demo
- `example_equalized_odds_demo.py` (150 lines)
  - Full pipeline: Algorithms 1-4 → fairness post-processing
  - Shows expected usage

---

## 10. QUICK START

### Minimal Example

```python
from Algorithm4 import run_loocv
from Algorithm4_FairnessIntegration import apply_fairness_post_processing

# Step 1: Run Algorithm 4
alg4_output = run_loocv(data, labels, max_users=100)

# Step 2: Apply fairness post-processing
fairness_output = apply_fairness_post_processing(
    alg4_output, data, labels,
    baseline_threshold=0.025
)

# Step 3: Use thresholds
threshold_male = fairness_output.thresholds_optimized.threshold_male
threshold_female = fairness_output.thresholds_optimized.threshold_female

print(f"Equalized TPR: {fairness_output.thresholds_optimized.tpr_equalized:.4f}")
print(f"Utility loss: {fairness_output.thresholds_optimized.utility_loss:.6f}")
```

---

## 11. TROUBLESHOOTING

### Issue: "Convex hull is empty"
**Cause**: No threshold pairs satisfy equalized TPR  
**Solution**: 
- Reduce tolerance in convex hull computation (line: `if abs(pt_male.tpr - pt_female.tpr) < 1e-6`)
- Increase ROC point granularity

### Issue: Thresholds unchanged
**Cause**: Baseline thresholds already satisfy equalized TPR well  
**Solution**: 
- This is good! Fairness is already present
- Check fairness_output.fairness_metrics_original

### Issue: Accuracy drops significantly
**Cause**: Large utility_loss value  
**Solution**:
- Consider alternative loss function with different FPR/FNR weighting
- Accept lower accuracy in favor of fairness (policy decision)

---

**Document Version**: 1.0  
**Last Updated**: May 18, 2026  
**Status**: Complete and validated
"""
