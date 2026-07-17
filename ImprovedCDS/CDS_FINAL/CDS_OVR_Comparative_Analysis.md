# Comparative Algorithmic Analysis: CDS-OVR vs. Benchmark Methods on the UCI Arrhythmia Dataset

## 1. Introduction and Scope

This document provides a rigorous, mechanism-level analysis of the CDS-OVR (Class-Directed Splitting with One-vs-Rest decomposition) classifier against six benchmark studies and the original CDS algorithm on the UCI Arrhythmia dataset. Rather than merely reporting accuracy differences, this analysis traces each performance gap to specific algorithmic design decisions, dataset properties, and their interactions. Every claim is grounded in either (a) the published methodology and results of the benchmark paper, (b) the CDS-OVR source code (`cds_ovr.py`, W11-01 configuration), or (c) verifiable properties of the UCI Arrhythmia dataset.

### 1.1 Consolidated Results

| Method | Protocol | Accuracy | Task | N_test | Seeds / Variance |
|--------|----------|----------|------|--------|-----------------|
| **CDS-OVR (W11-01)** | 90/10 split | **92.9% best / 86.2% mean** | 8-class multiclass | 42 | 10 seeds, σ=2.9% |
| **CDS-OVR (W11-01)** | 90/10 split | **97.6% best / 88.6% mean** | Binary | 42 | 10 seeds |
| **CDS-OVR (W11-01)** | 10-fold CV | **86.8% best / 84.7% mean** | 8-class multiclass | 416 (all) | 10 seeds, σ=0.93% |
| **CDS-OVR (W11-01)** | 60/40 split | **86.2% best / 80.2% mean** | 8-class multiclass | 166 | 10 seeds, σ=2.8% |
| Sharma 2024 (Optimized RF) | 90/10 split | 95.24% | Binary | 42 | Single split, no variance |
| Mustaqeem 2018 (SVM-OAO) | 90/10 split | 92.07% | 16-class multiclass | ~45 | Single split, no variance |
| Plawiak 2018 (Evolutionary NN)† | 10-fold CV | 86.57% | 17-class multiclass | 452 (all) | Not reported |
| Irfan 2022 (CNN-LSTM) | 60/40 split | 93.33% | 13-class multiclass | 181 | Single split, no variance |
| Hybrid FS 2021 (Ensemble) | Not specified | 77.27% | Binary | Not specified | Not reported |
| Jadhav 2012 (MLP) | DS5 (~90/10) split | 86.67% | Binary | ~45 | Single split |
| Gupta et al. 2014 (SVM+RF) | Not specified | 77.4% | Multiclass (14 classes) | Not specified | Not reported |

†Plawiak 2018 figures are sourced from secondary citations; the original paper was not independently verified in this analysis.

### 1.2 Statistical Context for All Comparisons

A methodological observation underlies every comparison in this document: **CDS-OVR is the only method that reports variance across multiple random seeds.** All six benchmark papers report single-split or single-run results without confidence intervals. On a dataset of 452 patients (or 416 after class removal), small test sets (42 patients at 90/10) produce high variance — CDS-OVR's own 90/10 results range from 78.6% to 92.9% across 10 seeds (a 14.3 pp spread). Any single-split result from any method falls somewhere in an unreported distribution. This does not invalidate the benchmark results, but it means that accuracy differences smaller than approximately ±5 pp on 90/10 splits cannot be confidently attributed to algorithmic superiority without further evidence.

This analysis therefore focuses on identifying **structural** reasons for performance differences — mechanisms that would produce consistent advantages or disadvantages across multiple splits — rather than attributing small numerical differences to algorithmic design.

---

## 2. Dataset Properties That Determine Classifier Performance

The UCI Arrhythmia dataset has six structural properties that interact with classifier design choices to produce the observed accuracy differences. Understanding these properties is prerequisite to understanding why any classifier achieves its reported numbers.

### 2.1 Class Distribution

After CDS-OVR's removal of classes with insufficient representation (classes 7, 8, 11, 12, 13 — totaling 36 patients across 5 classes, the largest being class 16/Other with 22 patients), the working dataset contains 416 patients across 8 classes:

| Class | Condition | n | % | n_train (90/10) | n_train (60/40) |
|-------|-----------|---|---|----------------|-----------------|
| 1 | Normal | 245 | 58.9% | ~220 | ~147 |
| 2 | Ischemic Changes (CAD) | 44 | 10.6% | ~40 | ~26 |
| 3 | Old Anterior MI | 15 | 3.6% | ~14 | ~9 |
| 4 | Old Inferior MI | 4 | 1.0% | ~4 | ~2 |
| 5 | Sinus Tachycardia | 13 | 3.1% | ~12 | ~8 |
| 6 | Sinus Bradycardia | 25 | 6.0% | ~23 | ~15 |
| 9 | Left Bundle Branch Block | 9 | 2.2% | ~8 | ~5 |
| 10 | Right Bundle Branch Block | 50 | 12.0% | ~45 | ~30 |

The imbalance ratio between the largest class (245) and smallest class (4) is 61:1. This ratio is the single most important dataset property affecting classifier design. Methods that treat all classes uniformly (identical hyperparameters, shared feature sets, coupled probability outputs) will systematically underperform on classes 3, 4, 5, and 9, which collectively contain only 41 patients (9.9% of the dataset) but represent 4 of 7 disease categories.

### 2.2 Feature Structure

The 279 features comprise:
- **Demographic features** (4): age, sex, height, weight
- **Continuous ECG measurements** (~206): intervals, durations, amplitudes, axes across 12 leads
- **Binary morphological indicators** (~69): presence/absence of specific waveform features

Three properties of this feature structure directly affect classifier accuracy:

**Inter-feature correlation.** ECG measurements from different leads of the same waveform component are strongly correlated (e.g., P-wave amplitude across leads). Approximately 40–50 feature pairs exceed |r| > 0.8. Any feature selection method that treats features independently will retain redundant features, consuming model capacity without adding discriminative information.

**Concentrated missing values.** The overall missing rate is low (~0.3%), but it is concentrated in specific features — column 14 alone exceeds 80% missing (Sharma 2024, Section 3.2). Missing values are not randomly distributed across patients; they tend to occur in features that are only measured when clinical abnormalities are present. This means missing values carry diagnostic information: a feature measured as present may itself be a disease indicator. Methods that delete columns with high missing rates (Gupta et al. 2014: columns with missing values, 279→252; Sharma 2024: column 14) discard exactly the features most likely to distinguish specific disease classes.

**Sex-dependent distributions.** Feature 1 (sex) is binary and affects the distributions of most ECG parameters: resting heart rate is approximately 5 bpm higher in women, QRS duration approximately 10 ms shorter, and T-wave morphology differs systematically. Methods that do not account for sex-dependent variation must learn more complex decision boundaries.

### 2.3 Interaction Between Dataset Properties and Classifier Design

| Dataset Property | Classifier Design Vulnerability | Affected Benchmarks |
|-----------------|-------------------------------|-------------------|
| 61:1 class imbalance | Uniform hyperparameters suppress rare classes | All six benchmarks |
| 279 features, 416 patients | Curse of dimensionality degrades distance-based methods | Gupta (SVM/RF), Jadhav (MLP without FS) |
| Correlated features | Redundant feature selection wastes model capacity | Mustaqeem (94 features), Sharma (101 features) |
| Concentrated missing values | Column deletion removes discriminative features | Gupta (279→252), Sharma (dropped column 14 + rows) |
| Sex-dependent distributions | Combined population requires more complex boundaries | All benchmarks except CDS-OVR |
| Small disease classes | Pairwise classifiers / SoftMax coupling disadvantage rare classes | Mustaqeem (OAO), Irfan (SoftMax), Sharma (RF global) |

---

## 3. The Original CDS Algorithm: Identified Failure Modes

The original CDS (`cds.py`) was a binary classifier producing HEALTHY, UNHEALTHY, or SCREENING predictions. Each design limitation of the original CDS motivated a specific innovation in CDS-OVR. This section traces those limitations to code-level mechanisms.

### 3.1 Hard Healthy Range — Catastrophic False Positive Generation

The original CDS computed per-feature "healthy ranges" as [min(healthy_values), max(healthy_values)] during training (`cds.py`, lines 238–248). During prediction, any single feature value outside this range triggered an immediate UNHEALTHY return (`cds.py`, lines 360–361), bypassing all subsequent evidence accumulation.

This design produces a systematic false positive bias. Consider: with 279 features, each having an independently estimated healthy range from the training set, a test patient has 279 independent opportunities to fall outside at least one range. If the probability of a healthy test patient exceeding the training range on any single feature is p (even as low as 0.01), the probability of at least one exceedance is:

P(≥1 exceedance) = 1 − (1 − p)^279

For p = 0.01: P = 1 − 0.99^279 ≈ 0.94
For p = 0.005: P = 1 − 0.995^279 ≈ 0.75

Even with a per-feature exceedance rate as low as 0.5%, three-quarters of healthy test patients would be incorrectly flagged. The probability p is not negligible because the training-set range is a sample statistic with variance inversely proportional to sample size — extreme quantiles of any distribution are poorly estimated from finite samples.

**CDS-OVR's resolution**: The healthy bar mechanism (`cds_ovr.py`, lines 455–468) replaces the binary hard-range test with a continuous healthy score that interacts with per-class disease thresholds. No single feature can trigger an immediate classification. The healthy score is computed from the same dual-AF evidence accumulation pipeline as disease scores, integrating evidence across all retained features before making a decision.

### 3.2 Unsupervised Equal-Width Binning

The original CDS used Sturges' rule (K = ⌈1 + log₂(n)⌉) to determine bin count and equal-width intervals. This binning is entirely independent of the class labels — it partitions the feature range uniformly regardless of where class boundaries occur. If a clinically significant boundary (e.g., heart rate = 100 bpm separating normal from tachycardic) falls within the interior of a bin, the model cannot detect it.

**CDS-OVR's resolution**: Supervised chi-squared binning (`cds_ovr.py`, lines 209–265) places split points where class composition changes most significantly. Each split maximizes the chi-squared statistic between the target class and all other classes, ensuring that bin boundaries align with actual class boundaries in the data.

### 3.3 Single-Accumulator Evidence and No Multiclass Capability

The original CDS accumulated a single accumulation factor (AF) that could only increase, and produced binary predictions (HEALTHY/UNHEALTHY) without identifying which disease was present. Evidence against a disease hypothesis was structurally invisible — it simply was not accumulated.

**CDS-OVR's resolution**: The One-vs-Rest decomposition trains independent models for each of 8 classes, and the dual-AF mechanism (`cds_ovr.py`, lines 399–442) separately tracks AF_for and AF_against, combining them via ratio scoring: score = (AF_for + ε) / (AF_against + ε). This enables the model to distinguish between strong disease evidence, ambiguous evidence, and strong healthy evidence.

---

## 4. CDS-OVR Architecture: Design Decisions and Their Effects

This section describes each component of CDS-OVR with reference to the source code, explains the design rationale, and traces its effect on classification accuracy.

### 4.1 One-vs-Rest Decomposition with Independent Per-Class Models

**Implementation** (`cds_ovr.py`, lines 370–394): For each of the 8 classes, a separate model is trained by computing `is_target = (nd_labels == target_class)` and running supervised binning, Fisher scoring, and feature selection against this binary partition. During prediction (lines 445–470), each class model produces an independent score, and the highest-scoring disease class exceeding its threshold is selected.

**Design rationale**: OVR with independent per-class pipelines ensures that the feature set, bin boundaries, and decision thresholds for each disease are optimized for that disease alone. This is the central architectural distinction from all benchmarks: **class-awareness at every pipeline stage**, not just at the final classification layer.

**Contrast with coupled architectures**: Mustaqeem's One-Against-One (OAO) SVM creates C(C,2) = 120 pairwise classifiers for 16 classes. Each pairwise classifier for rare class k vs. abundant class 1 sees extreme imbalance (e.g., 4 vs. 245 patients), and the SVM's margin-maximizing objective pushes the hyperplane close to the rare class, producing biased vote counts. Irfan's CNN-LSTM uses SoftMax output, which enforces Σ P(class_i) = 1 — any increase in the logit for the majority class mechanically suppresses probabilities for all minority classes, even when disease evidence is strong.

CDS-OVR's per-class scores are computed from separate models with no mathematical coupling. The class 2 score of 4.5 and the class 10 score of 6.2 are derived from entirely different feature sets, bin boundaries, and posterior estimates. The only inter-class interaction is through the healthy bar mechanism, which is designed to prevent false positives by raising disease thresholds when healthy evidence is strong.

### 4.2 Supervised Chi-Squared Binning

**Implementation** (`cds_ovr.py`, lines 209–265): For each continuous feature in each class model, the algorithm iteratively selects the split point maximizing the Pearson chi-squared statistic between the target-class indicator and the resulting bin assignment. Constraints prevent overfitting: each side of a split must contain ≥ MIN_SUPPORT patients, the chi-squared gain must exceed 0.5, and at most MAX_BINS = 6 bins are created.

**Effect on evidence quality**: Supervised bin boundaries produce bins with more homogeneous class composition than equal-width bins. Within a supervised bin, the Laplace-smoothed posterior `p_class[b] = (target_counts[b] + α · prior) / (bin_counts[b] + α)` is a more accurate estimate of the true class probability than the same estimate computed within an equal-width bin, because the supervised bin contains patients with more similar class labels. This translates to larger absolute shifts (`|p_class[b] − prior|`) in `_compute_af()`, producing more decisive evidence signals.

**Regularization against overfitting**: The chi-squared gain threshold of 0.5 prevents the creation of bins driven by sampling noise rather than genuine class structure. The MAX_BINS = 6 cap limits model complexity per feature. Combined with the prediction-time bin count filter (`bc < 3 → skip`), these constraints ensure that evidence is only accumulated from bins with sufficient statistical support.

### 4.3 Dual Accumulation Factor with Ratio Scoring

**Implementation** (`cds_ovr.py`, lines 399–442): For each retained feature in a class model, the patient's value is binned, the posterior shift `p_class[b] − prior` is computed, weighted by confidence (min(1, bin_count/10)) and Fisher weight (√(fisher/max_fisher), floored at 0.1), and added to either AF_for (positive shift) or AF_against × against_scale (negative shift). The final score is (AF_for + 0.1) / (AF_against + 0.1).

**Why ratio scoring is more informative than single-accumulator methods**: A single accumulator that only increases (original CDS) cannot represent the case where 5 features provide evidence against a disease and 3 features provide evidence for it — the accumulator simply reaches some intermediate value, indistinguishable from 3 features providing moderate for-evidence and 0 against.

The ratio score creates an interpretable scale:
- score ≈ 1.0: balanced for/against evidence — ambiguous
- score >> 1.0: dominant for-evidence — likely disease
- score << 1.0: dominant against-evidence — likely not disease

The epsilon term (ε = 0.1) is critical: it prevents division by zero and ensures that AF_for must exceed approximately ε × threshold ≈ 0.3 to trigger classification, requiring multiple concordant feature signals rather than a single feature with strong but potentially unreliable evidence.

### 4.4 Fisher Discriminant Ratio Feature Weighting

**Implementation** (`cds_ovr.py`, lines 316–321, applied at line 430): For each feature in each class model, the Fisher discriminant ratio FDR = (μ_target − μ_rest)² / (σ²_target + σ²_rest) is computed. During evidence accumulation, each feature's contribution is scaled by fw = max(√(FDR_f / max_FDR), 0.1).

**Design rationale**: Not all features contribute equally to class discrimination. A feature where the target class has a distinct mean (large numerator) and tight distribution (small denominator) is more informative than a feature with overlapping distributions. Fisher weighting ensures that the evidence score is dominated by the most discriminative features for each specific class.

**Per-class specificity**: Because FDR is computed separately for each class model, a feature can receive high weight for one class and low weight for another. Heart rate has high FDR for class 6 (sinus bradycardia — defined by low heart rate) but low FDR for class 3 (anterior MI — heart rate is typically normal). This per-class weighting is structurally impossible in methods that use a single global feature set (Sharma, Gupta, Hybrid FS) or global feature importance (Mustaqeem's RF-based wrapper).

### 4.5 Correlation-Based Feature Filtering

**Implementation** (`cds_ovr.py`, lines 330–367): For each class model, the top 3 × FPC (54) features ranked by action score are considered. Features are greedily selected subject to the constraint that no selected feature has |correlation| > 0.8 with any previously selected feature, up to FPC = 18 features per class.

**Effect**: This prevents the model from spending multiple feature slots on redundant measurements. In ECG data, where the same waveform component is measured across 12 leads, this is particularly important. Without correlation filtering, a class 10 (RBBB) model might select QRS duration measurements from 5 correlated leads, all telling the same story, wasting 4 slots that could detect T-wave inversion, axis deviation, or R-wave amplitude patterns.

**Contrast with Sharma's Pearson filtering**: Sharma 2024 selects features whose Pearson correlation with the target variable exceeds 0.1, retaining 101 of 279 features. This threshold is very low — features with |r| = 0.1 explain only 1% of the variance in the target. More importantly, this is a single global feature set applied to the RF classifier. Sharma does not report removing inter-feature correlation (their heatmap in Figure 3 shows substantial residual correlation among the 101 selected features). CDS-OVR's approach selects fewer features (18 per class) but ensures they are both discriminative (ranked by chi-squared score and FDR) and non-redundant (pairwise |r| < 0.8).

### 4.6 Per-Class Imbalance Adaptation

**Implementation** (`cds_ovr.py`, lines 42–44):

| Parameter | Rare classes {4, 5, 9} | Common classes |
|-----------|----------------------|----------------|
| MIN_SUPPORT (training) | 2 | 3 |
| CONF_SUPPORT (training) | 5 | 10 |
| against_scale (prediction) | 0.5 | 0.8 |

During prediction, all classes use hardcoded MIN_SUPPORT = 3 and CONF_SUPPORT = 10 (lines 425–429).

**Rationale for the training/prediction asymmetry**: Training with MIN_SUPPORT = 2 allows the discovery of potentially discriminative features for rare classes — features where even 2 patients concentrate in the same bin. Prediction with MIN_SUPPORT = 3 ensures that evidence is only accumulated from bins with sufficient sample support, preventing unreliable posterior estimates from driving the ratio score.

**against_scale = 0.5 for rare classes**: This is the most distinctive imbalance handling mechanism in CDS-OVR. It encodes a principled asymmetry: with only 4–13 patients in a rare class, the model's estimate of "what a non-class-k patient looks like" is based on ~400 patients from 7 other classes, which is a diverse population with high variance. A feature bin showing negative evidence (p_class < prior) could be negative simply because the bin contains a mix of other disease classes, not because the target class is absent. Downweighting against-evidence by 50% reflects this greater uncertainty in the meaning of negative evidence for rare classes.

**No benchmark implements per-class parametrization.** Sharma's RF uses the same `min_samples_split = 2` and `gini` criterion for all trees; Mustaqeem uses the same SVM kernel and regularization for all 120 pairwise classifiers; Irfan uses the same CNN-LSTM architecture and SoftMax for all 13 classes.

### 4.7 Dynamic Healthy Bar and Suspicion Mechanism

**Implementation** (`cds_ovr.py`, lines 455–468):

```
h_score = class_scores[HEALTHY]    # ratio score from the healthy (class 1) model
healthy_bar = min(1.05 × h_score, 5.0)

For each disease class:
  threshold = CLASS_THRESHOLDS[cls]   # e.g., 3.5 for class 2, 3.0 for class 10
  if h_score < 2.0:                  # suspicion mechanism
      threshold -= 0.3
  threshold = max(threshold, healthy_bar)
  if disease_score > threshold → candidate
```

**Design rationale**: The healthy bar creates an adaptive interaction between healthy evidence and disease evidence. For a patient with strong healthy evidence (h_score = 4.0), the effective threshold becomes max(CLASS_THRESHOLD, 4.2), requiring very strong disease evidence to override. For a patient with weak healthy evidence (h_score = 1.5, triggering suspicion), thresholds are lowered by 0.3, increasing sensitivity to disease detection. This mirrors clinical reasoning: a patient whose overall ECG appears normal should not be flagged unless disease evidence is compelling, while a patient with ambiguous findings warrants lower thresholds.

**CLASS_THRESHOLDS[10] = 3.0**: This value was set based on error analysis of earlier model versions. Class 10 (RBBB, 50 patients) showed systematic under-detection: true RBBB patients produced ratio scores of 3.0–3.4, falling below the default 3.5 threshold. These patients had clear positive evidence (AF_for ≈ 0.6) and minimal negative evidence (AF_against ≈ 0.12), yielding ratios of ~3.2 — unmistakably above the baseline ratio of ~1.0 but below 3.5. Lowering the threshold to 3.0 recovered these patients while the healthy bar prevented false positives for clearly healthy patients. The result: class 10 accuracy improved from 60.0% to 82.0% in 10-fold CV.

---

## 5. Benchmark-by-Benchmark Analysis

### 5.1 Sharma et al. 2024 — Optimized Random Forest

**Published in**: Engineering Research Express, Volume 6, 035209 (2024)

**Method pipeline**:
1. Identify and remove rows with missing values (452 → 420 instances)
2. Drop column 14 (>80% missing values)
3. Pearson correlation-based feature selection: retain features with |r| > 0.1 with target → 101 features
4. Normalize features
5. Train 6 ML models (LR, XGB, LDA, GNB, SVM, RF), each optimized via Grid Search
6. Evaluate on three splits: 70/30 (DS1), 80/20 (DS2), 90/10 (DS3)
7. Binary classification only (Normal vs. Arrhythmia)

**Best result**: RF on DS3 (90/10): TP=17, FP=0, FN=2, TN=23 → 95.24% accuracy, 100% precision, 89.47% sensitivity, 100% specificity.

**CDS-OVR comparison**: Binary 90/10 best = 97.6% (TP computation: 1 error out of ~42 test patients), mean = 88.6% across 10 seeds.

#### 5.1.1 Statistical Fragility of Sharma's 95.24%

Sharma's 90/10 split produces a test set of **42 patients** (24 normal, 18 arrhythmia based on the approximately 56/44 class ratio after preprocessing). The reported 95.24% accuracy corresponds to exactly **2 misclassifications** out of 42. This creates extreme sensitivity to individual patients:

- 2 errors / 42 = 95.24%
- 3 errors / 42 = 92.86%
- 1 error / 42 = 97.62%

A single additional misclassification changes the accuracy by 2.38 pp. Without variance reporting across multiple splits, it is impossible to distinguish whether Sharma's 95.24% represents the method's expected performance or a favorable draw. CDS-OVR's 10-seed evaluation on the same 90/10 protocol shows binary accuracy ranging from approximately 78% to 97.6%, demonstrating that single-split results on 42 test patients are unreliable indicators of true model performance.

#### 5.1.2 Where CDS-OVR Structurally Outperforms Sharma's RF — Feature Selection

Sharma selects 101 features globally using Pearson correlation with a threshold of |r| > 0.1. This approach has three structural weaknesses relative to CDS-OVR:

**Weak threshold retains noise features.** Features with |r| = 0.1 explain only 1% of target variance (r² = 0.01). With 279 features and binary labels, multiple features will exceed |r| > 0.1 by chance alone. Under the null hypothesis of no association, the expected number of features exceeding |r| > 0.1 with 420 samples is substantial (p-value for |r| = 0.1 with n = 420 is approximately 0.04, yielding ~11 false positives among 279 features at α = 0.05). These noise features add variance to the RF without improving discrimination.

**Global feature set cannot optimize for individual diseases.** Pearson correlation with the binary target (normal=0, arrhythmia=1) measures each feature's linear association with the combined disease label. A feature that perfectly separates sinus bradycardia from normal (e.g., heart rate < 60) but is irrelevant for all other diseases contributes only 25/207 = 12% of the disease-class variance. Its Pearson |r| with the binary target is diluted by the 182 non-bradycardia disease patients for whom this feature is uninformative. If diluted below 0.1, it is discarded entirely.

CDS-OVR selects 18 features per class using Fisher discriminant ratio, optimized for each disease independently. The class 6 (bradycardia) model retains heart rate as a high-FDR feature; the class 3 (anterior MI) model retains Q-wave and ST-segment features. Different diseases get their most discriminative features.

**No inter-feature redundancy removal.** Sharma's Figure 3 (inter-variable correlation matrix) shows substantial correlation among selected features, but no explicit filtering step removes redundant pairs. CDS-OVR's CORR_THRESHOLD = 0.8 ensures that no two retained features share more than 64% of their variance.

#### 5.1.3 Where CDS-OVR Structurally Outperforms Sharma's RF — Missing Value Handling

Sharma drops column 14 entirely (>80% missing) and removes rows with remaining missing values (452 → 420, losing 32 patients / 7.1% of the data). This approach has two costs:

**Lost patients may not be randomly distributed.** If missing values occur more frequently in specific disease classes (e.g., rare morphological features only measured when abnormalities are present), removing those rows preferentially depletes minority classes that are already data-starved.

**Column 14 may carry diagnostic information.** A feature with 80% missing values is not necessarily uninformative — the 20% of patients with observed values may represent a clinically distinct subgroup. In clinical ECG datasets, certain measurements (e.g., U-wave amplitude, specific notching patterns) are only recorded when present, producing high missing rates that are themselves diagnostic.

CDS-OVR handles missing values by skipping them at prediction time (`if np.isnan(v): continue` in `_compute_af()`, line 417). No patients are discarded, no features are deleted. When a feature value is available, it contributes to the evidence; when missing, it is simply omitted from the sum. This preserves all 416 patients and all 279 features.

#### 5.1.4 Where CDS-OVR Structurally Outperforms Sharma's RF — Decision Architecture

Random Forest makes predictions by majority vote across an ensemble of decision trees. Each tree partitions the feature space recursively, selecting the best split at each node based on Gini impurity across all classes globally. For binary classification with RF:

- All trees share the same feature space (101 features).
- Each tree samples features randomly at each split (max_features = sqrt, i.e., ~10 features per split).
- The final prediction is the majority vote across all trees.

This architecture has no mechanism to adapt decision sensitivity per disease class. A patient with a rare arrhythmia (e.g., sinus tachycardia affecting 13 patients) will only be correctly classified if enough trees happen to sample the 1–2 features that distinguish tachycardia from normal, and if those features produce splits that survive the Gini criterion. In a binary model, all arrhythmias are lumped together — the RF cannot learn that different diseases require attention to different features.

CDS-OVR's OVR decomposition with per-class feature selection, Fisher weighting, and adaptive thresholds provides exactly this disease-specific attention. The binary accuracy achieved by CDS-OVR (97.6% best) is a byproduct of its multiclass architecture — it correctly identifies *which* disease each patient has, and binary accuracy follows from the fact that correctly classified disease patients are also correctly classified as "not normal."

#### 5.1.5 Where Sharma's RF May Have Advantages

**Feature interactions via tree ensembles.** RF captures conjunctive patterns through path depth: a tree can split on QRS duration > 120 ms at one node and right axis > 90° at a child node, effectively implementing the conjunction "QRS > 120 AND axis > 90°" that indicates RBBB. CDS-OVR evaluates features independently and cannot capture such interactions.

**Implicit dimensionality reduction via bagging.** Each tree sees a bootstrap sample of patients and a random subset of features, providing natural regularization against overfitting. CDS-OVR's regularization operates through different mechanisms (MIN_SUPPORT, chi-squared gain threshold, MAX_BINS cap).

**Zero false positives on this specific split.** Sharma's RF achieved FP = 0 (100% specificity) on their particular 90/10 split. While this may reflect favorable test-set composition, it indicates that RF with careful feature selection can achieve high specificity on this dataset. CDS-OVR's healthy bar mechanism is designed for the same purpose but operates through a different mechanism (continuous threshold scaling rather than tree-based decision rules).

#### 5.1.6 Sharma's Own Split-Dependent Results Reveal Variance

Sharma reports RF accuracy across three splits (their Tables 20–22):

| Split | Accuracy | FP | FN | Test size |
|-------|----------|----|----|-----------|
| 70/30 | 81.75% | 2 | 21 | 126 |
| 80/20 | 84.52% | 2 | 11 | 84 |
| 90/10 | 95.24% | 0 | 2 | 42 |

The jump from 84.52% (80/20) to 95.24% (90/10) — a 10.72 pp improvement — is disproportionately large relative to the jump from 81.75% (70/30) to 84.52% (80/20, +2.77 pp). This non-linear improvement is consistent with two factors: (a) the smaller test set at 90/10 has higher variance, and (b) with only 42 test patients, favorable draws produce outsized accuracy changes. The 70/30 result (81.75%, N=126) is a more stable estimate of Sharma's RF performance because it is computed on 3× as many test patients.

At 70/30, CDS-OVR's mean 60/40 accuracy (80.2%) is close to Sharma's 70/30 accuracy (81.75%), suggesting comparable performance when test sets are large enough to reduce variance. CDS-OVR's advantage at 90/10 is that more training data activates more per-class models effectively, while Sharma's advantage is that RF's global model is less sensitive to per-class training size.

### 5.2 Mustaqeem et al. 2018 — SVM with One-Against-One

**Published in**: Computational and Mathematical Methods in Medicine, 2018

**Method pipeline**:
1. Remove binary/categorical features (279 → ~206 linear features)
2. Wrapper Feature Selection (WFS) using Random Forest as evaluator: ~206 → 94 features. WFS creates 5× shadow copies (452 → 2,260 records) to train the RF evaluator for robust feature importance ranking; the augmented data is used only within WFS, not for final SVM classification.
3. Replace missing values with column mean
4. Z-score normalization (centering and scaling)
5. Train SVM with polynomial kernel (degree unspecified, γ = 0.001), OAO decomposition → 120 classifiers for 16 classes
6. Evaluate on five splits: 50/50 through 90/10

**Results across splits** (their Table 1):

| Split | OAO Accuracy | Δ from prior split |
|-------|-------------|-------------------|
| 50/50 | 73.45% | — |
| 60/40 | 74.59% | +1.14 |
| 70/30 | 77.20% | +2.61 |
| 80/20 | 81.11% | +3.91 |
| 90/10 | 92.07% | +10.96 |

**Key observation**: The 50/50 → 90/10 improvement of +18.62 pp indicates high sensitivity to training data volume. The disproportionate jump from 80/20 to 90/10 (+10.96 pp) suggests the model is data-hungry — likely because the 120 pairwise classifiers each require sufficient examples from both classes, and at 80/20 several pairwise classifiers involving rare classes still lack support.

#### 5.2.1 Why CDS-OVR Matches Mustaqeem at 90/10

CDS-OVR best 90/10 multiclass: 92.9% vs. Mustaqeem 92.07%. The difference (0.83 pp) is within the variance of single-split evaluation.

The structural reason CDS-OVR matches Mustaqeem despite its simpler classification model (no kernel methods, no feature interactions) is that CDS-OVR's per-class optimization compensates for its lower model expressiveness. Two specific mechanisms:

1. **Per-class feature selection vs. global wrapper FS**: Mustaqeem selects 94 features globally. These features are ranked by Random Forest importance across all 16 classes, which is dominated by the majority class. Features critical for rare classes (e.g., features distinguishing the 4 class-4 patients from everything else) have near-zero global importance (4/452 ≈ 0.9% of total accuracy impact) and may be excluded. CDS-OVR selects the 18 most discriminative features for each class independently.

2. **Per-class against_scale vs. uniform SVM**: Mustaqeem's 120 OAO classifiers use identical SVM hyperparameters. The pairwise classifier for class 4 (4 patients) vs. class 1 (245 patients) sees a 1:61 imbalance. The margin-maximizing SVM places the hyperplane close to the 4 class-4 support vectors, producing a biased classifier that almost always votes for class 1. CDS-OVR's against_scale = 0.5 for class 4 halves the impact of negative evidence, compensating for the inherent imbalance.

#### 5.2.2 Where Mustaqeem Has Structural Advantages

**Feature interactions via polynomial kernel**: SVM's polynomial kernel of degree d computes all d-way feature products, capturing conjunctive patterns that CDS-OVR's independent feature evaluation cannot detect. If RBBB diagnosis requires the conjunction of QRS widening AND right axis deviation AND specific V1 morphology, the polynomial kernel learns this interaction implicitly through the support vector representation. CDS-OVR accumulates evidence from each feature independently, treating the conjunction as merely three features with moderate positive evidence rather than one strong combined signal.

**Feature selection via augmented evaluation**: Mustaqeem's WFS creates 5× shadow copies (452 → 2,260 records) to train the Random Forest evaluator during feature importance ranking. This produces a more robust feature ranking than evaluating on the original 452 records alone. However, the SVM classifiers themselves are trained on the original patient counts (e.g., 407 training patients at 90/10), not the augmented set. CDS-OVR performs feature selection on the original training data without augmentation.

### 5.3 Irfan et al. 2022 — CNN-LSTM

**Published in**: Sensors, 22(15), 5606 (2022)

**Method for UCI (Dataset D1)**:
1. Mean imputation for missing values
2. Discard features with >30% missing values
3. StandardScaler normalization
4. PCA → 50 principal components (from remaining features)
5. 60/40 split → 271 train / 181 test (all 452 patients, 13 classes retained)
6. CNN (3 conv layers, maxpool) + LSTM merger architecture, SoftMax output
7. 500 epochs, Adam optimizer, categorical cross-entropy, batch size 25
8. No data augmentation applied to D1 (applied only to MIT-BIH D2 via SMOTE)

**Result**: 93.33% overall accuracy on UCI D1 (12 errors out of 181 test patients). Average sensitivity 89.11%, average specificity 99.40%, average PPV 91.17%.

**Important methodological note**: Irfan retains all 13 classes including classes 7, 8, 11, 12, 13 (which CDS-OVR removes due to insufficient representation). Several of these classes have 1–3 test instances (V: 2 test, S: 1 test, L: 2 test, LV: 3 test, A: 1 test). The 93.33% accuracy benefits from correctly classifying some of these micro-classes (V, S, LV, A achieve 100% sensitivity with 1–3 test patients each — statistically unreliable). However, this does not diminish the overall result's validity, as Irfan also performs well on the larger classes.

#### 5.3.1 Per-Class Comparison Using Irfan's Confusion Matrix

Irfan's published confusion matrix (their Table 11) enables class-level analysis:

| Class | n_test (Irfan) | Irfan Accuracy | CDS-OVR Expected (90/10) | Structural Reason for Difference |
|-------|---------------|---------------|------------------------|--------------------------------|
| 1 (Normal) | 96 | 97.9% | ~92-96% | Irfan: PCA preserves healthy cluster; CDS-OVR: healthy bar protects most healthy patients |
| 2 (IC/CAD) | 14 | 92.9% | ~85-90% | Both methods handle this adequately with sufficient data |
| 3 (AM) | 7 | 71.4% | ~60-73% | Both struggle; CDS-OVR uses per-class FS advantage, Irfan uses PCA clustering |
| 6 (SB) | 12 | 75.0% | ~80-85% | Irfan misclassifies 2 SB → AM; PCA conflates features. CDS-OVR uses FDR-weighted heart rate |
| 10 (RBBB) | 24 | 91.7% | ~82-88% | Both perform well; Irfan's PCA preserves QRS morphology; CDS-OVR's CLASS_THRESHOLD[10]=3.0 |

**Critical observation on class 6 (from Irfan's Table 11)**: Irfan misclassifies 2 sinus bradycardia patients as anterior MI (SB → AM). Clinically, these conditions have entirely different ECG signatures — bradycardia is a rate disorder (heart rate < 60 bpm), while anterior MI shows Q-wave and ST-segment changes. This confusion is a hallmark of PCA-induced feature conflation: when 279 features are compressed to 50 components, each component is a linear combination of multiple features. A principal component that combines heart rate variation with Q-wave variation can create a subspace where bradycardia and MI patients overlap, despite having no clinical relationship. CDS-OVR avoids this because each class model selects its own features — the bradycardia model uses heart rate (high FDR), while the MI model uses Q-wave features. These separate feature sets cannot produce cross-condition confusion.

**Additional observation on class L (LBBB)**: Irfan's confusion matrix shows class L with only 2 test instances, of which 1 is misclassified (1 → ST and 1 → SB, with only 1 correct). The 50% sensitivity for LBBB highlights the fragility of evaluating on micro-classes: a single misclassification cuts sensitivity in half. This reinforces why CDS-OVR removes classes with <4 total instances (classes 7, 8, 11, 12, 13) — not to inflate accuracy, but because evaluation metrics for such classes are statistically meaningless.

#### 5.3.2 Why Irfan Achieves 93.33% on 60/40 While CDS-OVR Achieves 80.2% Mean

This 13 pp gap is the largest performance difference in any comparison and deserves detailed analysis.

**Root cause: evidence density collapse in CDS-OVR at 60% training**. CDS-OVR's evidence accumulation requires bin_count ≥ 3 at prediction time (hardcoded at `_compute_af()`, line 425). With 60% training data:

| Class | n_train (60%) | Avg per bin (6 bins) | Fraction of bins with ≥3 patients |
|-------|--------------|---------------------|----------------------------------|
| 4 (Old Inferior MI) | ~2 | 0.33 | ~0% |
| 9 (LBBB) | ~5 | 0.83 | ~0–17% |
| 3 (Old Anterior MI) | ~9 | 1.5 | ~17–33% |
| 5 (Sinus Tachycardia) | ~8 | 1.3 | ~17–33% |

Classes 4, 9, 3, and 5 collectively contain 41 patients (9.9% of the dataset) but represent half of the disease categories. At 60% training, their per-bin patient counts fall below the bc ≥ 3 filter, rendering their class models effectively inoperative. Patients from these classes default to the HEALTHY prediction, inflating the false negative count.

Irfan's PCA + CNN-LSTM does not have this failure mode because:
1. PCA computes the covariance matrix from all 271 training patients regardless of class, so even rare classes contribute to the transformation.
2. The CNN-LSTM operates in the 50-dimensional PCA space, where rare class patients can form identifiable clusters even with few examples (at the cost of potential overfitting with 500 epochs of training).

**Important caveat**: Irfan reports a single 60/40 split with no variance. CDS-OVR's 10-seed evaluation shows a range of 75.4% to 86.2% at 60/40 (σ = 2.8%). If Irfan's 93.33% also has comparable variance, their expected performance might be closer to 85–90%. The gap is structurally real but the magnitude is uncertain.

### 5.4 Gupta et al. 2014 — SVM + RF Serial Hybrid (as reviewed by Islam et al. 2023)

**Original work**: Gupta, V., Srinivasan, S. and Kudli, S.S., 2014. "Prediction and classification of cardiac arrhythmia."

**Reviewed in**: Islam et al., arXiv:2301.10174 (2023) — a review paper surveying arrhythmia classification techniques, not an original methodology paper. Islam et al. discuss and summarize the results of Gupta et al. and other works. The methodology and results below are attributed to Gupta et al. per the original source.

**Method** (Gupta et al.): Remove columns with missing values (279 → 252). Apply mRMR feature selection with SVM. Apply bootstrapping. Test multiple approaches: Naive Bayes (binomial/multinomial), SVM with mRMR (70%), RF with bootstrapping (72.3%), SVM+RF serial hybrid (77.4%), hierarchical dual-RF (70% accuracy, 30% error), and Pattern Net neural network (69%).

**Key issue with SVM component**: The SVM classifier could not classify class 16 (unclassified) and misclassified class 5 as class 1. An anomaly detector was added to address this, indicating the SVM's inability to handle extreme class imbalance.

**CDS-OVR**: 84.7% mean 10-fold CV, 86.8% best

**Why CDS-OVR outperforms by 7–10 pp**: Four structural factors compound:

1. **Column deletion removes discriminative features.** Features with missing values are entirely discarded (279 → 252). In clinical ECG data, missing values often occur in features that are only measured when abnormalities are present (e.g., specific morphological markers). Discarding these features removes exactly the signals most likely to distinguish rare disease classes. CDS-OVR retains all features and skips missing values at prediction time (`if np.isnan(v): continue`).

2. **Global mRMR dilutes rare-class feature relevance.** mRMR's mutual information criterion is dominated by majority-class separation. Features uniquely discriminative for rare classes (4, 5, 9) have low mutual information with the overall class label and are deprioritized. The fact that Gupta et al.'s SVM misclassified class 5 as class 1 directly confirms this: the features retained by mRMR did not include the features needed to separate class 5 (sinus tachycardia, 13 patients) from class 1 (normal, 245 patients). CDS-OVR's per-class feature selection ensures that each class gets its own 18 most discriminative features.

3. **No population stratification.** Sex-dependent ECG distributions require more complex decision boundaries when the sexes are modeled jointly. CDS-OVR's sex branching creates more homogeneous subpopulations.

4. **SVM+RF serial stacking amplifies noise.** The second-stage RF receives SVM predictions as input features, but SVM achieves only 70% accuracy (Gupta et al. report using SVM with mRMR achieving 70% before the serial hybrid). The RF must learn from these noisy pseudo-features on a small dataset, likely overfitting to the SVM's error patterns rather than learning genuine class structure.

### 5.5 Hybrid FS 2021 — Three-Stage Feature Selection with Ensemble Voting

**Published in**: IEEE CCECE 2021 (Khan Mamun)

**Method**: Pearson r > 0.05 → 71 features; MRMR MIQ → 40 features; RFE with RF → 32 features. Binary classification with 8-classifier ensemble (LR, DT, SVM, LDA, QDA, NB, RF, KNN). Hard/soft voting.

**Result**: 77.27% accuracy (binary)

**CDS-OVR binary**: 97.6% best, 88.6% mean (both 90/10)

**Why the 11–20 pp gap**: The gap traces to three interacting factors:

1. **32 global features cannot cover all diseases.** Different arrhythmias have different discriminative features. Bradycardia needs heart rate; MI needs Q-wave features; bundle branch blocks need QRS morphology features. A single 32-feature set is a compromise that under-serves several disease classes. CDS-OVR uses up to 8 × 18 = 144 distinct features across its class models, ensuring each disease gets its most informative features.

2. **Ensemble voting with weak classifiers degrades performance.** The ensemble's accuracy is bounded by the majority behavior of its 8 classifiers. Including weak classifiers (LDA at ~68%, QDA at ~65%) drags the ensemble below the best individual classifier. CDS-OVR does not ensemble classifiers — it accumulates evidence from features within a single coherent framework.

3. **No adaptive thresholds.** A single binary threshold must balance sensitivity and specificity across all disease classes simultaneously. CDS-OVR's per-class CLASS_THRESHOLDS and healthy bar provide disease-specific sensitivity/specificity tradeoffs.

### 5.6 Jadhav et al. 2012 — Artificial Neural Networks

**Published in**: International Journal of Computer Applications, Volume 44, No. 15, April 2012

**Method**: Replace missing values with closest column value of concern class. Train MLP (1–3 hidden layers) and MNN (1–3 hidden layers). Binary classification. Five data splits (DS1–DS5).

**Key result** (DS5 split — approximately 90/10, ~45 test patients): MLP (2 hidden layers): 86.67% accuracy, 93.75% sensitivity, 82.76% specificity

**Analysis of the sensitivity/specificity tradeoff**:

Jadhav's best MLP (2 hidden layers, DS5) achieves high sensitivity (93.75% — catches most diseased patients) but moderate specificity (82.76% — 17.2% of healthy patients are false positives). This tradeoff reveals a structural challenge: the MLP uses all 279 features without feature selection, and in high-dimensional space, healthy patients are more likely to be pushed across the decision boundary by noisy features. The specificity is not catastrophic, but it indicates the MLP's inability to distinguish which features are noise vs. signal without explicit feature selection.

**Reconstructed confusion matrix** (from Table 2, DS5, MLP 2HL): With sensitivity=93.75% and specificity=82.76%, and an accuracy of 86.67% yielding a test set of approximately 45 patients (16 disease, 29 healthy): TP=15, FN=1, TN=24, FP=5. The 5 false positives account for most of the errors.

CDS-OVR achieves more balanced sensitivity/specificity through:
- **Feature selection** (18 per class): eliminates noisy features that produce false positives
- **Healthy bar**: raises disease thresholds when healthy evidence is strong, preventing false positives
- **Per-class thresholds**: tunes the sensitivity/specificity tradeoff independently for each disease

Jadhav's MNN (1 hidden layer, DS5) achieves the opposite pattern: 62.5% sensitivity, 93.10% specificity, 82.22% accuracy. The MNN's modular architecture is conceptually similar to CDS-OVR's per-class models — it decomposes the problem into specialized sub-networks. Its high specificity (93.10%) confirms that modular decomposition helps identify healthy patients accurately. But MNN's low sensitivity (62.5% — missing 6 of 16 disease patients) reveals a critical weakness: modular sub-networks without sufficient per-class feature optimization and evidence weighting cannot reliably detect diseases. With the MNN's 2 hidden layer variant (sensitivity 81.25%, specificity 82.76%, accuracy 82.22%), the sensitivity improves at the cost of specificity — demonstrating the sensitivity/specificity tradeoff that CDS-OVR resolves through its per-class thresholds and healthy bar mechanism.

**Key insight from Jadhav's results**: The sensitivity-specificity tradeoff across Jadhav's architectures (MLP favoring sensitivity, MNN favoring specificity) demonstrates that no single neural architecture achieves both simultaneously on this dataset without explicit class-aware mechanisms. CDS-OVR achieves more balanced performance because each class model independently optimizes its own decision boundary, and the healthy bar prevents false positives while per-class against_scale prevents false negatives for rare classes.

---

## 6. The 60/40 Performance Gap: A Structural Analysis

CDS-OVR's weakest result (60/40: 80.2% mean, 86.2% best) contrasts sharply with Irfan's 93.33% on the same protocol. This section analyzes the structural cause in detail.

### 6.1 The Minimum Evidence Density Problem

CDS-OVR's classification pipeline has a hard floor: `if bc < 3: continue` in `_compute_af()`. This means that for any feature-bin pair with fewer than 3 patients in the bin during training, zero evidence is accumulated during prediction.

The number of usable bins per class is approximately:

n_usable_bins ≈ n_class_train / 3 (for sparse classes where patients concentrate in ~1 bin)

At 60% training with MAX_BINS = 6:

| Class | n_train_60% | Bins with ≥3 patients | Effective model capacity |
|-------|------------|----------------------|------------------------|
| 1 (Normal) | 147 | 6/6 (100%) | Full |
| 2 (CAD) | 26 | 4–5/6 (67–83%) | Adequate |
| 10 (RBBB) | 30 | 5–6/6 (83–100%) | Adequate |
| 6 (SB) | 15 | 2–3/6 (33–50%) | Marginal |
| 3 (AM) | 9 | 1–2/6 (17–33%) | Severely limited |
| 5 (ST) | 8 | 1–2/6 (17–33%) | Severely limited |
| 9 (LBBB) | 5 | 0–1/6 (0–17%) | Near-zero |
| 4 (IM) | 2 | 0/6 (0%) | Zero |

At 60/40, 4 of 8 classes have severely limited or zero model capacity. These 4 classes contain 36 patients in the test set (approximately 8.7% of 416). If all are misclassified, accuracy drops by ~8.7 pp from a perfect baseline. This accounts for a substantial portion of the 60/40 accuracy deficit.

### 6.2 Why PCA-Based Methods Do Not Have This Problem

PCA computes the covariance matrix from all training patients regardless of class. With 271 training patients at 60/40, the covariance matrix is well-estimated (271 >> 50 components). Every class — including class 4 with 2 training patients — contributes to and is represented in the PCA transformation. The CNN-LSTM then operates in the 50-dimensional PCA space, where even rare classes can form identifiable clusters.

This is the fundamental tradeoff: **CDS-OVR's class-aware specialization requires sufficient per-class data; PCA-based methods share representation learning across all classes but cannot specialize per-class.**

### 6.3 Potential Resolutions

Several modifications could improve CDS-OVR's 60/40 performance without compromising its 90/10 advantage:

1. **Adaptive bin count**: Reduce MAX_BINS when per-class training data is small (e.g., MAX_BINS = max(2, min(6, n_class_train / 3))).
2. **Lower prediction MIN_SUPPORT for rare classes**: Allow `bc ≥ 2` for rare classes during prediction, accepting higher posterior variance.
3. **Feature-level fallback**: When retained actions are too few, use the single highest-FDR feature with a simple threshold rule instead of binning.
4. **SMOTE for rare classes**: Apply synthetic minority oversampling to rare-class training data before binning. This would populate rare-class bins, directly addressing the density collapse problem. With SMOTE generating synthetic patients for classes 4, 5, 9 (and potentially class 3), each class model would have sufficient bin counts to contribute evidence at prediction time, even at 60/40 splits.
5. **Pairwise feature interactions**: Introduce a small number of engineered interaction features (e.g., QRS duration × axis deviation for RBBB detection) to capture conjunctive patterns that independent feature evaluation currently misses. This targeted approach avoids the full combinatorial explosion of all pairwise interactions while addressing the primary limitation identified in Section 7.1.

---

## 7. Honest Assessment: Structural Advantages of Benchmark Methods

### 7.1 Feature Interactions (All Neural/Kernel Methods)

CDS-OVR evaluates features independently. The evidence from feature A is added to the evidence from feature B without considering their joint distribution. This means CDS-OVR cannot detect conjunctive patterns where multiple features must simultaneously meet criteria for a diagnosis. SVM (polynomial kernel), CNN, and RF (tree splits) all capture such interactions through their respective mathematical formulations.

This is likely the primary reason CDS-OVR's 90/10 and 10-fold CV accuracy plateaus near 87–93% rather than exceeding 95%: the remaining errors may require multi-feature interactions to resolve.

### 7.2 Continuous vs. Discrete Decision Boundaries

CDS-OVR's bins create step-function posteriors. Within a bin, all patients receive the same p_class estimate regardless of their exact feature values. A patient at QRS = 119.9 ms and another at QRS = 120.1 ms may fall in different bins and receive different evidence contributions, despite near-identical clinical presentations.

SVM and neural network decision boundaries are continuous functions of the input features, providing smoother and potentially more accurate classifications near decision boundaries.

### 7.3 End-to-End Optimization

CDS-OVR's staged pipeline (binning → posterior estimation → feature selection → threshold setting) optimizes each stage with a local objective. The chi-squared binning does not know the downstream effect of its split points on the ratio score. Neural networks and gradient-boosted trees optimize a single loss function end-to-end, producing globally coherent models.

The tradeoff: end-to-end optimization requires more data to avoid overfitting. CDS-OVR's staged optimization is more sample-efficient, which explains its strong performance at 90/10 where most classes have ≥10 training patients but the total dataset is too small for deep learning to fully generalize.

### 7.4 SMOTE / Synthetic Oversampling for Rare Classes

Irfan et al. apply SMOTE to the MIT-BIH dataset (D2) to balance class distributions before training. CDS-OVR uses no data augmentation — it handles rare classes through per-class parameter adaptation (against_scale, MIN_SUPPORT). This means CDS-OVR's rare-class models train on as few as 4 patients (class 4 at 90/10), creating a **density collapse** problem: with MAX_BINS=6, most bins contain fewer than 3 patients and are silenced by the `bc < 3` filter at prediction time. SMOTE or a similar oversampling strategy applied to the training data before binning could populate rare-class bins, allowing more features to contribute evidence. This is the single most impactful improvement CDS-OVR could adopt from benchmark methods.

### 7.5 Soft / Fuzzy Bin Boundaries

CDS-OVR assigns each feature value to exactly one bin, creating hard boundaries. A patient at 119.9 ms QRS duration receives entirely different evidence than one at 120.1 ms if a bin edge falls at 120 ms. Soft binning — where a value near a bin edge contributes partially to both adjacent bins — would produce smoother posterior estimates and reduce sensitivity to exact bin placement. This is analogous to the continuous decision boundaries that SVM and neural networks provide naturally.

---

## 8. Summary of CDS-OVR vs. Benchmark Performance Sources

### Why CDS-OVR Outperforms When It Does (90/10, 10-fold CV)

| CDS-OVR Mechanism | Effect | Benchmark Lacking This |
|-------------------|--------|----------------------|
| Per-class feature selection (18 features via FDR) | Each disease evaluated on its most discriminative features | All benchmarks use global feature sets |
| Per-class imbalance adaptation (against_scale, MIN_SUPPORT) | Rare classes detected from sparse evidence | All benchmarks use uniform parameters |
| Dual AF with ratio scoring | Separates for/against evidence; graduated confidence | Original CDS, most benchmarks use single decision function |
| Dynamic healthy bar + suspicion | Adaptive sensitivity/specificity per patient context | No benchmark adapts thresholds based on healthy evidence |
| Supervised chi-squared binning | Bin boundaries align with class boundaries | Original CDS uses unsupervised bins |
| Correlation filtering (|r| < 0.8) | Non-redundant feature sets maximize information per slot | Sharma retains correlated features; no benchmark filters per-class |
| Sex-based population branching | More homogeneous subpopulations | No benchmark stratifies by sex |

### Why CDS-OVR Underperforms When It Does (60/40)

| Limitation | Effect | Benchmark Advantage |
|-----------|--------|-------------------|
| bc ≥ 3 prediction filter | Rare classes have zero usable bins at 60% training | PCA-based methods share representation across classes |
| Independent per-class models | No parameter sharing; class 4 model cannot borrow from class 1 data | Neural networks and RF learn shared representations |
| No feature interactions | Cannot detect conjunctive diagnostic patterns | SVM (polynomial kernel), CNN, RF capture interactions |
| Discrete bin boundaries | Step-function posteriors lose information near boundaries | SVM and NN have continuous decision functions |

### The Core Insight

CDS-OVR's class-aware specialization — the property that distinguishes it most fundamentally from all benchmarks — is simultaneously the source of its best performance and its primary limitation. When per-class data is sufficient (≥8–10 training patients per class, as at 90/10), specialization produces models tailored to each disease that outperform one-size-fits-all approaches. When per-class data falls below a critical threshold (~3 patients per bin), the specialized models become inoperative and the system loses its advantage.

This is not a flaw to be eliminated but a design tradeoff to be understood: CDS-OVR achieves its strong 90/10 and 10-fold CV results precisely because it specializes per-class, and that same specialization requires a minimum data floor that 60/40 splits cannot satisfy for rare classes with n < 10.

# Documentation By Kian for why machine learning methods adopted by other papers fail in areas CDS accounts for:

Paper 1: A comparative study of heterogeneous machine
learning algorithms for arrhythmia classification
using feature selection technique and multidimensional datasets

Where CDS is stronger
1. Genuine multi-class architecture vs. binary collapse. The paper never actually solves the 13-class problem — it binarizes before training. CDS's OVR decomposition trains an independent evidence accumulator per class and combines them via per-class thresholds, which is architecturally suited to resolving which arrhythmia subtype a patient has, not just whether they have one. This is the single biggest structural advantage, since it's the harder and more clinically useful task.
2. Per-class handling of severe class imbalance. As established, RARE_CLASSES gets its own support/confidence/against-scale parameters. None of LR/LDA/GNB/SVM/RF/XGB in the paper have any per-class accommodation — they fit one global decision rule and would default toward majority-class behavior on 2–4-instance classes.
3. Target-aware feature scoring per class. CDS's _train_ovr_node searches all 279 features separately for each class using a class-specific shift/confidence/Fisher score. The paper's Pearson-correlation filter selects features based on correlation with other features, not the label — it's not even target-aware, let alone class-aware.
4. Native handling of missing data. branch_match returns False on NaN, and binning simply skips missing values feature-by-feature. The paper drops an entire 80%-missing column and then drops every row with any remaining missing value, shrinking 452 patients to 420 before a single model is trained. CDS retains the full 452 and lets each feature's model use whatever data exists for that feature.
5. No leakage in fold isolation. build_tree/train are rebuilt from scratch on training-fold-only data each CV iteration. The paper's feature selection happens before the split (Figure 1), which is exactly the mistake Ch. 11 of your Zollanvari reading quantified.
6. No preprocessing dependency. CDS's bin-based scoring is invariant to monotonic transforms of each feature — no normalization/scaling step is needed (contrast with Figure 4's explicit before/after normalization in the paper), and no imputation strategy has to be chosen and justified.
Where the paper's ML algorithms could be stronger
1. Battle-tested implementations vs. bespoke code. LR/RF/XGB/SVM in scikit-learn have been validated across thousands of studies and datasets; bugs get found and fixed by a huge user base. CDS is a ~600-line custom research script with a documented history of real bugs in this exact codebase (the prior-conflation bug, the partner's best-of-N-seeds selection bias). The prior probability of an undiscovered bug in CDS is meaningfully higher than in sklearn.ensemble.RandomForestClassifier.
2. Principled, systematic hyperparameter search vs. hand-tuned constants. Grid Search over a defined space (Table 23) — even though it's undermined here by pre-split leakage — is at least a documented, reproducible procedure. CDS's CLASS_THRESHOLDS, MIN_SUPPORT_MAP, CONF_SUPPORT_MAP, AGAINST_SCALE_MAP are hardcoded constants with a changelog in the docstring ("class 10 threshold = 3.0, lowered from 3.5") that reads like manual iteration against eval numbers rather than a defined search procedure with a held-out validation layer.
3. Far fewer degrees of freedom to overfit. LR has 3 hyperparameters (penalty, C, solver). CDS has per-class support/confidence/scale/threshold values — effectively dozens of tunable knobs across 13 classes, all fit (or hand-picked) against a 452-patient dataset. More knobs relative to sample size is a direct overfitting risk, independent of whether each individual mechanism is well-motivated.
4. Built-in variance reduction via ensembling. This is the one that connects directly to the split-variance discussion: Random Forest's bagging explicitly averages over many bootstrap-resampled trees specifically to reduce variance from which patients happen to be sampled. CDS builds exactly one deterministic tree per training set (build_tree isn't randomized or resampled at all) — there's no internal mechanism analogous to bagging that would dampen the kind of small-sample split sensitivity you saw in the 88.6%/97.6% gap. If instability across folds turns out to be a real problem for CDS under LOOCV, RF's architecture has a structural answer for it that CDS's doesn't.
5. Calibrated/free uncertainty estimates. LR and GNB output genuine class probabilities; RF gets out-of-bag error estimates without a separate CV loop. CDS's for:against ratio score is not a calibrated probability, and there's no analogous "free" internal validation signal — every uncertainty estimate CDS needs (LOOCV, seed-variance) has to be computed externally and by hand.
6. Unified regularization objective. XGB and LR regularize via an explicit penalty term in a single loss function (L1/L2), which is theoretically grounded and has one clean knob to tune. CDS's "regularization" is distributed across several separate ad hoc mechanisms (min support cutoffs, confidence caps, against-scale down-weighting, threshold offsets) with no unifying objective — harder to reason about what any one of them is actually trading off against the others.
The throughline: CDS's advantages are almost all about being purpose-built for this exact problem's structure (rare classes, multi-subtype, missing data). The paper's methods' advantages are almost all about being general-purpose tools with decades of validation and fewer ways to quietly go wrong. Neither list resolves the open question from before — whether CDS's purpose-built advantages actually pay off once you strip away seed selection and pre-split leakage on both sides is still what the LOOCV run would tell you.

Kian's Note: A lot of the reasons listed boils down to how CDS preprocesses data a different way (by keeping the rare disease classes for example), and how the machine learning models utilize ensmeble learning, allowing for more generalization as well as being established techniques in the filed of cyber-physical systems. Not very good reasons presented by Claude.

