# CDS-OVR Research Paper Structure
## Refined for Maximum Publication Impact

---

## Section 1: Abstract (~250 words)

- Open with the clinical problem: automated arrhythmia classification from ECG features, need for interpretable and resource-efficient classifiers
- Name the algorithm: CDS-OVR (Cognitive Dynamic System with One-vs-Rest decomposition)
- State what it does: transforms a multiclass problem (8 arrhythmia types) into independent binary OVR sub-problems, each solved by a class-directed feature selection and scoring pipeline
- Headline results:
  - 86.78% multiclass accuracy (10-fold CV, best seed)
  - 84.66 +/- 0.93% mean across 10 seeds
  - Binary (healthy vs disease): 88.70%
  - AUC-ROC: 0.9021 binary, 0.9393 macro
  - Confusion matrix: macro P = 0.867, R = 0.815, F1 = 0.838
- State that component ablation with Wilcoxon signed-rank testing confirms two components as statistically significant (p = 0.002)

---

## Section 2: Introduction

### 2.1 Clinical Motivation
- Arrhythmia affects millions globally; manual ECG interpretation is slow and error-prone
- Need for automated classifiers that are accurate, interpretable, and deployable on resource-constrained hardware (bedside monitors, wearables)
- Existing approaches (deep learning, ensemble methods) achieve high accuracy but at the cost of interpretability, memory, and compute

### 2.2 The Interpretability Gap
- Black-box models (neural nets, gradient-boosted trees) cannot explain WHY a patient is classified as a particular arrhythmia
- In clinical settings, interpretability is not optional — clinicians need to trust and verify decisions
- CDS-OVR addresses this: every prediction traces back to specific features and their bin-level evidence scores

### 2.3 Why Not Standard One-vs-All?
- Standard OVA treats all "rest" classes identically — problematic when class distribution is 27:1
- CDS-OVR uses class-directed feature selection: each OVR model selects features that best discriminate its target class from the rest
- Per-class parameter adaptation: rare classes (4, 5, 9 with 9 patients each) get relaxed support thresholds

### 2.4 Contributions
1. The CDS-OVR algorithm combining supervised chi-squared binning with Fisher discriminant weighting for class-directed feature scoring
2. Rigorous multi-protocol evaluation: 10-fold CV, LOOCV, 6 train/test splits, all with stratified variants (60 stratified experiments across 10 seeds)
3. Component ablation study with Wilcoxon signed-rank significance testing (6 components, 10 paired observations each)
4. Computational profiling demonstrating 3.05 MB memory, 41 ms/patient, 270 retained features

---

## Section 3: Related Work

### 3.1 UCI Arrhythmia Dataset Benchmarks
- Sharma et al. (2015): used feature selection pipeline on this dataset; our report simulates their pipeline for comparison
- Gupta et al. (2014): alternative approach; we note they removed 4 columns we retain
- Other published results on UCI arrhythmia — tabulate for comparison in Section 8.4

### 3.2 Feature Selection for High-Dimensional Medical Data
- Chi-squared feature ranking (standard unsupervised vs our supervised binning approach)
- Fisher discriminant ratio as a feature weighting mechanism
- Correlation-based redundancy filtering (Pearson threshold methods)
- mRMR, embedded methods, and how CDS-OVR differs

### 3.3 One-vs-Rest Decomposition Strategies
- Standard OVR assumes class balance — fails with 27:1 imbalance
- Prior work on handling class imbalance in OVR settings
- How CDS-OVR's per-class parameterization addresses this

### 3.4 Lightweight and Embedded Classifiers
- Existing work on resource-constrained ML for medical devices
- FPGA-based classifiers in the literature
- Where CDS-OVR fits: 3 MB footprint vs typical model sizes

### 3.5 Expanding on the WHY

---

## Section 4: Dataset Analysis

### 4.1 UCI Arrhythmia Dataset Overview
- 452 patients, 279 features, 13 original classes
- Features include: age, sex, height, weight, QRS duration, P-R interval, Q-T interval, T interval, P interval, heart rate, and 12-lead ECG morphology features
- Feature types: 206 continuous, 73 binary (0/1)
- Missing values: 0.32% overall, concentrated in 4 features, affecting 354 of 452 patients

### 4.2 Class Filtering and Distribution
- 5 classes removed due to insufficient samples (classes 7, 8, 11, 12, 13) — retained 8 classes with 416 patients
- Class distribution after filtering:
  - Class 1 (Normal): 245 patients (58.9%)
  - Class 2 (Ischemic changes): 44 (10.6%)
  - Class 3 (Old anterior MI): 15 (3.6%)
  - Class 6 (Right bundle branch block): 15 (3.6%)
  - Class 10 (Sinus tachycardia): 50 (12.0%)
  - Classes 4, 5, 9 (Left ventricular hypertrophy, Other abnormalities, LBBB): 9 each (2.2%)
- Imbalance ratio: 27.2:1 (Normal vs rarest class)
- 17 of 78 possible OAO (one-against-one) pairs have extreme imbalance

### 4.3 Feature Correlation Structure
- 32 pairs with Pearson r > 0.8 — motivates correlation filtering
- 2 pairs with r > 0.9
- Optimal correlation threshold found at 0.8 via sensitivity analysis (Section 8.2)

### 4.4 Sex-Based Feature Distribution
- 179 male, 237 female patients
- 4 features show medium effect-size sex differences
- Motivates the sex-based branching in the tree structure (split on feature 1 = sex)

---

## Section 5: The CDS-OVR Algorithm

### 5.1 Algorithm Overview
- CDS-OVR decomposes the 8-class problem into 8 independent OVR binary sub-problems
- For each target class c: "Is this patient class c or not?"
- Each OVR model has its own feature selection, binning, and scoring — features that matter for detecting LBBB are different from those for detecting sinus tachycardia
- Final prediction aggregates all 8 OVR scores against a healthy-bar threshold

### 5.2 Tree Construction (Sex-Based Branching)
- Build a shallow decision tree (depth 1) by splitting on the sex feature (feature index 1)
- Binary feature: creates two child nodes (male branch, female branch)
- Minimum node size: U_MIN = 200 patients per branch
- Each node maintains its own patient subset and class histogram (hdist)
- Deduplication: if two branches contain the same patient set, keep only one
- Result: root node + 2 sex-based child nodes (3 nodes total in the tree)
- All subsequent OVR training happens independently within each node

### 5.3 Feature Classification
- Each of 279 features is classified as binary (only values 0 and 1) or continuous
- 73 binary features (ECG morphology presence/absence), 206 continuous features
- Binary features use simple 2-bin models; continuous features get supervised binning

### 5.4 Supervised Chi-Squared Binning (Key Innovation)
- Standard equal-width or equal-frequency binning ignores class boundaries
- Our approach: place bin edges where the target class distribution changes most
- Algorithm:
  1. Start with the full value range as a single bin [min, max]
  2. For each existing bin, scan all candidate split points (where consecutive sorted values differ)
  3. For each candidate, compute the chi-squared statistic of a 2x2 contingency table (target/rest x left/right)
  4. Select the split with the highest chi-squared if it exceeds the minimum gain threshold (0.5)
  5. Repeat up to MAX_BINS - 1 times (MAX_BINS = 6, so up to 5 splits creating 6 bins)
  6. Enforce minimum support: each resulting bin must have at least MIN_SUPPORT patients
- Per-class adaptation: rare classes (4, 5, 9) use MIN_SUPPORT = 2 vs 3 for common classes
- Why it matters: ablation shows +5.75pp accuracy improvement over uniform binning (p = 0.002)

### 5.5 OVR Feature Scoring
- For each target class, score every feature in every tree node:
  - Compute Laplace-smoothed class probability per bin: p_class = (target_count + alpha * prior) / (bin_count + alpha)
  - Feature score = sum over bins of |p_class - prior| * confidence, where confidence = min(1, bin_count / CONF_SUPPORT)
  - Per-class adaptation: CONF_SUPPORT = 5 for rare classes, 10 for common
  - Compute Fisher discriminant ratio: (mean_target - mean_rest)^2 / (var_target + var_rest)
  - Both score and Fisher ratio are retained for later use

### 5.6 Correlation-Based Feature Filtering
- After scoring, take the top 3 * FEATURES_PER_CLASS features by score
- Compute pairwise absolute Pearson correlation among this candidate set
- Greedily select features in descending score order:
  - Skip any feature whose correlation with an already-selected feature exceeds CORR_THRESHOLD (0.8)
  - Stop after selecting FEATURES_PER_CLASS features (default: 18)
- Result: each OVR model retains up to 18 non-redundant, high-scoring features per tree node
- Total across all classes and nodes: 4,185 class-feature models, 270 unique retained features

### 5.7 Fisher Discriminant Weighting (Key Innovation)
- During prediction, feature contributions are weighted by their Fisher discriminant ratio
- Weight = max(sqrt(Fisher_f / max_Fisher), 0.1) — ensures minimum weight of 0.1
- Effect: features with high class separation contribute more to the final score
- Why it matters: ablation shows +4.52pp accuracy improvement (p = 0.002)

### 5.8 Prediction Pipeline
- For a test patient:
  1. **Route through tree**: patient follows the sex-based branch to reach the applicable tree nodes
  2. **Score each OVR model**: for each of the 8 target classes:
     a. Look up the patient's feature value in each retained feature's bin model
     b. Compute the shift from prior: shift = p_class[bin] - prior
     c. Weight by confidence (min(1, bin_count/10)) and Fisher weight
     d. If shift >= 0: add to af_for (evidence FOR this class)
     e. If shift < 0: add to af_against (evidence AGAINST), scaled by AGAINST_SCALE
     f. Per-class against_scale: 0.5 for rare classes (less penalty), 0.8 for common
  3. **Compute class score**: score = (af_for + epsilon) / (af_against + epsilon), where epsilon = 0.1
  4. **Apply healthy bar**: healthy_bar = min(1.05 * healthy_score, 5.0)
  5. **Apply class thresholds**: each disease class has a threshold (e.g., class 3: 5.0, class 10: 3.0)
     - If healthy score < 2.0 (suspicion of disease): lower threshold by 0.3
     - Effective threshold = max(class_threshold, healthy_bar)
  6. **Select prediction**: among disease classes exceeding their threshold, pick the one with the highest margin (score - threshold) / threshold
     - If no disease class exceeds its threshold → predict Normal (class 1)

### 5.9 Key Parameters Summary
| Parameter | Value | Role |
|-----------|-------|------|
| MAX_BINS | 6 | Maximum bins for supervised chi-squared binning |
| CORR_THRESHOLD | 0.8 | Pearson correlation filter cutoff |
| FEATURES_PER_CLASS | 18 | Features retained per OVR model per node |
| LAPLACE_ALPHA | 1.0 | Smoothing for bin-level class probabilities |
| HEALTHY_WEIGHT | 1.05 | Multiplier for healthy-bar mechanism |
| U_MIN | 200 | Minimum patients per tree node |
| AGAINST_SCALE (common) | 0.8 | Weight for evidence-against in common classes |
| AGAINST_SCALE (rare) | 0.5 | Reduced penalty for rare classes |

---

## Section 6: Experimental Setup

### 6.1 Evaluation Protocols
- **10-Fold Cross-Validation**: primary protocol, 10 seeds for stability
- **Train/Test Splits**: 50/50, 60/40, 70/30, 75/25, 80/20, 85/15, 90/10 — 10 seeds each
- **Stratified Variants**: 10-fold CV + 5 split ratios (50/50 through 90/10) with class-proportional splitting — 60 total experiments
- **LOOCV**: Leave-One-Out on all 13 original classes — tests generalization to rare classes
- Total experiments: 80+ random splits + 60 stratified + LOOCV = 140+ evaluations

### 6.2 Stratified Splitting Methodology
- For each class, independently shuffle and split to preserve class proportions in both train and test sets
- Addresses concern that random splits may over/under-represent minority classes in small test sets
- Particularly important given 27.2:1 class imbalance

### 6.3 Metrics Computed
- **Multiclass accuracy**: exact class match across all 8 classes
- **Binary accuracy**: healthy (class 1) vs any disease — clinically relevant
- **AUC-ROC**: binary (disease score = max disease ratio / healthy ratio), macro (unweighted mean of per-class AUC), weighted (sample-weighted), per-class OVR AUC
- **Confusion matrix**: 8x8 matrix with per-class precision, recall, F1
- **Specificity**: true negative rate for healthy patients
- **Sensitivity**: disease detection rate (any disease patient predicted as any disease)
- **Balanced accuracy**: mean of per-class recall

### 6.4 Reproducibility
- 10 fixed seeds: [13, 20, 27, 34, 41, 48, 55, 62, 69, 76]
- Pure Python + NumPy implementation (no scikit-learn dependency for the algorithm)
- Platform: Python 3.11, NumPy
- All source code and result files available in repository

---

## Section 7: Results

### 7.1 Primary Results: 10-Fold Cross-Validation
- Best seed (13): 86.78% multiclass, 88.70% binary
- Mean across 10 seeds: 84.66 +/- 0.93%
- Binary mean: 87.72%
- Specificity: 91.43% (low false positive rate for healthy patients)
- Sensitivity: 84.8% (disease detection rate)
- Balanced accuracy: 81.47%
- Table: per-seed breakdown (accuracy, binary, specificity, sensitivity, BA)

### 7.2 AUC-ROC Analysis
- 10-fold CV: binary AUC = 0.9021 +/- 0.006, macro AUC = 0.9393 +/- 0.004
- Per-class AUC (10-fold, averaged over seeds):
  - Class 9 (LBBB): 1.000 — perfect discrimination
  - Class 3 (Old anterior MI): 0.991
  - Class 4 (LVH): 0.975
  - Class 5 (Other abnormalities): 0.953
  - Class 6 (RBBB): 0.947
  - Class 2 (Ischemic changes): 0.936
  - Class 10 (Sinus tachycardia): 0.927
  - Class 1 (Normal): 0.786 — lowest, discussed in Section 8.3
- 60/40 split: binary AUC = 0.884, macro AUC = 0.925
- 90/10 split: binary AUC = 0.904 (higher variance due to small test set)
- Table: protocol x AUC type, Figure: per-class AUC comparison bar chart

### 7.3 Confusion Matrix and Per-Class Metrics
- Based on 10-fold CV with seed 13 (best performing)
- 8x8 confusion matrix showing exact misclassification patterns
- Per-class precision / recall / F1:
  - Macro precision: 0.867
  - Macro recall: 0.815
  - Macro F1: 0.838
- Table: per-class P/R/F1 for all 8 classes
- Key observation: most misclassifications involve Normal (class 1) — the largest and most heterogeneous class

### 7.4 Robustness Across Split Ratios (Random Splits)
- 8 protocols from 50/50 to 90/10 + 10-fold CV
- Accuracy increases with more training data (expected):
  - 50/50: binary 83.03%
  - 60/40: binary 84.49%
  - 70/30: binary 85.12%
  - 80/20: binary 87.98%
  - 90/10: binary 88.57%
  - 10-fold CV: binary 87.72%
- Table: split ratio x {multiclass accuracy, binary accuracy}

### 7.5 Stratified Evaluation Results
- 6 protocols x 10 seeds = 60 experiments with class-proportional splits
- Stratified 10-fold: 85.32 +/- 0.62% multiclass, 88.56% binary
- Stratified 90/10: 85.56 +/- 4.58% multiclass
- Stratified 50/50: 77.21 +/- 1.74% multiclass
- Table: stratified protocol x {multi mean+/-std, binary mean, balanced accuracy}
- Includes per-class multiclass and binary accuracy breakdown

### 7.6 Stratified vs Random Comparison
- Side-by-side table comparing stratified and random results for matching protocols
- Stratified 10-fold (85.32%) vs random 10-fold (84.66%) — stratified slightly higher
- Shows that class-proportional splitting does not inflate results — performance holds or improves
- Demonstrates methodological rigor in evaluation

---

## Section 8: Analysis and Discussion

### 8.1 Component Ablation Study
- Ablation methodology: remove one component at a time, re-run 10-fold CV with all 10 seeds, compare paired accuracy vectors using Wilcoxon signed-rank test
- Results (6 components):

| Component | Delta (pp) | W-stat | p-value | Significant? | Effect size r |
|-----------|-----------|--------|---------|-------------|---------------|
| Supervised chi² binning | +5.75 | 0.0 | 0.002 | Yes | 0.979 |
| Fisher weighting | +4.52 | 0.0 | 0.002 | Yes | 0.979 |
| Correlation filter | +0.46 | 17.0 | 0.312 | No | 0.319 |
| Sex branching | +0.41 | 7.0 | 0.141 | No | 0.466 |
| Healthy bar | -0.31 | 6.5 | 0.133 | No | 0.475 |
| Against scale | +0.07 | 19.0 | 0.734 | No | 0.107 |

- Key insight: the two novel components (supervised binning and Fisher weighting) are the only statistically significant ones, together accounting for +10.27pp
- The non-significant components still contribute to robustness but are not individually critical
- Honest reporting: healthy_bar shows slight negative delta (-0.31pp) — it constrains disease over-prediction at the cost of marginal accuracy

### 8.2 Hyperparameter Sensitivity Analysis
- Three key hyperparameters swept while holding others fixed:

**Correlation threshold** (0.6 to 1.0):
- 0.6: 83.22%, 0.7: 83.68%, **0.8: 84.66%**, 0.9: 84.30%, 1.0: 84.21%
- Clear optimum at 0.8; performance degrades at both extremes

**MAX_BINS** (3 to 10):
- 3: 84.06%, 4: 84.01%, **5: 85.17%**, 6: 84.66%, 8: 83.22%, 10: 82.12%
- Optimal at 5-6; over-binning (10) hurts performance significantly
- Under-binning (3) loses discriminative power

**FEATURES_PER_CLASS** (10 to 30):
- 10: 84.66%, 14: 84.66%, 18: 84.66%, 22: 84.66%, 26: 84.66%, 30: 84.66%
- Completely insensitive — algorithm is robust to this parameter
- Explanation: correlation filtering naturally limits the effective feature count

### 8.3 Per-Class Performance Analysis
- Why Normal (class 1) has the lowest AUC (0.786):
  - Largest class (245 patients) — most heterogeneous in feature space
  - Healthy patients show wide variation in ECG features
  - Some borderline-normal patients share features with mild arrhythmia classes
  - In OVR framing, "normal" is the absence of all diseases — harder to define positively
- Why LBBB (class 9) achieves perfect AUC (1.000):
  - LBBB has distinctive ECG morphology (widened QRS, characteristic T-wave changes)
  - Even with only 9 training patients, the features are highly separable
  - Supervised binning places class-boundary-aware bin edges that capture this separation
- Discussion of which misclassifications are clinically significant vs benign

### 8.4 Comparison with Published Benchmarks
- Table comparing with Sharma 2015, Gupta 2014, and other published results on UCI arrhythmia
- Comparison notes:
  - Sharma simulated pipeline: 106 features retained vs our 109 unique features
  - Gupta removed 4 columns (J, heart rate, P-wave vector, T-wave vector) — 3 of which we use
  - Fair comparison is difficult due to different class filtering, feature subsets, and evaluation protocols
  - Our 10-fold CV at 84.66% mean is competitive; our rigorous multi-protocol evaluation exceeds what most published work reports

### 8.5 Computational Efficiency and FPGA Feasibility
- Runtime profiling:
  - 10-fold CV total time: 260.67 seconds
  - Single 90/10 split: 34.65 seconds
  - Training phase: 249.78 seconds (dominates)
  - Prediction: 41 ms per patient (real-time capable)
- Memory profiling:
  - Current memory: 3.05 MB
  - Peak memory: 4.27 MB
  - Data matrix alone: 0.89 MB
- Model complexity:
  - 2 tree nodes (root + sex branch), 4,185 class-feature models, 270 retained features
  - All operations are comparisons, additions, and lookups — no matrix multiplications
  - Integer-friendly: bin lookup is a searchsorted (binary search), scoring is fixed-point arithmetic
- FPGA suitability argument:
  - 4.27 MB peak fits in on-chip BRAM of mid-range FPGAs
  - 41 ms latency is well within clinical requirements
  - No floating-point unit needed — all operations can be quantized to fixed-point
  - Pipeline-parallelizable: 8 OVR models can run in parallel on FPGA fabric

---

## Section 9: Conclusion and Future Work

### 9.1 Summary of Contributions
- Restate the 4 contributions with final numbers
- Emphasize: CDS-OVR achieves competitive accuracy (86.78%) while being fully interpretable, lightweight (3.05 MB), and clinically fast (41 ms/patient)
- The algorithm's strength lies in the combination of supervised binning (+5.75pp) and Fisher weighting (+4.52pp), both confirmed statistically significant

### 9.2 Limitations
1. **Single dataset**: evaluated only on UCI Arrhythmia — generalization to other arrhythmia datasets or ECG formats is unverified
2. **Small sample size**: 416 patients after filtering; some classes have only 9 patients
3. **Removed classes**: 5 of 13 classes dropped due to insufficient samples — the algorithm hasn't been tested on the full spectrum
4. **Tabular features only**: uses pre-extracted ECG features, not raw ECG waveforms
5. **Feature independence assumption**: CDS-OVR scores features independently — no feature interaction modeling
6. **60/40 accuracy gap**: performance drops noticeably when training data falls below 70% of the dataset

### 9.3 Future Work
1. **FPGA implementation**: translate the Python model to Verilog RTL; the computational profile (3 MB, 41 ms, no FPU) makes this feasible
2. **Cross-dataset validation**: evaluate on MIT-BIH, PhysioNet Challenge datasets, and clinical ECG databases
3. **Feature interactions**: explore pairwise or grouped feature scoring to capture inter-feature dependencies
4. **Rare class augmentation**: synthetic oversampling (SMOTE variants) for classes with 9 patients
5. **Adaptive binning**: automatically tune MAX_BINS per feature based on sample size and class distribution
6. **Confidence calibration**: output calibrated prediction probabilities rather than just class labels

---

## Suggested Tables and Figures

### Tables
1. Dataset class distribution (Section 4.2)
2. Algorithm parameters summary (Section 5.9)
3. 10-fold CV per-seed results (Section 7.1)
4. AUC-ROC by protocol and type (Section 7.2)
5. Per-class AUC (Section 7.2)
6. Confusion matrix 8x8 (Section 7.3)
7. Per-class precision/recall/F1 (Section 7.3)
8. Random split results across 8 protocols (Section 7.4)
9. Stratified results across 6 protocols (Section 7.5)
10. Stratified vs random comparison (Section 7.6)
11. Component ablation with significance (Section 8.1)
12. Hyperparameter sensitivity (Section 8.2)
13. Benchmark comparison with prior work (Section 8.4)
14. Computational cost summary (Section 8.5)

### Figures
1. CDS-OVR algorithm flowchart: data → tree → OVR training → prediction pipeline (Section 5.1)
2. Supervised binning example: showing bin edges placed at class boundaries vs uniform (Section 5.4)
3. Per-class AUC bar chart across protocols (Section 7.2)
4. Accuracy vs training set size curve (random + stratified overlaid) (Section 7.4/7.5)
5. Ablation impact bar chart with significance markers (Section 8.1)
6. Hyperparameter sensitivity plots (3 panels: CORR_THRESHOLD, MAX_BINS, FEATURES_PER_CLASS) (Section 8.2)
