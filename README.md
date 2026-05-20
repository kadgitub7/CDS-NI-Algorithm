# CDS-NI-Algorithm
Implementing and improving the CDS NI Algorithm here: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8930480

## Project Structure

```
CDS-NI-Algorithm/
│
├── CDS_Paper_Algorithms.py          # Algorithm 1: CDS Decision Tree (foundation)
├── Algorithm1_forcedBranch.py       # Algorithm 1 variant: forced sex branching
├── CDS_Paper_Algorithms_ForcedBranchingM1.py  # Algorithm 1 variant
├── Algorithm2.py                    # Algorithm 2: Perceptor & Executive Training
├── Algorithm2_2.py                  # Algorithm 2 variant
├── Algorithm3.py                    # Algorithm 3: Executive Actions Refining
├── Algorithm4.py                    # Algorithm 4: User-Health Prediction
├── Algorithm4_NoNodeFilter.py       # Algorithm 4 variant
├── fairness_config.py               # Global fairness configuration flags
├── dataplot.py                      # Feature distribution visualization
│
├── data/                            # Datasets
│   ├── arrhythmia.data              # UCI Arrhythmia dataset (452 users, 279 features)
│   ├── arrhythmia.names             # Dataset feature descriptions
│   ├── arrhythmia_augmented.data    # Augmented dataset (952 rows)
│   └── wrongUserData.txt            # Misclassified user diagnostic output
│
├── extensions/                      # Fairness and bias mitigation modules
│   ├── adversarial_debiasing.py     # Adversarial debiasing (post-processing)
│   ├── augmentation_strategies.py   # Data augmentation for LOOCV pipeline
│   ├── Fairness_EqualizedOdds.py    # Equalized odds post-processing
│   ├── Algorithm4_FairnessIntegration.py  # Fairness integration for Algorithm 4
│   └── example_equalized_odds_demo.py     # Demo script
│
├── tests/                           # Test suites
│   ├── test_fairness_implementation.py    # Equalized odds unit tests
│   └── FATests/
│       ├── run_all_tests.py         # Comprehensive algorithm test runner
│       └── results.txt              # Test results
│
├── analysis/                        # Gender analysis and reporting scripts
│   ├── createAugmentedDataset.py    # Generates arrhythmia_augmented.data
│   ├── dataAnalysis.py              # Gender-stratified missingness analysis
│   ├── dataAugmentation.py          # SMOTE augmentation experiments
│   ├── featureImportanceByGender.py # Per-gender feature importance
│   ├── machineLearningImplementation.py  # Baseline ML models
│   ├── plot_algorithm4_results.py   # Confusion matrices and error plots
│   ├── reproducibilityReport.py     # Reproducibility analysis
│   └── technicalReport.py           # Technical report generation
│
├── output/                          # Generated results (not source code)
│   ├── csv/                         # All generated CSV files
│   ├── graphs/                      # Confusion matrices, error plots
│   ├── featureGraphs/               # Per-feature distribution PNGs
│   └── reports/                     # Text reports
│
└── docs/                            # Documentation and papers
    ├── papers/                      # Reference papers (PDFs)
    └── *.md                         # Design docs, guides, summaries
```

## Quick Start

```bash
# Run Algorithm 1 (decision tree construction)
python CDS_Paper_Algorithms.py

# Run full pipeline with LOOCV (Algorithm 1 -> 2 -> 3 -> 4)
python Algorithm4.py

# Run tests
python tests/FATests/run_all_tests.py
```
