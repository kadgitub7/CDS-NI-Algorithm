"""
================================================================================
Machine Learning Implementation: Arrhythmia Detection
Neural Network Model for Health Status Classification
================================================================================

DATASET: Arrhythmia Database
- 452 instances with 279 features
- Classes: 1 (Healthy), 2-16 (Unhealthy - arrhythmias)
- Task: Binary classification (Healthy vs Unhealthy)

MODEL: Simple 3-Layer Neural Network
- Input Layer: 279 features
- Hidden Layer 1: 64 neurons with ReLU activation
- Hidden Layer 2: 32 neurons with ReLU activation
- Output Layer: 2 neurons (softmax) for Healthy/Unhealthy

VALIDATION METHODS:
1. Leave-One-Out Cross-Validation (LOOCV) - Primary method
2. 10-Fold Cross-Validation (optional, commented)

================================================================================
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – DATA LOADING AND PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_arrhythmia_data(data_path: str, names_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load arrhythmia dataset from CSV file.
    
    Parameters
    ----------
    data_path : str
        Path to arrhythmia.data file (279 features + 1 label)
    names_path : str
        Path to arrhythmia.names file (for documentation)
    
    Returns
    -------
    data : np.ndarray of shape (n_samples, 279)
        Feature matrix (missing values '?' are replaced with NaN)
    labels : np.ndarray of shape (n_samples,)
        Class labels (1=Healthy, 2-16=Arrhythmia)
    """
    # Load raw data
    raw_data = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by commas, handle '?' as missing
            values = [np.nan if v == '?' else float(v) for v in line.split(',')]
            raw_data.append(values)
    
    data_array = np.array(raw_data)
    
    # Last column is label (class)
    labels = data_array[:, -1].astype(int)
    features = data_array[:, :-1]
    
    print(f"✓ Loaded arrhythmia dataset:")
    print(f"  - Samples: {features.shape[0]}")
    print(f"  - Features: {features.shape[1]}")
    print(f"  - Classes: {np.unique(labels)}")
    print(f"  - Missing values: {np.isnan(features).sum()} ({np.isnan(features).sum()/features.size*100:.2f}%)")
    
    return features, labels


def preprocess_data(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess features for neural network.
    
    Steps:
    1. Handle missing values (NaN) by replacing with column mean
    2. Map multi-class labels to binary (Healthy/Unhealthy)
    3. Normalize features to zero mean and unit variance
    
    Parameters
    ----------
    features : np.ndarray of shape (n_samples, 279)
    labels : np.ndarray of shape (n_samples,)
    
    Returns
    -------
    features_clean : np.ndarray, preprocessed
    labels_mapped : np.ndarray, mapped to 2 classes (0=Healthy, 1=Unhealthy)
    """
    # Handle missing values: replace with column mean
    for col_idx in range(features.shape[1]):
        col = features[:, col_idx]
        col_mean = np.nanmean(col)
        col[np.isnan(col)] = col_mean
    
    # Map original labels to binary classes:
    # 1 -> 0 (Healthy)
    # 2-16 -> 1 (Unhealthy - all arrhythmias including unclassified)
    labels_mapped = np.zeros_like(labels)
    labels_mapped[labels == 1] = 0  # Healthy
    labels_mapped[(labels >= 2) & (labels <= 16)] = 1  # Unhealthy
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    print(f"✓ Preprocessing complete:")
    print(f"  - Missing values handled (replaced with column mean)")
    print(f"  - Labels mapped: 1->{0} Healthy, 2-16->{1} Unhealthy")
    print(f"  - Class distribution: {np.bincount(labels_mapped)}")
    print(f"  - Features normalized (StandardScaler)")
    
    return features_normalized, labels_mapped


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – NEURAL NETWORK MODEL
# ─────────────────────────────────────────────────────────────────────────────

def create_neural_network() -> MLPClassifier:
    """
    Create a simple 3-layer neural network.
    
    Architecture:
    - Input: 279 features
    - Hidden Layer 1: 64 neurons, ReLU activation
    - Hidden Layer 2: 32 neurons, ReLU activation
    - Output: 2 neurons, Softmax (binary classification: Healthy/Unhealthy)
    
    Returns
    -------
    model : MLPClassifier
        Untrained neural network
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),      # 2 hidden layers: 64 and 32 neurons
        activation='relu',                  # ReLU activation
        solver='adam',                      # Adam optimizer
        learning_rate='adaptive',           # Adaptive learning rate
        learning_rate_init=0.001,           # Initial learning rate
        max_iter=500,                       # Max iterations
        batch_size=32,                      # Batch size for gradient descent
        random_state=42,
        verbose=False,
        early_stopping=True,                # Stop if validation score plateaus
        validation_fraction=0.1,            # 10% for validation during training
        n_iter_no_change=10,                # Stop after 10 iterations without improvement
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – LOOCV VALIDATION (PRIMARY METHOD)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelPerformance:
    """Store model performance metrics."""
    accuracy: float
    sensitivity: float      # True Positive Rate (Unhealthy) 
    specificity: float      # True Negative Rate (Healthy)
    confusion_matrix: np.ndarray
    predictions: List[int]
    true_labels: List[int]


def run_loocv(features: np.ndarray, labels: np.ndarray, 
              verbose: bool = True) -> ModelPerformance:
    """
    Leave-One-Out Cross-Validation (LOOCV).
    
    For each sample i (1 to N):
      1. Train model on all samples EXCEPT i
      2. Test model on sample i
      3. Record prediction
    
    This is the most rigorous validation method but computationally expensive.
    
    Parameters
    ----------
    features : np.ndarray of shape (n_samples, 279)
    labels : np.ndarray of shape (n_samples,)
    verbose : bool, print progress
    
    Returns
    -------
    performance : ModelPerformance
    """
    n_samples = features.shape[0]
    predictions = []
    true_labels_list = []
    
    print(f"\n{'='*70}")
    print(f"LOOCV Validation (Leave-One-Out Cross-Validation)")
    print(f"{'='*70}")
    print(f"Total samples: {n_samples}")
    print(f"Training iterations: {n_samples} (train on N-1, test on 1)")
    print()
    
    for i in range(n_samples):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples} samples processed...")
        
        # Split: training set (all except i), test set (sample i)
        train_indices = np.concatenate([np.arange(0, i), np.arange(i+1, n_samples)])
        test_index = i
        
        X_train = features[train_indices]
        y_train = labels[train_indices]
        X_test = features[test_index].reshape(1, -1)
        y_test = labels[test_index]
        
        # Create and train model
        model = create_neural_network()
        model.fit(X_train, y_train)
        
        # Predict on single test sample
        pred = model.predict(X_test)[0]
        predictions.append(pred)
        true_labels_list.append(y_test)
    
    # Compute metrics
    predictions = np.array(predictions)
    true_labels_arr = np.array(true_labels_list)
    
    accuracy = np.mean(predictions == true_labels_arr)
    
    # Sensitivity: Recall for class 1 (Unhealthy)
    unhealthy_indices = true_labels_arr == 1
    sensitivity = np.sum((predictions[unhealthy_indices] == 1)) / np.sum(unhealthy_indices) if np.sum(unhealthy_indices) > 0 else 0.0
    
    # Specificity: Recall for class 0 (Healthy)
    healthy_indices = true_labels_arr == 0
    specificity = np.sum((predictions[healthy_indices] == 0)) / np.sum(healthy_indices) if np.sum(healthy_indices) > 0 else 0.0
    
    cm = confusion_matrix(true_labels_arr, predictions, labels=[0, 1])
    
    print(f"✓ LOOCV Complete!")
    print(f"  - Accuracy: {accuracy*100:.2f}%")
    print(f"  - Sensitivity (Unhealthy detection): {sensitivity*100:.2f}%")
    print(f"  - Specificity (Healthy detection): {specificity*100:.2f}%")
    
    return ModelPerformance(
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        confusion_matrix=cm,
        predictions=predictions.tolist(),
        true_labels=true_labels_arr.tolist()
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – 10-FOLD CROSS-VALIDATION (OPTIONAL, COMMENTED)
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold_cv(features: np.ndarray, labels: np.ndarray, 
                 n_splits: int = 10, verbose: bool = True) -> ModelPerformance:
    """
    k-Fold Cross-Validation (default: 10-fold).
    
    For each fold k (1 to k):
      1. Split data into k-th test fold (1/k of data) and training (k-1/k of data)
      2. Train model on training fold
      3. Test model on test fold
      4. Record predictions
    
    Much faster than LOOCV but less thorough.
    
    Parameters
    ----------
    features : np.ndarray of shape (n_samples, 279)
    labels : np.ndarray of shape (n_samples,)
    n_splits : int, number of folds (default 10)
    verbose : bool, print progress
    
    Returns
    -------
    performance : ModelPerformance
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_predictions = []
    all_true_labels = []
    fold_accuracies = []
    
    print(f"\n{'='*70}")
    print(f"{n_splits}-Fold Cross-Validation")
    print(f"{'='*70}")
    print(f"Total samples: {features.shape[0]}")
    print(f"Fold size: ~{features.shape[0]//n_splits} samples per fold")
    print(f"Training iterations: {n_splits}")
    print()
    
    fold_num = 1
    for train_indices, test_indices in kfold.split(features):
        if verbose:
            print(f"  Fold {fold_num}/{n_splits}...", end=" ")
        
        X_train = features[train_indices]
        y_train = labels[train_indices]
        X_test = features[test_indices]
        y_test = labels[test_indices]
        
        # Create and train model
        model = create_neural_network()
        model.fit(X_train, y_train)
        
        # Predict on test fold
        predictions = model.predict(X_test)
        fold_accuracy = accuracy_score(y_test, predictions)
        fold_accuracies.append(fold_accuracy)
        
        all_predictions.extend(predictions)
        all_true_labels.extend(y_test)
        
        if verbose:
            print(f"Accuracy: {fold_accuracy*100:.2f}%")
        
        fold_num += 1
    
    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    accuracy = np.mean(all_predictions == all_true_labels)
    
    # Sensitivity: Recall for class 1 (Unhealthy)
    unhealthy_indices = all_true_labels == 1
    sensitivity = np.sum((all_predictions[unhealthy_indices] == 1)) / np.sum(unhealthy_indices) if np.sum(unhealthy_indices) > 0 else 0.0
    
    # Specificity: Recall for class 0 (Healthy)
    healthy_indices = all_true_labels == 0
    specificity = np.sum((all_predictions[healthy_indices] == 0)) / np.sum(healthy_indices) if np.sum(healthy_indices) > 0 else 0.0
    
    cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1])
    
    print(f"\n✓ {n_splits}-Fold CV Complete!")
    print(f"  - Mean Accuracy: {accuracy*100:.2f}% (±{np.std(fold_accuracies)*100:.2f}%)")
    print(f"  - Sensitivity (Unhealthy detection): {sensitivity*100:.2f}%")
    print(f"  - Specificity (Healthy detection): {specificity*100:.2f}%")
    print(f"  - Fold accuracies: {[f'{x*100:.2f}%' for x in fold_accuracies]}")
    
    return ModelPerformance(
        accuracy=accuracy,
        sensitivity=sensitivity,
        specificity=specificity,
        confusion_matrix=cm,
        predictions=all_predictions.tolist(),
        true_labels=all_true_labels.tolist()
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – RESULTS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def print_detailed_results(performance: ModelPerformance, method_name: str):
    """
    Print detailed performance metrics and confusion matrix.
    
    Parameters
    ----------
    performance : ModelPerformance
    method_name : str, name of validation method (e.g., "LOOCV", "10-Fold CV")
    """
    print(f"\n{'='*70}")
    print(f"DETAILED RESULTS: {method_name}")
    print(f"{'='*70}\n")
    
    print("Confusion Matrix:")
    print("                  Predicted")
    print("              Healthy  Unhealthy")
    print(f"Healthy           {performance.confusion_matrix[0,0]:3d}        {performance.confusion_matrix[0,1]:3d}")
    print(f"Unhealthy         {performance.confusion_matrix[1,0]:3d}        {performance.confusion_matrix[1,1]:3d}")
    
    print(f"\n{'─'*70}")
    print("Performance Metrics:")
    print(f"{'─'*70}")
    print(f"Overall Accuracy:            {performance.accuracy*100:6.2f}%")
    print(f"Sensitivity (Unhealthy):     {performance.sensitivity*100:6.2f}%")
    print(f"Specificity (Healthy):       {performance.specificity*100:6.2f}%")
    
    # Calculate additional metrics
    tp = performance.confusion_matrix[1, 1]  # True Positives (Unhealthy)
    fp = performance.confusion_matrix[0, 1]  # False Positives
    tn = performance.confusion_matrix[0, 0]  # True Negatives (Healthy)
    fn = performance.confusion_matrix[1, 0]  # False Negatives
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        print(f"Precision (Unhealthy):       {precision*100:6.2f}%")
    
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Main execution function: Load data, preprocess, run validation, print results.
    """
    # Paths
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.normpath(os.path.join(_here, ".."))
    data_path = os.path.join(_root, "data", "arrhythmia.data")
    names_path = os.path.join(_root, "data", "arrhythmia.names")
    
    print(f"\n{'='*70}")
    print(f"ARRHYTHMIA DETECTION - NEURAL NETWORK MODEL")
    print(f"{'='*70}\n")
    
    # Load and preprocess data
    features, labels = load_arrhythmia_data(data_path, names_path)
    features_clean, labels_mapped = preprocess_data(features, labels)
    
    # ─────────────────────────────────────────────────────────────────────────
    # RUN LOOCV (Primary Validation Method)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print("VALIDATION METHOD 1: LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)")
    print(f"{'#'*70}")
    loocv_performance = run_loocv(features_clean, labels_mapped, verbose=True)
    print_detailed_results(loocv_performance, "LOOCV")
    
    # ─────────────────────────────────────────────────────────────────────────
    # RUN 10-FOLD CROSS-VALIDATION (Optional, Commented Below)
    # Uncomment the following lines to run 10-fold CV instead of LOOCV
    # ─────────────────────────────────────────────────────────────────────────
    
    """
    OPTIONAL: 10-FOLD CROSS-VALIDATION
    
    Uncomment the code below to run 10-fold cross-validation in addition to 
    (or instead of) LOOCV. This method is MUCH faster but less thorough.
    """
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print("VALIDATION METHOD 2: 10-FOLD CROSS-VALIDATION (OPTIONAL)")
    print(f"{'#'*70}")
    kfold_performance = run_kfold_cv(features_clean, labels_mapped, n_splits=10, verbose=True)
    print_detailed_results(kfold_performance, "10-Fold CV")
    
    # Compare results if both methods are run
    print(f"\n{'='*70}")
    print("COMPARISON: LOOCV vs 10-Fold CV")
    print(f"{'='*70}")
    print(f"LOOCV Accuracy:    {loocv_performance.accuracy*100:.2f}%")
    print(f"10-Fold Accuracy:  {kfold_performance.accuracy*100:.2f}%")
    print(f"Difference:        {abs(loocv_performance.accuracy - kfold_performance.accuracy)*100:.2f}%")
    print()
    
    print(f"\n{'='*70}")
    print("Arrhythmia Detection Model - Execution Complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
