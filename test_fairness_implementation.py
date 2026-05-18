"""
================================================================================
test_fairness_implementation.py – Validation & Testing of Equalized Odds
================================================================================

PURPOSE
-------
Comprehensive test suite for the fairness post-processing implementation.
Tests:
  1. ROC curve computation
  2. Convex hull detection
  3. Loss optimization
  4. Fair threshold application
  5. Fairness metrics computation
  6. Integration with Algorithm 4

RUN AS
------
    python test_fairness_implementation.py

================================================================================
"""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Fairness_EqualizedOdds import (
    ROCCurve,
    compute_roc_curve,
    compute_convex_hull,
    compute_loss,
    compute_bayes_optimal_predictor,
    apply_equalized_odds_thresholds,
)

# ─────────────────────────────────────────────────────────────────────────────
# TEST UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

class TestResult:
    """Record of one test result."""
    def __init__(self, test_name: str):
        self.name = test_name
        self.passed = False
        self.error = None
        self.message = ""

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        result = f"{status}: {self.name}"
        if self.message:
            result += f"\n  {self.message}"
        if self.error:
            result += f"\n  Error: {self.error}"
        return result


def run_test(test_name: str, test_fn) -> TestResult:
    """Run a single test."""
    result = TestResult(test_name)
    try:
        test_fn(result)
        result.passed = True
    except AssertionError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"Exception: {e}"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: ROC CURVE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def test_roc_curve_basic(result: TestResult):
    """Test basic ROC curve computation."""
    # Synthetic data: perfect separation
    scores = np.array([0.1, 0.2, 0.8, 0.9])      # scores for users
    labels = np.array([1,   1,   2,   2])         # 1=healthy, 2=diseased

    roc = compute_roc_curve(scores, labels, "Test", n_points=10)

    assert len(roc.points) > 0, "ROC curve should have points"
    assert roc.n_positive == 2, "Should detect 2 diseased users"
    assert roc.n_negative == 2, "Should detect 2 healthy users"

    result.message = f"Generated {len(roc.points)} ROC points"


def test_roc_curve_extremes(result: TestResult):
    """Test ROC curve extreme thresholds."""
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([1,   2,   2])

    roc = compute_roc_curve(scores, labels, "Test", n_points=10)

    # Find extreme points
    max_threshold = max(p.threshold for p in roc.points)
    min_threshold = min(p.threshold for p in roc.points)

    # At threshold = -inf: all predictions = unhealthy
    point_all_positive = [p for p in roc.points if np.isinf(p.threshold) and p.threshold < 0]
    assert len(point_all_positive) > 0, "Should have point at -inf threshold"

    result.message = "Extreme thresholds handled correctly"


def test_roc_curve_tpr_fpr_bounds(result: TestResult):
    """Test that TPR and FPR are in [0, 1]."""
    np.random.seed(42)
    scores = np.random.uniform(0, 1, 100)
    labels = np.random.choice([1, 2], size=100)

    roc = compute_roc_curve(scores, labels, "Test", n_points=50)

    for point in roc.points:
        assert 0 <= point.tpr <= 1, f"TPR {point.tpr} out of bounds"
        assert 0 <= point.fpr <= 1, f"FPR {point.fpr} out of bounds"

    result.message = "All TPR/FPR values in [0, 1]"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: CONVEX HULL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def test_convex_hull_empty(result: TestResult):
    """Test convex hull with groups that have different optimal TPRs."""
    # Group A: can achieve TPR=0.8 or TPR=0.9
    scores_a = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels_a = np.array([1,   1,   1,   2,   2])

    # Group B: can only achieve TPR=0.5
    scores_b = np.array([0.4, 0.5])
    labels_b = np.array([1,   2])

    roc_a = compute_roc_curve(scores_a, labels_a, "A", n_points=20)
    roc_b = compute_roc_curve(scores_b, labels_b, "B", n_points=20)

    hull = compute_convex_hull(roc_a, roc_b)

    # Groups may not have many common TPRs, but hull should be computed
    result.message = f"Hull computed with {len(hull.hull_points)} candidate points"


def test_convex_hull_identical_groups(result: TestResult):
    """Test convex hull with identical group distributions."""
    # Both groups identical
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([1,   1,   2,   2])

    roc_a = compute_roc_curve(scores.copy(), labels.copy(), "A", n_points=20)
    roc_b = compute_roc_curve(scores.copy(), labels.copy(), "B", n_points=20)

    hull = compute_convex_hull(roc_a, roc_b)

    assert len(hull.hull_points) > 0, "Should find points with identical distributions"
    result.message = f"Found {len(hull.hull_points)} equalized-TPR points"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: LOSS OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_optimization_basic(result: TestResult):
    """Test loss optimization on convex hull."""
    np.random.seed(42)

    # Synthetic data: two groups
    scores_m = np.random.uniform(0, 1, 150)
    labels_m = np.random.choice([1, 2], size=150, p=[0.6, 0.4])

    scores_f = np.random.uniform(0, 1, 120)
    labels_f = np.random.choice([1, 2], size=120, p=[0.65, 0.35])

    roc_m = compute_roc_curve(scores_m, labels_m, "Male", n_points=50)
    roc_f = compute_roc_curve(scores_f, labels_f, "Female", n_points=50)

    hull = compute_convex_hull(roc_m, roc_f)

    if len(hull.hull_points) == 0:
        result.message = "No feasible hull (groups too different)"
        return

    loss = compute_loss(hull, baseline_threshold=0.5)

    assert loss.threshold_male > 0, "Threshold should be positive"
    assert loss.threshold_female > 0, "Threshold should be positive"
    assert 0 <= loss.tpr_equalized <= 1, "TPR should be in [0, 1]"

    result.message = (
        f"Optimal thresholds: M={loss.threshold_male:.4f}, "
        f"F={loss.threshold_female:.4f}, "
        f"Equalized TPR={loss.tpr_equalized:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: APPLY THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_thresholds_basic(result: TestResult):
    """Test threshold application to data."""
    from Fairness_EqualizedOdds import EqualizedOddsResult

    # Create synthetic data
    n = 100
    scores = np.random.uniform(0, 1, n)
    
    # Create fake data matrix with gender column
    data = np.zeros((n, 279))
    data[:50, 1] = 0.0   # First 50 are male
    data[50:, 1] = 1.0   # Last 50 are female

    # Create fake fairness result
    result_obj = EqualizedOddsResult(
        threshold_male=0.4,
        threshold_female=0.6,
        tpr_male=0.8,
        tpr_female=0.8,
        tpr_equalized=0.8,
        fpr_male=0.1,
        fpr_female=0.1,
        utility_loss=0.01,
        original_threshold=0.5,
        n_male=50,
        n_female=50,
    )

    predictions = apply_equalized_odds_thresholds(scores, data, result_obj)

    assert len(predictions) == n, "Should have predictions for all users"
    assert np.all((predictions == 0) | (predictions == 1)), "Predictions should be binary"

    result.message = f"Applied thresholds: {np.sum(predictions)} users predicted unhealthy"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline(result: TestResult):
    """Test full pipeline: scores → ROCs → hull → loss → predictions."""
    np.random.seed(42)

    # Synthetic dataset
    n_male, n_female = 200, 180
    n_total = n_male + n_female

    # Male users
    scores_m = np.concatenate([
        np.random.normal(0.2, 0.1, 160),  # 160 healthy
        np.random.normal(0.7, 0.1, 40),   # 40 diseased
    ])
    labels_m = np.concatenate([np.ones(160), np.ones(40) * 2])

    # Female users
    scores_f = np.concatenate([
        np.random.normal(0.25, 0.1, 150),  # 150 healthy
        np.random.normal(0.65, 0.1, 30),   # 30 diseased
    ])
    labels_f = np.concatenate([np.ones(150), np.ones(30) * 2])

    # Full data
    scores = np.concatenate([scores_m, scores_f])
    labels = np.concatenate([labels_m, labels_f])

    # Create data matrix with gender
    data = np.zeros((n_total, 279))
    data[:n_male, 1] = 0.0
    data[n_male:, 1] = 1.0

    # Run full pipeline
    fairness_result = compute_bayes_optimal_predictor(
        scores, labels, data,
        baseline_threshold=0.5,
        verbose=False
    )

    assert fairness_result is not None, "Should return result"
    assert fairness_result.threshold_male > 0, "Male threshold should be positive"
    assert fairness_result.threshold_female > 0, "Female threshold should be positive"
    assert abs(fairness_result.tpr_male - fairness_result.tpr_female) < 0.01, (
        "TPRs should be equal (within tolerance)"
    )

    result.message = (
        f"Full pipeline successful\n"
        f"  Male TPR: {fairness_result.tpr_male:.4f}\n"
        f"  Female TPR: {fairness_result.tpr_female:.4f}\n"
        f"  Utility loss: {fairness_result.utility_loss:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

def test_single_class_group(result: TestResult):
    """Test ROC curve with group containing only one class."""
    # Group with only healthy users (no diseased)
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1,   1,   1])

    roc = compute_roc_curve(scores, labels, "Healthy-only", n_points=10)

    # Should still create ROC curve, but with n_positive=0
    assert roc.n_positive == 0, "Should have no positive users"
    assert roc.n_negative == 3, "Should have 3 negative users"

    result.message = "Single-class group handled gracefully"


def test_large_dataset(result: TestResult):
    """Test with larger dataset."""
    np.random.seed(42)

    n = 5000
    scores = np.random.uniform(0, 1, n)
    labels = np.random.choice([1, 2], size=n, p=[0.7, 0.3])

    data = np.zeros((n, 279))
    data[:, 1] = np.where(np.arange(n) < n // 2, 0.0, 1.0)

    fairness_result = compute_bayes_optimal_predictor(
        scores, labels, data,
        baseline_threshold=0.5,
        verbose=False
    )

    assert fairness_result is not None
    result.message = f"Processed {n} users successfully"


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run all tests."""
    print("=" * 80)
    print("FAIRNESS IMPLEMENTATION TEST SUITE")
    print("=" * 80)

    tests = [
        ("ROC Curve – Basic", test_roc_curve_basic),
        ("ROC Curve – Extreme Thresholds", test_roc_curve_extremes),
        ("ROC Curve – TPR/FPR Bounds", test_roc_curve_tpr_fpr_bounds),
        ("Convex Hull – Different Groups", test_convex_hull_empty),
        ("Convex Hull – Identical Groups", test_convex_hull_identical_groups),
        ("Loss Optimization – Basic", test_loss_optimization_basic),
        ("Apply Thresholds – Basic", test_apply_thresholds_basic),
        ("Full Pipeline", test_full_pipeline),
        ("Edge Case – Single Class", test_single_class_group),
        ("Edge Case – Large Dataset", test_large_dataset),
    ]

    results = []
    for test_name, test_fn in tests:
        print(f"\nRunning: {test_name}...")
        result = run_test(test_name, test_fn)
        results.append(result)
        print(f"  {result}")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  • {r.name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
