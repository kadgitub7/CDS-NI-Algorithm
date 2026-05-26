"""
================================================================================
test_fairness_implementation.py – Validation & Testing of Equalized Odds
================================================================================

PURPOSE
-------
Comprehensive test suite for the fairness post-processing implementation.
Tests:
  1. ROC curve computation
  2. Convex hull intersection (both FPR and TPR equalized)
  3. Randomized threshold classifiers
  4. Fair threshold application
  5. Full pipeline integration
  6. Edge cases

RUN AS
------
    python test_fairness_implementation.py

================================================================================
"""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "extensions"))

from Fairness_EqualizedOdds import (
    ROCCurve,
    RandomizedThreshold,
    EqualizedOddsResult,
    compute_roc_curve,
    compute_roc_curves_by_gender,
    compute_bayes_optimal_predictor,
    apply_equalized_odds_thresholds,
    _roc_to_convex_hull_points,
    _find_feasible_region,
    _realize_operating_point,
    _apply_randomized_threshold,
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
        status = "PASS" if self.passed else "FAIL"
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
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([1,   1,   2,   2])

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
# TEST 2: CONVEX HULL AND FEASIBLE REGION
# ─────────────────────────────────────────────────────────────────────────────

def test_convex_hull_points(result: TestResult):
    """Test convex hull computation from ROC curve."""
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels = np.array([1,   1,   1,   2,   2])

    roc = compute_roc_curve(scores, labels, "Test", n_points=20)
    hull = _roc_to_convex_hull_points(roc)

    assert len(hull) > 0, "Hull should have points"
    assert all(hull[:, 1] >= hull[:, 0] - 1e-9), "Hull points should be above diagonal"

    result.message = f"Convex hull has {len(hull)} points"


def test_feasible_region_identical_groups(result: TestResult):
    """Test feasible region with identical group distributions."""
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([1,   1,   2,   2])

    roc_a = compute_roc_curve(scores.copy(), labels.copy(), "A", n_points=20)
    roc_b = compute_roc_curve(scores.copy(), labels.copy(), "B", n_points=20)

    hull_a = _roc_to_convex_hull_points(roc_a)
    hull_b = _roc_to_convex_hull_points(roc_b)
    feasible = _find_feasible_region(hull_a, hull_b)

    assert len(feasible) > 0, "Identical groups must have non-empty intersection"
    result.message = f"Found {len(feasible)} feasible (FPR, TPR) points"


def test_feasible_region_different_groups(result: TestResult):
    """Test feasible region with different group distributions."""
    scores_a = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    labels_a = np.array([1,   1,   1,   2,   2])

    scores_b = np.array([0.4, 0.5])
    labels_b = np.array([1,   2])

    roc_a = compute_roc_curve(scores_a, labels_a, "A", n_points=20)
    roc_b = compute_roc_curve(scores_b, labels_b, "B", n_points=20)

    hull_a = _roc_to_convex_hull_points(roc_a)
    hull_b = _roc_to_convex_hull_points(roc_b)
    feasible = _find_feasible_region(hull_a, hull_b)

    result.message = f"Feasible region: {len(feasible)} points (may be 0 for very different groups)"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: RANDOMIZED THRESHOLD CLASSIFIERS
# ─────────────────────────────────────────────────────────────────────────────

def test_randomized_threshold_deterministic(result: TestResult):
    """Test that deterministic thresholds are detected correctly."""
    rt = RandomizedThreshold(t_lo=0.5, t_hi=0.5, p=1.0)
    assert rt.is_deterministic, "Same t_lo and t_hi should be deterministic"

    rt2 = RandomizedThreshold(t_lo=0.3, t_hi=0.7, p=0.5)
    assert not rt2.is_deterministic, "Different thresholds with 0<p<1 should be randomized"

    result.message = "Deterministic/randomized detection correct"


def test_randomized_threshold_application(result: TestResult):
    """Test applying randomized thresholds produces binary predictions."""
    rng = np.random.default_rng(42)
    scores = np.linspace(0, 1, 100)

    rt = RandomizedThreshold(t_lo=0.3, t_hi=0.7, p=0.5)
    preds = _apply_randomized_threshold(scores, rt, rng)

    assert len(preds) == 100, "Should produce prediction for each score"
    assert np.all((preds == 0) | (preds == 1)), "Predictions should be binary"

    below = preds[scores < 0.3]
    assert np.all(below == 0), "Scores below t_lo should always predict 0"

    above = preds[scores > 0.7]
    assert np.all(above == 1), "Scores above t_hi should always predict 1"

    result.message = f"Randomized predictions: {np.sum(preds)} positive out of 100"


def test_realize_operating_point(result: TestResult):
    """Test realizing a target (FPR, TPR) as a randomized threshold."""
    np.random.seed(42)
    scores = np.concatenate([
        np.random.normal(0.3, 0.1, 80),
        np.random.normal(0.7, 0.1, 20),
    ])
    labels = np.concatenate([np.ones(80), np.ones(20) * 2])
    roc = compute_roc_curve(scores, labels, "Test", n_points=50)

    mid_point = roc.points[len(roc.points) // 2]
    rt = _realize_operating_point(roc, mid_point.fpr, mid_point.tpr)

    assert rt.t_lo <= rt.t_hi, "t_lo should be <= t_hi"
    assert 0 <= rt.p <= 1, "Mixing probability should be in [0,1]"

    result.message = f"Realized: t_lo={rt.t_lo:.4f}, t_hi={rt.t_hi:.4f}, p={rt.p:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: APPLY THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_thresholds_basic(result: TestResult):
    """Test threshold application to data (backward compat without randomized thresholds)."""
    n = 100
    np.random.seed(42)
    scores = np.random.uniform(0, 1, n)

    data = np.zeros((n, 279))
    data[:50, 1] = 0.0
    data[50:, 1] = 1.0

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


def test_apply_thresholds_with_randomization(result: TestResult):
    """Test that randomized thresholds produce reproducible results with seed."""
    n = 200
    np.random.seed(42)
    scores = np.random.uniform(0, 1, n)

    data = np.zeros((n, 279))
    data[:100, 1] = 0.0
    data[100:, 1] = 1.0

    rt_m = RandomizedThreshold(t_lo=0.3, t_hi=0.6, p=0.5)
    rt_f = RandomizedThreshold(t_lo=0.35, t_hi=0.65, p=0.4)

    result_obj = EqualizedOddsResult(
        threshold_male=0.45,
        threshold_female=0.5,
        tpr_male=0.8,
        tpr_female=0.8,
        tpr_equalized=0.8,
        fpr_male=0.15,
        fpr_female=0.15,
        utility_loss=0.01,
        original_threshold=0.5,
        n_male=100,
        n_female=100,
        randomized_threshold_male=rt_m,
        randomized_threshold_female=rt_f,
    )

    preds1 = apply_equalized_odds_thresholds(scores, data, result_obj, seed=123)
    preds2 = apply_equalized_odds_thresholds(scores, data, result_obj, seed=123)

    assert np.array_equal(preds1, preds2), "Same seed should produce same predictions"

    preds3 = apply_equalized_odds_thresholds(scores, data, result_obj, seed=456)
    # Different seeds may produce different results in the randomized region
    # (though not guaranteed for all data)

    result.message = f"Reproducible: seed=123 gave {np.sum(preds1)} positives both times"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: FULL PIPELINE – EQUALIZED ODDS (BOTH TPR AND FPR)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline(result: TestResult):
    """Test full pipeline: both TPR and FPR should be equalized."""
    np.random.seed(42)

    n_male, n_female = 200, 180
    n_total = n_male + n_female

    scores_m = np.concatenate([
        np.random.normal(0.2, 0.1, 160),
        np.random.normal(0.7, 0.1, 40),
    ])
    labels_m = np.concatenate([np.ones(160), np.ones(40) * 2])

    scores_f = np.concatenate([
        np.random.normal(0.25, 0.1, 150),
        np.random.normal(0.65, 0.1, 30),
    ])
    labels_f = np.concatenate([np.ones(150), np.ones(30) * 2])

    scores = np.concatenate([scores_m, scores_f])
    labels = np.concatenate([labels_m, labels_f])

    data = np.zeros((n_total, 279))
    data[:n_male, 1] = 0.0
    data[n_male:, 1] = 1.0

    fairness_result = compute_bayes_optimal_predictor(
        scores, labels, data,
        baseline_threshold=0.5,
        verbose=False
    )

    assert fairness_result is not None, "Should return result"

    # Equalized odds: BOTH TPR and FPR should be equal
    assert abs(fairness_result.tpr_male - fairness_result.tpr_female) < 0.01, (
        f"TPRs should be equal: male={fairness_result.tpr_male:.4f}, "
        f"female={fairness_result.tpr_female:.4f}"
    )
    assert abs(fairness_result.fpr_male - fairness_result.fpr_female) < 0.01, (
        f"FPRs should be equal: male={fairness_result.fpr_male:.4f}, "
        f"female={fairness_result.fpr_female:.4f}"
    )

    result.message = (
        f"Full pipeline successful (BOTH TPR and FPR equalized)\n"
        f"  Male TPR: {fairness_result.tpr_male:.4f}, Female TPR: {fairness_result.tpr_female:.4f}\n"
        f"  Male FPR: {fairness_result.fpr_male:.4f}, Female FPR: {fairness_result.fpr_female:.4f}\n"
        f"  Utility loss: {fairness_result.utility_loss:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

def test_single_class_group(result: TestResult):
    """Test ROC curve with group containing only one class."""
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1,   1,   1])

    roc = compute_roc_curve(scores, labels, "Healthy-only", n_points=10)

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


def test_diagnostics_contain_randomization_info(result: TestResult):
    """Test that diagnostics report whether randomization was used."""
    np.random.seed(42)

    n_male, n_female = 150, 130
    n_total = n_male + n_female

    scores_m = np.concatenate([
        np.random.normal(0.2, 0.15, 120),
        np.random.normal(0.8, 0.1, 30),
    ])
    labels_m = np.concatenate([np.ones(120), np.ones(30) * 2])

    scores_f = np.concatenate([
        np.random.normal(0.3, 0.15, 100),
        np.random.normal(0.7, 0.1, 30),
    ])
    labels_f = np.concatenate([np.ones(100), np.ones(30) * 2])

    scores = np.concatenate([scores_m, scores_f])
    labels = np.concatenate([labels_m, labels_f])

    data = np.zeros((n_total, 279))
    data[:n_male, 1] = 0.0
    data[n_male:, 1] = 1.0

    fr = compute_bayes_optimal_predictor(
        scores, labels, data,
        baseline_threshold=0.5,
        verbose=False,
    )

    assert "randomized_male" in fr.diagnostics, "Should report male randomization status"
    assert "randomized_female" in fr.diagnostics, "Should report female randomization status"
    assert "optimal_fpr" in fr.diagnostics, "Should report optimal FPR"
    assert "optimal_tpr" in fr.diagnostics, "Should report optimal TPR"

    result.message = (
        f"Diagnostics OK: randomized_male={fr.diagnostics['randomized_male']}, "
        f"randomized_female={fr.diagnostics['randomized_female']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run all tests."""
    print("=" * 80)
    print("FAIRNESS IMPLEMENTATION TEST SUITE")
    print("  Testing: Equalized Odds (Hardt et al. 2016)")
    print("  Constraint: BOTH TPR and FPR equalized across groups")
    print("=" * 80)

    tests = [
        ("ROC Curve - Basic", test_roc_curve_basic),
        ("ROC Curve - Extreme Thresholds", test_roc_curve_extremes),
        ("ROC Curve - TPR/FPR Bounds", test_roc_curve_tpr_fpr_bounds),
        ("Convex Hull - Points", test_convex_hull_points),
        ("Feasible Region - Identical Groups", test_feasible_region_identical_groups),
        ("Feasible Region - Different Groups", test_feasible_region_different_groups),
        ("Randomized Threshold - Deterministic Detection", test_randomized_threshold_deterministic),
        ("Randomized Threshold - Application", test_randomized_threshold_application),
        ("Randomized Threshold - Realize Operating Point", test_realize_operating_point),
        ("Apply Thresholds - Basic", test_apply_thresholds_basic),
        ("Apply Thresholds - Randomized Reproducibility", test_apply_thresholds_with_randomization),
        ("Full Pipeline - Equalized Odds", test_full_pipeline),
        ("Edge Case - Single Class", test_single_class_group),
        ("Edge Case - Large Dataset", test_large_dataset),
        ("Diagnostics - Randomization Info", test_diagnostics_contain_randomization_info),
    ]

    results = []
    for test_name, test_fn in tests:
        print(f"\nRunning: {test_name}...")
        r = run_test(test_name, test_fn)
        results.append(r)
        print(f"  {r}")

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\nALL TESTS PASSED")
        return 0
    else:
        print(f"\n{total - passed} TEST(S) FAILED")
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
