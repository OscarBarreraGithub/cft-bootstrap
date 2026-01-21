"""
Tests for mixed correlator bootstrap implementation.

Tests cover:
1. MixedCrossingVector F-vector computation
2. TwoCorrelatorBootstrapSolver
3. MixedCorrelatorBootstrapSolver
4. Comparison with single correlator bounds
"""

import numpy as np
import sys
import os
import time

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create a dummy pytest module for decorators
    class DummyPytest:
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    return func
                return decorator
            @staticmethod
            def slow(func):
                return func
    pytest = DummyPytest()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixed_correlator_blocks import MixedCrossingVector, test_mixed_crossing_vectors
from mixed_correlator_bootstrap import (
    TwoCorrelatorBootstrapSolver,
    MixedCorrelatorBootstrapSolver,
    compare_single_vs_mixed
)
from taylor_conformal_blocks import TaylorCrossingVector, HighOrderGapBootstrapSolver

# Check for CVXPY
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


class TestMixedCrossingVector:
    """Tests for MixedCrossingVector F-vector computation."""

    def test_ssee_reduces_to_ssss_equal_dims(self):
        """
        When Delta_epsilon = Delta_sigma, ssee should equal ssss.

        This is a critical consistency check: when the external dimensions
        are equal, the ssee crossing function has the same prefactors as ssss.
        """
        ds = 0.518
        cross = MixedCrossingVector(ds, ds, max_deriv=11)

        for delta in [0, 1.5, 2.0, 3.0, 5.0]:
            F_ssss = cross.build_F_vector_ssss(delta)
            F_ssee = cross.build_F_vector_ssee(delta)

            # Should match to high precision
            np.testing.assert_allclose(
                F_ssss, F_ssee, rtol=1e-6,
                err_msg=f"ssee != ssss at delta={delta} when external dims equal"
            )

    def test_eeee_matches_taylor_crossing_vector(self):
        """
        The eeee F-vector should match TaylorCrossingVector with Delta_epsilon.

        eeee is just the ssss formula with external dimension = Delta_epsilon.
        """
        ds = 0.518
        de = 1.41
        cross = MixedCrossingVector(ds, de, max_deriv=11)

        # Direct construction with Delta_epsilon as external
        direct_eeee = TaylorCrossingVector(de, max_deriv=11)

        for delta in [0, 1.5, 2.0, 3.0, 5.0]:
            F_mixed = cross.build_F_vector_eeee(delta)
            F_direct = direct_eeee.build_F_vector(delta)

            np.testing.assert_allclose(
                F_mixed, F_direct, rtol=1e-10,
                err_msg=f"eeee mismatch at delta={delta}"
            )

    def test_ssss_matches_taylor_crossing_vector(self):
        """
        The ssss F-vector should match TaylorCrossingVector with Delta_sigma.
        """
        ds = 0.518
        de = 1.41
        cross = MixedCrossingVector(ds, de, max_deriv=11)

        direct_ssss = TaylorCrossingVector(ds, max_deriv=11)

        for delta in [0, 1.5, 2.0, 3.0]:
            F_mixed = cross.build_F_vector_ssss(delta)
            F_direct = direct_ssss.build_F_vector(delta)

            np.testing.assert_allclose(
                F_mixed, F_direct, rtol=1e-10,
                err_msg=f"ssss mismatch at delta={delta}"
            )

    def test_f_vectors_finite(self):
        """All F-vectors should be finite (no NaN or inf)."""
        cross = MixedCrossingVector(0.518, 1.41, max_deriv=21)

        for delta in [0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0]:
            F_ssss = cross.build_F_vector_ssss(delta)
            F_ssee = cross.build_F_vector_ssee(delta)
            F_eeee = cross.build_F_vector_eeee(delta)

            assert np.all(np.isfinite(F_ssss)), f"F_ssss not finite at delta={delta}"
            assert np.all(np.isfinite(F_ssee)), f"F_ssee not finite at delta={delta}"
            assert np.all(np.isfinite(F_eeee)), f"F_eeee not finite at delta={delta}"

    def test_f_vector_shape(self):
        """F-vectors should have correct shape."""
        for max_deriv in [5, 11, 21]:
            cross = MixedCrossingVector(0.518, 1.41, max_deriv=max_deriv)
            expected_len = (max_deriv + 1) // 2

            assert cross.n_constraints == expected_len

            F_ssss = cross.build_F_vector_ssss(1.5)
            F_ssee = cross.build_F_vector_ssee(1.5)
            F_eeee = cross.build_F_vector_eeee(1.5)

            assert len(F_ssss) == expected_len
            assert len(F_ssee) == expected_len
            assert len(F_eeee) == expected_len

    def test_ssee_different_from_ssss_at_ising(self):
        """
        At the Ising point, ssee should differ from ssss (different external dims).
        """
        cross = MixedCrossingVector(0.518, 1.41, max_deriv=11)

        # For a non-identity operator
        delta = 2.0
        F_ssss = cross.build_F_vector_ssss(delta)
        F_ssee = cross.build_F_vector_ssee(delta)

        # They should be different (since external dims differ)
        assert not np.allclose(F_ssss, F_ssee, rtol=0.1), \
            "ssss and ssee should differ when external dims are different"

    def test_cache_consistency(self):
        """Cached F-vectors should match fresh computations."""
        cross = MixedCrossingVector(0.518, 1.41, max_deriv=11)

        # First call (computed)
        F1 = cross.build_F_vector_ssee(2.0)

        # Second call (should be cached)
        F2 = cross.build_F_vector_ssee(2.0)

        # Should be identical (same object)
        assert np.array_equal(F1, F2)

        # Clear cache and recompute
        cross.clear_cache()
        F3 = cross.build_F_vector_ssee(2.0)

        # Should still be equal (same computation)
        np.testing.assert_allclose(F1, F3, rtol=1e-14)


@pytest.mark.skipif(not HAS_CVXPY, reason="CVXPY not installed")
class TestTwoCorrelatorBootstrap:
    """Tests for TwoCorrelatorBootstrapSolver."""

    def test_ising_point_low_gap_allowed(self):
        """Ising point with low gap should be allowed."""
        solver = TwoCorrelatorBootstrapSolver(d=3, max_deriv=11)

        # Low gap should be allowed
        excluded = solver.is_excluded(0.518, 1.41, 1.8, n_samples=100)
        assert not excluded, "Ising point with Delta_eps'=1.8 should be allowed"

    def test_ising_point_high_gap_excluded(self):
        """Ising point with very high gap should be excluded."""
        solver = TwoCorrelatorBootstrapSolver(d=3, max_deriv=11)

        # Very high gap should be excluded
        excluded = solver.is_excluded(0.518, 1.41, 6.0, n_samples=100)
        assert excluded, "Ising point with Delta_eps'=6.0 should be excluded"

    def test_bound_in_reasonable_range(self):
        """The bound should be in a reasonable range."""
        solver = TwoCorrelatorBootstrapSolver(d=3, max_deriv=11)

        bound = solver.find_delta_epsilon_prime_bound(
            0.518, 1.41, tolerance=0.1
        )

        # Bound should be between 2.0 and 5.0 (roughly)
        assert 2.0 < bound < 5.0, f"Bound {bound} outside expected range [2.0, 5.0]"

    def test_monotonicity_in_gap(self):
        """Exclusion should be monotonic: if gap G is excluded, G+eps also excluded."""
        solver = TwoCorrelatorBootstrapSolver(d=3, max_deriv=11)

        # Find approximate bound
        bound = solver.find_delta_epsilon_prime_bound(
            0.518, 1.41, tolerance=0.2
        )

        # Slightly above bound should be excluded
        excluded_above = solver.is_excluded(0.518, 1.41, bound + 0.5, n_samples=100)
        assert excluded_above, "Gap above bound should be excluded"

        # Slightly below bound should be allowed
        excluded_below = solver.is_excluded(0.518, 1.41, bound - 0.5, n_samples=100)
        assert not excluded_below, "Gap below bound should be allowed"


@pytest.mark.skipif(not HAS_CVXPY, reason="CVXPY not installed")
class TestMixedCorrelatorBootstrap:
    """Tests for full MixedCorrelatorBootstrapSolver."""

    def test_ising_point_basic(self):
        """Basic test at Ising point."""
        solver = MixedCorrelatorBootstrapSolver(d=3, max_deriv=11)

        # Low gap should be allowed
        excluded = solver.is_excluded(0.518, 1.41, 2.0, n_samples=50)
        assert not excluded, "Ising point with Delta_eps'=2.0 should be allowed"

    @pytest.mark.slow
    def test_bound_computation(self):
        """Test bound computation with matrix SDP."""
        solver = MixedCorrelatorBootstrapSolver(d=3, max_deriv=11)

        bound = solver.find_delta_epsilon_prime_bound(
            0.518, 1.41, tolerance=0.2
        )

        # Bound should be reasonable
        assert 2.0 < bound < 6.0, f"Bound {bound} outside expected range"


@pytest.mark.skipif(not HAS_CVXPY, reason="CVXPY not installed")
class TestComparison:
    """Tests comparing single vs mixed correlator bounds."""

    def test_two_correlator_vs_single(self):
        """Two-correlator bound should be >= single correlator bound."""
        single_solver = HighOrderGapBootstrapSolver(d=3, max_deriv=11)
        two_solver = TwoCorrelatorBootstrapSolver(d=3, max_deriv=11)

        single_bound = single_solver.find_delta_epsilon_prime_bound(
            0.518, 1.41, tolerance=0.1
        )
        two_bound = two_solver.find_delta_epsilon_prime_bound(
            0.518, 1.41, tolerance=0.1
        )

        # Two-correlator should give same or higher bound
        # (more constraints = can only exclude more configurations)
        # Note: numerically they might be very close
        assert two_bound >= single_bound - 0.3, \
            f"Two-correlator bound ({two_bound}) should be >= single ({single_bound})"

    def test_compare_function(self):
        """Test the compare_single_vs_mixed function."""
        results = compare_single_vs_mixed(
            delta_sigma=0.518,
            delta_epsilon=1.41,
            max_deriv=11,
            tolerance=0.1,
            verbose=False
        )

        assert 'single_correlator' in results
        assert 'two_correlator' in results
        assert isinstance(results['single_correlator'], float)
        assert isinstance(results['two_correlator'], float)


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Running Mixed Correlator Bootstrap Tests")
    print("=" * 70)

    # Test 1: Crossing vectors
    print("\n1. Testing MixedCrossingVector:")
    print("-" * 40)
    test_mixed_crossing_vectors()

    if not HAS_CVXPY:
        print("\n[SKIP] CVXPY not installed - skipping solver tests")
        return

    # Test 2: Two-correlator solver
    print("\n2. Testing TwoCorrelatorBootstrapSolver:")
    print("-" * 40)

    solver = TwoCorrelatorBootstrapSolver(d=3, max_deriv=11)
    print(f"   Constraints: {solver.n_constraints}")

    t0 = time.time()
    excluded_low = solver.is_excluded(0.518, 1.41, 2.0, n_samples=100)
    excluded_high = solver.is_excluded(0.518, 1.41, 5.0, n_samples=100)
    t1 = time.time()

    print(f"   Delta_eps' = 2.0: {'EXCLUDED' if excluded_low else 'ALLOWED'}")
    print(f"   Delta_eps' = 5.0: {'EXCLUDED' if excluded_high else 'ALLOWED'}")
    print(f"   Time: {t1-t0:.2f}s")

    # Test 3: Find bound
    print("\n3. Finding two-correlator bound:")
    print("-" * 40)

    t0 = time.time()
    bound = solver.find_delta_epsilon_prime_bound(0.518, 1.41, tolerance=0.1)
    t1 = time.time()

    print(f"   Bound: Delta_eps' <= {bound:.3f}")
    print(f"   Time: {t1-t0:.2f}s")

    # Test 4: Compare with single correlator
    print("\n4. Comparison with single correlator:")
    print("-" * 40)

    single_solver = HighOrderGapBootstrapSolver(d=3, max_deriv=11)
    single_bound = single_solver.find_delta_epsilon_prime_bound(
        0.518, 1.41, tolerance=0.1
    )

    print(f"   Single correlator: {single_bound:.3f}")
    print(f"   Two correlator:    {bound:.3f}")
    print(f"   Improvement:       {bound - single_bound:+.3f}")
    print(f"   Reference:         ~3.8")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
