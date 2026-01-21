"""
Comprehensive tests for polynomial positivity constraints.

These tests verify:
1. Polynomial fitting accuracy
2. SOS constraint structure
3. Exclusion/feasibility checks
4. Bound computation
5. Comparison with discrete sampling

Run with: pytest tests/test_polynomial_positivity.py -v
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polynomial_positivity import (
    PolynomialFitter,
    PolynomialFVector,
    FittedPolynomial,
    SOSPositivityConstraint,
    PolynomialPositivitySolver,
    PolynomialPositivityGapSolver,
)
from taylor_conformal_blocks import TaylorCrossingVector


class TestPolynomialFitting:
    """Tests for polynomial fitting to F-vectors."""

    def test_chebyshev_nodes(self):
        """Test Chebyshev node generation."""
        fitter = PolynomialFitter(delta_sigma=0.518, max_deriv=11, poly_degree=10)

        nodes = fitter.chebyshev_nodes(5, 0, 1)

        # Chebyshev nodes should be in [0, 1]
        assert np.all(nodes >= 0)
        assert np.all(nodes <= 1)
        assert len(nodes) == 5

        # Nodes should be distinct
        assert len(np.unique(nodes)) == 5

    def test_chebyshev_nodes_mapping(self):
        """Test Chebyshev nodes map correctly to [a, b]."""
        fitter = PolynomialFitter(delta_sigma=0.518, max_deriv=11, poly_degree=10)

        a, b = 2.0, 10.0
        nodes = fitter.chebyshev_nodes(10, a, b)

        assert np.all(nodes >= a)
        assert np.all(nodes <= b)
        # Endpoints should be close to but not exactly a and b
        assert nodes.min() > a
        assert nodes.max() < b

    def test_polynomial_fit_basic(self):
        """Test basic polynomial fitting."""
        fitter = PolynomialFitter(delta_sigma=0.518, max_deriv=11, poly_degree=12)

        poly_F = fitter.fit_polynomial(delta_gap=1.5, delta_max=30.0)

        assert isinstance(poly_F, PolynomialFVector)
        assert poly_F.n_components == 6  # (11 + 1) // 2 = 6 constraints
        assert poly_F.max_degree == 12
        assert poly_F.delta_gap == 1.5
        assert poly_F.delta_max == 30.0

    def test_polynomial_fit_accuracy(self):
        """Test that polynomial fit is accurate."""
        fitter = PolynomialFitter(delta_sigma=0.518, max_deriv=11, poly_degree=15)

        poly_F = fitter.fit_polynomial(delta_gap=1.5, delta_max=30.0)

        # Validate fit at test points
        max_err, comp_err = fitter.validate_fit(poly_F, n_test_points=50)

        # Should have reasonable accuracy
        assert max_err < 0.01, f"Max relative error too high: {max_err}"

        # All components should be well-fitted
        assert np.all(comp_err < 0.01), f"Some components poorly fitted: {comp_err}"

    def test_polynomial_evaluate(self):
        """Test polynomial evaluation."""
        fitter = PolynomialFitter(delta_sigma=0.518, max_deriv=11, poly_degree=12)
        poly_F = fitter.fit_polynomial(delta_gap=1.5, delta_max=30.0)

        # Evaluate at a test point
        delta_test = 5.0
        F_approx = poly_F.evaluate(delta_test)

        assert len(F_approx) == poly_F.n_components

        # Compare with exact F-vector
        cross = TaylorCrossingVector(0.518, 11)
        F_exact = cross.build_F_vector(delta_test)

        # Should be reasonably close
        rel_error = np.abs(F_approx - F_exact) / (np.abs(F_exact) + 1e-10)
        assert np.all(rel_error < 0.05), f"Evaluation error too high: {rel_error}"

    def test_coefficient_matrix(self):
        """Test coefficient matrix extraction."""
        fitter = PolynomialFitter(delta_sigma=0.518, max_deriv=11, poly_degree=10)
        poly_F = fitter.fit_polynomial(delta_gap=1.5, delta_max=30.0)

        coeff_matrix = poly_F.get_coefficient_matrix()

        assert coeff_matrix.shape == (6, 11)  # 6 components, degree 10 + 1 coeffs


class TestSOSConstraints:
    """Tests for sum-of-squares positivity constraints."""

    def test_sos_structure(self):
        """Test SOS constraint structure."""
        sos = SOSPositivityConstraint(poly_degree=10)

        # For degree 10: k0 = 5, k1 = 4
        size0, size1 = sos.gram_matrix_size()
        assert size0 == 6  # k0 + 1
        assert size1 == 5  # k1 + 1

    def test_sos_odd_degree(self):
        """Test SOS with odd degree polynomial."""
        sos = SOSPositivityConstraint(poly_degree=11)

        size0, size1 = sos.gram_matrix_size()
        assert size0 == 6  # 11 // 2 + 1
        assert size1 == 5  # (11-1) // 2 + 1

    def test_coefficient_map_structure(self):
        """Test coefficient map has correct structure."""
        sos = SOSPositivityConstraint(poly_degree=6)

        # map0 should cover degrees 0 to 2*k0
        assert len(sos.map0) == 2 * sos.k0 + 1

        # Each entry should have valid (i, j) pairs
        for n, pairs in sos.map0.items():
            for (i, j) in pairs:
                assert i + j == n
                assert 0 <= i <= sos.k0
                assert 0 <= j <= sos.k0

    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('cvxpy'),
        reason="CVXPY not installed"
    )
    def test_sos_cvxpy_constraints(self):
        """Test CVXPY constraint building."""
        import cvxpy as cp

        sos = SOSPositivityConstraint(poly_degree=8)
        n_components = 5
        n_coeffs = 9  # poly_degree + 1

        # Random coefficient matrix
        coefficients = np.random.randn(n_components, n_coeffs)

        # Create alpha variable
        alpha = cp.Variable(n_components)

        # Build constraints
        constraints, Q0, Q1 = sos.build_constraints_cvxpy(coefficients, alpha)

        # Should have 2 PSD constraints + n_coeffs equality constraints
        assert len(constraints) == 2 + n_coeffs

        # Check Q0 shape
        assert Q0.shape == (sos.k0 + 1, sos.k0 + 1)

        # Check Q1 shape
        assert Q1.shape == (sos.k1 + 1, sos.k1 + 1)


class TestPolynomialPositivitySolver:
    """Tests for the main polynomial positivity solver."""

    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('cvxpy'),
        reason="CVXPY not installed"
    )
    def test_solver_initialization(self):
        """Test solver initialization."""
        solver = PolynomialPositivitySolver(
            delta_sigma=0.518,
            max_deriv=11,
            poly_degree=12
        )

        assert solver.delta_sigma == 0.518
        assert solver.max_deriv == 11
        assert solver.n_constraints == 6
        assert solver.poly_degree == 12

    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('cvxpy'),
        reason="CVXPY not installed"
    )
    def test_exclusion_low_gap(self):
        """Test that very low Δε' values are ALLOWED (not excluded)."""
        solver = PolynomialPositivitySolver(
            delta_sigma=0.518,
            max_deriv=11,
            poly_degree=12
        )

        # Very low gap should be allowed
        excluded, info = solver.is_excluded(
            delta_epsilon=1.41,
            delta_epsilon_prime=1.5
        )

        assert not excluded, "Low gap should be ALLOWED"

    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('cvxpy'),
        reason="CVXPY not installed"
    )
    def test_exclusion_high_gap(self):
        """Test that very high Δε' values are EXCLUDED."""
        solver = PolynomialPositivitySolver(
            delta_sigma=0.518,
            max_deriv=11,
            poly_degree=12
        )

        # Very high gap should be excluded
        excluded, info = solver.is_excluded(
            delta_epsilon=1.41,
            delta_epsilon_prime=5.0
        )

        assert excluded, "High gap should be EXCLUDED"

    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('cvxpy'),
        reason="CVXPY not installed"
    )
    def test_bound_computation(self):
        """Test bound computation at Ising point."""
        solver = PolynomialPositivitySolver(
            delta_sigma=0.518,
            max_deriv=11,
            poly_degree=12
        )

        bound = solver.find_delta_epsilon_prime_bound(
            delta_epsilon=1.41,
            tolerance=0.1,
            verbose=False
        )

        # Bound should be finite and reasonable
        assert np.isfinite(bound)
        assert bound > 1.5  # Above first scalar
        assert bound < 10.0  # Below some maximum

    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('cvxpy'),
        reason="CVXPY not installed"
    )
    def test_hybrid_method(self):
        """Test hybrid method (polynomial + discrete samples)."""
        solver = PolynomialPositivitySolver(
            delta_sigma=0.518,
            max_deriv=11,
            poly_degree=12
        )

        # Test with hybrid method
        excluded, info = solver.is_excluded_hybrid(
            delta_epsilon=1.41,
            delta_epsilon_prime=3.0,
            n_discrete_samples=20
        )

        assert isinstance(excluded, bool)
        assert "n_discrete_samples" in info


class TestPolynomialPositivityGapSolver:
    """Tests for the gap solver with polynomial positivity."""

    def test_literature_boundary(self):
        """Test literature boundary interpolation."""
        de = PolynomialPositivityGapSolver.delta_epsilon_boundary_literature(0.518)
        assert abs(de - 1.41) < 0.01

        # Free field
        de_free = PolynomialPositivityGapSolver.delta_epsilon_boundary_literature(0.5)
        assert abs(de_free - 1.0) < 0.01

        # Extrapolation
        de_ext = PolynomialPositivityGapSolver.delta_epsilon_boundary_literature(0.65)
        assert de_ext > 1.85  # Should extrapolate beyond last point

    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('cvxpy'),
        reason="CVXPY not installed"
    )
    def test_gap_solver_initialization(self):
        """Test gap solver initialization."""
        computer = PolynomialPositivityGapSolver(
            max_deriv=11,
            poly_degree=12
        )

        assert computer.max_deriv == 11
        assert computer.poly_degree == 12
        assert computer.n_constraints == 6

    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('cvxpy'),
        reason="CVXPY not installed"
    )
    def test_single_point_bound(self):
        """Test bound at single point."""
        computer = PolynomialPositivityGapSolver(
            max_deriv=11,
            poly_degree=12
        )

        bound = computer.compute_bound_at_point(
            delta_sigma=0.518,
            delta_epsilon=1.41,
            tolerance=0.1,
            verbose=False
        )

        assert np.isfinite(bound)
        assert bound > 1.5


class TestFittedPolynomial:
    """Tests for the FittedPolynomial dataclass."""

    def test_degree_property(self):
        """Test degree property."""
        poly = FittedPolynomial(
            coefficients=np.array([1, 2, 3, 4]),
            delta_gap=1.0,
            delta_max=10.0
        )

        assert poly.degree == 3

    def test_evaluate(self):
        """Test polynomial evaluation."""
        # p(x) = 1 + 2x + 3x^2
        poly = FittedPolynomial(
            coefficients=np.array([1, 2, 3]),
            delta_gap=1.0,
            delta_max=10.0
        )

        # At delta = 2: x = 1, p(1) = 1 + 2 + 3 = 6
        val = poly.evaluate(2.0)
        assert abs(val - 6.0) < 1e-10

        # At delta = 3: x = 2, p(2) = 1 + 4 + 12 = 17
        val = poly.evaluate(3.0)
        assert abs(val - 17.0) < 1e-10

    def test_evaluate_array(self):
        """Test array evaluation."""
        poly = FittedPolynomial(
            coefficients=np.array([1, 2, 3]),
            delta_gap=1.0,
            delta_max=10.0
        )

        deltas = np.array([2.0, 3.0, 4.0])
        vals = poly.evaluate_array(deltas)

        assert len(vals) == 3
        assert abs(vals[0] - 6.0) < 1e-10   # p(1) = 6
        assert abs(vals[1] - 17.0) < 1e-10  # p(2) = 17
        assert abs(vals[2] - 34.0) < 1e-10  # p(3) = 1 + 6 + 27 = 34


class TestPolynomialFVector:
    """Tests for the PolynomialFVector dataclass."""

    def test_properties(self):
        """Test PolynomialFVector properties."""
        polys = [
            FittedPolynomial(np.array([1, 2]), 1.0, 10.0),
            FittedPolynomial(np.array([1, 2, 3]), 1.0, 10.0),
            FittedPolynomial(np.array([1]), 1.0, 10.0),
        ]

        pf = PolynomialFVector(polynomials=polys, delta_gap=1.0, delta_max=10.0)

        assert pf.n_components == 3
        assert pf.max_degree == 2  # Max of [1, 2, 0]

    def test_coefficient_matrix(self):
        """Test coefficient matrix extraction."""
        polys = [
            FittedPolynomial(np.array([1, 2]), 1.0, 10.0),
            FittedPolynomial(np.array([3, 4, 5]), 1.0, 10.0),
        ]

        pf = PolynomialFVector(polynomials=polys, delta_gap=1.0, delta_max=10.0)
        matrix = pf.get_coefficient_matrix()

        assert matrix.shape == (2, 3)
        np.testing.assert_array_equal(matrix[0], [1, 2, 0])
        np.testing.assert_array_equal(matrix[1], [3, 4, 5])


class TestIntegration:
    """Integration tests comparing methods."""

    @pytest.mark.skipif(
        not __import__('importlib.util').util.find_spec('cvxpy'),
        reason="CVXPY not installed"
    )
    @pytest.mark.slow
    def test_polynomial_vs_discrete(self):
        """Test that polynomial method gives reasonable bounds vs discrete."""
        from taylor_conformal_blocks import HighOrderGapBootstrapSolver

        max_deriv = 9
        delta_sigma = 0.518
        delta_epsilon = 1.41

        # Discrete method
        discrete_solver = HighOrderGapBootstrapSolver(d=3, max_deriv=max_deriv)
        discrete_bound = discrete_solver.find_delta_epsilon_prime_bound(
            delta_sigma, delta_epsilon, tolerance=0.1
        )

        # Polynomial method
        poly_solver = PolynomialPositivitySolver(
            delta_sigma=delta_sigma,
            max_deriv=max_deriv,
            poly_degree=10
        )
        poly_bound = poly_solver.find_delta_epsilon_prime_bound(
            delta_epsilon=delta_epsilon, tolerance=0.1, verbose=False
        )

        # Both should give reasonable bounds
        assert np.isfinite(discrete_bound)
        assert np.isfinite(poly_bound)

        # Bounds should be in reasonable range
        assert 1.5 < discrete_bound < 10.0
        assert 1.5 < poly_bound < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
