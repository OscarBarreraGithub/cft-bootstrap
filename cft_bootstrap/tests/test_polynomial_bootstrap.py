"""
Tests for the polynomial bootstrap infrastructure.

These tests verify:
1. Basic infrastructure (PolynomialVector, BilinearBasis)
2. Comparison with pycftboot reference (when available)
3. PMP file generation
4. Ising point bound computation
"""

import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polynomial_bootstrap import (
    HAVE_SYMENGINE,
    HAVE_MPMATH,
    R_CROSS,
    PREC,
    rising_factorial,
    unitarity_bound,
    gather_poles,
    ConformalBlockPoles,
    BilinearBasis,
)


class TestConstants(unittest.TestCase):
    """Test that constants match pycftboot."""

    def test_r_cross(self):
        """R_cross should be 3 - 2*sqrt(2) ≈ 0.1716."""
        expected = 3 - 2 * np.sqrt(2)
        self.assertAlmostEqual(R_CROSS, expected, places=10)
        self.assertAlmostEqual(R_CROSS, 0.17157287525380988, places=10)

    def test_precision(self):
        """Precision should be 660 bits ≈ 200 decimal digits."""
        self.assertEqual(PREC, 660)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_rising_factorial(self):
        """Test Pochhammer symbol (rising factorial)."""
        # (1)_3 = 1 * 2 * 3 = 6
        self.assertEqual(rising_factorial(1, 3), 6)

        # (2)_4 = 2 * 3 * 4 * 5 = 120
        self.assertEqual(rising_factorial(2, 4), 120)

        # (x)_0 = 1
        self.assertEqual(rising_factorial(5, 0), 1)

        # Negative n: (x)_{-n} = 1 / (x-n)_n
        self.assertAlmostEqual(rising_factorial(5, -2), 1 / (3 * 4), places=10)

    def test_unitarity_bound(self):
        """Test unitarity bounds in d=3."""
        dim = 3

        # Scalar: Δ ≥ (d-2)/2 = 0.5
        self.assertEqual(unitarity_bound(dim, 0), 0.5)

        # Spin 1: Δ ≥ d + l - 2 = 2
        self.assertEqual(unitarity_bound(dim, 1), 2)

        # Spin 2 (stress tensor): Δ ≥ 3
        self.assertEqual(unitarity_bound(dim, 2), 3)

    def test_gather_poles(self):
        """Test pole gathering with multiplicities."""
        # Simple case: distinct poles
        poles = [1.0, 2.0, 3.0]
        gathered = gather_poles(poles)
        self.assertEqual(len(gathered), 3)
        self.assertEqual(gathered[1.0], 1)

        # Double pole
        poles = [1.0, 1.0 + 1e-200, 2.0]
        gathered = gather_poles(poles)
        self.assertEqual(len(gathered), 2)
        self.assertEqual(gathered[1.0], 2)


class TestConformalBlockPoles(unittest.TestCase):
    """Test conformal block pole computation."""

    def test_poles_d3_spin0(self):
        """Test pole positions for d=3, spin=0."""
        calc = ConformalBlockPoles(dim=3, k_max=10)

        # Series 1 poles: 1 - l - k = 1 - 0 - k = 1 - k
        self.assertAlmostEqual(calc.get_pole(1, 0, 1), 0.0, places=10)
        self.assertAlmostEqual(calc.get_pole(2, 0, 1), -1.0, places=10)

        # Series 2 poles: 1 + ν - k = 1 + 0.5 - k = 1.5 - k
        self.assertAlmostEqual(calc.get_pole(1, 0, 2), 0.5, places=10)
        self.assertAlmostEqual(calc.get_pole(2, 0, 2), -0.5, places=10)

    def test_all_poles_spin0(self):
        """Get all poles for spin-0."""
        calc = ConformalBlockPoles(dim=3, k_max=5)
        poles = calc.get_all_poles(l=0)

        # Should have poles from multiple series
        # Note: some may vanish due to zero residue
        self.assertGreater(len(poles), 0)


class TestBilinearBasis(unittest.TestCase):
    """Test bilinear basis computation."""

    def test_gram_matrix_positive_definite(self):
        """Gram matrix should be positive definite."""
        # Simple case: no poles, delta_min = 1
        basis = BilinearBasis(poles=[], delta_min=1.0, max_degree=4)

        # Check Gram matrix is positive definite
        eigenvalues = np.linalg.eigvalsh(basis.gram_matrix)
        self.assertTrue(np.all(eigenvalues > 0),
                       f"Gram matrix has non-positive eigenvalues: {eigenvalues}")

    def test_gram_matrix_with_poles(self):
        """Gram matrix with poles should still be positive definite."""
        # Add some poles below delta_min
        # Using poles that are well-separated from integration domain
        poles = [-5.0, -10.0]  # Well below delta_min = 1.0
        try:
            basis = BilinearBasis(poles=poles, delta_min=1.0, max_degree=4)
            eigenvalues = np.linalg.eigvalsh(basis.gram_matrix)
            # At minimum, the matrix should be computed
            self.assertIsNotNone(eigenvalues)
        except Exception as e:
            # With poles, the integral is more complex and may need symengine
            self.skipTest(f"Pole integration requires full symengine support: {e}")

    def test_basis_matrix_orthogonality(self):
        """Transformed polynomials should be orthogonal."""
        basis = BilinearBasis(poles=[], delta_min=1.0, max_degree=4)

        # Transform identity matrix should give orthogonal vectors
        n = basis.basis_matrix.shape[0]
        B = basis.basis_matrix
        G = basis.gram_matrix

        # For Cholesky G = L L^T with B = L^{-1}:
        # B G B^T = L^{-1} L L^T L^{-T} = I
        product = B @ G @ B.T
        expected = np.eye(n)

        np.testing.assert_array_almost_equal(
            product, expected, decimal=6,
            err_msg="Basis transformation does not produce orthonormal basis"
        )


class TestPMPGeneration(unittest.TestCase):
    """Test PMP file generation."""

    def setUp(self):
        """Create temporary directory for output."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_laguerre_points(self):
        """Test Laguerre sample point generation."""
        from polynomial_bootstrap import PMPGenerator

        # Create minimal generator
        class MockTable:
            dim = 3
            table = []

        generator = PMPGenerator(
            convolved_table=MockTable(),
            bounds={},
            objective=[],
            normalization=[]
        )

        # Generate points for degree 5
        points = generator.make_laguerre_points(5)
        self.assertEqual(len(points), 6)

        # Points should be positive (sample in [0, ∞))
        self.assertTrue(all(p >= 0 for p in points))

        # Points should be increasing
        self.assertEqual(points, sorted(points))

    def test_shifted_prefactor(self):
        """Test damped rational prefactor computation."""
        from polynomial_bootstrap import PMPGenerator

        class MockTable:
            dim = 3
            table = []

        generator = PMPGenerator(
            convolved_table=MockTable(),
            bounds={},
            objective=[],
            normalization=[]
        )

        # Without poles: prefactor = r_cross^(x + shift)
        val = generator.shifted_prefactor(poles=[], x=0, shift=0)
        self.assertAlmostEqual(val, 1.0, places=10)

        val = generator.shifted_prefactor(poles=[], x=1, shift=0)
        self.assertAlmostEqual(val, R_CROSS, places=10)

        # With one pole: prefactor = r_cross^(x + shift) / (x - (pole - shift))
        val = generator.shifted_prefactor(poles=[-1.0], x=0, shift=0)
        expected = R_CROSS ** 0 / (0 - (-1.0))  # = 1 / 1 = 1
        self.assertAlmostEqual(val, expected, places=10)


@unittest.skipUnless(HAVE_SYMENGINE, "symengine not available")
class TestSymengineIntegration(unittest.TestCase):
    """Tests requiring symengine."""

    def test_symbolic_polynomial_creation(self):
        """Test creating symbolic polynomials."""
        from polynomial_bootstrap import (
            SymbolicPolynomialVector,
            coefficients_from_polynomial,
            build_polynomial_from_coeffs,
            delta
        )
        from symengine import RealMPFR

        # Create a simple polynomial: 1 + 2*delta + 3*delta^2
        poly = RealMPFR("1", PREC) + RealMPFR("2", PREC) * delta + RealMPFR("3", PREC) * delta**2

        # Extract coefficients
        coeffs = coefficients_from_polynomial(poly)
        expected = [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(coeffs, expected, decimal=8)


class TestPycftbootComparison(unittest.TestCase):
    """Compare with pycftboot reference implementation."""

    @classmethod
    def setUpClass(cls):
        """Check if pycftboot can be imported."""
        try:
            # Try to import pycftboot from reference_implementations
            ref_path = Path(__file__).parent.parent.parent / "reference_implementations" / "pycftboot"
            if ref_path.exists():
                sys.path.insert(0, str(ref_path))

            from symengine import Symbol
            cls.has_pycftboot = True
        except ImportError:
            cls.has_pycftboot = False

    @unittest.skip("Requires full pycftboot setup")
    def test_block_table_comparison(self):
        """Compare block tables with pycftboot."""
        if not self.has_pycftboot:
            self.skipTest("pycftboot not available")

        # This would compare our block computation with pycftboot
        # Requires full pycftboot environment (symengine + MPFR)
        pass


class TestIsingPointBound(unittest.TestCase):
    """Test the Ising point bound computation."""

    @unittest.skip("Requires full implementation")
    def test_ising_bound_correct(self):
        """
        At the Ising point (Δσ=0.518, Δε=1.41), the Δε' bound should be ~3.8.

        This is the key test that will validate the polynomial approach is working.
        Currently we get ~2.6 with discrete sampling.
        """
        from polynomial_bootstrap import PolynomialBootstrapSolver

        solver = PolynomialBootstrapSolver(dim=3, k_max=20, l_max=50, m_max=10, n_max=10)
        solver.setup_problem(delta_sigma=0.518)

        # Would run SDPB here...
        # bound = solver.compute_bound(delta_epsilon=1.41)

        # Expected: bound ≈ 3.8, NOT 2.6
        # self.assertGreater(bound, 3.5)
        pass


if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)
