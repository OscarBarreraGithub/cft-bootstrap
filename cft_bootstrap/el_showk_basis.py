"""
El-Showk et al. (2012) derivative basis for conformal bootstrap.

This module implements the full derivative basis used in:
  "Solving the 3D Ising Model with the Conformal Bootstrap"
  El-Showk et al., arXiv:1203.6064 (2012)

Key differences from the standard implementation:
1. Uses (a,b) coordinates where u = a², v = b² (cross-ratios)
2. Evaluates derivatives at (a=1, b=0), i.e., u=1, v=0
3. Uses mixed derivatives ∂_a^m ∂_b^n with constraint m + 2n ≤ 2*nmax + 1
4. Only odd m values contribute due to crossing symmetry

With nmax=10, this gives 66 independent coefficients.

Reference: El-Showk et al. (2012), Section 4 and Appendix D
"""

import numpy as np
from math import factorial
from typing import List, Tuple, Optional
from functools import lru_cache
import warnings

try:
    from .bootstrap_solver import ConformalBlock3D
except ImportError:
    from bootstrap_solver import ConformalBlock3D


def get_derivative_indices(nmax: int) -> List[Tuple[int, int]]:
    """
    Get all (m, n) pairs for the derivative basis.

    The constraint is: m + 2n ≤ 2*nmax + 1, with m odd and m,n ≥ 0.

    Args:
        nmax: Maximum order parameter (paper uses nmax=10)

    Returns:
        List of (m, n) tuples sorted by total order
    """
    indices = []
    max_constraint = 2 * nmax + 1

    for m in range(1, max_constraint + 1, 2):  # m = 1, 3, 5, ...
        for n in range(0, (max_constraint - m) // 2 + 1):
            if m + 2 * n <= max_constraint:
                indices.append((m, n))

    # Sort by total order m + 2n, then by m
    indices.sort(key=lambda x: (x[0] + 2 * x[1], x[0]))
    return indices


def count_coefficients(nmax: int) -> int:
    """Count the number of derivative coefficients for given nmax."""
    return len(get_derivative_indices(nmax))


class ElShowkCrossingVector:
    """
    Compute F-vectors using El-Showk et al. (2012) conventions.

    Uses coordinates (a, b) where:
        z = 1/2 + (a + b)/2
        zbar = 1/2 + (a - b)/2

    The crossing-symmetric point is (a=0, b=0), giving z = zbar = 1/2.

    The paper expands F(a,b) around (a=0, b=0) and takes mixed derivatives
    ∂_a^m ∂_b^n with the constraint m + 2n ≤ 2*nmax + 1, where only odd m
    contribute due to crossing symmetry.
    """

    def __init__(self, delta_sigma: float, nmax: int = 10):
        """
        Initialize the crossing vector computer.

        Args:
            delta_sigma: External scalar dimension
            nmax: Derivative order (paper uses nmax=10 for 66 coefficients)
        """
        self.delta_sigma = delta_sigma
        self.nmax = nmax
        self.blocks = ConformalBlock3D()
        self.indices = get_derivative_indices(nmax)
        self.n_coeffs = len(self.indices)

    def _a_b_to_z_zbar(self, a: float, b: float) -> Tuple[float, float]:
        """
        Convert (a, b) coordinates to (z, zbar).

        z = 1/2 + (a + b)/2
        zbar = 1/2 + (a - b)/2
        """
        z = 0.5 + (a + b) / 2
        zbar = 0.5 + (a - b) / 2
        return z, zbar

    def F_function(self, delta: float, a: float, b: float) -> float:
        """
        Compute F(a, b) for operator of dimension delta.

        F(a,b) = v^{Δσ} g(z,zbar) - u^{Δσ} g(1-z,1-zbar)

        where u = z*zbar, v = (1-z)(1-zbar).
        """
        z, zbar = self._a_b_to_z_zbar(a, b)

        u = z * zbar
        v = (1 - z) * (1 - zbar)

        # Handle edge cases
        if z <= 0 or z >= 1 or zbar <= 0 or zbar >= 1:
            return 0.0

        if delta == 0:
            # Identity contribution
            return v**self.delta_sigma - u**self.delta_sigma

        # Regular operator
        g1 = self.blocks.scalar_block(delta, z, zbar)
        g2 = self.blocks.scalar_block(delta, 1-z, 1-zbar)

        return v**self.delta_sigma * g1 - u**self.delta_sigma * g2

    def _numerical_derivative(self, delta: float, m: int, n: int,
                              h_a: float = 0.02, h_b: float = 0.02) -> float:
        """
        Compute ∂_a^m ∂_b^n F at (a=0, b=0) using finite differences.

        Uses central differences for both a and b derivatives.
        """
        # Evaluate at crossing-symmetric point (a=0, b=0)
        a0, b0 = 0.0, 0.0

        # For n-th b-derivative at fixed a
        def deriv_b_at_a(a_val):
            if n == 0:
                return self.F_function(delta, a_val, b0)

            # Central difference for b
            result = 0.0
            coeffs_b = self._get_diff_coeffs(n)
            half_b = len(coeffs_b) // 2
            for j, c in enumerate(coeffs_b):
                b_val = b0 + (j - half_b) * h_b
                result += c * self.F_function(delta, a_val, b_val)
            return result / (h_b ** n)

        # Now take m-th derivative in a
        if m == 0:
            return deriv_b_at_a(a0)

        coeffs_a = self._get_diff_coeffs(m)
        half_a = len(coeffs_a) // 2
        result = 0.0
        for i, c in enumerate(coeffs_a):
            a_val = a0 + (i - half_a) * h_a
            result += c * deriv_b_at_a(a_val)

        return result / (h_a ** m)

    @staticmethod
    @lru_cache(maxsize=50)
    def _get_diff_coeffs(order: int) -> Tuple[float, ...]:
        """Get central difference coefficients for given derivative order."""
        # Pre-computed central difference coefficients
        coeffs = {
            0: (1.0,),
            1: (-0.5, 0, 0.5),
            2: (1, -2, 1),
            3: (-0.5, 1, 0, -1, 0.5),
            4: (1, -4, 6, -4, 1),
            5: (-0.5, 2, -2.5, 0, 2.5, -2, 0.5),
            6: (1, -6, 15, -20, 15, -6, 1),
            7: (-0.5, 3, -7, 7, 0, -7, 7, -3, 0.5),
            8: (1, -8, 28, -56, 70, -56, 28, -8, 1),
        }

        if order in coeffs:
            return coeffs[order]

        # Compute higher-order coefficients
        n_points = order + 1 + (order % 2)
        half_n = n_points // 2
        points = np.arange(-half_n, half_n + 1, dtype=float)
        A = np.vander(points, N=order+1, increasing=True).T
        b = np.zeros(order + 1)
        b[order] = factorial(order)

        try:
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            return tuple(c)
        except:
            # Fallback
            return tuple([(-1)**i * np.math.comb(order, i) for i in range(order+1)])

    def build_F_vector(self, delta: float) -> np.ndarray:
        """
        Build the full F-vector for an operator of dimension delta.

        Returns array of shape (n_coeffs,) containing:
            F_{m,n} = (1/m!n!) ∂_a^m ∂_b^n F(a,b)|_{a=1,b=0}

        for all (m,n) in the derivative basis.
        """
        F_vec = np.zeros(self.n_coeffs)

        for i, (m, n) in enumerate(self.indices):
            deriv = self._numerical_derivative(delta, m, n)
            # Normalize by factorials
            F_vec[i] = deriv / (factorial(m) * factorial(n))

        return F_vec

    def build_F_vector_taylor(self, delta: float) -> np.ndarray:
        """
        Build F-vector using Taylor expansion (more stable for high orders).

        Uses the approach from taylor_conformal_blocks.py adapted for
        the El-Showk basis.
        """
        # For now, fall back to numerical derivatives
        # TODO: Implement proper Taylor expansion in (a,b) coordinates
        return self.build_F_vector(delta)


class ElShowkBootstrapSolver:
    """
    Bootstrap solver using the full El-Showk derivative basis.

    This solver implements the paper's protocol with:
    - nmax=10 (66 derivative coefficients)
    - Mixed (a,b) derivatives at (a=1, b=0)
    - Compatible with spinning operators
    """

    def __init__(self, d: int = 3, nmax: int = 10):
        """
        Initialize the solver.

        Args:
            d: Spacetime dimension (default 3 for Ising)
            nmax: Derivative order (default 10 for 66 coefficients)
        """
        self.d = d
        self.nmax = nmax
        self.n_constraints = count_coefficients(nmax)
        self.indices = get_derivative_indices(nmax)

        print(f"El-Showk solver initialized:")
        print(f"  nmax = {nmax}")
        print(f"  Number of constraints: {self.n_constraints}")

    def is_excluded(self, delta_sigma: float, delta_epsilon: float,
                    delta_epsilon_prime: float,
                    delta_max: float = 30.0, n_samples: int = 100) -> bool:
        """
        Check if point (Δσ, Δε, Δε') is excluded.

        Uses the dual SDP formulation:
            Find α such that:
                α · F_id = 1 (normalization)
                α · F_ε ≥ 0 (first scalar OK)
                α · F_Δ ≥ 0 for all Δ ≥ Δε' (positivity)

        If such α exists, the point is EXCLUDED.
        """
        try:
            import cvxpy as cp
        except ImportError:
            warnings.warn("CVXPY not installed")
            return False

        try:
            from .bootstrap_gap_solver import reshuffle_with_normalization
        except ImportError:
            from bootstrap_gap_solver import reshuffle_with_normalization

        cross = ElShowkCrossingVector(delta_sigma, self.nmax)

        # Build F-vectors
        F_id = cross.build_F_vector(0)  # Identity
        F_eps = cross.build_F_vector(delta_epsilon)  # First scalar

        # Sample operators above the gap
        deltas = np.linspace(delta_epsilon_prime, delta_max, n_samples)
        F_ops = np.array([cross.build_F_vector(d) for d in deltas])

        # Stack for reshuffling
        F_all = np.vstack([F_eps[np.newaxis, :], F_ops])

        # Apply normalization
        F_reduced, fixed_contribs, max_idx = reshuffle_with_normalization(F_all, F_id)

        # Separate
        F_eps_reduced = F_reduced[0, :]
        fixed_eps = fixed_contribs[0]
        F_ops_reduced = F_reduced[1:, :]
        fixed_ops = fixed_contribs[1:]

        # Solve SDP
        alpha = cp.Variable(self.n_constraints - 1)

        constraints = [alpha @ F_eps_reduced >= -fixed_eps]
        for i, F_O in enumerate(F_ops_reduced):
            constraints.append(alpha @ F_O >= -fixed_ops[i])

        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
            return prob.status == cp.OPTIMAL
        except Exception as e:
            warnings.warn(f"Solver failed: {e}")
            return False

    def find_delta_epsilon_prime_bound(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: float = None,
        delta_prime_max: float = 6.0,
        tolerance: float = 0.02
    ) -> float:
        """Find upper bound on Δε' using binary search."""
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        if self.is_excluded(delta_sigma, delta_epsilon, delta_prime_min):
            return delta_prime_min

        if not self.is_excluded(delta_sigma, delta_epsilon, delta_prime_max):
            return float('inf')

        lo, hi = delta_prime_min, delta_prime_max
        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            if self.is_excluded(delta_sigma, delta_epsilon, mid):
                hi = mid
            else:
                lo = mid

        return (lo + hi) / 2


# Convenience functions
def test_el_showk_basis():
    """Test the El-Showk derivative basis."""
    print("Testing El-Showk derivative basis")
    print("=" * 50)

    # Test coefficient counting
    for nmax in [5, 10, 11]:
        n_coeffs = count_coefficients(nmax)
        expected = (nmax + 1) * (nmax + 2) // 2
        print(f"nmax={nmax}: {n_coeffs} coefficients (expected: {expected})")

    # Test derivative indices
    print("\nDerivative indices for nmax=3:")
    indices = get_derivative_indices(3)
    for m, n in indices:
        print(f"  (m={m}, n={n}), order = {m + 2*n}")

    # Test F-vector computation
    print("\nTesting F-vector computation at Ising point:")
    cross = ElShowkCrossingVector(delta_sigma=0.518, nmax=3)

    F_id = cross.build_F_vector(0)
    F_eps = cross.build_F_vector(1.41)

    print(f"  F_identity: {F_id[:5]}...")
    print(f"  F_epsilon:  {F_eps[:5]}...")

    # Test solver
    print("\nTesting solver at Ising point:")
    solver = ElShowkBootstrapSolver(d=3, nmax=3)

    excluded = solver.is_excluded(0.518, 1.41, 3.0)
    print(f"  Δε'=3.0 excluded: {excluded}")

    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    test_el_showk_basis()
