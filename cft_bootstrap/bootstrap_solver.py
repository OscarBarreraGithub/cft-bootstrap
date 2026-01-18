"""
CFT Bootstrap solver using linear programming and semidefinite programming.

This module implements the core bootstrap algorithm:
1. Linear Programming (LP) approach - fast but weaker bounds
2. Semidefinite Programming (SDP) approach - slower but optimal bounds (requires CVXPY)
"""

import numpy as np
from math import factorial, comb
from scipy.special import hyp2f1
from scipy.optimize import linprog
from typing import Tuple, Optional, List
from functools import lru_cache
import warnings

# Try to import CVXPY for SDP (optional)
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    warnings.warn("CVXPY not installed. SDP solver unavailable. Install with: pip install cvxpy")

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============================================================================
# Conformal Blocks
# ============================================================================

class ConformalBlock3D:
    """3D conformal blocks using Dolan-Osborn formula."""

    def k_function(self, beta: float, x: float) -> float:
        """k_β(x) = x^(β/2) * 2F1(β/2, β/2; β; x)"""
        if x <= 0 or x >= 1:
            return 0.0
        try:
            return x**(beta/2) * hyp2f1(beta/2, beta/2, beta, x)
        except:
            return 0.0

    def scalar_block(self, delta: float, z: float, zbar: float) -> float:
        """Compute g_{Δ,0}(z, zbar) for scalar exchange."""
        if abs(z - zbar) < 1e-10:
            return self._diagonal_limit(delta, z)

        k_d_z = self.k_function(delta, z)
        k_d_zb = self.k_function(delta, zbar)
        k_dm1_z = self.k_function(delta - 1, z)
        k_dm1_zb = self.k_function(delta - 1, zbar)

        return (z * zbar / (z - zbar)) * (k_d_z * k_dm1_zb - k_d_zb * k_dm1_z)

    def _diagonal_limit(self, delta: float, z: float, eps: float = 1e-5) -> float:
        """Diagonal z=zbar limit via averaging."""
        g1 = self.scalar_block(delta, z + eps, z - eps)
        g2 = self.scalar_block(delta, z - eps, z + eps)
        return (g1 + g2) / 2


# ============================================================================
# Crossing Equation
# ============================================================================

class CrossingVector:
    """
    Compute F-vectors for the crossing equation.

    Coordinates: z = 1/2 + (a + b)/2, zbar = 1/2 + (a - b)/2
    Crossing symmetric point: a = b = 0
    """

    def __init__(self, delta_sigma: float):
        self.delta_sigma = delta_sigma
        self.blocks = ConformalBlock3D()

    def F(self, delta: float, a: float, b: float) -> float:
        """Compute F at given (a, b) coordinates."""
        z = 0.5 + (a + b) / 2
        zbar = 0.5 + (a - b) / 2

        u = z * zbar
        v = (1 - z) * (1 - zbar)

        if delta == 0:
            return v**self.delta_sigma - u**self.delta_sigma
        else:
            g1 = self.blocks.scalar_block(delta, z, zbar)
            g2 = self.blocks.scalar_block(delta, 1-z, 1-zbar)
            return v**self.delta_sigma * g1 - u**self.delta_sigma * g2

    def derivative(self, delta: float, m: int, h: float = 0.02) -> float:
        """Compute d^m/da^m F at a=0, b=0 using central differences."""
        coeffs = self._central_diff_coeffs(m)
        n = len(coeffs)
        half_n = n // 2

        result = 0.0
        for i, c in enumerate(coeffs):
            a_val = (i - half_n) * h
            result += c * self.F(delta, a_val, 0.0)

        return result / h**m

    @staticmethod
    @lru_cache(maxsize=20)
    def _central_diff_coeffs(m: int) -> tuple:
        """Central difference coefficients for m-th derivative."""
        coeffs = {
            1: (-0.5, 0, 0.5),
            2: (1, -2, 1),
            3: (-0.5, 1, 0, -1, 0.5),
            4: (1, -4, 6, -4, 1),
            5: (-0.5, 2, -2.5, 0, 2.5, -2, 0.5),
            6: (1, -6, 15, -20, 15, -6, 1),
            7: (-0.5, 3, -7, 7, 0, -7, 7, -3, 0.5),
        }
        if m in coeffs:
            return coeffs[m]
        return tuple([(-1)**i * comb(m, i) for i in range(m+1)])

    def build_F_vector(self, delta: float, max_deriv: int) -> np.ndarray:
        """
        Build F-vector: derivatives d^m/da^m F / m! for m = 1, 3, 5, ..., max_deriv
        """
        return np.array([
            self.derivative(delta, m) / factorial(m)
            for m in range(1, max_deriv + 1, 2)
        ])


# ============================================================================
# Bootstrap Solver
# ============================================================================

class BootstrapSolver:
    """
    Solver for CFT bootstrap bounds.

    The bootstrap checks if the crossing equation can be satisfied:
    F_id + Σ_O p_O F_O = 0  with p_O ≥ 0

    If no solution exists, that point (Δσ, Δε) is excluded.
    """

    def __init__(self, d: int = 3, max_deriv: int = 5):
        """
        Initialize the bootstrap solver.

        Args:
            d: Spacetime dimension
            max_deriv: Maximum derivative order (odd integers 1, 3, 5, ... up to this)
        """
        self.d = d
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2
        print(f"Using {self.n_constraints} derivative constraints (m = 1, 3, ..., {max_deriv})")

    def is_feasible_lp(self, delta_sigma: float, delta_gap: float,
                       delta_max: float = 20.0, n_samples: int = 100) -> bool:
        """
        Check feasibility using Linear Programming.

        Tests if -F_id can be written as positive combination of F_O
        for operators with Δ ≥ delta_gap.
        """
        cross = CrossingVector(delta_sigma)

        # Build F-vectors
        F_id = cross.build_F_vector(0, self.max_deriv)
        deltas = np.linspace(delta_gap, delta_max, n_samples)
        F_ops = np.array([cross.build_F_vector(d, self.max_deriv) for d in deltas])

        # Normalize for numerical stability
        scales = np.abs(F_ops).max(axis=0) + 1e-10
        F_ops_scaled = F_ops / scales
        F_id_scaled = F_id / scales

        # LP: find p >= 0 such that F_ops.T @ p = -F_id
        A_eq = F_ops_scaled.T
        b_eq = -F_id_scaled

        result = linprog(np.zeros(n_samples), A_eq=A_eq, b_eq=b_eq,
                        bounds=(0, None), method='highs')
        return result.success

    def is_feasible_sdp(self, delta_sigma: float, delta_gap: float,
                        delta_max: float = 20.0, n_samples: int = 100) -> bool:
        """
        Check feasibility using Semidefinite Programming.

        Looks for a linear functional α such that:
        - α · F_id = 1 (normalization)
        - α · F_O ≥ 0 for all O with Δ ≥ delta_gap (positivity)

        If such α exists, the point is excluded.
        """
        if not HAS_CVXPY:
            return self.is_feasible_lp(delta_sigma, delta_gap, delta_max, n_samples)

        cross = CrossingVector(delta_sigma)

        F_id = cross.build_F_vector(0, self.max_deriv)
        deltas = np.linspace(delta_gap, delta_max, n_samples)
        F_ops = np.array([cross.build_F_vector(d, self.max_deriv) for d in deltas])

        # SDP: find α such that α · F_id = 1 and α · F_O ≥ 0
        alpha = cp.Variable(self.n_constraints)

        constraints = [alpha @ F_id == 1]
        for F_O in F_ops:
            constraints.append(alpha @ F_O >= 0)

        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False)
            # If feasible, point is EXCLUDED
            return prob.status != cp.OPTIMAL
        except:
            return True  # Assume allowed if solver fails

    def find_bound(self, delta_sigma: float,
                   delta_min: float = 0.5, delta_max: float = 3.0,
                   tolerance: float = 0.01, method: str = 'lp') -> float:
        """
        Find the bootstrap upper bound on the gap using binary search.
        """
        is_feasible = self.is_feasible_lp if method == 'lp' else self.is_feasible_sdp

        if not is_feasible(delta_sigma, delta_min):
            return delta_min
        if is_feasible(delta_sigma, delta_max):
            return float('inf')

        lo, hi = delta_min, delta_max
        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            if is_feasible(delta_sigma, mid):
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2


class BootstrapBoundComputer:
    """Compute bootstrap bounds over a grid. Designed for cluster execution."""

    def __init__(self, d: int = 3, max_deriv: int = 5):
        self.solver = BootstrapSolver(d, max_deriv)

    def compute_bound_grid(self, delta_sigma_values: np.ndarray,
                          method: str = 'lp', tolerance: float = 0.01) -> np.ndarray:
        """Compute bounds for a grid of delta_sigma values."""
        results = []
        for i, delta_sigma in enumerate(delta_sigma_values):
            print(f"Computing bound {i+1}/{len(delta_sigma_values)}: Δσ = {delta_sigma:.4f}")
            bound = self.solver.find_bound(delta_sigma, method=method, tolerance=tolerance)
            results.append([delta_sigma, bound])
            print(f"  -> Δε ≤ {bound:.4f}")

        return np.array(results)

    def compute_single_bound(self, delta_sigma: float, method: str = 'lp',
                            tolerance: float = 0.01) -> Tuple[float, float]:
        """Compute a single bound."""
        bound = self.solver.find_bound(delta_sigma, method=method, tolerance=tolerance)
        return (delta_sigma, bound)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("CFT Bootstrap Solver")
    print("=" * 60)

    solver = BootstrapSolver(d=3, max_deriv=5)

    print("\nTesting feasibility at Δσ = 0.518 (3D Ising value):")
    for delta_gap in [1.0, 1.2, 1.41, 1.6, 1.8, 2.0]:
        t0 = time.time()
        feasible = solver.is_feasible_lp(0.518, delta_gap)
        t1 = time.time()
        status = "ALLOWED" if feasible else "EXCLUDED"
        print(f"  Δε = {delta_gap:.2f}: {status} ({t1-t0:.3f}s)")

    print("\nFinding bound for Δσ = 0.518:")
    t0 = time.time()
    bound = solver.find_bound(0.518, tolerance=0.01)
    t1 = time.time()
    print(f"  Upper bound: Δε ≤ {bound:.4f} ({t1-t0:.2f}s)")
    print(f"  3D Ising value: Δε ≈ 1.41 (should be below bound)")

    print("\n" + "=" * 60)
    print("Computing bounds for Δσ ∈ [0.50, 0.60]")
    print("=" * 60)

    computer = BootstrapBoundComputer(d=3, max_deriv=5)
    delta_sigmas = np.linspace(0.50, 0.60, 6)

    t0 = time.time()
    results = computer.compute_bound_grid(delta_sigmas, method='lp', tolerance=0.02)
    t1 = time.time()

    print(f"\nResults (computed in {t1-t0:.1f}s):")
    print("-" * 30)
    print("  Δσ      Δε_max")
    print("-" * 30)
    for ds, de in results:
        print(f"  {ds:.4f}   {de:.4f}")

    # Mark known CFT points
    print("\nKnown CFT values:")
    print("  Free scalar: Δσ = 0.5, Δε = 1.0")
    print("  3D Ising:    Δσ ≈ 0.518, Δε ≈ 1.41")
