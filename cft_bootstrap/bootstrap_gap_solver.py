"""
CFT Bootstrap solver with gap assumptions for Δε' bounds.

This module extends the basic bootstrap to compute bounds on the SECOND
Z2-even scalar operator dimension Δε', assuming a gap to the first
scalar at Δε.

This reproduces results from:
  "Solving the 3D Ising Model with the Conformal Bootstrap"
  El-Showk et al., arXiv:1203.6064 (2012), Figure 7

The key insight: instead of just demanding Δ ≥ Δε_gap for ALL scalars
in the OPE, we:
1. Include the identity operator (Δ = 0)
2. Include a SINGLE scalar at Δε (the assumed first scalar)
3. Demand ALL OTHER scalars have Δ ≥ Δε' (the bound we're computing)
"""

import numpy as np
from math import factorial
from scipy.optimize import linprog
from typing import Tuple, Optional
import warnings

# Try to import CVXPY for SDP (optional)
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

warnings.filterwarnings('ignore', category=RuntimeWarning)

from bootstrap_solver import ConformalBlock3D, CrossingVector


class GapBootstrapSolver:
    """
    Bootstrap solver with gap assumptions for Δε' bounds.

    The crossing equation structure is:
        F_id + p_ε * F_ε + Σ_{Δ ≥ Δε'} p_Δ * F_Δ = 0

    where:
        - F_id is the identity contribution
        - F_ε is the contribution from the first Z2-even scalar at Δε
        - F_Δ are contributions from scalars with Δ ≥ Δε' (second scalar onwards)

    We search for a linear functional α such that:
        - α · F_id = 1 (normalization)
        - α · F_ε ≥ 0 (first scalar is physical)
        - α · F_Δ ≥ 0 for all Δ ≥ Δε' (positivity)

    If such α exists, the point is EXCLUDED.
    """

    def __init__(self, d: int = 3, max_deriv: int = 5):
        self.d = d
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2
        self.blocks = ConformalBlock3D()

    def is_excluded(self, delta_sigma: float, delta_epsilon: float,
                    delta_epsilon_prime: float,
                    delta_max: float = 30.0, n_samples: int = 150) -> bool:
        """
        Check if point (Δσ, Δε, Δε') is excluded by the bootstrap.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: Assumed first Z2-even scalar dimension
            delta_epsilon_prime: Gap for second Z2-even scalar (what we're bounding)
            delta_max: Maximum dimension to sample
            n_samples: Number of operators to sample above the gap

        Returns:
            True if point is EXCLUDED, False if ALLOWED
        """
        cross = CrossingVector(delta_sigma)

        # Build F-vectors
        F_id = cross.build_F_vector(0, self.max_deriv)  # Identity
        F_eps = cross.build_F_vector(delta_epsilon, self.max_deriv)  # First scalar

        # Sample operators above the gap Δε'
        deltas = np.linspace(delta_epsilon_prime, delta_max, n_samples)
        F_ops = np.array([cross.build_F_vector(d, self.max_deriv) for d in deltas])

        if HAS_CVXPY:
            return self._is_excluded_sdp(F_id, F_eps, F_ops)
        else:
            return self._is_excluded_lp(F_id, F_eps, F_ops)

    def _is_excluded_lp(self, F_id: np.ndarray, F_eps: np.ndarray,
                        F_ops: np.ndarray) -> bool:
        """
        LP feasibility check.

        Tests if there exist p_ε ≥ 0 and p_Δ ≥ 0 such that:
            F_id + p_ε * F_eps + Σ p_Δ * F_Δ = 0

        If NO such solution exists, the point is EXCLUDED.
        """
        n_ops = len(F_ops)

        # Variables: [p_eps, p_1, p_2, ..., p_n] all ≥ 0
        # Constraint: F_eps * p_eps + F_ops.T @ p_ops = -F_id

        A_eq = np.column_stack([F_eps, F_ops.T])
        b_eq = -F_id

        # Normalize for numerical stability
        scales = np.abs(A_eq).max(axis=0) + 1e-10
        A_eq_scaled = A_eq / scales

        result = linprog(
            c=np.zeros(1 + n_ops),  # No objective
            A_eq=A_eq_scaled,
            b_eq=b_eq,
            bounds=(0, None),  # All coefficients non-negative
            method='highs'
        )

        # If LP is infeasible, point is EXCLUDED
        return not result.success

    def _is_excluded_sdp(self, F_id: np.ndarray, F_eps: np.ndarray,
                         F_ops: np.ndarray) -> bool:
        """
        SDP feasibility check (more powerful than LP).

        Find α such that:
            α · F_id = 1 (normalization)
            α · F_eps ≥ 0 (first scalar OK)
            α · F_Δ ≥ 0 for all Δ ≥ Δε' (positivity)

        If such α exists, point is EXCLUDED.
        """
        alpha = cp.Variable(self.n_constraints)

        constraints = [
            alpha @ F_id == 1,
            alpha @ F_eps >= 0,
        ]

        for F_O in F_ops:
            constraints.append(alpha @ F_O >= 0)

        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
            # If feasible (optimal found), point is EXCLUDED
            return prob.status == cp.OPTIMAL
        except Exception:
            # Solver failure - fall back to LP
            return self._is_excluded_lp(F_id, F_eps, F_ops)

    def find_delta_epsilon_prime_bound(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: float = None,
        delta_prime_max: float = 6.0,
        tolerance: float = 0.01
    ) -> float:
        """
        Find the upper bound on Δε' using binary search.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: Assumed first Z2-even scalar dimension
            delta_prime_min: Minimum Δε' to consider (defaults to delta_epsilon + 0.1)
            delta_prime_max: Maximum Δε' to consider
            tolerance: Binary search tolerance

        Returns:
            Upper bound on Δε'
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        # Ensure delta_prime_min > delta_epsilon
        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        # Check boundary conditions
        if self.is_excluded(delta_sigma, delta_epsilon, delta_prime_min):
            return delta_prime_min

        if not self.is_excluded(delta_sigma, delta_epsilon, delta_prime_max):
            return float('inf')

        # Binary search
        lo, hi = delta_prime_min, delta_prime_max
        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            if self.is_excluded(delta_sigma, delta_epsilon, mid):
                hi = mid
            else:
                lo = mid

        return (lo + hi) / 2


class DeltaEpsilonPrimeBoundComputer:
    """
    Compute Δε' bounds over a 2D grid of (Δσ, Δε) values.

    This allows reproducing Figure 7 from El-Showk et al. (2012).
    """

    def __init__(self, d: int = 3, max_deriv: int = 5):
        self.solver = GapBootstrapSolver(d, max_deriv)
        self.max_deriv = max_deriv

    def compute_bound_along_curve(
        self,
        delta_sigma_values: np.ndarray,
        delta_epsilon_func,
        tolerance: float = 0.02,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute Δε' bounds along a curve in (Δσ, Δε) space.

        For the Ising-like plot, delta_epsilon_func should return the
        boundary value of Δε at each Δσ (approximately where the Ising
        model lives).

        Args:
            delta_sigma_values: Array of Δσ values
            delta_epsilon_func: Function mapping Δσ -> Δε
            tolerance: Binary search tolerance
            verbose: Print progress

        Returns:
            Array of shape (N, 3) with [Δσ, Δε, Δε'_bound]
        """
        results = []
        n_total = len(delta_sigma_values)

        for i, ds in enumerate(delta_sigma_values):
            de = delta_epsilon_func(ds)

            if verbose:
                print(f"[{i+1}/{n_total}] Δσ={ds:.4f}, Δε={de:.4f} ... ", end='', flush=True)

            bound = self.solver.find_delta_epsilon_prime_bound(
                ds, de, tolerance=tolerance
            )
            results.append([ds, de, bound])

            if verbose:
                print(f"Δε' ≤ {bound:.4f}")

        return np.array(results)

    def compute_ising_plot(
        self,
        delta_sigma_min: float = 0.50,
        delta_sigma_max: float = 0.60,
        n_points: int = 50,
        tolerance: float = 0.02,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute data for reproducing the Ising Δε' bound plot.

        Uses the approximate relation between Δσ and Δε near the Ising point.
        The Ising model sits at roughly:
            Δσ ≈ 0.5182, Δε ≈ 1.4127

        For the plot, we need to track the boundary curve. A simple
        approximation is to use the unitarity bound Δε ≥ 1 as baseline
        and linearly interpolate toward the Ising value.
        """
        delta_sigmas = np.linspace(delta_sigma_min, delta_sigma_max, n_points)

        # Approximate Δε as function of Δσ along the boundary
        # This is a crude approximation - the real boundary comes from
        # the bootstrap itself. For demonstration, we use:
        # - At Δσ = 0.5 (free field): Δε ≈ 1.0
        # - At Δσ = 0.518 (Ising): Δε ≈ 1.41
        # - Linear interpolation for others
        def delta_epsilon_boundary(ds):
            # Piecewise linear approximation to the kink
            if ds <= 0.518:
                # Steep rise from free field to Ising
                return 1.0 + (ds - 0.5) * (1.41 - 1.0) / (0.518 - 0.5)
            else:
                # Gradual rise after the kink
                return 1.41 + (ds - 0.518) * 2.5

        if verbose:
            print(f"Computing Δε' bounds for {n_points} points")
            print(f"Δσ range: [{delta_sigma_min}, {delta_sigma_max}]")
            print(f"Max derivative order: {self.max_deriv}")
            print("=" * 50)

        return self.compute_bound_along_curve(
            delta_sigmas, delta_epsilon_boundary, tolerance, verbose
        )


# For quick testing
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Gap Bootstrap Solver - Δε' Bounds")
    print("=" * 60)

    solver = GapBootstrapSolver(d=3, max_deriv=5)

    # Test at the 3D Ising point
    ds_ising = 0.518
    de_ising = 1.41

    print(f"\nTesting at 3D Ising point: Δσ={ds_ising}, Δε={de_ising}")
    print("-" * 50)

    for de_prime in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        t0 = time.time()
        excluded = solver.is_excluded(ds_ising, de_ising, de_prime)
        t1 = time.time()
        status = "EXCLUDED" if excluded else "ALLOWED"
        print(f"  Δε' = {de_prime:.2f}: {status} ({t1-t0:.3f}s)")

    print("\nFinding Δε' bound at Ising point...")
    t0 = time.time()
    bound = solver.find_delta_epsilon_prime_bound(ds_ising, de_ising, tolerance=0.05)
    t1 = time.time()
    print(f"  Upper bound: Δε' ≤ {bound:.4f} ({t1-t0:.2f}s)")
    print(f"  Literature value (El-Showk 2012): Δε' ~ 3.8 at the kink")

    # Quick grid test
    print("\n" + "=" * 60)
    print("Computing bounds along approximate Ising curve")
    print("=" * 60)

    computer = DeltaEpsilonPrimeBoundComputer(d=3, max_deriv=5)

    t0 = time.time()
    results = computer.compute_ising_plot(
        delta_sigma_min=0.50,
        delta_sigma_max=0.55,
        n_points=6,
        tolerance=0.1,
        verbose=True
    )
    t1 = time.time()

    print(f"\nTotal time: {t1-t0:.1f}s")
    print("\nResults:")
    print("-" * 40)
    print("  Δσ      Δε      Δε'_max")
    print("-" * 40)
    for ds, de, dep in results:
        print(f"  {ds:.4f}  {de:.4f}  {dep:.4f}")
