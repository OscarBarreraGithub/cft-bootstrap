"""
Mixed correlator bootstrap for the 3D Ising model.

This module implements the mixed correlator conformal bootstrap using multiple
four-point functions simultaneously:
- <sigma sigma sigma sigma> (ssss)
- <sigma sigma epsilon epsilon> (ssee)
- <epsilon epsilon epsilon epsilon> (eeee)

The mixed correlator approach provides much stronger constraints than single
correlator bootstrap, creating the sharp kink at the Ising point that allows
precise determination of critical exponents.

References:
    - El-Showk et al., "Solving the 3D Ising Model with the Conformal Bootstrap"
      arXiv:1203.6064 (2012)
    - Kos, Poland, Simmons-Duffin, "Bootstrapping Mixed Correlators in the 3D Ising Model"
      arXiv:1406.4858 (2014)
"""

import numpy as np
import warnings
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    cp = None

try:
    from .taylor_conformal_blocks import TaylorCrossingVector
except ImportError:
    from taylor_conformal_blocks import TaylorCrossingVector


@dataclass
class MixedCorrelatorResult:
    """Result from mixed correlator bootstrap computation."""
    excluded: bool
    delta_sigma: float
    delta_epsilon: float
    delta_epsilon_prime: float
    method: str
    solver_status: Optional[str] = None
    computation_time: Optional[float] = None


class TwoCorrelatorBootstrapSolver:
    """
    Two-correlator bootstrap using ssss + eeee (simplified, no matrix SDP).

    This solver requires that BOTH correlators have consistent constraints:
    - alpha_ssss . F^{ssss}_O >= 0 for all operators O
    - alpha_eeee . F^{eeee}_O >= 0 for all operators O

    While this ignores the OPE coefficient correlations (no ssee cross-terms),
    it still provides stronger constraints than single correlator because the
    same operator spectrum must satisfy both systems simultaneously.

    This is a "quick win" implementation that reuses existing TaylorCrossingVector.

    Attributes:
        d: Spacetime dimension (only d=3 implemented)
        max_deriv: Maximum derivative order for constraints
        n_constraints: Number of F-vector components = (max_deriv + 1) // 2
    """

    def __init__(self, d: int = 3, max_deriv: int = 21):
        """
        Initialize the two-correlator bootstrap solver.

        Args:
            d: Spacetime dimension (only d=3 implemented)
            max_deriv: Maximum derivative order (21 gives 11 constraints)
        """
        if d != 3:
            raise ValueError("Only d=3 is implemented")
        if not HAS_CVXPY:
            raise ImportError(
                "CVXPY is required for mixed correlator bootstrap. "
                "Install with: pip install cvxpy"
            )

        self.d = d
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2

    def is_excluded(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        delta_max: float = 30.0,
        n_samples: int = 150,
        verbose: bool = False
    ) -> bool:
        """
        Check if point (Delta_sigma, Delta_epsilon, Delta_epsilon') is excluded.

        The point is excluded if we can find linear functionals alpha_ssss and
        alpha_eeee such that:
        - Normalization: alpha_ssss . F^{ssss}_id = 1
        - Positivity for epsilon: alpha_ssss . F^{ssss}_eps >= 0
                                  alpha_eeee . F^{eeee}_eps >= 0
        - Positivity for gap: alpha_ssss . F^{ssss}_O >= 0 for all O with Delta >= Delta_eps'
                              alpha_eeee . F^{eeee}_O >= 0 for all O with Delta >= Delta_eps'

        Args:
            delta_sigma: External operator dimension (sigma)
            delta_epsilon: Dimension of first Z2-even scalar (epsilon)
            delta_epsilon_prime: Gap for second Z2-even scalar
            delta_max: Maximum dimension to sample
            n_samples: Number of operators to sample above the gap
            verbose: Print solver status

        Returns:
            True if EXCLUDED (constraints are satisfiable, point ruled out)
            False if ALLOWED (no excluding functional found)
        """
        # Create crossing vectors for both correlators
        # ssss uses delta_sigma as external dimension
        # eeee uses delta_epsilon as external dimension
        cross_ssss = TaylorCrossingVector(delta_sigma, self.max_deriv)
        cross_eeee = TaylorCrossingVector(delta_epsilon, self.max_deriv)

        # Identity F-vectors
        F_id_ssss = cross_ssss.build_F_vector(0)
        F_id_eeee = cross_eeee.build_F_vector(0)

        # First scalar (epsilon) F-vectors
        # In ssss: epsilon appears with dimension delta_epsilon
        # In eeee: epsilon also appears (it's the external operator's OPE)
        F_eps_ssss = cross_ssss.build_F_vector(delta_epsilon)
        F_eps_eeee = cross_eeee.build_F_vector(delta_epsilon)

        # Sample operators above the gap
        deltas = np.linspace(delta_epsilon_prime, delta_max, n_samples)

        # Setup CVXPY problem
        n = self.n_constraints
        alpha_ssss = cp.Variable(n)
        alpha_eeee = cp.Variable(n)

        constraints = [
            # Normalization on ssss (standard choice)
            alpha_ssss @ F_id_ssss == 1,
            # First scalar (epsilon) must have positive coefficient in both
            alpha_ssss @ F_eps_ssss >= 0,
            alpha_eeee @ F_eps_eeee >= 0,
        ]

        # Positivity for all operators above the gap (BOTH correlators)
        for delta in deltas:
            F_ssss = cross_ssss.build_F_vector(delta)
            F_eeee = cross_eeee.build_F_vector(delta)
            constraints.append(alpha_ssss @ F_ssss >= 0)
            constraints.append(alpha_eeee @ F_eeee >= 0)

        # Solve feasibility problem
        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)

            if verbose:
                print(f"  Solver status: {prob.status}")

            return prob.status == cp.OPTIMAL

        except Exception as e:
            if verbose:
                print(f"  Solver failed: {e}")
            return False

    def find_delta_epsilon_prime_bound(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: Optional[float] = None,
        delta_prime_max: float = 8.0,
        tolerance: float = 0.01,
        verbose: bool = False
    ) -> float:
        """
        Find the upper bound on Delta_epsilon' using binary search.

        Searches for the smallest Delta_eps' such that the point is excluded.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: Assumed first Z2-even scalar dimension
            delta_prime_min: Minimum Delta_eps' to consider (default: delta_epsilon + 0.1)
            delta_prime_max: Maximum Delta_eps' to consider
            tolerance: Binary search tolerance
            verbose: Print progress

        Returns:
            Upper bound on Delta_epsilon' (inf if no bound found)
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        if verbose:
            print(f"Finding Delta_eps' bound for Delta_sigma={delta_sigma}, Delta_eps={delta_epsilon}")
            print(f"  Search range: [{delta_prime_min}, {delta_prime_max}]")

        # Check boundaries
        if self.is_excluded(delta_sigma, delta_epsilon, delta_prime_min, verbose=verbose):
            if verbose:
                print(f"  Lower bound already excluded")
            return delta_prime_min

        if not self.is_excluded(delta_sigma, delta_epsilon, delta_prime_max, verbose=verbose):
            if verbose:
                print(f"  Upper bound not excluded - no finite bound")
            return float('inf')

        # Binary search
        lo, hi = delta_prime_min, delta_prime_max
        iteration = 0

        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            excluded = self.is_excluded(delta_sigma, delta_epsilon, mid)

            if verbose:
                status = "EXCLUDED" if excluded else "ALLOWED"
                print(f"  Iteration {iteration}: Delta_eps'={mid:.4f} -> {status}")

            if excluded:
                hi = mid
            else:
                lo = mid
            iteration += 1

        bound = (lo + hi) / 2
        if verbose:
            print(f"  Final bound: Delta_eps' <= {bound:.4f}")

        return bound


class MixedCorrelatorBootstrapSolver:
    """
    Full mixed correlator bootstrap with 2x2 matrix SDP constraints.

    For each operator O in the OPE, the constraint is a 2x2 positive semidefinite
    matrix capturing the correlations between OPE coefficients:

        | alpha^{ssss} . F^{ssss}_O    alpha^{ssee} . F^{ssee}_O |
        | alpha^{ssee} . F^{ssee}_O    alpha^{eeee} . F^{eeee}_O |  >> 0

    This matrix structure arises because:
    - Operator O contributes to ssss with coefficient lambda_{sigma sigma O}^2
    - Operator O contributes to eeee with coefficient lambda_{epsilon epsilon O}^2
    - Operator O contributes to ssee with coefficient lambda_{sigma sigma O} * lambda_{epsilon epsilon O}

    The matrix being positive semidefinite ensures:
        (alpha^{ssss} . F^{ssss}_O) * (alpha^{eeee} . F^{eeee}_O) >= (alpha^{ssee} . F^{ssee}_O)^2

    which is the Cauchy-Schwarz inequality for OPE coefficients.

    Attributes:
        d: Spacetime dimension (only d=3 implemented)
        max_deriv: Maximum derivative order for constraints
        n_constraints: Number of F-vector components per correlator
    """

    def __init__(self, d: int = 3, max_deriv: int = 21):
        """
        Initialize the full mixed correlator bootstrap solver.

        Args:
            d: Spacetime dimension (only d=3 implemented)
            max_deriv: Maximum derivative order (21 gives 11 constraints)
        """
        if d != 3:
            raise ValueError("Only d=3 is implemented")
        if not HAS_CVXPY:
            raise ImportError(
                "CVXPY is required for mixed correlator bootstrap. "
                "Install with: pip install cvxpy"
            )

        self.d = d
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2

    def is_excluded(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        delta_max: float = 30.0,
        n_samples: int = 100,
        verbose: bool = False
    ) -> bool:
        """
        Check if point is excluded using full matrix SDP constraints.

        This uses 2x2 matrix positive semidefiniteness constraints for each
        operator, capturing the OPE coefficient correlations.

        Args:
            delta_sigma: External operator dimension (sigma)
            delta_epsilon: Dimension of first Z2-even scalar (epsilon)
            delta_epsilon_prime: Gap for second Z2-even scalar
            delta_max: Maximum dimension to sample
            n_samples: Number of operators to sample above the gap
            verbose: Print solver status

        Returns:
            True if EXCLUDED, False if ALLOWED
        """
        # Import MixedCrossingVector for ssee computation
        try:
            from .mixed_correlator_blocks import MixedCrossingVector
        except ImportError:
            try:
                from mixed_correlator_blocks import MixedCrossingVector
            except ImportError:
                raise ImportError(
                    "MixedCrossingVector not available. "
                    "Falling back to TwoCorrelatorBootstrapSolver."
                )

        cross = MixedCrossingVector(delta_sigma, delta_epsilon, self.max_deriv)
        n = self.n_constraints

        # Three linear functionals (one per correlator)
        alpha_ssss = cp.Variable(n)
        alpha_ssee = cp.Variable(n)
        alpha_eeee = cp.Variable(n)

        # Identity F-vectors
        F_id_ssss = cross.build_F_vector_ssss(0)
        F_id_ssee = cross.build_F_vector_ssee(0)
        F_id_eeee = cross.build_F_vector_eeee(0)

        constraints = [
            # Normalization
            alpha_ssss @ F_id_ssss == 1,
        ]

        # First scalar (epsilon) - 2x2 matrix constraint
        F_eps_ssss = cross.build_F_vector_ssss(delta_epsilon)
        F_eps_ssee = cross.build_F_vector_ssee(delta_epsilon)
        F_eps_eeee = cross.build_F_vector_eeee(delta_epsilon)

        M_eps = cp.bmat([
            [cp.reshape(alpha_ssss @ F_eps_ssss, (1, 1)),
             cp.reshape(alpha_ssee @ F_eps_ssee, (1, 1))],
            [cp.reshape(alpha_ssee @ F_eps_ssee, (1, 1)),
             cp.reshape(alpha_eeee @ F_eps_eeee, (1, 1))]
        ])
        constraints.append(M_eps >> 0)

        # Operators above the gap - 2x2 matrix constraints
        deltas = np.linspace(delta_epsilon_prime, delta_max, n_samples)

        for delta in deltas:
            F_ssss = cross.build_F_vector_ssss(delta)
            F_ssee = cross.build_F_vector_ssee(delta)
            F_eeee = cross.build_F_vector_eeee(delta)

            M = cp.bmat([
                [cp.reshape(alpha_ssss @ F_ssss, (1, 1)),
                 cp.reshape(alpha_ssee @ F_ssee, (1, 1))],
                [cp.reshape(alpha_ssee @ F_ssee, (1, 1)),
                 cp.reshape(alpha_eeee @ F_eeee, (1, 1))]
            ])
            constraints.append(M >> 0)

        # Solve
        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)

            if verbose:
                print(f"  Solver status: {prob.status}")

            return prob.status == cp.OPTIMAL

        except Exception as e:
            if verbose:
                print(f"  Matrix SDP solver failed: {e}")
            # Fallback to two-correlator
            two_corr = TwoCorrelatorBootstrapSolver(self.d, self.max_deriv)
            return two_corr.is_excluded(
                delta_sigma, delta_epsilon, delta_epsilon_prime,
                delta_max, n_samples, verbose
            )

    def find_delta_epsilon_prime_bound(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: Optional[float] = None,
        delta_prime_max: float = 8.0,
        tolerance: float = 0.01,
        verbose: bool = False
    ) -> float:
        """
        Find the upper bound on Delta_epsilon' using binary search.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: Assumed first Z2-even scalar dimension
            delta_prime_min: Minimum Delta_eps' to consider
            delta_prime_max: Maximum Delta_eps' to consider
            tolerance: Binary search tolerance
            verbose: Print progress

        Returns:
            Upper bound on Delta_epsilon'
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        if verbose:
            print(f"Finding Delta_eps' bound (matrix SDP)")
            print(f"  Delta_sigma={delta_sigma}, Delta_eps={delta_epsilon}")

        # Check boundaries
        if self.is_excluded(delta_sigma, delta_epsilon, delta_prime_min, verbose=verbose):
            return delta_prime_min

        if not self.is_excluded(delta_sigma, delta_epsilon, delta_prime_max, verbose=verbose):
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


def compare_single_vs_mixed(
    delta_sigma: float = 0.518,
    delta_epsilon: float = 1.41,
    max_deriv: int = 11,
    tolerance: float = 0.05,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compare single correlator vs two-correlator bootstrap bounds.

    Args:
        delta_sigma: External dimension (default: Ising point)
        delta_epsilon: First scalar dimension (default: Ising epsilon)
        max_deriv: Maximum derivative order
        tolerance: Binary search tolerance
        verbose: Print results

    Returns:
        Dictionary with bounds from each method
    """
    from .taylor_conformal_blocks import HighOrderGapBootstrapSolver

    results = {}

    if verbose:
        print("=" * 60)
        print("Comparing single vs two-correlator bootstrap")
        print(f"Point: Delta_sigma={delta_sigma}, Delta_epsilon={delta_epsilon}")
        print(f"Constraints: {(max_deriv + 1) // 2}")
        print("=" * 60)

    # Single correlator
    if verbose:
        print("\n1. Single correlator (ssss only):")
    single_solver = HighOrderGapBootstrapSolver(d=3, max_deriv=max_deriv)
    single_bound = single_solver.find_delta_epsilon_prime_bound(
        delta_sigma, delta_epsilon, tolerance=tolerance
    )
    results['single_correlator'] = single_bound
    if verbose:
        print(f"   Bound: Delta_eps' <= {single_bound:.4f}")

    # Two correlator
    if verbose:
        print("\n2. Two correlator (ssss + eeee):")
    two_solver = TwoCorrelatorBootstrapSolver(d=3, max_deriv=max_deriv)
    two_bound = two_solver.find_delta_epsilon_prime_bound(
        delta_sigma, delta_epsilon, tolerance=tolerance
    )
    results['two_correlator'] = two_bound
    if verbose:
        print(f"   Bound: Delta_eps' <= {two_bound:.4f}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Single correlator: {single_bound:.4f}")
        print(f"  Two correlator:    {two_bound:.4f}")
        print(f"  Improvement:       {two_bound - single_bound:+.4f}")
        print(f"  Reference (El-Showk 2012): ~3.8")
        print("=" * 60)

    return results


# Test function
def test_two_correlator():
    """Test the two-correlator bootstrap implementation."""
    print("Testing Two-Correlator Bootstrap")
    print("=" * 60)

    solver = TwoCorrelatorBootstrapSolver(d=3, max_deriv=11)

    # Ising point
    ds_ising = 0.518
    de_ising = 1.41

    print(f"\nIsing point: Delta_sigma={ds_ising}, Delta_epsilon={de_ising}")
    print(f"Number of constraints: {solver.n_constraints}")

    # Test a few gaps
    print("\nTesting exclusion at various gaps:")
    for de_prime in [2.0, 2.5, 3.0, 3.5, 4.0]:
        excluded = solver.is_excluded(ds_ising, de_ising, de_prime)
        status = "EXCLUDED" if excluded else "ALLOWED"
        print(f"  Delta_eps' = {de_prime:.1f}: {status}")

    # Find bound
    print("\nFinding bound with binary search:")
    bound = solver.find_delta_epsilon_prime_bound(
        ds_ising, de_ising, tolerance=0.05, verbose=True
    )
    print(f"\nFinal bound: Delta_eps' <= {bound:.4f}")
    print(f"Reference (El-Showk 2012): ~3.8")


if __name__ == "__main__":
    test_two_correlator()
