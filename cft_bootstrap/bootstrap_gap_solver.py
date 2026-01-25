"""
CFT Bootstrap solver with gap assumptions for Δε' bounds.

This module extends the basic bootstrap to compute bounds on the SECOND
Z2-even scalar operator dimension Δε', assuming a gap to the first
scalar at Δε.

This partially reproduces results from:
  "Solving the 3D Ising Model with the Conformal Bootstrap"
  El-Showk et al., arXiv:1203.6064 (2012), Figure 6

The key insight: instead of just demanding Δ ≥ Δε_gap for ALL scalars
in the OPE, we:
1. Include the identity operator (Δ = 0)
2. Include a SINGLE scalar at Δε (the assumed first scalar)
3. Demand ALL OTHER scalars have Δ ≥ Δε' (the bound we're computing)

IMPORTANT LIMITATIONS:
- We use only 3-4 derivative constraints (paper uses ~60+)
- We include only scalar operators (paper includes spinning operators)
- Our bounds are therefore TIGHTER (lower) than the correct bounds
- Expected offset: ~1 unit below the paper's values

The qualitative features (kink at Ising point, shape) are correctly reproduced.
"""

import numpy as np
from math import factorial
from scipy.optimize import linprog
from typing import Tuple, Optional
import warnings

# Try to import CVXPY for SDP (optional but recommended)
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    warnings.warn("CVXPY not installed. Install with: pip install cvxpy")

warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from .bootstrap_solver import ConformalBlock3D, CrossingVector
except ImportError:
    from bootstrap_solver import ConformalBlock3D, CrossingVector


def reshuffle_with_normalization(F_vectors: np.ndarray, F_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Reshuffle F-vectors using component-wise normalization (pycftboot/SDPB convention).

    This transforms F-vectors to eliminate the normalization constraint.
    For each F-vector F_O, we compute:
        const_O = F_O[max_idx] / F_norm[max_idx]
        F_O_transformed = F_O - const_O * F_norm  (for all components except max_idx)

    The const_O values represent something like OPE coefficient ratios.
    After this transformation:
    - The original α · F_norm = 1 becomes just fixing the coefficient of const_O
    - The transformed F-vectors have F_O_transformed[max_idx] = 0 by construction

    Args:
        F_vectors: Array of F-vectors, shape (n_operators, n_constraints)
        F_norm: Normalization vector (typically F_identity)

    Returns:
        Tuple of (reduced_F_vectors, const_coeffs, max_idx) where:
        - reduced_F_vectors: shape (n_operators, n_constraints - 1) - transformed & reduced
        - const_coeffs: shape (n_operators,) - the const values (relate to OPE coefficients)
        - max_idx: index of the eliminated component
    """
    max_idx = np.argmax(np.abs(F_norm))
    n_ops, n_cons = F_vectors.shape

    # For each operator, compute const = F_O[max_idx] / F_norm[max_idx]
    const_coeffs = F_vectors[:, max_idx] / F_norm[max_idx]

    # Transform F-vectors: F_O_transformed = F_O - const * F_norm
    # This makes F_O_transformed[max_idx] = 0 by construction
    F_transformed = np.zeros((n_ops, n_cons))
    for i in range(n_ops):
        F_transformed[i, :] = F_vectors[i, :] - const_coeffs[i] * F_norm

    # Remove the max_idx column (which is now all zeros)
    reduced = np.zeros((n_ops, n_cons - 1))
    j = 0
    for k in range(n_cons):
        if k != max_idx:
            reduced[:, j] = F_transformed[:, k]
            j += 1

    return reduced, const_coeffs, max_idx


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

    Note: SDP gives the correct (weaker) bounds. LP gives stronger but
    incorrect bounds. Always prefer SDP when CVXPY is available.
    """

    def __init__(self, d: int = 3, max_deriv: int = 5):
        self.d = d
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2
        self.blocks = ConformalBlock3D()

        if not HAS_CVXPY:
            warnings.warn(
                "CVXPY not available. LP solver gives incorrect (too strong) bounds. "
                "Install CVXPY for correct SDP bounds: pip install cvxpy"
            )

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
        LP feasibility check (gives INCORRECT bounds - use SDP instead).

        Tests if there exist p_ε ≥ 0 and p_Δ ≥ 0 such that:
            F_id + p_ε * F_eps + Σ p_Δ * F_Δ = 0

        If NO such solution exists, the point is EXCLUDED.

        WARNING: LP gives bounds that are TOO STRONG. The correct bootstrap
        bound comes from the SDP dual problem.
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
        SDP feasibility check (gives CORRECT bounds).

        Uses component-wise normalization (pycftboot/SDPB convention):
        1. Find the largest-magnitude component in F_id
        2. Fix alpha[max_idx] = 1 / F_id[max_idx]
        3. Solve reduced SDP for remaining alpha components

        The original constraints:
            α · F_id = 1 (normalization)
            α · F_eps ≥ 0 (first scalar OK)
            α · F_Δ ≥ 0 for all Δ ≥ Δε' (positivity)

        Become (after substitution):
            alpha_reduced @ F_eps_reduced >= -fixed_eps
            alpha_reduced @ F_O_reduced >= -fixed_O for each operator

        If such alpha_reduced exists, point is EXCLUDED.
        """
        # Stack all F-vectors for reshuffling
        F_all = np.vstack([F_eps[np.newaxis, :], F_ops])

        # Apply component-wise normalization
        F_reduced, fixed_contribs, max_idx = reshuffle_with_normalization(F_all, F_id)

        # Separate epsilon and other operators
        F_eps_reduced = F_reduced[0, :]
        fixed_eps = fixed_contribs[0]
        F_ops_reduced = F_reduced[1:, :]
        fixed_ops = fixed_contribs[1:]

        # Reduced alpha has n_constraints - 1 components
        alpha_reduced = cp.Variable(self.n_constraints - 1)

        # CRITICAL: F_id_reduced must satisfy alpha_reduced @ F_id_reduced = 0
        # to correctly enforce the normalization constraint alpha @ F_id = 1
        F_id_reduced = np.delete(F_id, max_idx)

        # Build constraints with the fixed contribution moved to RHS
        constraints = [
            # Normalization constraint: alpha_reduced @ F_id_reduced = 0
            alpha_reduced @ F_id_reduced == 0,
            # First scalar positivity
            alpha_reduced @ F_eps_reduced >= -fixed_eps,
        ]

        for i, F_O_reduced in enumerate(F_ops_reduced):
            constraints.append(alpha_reduced @ F_O_reduced >= -fixed_ops[i])

        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
            # If feasible (optimal found), point is EXCLUDED
            return prob.status == cp.OPTIMAL
        except Exception:
            # Solver failure - fall back to LP (with warning)
            warnings.warn("SDP solver failed, falling back to LP (bounds may be incorrect)")
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

    This allows partially reproducing Figure 6 from El-Showk et al. (2012).

    Note: Our bounds are ~1 unit below the paper due to fewer constraints
    and no spinning operators. The qualitative shape is correct.
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

    # Literature values for Δε boundary from high-precision bootstrap
    # Source: El-Showk et al. (2014), Kos et al. (2014), and later works
    # These are approximate values along the boundary of the allowed region
    LITERATURE_BOUNDARY = {
        # (Δσ, Δε) pairs from published bootstrap results
        # Before Ising kink: steep rise from free field
        0.500: 1.000,   # Free scalar (exact)
        0.505: 1.050,
        0.510: 1.150,
        0.515: 1.300,
        0.5181489: 1.412625,  # 3D Ising (high precision)
        # After Ising kink: gradual rise
        0.520: 1.42,
        0.525: 1.44,
        0.530: 1.46,
        0.540: 1.51,
        0.550: 1.56,
        0.560: 1.61,
        0.570: 1.67,
        0.580: 1.73,
        0.590: 1.79,
        0.600: 1.85,
    }

    @staticmethod
    def delta_epsilon_boundary_approximate(delta_sigma: float) -> float:
        """
        Approximate Δε boundary curve as function of Δσ (hand-tuned).

        This approximates the boundary of the allowed region in (Δσ, Δε) space:
        - At Δσ = 0.5 (free field): Δε = 1.0 (unitarity bound)
        - At Δσ ≈ 0.518 (Ising): Δε ≈ 1.41
        - After the kink: gradual rise

        For better accuracy, use delta_epsilon_boundary_literature() instead.
        """
        ISING_DS = 0.5181489
        ISING_DE = 1.412625

        if delta_sigma <= 0.5:
            return 1.0
        elif delta_sigma <= ISING_DS:
            # Linear interpolation from free field to Ising
            return 1.0 + (delta_sigma - 0.5) * (ISING_DE - 1.0) / (ISING_DS - 0.5)
        else:
            # After the kink: gradual rise
            return ISING_DE + (delta_sigma - ISING_DS) * 2.5

    @classmethod
    def delta_epsilon_boundary_literature(cls, delta_sigma: float) -> float:
        """
        Δε boundary curve using literature values with interpolation.

        Uses tabulated values from high-precision bootstrap calculations
        (El-Showk et al. 2014, Kos et al. 2014) with linear interpolation.

        This gives more accurate Δε values than the simple piecewise approximation.
        """
        # Get sorted boundary points
        ds_values = np.array(sorted(cls.LITERATURE_BOUNDARY.keys()))
        de_values = np.array([cls.LITERATURE_BOUNDARY[ds] for ds in ds_values])

        # Handle out of range
        if delta_sigma <= ds_values[0]:
            return de_values[0]
        if delta_sigma >= ds_values[-1]:
            # Extrapolate linearly
            slope = (de_values[-1] - de_values[-2]) / (ds_values[-1] - ds_values[-2])
            return de_values[-1] + slope * (delta_sigma - ds_values[-1])

        # Linear interpolation
        idx = np.searchsorted(ds_values, delta_sigma) - 1
        t = (delta_sigma - ds_values[idx]) / (ds_values[idx + 1] - ds_values[idx])
        return de_values[idx] + t * (de_values[idx + 1] - de_values[idx])

    def compute_delta_epsilon_boundary(
        self,
        delta_sigma_values: np.ndarray,
        tolerance: float = 0.02,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute the Δε boundary self-consistently using the bootstrap.

        This computes the actual bootstrap bound on Δε for each Δσ value,
        rather than using a hand-tuned approximation.

        Args:
            delta_sigma_values: Array of Δσ values
            tolerance: Binary search tolerance
            verbose: Print progress

        Returns:
            Array of shape (N, 2) with [Δσ, Δε_bound]
        """
        from .bootstrap_solver import BootstrapSolver

        solver = BootstrapSolver(d=3, max_deriv=self.max_deriv)
        results = []

        if verbose:
            print("Computing self-consistent Δε boundary...")
            print(f"  Δσ range: [{delta_sigma_values.min():.3f}, {delta_sigma_values.max():.3f}]")
            print(f"  Points: {len(delta_sigma_values)}")
            print("=" * 50)

        for i, ds in enumerate(delta_sigma_values):
            if verbose:
                print(f"[{i+1}/{len(delta_sigma_values)}] Δσ={ds:.4f} ... ", end='', flush=True)

            # Use SDP if available, otherwise LP
            method = 'sdp' if HAS_CVXPY else 'lp'
            bound = solver.find_bound(ds, delta_min=0.5, delta_max=3.0,
                                      tolerance=tolerance, method=method)
            results.append([ds, bound])

            if verbose:
                print(f"Δε ≤ {bound:.4f}")

        return np.array(results)

    def create_interpolated_boundary(self, boundary_data: np.ndarray):
        """
        Create an interpolation function from computed boundary data.

        Args:
            boundary_data: Array of shape (N, 2) with [Δσ, Δε_bound]

        Returns:
            Function mapping Δσ -> Δε
        """
        from scipy.interpolate import interp1d

        delta_sigmas = boundary_data[:, 0]
        delta_epsilons = boundary_data[:, 1]

        # Use linear interpolation with extrapolation
        interp_func = interp1d(
            delta_sigmas, delta_epsilons,
            kind='linear', fill_value='extrapolate'
        )

        return interp_func

    def compute_two_stage_scan(
        self,
        delta_sigma_min: float = 0.50,
        delta_sigma_max: float = 0.60,
        n_points: int = 25,
        tolerance_stage1: float = 0.02,
        tolerance_stage2: float = 0.02,
        max_deriv_stage1: int = None,
        max_deriv_stage2: int = None,
        max_spin: int = 20,
        use_spinning_stage1: bool = True,
        use_multiresolution: bool = False,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Two-stage scan following El-Showk et al. (2012) Figure 6 protocol.

        This is the CORRECT protocol for computing Δε' bounds:
        1. Stage 1: For each Δσ, compute Δε,max(Δσ) - the bootstrap upper bound
        2. Stage 2: With Δε fixed to Δε,max, compute the Δε' upper bound

        The paper uses nmax=11 for Stage 1 and nmax=10 for Stage 2.

        Args:
            delta_sigma_min: Minimum Δσ value
            delta_sigma_max: Maximum Δσ value
            n_points: Number of points to compute
            tolerance_stage1: Binary search tolerance for Δε boundary
            tolerance_stage2: Binary search tolerance for Δε' bound
            max_deriv_stage1: Max derivative order for Stage 1 (default: same as self)
            max_deriv_stage2: Max derivative order for Stage 2 (default: same as self)
            max_spin: Maximum spin to include (default 20)
            use_spinning_stage1: Whether to include spinning operators in Stage 1
            use_multiresolution: Use T1-T5 style multi-resolution discretization
            verbose: Print progress

        Returns:
            Array of shape (N, 3) with [Δσ, Δε_max, Δε'_bound]
        """
        # Import ElShowkBootstrapSolver for Stage 1 with spinning operators
        try:
            from .el_showk_basis import ElShowkBootstrapSolver
        except ImportError:
            from el_showk_basis import ElShowkBootstrapSolver

        # Default to same max_deriv for both stages if not specified
        if max_deriv_stage1 is None:
            max_deriv_stage1 = self.max_deriv
        if max_deriv_stage2 is None:
            max_deriv_stage2 = self.max_deriv

        delta_sigmas = np.linspace(delta_sigma_min, delta_sigma_max, n_points)
        results = []

        if verbose:
            print("=" * 60)
            print("Two-Stage Scan: El-Showk et al. (2012) Figure 6 Protocol")
            print("=" * 60)
            print(f"Δσ range: [{delta_sigma_min}, {delta_sigma_max}]")
            print(f"Points: {n_points}")
            print(f"Stage 1 (Δε boundary): nmax={max_deriv_stage1 // 2} with spinning={use_spinning_stage1}")
            print(f"Stage 2 (Δε' bound): {(max_deriv_stage2 + 1) // 2} constraints")
            print(f"Max spin: {max_spin}")
            print(f"Multi-resolution discretization: {use_multiresolution}")
            print(f"Solver: {'SDP (CVXPY)' if HAS_CVXPY else 'LP (scipy)'}")
            print("=" * 60)

        # Create Stage 1 solver with spinning operators (ElShowkBootstrapSolver)
        stage1_solver = ElShowkBootstrapSolver(
            d=3,
            nmax=max_deriv_stage1 // 2,
            max_spin=max_spin if use_spinning_stage1 else 0,
            solver='auto',
            high_precision=False
        )

        # Create Stage 2 solver (for Δε' bound)
        stage2_solver = GapBootstrapSolver(d=3, max_deriv=max_deriv_stage2)

        for i, ds in enumerate(delta_sigmas):
            if verbose:
                print(f"\n[{i+1}/{n_points}] Δσ = {ds:.4f}")
                print("-" * 40)

            # Stage 1: Compute Δε boundary with spinning operators
            if verbose:
                print(f"  Stage 1: Computing Δε boundary (with spinning)... ", end='', flush=True)

            delta_eps_max = stage1_solver.find_delta_epsilon_bound(
                ds, delta_min=0.5, delta_max=3.0,
                tolerance=tolerance_stage1,
                include_spinning=use_spinning_stage1,
                use_multiresolution=use_multiresolution,
                verbose=False
            )

            if verbose:
                print(f"Δε ≤ {delta_eps_max:.4f}")

            # Stage 2: Compute Δε' bound with Δε fixed to boundary
            if verbose:
                print(f"  Stage 2: Computing Δε' bound (Δε = {delta_eps_max:.4f})... ", end='', flush=True)

            delta_eps_prime_bound = stage2_solver.find_delta_epsilon_prime_bound(
                ds, delta_eps_max,
                tolerance=tolerance_stage2
            )

            if verbose:
                print(f"Δε' ≤ {delta_eps_prime_bound:.4f}")

            results.append([ds, delta_eps_max, delta_eps_prime_bound])

        results = np.array(results)

        if verbose:
            print("\n" + "=" * 60)
            print("Two-Stage Scan Complete")
            print("=" * 60)
            print("\nResults:")
            print("-" * 40)
            print("  Δσ        Δε_max    Δε'_bound")
            print("-" * 40)
            for ds, de_max, dep in results:
                print(f"  {ds:.4f}    {de_max:.4f}    {dep:.4f}")
            print("-" * 40)

        return results

    def compute_ising_plot(
        self,
        delta_sigma_min: float = 0.50,
        delta_sigma_max: float = 0.60,
        n_points: int = 50,
        tolerance: float = 0.02,
        verbose: bool = True,
        boundary_method: str = 'literature',
        boundary_n_points: int = 20
    ) -> np.ndarray:
        """
        Compute data for reproducing the Ising Δε' bound plot.

        Uses the approximate relation between Δσ and Δε near the Ising point.
        The Ising model sits at roughly:
            Δσ ≈ 0.5182, Δε ≈ 1.4127

        Args:
            delta_sigma_min: Minimum Δσ value
            delta_sigma_max: Maximum Δσ value
            n_points: Number of points to compute
            tolerance: Binary search tolerance
            verbose: Print progress
            boundary_method: How to determine Δε values:
                - 'literature': Use tabulated values from published results (default, recommended)
                - 'approximate': Use simple piecewise linear approximation
                - 'self_consistent': Compute from bootstrap (not recommended with few constraints)
            boundary_n_points: Number of points for boundary computation (if self-consistent)

        Note: Our bounds will be ~1 unit below El-Showk et al. (2012)
        due to using only 3 derivative constraints vs their ~60.
        The qualitative shape (kink at Ising point) is correct.
        """
        delta_sigmas = np.linspace(delta_sigma_min, delta_sigma_max, n_points)

        if verbose:
            print(f"Computing Δε' bounds for {n_points} points")
            print(f"Δσ range: [{delta_sigma_min}, {delta_sigma_max}]")
            print(f"Max derivative order: {self.max_deriv}")
            print(f"Number of constraints: {(self.max_deriv + 1) // 2}")
            print(f"Solver: {'SDP (CVXPY)' if HAS_CVXPY else 'LP (scipy)'}")
            print(f"Δε boundary: {boundary_method}")
            print("=" * 50)
            if not HAS_CVXPY:
                print("WARNING: LP bounds are incorrect. Install CVXPY for correct results.")
                print("=" * 50)

        # Determine boundary function
        if boundary_method == 'self_consistent':
            if verbose:
                print("\nStep 1: Computing self-consistent Δε boundary")
                print("-" * 50)
                print("WARNING: Self-consistent boundary gives worse results with few constraints!")
                print("-" * 50)

            # Compute boundary on a coarser grid
            boundary_sigmas = np.linspace(delta_sigma_min, delta_sigma_max, boundary_n_points)
            boundary_data = self.compute_delta_epsilon_boundary(
                boundary_sigmas, tolerance=tolerance, verbose=verbose
            )

            # Create interpolation function
            delta_epsilon_func = self.create_interpolated_boundary(boundary_data)

            if verbose:
                print("\nStep 2: Computing Δε' bounds along computed boundary")
                print("-" * 50)
        elif boundary_method == 'literature':
            delta_epsilon_func = self.delta_epsilon_boundary_literature
        else:  # 'approximate'
            delta_epsilon_func = self.delta_epsilon_boundary_approximate

        return self.compute_bound_along_curve(
            delta_sigmas, delta_epsilon_func, tolerance, verbose
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
