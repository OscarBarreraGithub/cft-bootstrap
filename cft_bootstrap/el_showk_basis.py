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
from typing import List, Tuple, Optional, Dict
from functools import lru_cache
import warnings

try:
    from .bootstrap_solver import ConformalBlock3D
except ImportError:
    from bootstrap_solver import ConformalBlock3D

try:
    from .spinning_conformal_blocks import SpinningConformalBlock
except ImportError:
    from spinning_conformal_blocks import SpinningConformalBlock


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


# ============================================================================
# Multi-Resolution Discretization (T1-T5 Tables from El-Showk et al. 2012)
# ============================================================================

# Table specifications from the paper (Table 2, page 15)
# Each table: (step_size, delta_max, spin_max)
T1_T5_TABLES = {
    'T1': {'step': 2e-5, 'delta_max': 3.0,  'spin_max': 0},     # Fine grid for scalars near gap
    'T2': {'step': 5e-4, 'delta_max': 8.0,  'spin_max': 6},     # Medium grid
    'T3': {'step': 2e-3, 'delta_max': 22.0, 'spin_max': 20},    # Coarser grid
    'T4': {'step': 0.02, 'delta_max': 100.0, 'spin_max': 50},   # Coarse grid for high Δ
    'T5': {'step': 1.0,  'delta_max': 500.0, 'spin_max': 100},  # Very coarse for far tail
}


def get_multiresolution_operators(
    delta_gap: float,
    unitarity_bound_func,
    tables: Dict[str, Dict] = None,
    verbose: bool = False
) -> List[Tuple[float, int]]:
    """
    Generate operator list using multi-resolution T1-T5 discretization.

    Uses progressively coarser grids at higher dimensions and spins,
    matching the El-Showk et al. (2012) discretization strategy.

    Args:
        delta_gap: The gap Δε' (scalars above this dimension)
        unitarity_bound_func: Function(spin) -> min_delta for that spin
        tables: Custom table specs (default: T1-T5 from paper)
        verbose: Print info about operator count

    Returns:
        List of (delta, spin) pairs for all operators to include
    """
    if tables is None:
        tables = T1_T5_TABLES

    operators = []
    seen = set()  # Avoid duplicates

    for table_name, spec in tables.items():
        step = spec['step']
        delta_max = spec['delta_max']
        spin_max = spec['spin_max']

        # For each spin in this table's range
        for spin in range(0, spin_max + 1, 2):  # Even spins only
            delta_min = unitarity_bound_func(spin)

            # For scalars (spin=0), start from the gap
            if spin == 0:
                delta_min = max(delta_min, delta_gap)

            # Sample operators from delta_min to delta_max with given step
            n_samples = int((delta_max - delta_min) / step) + 1
            if n_samples < 1:
                continue

            for i in range(n_samples):
                delta = delta_min + i * step
                if delta > delta_max:
                    break

                key = (round(delta, 6), spin)  # Avoid float precision issues
                if key not in seen:
                    seen.add(key)
                    operators.append((delta, spin))

    # Sort by (spin, delta) for organized processing
    operators.sort(key=lambda x: (x[1], x[0]))

    if verbose:
        n_scalars = sum(1 for _, s in operators if s == 0)
        n_spinning = len(operators) - n_scalars
        max_spin_found = max(s for _, s in operators)
        print(f"Multi-resolution discretization:")
        print(f"  Total operators: {len(operators)}")
        print(f"  Scalars (ℓ=0): {n_scalars}")
        print(f"  Spinning (ℓ>0): {n_spinning}")
        print(f"  Max spin: {max_spin_found}")

    return operators


def get_simplified_multiresolution(
    delta_gap: float,
    unitarity_bound_func,
    max_spin: int = 50,
    n_regions: int = 3,
    verbose: bool = False
) -> List[Tuple[float, int]]:
    """
    Simplified multi-resolution discretization (faster than full T1-T5).

    Uses 3 resolution regions:
    - Region 1: Fine grid near unitarity bound (step = 0.01)
    - Region 2: Medium grid (step = 0.1)
    - Region 3: Coarse grid for high Δ (step = 0.5)

    Args:
        delta_gap: The gap Δε'
        unitarity_bound_func: Function(spin) -> min_delta
        max_spin: Maximum spin to include
        n_regions: Number of resolution regions (1-3)
        verbose: Print info

    Returns:
        List of (delta, spin) pairs
    """
    operators = []
    seen = set()

    # Define regions
    regions = [
        {'step': 0.01, 'range': 3.0},   # Fine: unitarity to +3
        {'step': 0.1,  'range': 10.0},  # Medium: +3 to +13
        {'step': 0.5,  'range': 30.0},  # Coarse: +13 to +43
    ][:n_regions]

    for spin in range(0, max_spin + 1, 2):
        delta_min = unitarity_bound_func(spin)

        # For scalars, start from gap
        if spin == 0:
            delta_min = max(delta_min, delta_gap)

        current_delta = delta_min

        for region in regions:
            step = region['step']
            delta_end = current_delta + region['range']

            while current_delta <= delta_end:
                key = (round(current_delta, 4), spin)
                if key not in seen:
                    seen.add(key)
                    operators.append((current_delta, spin))
                current_delta += step

    operators.sort(key=lambda x: (x[1], x[0]))

    if verbose:
        n_scalars = sum(1 for _, s in operators if s == 0)
        n_spinning = len(operators) - n_scalars
        print(f"Simplified multi-resolution:")
        print(f"  Total operators: {len(operators)}")
        print(f"  Scalars (ℓ=0): {n_scalars}")
        print(f"  Spinning (ℓ>0): {n_spinning}")

    return operators


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

    def build_F_vector_spinning(self, delta: float, ell: int, n_max: int = 40) -> np.ndarray:
        """
        Build F-vector for spinning operator with dimension delta and spin ell.

        For ℓ=0, delegates to build_F_vector().
        For ℓ>0, uses the radial expansion from SpinningConformalBlock.

        Args:
            delta: Operator scaling dimension
            ell: Operator spin (0, 2, 4, ...)
            n_max: Maximum order in radial expansion

        Returns:
            F-vector of shape (n_coeffs,)
        """
        if ell == 0:
            return self.build_F_vector(delta)

        F_vec = np.zeros(self.n_coeffs)

        # For spinning operators, compute numerically using finite differences
        # We need F(a,b) = v^{Δσ} g_{Δ,ℓ}(z,zbar) - u^{Δσ} g_{Δ,ℓ}(1-z,1-zbar)
        block = SpinningConformalBlock(delta, ell, n_max=n_max)

        for i, (m, n) in enumerate(self.indices):
            deriv = self._numerical_derivative_spinning(delta, ell, m, n, block)
            F_vec[i] = deriv / (factorial(m) * factorial(n))

        return F_vec

    def _numerical_derivative_spinning(self, delta: float, ell: int, m: int, n: int,
                                        block: SpinningConformalBlock,
                                        h_a: float = 0.02, h_b: float = 0.02) -> float:
        """
        Compute ∂_a^m ∂_b^n F at (a=0, b=0) for a spinning operator.
        """
        a0, b0 = 0.0, 0.0

        def F_spinning(a_val: float, b_val: float) -> float:
            """F for spinning operator."""
            z, zbar = self._a_b_to_z_zbar(a_val, b_val)

            if z <= 0 or z >= 1 or zbar <= 0 or zbar >= 1:
                return 0.0

            u = z * zbar
            v = (1 - z) * (1 - zbar)

            # For diagonal b=0 case, use diagonal evaluation
            if abs(b_val) < 1e-10:
                g1 = block.evaluate_diagonal(z)
                g2 = block.evaluate_diagonal(1 - z)
            else:
                # Off-diagonal: need full (r, eta) evaluation
                r = np.sqrt(z * zbar)
                eta = (z + zbar) / (2 * r) if r > 1e-10 else 1.0
                eta = np.clip(eta, -1, 1)

                r_1m = np.sqrt((1-z) * (1-zbar))
                eta_1m = ((1-z) + (1-zbar)) / (2 * r_1m) if r_1m > 1e-10 else 1.0
                eta_1m = np.clip(eta_1m, -1, 1)

                # Convert to radial coordinate ρ
                # For small z: ρ ≈ z/4, for z near 1: ρ ≈ 1
                rho = r  # Simplified - radial coordinate
                rho_1m = r_1m

                g1 = block.evaluate(rho, eta)
                g2 = block.evaluate(rho_1m, eta_1m)

            return v**self.delta_sigma * g1 - u**self.delta_sigma * g2

        # For n-th b-derivative at fixed a
        def deriv_b_at_a(a_val):
            if n == 0:
                return F_spinning(a_val, b0)

            coeffs_b = self._get_diff_coeffs(n)
            half_b = len(coeffs_b) // 2
            result = 0.0
            for j, c in enumerate(coeffs_b):
                b_val = b0 + (j - half_b) * h_b
                result += c * F_spinning(a_val, b_val)
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


class ElShowkBootstrapSolver:
    """
    Bootstrap solver using the full El-Showk derivative basis.

    This solver implements the paper's protocol with:
    - nmax=10 (66 derivative coefficients)
    - Mixed (a,b) derivatives at (a=1, b=0)
    - Spinning operators up to Lmax (default 50)
    """

    # Solver options with tolerances matching serious LP solvers
    # Paper used IBM ILOG CPLEX (dual simplex)
    SOLVER_OPTIONS = {
        'scs': {
            'eps_abs': 1e-9,      # Absolute tolerance (default 1e-4)
            'eps_rel': 1e-9,      # Relative tolerance (default 1e-4)
            'max_iters': 50000,   # More iterations for convergence
            'verbose': False,
        },
        'ecos': {
            'abstol': 1e-9,
            'reltol': 1e-9,
            'feastol': 1e-9,
            'max_iters': 500,
            'verbose': False,
        },
        'clarabel': {
            'tol_gap_abs': 1e-9,
            'tol_gap_rel': 1e-9,
            'tol_feas': 1e-9,
            'max_iter': 500,
            'verbose': False,
        },
        'mosek': {
            'verbose': False,
        },
    }

    def __init__(self, d: int = 3, nmax: int = 10, max_spin: int = 50,
                 solver: str = 'auto'):
        """
        Initialize the solver.

        Args:
            d: Spacetime dimension (default 3 for Ising)
            nmax: Derivative order (default 10 for 66 coefficients)
            max_spin: Maximum spin to include (default 50, paper uses 100)
            solver: Solver to use ('auto', 'scs', 'ecos', 'clarabel', 'mosek')
        """
        self.d = d
        self.nmax = nmax
        self.max_spin = max_spin
        self.solver_name = solver
        self.n_constraints = count_coefficients(nmax)
        self.indices = get_derivative_indices(nmax)

        print(f"El-Showk solver initialized:")
        print(f"  nmax = {nmax}")
        print(f"  Number of constraints: {self.n_constraints}")
        print(f"  Max spin: {max_spin}")
        print(f"  Solver: {solver}")

    def unitarity_bound(self, ell: int) -> float:
        """
        Unitarity bound for spin ℓ operators in d dimensions.

        Δ ≥ ℓ + d - 2  for ℓ > 0
        Δ ≥ (d - 2)/2  for ℓ = 0

        In 3D: Δ ≥ ℓ + 1 for ℓ > 0, Δ ≥ 0.5 for ℓ = 0
        """
        if ell == 0:
            return (self.d - 2) / 2  # 0.5 for d=3
        return ell + self.d - 2  # ℓ + 1 for d=3

    def is_excluded(self, delta_sigma: float, delta_epsilon: float,
                    delta_epsilon_prime: float,
                    delta_max: float = 30.0, n_scalar_samples: int = 60,
                    n_spin_samples: int = 30, include_spinning: bool = True,
                    use_multiresolution: bool = False) -> bool:
        """
        Check if point (Δσ, Δε, Δε') is excluded.

        Uses the dual SDP formulation:
            Find α such that:
                α · F_id = 1 (normalization)
                α · F_ε ≥ 0 (first scalar OK)
                α · F_{Δ,ℓ} ≥ 0 for all operators above gap (positivity)

        If such α exists, the point is EXCLUDED.

        Args:
            delta_sigma: External scalar dimension
            delta_epsilon: First scalar dimension Δε
            delta_epsilon_prime: Gap for second scalar Δε'
            delta_max: Maximum dimension to sample
            n_scalar_samples: Number of scalar operators to sample above gap
            n_spin_samples: Number of spinning operators per spin
            include_spinning: Whether to include spinning operators (default True)
            use_multiresolution: Use T1-T5 style multi-resolution discretization
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

        # Collect all F-vectors for operators
        all_F_ops = []

        if use_multiresolution:
            # Use T1-T5 style multi-resolution discretization
            operators = get_simplified_multiresolution(
                delta_gap=delta_epsilon_prime,
                unitarity_bound_func=self.unitarity_bound,
                max_spin=self.max_spin if include_spinning else 0,
                n_regions=3,
                verbose=False
            )

            for delta, spin in operators:
                if spin == 0:
                    F = cross.build_F_vector(delta)
                else:
                    F = cross.build_F_vector_spinning(delta, spin)
                all_F_ops.append(F)
        else:
            # Original uniform sampling
            # 1. Scalar operators (ℓ=0) above the gap Δε'
            scalar_deltas = np.linspace(delta_epsilon_prime, delta_max, n_scalar_samples)
            for delta in scalar_deltas:
                F = cross.build_F_vector(delta)
                all_F_ops.append(F)

            # 2. Spinning operators (ℓ = 2, 4, 6, ..., max_spin)
            if include_spinning and self.max_spin >= 2:
                for ell in range(2, self.max_spin + 1, 2):
                    delta_min = self.unitarity_bound(ell)
                    spin_deltas = np.linspace(delta_min, delta_max, n_spin_samples)
                    for delta in spin_deltas:
                        F = cross.build_F_vector_spinning(delta, ell)
                        all_F_ops.append(F)

        if len(all_F_ops) == 0:
            warnings.warn("No operators to check")
            return False

        F_ops = np.array(all_F_ops)

        # Stack for reshuffling
        F_all = np.vstack([F_eps[np.newaxis, :], F_ops])

        # Apply normalization
        F_reduced, fixed_contribs, max_idx = reshuffle_with_normalization(F_all, F_id)

        # Separate
        F_eps_reduced = F_reduced[0, :]
        fixed_eps = fixed_contribs[0]
        F_ops_reduced = F_reduced[1:, :]
        fixed_ops = fixed_contribs[1:]

        # Solve SDP with high-precision tolerances
        alpha = cp.Variable(self.n_constraints - 1)

        constraints = [alpha @ F_eps_reduced >= -fixed_eps]
        for i, F_O in enumerate(F_ops_reduced):
            constraints.append(alpha @ F_O >= -fixed_ops[i])

        prob = cp.Problem(cp.Minimize(0), constraints)

        # Try solvers in order of preference
        solvers_to_try = self._get_solvers_to_try()

        for solver_name, solver_cp, options in solvers_to_try:
            try:
                prob.solve(solver=solver_cp, **options)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    return True  # EXCLUDED
                elif prob.status == cp.INFEASIBLE:
                    return False  # ALLOWED
                # Otherwise try next solver
            except Exception:
                continue

        # All solvers failed or returned unclear status
        return False

    def _get_solvers_to_try(self):
        """Get list of (name, cvxpy_solver, options) to try."""
        import cvxpy as cp

        if self.solver_name != 'auto':
            # Use specified solver only
            solver_map = {
                'scs': cp.SCS,
                'ecos': cp.ECOS,
                'clarabel': cp.CLARABEL,
                'mosek': cp.MOSEK,
            }
            if self.solver_name in solver_map:
                solver_cp = solver_map[self.solver_name]
                options = self.SOLVER_OPTIONS.get(self.solver_name, {})
                return [(self.solver_name, solver_cp, options)]
            else:
                warnings.warn(f"Unknown solver {self.solver_name}, using SCS")
                return [('scs', cp.SCS, self.SOLVER_OPTIONS['scs'])]

        # Auto mode: try solvers in order of preference
        solvers = []

        # Try CLARABEL first (modern, high-precision)
        try:
            import clarabel
            solvers.append(('clarabel', cp.CLARABEL, self.SOLVER_OPTIONS['clarabel']))
        except ImportError:
            pass

        # Try ECOS (good for LP/SOCP)
        try:
            import ecos
            solvers.append(('ecos', cp.ECOS, self.SOLVER_OPTIONS['ecos']))
        except ImportError:
            pass

        # Try MOSEK (commercial, very accurate)
        try:
            import mosek
            solvers.append(('mosek', cp.MOSEK, self.SOLVER_OPTIONS['mosek']))
        except ImportError:
            pass

        # SCS always available as fallback
        solvers.append(('scs', cp.SCS, self.SOLVER_OPTIONS['scs']))

        return solvers

    def find_delta_epsilon_prime_bound(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: float = None,
        delta_prime_max: float = 6.0,
        tolerance: float = 0.02,
        include_spinning: bool = True,
        use_multiresolution: bool = False,
        verbose: bool = False
    ) -> float:
        """
        Find upper bound on Δε' using binary search.

        Args:
            delta_sigma: External scalar dimension
            delta_epsilon: First scalar dimension
            delta_prime_min: Minimum gap to search (default: delta_epsilon + 0.1)
            delta_prime_max: Maximum gap to search
            tolerance: Search tolerance
            include_spinning: Whether to include spinning operators
            use_multiresolution: Use T1-T5 style multi-resolution discretization
            verbose: Print progress
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        if verbose:
            print(f"Searching for Δε' bound at Δσ={delta_sigma:.4f}, Δε={delta_epsilon:.4f}")
            print(f"  Range: [{delta_prime_min:.2f}, {delta_prime_max:.2f}]")
            print(f"  Including spinning operators: {include_spinning}")
            print(f"  Multi-resolution discretization: {use_multiresolution}")

        if self.is_excluded(delta_sigma, delta_epsilon, delta_prime_min,
                           include_spinning=include_spinning, use_multiresolution=use_multiresolution):
            return delta_prime_min

        if not self.is_excluded(delta_sigma, delta_epsilon, delta_prime_max,
                               include_spinning=include_spinning, use_multiresolution=use_multiresolution):
            return float('inf')

        lo, hi = delta_prime_min, delta_prime_max
        iterations = 0
        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            if self.is_excluded(delta_sigma, delta_epsilon, mid,
                               include_spinning=include_spinning, use_multiresolution=use_multiresolution):
                hi = mid
            else:
                lo = mid
            iterations += 1
            if verbose and iterations % 5 == 0:
                print(f"  Iteration {iterations}: [{lo:.3f}, {hi:.3f}]")

        bound = (lo + hi) / 2
        if verbose:
            print(f"  Final bound: Δε' ≤ {bound:.4f}")
        return bound


# Convenience functions
def test_el_showk_basis():
    """Test the El-Showk derivative basis."""
    print("Testing El-Showk derivative basis")
    print("=" * 60)

    # Test coefficient counting
    print("\n1. Coefficient counting:")
    for nmax in [5, 10, 11]:
        n_coeffs = count_coefficients(nmax)
        expected = (nmax + 1) * (nmax + 2) // 2
        print(f"  nmax={nmax}: {n_coeffs} coefficients (expected: {expected})")

    # Test derivative indices
    print("\n2. Derivative indices for nmax=3:")
    indices = get_derivative_indices(3)
    for m, n in indices:
        print(f"  (m={m}, n={n}), order = {m + 2*n}")

    # Test F-vector computation
    print("\n3. F-vector computation at Ising point (scalars):")
    cross = ElShowkCrossingVector(delta_sigma=0.518, nmax=3)

    F_id = cross.build_F_vector(0)
    F_eps = cross.build_F_vector(1.41)

    print(f"  F_identity: {F_id[:5]}...")
    print(f"  F_epsilon:  {F_eps[:5]}...")

    # Test spinning F-vectors
    print("\n4. F-vector computation for spinning operators:")
    F_spin2 = cross.build_F_vector_spinning(3.0, ell=2)  # Stress tensor
    F_spin4 = cross.build_F_vector_spinning(5.0, ell=4)
    print(f"  F(Δ=3, ℓ=2): {F_spin2[:5]}...")
    print(f"  F(Δ=5, ℓ=4): {F_spin4[:5]}...")

    # Test solver with scalars only
    print("\n5. Solver test (scalars only, nmax=3, max_spin=0):")
    solver_scalar = ElShowkBootstrapSolver(d=3, nmax=3, max_spin=0)
    excluded_scalar = solver_scalar.is_excluded(0.518, 1.41, 3.0, include_spinning=False)
    print(f"  Δε'=3.0 excluded (scalars): {excluded_scalar}")

    # Test solver with spinning operators
    print("\n6. Solver test (with spinning, nmax=3, max_spin=6):")
    solver_spin = ElShowkBootstrapSolver(d=3, nmax=3, max_spin=6)
    excluded_spin = solver_spin.is_excluded(0.518, 1.41, 3.0,
                                            n_scalar_samples=30, n_spin_samples=10,
                                            include_spinning=True)
    print(f"  Δε'=3.0 excluded (with spin): {excluded_spin}")

    print("\n" + "=" * 60)
    print("Test complete!")


def test_spinning_operators():
    """Test spinning operators in more detail."""
    print("\nTesting spinning operators in El-Showk basis")
    print("=" * 60)

    delta_sigma = 0.518
    delta_epsilon = 1.41

    print(f"\nIsing point: Δσ = {delta_sigma}, Δε = {delta_epsilon}")
    print("Reference: Δε' ≤ 3.8 (El-Showk 2012)")

    # Test with different max_spin values
    for max_spin in [0, 2, 6, 10]:
        print(f"\n--- max_spin = {max_spin} ---")
        solver = ElShowkBootstrapSolver(d=3, nmax=5, max_spin=max_spin)

        # Test a few gap values
        for gap in [2.5, 3.0, 3.5]:
            if max_spin > 0:
                excluded = solver.is_excluded(delta_sigma, delta_epsilon, gap,
                                            n_scalar_samples=40, n_spin_samples=15,
                                            include_spinning=True)
            else:
                excluded = solver.is_excluded(delta_sigma, delta_epsilon, gap,
                                            include_spinning=False)
            status = "EXCLUDED" if excluded else "ALLOWED"
            print(f"  Δε' = {gap}: {status}")

    print("\n" + "=" * 60)


def test_multiresolution():
    """Test multi-resolution discretization."""
    print("\nTesting multi-resolution discretization (T1-T5 style)")
    print("=" * 60)

    # Define unitarity bound function for 3D
    def unitarity_bound(ell):
        if ell == 0:
            return 0.5
        return ell + 1

    # Test the simplified multi-resolution
    print("\n1. Simplified multi-resolution (faster):")
    operators = get_simplified_multiresolution(
        delta_gap=1.51,  # Gap above Δε
        unitarity_bound_func=unitarity_bound,
        max_spin=10,
        n_regions=3,
        verbose=True
    )

    print(f"\n  Sample operators:")
    print(f"  First 5 scalars: {[(d, s) for d, s in operators if s == 0][:5]}")
    print(f"  First 5 spin-2:  {[(d, s) for d, s in operators if s == 2][:5]}")
    print(f"  First 5 spin-4:  {[(d, s) for d, s in operators if s == 4][:5]}")

    # Test full T1-T5 discretization
    print("\n2. Full T1-T5 discretization (paper style):")
    operators_full = get_multiresolution_operators(
        delta_gap=1.51,
        unitarity_bound_func=unitarity_bound,
        verbose=True
    )

    # Test solver with multi-resolution
    print("\n3. Solver test with multi-resolution:")
    solver = ElShowkBootstrapSolver(d=3, nmax=3, max_spin=6)

    delta_sigma = 0.518
    delta_epsilon = 1.41

    for gap in [2.5, 3.0]:
        excluded_uniform = solver.is_excluded(
            delta_sigma, delta_epsilon, gap,
            n_scalar_samples=30, n_spin_samples=10,
            include_spinning=True, use_multiresolution=False
        )
        excluded_multi = solver.is_excluded(
            delta_sigma, delta_epsilon, gap,
            include_spinning=True, use_multiresolution=True
        )
        print(f"  Δε' = {gap}: uniform={'EXCL' if excluded_uniform else 'ALLOW'}, "
              f"multiresolution={'EXCL' if excluded_multi else 'ALLOW'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--spinning":
        test_spinning_operators()
    elif len(sys.argv) > 1 and sys.argv[1] == "--multiresolution":
        test_multiresolution()
    else:
        test_el_showk_basis()
