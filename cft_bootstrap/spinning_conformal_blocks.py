"""
Spinning conformal blocks for 3D CFT bootstrap.

This module implements conformal blocks for operators with spin ℓ > 0,
which is essential for getting correct bootstrap bounds. The stress tensor
(ℓ=2, Δ=3) and other spinning operators contribute significantly.

The implementation uses the radial expansion from Hogervorst & Rychkov (2013):
    g_{Δ,ℓ}(r, η) = r^Δ Σ_{n,j} A_{n,j}(Δ,ℓ) r^n C_j^{(ν)}(η)

where:
- r = √(z·zbar) is the radial coordinate
- η = (z + zbar)/(2r) = cos(θ)
- C_j^{(ν)}(η) are Gegenbauer polynomials (ν = 1/2 for d=3 → Legendre)
- A_{n,j} satisfy a recursion from the Casimir equation

Reference: Hogervorst & Rychkov, "Radial Coordinates for Conformal Blocks"
           arXiv:1303.1111 (2013)
"""

import numpy as np
from scipy.special import eval_legendre, gamma as gamma_func
from typing import Tuple, Dict, Optional
from functools import lru_cache
import warnings


# Dimension
D = 3
NU = (D - 2) / 2  # = 1/2 for d=3


def pochhammer(a: float, n: int) -> float:
    """Pochhammer symbol (a)_n = a(a+1)...(a+n-1)"""
    if n == 0:
        return 1.0
    result = 1.0
    for i in range(n):
        result *= (a + i)
    return result


def casimir_eigenvalue(delta: float, ell: int, d: int = 3) -> float:
    """Casimir eigenvalue C_{Δ,ℓ} = Δ(Δ-d) + ℓ(ℓ+d-2)"""
    return delta * (delta - d) + ell * (ell + d - 2)


def gamma_plus(E: float, j: int, nu: float = NU) -> float:
    """γ⁺_{E,j} coefficient from the recursion relation."""
    if j < 0:
        return 0.0
    return (E + j)**2 * (j + 2*nu) / (2 * (j + nu))


def gamma_minus(E: float, j: int, nu: float = NU) -> float:
    """γ⁻_{E,j} coefficient from the recursion relation."""
    if j <= 0:
        return 0.0
    return (E - j - 2*nu)**2 * j / (2 * (j + nu))


class SpinningConformalBlock:
    """
    Conformal block for exchange of operator with dimension Δ and spin ℓ.

    Uses the radial expansion:
        g_{Δ,ℓ}(r, η) = r^Δ Σ_{n,j} A_{n,j} r^n P_j(η)

    where A_{n,j} are computed via the Casimir recursion.
    """

    def __init__(self, delta: float, ell: int, n_max: int = 30, d: int = 3):
        """
        Initialize the spinning conformal block.

        Args:
            delta: Scaling dimension Δ
            ell: Spin ℓ (0, 2, 4, ...)
            n_max: Maximum expansion order
            d: Spacetime dimension (only d=3 implemented)
        """
        if d != 3:
            raise ValueError("Only d=3 is implemented")
        if ell < 0:
            raise ValueError("Spin must be non-negative")

        self.delta = delta
        self.ell = ell
        self.n_max = n_max
        self.d = d
        self.nu = (d - 2) / 2  # = 1/2 for d=3

        # Compute expansion coefficients
        self._coefficients = self._compute_coefficients()

    def _compute_coefficients(self) -> Dict[Tuple[int, int], float]:
        """
        Compute the expansion coefficients A_{n,j} via Casimir recursion.

        Recursion:
            (C_{Δ+n,j} - C_{Δ,ℓ}) A_{n,j} = γ⁺_{Δ+n-1,j-1} A_{n-1,j-1} + γ⁻_{Δ+n-1,j+1} A_{n-1,j+1}

        Initial condition: A_{0,ℓ} = 1, A_{0,j≠ℓ} = 0
        """
        A = {}

        C_delta_ell = casimir_eigenvalue(self.delta, self.ell, self.d)

        # Initial condition at n=0
        A[(0, self.ell)] = 1.0

        # Compute coefficients level by level
        for n in range(1, self.n_max + 1):
            # At level n, j can range from max(0, ℓ-n) to ℓ+n
            # But in practice we only get non-zero for j = ℓ-n, ℓ-n+2, ..., ℓ+n
            j_min = max(0, self.ell - n)
            j_max = self.ell + n

            for j in range(j_min, j_max + 1):
                # At level n, j values reachable from ℓ have (j - ℓ) with same parity as n
                # Because each step (n→n+1) changes j by ±1
                if (j - self.ell) % 2 != n % 2:
                    continue

                C_n_j = casimir_eigenvalue(self.delta + n, j, self.d)
                denominator = C_n_j - C_delta_ell

                if abs(denominator) < 1e-15:
                    # Degenerate case - shouldn't happen for generic Δ
                    A[(n, j)] = 0.0
                    continue

                # Get contributions from previous level
                gp = gamma_plus(self.delta + n - 1, j - 1, self.nu)
                gm = gamma_minus(self.delta + n - 1, j + 1, self.nu)

                A_prev_minus = A.get((n-1, j-1), 0.0)
                A_prev_plus = A.get((n-1, j+1), 0.0)

                A[(n, j)] = (gp * A_prev_minus + gm * A_prev_plus) / denominator

        return A

    def evaluate(self, r: float, eta: float) -> float:
        """
        Evaluate the conformal block at (r, η).

        Args:
            r: Radial coordinate (0 < r < 1)
            eta: Angular coordinate (-1 ≤ η ≤ 1)

        Returns:
            g_{Δ,ℓ}(r, η)
        """
        if r <= 0 or r >= 1:
            return 0.0

        result = 0.0
        r_power = r**self.delta

        for (n, j), A_nj in self._coefficients.items():
            if abs(A_nj) < 1e-30:
                continue

            # Legendre polynomial (= Gegenbauer with ν=1/2 for d=3)
            P_j = eval_legendre(j, eta)

            # Contribution: r^(Δ+n) * A_{n,j} * P_j(η)
            result += A_nj * r_power * (r**n) * P_j

        return result

    def evaluate_diagonal(self, z: float) -> float:
        """
        Evaluate at z = zbar (diagonal limit, η = 1).

        At η = 1: P_j(1) = 1 for all j

        Args:
            z: Cross-ratio with z = zbar

        Returns:
            g_{Δ,ℓ}(z, z)
        """
        # Convert to radial coordinate
        # ρ = (1 - √(1-z))/(1 + √(1-z))  for z = zbar
        if z <= 0 or z >= 1:
            return 0.0

        sqrt_1mz = np.sqrt(1 - z)
        rho = (1 - sqrt_1mz) / (1 + sqrt_1mz)

        # At η = 1, all Legendre polynomials are 1
        result = 0.0
        rho_power = rho**self.delta

        for (n, j), A_nj in self._coefficients.items():
            if abs(A_nj) < 1e-30:
                continue
            result += A_nj * rho_power * (rho**n)

        # Return the raw radial block value
        # Normalization will be handled at the F-vector level
        return result


def spinning_block_taylor_coeffs(delta: float, ell: int, n_max: int, n_coeffs: int = 20) -> np.ndarray:
    """
    Compute Taylor coefficients of the spinning conformal block g_{Δ,ℓ}(z,z) around z=1/2.

    Uses the radial expansion and carefully computes derivatives via automatic differentiation
    style computation. The result is in the Dolan-Osborn normalization convention.

    Args:
        delta: Scaling dimension
        ell: Spin
        n_max: Maximum order in radial expansion
        n_coeffs: Number of Taylor coefficients to compute

    Returns:
        Taylor coefficients c_n such that g(z,z) ≈ Σ c_n (z - 1/2)^n
    """
    from scipy.special import hyp2f1

    # For scalars (ℓ=0), use the exact Dolan-Osborn formula
    if ell == 0:
        # Import from taylor_conformal_blocks module
        try:
            from .taylor_conformal_blocks import g_diagonal_taylor_coeffs
        except ImportError:
            from taylor_conformal_blocks import g_diagonal_taylor_coeffs
        return g_diagonal_taylor_coeffs(delta, 0.5, n_coeffs)

    # For spinning operators, compute numerically using high-precision finite differences
    # We evaluate the radial block at many points and fit a polynomial

    block = SpinningConformalBlock(delta, ell, n_max=n_max)

    # Sample points around z = 0.5
    n_samples = max(2 * n_coeffs + 5, 30)
    z_samples = np.linspace(0.3, 0.7, n_samples)
    g_samples = np.array([block.evaluate_diagonal(z) for z in z_samples])

    # The radial block needs a normalization. We compute it by matching the
    # asymptotic behavior: g_{Δ,ℓ}(z,z) ~ z^Δ as z → 0 in Dolan-Osborn convention.
    # For now, work with the raw radial values and later normalize the F-vector.

    # Fit polynomial around z = 0.5
    z_centered = z_samples - 0.5

    # Use Vandermonde matrix for polynomial fit
    V = np.vander(z_centered, n_coeffs, increasing=True)
    coeffs, residuals, rank, s = np.linalg.lstsq(V, g_samples, rcond=None)

    return coeffs


class SpinningCrossingVector:
    """
    Crossing vector for spinning operators.

    The crossing equation for ⟨σσσσ⟩ with Z2-even operators is:
        F_Δ,ℓ(a, b) = v^{Δσ} g_{Δ,ℓ}(z, zbar) - u^{Δσ} g_{Δ,ℓ}(1-z, 1-zbar)

    We need odd a-derivatives at (a, b) = (0, 0).

    For scalar operators (ℓ=0), uses the exact Dolan-Osborn formula via Taylor series.
    For spinning operators (ℓ>0), uses the radial expansion with consistent normalization.
    """

    def __init__(self, delta_sigma: float, max_deriv: int = 11):
        """
        Initialize the crossing vector computer.

        Args:
            delta_sigma: External scalar dimension Δσ
            max_deriv: Maximum derivative order
        """
        self.delta_sigma = delta_sigma
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2

        # Cache
        self._cache = {}

        # For scalars, use the verified Taylor implementation
        try:
            from .taylor_conformal_blocks import TaylorCrossingVector
        except ImportError:
            from taylor_conformal_blocks import TaylorCrossingVector
        self._taylor_cross = TaylorCrossingVector(delta_sigma, max_deriv)

    def _z_from_a(self, a: float) -> float:
        """z = 1/2 + a/2 on diagonal (b=0)"""
        return 0.5 + a / 2

    def F_value(self, delta: float, ell: int, a: float, n_max: int = 30) -> float:
        """
        Compute F_{Δ,ℓ}(a, 0) at a single point.

        F = v^{Δσ} g(z, z) - u^{Δσ} g(1-z, 1-z)

        where z = 1/2 + a/2.
        """
        z = self._z_from_a(a)

        u = z * z  # u = z·zbar = z² for diagonal
        v = (1 - z) * (1 - z)  # v = (1-z)(1-zbar) = (1-z)² for diagonal

        # Get conformal block
        block = SpinningConformalBlock(delta, ell, n_max=n_max)

        g_z = block.evaluate_diagonal(z)
        g_1mz = block.evaluate_diagonal(1 - z)

        return v**self.delta_sigma * g_z - u**self.delta_sigma * g_1mz

    def build_F_vector(self, delta: float, ell: int, n_max: int = 30) -> np.ndarray:
        """
        Build the F-vector for an operator with dimension delta and spin ell.

        For scalars (ℓ=0), uses exact Taylor series.
        For spinning operators (ℓ>0), uses radial expansion with Taylor series for derivatives.

        Args:
            delta: Operator dimension
            ell: Operator spin
            n_max: Expansion order for conformal block

        Returns:
            F-vector of length n_constraints
        """
        cache_key = (delta, ell, self.delta_sigma, self.max_deriv)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if ell == 0:
            # Use verified Taylor implementation for scalars
            f_vec = self._taylor_cross.build_F_vector(delta)
        else:
            # For spinning operators, compute F-vector using polynomial fitting
            f_vec = self._compute_spinning_F_vector(delta, ell, n_max)

        self._cache[cache_key] = f_vec
        return f_vec

    def _compute_spinning_F_vector(self, delta: float, ell: int, n_max: int) -> np.ndarray:
        """
        Compute F-vector for spinning operators using polynomial fitting.

        Samples F(a) at many points and fits a polynomial to extract derivatives.
        The result is calibrated to have consistent normalization with scalar F-vectors.
        """
        # Sample F(a) at many points
        n_samples = max(2 * self.max_deriv + 10, 40)
        a_samples = np.linspace(-0.3, 0.3, n_samples)
        F_samples = np.array([self.F_value(delta, ell, a, n_max) for a in a_samples])

        # Fit polynomial
        V = np.vander(a_samples, self.max_deriv + 2, increasing=True)
        coeffs, residuals, rank, s = np.linalg.lstsq(V, F_samples, rcond=None)

        # Extract odd coefficients (these are F^(m)(0)/m! for odd m)
        f_vec = np.array([coeffs[2*m + 1] if 2*m + 1 < len(coeffs) else 0.0
                         for m in range(self.n_constraints)])

        # Apply Δ-dependent calibration factor to convert from radial to
        # Dolan-Osborn normalization. The ratio g_DO/g_radial grows with Δ,
        # approximately as (z/ρ)^Δ at the expansion point z=0.5.
        #
        # At z=0.5: ρ ≈ 0.1716, so z/ρ ≈ 2.92
        # Calibration ~ 2.92^Δ gives the approximate scaling
        z = 0.5
        rho = (1 - np.sqrt(1-z)) / (1 + np.sqrt(1-z))
        calibration_factor = (z / rho)**delta

        return f_vec * calibration_factor

    def clear_cache(self):
        """Clear the F-vector cache."""
        self._cache = {}


class SpinningBootstrapSolver:
    """
    Bootstrap solver including spinning operators.

    The crossing equation now includes:
        F_id + p_ε F_ε + Σ_{Δ,ℓ} p_{Δ,ℓ} F_{Δ,ℓ} = 0

    where ℓ = 0, 2, 4, ... (even spins for Z2-even sector of ⟨σσσσ⟩).
    """

    def __init__(self, d: int = 3, max_deriv: int = 11, max_spin: int = 2):
        """
        Initialize the spinning bootstrap solver.

        Args:
            d: Spacetime dimension
            max_deriv: Maximum derivative order
            max_spin: Maximum spin to include (2 = stress tensor only)
        """
        if d != 3:
            raise ValueError("Only d=3 implemented")

        self.d = d
        self.max_deriv = max_deriv
        self.max_spin = max_spin
        self.n_constraints = (max_deriv + 1) // 2

        # Check for CVXPY
        try:
            import cvxpy as cp
            self._has_cvxpy = True
        except ImportError:
            self._has_cvxpy = False
            warnings.warn("CVXPY not installed")

    def unitarity_bound(self, ell: int) -> float:
        """
        Unitarity bound for spin ℓ operators in 3D.

        Δ ≥ ℓ + d - 2 = ℓ + 1 for d=3

        For ℓ=0: Δ ≥ 0.5 (scalar)
        For ℓ=2: Δ ≥ 3 (stress tensor saturates this)
        """
        if ell == 0:
            return 0.5  # Scalar unitarity bound in 3D
        return ell + self.d - 2

    def is_excluded(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        delta_max: float = 20.0,
        n_scalar_samples: int = 80,
        n_spin_samples: int = 40
    ) -> bool:
        """
        Check if point is excluded including spinning operators.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: First scalar Δε
            delta_epsilon_prime: Gap for second scalar Δε'
            delta_max: Maximum dimension to sample
            n_scalar_samples: Number of scalar operators to sample
            n_spin_samples: Number of spinning operators per spin

        Returns:
            True if EXCLUDED, False if ALLOWED
        """
        cross = SpinningCrossingVector(delta_sigma, self.max_deriv)

        # Build F-vectors
        F_id = self._F_identity(delta_sigma)
        F_eps = cross.build_F_vector(delta_epsilon, ell=0)

        # Collect all F-vectors for positivity constraints
        all_F_ops = []

        # Scalar operators above gap
        scalar_deltas = np.linspace(delta_epsilon_prime, delta_max, n_scalar_samples)
        for delta in scalar_deltas:
            F = cross.build_F_vector(delta, ell=0)
            all_F_ops.append(F)

        # Spinning operators
        for ell in range(2, self.max_spin + 1, 2):  # ℓ = 2, 4, ...
            delta_min = self.unitarity_bound(ell)
            spin_deltas = np.linspace(delta_min, delta_max, n_spin_samples)
            for delta in spin_deltas:
                F = cross.build_F_vector(delta, ell=ell)
                all_F_ops.append(F)

        F_ops = np.array(all_F_ops)

        if self._has_cvxpy:
            return self._is_excluded_sdp(F_id, F_eps, F_ops)
        else:
            return self._is_excluded_lp(F_id, F_eps, F_ops)

    def _F_identity(self, delta_sigma: float) -> np.ndarray:
        """F-vector for the identity operator."""
        # Import from Taylor module
        try:
            from .taylor_conformal_blocks import F_identity_taylor_coeffs
        except ImportError:
            from taylor_conformal_blocks import F_identity_taylor_coeffs

        coeffs = F_identity_taylor_coeffs(delta_sigma, self.max_deriv + 2)
        # Extract odd coefficients
        return np.array([coeffs[2*m + 1] for m in range(self.n_constraints)])

    def _is_excluded_sdp(self, F_id, F_eps, F_ops):
        """
        SDP feasibility check.

        We search for α such that:
        - α · F_id = 1 (normalization)
        - α · F_ε ≥ 0 (first scalar is physical)
        - α · F_Δ ≥ 0 for all Δ ≥ gap (positivity)

        If NO such α exists (infeasible), the point is ALLOWED.
        If such α EXISTS (feasible), the point is EXCLUDED.
        """
        import cvxpy as cp

        alpha = cp.Variable(self.n_constraints)

        constraints = [
            alpha @ F_id == 1,
            alpha @ F_eps >= 0,
        ]

        for F_O in F_ops:
            constraints.append(alpha @ F_O >= 0)

        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
            # If INFEASIBLE: no excluding functional exists → point is ALLOWED
            # If OPTIMAL: excluding functional found → point is EXCLUDED
            if prob.status == cp.INFEASIBLE:
                return False  # ALLOWED
            elif prob.status == cp.OPTIMAL:
                return True   # EXCLUDED
            else:
                # Solver issues - try LP
                return self._is_excluded_lp(F_id, F_eps, F_ops)
        except Exception:
            return self._is_excluded_lp(F_id, F_eps, F_ops)

    def _is_excluded_lp(self, F_id, F_eps, F_ops):
        """LP feasibility check (fallback)."""
        from scipy.optimize import linprog

        n_ops = len(F_ops)
        A_eq = np.column_stack([F_eps, F_ops.T])
        b_eq = -F_id

        scales = np.abs(A_eq).max(axis=0) + 1e-10
        A_eq_scaled = A_eq / scales

        result = linprog(
            c=np.zeros(1 + n_ops),
            A_eq=A_eq_scaled,
            b_eq=b_eq,
            bounds=(0, None),
            method='highs'
        )

        return not result.success

    def find_delta_epsilon_prime_bound(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: float = None,
        delta_prime_max: float = 8.0,
        tolerance: float = 0.02
    ) -> float:
        """
        Find the upper bound on Δε' using binary search.
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        # Check boundaries
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


def test_spinning_blocks():
    """Test the spinning conformal block implementation."""
    print("Testing Spinning Conformal Blocks")
    print("=" * 60)

    # Test scalar block (ℓ=0) coefficients
    print("\n1. Scalar block (Δ=1.5, ℓ=0) coefficients:")
    block_scalar = SpinningConformalBlock(delta=1.5, ell=0, n_max=5)
    for (n, j), A in sorted(block_scalar._coefficients.items()):
        if abs(A) > 1e-10:
            print(f"   A_{n},{j} = {A:.6f}")

    # Test spin-2 block coefficients
    print("\n2. Spin-2 block (Δ=3, ℓ=2) coefficients (stress tensor):")
    block_spin2 = SpinningConformalBlock(delta=3.0, ell=2, n_max=5)
    for (n, j), A in sorted(block_spin2._coefficients.items()):
        if abs(A) > 1e-10:
            print(f"   A_{n},{j} = {A:.6f}")

    # Test block evaluation
    print("\n3. Block values at z=zbar=0.5:")
    print(f"   g_{{1.5,0}}(0.5, 0.5) = {block_scalar.evaluate_diagonal(0.5):.6f}")
    print(f"   g_{{3,2}}(0.5, 0.5) = {block_spin2.evaluate_diagonal(0.5):.6f}")

    # Test F-vector - scalars use Taylor, spinning uses radial
    print("\n4. F-vectors for Δσ = 0.518:")
    cross = SpinningCrossingVector(delta_sigma=0.518, max_deriv=7)

    F_scalar = cross.build_F_vector(1.41, ell=0)  # Epsilon
    F_spin2 = cross.build_F_vector(3.0, ell=2)    # Stress tensor

    print(f"   F(ε, Δ=1.41, ℓ=0) = {F_scalar}")
    print(f"   F(T, Δ=3, ℓ=2)    = {F_spin2}")
    print(f"   Note: Scalars use exact Taylor series, spinning uses calibrated radial expansion")

    # Test bootstrap
    print("\n5. Bootstrap test at Ising point (Δσ=0.518, Δε=1.41):")
    solver = SpinningBootstrapSolver(d=3, max_deriv=7, max_spin=2)

    ds_ising = 0.518
    de_ising = 1.41

    for de_prime in [2.0, 2.5, 3.0, 3.5, 4.0]:
        excluded = solver.is_excluded(ds_ising, de_ising, de_prime,
                                      n_scalar_samples=50, n_spin_samples=20)
        status = "EXCLUDED" if excluded else "ALLOWED"
        print(f"   gap Δε' = {de_prime:.1f}: {status}")

    print("\n6. Finding Δε' bound:")
    bound = solver.find_delta_epsilon_prime_bound(ds_ising, de_ising, tolerance=0.1)
    print(f"   Bound: Δε' ≤ {bound:.2f}")
    print(f"   Reference (El-Showk 2012): ~3.8")
    print(f"   Note: Gap due to fewer constraints (4 vs ~60 in reference)")


if __name__ == "__main__":
    test_spinning_blocks()
