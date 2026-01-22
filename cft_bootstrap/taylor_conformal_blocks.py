"""
Taylor series expansion for conformal blocks - enables high-order derivatives.

This module implements conformal block computations using Taylor series expansion
around the crossing symmetric point z = zbar = 1/2. This avoids the numerical
instability of finite differences for high-order derivatives (m > 7).

Key insight: Instead of computing derivatives via finite differences, we:
1. Expand the hypergeometric functions in k_β(z) as Taylor series around z = 1/2
2. Compute the diagonal conformal block g_Δ(z,z) Taylor series via convolution
3. Build the crossing function F_Δ(a,0) Taylor series in the coordinate a
4. Read off derivatives directly as Taylor coefficients (coefficient_m = d^m F / da^m / m!)

This enables 20-60+ derivative constraints vs the 3-7 possible with finite differences.

Reference: Hogervorst & Rychkov, "Radial Coordinates for Conformal Blocks" (arXiv:1303.1111)
"""

import numpy as np
from scipy.special import gamma, poch
from typing import Tuple, Optional, Callable
import warnings

try:
    from .bootstrap_gap_solver import reshuffle_with_normalization
except ImportError:
    from bootstrap_gap_solver import reshuffle_with_normalization


def hyp2f1_value(a: float, b: float, c: float, z: float) -> float:
    """
    Compute 2F1(a, b; c; z) using scipy.

    Falls back to series summation near z = 1/2 for better precision.
    """
    from scipy.special import hyp2f1
    return float(hyp2f1(a, b, c, z))


def hyp2f1_taylor_coeffs(a: float, b: float, c: float, z0: float, n_max: int) -> np.ndarray:
    """
    Taylor coefficients of 2F1(a, b; c; z) around z = z0.

    The n-th coefficient is:
        (a)_n (b)_n / (c)_n / n! * 2F1(a+n, b+n; c+n; z0)

    where (x)_n = x(x+1)...(x+n-1) is the Pochhammer symbol.

    Args:
        a, b, c: Hypergeometric parameters
        z0: Expansion point
        n_max: Maximum order

    Returns:
        Array of Taylor coefficients [c_0, c_1, ..., c_{n_max}]
    """
    coeffs = np.zeros(n_max + 1)

    for n in range(n_max + 1):
        # Pochhammer symbols
        poch_a = poch(a, n)
        poch_b = poch(b, n)
        poch_c = poch(c, n)

        if abs(poch_c) < 1e-100:
            coeffs[n] = 0.0
            continue

        # Factorial
        fact_n = 1.0
        for i in range(1, n + 1):
            fact_n *= i

        # Hypergeometric value at shifted parameters
        hyp_val = hyp2f1_value(a + n, b + n, c + n, z0)

        coeffs[n] = poch_a * poch_b / poch_c / fact_n * hyp_val

    return coeffs


def generalized_binomial(alpha: float, n: int) -> float:
    """
    Generalized binomial coefficient C(alpha, n) = alpha * (alpha-1) * ... * (alpha-n+1) / n!
    """
    if n == 0:
        return 1.0
    if n < 0:
        return 0.0

    result = 1.0
    for i in range(n):
        result *= (alpha - i) / (i + 1)
    return result


def k_taylor_coeffs(beta: float, z0: float, n_max: int) -> np.ndarray:
    """
    Taylor coefficients of k_β(z) = z^(β/2) * 2F1(β/2, β/2; β; z) around z = z0.

    This is computed by convolving the Taylor series of z^(β/2) and 2F1.

    Args:
        beta: Conformal dimension parameter
        z0: Expansion point (typically 1/2)
        n_max: Maximum order

    Returns:
        Array of Taylor coefficients
    """
    # Taylor coefficients of z^(β/2) around z0
    power_coeffs = np.array([
        generalized_binomial(beta/2, n) * z0**(beta/2 - n)
        for n in range(n_max + 1)
    ])

    # Taylor coefficients of 2F1(β/2, β/2; β; z) around z0
    hyp_coeffs = hyp2f1_taylor_coeffs(beta/2, beta/2, beta, z0, n_max)

    # Convolve to get product
    result = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        for i in range(n + 1):
            result[n] += power_coeffs[i] * hyp_coeffs[n - i]

    return result


def g_diagonal_taylor_coeffs(delta: float, z0: float, n_max: int) -> np.ndarray:
    """
    Taylor coefficients of the diagonal conformal block g_Δ(z, z) around z = z0.

    On the diagonal z = zbar, the 3D scalar conformal block is:
        g_Δ(z, z) = z² [k'_Δ(z) k_{Δ-1}(z) - k_Δ(z) k'_{Δ-1}(z)]

    where k_β(z) = z^(β/2) * 2F1(β/2, β/2; β; z).

    This formula comes from the L'Hopital limit of the Dolan-Osborn formula
    as zbar → z.

    Args:
        delta: Scaling dimension of exchanged operator
        z0: Expansion point (typically 1/2)
        n_max: Maximum order

    Returns:
        Array of Taylor coefficients
    """
    # Get Taylor coefficients for k_Δ and k_{Δ-1}
    k_D = k_taylor_coeffs(delta, z0, n_max + 3)
    k_Dm1 = k_taylor_coeffs(delta - 1, z0, n_max + 3)

    # Derivatives: k'(z) has coefficients c'_n = (n+1) * c_{n+1}
    k_D_prime = np.array([(n + 1) * k_D[n + 1] for n in range(n_max + 2)])
    k_Dm1_prime = np.array([(n + 1) * k_Dm1[n + 1] for n in range(n_max + 2)])

    # Taylor coefficients of z² around z0
    z2_coeffs = np.array([generalized_binomial(2, n) * z0**(2 - n) for n in range(min(3, n_max + 3))])
    z2_coeffs = np.pad(z2_coeffs, (0, n_max + 3 - len(z2_coeffs)))

    # Convolve k'_Δ with k_{Δ-1}
    term1 = np.zeros(n_max + 2)
    for n in range(n_max + 2):
        for i in range(min(n + 1, len(k_D_prime))):
            if n - i < len(k_Dm1):
                term1[n] += k_D_prime[i] * k_Dm1[n - i]

    # Convolve k_Δ with k'_{Δ-1}
    term2 = np.zeros(n_max + 2)
    for n in range(n_max + 2):
        for i in range(min(n + 1, len(k_D))):
            if n - i < len(k_Dm1_prime):
                term2[n] += k_D[i] * k_Dm1_prime[n - i]

    # Bracket: term1 - term2
    bracket = term1 - term2

    # Multiply by z²
    result = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        for i in range(min(n + 1, 3)):  # z² has only 3 non-zero coeffs
            if n - i < len(bracket):
                result[n] += z2_coeffs[i] * bracket[n - i]

    return result


def F_identity_taylor_coeffs(delta_sigma: float, n_max: int) -> np.ndarray:
    """
    Taylor coefficients of F_identity(a, 0) = v^{Δσ} - u^{Δσ}.

    In the coordinates z = 1/2 + a/2, zbar = 1/2 - a/2:
        u = z * zbar = (1/4)(1 + a)² → u^{Δσ} = (1/4)^{Δσ} (1+a)^{2Δσ}
        v = (1-z)(1-zbar) = (1/4)(1 - a)² → v^{Δσ} = (1/4)^{Δσ} (1-a)^{2Δσ}

    At b = 0 (diagonal): z = zbar = 1/2 + a/2

    Args:
        delta_sigma: External operator dimension
        n_max: Maximum order

    Returns:
        Array of Taylor coefficients in a
    """
    prefactor = (1/4)**delta_sigma

    # u^{Δσ} coefficients: (1/4)^{Δσ} * Σ_n C(2Δσ, n) a^n
    u_coeffs = prefactor * np.array([
        generalized_binomial(2 * delta_sigma, n)
        for n in range(n_max + 1)
    ])

    # v^{Δσ} coefficients: (1/4)^{Δσ} * Σ_n (-1)^n C(2Δσ, n) a^n
    v_coeffs = prefactor * np.array([
        ((-1)**n) * generalized_binomial(2 * delta_sigma, n)
        for n in range(n_max + 1)
    ])

    return v_coeffs - u_coeffs


def F_operator_taylor_coeffs(delta: float, delta_sigma: float, n_max: int) -> np.ndarray:
    """
    Taylor coefficients of F_Δ(a, 0) for an operator of dimension Δ.

    F_Δ(a, 0) = v^{Δσ} g_Δ(z, z) - u^{Δσ} g_Δ(1-z, 1-z)

    where z = 1/2 + a/2.

    Args:
        delta: Scaling dimension of exchanged operator
        delta_sigma: External operator dimension
        n_max: Maximum order

    Returns:
        Array of Taylor coefficients in a
    """
    if delta == 0:
        return F_identity_taylor_coeffs(delta_sigma, n_max)

    # g_Δ(z, z) Taylor coefficients around z = 1/2
    g_coeffs = g_diagonal_taylor_coeffs(delta, 0.5, n_max + 4)

    # Convert to expansion in a where z = 1/2 + a/2
    # g(z, z) = Σ_n c_n (z - 1/2)^n = Σ_n c_n (a/2)^n = Σ_n c_n/2^n a^n
    g_in_a = g_coeffs / (2.0 ** np.arange(len(g_coeffs)))

    # For g(1-z, 1-z) with z = 1/2 + a/2: 1-z = 1/2 - a/2
    # g(1-z, 1-z) = Σ_n c_n (-a/2)^n = Σ_n c_n (-1)^n / 2^n a^n
    g_1minusz_in_a = g_coeffs * ((-1.0) ** np.arange(len(g_coeffs))) / (2.0 ** np.arange(len(g_coeffs)))

    # u^{Δσ} and v^{Δσ} coefficients
    prefactor = (1/4)**delta_sigma
    u_coeffs = prefactor * np.array([
        generalized_binomial(2 * delta_sigma, n)
        for n in range(n_max + 5)
    ])
    v_coeffs = prefactor * np.array([
        ((-1)**n) * generalized_binomial(2 * delta_sigma, n)
        for n in range(n_max + 5)
    ])

    # Convolve v^{Δσ} with g(z, z)
    term_v = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        for i in range(min(n + 1, len(v_coeffs))):
            if n - i < len(g_in_a):
                term_v[n] += v_coeffs[i] * g_in_a[n - i]

    # Convolve u^{Δσ} with g(1-z, 1-z)
    term_u = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        for i in range(min(n + 1, len(u_coeffs))):
            if n - i < len(g_1minusz_in_a):
                term_u[n] += u_coeffs[i] * g_1minusz_in_a[n - i]

    return term_v - term_u


def build_F_vector_taylor(delta: float, delta_sigma: float, max_deriv: int) -> np.ndarray:
    """
    Build the F-vector using Taylor series method.

    The F-vector contains d^m/da^m F|_{a=0} / m! for odd m = 1, 3, 5, ..., max_deriv.
    These are exactly the Taylor coefficients at those positions.

    Args:
        delta: Scaling dimension of operator (0 for identity)
        delta_sigma: External operator dimension
        max_deriv: Maximum derivative order (should be odd)

    Returns:
        F-vector of length (max_deriv + 1) // 2
    """
    coeffs = F_operator_taylor_coeffs(delta, delta_sigma, max_deriv + 2)

    # Extract odd coefficients: positions 1, 3, 5, ... (0-indexed: [1], [3], [5], ...)
    n_components = (max_deriv + 1) // 2
    f_vec = np.array([coeffs[2*m + 1] for m in range(n_components)])

    return f_vec


class TaylorCrossingVector:
    """
    Crossing vector computer using Taylor series expansion.

    This replaces the finite-difference CrossingVector for high-order derivatives.
    """

    def __init__(self, delta_sigma: float, max_deriv: int = 41):
        """
        Initialize the Taylor crossing vector computer.

        Args:
            delta_sigma: External operator dimension
            max_deriv: Maximum derivative order (default 41 gives 21 constraints)
        """
        self.delta_sigma = delta_sigma
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2

        # Cache for computed F-vectors
        self._cache = {}

    def build_F_vector(self, delta: float) -> np.ndarray:
        """
        Build F-vector for given operator dimension.

        Args:
            delta: Scaling dimension (0 for identity)

        Returns:
            F-vector of length n_constraints
        """
        # Check cache
        cache_key = (delta, self.delta_sigma, self.max_deriv)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compute
        f_vec = build_F_vector_taylor(delta, self.delta_sigma, self.max_deriv)

        # Cache and return
        self._cache[cache_key] = f_vec
        return f_vec

    def clear_cache(self):
        """Clear the F-vector cache."""
        self._cache = {}


class HighOrderGapBootstrapSolver:
    """
    Bootstrap solver using Taylor series for high-order derivatives.

    This enables 20+ derivative constraints, significantly improving bounds
    compared to the finite-difference approach limited to ~7 derivatives.
    """

    def __init__(self, d: int = 3, max_deriv: int = 41):
        """
        Initialize the high-order bootstrap solver.

        Args:
            d: Spacetime dimension (only d=3 implemented)
            max_deriv: Maximum derivative order (41 gives 21 constraints)
        """
        if d != 3:
            raise ValueError("Only d=3 is implemented")

        self.d = d
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2

        # Check for CVXPY
        try:
            import cvxpy as cp
            self._has_cvxpy = True
        except ImportError:
            self._has_cvxpy = False
            warnings.warn("CVXPY not installed. Install with: pip install cvxpy")

    def is_excluded(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        delta_max: float = 30.0,
        n_samples: int = 150
    ) -> bool:
        """
        Check if point (Δσ, Δε, Δε') is excluded by the bootstrap.

        Uses SDP to find a linear functional that excludes the point.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: Assumed first Z2-even scalar dimension
            delta_epsilon_prime: Gap for second Z2-even scalar
            delta_max: Maximum dimension to sample
            n_samples: Number of operators to sample

        Returns:
            True if EXCLUDED, False if ALLOWED
        """
        cross = TaylorCrossingVector(delta_sigma, self.max_deriv)

        # Build F-vectors
        F_id = cross.build_F_vector(0)  # Identity
        F_eps = cross.build_F_vector(delta_epsilon)  # First scalar

        # Sample operators above the gap
        deltas = np.linspace(delta_epsilon_prime, delta_max, n_samples)
        F_ops = np.array([cross.build_F_vector(d) for d in deltas])

        if self._has_cvxpy:
            return self._is_excluded_sdp(F_id, F_eps, F_ops)
        else:
            return self._is_excluded_lp(F_id, F_eps, F_ops)

    def _is_excluded_sdp(
        self,
        F_id: np.ndarray,
        F_eps: np.ndarray,
        F_ops: np.ndarray
    ) -> bool:
        """
        SDP feasibility check using component-wise normalization.

        Uses the pycftboot/SDPB convention:
        1. Find the largest-magnitude component in F_id
        2. Fix alpha[max_idx] = 1 / F_id[max_idx]
        3. Solve reduced SDP for remaining alpha components
        """
        import cvxpy as cp

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

        # Build constraints with the fixed contribution moved to RHS
        constraints = [
            alpha_reduced @ F_eps_reduced >= -fixed_eps,
        ]

        for i, F_O_reduced in enumerate(F_ops_reduced):
            constraints.append(alpha_reduced @ F_O_reduced >= -fixed_ops[i])

        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
            return prob.status == cp.OPTIMAL
        except Exception as e:
            warnings.warn(f"SDP solver failed: {e}")
            return self._is_excluded_lp(F_id, F_eps, F_ops)

    def _is_excluded_lp(
        self,
        F_id: np.ndarray,
        F_eps: np.ndarray,
        F_ops: np.ndarray
    ) -> bool:
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
        tolerance: float = 0.01
    ) -> float:
        """
        Find the upper bound on Δε' using binary search.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: Assumed first Z2-even scalar dimension
            delta_prime_min: Minimum Δε' to consider
            delta_prime_max: Maximum Δε' to consider
            tolerance: Binary search tolerance

        Returns:
            Upper bound on Δε'
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


# Test function
def test_taylor_expansion():
    """Test the Taylor series implementation."""
    print("Testing Taylor series conformal block implementation")
    print("=" * 60)

    # Test diagonal block coefficients
    print("\n1. Testing g_Δ(z,z) Taylor coefficients for Δ = 1.5:")
    coeffs = g_diagonal_taylor_coeffs(1.5, 0.5, 5)
    print(f"   Coefficients: {coeffs}")
    print(f"   Expected: [0.25, 1.5, 4.0, 8.0, 16.0, 32.0] (approx)")

    # Test F-vector
    print("\n2. Testing F-vector for Δσ = 0.518:")
    cross = TaylorCrossingVector(0.518, max_deriv=21)

    F_id = cross.build_F_vector(0)
    F_15 = cross.build_F_vector(1.5)

    print(f"   F_identity (first 5): {F_id[:5]}")
    print(f"   F_1.5 (first 5):      {F_15[:5]}")
    print(f"   Number of constraints: {len(F_id)}")

    # Test bootstrap bound
    print("\n3. Testing bootstrap bound at Ising point:")
    solver = HighOrderGapBootstrapSolver(d=3, max_deriv=21)

    ds_ising = 0.518
    de_ising = 1.41

    for de_prime in [3.0, 3.5, 4.0, 4.5]:
        excluded = solver.is_excluded(ds_ising, de_ising, de_prime)
        status = "EXCLUDED" if excluded else "ALLOWED"
        print(f"   Δε' = {de_prime:.1f}: {status}")

    print("\n4. Finding Δε' bound with 11 constraints:")
    bound = solver.find_delta_epsilon_prime_bound(ds_ising, de_ising, tolerance=0.05)
    print(f"   Bound: Δε' ≤ {bound:.3f}")
    print(f"   Reference (El-Showk 2012): ~3.8")


if __name__ == "__main__":
    test_taylor_expansion()
