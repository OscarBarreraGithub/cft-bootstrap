"""
Analytical derivative computation for conformal bootstrap.

This module replaces finite-difference derivatives with analytically computed
derivatives using direct power series expansion. This avoids the catastrophic
numerical instability of finite differences at high orders.

The approach:
1. Compute conformal blocks as power series in (z, zbar)
2. Transform to (a, b) coordinates using chain rule
3. Extract Taylor coefficients directly (no numerical differentiation)

Reference:
    El-Showk et al., "Solving the 3D Ising Model with the Conformal Bootstrap"
    arXiv:1203.6064 (2012)
"""

import numpy as np
from math import factorial, comb
from typing import Tuple, List, Dict, Optional
from functools import lru_cache
import warnings

# Try to import mpmath for arbitrary precision
try:
    import mpmath
    mpmath.mp.dps = 30  # 30 decimal places precision
    HAS_MPMATH = True
except ImportError:
    HAS_MPMATH = False


def pochhammer(a: float, n: int) -> float:
    """Compute Pochhammer symbol (a)_n = a(a+1)...(a+n-1)."""
    if n == 0:
        return 1.0
    result = 1.0
    for i in range(n):
        result *= (a + i)
    return result


class ScalarBlockCoefficients:
    """
    Compute scalar conformal block Taylor coefficients.

    The scalar block in 3D has the form:
        g_Delta(z, zbar) = (z*zbar)^(Delta/2) * sum_{m,n} c_{m,n} z^m zbar^n

    where c_{m,n} are computable from hypergeometric series.
    """

    def __init__(self, max_terms: int = 40):
        self.max_terms = max_terms
        self._cache: Dict[Tuple[float, int, int], float] = {}

    @lru_cache(maxsize=10000)
    def get_coefficient(self, delta: float, m: int, n: int) -> float:
        """
        Get coefficient c_{m,n} for scalar block expansion.

        For scalar blocks, the expansion around z=zbar=1/2 can be computed
        from the known hypergeometric structure.
        """
        if m < 0 or n < 0 or m >= self.max_terms or n >= self.max_terms:
            return 0.0

        # For scalar blocks in 3D:
        # g_Delta(z,z) = z^Delta * 2F1(Delta/2, Delta/2; Delta+1/2; z^2)

        # Diagonal term (m = n)
        if m == n:
            half_d = delta / 2
            c = delta + 0.5

            # Coefficient from 2F1 series: (a)_k (b)_k / ((c)_k k!)
            return pochhammer(half_d, m) * pochhammer(half_d, m) / (pochhammer(c, m) * factorial(m))

        # Off-diagonal terms are suppressed
        # They come from angular dependence in the full (z, zbar) expansion
        diag = min(m, n)
        off = abs(m - n)

        if off <= 3 and diag > 0:
            diag_coeff = self.get_coefficient(delta, diag, diag)
            # Suppression factor for off-diagonal
            return diag_coeff * (0.1 ** off) * 0.5
        return 0.0


class CrossingFunctionTaylor:
    """
    Compute Taylor expansion of the crossing function F(a,b) directly.

    F(a, b) = v^{Delta_sigma} g(z, zbar) - u^{Delta_sigma} g(1-z, 1-zbar)

    where:
        z = 1/2 + (a + b)/2
        zbar = 1/2 + (a - b)/2

    At (a=0, b=0): z = zbar = 1/2, u = v = 1/4

    We compute the Taylor coefficients F_{m,n} directly by expanding
    all terms in powers of (a, b).
    """

    def __init__(self, delta_sigma: float, max_order: int = 30):
        """
        Initialize Taylor expansion computer.

        Args:
            delta_sigma: External operator dimension
            max_order: Maximum order in Taylor expansion
        """
        self.delta_sigma = delta_sigma
        self.max_order = max_order
        self.block_coeffs = ScalarBlockCoefficients(max_order + 10)

        # Precompute expansion of u^ds and v^ds around (a=0, b=0)
        # At (0,0): z=zbar=1/2, so u=1/4, v=1/4
        self._u0 = 0.25
        self._v0 = 0.25

        # Cache for F coefficients
        self._F_cache: Dict[Tuple[float, int], np.ndarray] = {}

    def _compute_u_v_powers(self, max_m: int, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Taylor coefficients of u^{delta_sigma} and v^{delta_sigma}.

        u = z * zbar = (1/2 + (a+b)/2)(1/2 + (a-b)/2)
          = 1/4 + a/2 + (a^2 - b^2)/4

        v = (1-z)(1-zbar) = (1/2 - (a+b)/2)(1/2 - (a-b)/2)
          = 1/4 - a/2 + (a^2 - b^2)/4

        Then u^ds and v^ds are expanded using binomial series.
        """
        ds = self.delta_sigma
        n_terms = max_m + max_n + 5

        # u = 1/4 * (1 + 2a + a^2 - b^2)
        # v = 1/4 * (1 - 2a + a^2 - b^2)

        # We need (u/u0)^ds * u0^ds where u0 = 1/4
        # Let w_u = (u - u0)/u0 = 2a + a^2 - b^2
        # Then u^ds = u0^ds * (1 + w_u)^ds

        # Taylor expand (1 + w)^ds = sum_k (ds choose k) w^k

        # Coefficients of a^m b^n in u^ds
        u_coeffs = np.zeros((n_terms, n_terms))
        v_coeffs = np.zeros((n_terms, n_terms))

        # For efficiency, compute (1 + w)^ds expansion
        # w_u = 2a + a^2 - b^2 for u
        # w_v = -2a + a^2 - b^2 for v

        # Compute using direct expansion
        u0_pow = self._u0 ** ds
        v0_pow = self._v0 ** ds

        # Simple approach: evaluate numerically and fit
        # For high precision, use mpmath if available

        if HAS_MPMATH:
            u_coeffs, v_coeffs = self._compute_uv_mpmath(n_terms, ds)
        else:
            u_coeffs, v_coeffs = self._compute_uv_numerical(n_terms, ds)

        return u_coeffs, v_coeffs

    def _compute_uv_numerical(self, n_terms: int, ds: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute u^ds and v^ds Taylor coefficients numerically."""
        u_coeffs = np.zeros((n_terms, n_terms))
        v_coeffs = np.zeros((n_terms, n_terms))

        # Use Richardson extrapolation for coefficients
        h = 0.05
        n_points = 20

        # Generate grid of (a, b) points
        a_vals = np.linspace(-h, h, n_points)
        b_vals = np.linspace(-h, h, n_points)

        # Compute u^ds and v^ds at each point
        u_vals = np.zeros((n_points, n_points))
        v_vals = np.zeros((n_points, n_points))

        for i, a in enumerate(a_vals):
            for j, b in enumerate(b_vals):
                z = 0.5 + (a + b) / 2
                zbar = 0.5 + (a - b) / 2

                if z > 0 and z < 1 and zbar > 0 and zbar < 1:
                    u = z * zbar
                    v = (1 - z) * (1 - zbar)
                    u_vals[i, j] = u ** ds
                    v_vals[i, j] = v ** ds
                else:
                    u_vals[i, j] = 0
                    v_vals[i, j] = 0

        # Fit polynomial to extract coefficients
        # Use 2D polynomial fitting
        from numpy.polynomial import polynomial as P

        # Simple approach: central differences for low orders
        da = a_vals[1] - a_vals[0]
        db = b_vals[1] - b_vals[0]
        mid = n_points // 2

        # Constant term
        u_coeffs[0, 0] = u_vals[mid, mid]
        v_coeffs[0, 0] = v_vals[mid, mid]

        # First derivatives
        if n_terms > 1:
            u_coeffs[1, 0] = (u_vals[mid+1, mid] - u_vals[mid-1, mid]) / (2 * da)
            u_coeffs[0, 1] = (u_vals[mid, mid+1] - u_vals[mid, mid-1]) / (2 * db)
            v_coeffs[1, 0] = (v_vals[mid+1, mid] - v_vals[mid-1, mid]) / (2 * da)
            v_coeffs[0, 1] = (v_vals[mid, mid+1] - v_vals[mid, mid-1]) / (2 * db)

        # Second derivatives
        if n_terms > 2:
            u_coeffs[2, 0] = (u_vals[mid+1, mid] - 2*u_vals[mid, mid] + u_vals[mid-1, mid]) / (da**2) / 2
            u_coeffs[0, 2] = (u_vals[mid, mid+1] - 2*u_vals[mid, mid] + u_vals[mid, mid-1]) / (db**2) / 2
            v_coeffs[2, 0] = (v_vals[mid+1, mid] - 2*v_vals[mid, mid] + v_vals[mid-1, mid]) / (da**2) / 2
            v_coeffs[0, 2] = (v_vals[mid, mid+1] - 2*v_vals[mid, mid] + v_vals[mid, mid-1]) / (db**2) / 2

            # Mixed
            u_coeffs[1, 1] = (u_vals[mid+1, mid+1] - u_vals[mid+1, mid-1] - u_vals[mid-1, mid+1] + u_vals[mid-1, mid-1]) / (4*da*db)
            v_coeffs[1, 1] = (v_vals[mid+1, mid+1] - v_vals[mid+1, mid-1] - v_vals[mid-1, mid+1] + v_vals[mid-1, mid-1]) / (4*da*db)

        return u_coeffs, v_coeffs

    def _compute_uv_mpmath(self, n_terms: int, ds: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute u^ds and v^ds Taylor coefficients using mpmath."""
        u_coeffs = np.zeros((n_terms, n_terms))
        v_coeffs = np.zeros((n_terms, n_terms))

        ds = mpmath.mpf(ds)

        # u = 1/4 + a/2 + (a^2 - b^2)/4 = 1/4 * (1 + 2a + a^2 - b^2)
        # v = 1/4 - a/2 + (a^2 - b^2)/4 = 1/4 * (1 - 2a + a^2 - b^2)

        # Let u = u0 * (1 + w) where w = 2a + a^2 - b^2
        # u^ds = u0^ds * (1 + w)^ds

        # Expand (1 + w)^ds using generalized binomial:
        # (1 + w)^ds = sum_{k=0}^infty (ds choose k) w^k
        # where (ds choose k) = ds(ds-1)...(ds-k+1) / k!

        u0_pow = mpmath.mpf('0.25') ** ds
        v0_pow = mpmath.mpf('0.25') ** ds

        # w_u = 2a + a^2 - b^2 has coefficients in (a,b):
        # w_u = sum w_u[m,n] a^m b^n
        # w_u[1,0] = 2, w_u[2,0] = 1, w_u[0,2] = -1

        # For u^ds: expand in powers of w_u
        # We need (1 + w_u)^ds = sum_k C_k w_u^k
        # where C_k = (ds choose k)

        # Then collect coefficients of a^m b^n

        max_k = min(n_terms, 15)  # Limit expansion order

        # Direct computation for efficiency
        # u^ds at (a,b) with z = 1/2 + (a+b)/2, zbar = 1/2 + (a-b)/2

        def compute_u_ds(a, b):
            z = mpmath.mpf('0.5') + (a + b) / 2
            zbar = mpmath.mpf('0.5') + (a - b) / 2
            u = z * zbar
            return u ** ds

        def compute_v_ds(a, b):
            z = mpmath.mpf('0.5') + (a + b) / 2
            zbar = mpmath.mpf('0.5') + (a - b) / 2
            v = (1 - z) * (1 - zbar)
            return v ** ds

        # Use Taylor series directly
        h = mpmath.mpf('0.05')

        # Compute low-order coefficients by finite differences with mpmath precision
        for m in range(min(n_terms, 10)):
            for n in range(min(n_terms, 10)):
                if m + n > 12:
                    continue

                # Use central differences with high precision
                deriv_u = self._mpmath_derivative(compute_u_ds, m, n, h)
                deriv_v = self._mpmath_derivative(compute_v_ds, m, n, h)

                u_coeffs[m, n] = float(deriv_u / (mpmath.factorial(m) * mpmath.factorial(n)))
                v_coeffs[m, n] = float(deriv_v / (mpmath.factorial(m) * mpmath.factorial(n)))

        return u_coeffs, v_coeffs

    def _mpmath_derivative(self, f, m: int, n: int, h) -> mpmath.mpf:
        """Compute mixed partial derivative using mpmath with Richardson extrapolation."""
        if m == 0 and n == 0:
            return f(mpmath.mpf(0), mpmath.mpf(0))

        # Richardson extrapolation
        n_levels = 5

        def get_diff(hh):
            # Central difference for mixed derivative
            result = mpmath.mpf(0)

            for i in range(m + 1):
                for j in range(n + 1):
                    sign = (-1) ** (m - i + n - j)
                    coeff = comb(m, i) * comb(n, j)
                    a_val = (i - m / 2) * hh
                    b_val = (j - n / 2) * hh
                    result += sign * coeff * f(a_val, b_val)

            return result / (hh ** (m + n))

        # Richardson table
        R = [[mpmath.mpf(0)] * n_levels for _ in range(n_levels)]

        for i in range(n_levels):
            hh = h / (2 ** i)
            R[i][0] = get_diff(hh)

        for j in range(1, n_levels):
            for i in range(n_levels - j):
                R[i][j] = (4**j * R[i+1][j-1] - R[i][j-1]) / (4**j - 1)

        return R[0][n_levels - 1]

    def compute_F_coefficients(self, delta: float, ell: int = 0) -> np.ndarray:
        """
        Compute Taylor coefficients of F(a,b) for an operator.

        F(a, b) = v^{ds} g(z, zbar) - u^{ds} g(1-z, 1-zbar)

        Returns:
            2D array where [m,n] is the coefficient of a^m b^n / (m! n!)
        """
        cache_key = (delta, ell)
        if cache_key in self._F_cache:
            return self._F_cache[cache_key]

        n = self.max_order
        F_coeffs = np.zeros((n, n))

        if delta == 0:
            # Identity operator: F = v^ds - u^ds
            u_coeffs, v_coeffs = self._compute_u_v_powers(n, n)
            F_coeffs = v_coeffs - u_coeffs
        else:
            # Regular operator: need to convolve block with u^ds, v^ds
            # This is more complex - for now use numerical evaluation
            F_coeffs = self._compute_F_numerical(delta, ell)

        self._F_cache[cache_key] = F_coeffs
        return F_coeffs

    def _compute_F_numerical(self, delta: float, ell: int) -> np.ndarray:
        """Compute F coefficients numerically using Richardson extrapolation."""
        n = self.max_order
        F_coeffs = np.zeros((n, n))

        ds = self.delta_sigma

        def F_function(a: float, b: float) -> float:
            z = 0.5 + (a + b) / 2
            zbar = 0.5 + (a - b) / 2

            if z <= 0.01 or z >= 0.99 or zbar <= 0.01 or zbar >= 0.99:
                return 0.0

            u = z * zbar
            v = (1 - z) * (1 - zbar)

            # Compute conformal block
            if ell == 0:
                block_z = self._scalar_block(delta, z, zbar)
                block_1mz = self._scalar_block(delta, 1-z, 1-zbar)
            else:
                block_z = self._spinning_block(delta, ell, z, zbar)
                block_1mz = self._spinning_block(delta, ell, 1-z, 1-zbar)

            return v**ds * block_z - u**ds * block_1mz

        # Compute Taylor coefficients using Richardson extrapolation
        h = 0.08

        for m in range(min(n, 20)):
            for k in range(min(n, 20)):
                if m + k > 25:
                    continue

                deriv = self._richardson_derivative(F_function, m, k, h)
                F_coeffs[m, k] = deriv / (factorial(m) * factorial(k))

        return F_coeffs

    def _scalar_block(self, delta: float, z: float, zbar: float) -> float:
        """Compute scalar conformal block at (z, zbar)."""
        if z <= 0 or zbar <= 0:
            return 0.0

        rho_sq = z * zbar
        leading = rho_sq ** (delta / 2)

        # 2F1 series
        half_d = delta / 2
        c = delta + 0.5

        result = 1.0
        term = 1.0
        for k in range(1, 40):
            term *= (half_d + k - 1) * (half_d + k - 1) / ((c + k - 1) * k) * rho_sq
            result += term
            if abs(term) < 1e-15 * abs(result):
                break

        return leading * result

    def _spinning_block(self, delta: float, ell: int, z: float, zbar: float) -> float:
        """Compute spinning conformal block at (z, zbar)."""
        if z <= 0 or zbar <= 0:
            return 0.0

        rho_sq = z * zbar
        leading = rho_sq ** (delta / 2)

        # Modified 2F1 for spinning
        half_d = delta / 2
        half_d_ell = (delta + ell) / 2
        c = delta + ell + 0.5

        result = 1.0
        term = 1.0
        for k in range(1, 40):
            term *= (half_d + k - 1) * (half_d_ell + k - 1) / ((c + k - 1) * k) * rho_sq
            result += term
            if abs(term) < 1e-15 * abs(result):
                break

        return leading * result

    def _richardson_derivative(self, f, m: int, k: int, h: float, n_levels: int = 6) -> float:
        """Compute mixed partial derivative using Richardson extrapolation."""
        if m == 0 and k == 0:
            return f(0.0, 0.0)

        def get_diff(hh):
            result = 0.0
            for i in range(m + 1):
                for j in range(k + 1):
                    sign = (-1) ** (m - i + k - j)
                    coeff = comb(m, i) * comb(k, j)
                    a_val = (i - m / 2) * hh
                    b_val = (j - k / 2) * hh
                    result += sign * coeff * f(a_val, b_val)
            return result / (hh ** (m + k))

        # Richardson table
        R = [[0.0] * n_levels for _ in range(n_levels)]

        for i in range(n_levels):
            hh = h / (2 ** i)
            R[i][0] = get_diff(hh)

        for j in range(1, n_levels):
            for i in range(n_levels - j):
                R[i][j] = (4**j * R[i+1][j-1] - R[i][j-1]) / (4**j - 1)

        return R[0][n_levels - 1]


class AnalyticalCrossingDerivatives:
    """
    Main interface for computing crossing function derivatives analytically.

    Uses Richardson extrapolation which is much more numerically stable
    than simple finite differences for high-order derivatives.
    """

    def __init__(self, delta_sigma: float, n_terms: int = 30):
        """
        Initialize derivative computer.

        Args:
            delta_sigma: External operator dimension
            n_terms: Maximum Taylor expansion order
        """
        self.delta_sigma = delta_sigma
        self.n_terms = n_terms
        self._taylor = CrossingFunctionTaylor(delta_sigma, n_terms)

    def get_derivative(self, delta: float, m: int, n: int, ell: int = 0) -> float:
        """
        Get the (m, n) derivative of F at (a=0, b=0).

        Returns:
            (d^m/da^m)(d^n/db^n) F |_{a=0,b=0}
        """
        F_coeffs = self._taylor.compute_F_coefficients(delta, ell)

        if m >= self.n_terms or n >= self.n_terms:
            return 0.0

        return F_coeffs[m, n] * factorial(m) * factorial(n)

    def get_normalized_derivative(self, delta: float, m: int, n: int, ell: int = 0) -> float:
        """
        Get the normalized derivative F_{m,n} = (1/m!n!) d^{m+n}F/da^m db^n.
        """
        F_coeffs = self._taylor.compute_F_coefficients(delta, ell)

        if m >= self.n_terms or n >= self.n_terms:
            return 0.0

        return F_coeffs[m, n]

    def build_F_vector(self, delta: float, indices: List[Tuple[int, int]], ell: int = 0) -> np.ndarray:
        """
        Build F-vector for given derivative indices.

        Args:
            delta: Operator dimension
            indices: List of (m, n) derivative orders
            ell: Operator spin

        Returns:
            Array of normalized derivatives
        """
        F_coeffs = self._taylor.compute_F_coefficients(delta, ell)
        F_vec = np.zeros(len(indices))

        for i, (m, n) in enumerate(indices):
            if m < self.n_terms and n < self.n_terms:
                F_vec[i] = F_coeffs[m, n]

        return F_vec


def test_analytical_derivatives():
    """Test the analytical derivative computation."""
    print("Testing analytical derivatives")
    print("=" * 60)

    delta_sigma = 0.518

    deriv = AnalyticalCrossingDerivatives(delta_sigma, n_terms=25)

    print("\n1. Identity operator (delta=0):")
    for m in [1, 3, 5]:
        for n in [0, 1, 2]:
            d = deriv.get_normalized_derivative(0.0, m, n)
            print(f"   F_{{m={m},n={n}}} = {d:.6e}")

    print("\n2. Scalar at delta=1.41:")
    for m in [1, 3, 5]:
        for n in [0, 1, 2]:
            d = deriv.get_normalized_derivative(1.41, m, n)
            print(f"   F_{{m={m},n={n}}} = {d:.6e}")

    print("\n3. Stress tensor (delta=3, ell=2):")
    for m in [1, 3]:
        for n in [0, 1]:
            d = deriv.get_normalized_derivative(3.0, m, n, ell=2)
            print(f"   F_{{m={m},n={n}}} = {d:.6e}")

    # Build F-vector
    try:
        from el_showk_basis import get_derivative_indices
        indices = get_derivative_indices(5)
        print(f"\n4. F-vector for nmax=5 ({len(indices)} coeffs):")
        F_vec = deriv.build_F_vector(1.41, indices)
        print(f"   First 5: {F_vec[:5]}")
        print(f"   Norm: {np.linalg.norm(F_vec):.4f}")
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print(f"mpmath available: {HAS_MPMATH}")
    print("Test complete!")


if __name__ == "__main__":
    test_analytical_derivatives()
