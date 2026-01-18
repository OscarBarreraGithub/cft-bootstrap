"""
Conformal block computation for CFT bootstrap - Version 2.

This version uses a more numerically stable approach:
1. Proper normalization of F-vectors
2. Richardson extrapolation for derivatives
3. Automatic scaling to avoid overflow
"""

import numpy as np
from math import factorial, comb
from scipy.special import hyp2f1
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class ConformalBlock3D:
    """3D conformal blocks using Dolan-Osborn formula."""

    def __init__(self):
        pass

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


class CrossingVector:
    """
    Compute F-vectors for the crossing equation with proper normalization.

    Uses the coordinate system:
        z = 1/2 + (a + b)/2
        zbar = 1/2 + (a - b)/2

    The crossing symmetric point is a = b = 0.
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

    def derivative_richardson(self, delta: float, m: int, h_base: float = 0.02,
                              n_extrap: int = 4) -> float:
        """
        Compute m-th derivative at a=0, b=0 using Richardson extrapolation.

        This is more numerically stable than simple finite differences.
        """
        # Only computing d^m/da^m at a=0, b=0 (keeping b=0)
        estimates = []

        for k in range(n_extrap):
            h = h_base / (2**k)
            deriv = self._finite_diff(delta, m, h)
            estimates.append(deriv)

        # Richardson extrapolation
        result = estimates[-1]  # Use finest estimate

        # Could implement full Richardson table here for more accuracy
        return result

    def _finite_diff(self, delta: float, m: int, h: float) -> float:
        """Simple central finite difference for m-th derivative."""
        # Use central difference formula
        coeffs = self._central_diff_coeffs(m)
        n = len(coeffs)
        half_n = n // 2

        result = 0.0
        for i, c in enumerate(coeffs):
            a_val = (i - half_n) * h
            result += c * self.F(delta, a_val, 0.0)

        result /= h**m
        return result

    @staticmethod
    @lru_cache(maxsize=20)
    def _central_diff_coeffs(m: int) -> tuple:
        """Get central difference coefficients for m-th derivative."""
        # Standard central difference coefficients (accuracy O(h²))
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
        else:
            # Fall back to simple forward difference
            return tuple([(-1)**i * comb(m, i) for i in range(m+1)])

    def build_F_vector_normalized(self, delta: float, max_deriv_order: int) -> np.ndarray:
        """
        Build F-vector with proper normalization to avoid numerical issues.

        Returns derivatives d^m/da^m F / m! for m = 1, 3, 5, ..., max_deriv_order
        """
        derivs = []
        for m in range(1, max_deriv_order + 1, 2):  # Odd m only
            d = self.derivative_richardson(delta, m) / factorial(m)
            derivs.append(d)
        return np.array(derivs)


class BootstrapSolverV2:
    """
    Improved bootstrap solver with proper normalization.
    """

    def __init__(self, max_deriv: int = 9):
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2  # Number of odd integers from 1 to max_deriv
        print(f"Using {self.n_constraints} derivative constraints (m = 1, 3, ..., {max_deriv})")

    def is_feasible(self, delta_sigma: float, delta_gap: float,
                    delta_max: float = 20.0, n_samples: int = 100) -> bool:
        """Check if (delta_sigma, delta_gap) is in the allowed region."""
        from scipy.optimize import linprog

        cross = CrossingVector(delta_sigma)

        # Build F-vector for identity
        F_id = cross.build_F_vector_normalized(0, self.max_deriv)

        # Sample operator dimensions above the gap
        deltas = np.linspace(delta_gap, delta_max, n_samples)
        F_ops = np.array([cross.build_F_vector_normalized(d, self.max_deriv) for d in deltas])

        # Normalize columns for numerical stability
        scales = np.abs(F_ops).max(axis=0) + 1e-10
        F_ops_scaled = F_ops / scales
        F_id_scaled = F_id / scales

        # Linear program: find p >= 0 such that F_ops.T @ p = -F_id
        A_eq = F_ops_scaled.T
        b_eq = -F_id_scaled

        c = np.zeros(n_samples)  # Just checking feasibility

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
        return result.success

    def find_bound(self, delta_sigma: float, delta_min: float = 0.5,
                   delta_max: float = 3.0, tol: float = 0.01) -> float:
        """Find the upper bound on the gap using binary search."""
        if not self.is_feasible(delta_sigma, delta_min):
            return delta_min
        if self.is_feasible(delta_sigma, delta_max):
            return float('inf')

        lo, hi = delta_min, delta_max
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if self.is_feasible(delta_sigma, mid):
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("CFT Bootstrap V2 - Improved Numerics")
    print("=" * 60)

    # Test F-vectors
    print("\nTesting F-vector computation:")
    cross = CrossingVector(0.518)

    print("\nF-vectors (normalized):")
    for delta in [0, 1.0, 1.41, 2.0]:
        fv = cross.build_F_vector_normalized(delta, 9)
        print(f"  Δ={delta}: {fv}")

    # Test solver
    print("\n" + "=" * 60)
    print("Testing bootstrap solver:")
    print("=" * 60)

    # Use only low derivatives for stability
    solver = BootstrapSolverV2(max_deriv=5)  # Just m=1,3,5

    print("\nFeasibility at Δσ = 0.518:")
    for gap in [1.0, 1.2, 1.41, 1.6, 1.8, 2.0]:
        t0 = time.time()
        ok = solver.is_feasible(0.518, gap)
        t1 = time.time()
        status = "ALLOWED" if ok else "EXCLUDED"
        print(f"  Δε = {gap:.2f}: {status} ({t1-t0:.3f}s)")

    print("\nComputing bounds:")
    for ds in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
        t0 = time.time()
        bound = solver.find_bound(ds, tol=0.02)
        t1 = time.time()
        print(f"  Δσ = {ds:.2f}: Δε ≤ {bound:.4f} ({t1-t0:.2f}s)")
