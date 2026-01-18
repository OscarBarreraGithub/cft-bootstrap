"""
Conformal block computation for CFT bootstrap.

This module provides efficient computation of conformal blocks in various dimensions,
optimized for the numerical bootstrap program.
"""

import numpy as np
from scipy.special import hyp2f1, gamma
from functools import lru_cache
import warnings

# Suppress overflow warnings for large arguments
warnings.filterwarnings('ignore', category=RuntimeWarning)


class ConformalBlock:
    """Base class for conformal blocks in d dimensions."""

    def __init__(self, d: int = 3):
        """
        Initialize conformal block calculator.

        Args:
            d: Spacetime dimension
        """
        self.d = d

    def k_function(self, beta: float, x: float) -> float:
        """
        Compute k_β(x) = x^(β/2) * 2F1(β/2, β/2; β; x)

        This is the basic building block for conformal blocks.
        """
        if x <= 0:
            return 0.0
        if x >= 1:
            return np.inf

        return x**(beta/2) * hyp2f1(beta/2, beta/2, beta, x)

    def scalar_block_3d(self, delta: float, z: float, zbar: float) -> float:
        """
        Compute the 3D scalar conformal block g_{Δ,0}(z, zbar).

        Uses the Dolan-Osborn formula:
        g_{Δ,0}(z,zbar) = (z*zbar)/(z-zbar) * [k_Δ(z)*k_{Δ-1}(zbar) - k_Δ(zbar)*k_{Δ-1}(z)]

        Args:
            delta: Scaling dimension of exchanged operator
            z, zbar: Cross-ratio coordinates

        Returns:
            Value of the conformal block
        """
        if abs(z - zbar) < 1e-10:
            # Diagonal limit: use L'Hopital's rule
            return self._scalar_block_3d_diagonal(delta, z)

        k_delta_z = self.k_function(delta, z)
        k_delta_zbar = self.k_function(delta, zbar)
        k_deltam1_z = self.k_function(delta - 1, z)
        k_deltam1_zbar = self.k_function(delta - 1, zbar)

        return (z * zbar / (z - zbar)) * (
            k_delta_z * k_deltam1_zbar - k_delta_zbar * k_deltam1_z
        )

    def _scalar_block_3d_diagonal(self, delta: float, z: float, eps: float = 1e-6) -> float:
        """Compute diagonal limit using finite difference."""
        g_plus = self.scalar_block_3d(delta, z + eps, z - eps)
        g_minus = self.scalar_block_3d(delta, z - eps, z + eps)
        return (g_plus + g_minus) / 2

    def spinning_block_3d(self, delta: float, ell: int, z: float, zbar: float) -> float:
        """
        Compute 3D conformal block with spin.

        For now, only implemented for ell=0 (scalar).
        TODO: Implement spinning blocks using recursion relations.
        """
        if ell == 0:
            return self.scalar_block_3d(delta, z, zbar)
        else:
            raise NotImplementedError(f"Spinning blocks (ell={ell}) not yet implemented")


class CrossingEquation:
    """
    Implements the crossing equation for <σσσσ> correlator.

    The crossing equation is:
    Σ_O p_O F_O(z, zbar) = 0

    where F_O = v^{Δσ} g_O(z,zbar) - u^{Δσ} g_O(1-z, 1-zbar)
    and u = z*zbar, v = (1-z)(1-zbar)
    """

    def __init__(self, delta_sigma: float, d: int = 3):
        """
        Initialize crossing equation.

        Args:
            delta_sigma: External operator dimension
            d: Spacetime dimension
        """
        self.delta_sigma = delta_sigma
        self.d = d
        self.blocks = ConformalBlock(d)

    def F_vector(self, delta: float, z: float, zbar: float) -> float:
        """
        Compute F_O(z, zbar) for operator with dimension delta.

        For identity (delta=0), g=1.
        """
        u = z * zbar
        v = (1 - z) * (1 - zbar)

        if delta == 0:
            # Identity contribution
            return v**self.delta_sigma - u**self.delta_sigma
        else:
            g_direct = self.blocks.scalar_block_3d(delta, z, zbar)
            g_crossed = self.blocks.scalar_block_3d(delta, 1-z, 1-zbar)
            return v**self.delta_sigma * g_direct - u**self.delta_sigma * g_crossed

    def F_derivative(self, delta: float, m: int, n: int, h: float = 0.001) -> float:
        """
        Compute ∂^m_a ∂^n_b F at the crossing symmetric point.

        Uses coordinates: z = 1/2 + (a+b)/2, zbar = 1/2 + (a-b)/2
        The crossing symmetric point is a=0, b=0 (i.e., z=zbar=1/2)

        Args:
            delta: Operator dimension
            m: Order of derivative in a
            n: Order of derivative in b
            h: Step size for finite difference

        Returns:
            Value of the derivative (normalized by m! * n!)
        """
        def F_ab(a, b):
            z = 0.5 + (a + b) / 2
            zbar = 0.5 + (a - b) / 2
            return self.F_vector(delta, z, zbar)

        # Compute derivative using finite differences
        result = 0.0
        for i in range(m + 1):
            for j in range(n + 1):
                coeff = ((-1)**(m - i + n - j) *
                        self._binomial(m, i) * self._binomial(n, j))
                a_val = (i - m/2) * h
                b_val = (j - n/2) * h
                result += coeff * F_ab(a_val, b_val)

        result /= h**(m + n)

        # Normalize by factorials
        result /= (self._factorial(m) * self._factorial(n))

        return result

    @staticmethod
    @lru_cache(maxsize=1000)
    def _binomial(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        return CrossingEquation._binomial(n-1, k-1) + CrossingEquation._binomial(n-1, k)

    @staticmethod
    @lru_cache(maxsize=100)
    def _factorial(n: int) -> int:
        if n <= 1:
            return 1
        return n * CrossingEquation._factorial(n - 1)

    def build_F_vector(self, delta: float, derivative_orders: list) -> np.ndarray:
        """
        Build the F-vector containing multiple derivatives.

        Args:
            delta: Operator dimension
            derivative_orders: List of (m, n) tuples specifying which derivatives

        Returns:
            numpy array of derivative values
        """
        return np.array([
            self.F_derivative(delta, m, n)
            for m, n in derivative_orders
        ])


def get_derivative_orders(max_order: int, symmetry: str = 'scalar', stable_only: bool = True) -> list:
    """
    Get the relevant derivative orders for the bootstrap.

    For <σσσσ> with identical scalars:
    - F is antisymmetric under (a,b) -> (-a,-b), so m+n must be odd
    - F is symmetric under b -> -b, so n must be even

    Args:
        max_order: Maximum total derivative order (m + n)
        symmetry: Type of symmetry ('scalar' for identical external operators)
        stable_only: If True, only use n=0 derivatives (more numerically stable)

    Returns:
        List of (m, n) tuples
    """
    orders = []
    if stable_only:
        # Only use derivatives along the a-direction (n=0)
        # These are much more numerically stable
        for m in range(1, max_order + 1, 2):  # m odd
            orders.append((m, 0))
    else:
        for m in range(1, max_order + 1, 2):  # m odd
            for n in range(0, max_order - m + 1, 2):  # n even
                if m + n <= max_order:
                    orders.append((m, n))
    return orders


if __name__ == "__main__":
    # Quick test
    blocks = ConformalBlock(d=3)

    print("Testing 3D scalar conformal block:")
    print(f"g(Δ=2, z=0.3, zbar=0.5) = {blocks.scalar_block_3d(2.0, 0.3, 0.5):.6f}")
    print(f"g(Δ=2, z=0.4, zbar=0.4) = {blocks.scalar_block_3d(2.0, 0.4, 0.4):.6f}")

    print("\nTesting crossing equation:")
    cross = CrossingEquation(delta_sigma=0.518, d=3)

    orders = get_derivative_orders(11, stable_only=True)
    print(f"Derivative orders (stable): {orders}")

    print("\nF-vectors:")
    F_id = cross.build_F_vector(0, orders)
    F_1 = cross.build_F_vector(1.0, orders)
    F_141 = cross.build_F_vector(1.41, orders)

    print(f"Identity: {F_id}")
    print(f"Δ=1.0:    {F_1}")
    print(f"Δ=1.41:   {F_141}")

    # Check signs - identity should have opposite sign from operators in some components
    print("\nSign analysis:")
    print(f"F_id signs:   {np.sign(F_id)}")
    print(f"F(1.0) signs: {np.sign(F_1)}")
    print(f"F(1.41) signs: {np.sign(F_141)}")
