"""
F-vectors for mixed correlator bootstrap.

This module computes crossing vectors (F-vectors) for the three correlators
needed in mixed correlator bootstrap:
- <sigma sigma sigma sigma> (ssss)
- <sigma sigma epsilon epsilon> (ssee)
- <epsilon epsilon epsilon epsilon> (eeee)

The key new component is the ssee F-vector which has different prefactors
than the symmetric correlators.

Mathematical Background:
    For the ssee correlator, the crossing equation involves:

    F^{ssee}_Delta = v^{Delta_sigma} g_Delta(z, zbar)
                   - u^{Delta_epsilon} v^{Delta_sigma - Delta_epsilon} g_Delta(1-z, 1-zbar)

    Compare to ssss:
    F^{ssss}_Delta = v^{Delta_sigma} g_Delta(z, zbar)
                   - u^{Delta_sigma} g_Delta(1-z, 1-zbar)

    The different powers of u and v in ssee arise from the different
    external operator dimensions.

References:
    - El-Showk et al., arXiv:1203.6064 (2012)
    - Kos, Poland, Simmons-Duffin, arXiv:1406.4858 (2014)
"""

import numpy as np
from typing import Optional, Tuple, Dict
from functools import lru_cache

try:
    from .taylor_conformal_blocks import (
        TaylorCrossingVector,
        g_diagonal_taylor_coeffs,
        generalized_binomial,
        F_identity_taylor_coeffs,
        F_operator_taylor_coeffs
    )
except ImportError:
    from taylor_conformal_blocks import (
        TaylorCrossingVector,
        g_diagonal_taylor_coeffs,
        generalized_binomial,
        F_identity_taylor_coeffs,
        F_operator_taylor_coeffs
    )


def convolve_truncate(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """
    Convolve two arrays and truncate to length n.

    Args:
        a: First array (polynomial coefficients)
        b: Second array (polynomial coefficients)
        n: Desired output length

    Returns:
        First n terms of convolution
    """
    result = np.convolve(a, b)
    if len(result) < n:
        return np.pad(result, (0, n - len(result)))
    return result[:n]


class MixedCrossingVector:
    """
    Crossing vectors for the mixed correlator bootstrap system.

    Computes F-vectors for all three correlators:
    - ssss: <sigma sigma sigma sigma>
    - ssee: <sigma sigma epsilon epsilon>
    - eeee: <epsilon epsilon epsilon epsilon>

    The ssss and eeee correlators reuse TaylorCrossingVector since they
    have the same functional form (just different external dimensions).
    The ssee correlator requires a new implementation due to the mixed
    prefactors.

    Attributes:
        delta_sigma: External dimension of sigma operator
        delta_epsilon: External dimension of epsilon operator
        max_deriv: Maximum derivative order
        n_constraints: Number of F-vector components = (max_deriv + 1) // 2
    """

    def __init__(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        max_deriv: int = 21
    ):
        """
        Initialize the mixed crossing vector computer.

        Args:
            delta_sigma: Dimension of sigma (spin operator)
            delta_epsilon: Dimension of epsilon (energy operator)
            max_deriv: Maximum derivative order (21 gives 11 constraints)
        """
        self.delta_sigma = delta_sigma
        self.delta_epsilon = delta_epsilon
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2

        # Reuse existing TaylorCrossingVector for ssss and eeee
        self._cross_ssss = TaylorCrossingVector(delta_sigma, max_deriv)
        self._cross_eeee = TaylorCrossingVector(delta_epsilon, max_deriv)

        # Cache for ssee F-vectors
        self._cache_ssee: Dict[float, np.ndarray] = {}

        # Precompute prefactor coefficients for ssee
        self._precompute_ssee_prefactors()

    def _precompute_ssee_prefactors(self):
        """
        Precompute the Taylor coefficients for ssee prefactors.

        For ssee:
        - Direct channel: v^{Delta_sigma}
        - Crossed channel: u^{Delta_epsilon} * v^{Delta_sigma - Delta_epsilon}

        On the diagonal with z = 1/2 + a/2:
        - u = z^2 = (1/2 + a/2)^2 = (1/4)(1+a)^2
        - v = (1-z)^2 = (1/2 - a/2)^2 = (1/4)(1-a)^2
        """
        ds = self.delta_sigma
        de = self.delta_epsilon
        n_max = self.max_deriv + 5

        # v^{ds} = (1/4)^{ds} * (1-a)^{2*ds}
        # Taylor coefficients of (1-a)^{2*ds} around a=0
        self._v_ds_coeffs = (0.25)**ds * np.array([
            ((-1)**n) * generalized_binomial(2*ds, n)
            for n in range(n_max)
        ])

        # u^{de} * v^{ds-de} = (1/4)^{ds} * (1+a)^{2*de} * (1-a)^{2*(ds-de)}
        # First compute (1+a)^{2*de} coefficients
        u_part = np.array([generalized_binomial(2*de, n) for n in range(n_max)])

        # Then (1-a)^{2*(ds-de)} coefficients
        v_part = np.array([
            ((-1)**n) * generalized_binomial(2*(ds - de), n)
            for n in range(n_max)
        ])

        # Convolve and scale
        self._u_de_v_diff_coeffs = (0.25)**ds * convolve_truncate(u_part, v_part, n_max)

    def build_F_vector_ssss(self, delta: float, ell: int = 0) -> np.ndarray:
        """
        Build F-vector for <sigma sigma sigma sigma>.

        Args:
            delta: Scaling dimension of exchanged operator (0 for identity)
            ell: Spin of exchanged operator (only ell=0 implemented)

        Returns:
            F-vector of length n_constraints
        """
        if ell != 0:
            raise NotImplementedError("Only scalar exchanges (ell=0) implemented")
        return self._cross_ssss.build_F_vector(delta)

    def build_F_vector_eeee(self, delta: float, ell: int = 0) -> np.ndarray:
        """
        Build F-vector for <epsilon epsilon epsilon epsilon>.

        Args:
            delta: Scaling dimension of exchanged operator (0 for identity)
            ell: Spin of exchanged operator (only ell=0 implemented)

        Returns:
            F-vector of length n_constraints
        """
        if ell != 0:
            raise NotImplementedError("Only scalar exchanges (ell=0) implemented")
        return self._cross_eeee.build_F_vector(delta)

    def build_F_vector_ssee(self, delta: float, ell: int = 0) -> np.ndarray:
        """
        Build F-vector for <sigma sigma epsilon epsilon>.

        The crossing function is:
            F^{ssee}_Delta = v^{ds} g_Delta(z,z) - u^{de} v^{ds-de} g_Delta(1-z,1-z)

        where ds = Delta_sigma and de = Delta_epsilon.

        Args:
            delta: Scaling dimension of exchanged operator (0 for identity)
            ell: Spin of exchanged operator (only ell=0 implemented)

        Returns:
            F-vector of length n_constraints
        """
        if ell != 0:
            raise NotImplementedError("Only scalar exchanges (ell=0) implemented")

        # Check cache
        if delta in self._cache_ssee:
            return self._cache_ssee[delta]

        # Compute
        if delta == 0:
            f_vec = self._compute_ssee_identity()
        else:
            f_vec = self._compute_ssee_operator(delta)

        # Cache and return
        self._cache_ssee[delta] = f_vec
        return f_vec

    def _compute_ssee_identity(self) -> np.ndarray:
        """
        Compute F^{ssee} for the identity operator.

        F^{ssee}_id = v^{ds} - u^{de} v^{ds-de}

        where ds = Delta_sigma and de = Delta_epsilon.

        Returns:
            F-vector of length n_constraints
        """
        # The difference of the two prefactors
        F_id_coeffs = self._v_ds_coeffs - self._u_de_v_diff_coeffs

        # Extract odd coefficients for F-vector
        # F-vector contains coefficients at positions 1, 3, 5, ... (odd powers of a)
        f_vec = np.array([
            F_id_coeffs[2*m + 1] if 2*m + 1 < len(F_id_coeffs) else 0.0
            for m in range(self.n_constraints)
        ])

        return f_vec

    def _compute_ssee_operator(self, delta: float) -> np.ndarray:
        """
        Compute F^{ssee} for an operator with dimension delta.

        F^{ssee}_Delta = v^{ds} g_Delta(z,z) - u^{de} v^{ds-de} g_Delta(1-z,1-z)

        On diagonal with z = 1/2 + a/2:
        - g(z,z) has Taylor coefficients g_n around z=1/2
        - g(1-z,1-z) = g(1/2 - a/2, 1/2 - a/2) has coefficients (-1)^n g_n

        Args:
            delta: Scaling dimension of exchanged operator

        Returns:
            F-vector of length n_constraints
        """
        n_max = self.max_deriv + 5

        # Get g_Delta(z,z) Taylor coefficients around z = 1/2
        g_coeffs = g_diagonal_taylor_coeffs(delta, 0.5, n_max)

        # Convert to expansion in a-coordinate: z = 1/2 + a/2
        # g(z,z) = sum_n c_n (z - 1/2)^n = sum_n c_n (a/2)^n
        g_in_a = g_coeffs / (2.0 ** np.arange(len(g_coeffs)))

        # For g(1-z, 1-z): 1-z = 1/2 - a/2
        # g(1-z, 1-z) = sum_n c_n (-a/2)^n = sum_n (-1)^n c_n (a/2)^n
        g_1mz_in_a = g_coeffs * ((-1.0) ** np.arange(len(g_coeffs))) / (2.0 ** np.arange(len(g_coeffs)))

        # Convolve: v^{ds} * g(z, z)
        term1 = convolve_truncate(self._v_ds_coeffs, g_in_a, n_max)

        # Convolve: u^{de} v^{ds-de} * g(1-z, 1-z)
        term2 = convolve_truncate(self._u_de_v_diff_coeffs, g_1mz_in_a, n_max)

        # F = term1 - term2
        F_coeffs = term1 - term2

        # Extract odd coefficients for F-vector
        f_vec = np.array([
            F_coeffs[2*m + 1] if 2*m + 1 < len(F_coeffs) else 0.0
            for m in range(self.n_constraints)
        ])

        return f_vec

    def clear_cache(self):
        """Clear all cached F-vectors."""
        self._cross_ssss.clear_cache()
        self._cross_eeee.clear_cache()
        self._cache_ssee.clear()

    def verify_ssee_limit(self, delta: float, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """
        Verify that ssee reduces to ssss when Delta_epsilon = Delta_sigma.

        When the external dimensions are equal, the ssee crossing function
        should equal the ssss crossing function.

        Args:
            delta: Operator dimension to test
            tolerance: Relative tolerance for comparison

        Returns:
            Tuple of (passed, max_relative_error)
        """
        if abs(self.delta_epsilon - self.delta_sigma) > tolerance:
            # External dimensions are different, can't verify this limit
            return True, 0.0

        F_ssss = self.build_F_vector_ssss(delta)
        F_ssee = self.build_F_vector_ssee(delta)

        # Compute relative error
        max_abs = max(np.max(np.abs(F_ssss)), np.max(np.abs(F_ssee)), 1e-100)
        rel_error = np.max(np.abs(F_ssss - F_ssee)) / max_abs

        return rel_error < tolerance, rel_error


def test_mixed_crossing_vectors():
    """Test the mixed crossing vector implementation."""
    print("Testing Mixed Crossing Vectors")
    print("=" * 60)

    # Test 1: ssee should equal ssss when delta_epsilon = delta_sigma
    print("\n1. Testing ssee == ssss when Delta_eps = Delta_sigma:")
    ds = 0.518
    cross_equal = MixedCrossingVector(ds, ds, max_deriv=11)

    for delta in [0, 1.5, 2.0, 3.0]:
        passed, error = cross_equal.verify_ssee_limit(delta)
        status = "PASS" if passed else "FAIL"
        print(f"   Delta={delta}: {status} (rel_error={error:.2e})")

    # Test 2: Compare F-vector components at Ising point
    print("\n2. F-vector components at Ising point:")
    ds_ising = 0.518
    de_ising = 1.41
    cross = MixedCrossingVector(ds_ising, de_ising, max_deriv=11)

    print(f"   External dims: Delta_sigma={ds_ising}, Delta_epsilon={de_ising}")
    print(f"   Number of constraints: {cross.n_constraints}")

    # Identity
    F_id_ssss = cross.build_F_vector_ssss(0)
    F_id_ssee = cross.build_F_vector_ssee(0)
    F_id_eeee = cross.build_F_vector_eeee(0)

    print(f"\n   Identity F-vectors (first 3 components):")
    print(f"     ssss: {F_id_ssss[:3]}")
    print(f"     ssee: {F_id_ssee[:3]}")
    print(f"     eeee: {F_id_eeee[:3]}")

    # Operator at delta = 2.0
    delta_test = 2.0
    F_op_ssss = cross.build_F_vector_ssss(delta_test)
    F_op_ssee = cross.build_F_vector_ssee(delta_test)
    F_op_eeee = cross.build_F_vector_eeee(delta_test)

    print(f"\n   F-vectors for Delta={delta_test} (first 3 components):")
    print(f"     ssss: {F_op_ssss[:3]}")
    print(f"     ssee: {F_op_ssee[:3]}")
    print(f"     eeee: {F_op_eeee[:3]}")

    # Test 3: Verify asymptotic behavior
    print("\n3. Asymptotic behavior (large Delta):")
    for delta in [5.0, 10.0, 20.0]:
        F_ssss = cross.build_F_vector_ssss(delta)
        F_ssee = cross.build_F_vector_ssee(delta)
        F_eeee = cross.build_F_vector_eeee(delta)

        print(f"   Delta={delta}: |F_ssss[0]|={abs(F_ssss[0]):.2e}, "
              f"|F_ssee[0]|={abs(F_ssee[0]):.2e}, |F_eeee[0]|={abs(F_eeee[0]):.2e}")

    print("\n" + "=" * 60)
    print("Mixed crossing vector tests complete")


if __name__ == "__main__":
    test_mixed_crossing_vectors()
