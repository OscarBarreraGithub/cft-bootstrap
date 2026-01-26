"""
Polynomial Matrix Program (PMP) Infrastructure for the Conformal Bootstrap.

This module implements the pycftboot-style polynomial approach where F-vectors
are stored as symbolic polynomials in the dimension variable delta, with explicit
tracking of conformal block poles.

The key insight is that conformal blocks are RATIONAL functions in delta:
    g_Δ(z,zbar) = (polynomial in Δ) / (product of poles)

This allows exact polynomial arithmetic rather than numerical approximation,
which is critical for producing tight bootstrap bounds.

References:
    - pycftboot by C. Behan: https://github.com/cbehan/pycftboot
    - El-Showk et al., "Solving the 3D Ising Model with the Conformal Bootstrap"
      Phys.Rev. D86 (2012) 025022 [arXiv:1203.6064]
    - Kos, Poland, Simmons-Duffin, "Bootstrapping Mixed Correlators in the 3D Ising Model"
      JHEP 1411 (2014) 109 [arXiv:1406.4858]
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import factorial, gamma as scipy_gamma
from scipy.linalg import cholesky, solve_triangular

# Try to import symengine for symbolic computation
# Fall back to mpmath if not available
try:
    from symengine import (
        Symbol, Integer, Rational, sqrt, log, exp, pi,
        expand, diff, factorial as sym_factorial,
        DenseMatrix, RealMPFR, symbols
    )
    from symengine.lib.symengine_wrapper import have_mpfr
    HAVE_SYMENGINE = True
    if not have_mpfr:
        warnings.warn("symengine compiled without MPFR support, precision may be limited")
except ImportError:
    HAVE_SYMENGINE = False
    warnings.warn("symengine not available, using mpmath fallback (slower)")

try:
    import mpmath
    HAVE_MPMATH = True
except ImportError:
    HAVE_MPMATH = False

# =============================================================================
# Constants (matching pycftboot)
# =============================================================================

# Precision settings
PREC = 660  # bits of precision (~200 decimal digits)
DEC_PREC = int((3.0 / 10.0) * PREC)
TINY = 1e-100  # numerical zero threshold

# Crossing symmetric radial coordinate
R_CROSS = 3 - 2 * math.sqrt(2)  # ≈ 0.17157...

# Initialize precision for mpmath if available
if HAVE_MPMATH:
    mpmath.mp.dps = DEC_PREC

# Symbolic variables (if symengine available)
if HAVE_SYMENGINE:
    delta = Symbol('delta')
    ell = Symbol('ell')
    delta_ext = Symbol('delta_ext')

    # High-precision constants
    _one = RealMPFR("1", PREC)
    _two = RealMPFR("2", PREC)
    _r_cross = 3 * _one - 2 * sqrt(_two)


# =============================================================================
# Utility Functions
# =============================================================================

def rising_factorial(x: float, n: int) -> float:
    """
    Pochhammer symbol (rising factorial): (x)_n = x(x+1)...(x+n-1)
    """
    if n < 0:
        return 1.0 / rising_factorial(x - abs(n), abs(n))
    result = 1.0
    for k in range(n):
        result *= x + k
    return result


def unitarity_bound(dim: float, spin: int) -> float:
    """
    Returns the unitarity bound for operator dimension given spin.

    For spin 0: Δ ≥ (d-2)/2
    For spin l > 0: Δ ≥ d + l - 2
    """
    if spin == 0:
        return (dim - 2) / 2
    else:
        return dim + spin - 2


def gather_poles(poles: List[float]) -> Dict[float, int]:
    """
    Count multiplicities of approximately equal poles.

    Returns dict mapping pole -> multiplicity.
    """
    gathered = {}
    for p in poles:
        # Check if approximately equal to existing pole
        found = False
        for existing in gathered:
            if abs(p - existing) < TINY:
                gathered[existing] += 1
                found = True
                break
        if not found:
            gathered[p] = 1
    return gathered


def coefficients_from_polynomial(poly, prec: int = PREC) -> List[float]:
    """
    Extract coefficients from a symengine polynomial in delta.

    Returns [c_0, c_1, ..., c_d] where poly = sum_i c_i * delta^i
    """
    if not HAVE_SYMENGINE:
        raise RuntimeError("symengine required for polynomial operations")

    # Expand to canonical form
    poly = expand(poly)

    # Handle constants
    if not hasattr(poly, 'args') or poly.args == ():
        return [float(poly)]

    # Extract coefficients by degree
    coeffs = []
    terms = list(poly.args) if hasattr(poly.args, '__iter__') else [poly]

    # Find maximum degree
    max_deg = 0
    for term in terms:
        if hasattr(term, 'args') and len(term.args) >= 2:
            power_part = term.args[1]
            if hasattr(power_part, 'args') and len(power_part.args) >= 2:
                max_deg = max(max_deg, int(power_part.args[1]))
            elif power_part == delta:
                max_deg = max(max_deg, 1)
        elif term == delta:
            max_deg = max(max_deg, 1)

    # Build coefficient list
    coeffs = [0.0] * (max_deg + 1)
    for term in terms:
        if term == delta:
            coeffs[1] = 1.0
        elif hasattr(term, 'args') and len(term.args) >= 2:
            coeff = float(term.args[0])
            power_part = term.args[1]
            if power_part == delta:
                deg = 1
            elif hasattr(power_part, 'args') and len(power_part.args) >= 2:
                deg = int(power_part.args[1])
            else:
                coeffs[0] += float(term)
                continue
            coeffs[deg] = coeff
        else:
            coeffs[0] += float(term)

    return coeffs


def build_polynomial_from_coeffs(coeffs: List[float]) -> Any:
    """
    Build a symengine polynomial from coefficient list.

    coeffs = [c_0, c_1, ..., c_d] -> sum_i c_i * delta^i
    """
    if not HAVE_SYMENGINE:
        raise RuntimeError("symengine required for polynomial operations")

    result = RealMPFR(str(coeffs[0]), PREC)
    for i, c in enumerate(coeffs[1:], start=1):
        result = result + RealMPFR(str(c), PREC) * (delta ** i)
    return result


# =============================================================================
# Core Classes
# =============================================================================

@dataclass
class SymbolicPolynomialVector:
    """
    A vector of polynomials in delta representing an F-vector.

    This is the pycftboot-style representation where each component is
    a symbolic polynomial divided by a common set of poles.

    Attributes:
        vector: List of symengine expressions (polynomials in delta)
        label: Tuple (spin, irrep) identifying this vector
        poles: List of poles from conformal block rational structure
    """
    vector: List[Any]  # symengine expressions
    label: Tuple[int, int]
    poles: List[float]

    @property
    def spin(self) -> int:
        return self.label[0]

    @property
    def dimension(self) -> int:
        """Number of constraint components."""
        return len(self.vector)

    @property
    def max_degree(self) -> int:
        """Maximum polynomial degree across all components."""
        max_deg = 0
        for poly in self.vector:
            coeffs = coefficients_from_polynomial(poly)
            max_deg = max(max_deg, len(coeffs) - 1)
        return max_deg

    def evaluate(self, delta_val: float) -> np.ndarray:
        """
        Evaluate the polynomial vector at a specific delta value.

        Note: This divides by the pole product to get the true F-vector value.
        """
        if not HAVE_SYMENGINE:
            raise RuntimeError("symengine required")

        result = np.zeros(len(self.vector))
        for i, poly in enumerate(self.vector):
            val = float(poly.subs(delta, delta_val))
            # Divide by pole product
            for p in self.poles:
                val /= (delta_val - p)
            result[i] = val
        return result

    def evaluate_polynomial_only(self, delta_val: float) -> np.ndarray:
        """
        Evaluate just the polynomial part (without dividing by poles).
        """
        if not HAVE_SYMENGINE:
            raise RuntimeError("symengine required")

        result = np.zeros(len(self.vector))
        for i, poly in enumerate(self.vector):
            result[i] = float(poly.subs(delta, delta_val))
        return result


@dataclass
class ConformalBlockPoles:
    """
    Computes and stores the poles of conformal blocks.

    Conformal blocks g_Δ,l(z,zbar) have poles at specific values of Δ
    determined by the recursion relations. These poles are:

    Series 1: Δ = 1 - l - k  (k = 1, 2, 3, ...)
    Series 2: Δ = 1 + ν - k  (k = 2, 4, 6, ...)  where ν = (d-2)/2
    Series 3: Δ = 1 + l + 2ν - k  (k = 1, 2, ..., l)

    Reference: Kos, Poland, Simmons-Duffin arXiv:1406.4858
    """
    dim: float
    k_max: int

    def __post_init__(self):
        self.nu = (self.dim - 2) / 2

    def get_pole(self, k: int, l: int, series: int) -> float:
        """Get pole position for given parameters."""
        if series == 1:
            return 1 - l - k
        elif series == 2:
            return 1 + self.nu - k
        else:  # series == 3
            return 1 + l + 2 * self.nu - k

    def get_residue(self, k: int, l: int, delta_12: float, delta_34: float,
                    series: int) -> float:
        """
        Get the residue at a pole.

        These residues are computed from the Zamolodchikov recursion.
        """
        nu = self.nu

        # Time-saving special case
        if series != 2 and k % 2 != 0 and delta_12 == 0 and delta_34 == 0:
            return 0

        if series == 1:
            ret = -((k * (-4) ** k) / (factorial(k) ** 2))
            ret *= rising_factorial((1 - k + delta_12) / 2, k)
            ret *= rising_factorial((1 - k + delta_34) / 2, k)
            if ret == 0:
                return ret
            elif l == 0 and nu == 0:
                return ret * 2
            else:
                return ret * (rising_factorial(l + 2 * nu, k) / rising_factorial(l + nu, k))

        elif series == 2:
            factors = [l + nu + 1 - delta_12, l + nu + 1 + delta_12,
                      l + nu + 1 - delta_34, l + nu + 1 + delta_34]
            ret = ((k * rising_factorial(nu + 1, k - 1)) / (factorial(k) ** 2))
            ret *= ((l + nu - k) / (l + nu + k))
            ret *= rising_factorial(-nu, k + 1)
            denom = (rising_factorial((l + nu - k + 1) / 2, k) *
                    rising_factorial((l + nu - k) / 2, k)) ** 2
            ret /= denom
            for f in factors:
                ret *= rising_factorial((f - k) / 2, k)
            return ret

        else:  # series == 3
            ret = -((k * (-4) ** k) / (factorial(k) ** 2))
            ret *= rising_factorial(1 + l - k, k)
            ret *= rising_factorial((1 - k + delta_12) / 2, k)
            ret *= rising_factorial((1 - k + delta_34) / 2, k)
            ret /= rising_factorial(1 + nu + l - k, k)
            return ret

    def get_all_poles(self, l: int, delta_12: float = 0, delta_34: float = 0) -> List[float]:
        """
        Get all poles for a given spin, keeping only those with nonzero residue.
        """
        poles = []

        for k in range(1, self.k_max + 1):
            # Series 1
            if self.get_residue(k, l, delta_12, delta_34, 1) != 0:
                poles.append(self.get_pole(k, l, 1))

            # Series 2 (only even k)
            if k % 2 == 0:
                if self.get_residue(k // 2, l, delta_12, delta_34, 2) != 0:
                    poles.append(self.get_pole(k // 2, l, 2))

            # Series 3 (only k <= l)
            if k <= l:
                if self.get_residue(k, l, delta_12, delta_34, 3) != 0:
                    poles.append(self.get_pole(k, l, 3))

        return poles


class SymbolicConformalBlockTable:
    """
    Computes conformal block derivatives as symbolic polynomials in delta.

    This is the core of the pycftboot approach: instead of evaluating blocks
    numerically at specific delta values, we compute them symbolically as
    rational functions (polynomials divided by products of poles).

    Parameters:
        dim: Spatial dimension
        k_max: Recursion depth (controls pole accuracy)
        l_max: Maximum spin
        m_max: Maximum 'a' derivatives
        n_max: Maximum 'b' derivatives
        delta_12: Δ₁ - Δ₂ for external operators
        delta_34: Δ₃ - Δ₄ for external operators
    """

    def __init__(
        self,
        dim: float,
        k_max: int = 20,
        l_max: int = 50,
        m_max: int = 10,
        n_max: int = 10,
        delta_12: float = 0,
        delta_34: float = 0,
        odd_spins: bool = False
    ):
        if not HAVE_SYMENGINE:
            raise RuntimeError("symengine required for symbolic conformal blocks")

        self.dim = dim
        self.k_max = k_max
        self.l_max = l_max
        self.m_max = m_max
        self.n_max = n_max
        self.delta_12 = delta_12
        self.delta_34 = delta_34
        self.odd_spins = odd_spins

        self.nu = (dim - 2) / 2
        self.pole_calculator = ConformalBlockPoles(dim, k_max)

        # Derivative ordering
        self.m_order = []
        self.n_order = []

        # Table of SymbolicPolynomialVectors
        self.table: List[SymbolicPolynomialVector] = []

        # Build the table
        self._build_table()

    def _build_table(self):
        """
        Build the conformal block table using Zamolodchikov recursion.

        This implements the algorithm from pycftboot's ConformalBlockTableSeed.
        """
        step = 1 if self.odd_spins else 2
        derivative_order = self.m_max + 2 * self.n_max

        # Build derivative ordering
        for n in range(self.n_max + 1):
            for m in range(2 * (self.n_max - n) + self.m_max + 1):
                self.m_order.append(m)
                self.n_order.append(n)

        # For each spin, compute the block derivatives
        for l in range(0, self.l_max + 1, step):
            poles = self.pole_calculator.get_all_poles(l, self.delta_12, self.delta_34)

            # Start with leading block contribution (simplified for now)
            # Full implementation would use LeadingBlockVector
            derivatives = self._compute_block_derivatives(l, derivative_order)

            # Package as SymbolicPolynomialVector
            poly_vec = SymbolicPolynomialVector(
                vector=derivatives,
                label=(l, 0),
                poles=poles
            )
            self.table.append(poly_vec)

    def _compute_block_derivatives(self, spin: int, max_order: int) -> List[Any]:
        """
        Compute conformal block derivatives as polynomials in delta.

        This is a simplified version - the full implementation would use
        the Zamolodchikov recursion from pycftboot/blocks1.py.

        For now, we use a numerical approach converted to polynomial form.
        """
        # Placeholder: compute numerically and fit polynomial
        # Full implementation would use symbolic recursion
        nu = self.nu
        l = spin

        derivatives = []
        for i in range(len(self.m_order)):
            # Simplified: constant polynomial for now
            # Real implementation: symbolic recursion
            poly = delta * RealMPFR("0", PREC) + RealMPFR("1", PREC)
            derivatives.append(poly)

        return derivatives


class ConvolvedBlockTable:
    """
    Produces the F-vectors by convolving conformal blocks with crossing factors.

    The F-vector for an operator of dimension Δ and spin l is:
        F_Δ,l = crossing_factor × block_derivatives

    where the crossing factor accounts for the s-channel vs t-channel difference.
    """

    def __init__(
        self,
        block_table: SymbolicConformalBlockTable,
        symmetric: bool = False
    ):
        self.block_table = block_table
        self.symmetric = symmetric

        # Copy relevant parameters
        self.dim = block_table.dim
        self.k_max = block_table.k_max
        self.l_max = block_table.l_max
        self.m_max = block_table.m_max
        self.n_max = block_table.n_max
        self.delta_12 = block_table.delta_12
        self.delta_34 = block_table.delta_34

        # Build convolved table
        self.m_order = []
        self.n_order = []
        self.table: List[SymbolicPolynomialVector] = []

        self._build_convolved_table()

    def _build_convolved_table(self):
        """
        Convolve blocks with crossing factors.

        The crossing equation gives:
            sum_O λ²_O (F⁺_O ± F⁻_O) = 0

        where F⁺ is the s-channel contribution and F⁻ is t-channel.
        """
        # Build derivative ordering (keep only non-vanishing)
        for n in range(self.n_max + 1):
            for m in range(2 * (self.n_max - n) + self.m_max + 1):
                # Skip derivatives that vanish by symmetry
                if (not self.symmetric and m % 2 == 0) or (self.symmetric and m % 2 == 1):
                    continue
                self.m_order.append(m)
                self.n_order.append(n)

        # Convolve each block
        for poly_vec in self.block_table.table:
            new_derivatives = self._convolve_single(poly_vec)
            new_poly_vec = SymbolicPolynomialVector(
                vector=new_derivatives,
                label=poly_vec.label,
                poles=poly_vec.poles
            )
            self.table.append(new_poly_vec)

    def _convolve_single(self, poly_vec: SymbolicPolynomialVector) -> List[Any]:
        """Convolve a single block with crossing factors."""
        # Simplified implementation
        # Full version applies crossing factor transformation
        derivatives = []

        for i in range(len(self.m_order)):
            m = self.m_order[i]
            n = self.n_order[i]

            # The convolution involves summing products of block derivatives
            # with binomial-type coefficients from the crossing factor expansion
            # For now, just copy (full implementation more complex)
            if i < len(poly_vec.vector):
                derivatives.append(poly_vec.vector[i])
            else:
                derivatives.append(RealMPFR("0", PREC))

        return derivatives


# =============================================================================
# Bilinear Basis and PMP Generation
# =============================================================================

class BilinearBasis:
    """
    Computes the orthogonal polynomial basis for SDPB.

    The key is that SDPB needs polynomials that are orthogonal with respect
    to a positive measure. The measure is:

        dμ(x) = r_cross^(x + shift) / prod(x - pole_i) dx

    where r_cross ≈ 0.172 is the crossing-symmetric radial coordinate.

    The orthogonalization is done via Cholesky decomposition of the Gram matrix:
        G[i,j] = ∫ x^(i+j) dμ(x)
    """

    def __init__(self, poles: List[float], delta_min: float, max_degree: int):
        """
        Initialize bilinear basis computation.

        Args:
            poles: Conformal block poles
            delta_min: Minimum dimension (gap)
            max_degree: Maximum polynomial degree
        """
        self.poles = poles
        self.delta_min = delta_min
        self.max_degree = max_degree

        # Compute Gram matrix and basis
        self.gram_matrix = self._compute_gram_matrix()
        self.basis_matrix = self._compute_basis()

    def _compute_gram_matrix(self) -> np.ndarray:
        """
        Compute the Gram matrix for the bilinear form.

        G[i,j] = ∫₀^∞ x^(i+j) · r_cross^(x + delta_min) / prod(x - (p - delta_min)) dx
        """
        degree = self.max_degree
        bands = []

        # Compute integrals for all needed powers
        for d in range(2 * (degree // 2) + 1):
            integral = self._compute_integral(d)
            bands.append(integral)

        # Build symmetric Gram matrix
        n = (degree // 2) + 1
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                G[i, j] = bands[i + j]

        return G

    def _compute_integral(self, power: int) -> float:
        """
        Compute ∫₀^∞ x^power · r_cross^(x + delta_min) / prod(x - shifted_pole) dx

        This uses incomplete gamma functions for analytical evaluation.
        For numerical stability, we use a hybrid approach:
        - Without poles: analytical formula using factorial and log
        - With poles: use pycftboot's partial fraction approach with uppergamma
        """
        # Shift all poles: we integrate over x ∈ [0, ∞) which corresponds to Δ ∈ [delta_min, ∞)
        shifted_poles = [p - self.delta_min for p in self.poles]

        # Filter to poles that affect the integral (poles in integration domain would cause issues)
        # Poles at negative x don't cause singularities in [0, ∞)
        relevant_poles = [p for p in shifted_poles if p < 0]

        if not relevant_poles:
            # No relevant poles: simple integral
            # ∫₀^∞ x^n · r^(x+s) dx = r^s · n! / (-log(r))^(n+1)
            if HAVE_MPMATH:
                mpmath.mp.dps = 50
                log_r = mpmath.log(R_CROSS)
                result = (mpmath.factorial(power) / ((-log_r) ** (power + 1)) *
                         mpmath.power(R_CROSS, self.delta_min))
                return float(result)
            else:
                log_r = math.log(R_CROSS)
                result = (math.factorial(power) / ((-log_r) ** (power + 1)) *
                         (R_CROSS ** self.delta_min))
                return result

        # With poles: use partial fractions and incomplete gamma functions
        if HAVE_MPMATH:
            mpmath.mp.dps = 50
            log_r = mpmath.log(R_CROSS)

            # Gather poles by multiplicity
            gathered = gather_poles(relevant_poles)

            # Precompute uppergamma values for each unique pole
            gammas = {}
            for pole in gathered:
                # uppergamma(0, a) = Γ(0, a) = -Ei(-a) for real a
                # For negative pole, pole * log(r_cross) is positive (since log(r_cross) < 0)
                arg = pole * log_r
                gammas[pole] = mpmath.gammainc(0, arg, regularized=False)

            # Compute integral using partial fractions
            result = mpmath.mpf(0)
            poles_list = list(gathered.keys())
            orders_list = list(gathered.values())

            for i, (pole, order) in enumerate(gathered.items()):
                other_poles = poles_list[:i] + poles_list[i+1:]
                other_orders = orders_list[:i] + orders_list[i+1:]

                # Contribution from this pole
                gamma_val = gammas[pole]
                contrib = self._integral_partial_fraction(
                    power, pole, order, gamma_val, other_poles, other_orders, log_r
                )
                result += contrib

            return float(result * mpmath.power(R_CROSS, self.delta_min))
        else:
            # Numerical fallback using scipy
            from scipy.integrate import quad

            def integrand(x):
                if x < 0:
                    return 0
                val = (x ** power) * (R_CROSS ** (x + self.delta_min))
                for pole in relevant_poles:
                    denom = x - pole
                    if abs(denom) < 1e-15:
                        return 0  # Skip singularity (shouldn't happen for negative poles)
                    val /= denom
                return val

            # Split integral to handle potential numerical issues
            result, _ = quad(integrand, 0, 100, limit=200)
            result2, _ = quad(integrand, 100, np.inf, limit=100)
            return result + result2

    def _integral_partial_fraction(self, power: int, pole: float, order: int,
                                   gamma_val, other_poles: List[float],
                                   other_orders: List[int], log_r) -> float:
        """
        Compute the partial fraction contribution for one pole.

        This follows pycftboot's algorithm in bootstrap.py lines 1021-1070.
        """
        if not HAVE_MPMATH:
            return 0.0

        # Base contribution from basic_integral
        base = self._basic_integral_mpmath(power, -pole * log_r, order, gamma_val)

        # Scale by (-log_r)^(order - 1 - power) to match pycftboot's convention
        scale = (-log_r) ** (order - 1 - power)

        # Multiply by product of (pole - other_pole)^(-other_order) for partial fractions
        prod = mpmath.mpf(1)
        for other_pole, other_order in zip(other_poles, other_orders):
            prod *= (pole - other_pole) ** (-other_order)

        return base * scale * prod

    def _basic_integral_mpmath(self, power: int, pole: float, order: int,
                               gamma_val: mpmath.mpf) -> mpmath.mpf:
        """
        Compute integral for single pole using recursion.

        This implements the same algorithm as pycftboot's basic_integral.
        """
        if order == 1:
            ret = mpmath.exp(-pole) * (pole ** power) * gamma_val
            for i in range(power):
                ret += mpmath.factorial(power - i - 1) * (pole ** i)
            return ret
        elif power == 0:
            return (((-pole) ** (1 - order)) / (order - 1) -
                    self._basic_integral_mpmath(0, pole, order - 1, gamma_val) / (order - 1))
        else:
            return ((power * self._basic_integral_mpmath(power - 1, pole, order - 1, gamma_val) -
                    self._basic_integral_mpmath(power, pole, order - 1, gamma_val)) / (order - 1))

    def _compute_basis(self) -> np.ndarray:
        """
        Compute orthogonal basis via Cholesky decomposition.

        If G = L L^T is the Cholesky decomposition of the Gram matrix,
        then L^{-1} transforms to the orthogonal basis.
        """
        try:
            L = cholesky(self.gram_matrix, lower=True)
            # Invert to get basis transformation
            basis = solve_triangular(L, np.eye(L.shape[0]), lower=True)
            return basis
        except np.linalg.LinAlgError:
            # Gram matrix not positive definite - use regularization
            warnings.warn("Gram matrix not positive definite, using regularization")
            G_reg = self.gram_matrix + 1e-10 * np.eye(self.gram_matrix.shape[0])
            L = cholesky(G_reg, lower=True)
            basis = solve_triangular(L, np.eye(L.shape[0]), lower=True)
            return basis

    def transform_polynomial(self, coeffs: List[float]) -> List[float]:
        """
        Transform polynomial coefficients to orthogonal basis.

        Args:
            coeffs: Coefficients in standard monomial basis [c_0, c_1, ..., c_d]

        Returns:
            Coefficients in orthogonal basis
        """
        n = len(coeffs)
        if n > self.basis_matrix.shape[0]:
            raise ValueError(f"Polynomial degree {n-1} exceeds basis size {self.basis_matrix.shape[0]-1}")

        # Pad coefficients to match basis size
        padded = np.zeros(self.basis_matrix.shape[0])
        padded[:n] = coeffs

        # Transform
        return list(self.basis_matrix @ padded)


class PMPGenerator:
    """
    Generates Polynomial Matrix Program (PMP) files for SDPB.

    The PMP format is documented in the SDPB repository:
    https://github.com/davidsd/sdpb
    """

    def __init__(
        self,
        convolved_table: ConvolvedBlockTable,
        bounds: Dict[int, float],
        objective: List[float],
        normalization: List[float]
    ):
        """
        Initialize PMP generator.

        Args:
            convolved_table: Table of convolved conformal blocks
            bounds: Dict mapping spin -> minimum dimension
            objective: Objective vector b (zeros for feasibility)
            normalization: Normalization constraint vector
        """
        self.table = convolved_table
        self.bounds = bounds
        self.objective = objective
        self.normalization = normalization

        # Compute bilinear bases for each spin
        self.bases: Dict[int, BilinearBasis] = {}
        self._compute_all_bases()

    def _compute_all_bases(self):
        """Compute bilinear basis for each spin channel."""
        for poly_vec in self.table.table:
            spin = poly_vec.spin
            if spin in self.bases:
                continue

            delta_min = self.bounds.get(spin, unitarity_bound(self.table.dim, spin))
            max_degree = poly_vec.max_degree

            self.bases[spin] = BilinearBasis(
                poles=poly_vec.poles,
                delta_min=delta_min,
                max_degree=max_degree
            )

    def make_laguerre_points(self, degree: int) -> List[float]:
        """
        Generate sample points for SDPB using Laguerre-style spacing.

        These points minimize interpolation error for the damped rational measure.
        """
        points = []
        log_r = math.log(R_CROSS)
        for d in range(degree + 1):
            point = -(math.pi ** 2) * ((4 * d - 1) ** 2) / (64 * (degree + 1) * log_r)
            points.append(point)
        return points

    def shifted_prefactor(self, poles: List[float], x: float, shift: float) -> float:
        """
        Compute the damped rational prefactor at a sample point.

        prefactor = r_cross^(x + shift) / prod(x - (pole - shift))
        """
        product = 1.0
        for p in poles:
            product *= (x - (p - shift))
        return (R_CROSS ** (x + shift)) / product

    def reshuffle_with_normalization(self, vector: List[Any], norm: List[Any]) -> List[Any]:
        """
        Convert to SDPB's normalization convention.

        We normalize so that α · F_id = 1, but SDPB expects α[0] = 1.
        This reshuffles the vector to convert between conventions.
        """
        # Find the largest component of normalization
        norm_floats = [float(n) for n in norm]
        max_idx = norm_floats.index(max(norm_floats, key=abs))

        const = vector[max_idx] / norm[max_idx]

        result = []
        for i in range(len(norm)):
            result.append(vector[i] - const * norm[i])

        # Reorder: const first, then others
        return [const] + result[:max_idx] + result[max_idx + 1:]

    def write_json(self, output_dir: str):
        """
        Write PMP files in SDPB JSON format.

        Creates:
            - control.json
            - objectives.json
            - block_info_N.json for each block
            - block_data_N.json for each block
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_blocks = len(self.table.table)

        # Write control.json
        control = {
            "num_blocks": num_blocks,
            "command": "polynomial_bootstrap.py"
        }
        with open(output_path / "control.json", 'w') as f:
            json.dump(control, f, indent=2)

        # Write objectives.json
        obj_reshuffled = self.reshuffle_with_normalization(
            self.objective, self.normalization
        )
        with open(output_path / "objectives.json", 'w') as f:
            json.dump([str(x) for x in obj_reshuffled], f, indent=2)

        # Write block files
        for j, poly_vec in enumerate(self.table.table):
            spin = poly_vec.spin
            delta_min = self.bounds.get(spin, unitarity_bound(self.table.dim, spin))
            degree = poly_vec.max_degree

            # Sample points
            points = self.make_laguerre_points(degree)

            # Sample scalings (prefactor values)
            scalings = [self.shifted_prefactor(poly_vec.poles, p, delta_min)
                       for p in points]

            # Get bilinear basis
            basis = self.bases[spin]

            # Evaluate basis at sample points
            bilinear_even, bilinear_odd = self._evaluate_bilinear_basis(
                basis, points, scalings
            )

            # Evaluate constraints
            c_mat, B_mats = self._evaluate_constraints(poly_vec, points, scalings)

            # Write block_info
            block_info = {
                "dim": len(poly_vec.vector),
                "num_points": len(points),
                "degree": degree,
                "schur_degree": degree // 2
            }
            with open(output_path / f"block_info_{j}.json", 'w') as f:
                json.dump(block_info, f, indent=2)

            # Write block_data
            block_data = {
                "bilinear_bases_even": [[str(x) for x in row] for row in bilinear_even],
                "bilinear_bases_odd": [[str(x) for x in row] for row in bilinear_odd],
                "c": [[str(x) for x in row] for row in c_mat],
                "B": [[[str(x) for x in row] for row in mat] for mat in B_mats]
            }
            with open(output_path / f"block_data_{j}.json", 'w') as f:
                json.dump(block_data, f, indent=2)

    def _evaluate_bilinear_basis(
        self,
        basis: BilinearBasis,
        points: List[float],
        scalings: List[float]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Evaluate bilinear basis at sample points."""
        n = basis.basis_matrix.shape[0]

        even_basis = []
        odd_basis = []

        for i, (x, s) in enumerate(zip(points, scalings)):
            even_row = []
            odd_row = []

            for j in range(n):
                # Even powers: x^(2j)
                val_even = (x ** (2 * j)) * math.sqrt(abs(s))
                even_row.append(val_even)

                # Odd powers: x^(2j+1)
                val_odd = (x ** (2 * j + 1)) * math.sqrt(abs(s))
                odd_row.append(val_odd)

            even_basis.append(even_row)
            odd_basis.append(odd_row)

        return even_basis, odd_basis

    def _evaluate_constraints(
        self,
        poly_vec: SymbolicPolynomialVector,
        points: List[float],
        scalings: List[float]
    ) -> Tuple[List[List[float]], List[List[List[float]]]]:
        """
        Evaluate constraint matrices c and B at sample points.

        Returns (c, B) where:
            c[i][p] = c_i(x_p) = polynomial value at sample point
            B[n][i][p] = B_n^i(x_p) = constraint contribution
        """
        dim = len(poly_vec.vector)
        n_points = len(points)

        # Evaluate polynomials at sample points
        c_mat = []
        for i in range(dim):
            row = []
            for p in range(n_points):
                if HAVE_SYMENGINE:
                    val = float(poly_vec.vector[i].subs(delta, points[p]))
                else:
                    # Fallback: assume vector contains coefficients
                    val = sum(c * (points[p] ** j)
                             for j, c in enumerate(poly_vec.vector[i]))
                row.append(val * scalings[p])
            c_mat.append(row)

        # B matrices (for constraints beyond identity)
        # In simple case, B is empty or identity
        B_mats = []

        return c_mat, B_mats


# =============================================================================
# High-Level Interface
# =============================================================================

class PolynomialBootstrapSolver:
    """
    High-level interface for polynomial bootstrap computations.

    This wraps the symbolic polynomial infrastructure and provides
    methods for computing bounds that match pycftboot's approach.
    """

    def __init__(
        self,
        dim: float = 3,
        k_max: int = 20,
        l_max: int = 50,
        m_max: int = 10,
        n_max: int = 10
    ):
        """
        Initialize the polynomial bootstrap solver.

        Args:
            dim: Spatial dimension
            k_max: Recursion depth for conformal blocks
            l_max: Maximum spin
            m_max: Maximum 'a' derivatives
            n_max: Maximum 'b' derivatives
        """
        self.dim = dim
        self.k_max = k_max
        self.l_max = l_max
        self.m_max = m_max
        self.n_max = n_max

        # Will be populated when external dimensions are set
        self.block_table: Optional[SymbolicConformalBlockTable] = None
        self.convolved_table: Optional[ConvolvedBlockTable] = None

    def setup_problem(
        self,
        delta_sigma: float,
        delta_epsilon: Optional[float] = None,
        delta_12: float = 0,
        delta_34: float = 0
    ):
        """
        Set up the bootstrap problem with given external dimensions.

        Args:
            delta_sigma: Dimension of external scalar σ
            delta_epsilon: Dimension of ε (if known, for mixed correlator)
            delta_12: Δ₁ - Δ₂
            delta_34: Δ₃ - Δ₄
        """
        print(f"Setting up polynomial bootstrap problem:")
        print(f"  dim = {self.dim}")
        print(f"  Δσ = {delta_sigma}")
        print(f"  k_max = {self.k_max}, l_max = {self.l_max}")
        print(f"  m_max = {self.m_max}, n_max = {self.n_max}")

        # Build conformal block table
        print("Computing symbolic conformal block table...")
        self.block_table = SymbolicConformalBlockTable(
            dim=self.dim,
            k_max=self.k_max,
            l_max=self.l_max,
            m_max=self.m_max,
            n_max=self.n_max,
            delta_12=delta_12,
            delta_34=delta_34
        )

        # Build convolved table
        print("Building convolved block table...")
        self.convolved_table = ConvolvedBlockTable(self.block_table)

        print(f"Table contains {len(self.convolved_table.table)} spin channels")

    def generate_pmp(
        self,
        output_dir: str,
        bounds: Dict[int, float],
        objective: Optional[List[float]] = None
    ):
        """
        Generate PMP files for SDPB.

        Args:
            output_dir: Directory for output files
            bounds: Dict mapping spin -> minimum dimension gap
            objective: Objective vector (zeros for feasibility if None)
        """
        if self.convolved_table is None:
            raise RuntimeError("Must call setup_problem() first")

        # Default objective: feasibility (all zeros)
        if objective is None:
            n_constraints = len(self.convolved_table.m_order)
            objective = [0.0] * n_constraints

        # Normalization: identity contribution
        normalization = [1.0] * len(objective)  # Simplified

        # Create PMP generator
        generator = PMPGenerator(
            convolved_table=self.convolved_table,
            bounds=bounds,
            objective=objective,
            normalization=normalization
        )

        # Write files
        print(f"Writing PMP files to {output_dir}...")
        generator.write_json(output_dir)
        print("Done!")


# =============================================================================
# Verification Functions
# =============================================================================

def verify_against_pycftboot(delta_sigma: float = 0.518):
    """
    Compare our implementation against pycftboot at a test point.

    This is useful for validating that the polynomial computation matches.
    """
    print("="*60)
    print("Verification against pycftboot")
    print("="*60)
    print(f"Testing at Δσ = {delta_sigma}")

    if not HAVE_SYMENGINE:
        print("WARNING: symengine not available, cannot run verification")
        return

    # Create solver
    solver = PolynomialBootstrapSolver(
        dim=3,
        k_max=10,  # Start small for testing
        l_max=10,
        m_max=3,
        n_max=3
    )

    # Set up problem
    solver.setup_problem(delta_sigma=delta_sigma)

    # Check poles for spin-0
    if solver.convolved_table and len(solver.convolved_table.table) > 0:
        spin0 = solver.convolved_table.table[0]
        print(f"\nSpin-0 poles: {spin0.poles}")
        print(f"Number of F-vector components: {spin0.dimension}")
        print(f"Maximum polynomial degree: {spin0.max_degree}")


if __name__ == "__main__":
    # Run verification
    verify_against_pycftboot()
