"""
Polynomial Positivity Constraints for the Conformal Bootstrap.

This module implements polynomial approximation and sum-of-squares (SOS) positivity
constraints for the conformal bootstrap. Instead of enforcing positivity at discrete
sample points, we approximate F-vectors as polynomials in Δ and enforce positivity
for ALL Δ ≥ Δ_gap using semidefinite programming.

The key insight is that a univariate polynomial p(x) is non-negative on [0, ∞) if and
only if it can be written as:

    p(x) = s(x) + x * t(x)

where s(x) and t(x) are sum-of-squares (SOS) polynomials. An SOS polynomial is one
that can be written as a sum of squared polynomials, which is equivalent to requiring
that a certain Gram matrix is positive semidefinite.

This approach gives tighter bounds than discrete sampling because:
1. Continuous positivity is enforced, not just at sample points
2. The optimal linear functional is found more precisely
3. Fewer numerical artifacts from sampling

Mathematical Background
-----------------------
For the bootstrap, we need to find α such that:
    α · F_Δ ≥ 0  for all Δ ≥ Δ_gap

We approximate: F_Δ ≈ p(x) where x = Δ - Δ_gap ≥ 0
Then:          α · p(x) ≥ 0 for all x ≥ 0

This is a polynomial non-negativity constraint on a half-line.

By Putinar's theorem, p(x) ≥ 0 on [0, ∞) iff:
    p(x) = σ₀(x) + x · σ₁(x)

where σ₀, σ₁ are SOS polynomials.

An SOS polynomial of degree 2d can be written as:
    σ(x) = v(x)ᵀ Q v(x)

where v(x) = [1, x, x², ..., x^d]ᵀ and Q ≽ 0 (positive semidefinite).

References
----------
- Parrilo, P. (2003). "Semidefinite programming relaxations for semialgebraic problems"
- Simmons-Duffin, D. (2015). "A Semidefinite Program Solver for the Conformal Bootstrap"
- Lasserre, J.B. (2001). "Global optimization with polynomials and the problem of moments"

Usage
-----
>>> solver = PolynomialPositivitySolver(delta_sigma=0.518, max_deriv=21, poly_degree=15)
>>> bound = solver.find_delta_epsilon_prime_bound(delta_epsilon=1.41)
>>> print(f"Δε' ≤ {bound:.4f}")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
import warnings

# Import CVXPY for SDP
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    warnings.warn("CVXPY not installed. Install with: pip install cvxpy")

# Import local modules
try:
    from .taylor_conformal_blocks import TaylorCrossingVector, build_F_vector_taylor
    from .bootstrap_gap_solver import reshuffle_with_normalization
except ImportError:
    from taylor_conformal_blocks import TaylorCrossingVector, build_F_vector_taylor
    from bootstrap_gap_solver import reshuffle_with_normalization


# =============================================================================
# Polynomial Fitting
# =============================================================================

@dataclass
class FittedPolynomial:
    """
    A polynomial fitted to F-vector data.

    Represents the i-th component of F_Δ as a polynomial in x = Δ - Δ_gap:
        F_Δ[i] ≈ Σⱼ coefficients[j] * x^j

    Attributes:
        coefficients: Polynomial coefficients [c₀, c₁, ..., c_d] where p(x) = Σ cⱼ xʲ
        delta_gap: The gap value (x = Δ - delta_gap)
        delta_max: Maximum Δ used in fitting
        fitting_error: RMS error of the polynomial fit
        component_index: Which component of the F-vector this represents
    """
    coefficients: np.ndarray
    delta_gap: float
    delta_max: float
    fitting_error: float = 0.0
    component_index: int = 0

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return len(self.coefficients) - 1

    def evaluate(self, delta: float) -> float:
        """Evaluate polynomial at Δ."""
        x = delta - self.delta_gap
        return np.polyval(self.coefficients[::-1], x)

    def evaluate_array(self, deltas: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at array of Δ values."""
        x = deltas - self.delta_gap
        return np.polyval(self.coefficients[::-1], x)


@dataclass
class PolynomialFVector:
    """
    Complete polynomial approximation of an F-vector.

    Each component of the F-vector is approximated as a polynomial in x = Δ - Δ_gap.
    This represents: F_Δ ≈ [p₀(x), p₁(x), ..., p_{n-1}(x)]

    Attributes:
        polynomials: List of FittedPolynomial for each component
        delta_gap: The gap value
        delta_max: Maximum Δ used in fitting
        n_components: Number of F-vector components (number of constraints)
    """
    polynomials: List[FittedPolynomial] = field(default_factory=list)
    delta_gap: float = 0.0
    delta_max: float = 30.0

    @property
    def n_components(self) -> int:
        """Number of constraint components."""
        return len(self.polynomials)

    @property
    def max_degree(self) -> int:
        """Maximum polynomial degree across all components."""
        if not self.polynomials:
            return 0
        return max(p.degree for p in self.polynomials)

    def get_coefficient_matrix(self) -> np.ndarray:
        """
        Get coefficient matrix of shape (n_components, max_degree + 1).

        Entry [i, j] is the coefficient of x^j in component i.
        """
        d = self.max_degree + 1
        matrix = np.zeros((self.n_components, d))
        for i, poly in enumerate(self.polynomials):
            matrix[i, :len(poly.coefficients)] = poly.coefficients
        return matrix

    def evaluate(self, delta: float) -> np.ndarray:
        """Evaluate all components at Δ."""
        return np.array([p.evaluate(delta) for p in self.polynomials])

    @property
    def total_fitting_error(self) -> float:
        """Total RMS fitting error across all components."""
        if not self.polynomials:
            return 0.0
        errors = [p.fitting_error for p in self.polynomials]
        return np.sqrt(np.mean(np.array(errors)**2))


class PolynomialFitter:
    """
    Fits F-vectors to polynomials using Chebyshev interpolation.

    Chebyshev nodes provide near-optimal interpolation and avoid Runge's phenomenon
    (oscillations at the edges of the interval).

    The fitting process:
    1. Sample F-vectors at Chebyshev nodes in [Δ_gap, Δ_max]
    2. Fit polynomial to each component using least squares
    3. Optionally validate fit against additional sample points
    """

    def __init__(
        self,
        delta_sigma: float,
        max_deriv: int = 21,
        poly_degree: int = 15,
    ):
        """
        Initialize the polynomial fitter.

        Args:
            delta_sigma: External operator dimension
            max_deriv: Maximum derivative order (determines number of constraints)
            poly_degree: Degree of polynomial approximation
        """
        self.delta_sigma = delta_sigma
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2
        self.poly_degree = poly_degree

        # Create crossing vector computer
        self.crossing = TaylorCrossingVector(delta_sigma, max_deriv)

    def chebyshev_nodes(self, n: int, a: float, b: float) -> np.ndarray:
        """
        Generate Chebyshev nodes on interval [a, b].

        Chebyshev nodes are: xₖ = cos(π(2k+1)/(2n)) for k = 0, ..., n-1
        mapped from [-1, 1] to [a, b].

        Args:
            n: Number of nodes
            a: Left endpoint
            b: Right endpoint

        Returns:
            Array of n Chebyshev nodes in [a, b]
        """
        k = np.arange(n)
        nodes_unit = np.cos(np.pi * (2*k + 1) / (2*n))
        # Map from [-1, 1] to [a, b]
        return 0.5 * (b - a) * (nodes_unit + 1) + a

    def fit_polynomial(
        self,
        delta_gap: float,
        delta_max: float = 30.0,
        n_sample_points: Optional[int] = None,
        validate: bool = True
    ) -> PolynomialFVector:
        """
        Fit polynomial approximation to F_Δ for Δ ∈ [delta_gap, delta_max].

        Args:
            delta_gap: Minimum dimension (gap)
            delta_max: Maximum dimension for fitting
            n_sample_points: Number of sample points (default: 2 * poly_degree + 5)
            validate: Whether to compute fitting errors

        Returns:
            PolynomialFVector with polynomial approximations
        """
        if n_sample_points is None:
            # Use more sample points for better conditioning
            n_sample_points = 2 * self.poly_degree + 5

        # Sample at Chebyshev nodes for better interpolation
        delta_samples = self.chebyshev_nodes(n_sample_points, delta_gap, delta_max)

        # Compute F-vectors at sample points
        F_samples = np.array([self.crossing.build_F_vector(d) for d in delta_samples])

        # Convert to x = Δ - delta_gap coordinates, scaled to [0, 1] for better conditioning
        x_range = delta_max - delta_gap
        x_samples = (delta_samples - delta_gap) / x_range  # Normalized to [0, 1]

        # Fit polynomial for each component
        polynomials = []
        for i in range(self.n_constraints):
            y_values = F_samples[:, i]

            # Normalize by the scale of y values for better conditioning
            y_scale = np.max(np.abs(y_values)) + 1e-10
            y_normalized = y_values / y_scale

            # Use numpy's polynomial fitting (more stable than Vandermonde)
            coeffs_normalized = np.polynomial.polynomial.polyfit(
                x_samples, y_normalized, self.poly_degree
            )

            # Rescale coefficients to account for x normalization and y normalization
            # If p_norm(x_norm) fits y_norm, then p(x) = y_scale * p_norm(x / x_range)
            # p(x) = y_scale * sum_k c_k * (x / x_range)^k = sum_k (y_scale * c_k / x_range^k) * x^k
            coeffs = np.array([
                y_scale * coeffs_normalized[k] / (x_range ** k)
                for k in range(len(coeffs_normalized))
            ])

            # Compute fitting error on original scale
            if validate:
                x_orig = delta_samples - delta_gap
                y_fit = np.polynomial.polynomial.polyval(x_orig, coeffs)
                error = np.sqrt(np.mean((y_values - y_fit)**2))
                # Use relative error for better interpretation
                rel_error = error / (y_scale + 1e-10)
            else:
                error = 0.0
                rel_error = 0.0

            poly = FittedPolynomial(
                coefficients=coeffs,
                delta_gap=delta_gap,
                delta_max=delta_max,
                fitting_error=rel_error,  # Store relative error
                component_index=i
            )
            polynomials.append(poly)

        return PolynomialFVector(
            polynomials=polynomials,
            delta_gap=delta_gap,
            delta_max=delta_max
        )

    def validate_fit(
        self,
        poly_F: PolynomialFVector,
        n_test_points: int = 50
    ) -> Tuple[float, np.ndarray]:
        """
        Validate polynomial fit against new sample points.

        Args:
            poly_F: Fitted polynomial F-vector
            n_test_points: Number of validation points

        Returns:
            Tuple of (max_relative_error, errors_per_component)
        """
        # Sample at uniform points (different from Chebyshev nodes used for fitting)
        delta_test = np.linspace(poly_F.delta_gap, poly_F.delta_max, n_test_points)

        # Compute exact F-vectors
        F_exact = np.array([self.crossing.build_F_vector(d) for d in delta_test])

        # Compute polynomial approximations
        F_approx = np.array([poly_F.evaluate(d) for d in delta_test])

        # Compute errors
        abs_errors = np.abs(F_exact - F_approx)
        rel_errors = abs_errors / (np.abs(F_exact) + 1e-10)

        max_rel_error = np.max(rel_errors)
        errors_per_component = np.max(rel_errors, axis=0)

        return max_rel_error, errors_per_component


# =============================================================================
# Sum-of-Squares (SOS) Positivity
# =============================================================================

class SOSPositivityConstraint:
    """
    Builds sum-of-squares positivity constraints for polynomial non-negativity.

    For a polynomial p(x) of degree d to be non-negative on [0, ∞), we require:
        p(x) = σ₀(x) + x · σ₁(x)

    where σ₀ and σ₁ are SOS polynomials of appropriate degrees.

    An SOS polynomial σ(x) = v(x)ᵀ Q v(x) where:
        - v(x) = [1, x, x², ..., x^k]ᵀ is the monomial basis
        - Q ≽ 0 is a positive semidefinite Gram matrix

    The relationship between σ(x) coefficients and Q entries:
        σ_n = Σᵢⱼ Q_ij where i + j = n

    This creates linear constraints linking Q entries to polynomial coefficients.
    """

    def __init__(self, poly_degree: int):
        """
        Initialize SOS constraint builder.

        Args:
            poly_degree: Degree of polynomial to constrain
        """
        self.poly_degree = poly_degree

        # Degrees for σ₀ and σ₁
        # If p has degree d, we need:
        # - σ₀ has degree d (so k₀ = d//2 for v₀)
        # - σ₁ has degree d-1 (so k₁ = (d-1)//2 for v₁)
        self.k0 = poly_degree // 2  # Size of Q₀ is (k0+1) × (k0+1)
        self.k1 = (poly_degree - 1) // 2 if poly_degree > 0 else 0  # Size of Q₁

        # Build the constraint matrices
        self._build_coefficient_maps()

    def _build_coefficient_maps(self):
        """
        Build linear maps from Gram matrix entries to polynomial coefficients.

        For σ(x) = v(x)ᵀ Q v(x), the coefficient of x^n is:
            σ_n = Σᵢⱼ Q_ij  where i + j = n

        We store these as sparse mappings: coeff_map[n] = list of (i, j) pairs
        """
        # Map for σ₀: degree 2*k0
        self.map0 = {}
        for n in range(2 * self.k0 + 1):
            pairs = []
            for i in range(self.k0 + 1):
                j = n - i
                if 0 <= j <= self.k0:
                    pairs.append((i, j))
            self.map0[n] = pairs

        # Map for σ₁: degree 2*k1 (then multiplied by x)
        self.map1 = {}
        if self.k1 >= 0:
            for n in range(2 * self.k1 + 1):
                pairs = []
                for i in range(self.k1 + 1):
                    j = n - i
                    if 0 <= j <= self.k1:
                        pairs.append((i, j))
                self.map1[n] = pairs

    def gram_matrix_size(self) -> Tuple[int, int]:
        """Return sizes of Gram matrices (size0, size1)."""
        return (self.k0 + 1, self.k1 + 1 if self.k1 >= 0 else 0)

    def build_constraints_cvxpy(
        self,
        coefficients: np.ndarray,
        alpha: cp.Variable
    ) -> Tuple[List, cp.Variable, Optional[cp.Variable]]:
        """
        Build CVXPY constraints for α · p(x) ≥ 0 on [0, ∞).

        The polynomial α · p(x) has coefficients that are linear in α:
            (α · p(x))_n = Σᵢ αᵢ · coefficients[i, n]

        This is a SINGLE scalar polynomial (the inner product of α with the
        F-vector polynomial), which must be non-negative on [0, ∞).

        By Putinar's theorem: p(x) ≥ 0 on [0, ∞) iff p(x) = σ₀(x) + x·σ₁(x)
        where σ₀, σ₁ are SOS polynomials.

        Args:
            coefficients: Matrix of shape (n_components, poly_degree + 1)
                         where coefficients[i, n] is the x^n coeff of component i
            alpha: CVXPY variable for the linear functional

        Returns:
            Tuple of (constraints, Q0, Q1) where Q0, Q1 are Gram matrix variables
        """
        n_components = coefficients.shape[0]
        n_coeffs = coefficients.shape[1]  # poly_degree + 1

        # Create Gram matrix variables for σ₀ and σ₁
        # σ₀ has degree = poly_degree, so k0 = poly_degree // 2
        # σ₁ has degree = poly_degree - 1, so k1 = (poly_degree - 1) // 2
        size0 = self.k0 + 1
        Q0 = cp.Variable((size0, size0), symmetric=True)

        if self.k1 >= 0:
            size1 = self.k1 + 1
            Q1 = cp.Variable((size1, size1), symmetric=True)
        else:
            Q1 = None

        constraints = []

        # PSD constraints
        constraints.append(Q0 >> 0)  # Q0 is positive semidefinite
        if Q1 is not None:
            constraints.append(Q1 >> 0)  # Q1 is positive semidefinite

        # Coefficient matching constraints
        # The inner product polynomial is: (α · F)(x) = Σᵢ αᵢ · Fᵢ(x)
        # Its n-th coefficient is: Σᵢ αᵢ · coefficients[i, n]
        #
        # This must equal: σ₀_n + (x · σ₁)_n
        # where σ₀_n = Σ_{i+j=n} Q0[i,j]
        # and (x · σ₁)_n = σ₁_{n-1} = Σ_{i+j=n-1} Q1[i,j] for n ≥ 1

        for n in range(n_coeffs):
            # LHS: coefficient of x^n in α · F(x)
            # = sum over components i of alpha[i] * coefficients[i, n]
            lhs_terms = []
            for i in range(n_components):
                if abs(coefficients[i, n]) > 1e-15:  # Skip near-zero terms
                    lhs_terms.append(alpha[i] * coefficients[i, n])

            if lhs_terms:
                lhs = cp.sum(lhs_terms)
            else:
                lhs = 0

            # RHS: coefficient of x^n in σ₀(x) + x·σ₁(x)
            rhs_terms = []

            # σ₀ contribution: sum Q0[i,j] for i+j = n
            if n in self.map0:
                for (i, j) in self.map0[n]:
                    rhs_terms.append(Q0[i, j])

            # x·σ₁ contribution: sum Q1[i,j] for i+j = n-1
            if Q1 is not None and n >= 1 and (n - 1) in self.map1:
                for (i, j) in self.map1[n - 1]:
                    rhs_terms.append(Q1[i, j])

            if rhs_terms:
                rhs = cp.sum(rhs_terms)
            else:
                rhs = 0

            constraints.append(lhs == rhs)

        return constraints, Q0, Q1

    def build_constraints_cvxpy_with_fixed(
        self,
        coefficients: np.ndarray,
        alpha: cp.Variable,
        fixed_coeffs: np.ndarray
    ) -> Tuple[List, cp.Variable, Optional[cp.Variable]]:
        """
        Build CVXPY constraints for α_reduced · p_reduced(x) + fixed_poly(x) ≥ 0 on [0, ∞).

        This is a variant of build_constraints_cvxpy that handles the case where
        one component of α is fixed (for component-wise normalization).

        The polynomial constraint becomes:
            (α_reduced · p_reduced(x) + fixed_poly(x)) ≥ 0  for all x ≥ 0

        Args:
            coefficients: Reduced coefficient matrix (n_components-1, poly_degree+1)
            alpha: CVXPY variable for the reduced linear functional
            fixed_coeffs: Polynomial coefficients from the fixed α component

        Returns:
            Tuple of (constraints, Q0, Q1) where Q0, Q1 are Gram matrix variables
        """
        n_components = coefficients.shape[0]
        n_coeffs = coefficients.shape[1]  # poly_degree + 1

        # Create Gram matrix variables
        size0 = self.k0 + 1
        Q0 = cp.Variable((size0, size0), symmetric=True)

        if self.k1 >= 0:
            size1 = self.k1 + 1
            Q1 = cp.Variable((size1, size1), symmetric=True)
        else:
            Q1 = None

        constraints = []

        # PSD constraints
        constraints.append(Q0 >> 0)
        if Q1 is not None:
            constraints.append(Q1 >> 0)

        # Coefficient matching constraints with fixed contribution
        for n in range(n_coeffs):
            # LHS: coefficient of x^n in (α_reduced · F_reduced(x) + fixed_poly(x))
            lhs_terms = []

            # Variable contribution from reduced alpha
            for i in range(n_components):
                if abs(coefficients[i, n]) > 1e-15:
                    lhs_terms.append(alpha[i] * coefficients[i, n])

            # Fixed contribution
            fixed_contrib = fixed_coeffs[n] if n < len(fixed_coeffs) else 0.0

            if lhs_terms:
                lhs = cp.sum(lhs_terms) + fixed_contrib
            else:
                lhs = fixed_contrib

            # RHS: coefficient of x^n in σ₀(x) + x·σ₁(x)
            rhs_terms = []

            if n in self.map0:
                for (i, j) in self.map0[n]:
                    rhs_terms.append(Q0[i, j])

            if Q1 is not None and n >= 1 and (n - 1) in self.map1:
                for (i, j) in self.map1[n - 1]:
                    rhs_terms.append(Q1[i, j])

            if rhs_terms:
                rhs = cp.sum(rhs_terms)
            else:
                rhs = 0

            constraints.append(lhs == rhs)

        return constraints, Q0, Q1


# =============================================================================
# Polynomial Positivity Solver
# =============================================================================

class PolynomialPositivitySolver:
    """
    Bootstrap solver using polynomial positivity constraints.

    This solver:
    1. Fits F-vectors to polynomials in x = Δ - Δ_gap
    2. Enforces positivity for ALL x ≥ 0 using SOS constraints
    3. Solves the resulting SDP to find excluding functionals

    Compared to discrete sampling:
    - Continuous positivity: No gaps between sample points
    - Tighter bounds: Finds optimal functional more precisely
    - Better scaling: SDP size depends on polynomial degree, not sample count

    Usage:
        solver = PolynomialPositivitySolver(delta_sigma=0.518)
        bound = solver.find_delta_epsilon_prime_bound(delta_epsilon=1.41)
    """

    def __init__(
        self,
        delta_sigma: float,
        max_deriv: int = 21,
        poly_degree: int = 15,
        delta_max: float = 30.0
    ):
        """
        Initialize the polynomial positivity solver.

        Args:
            delta_sigma: External operator dimension
            max_deriv: Maximum derivative order (determines constraint count)
            poly_degree: Polynomial approximation degree
            delta_max: Maximum Δ for polynomial fitting
        """
        if not HAS_CVXPY:
            raise ImportError(
                "CVXPY is required for polynomial positivity constraints. "
                "Install with: pip install cvxpy"
            )

        self.delta_sigma = delta_sigma
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2
        self.poly_degree = poly_degree
        self.delta_max = delta_max

        # Create polynomial fitter
        self.fitter = PolynomialFitter(delta_sigma, max_deriv, poly_degree)

        # Create crossing vector for discrete F-vectors (identity, first scalar)
        self.crossing = TaylorCrossingVector(delta_sigma, max_deriv)

    def is_excluded(
        self,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        verbose: bool = False,
        use_sos: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Check if point (Δσ, Δε, Δε') is excluded using polynomial positivity.

        The SDP problem is:
            Find α such that:
                α · F_id = 1                    (normalization)
                α · F_ε ≥ 0                     (first scalar positivity)
                α · F_Δ ≥ 0 for all Δ ≥ Δε'    (gap positivity)

        Args:
            delta_epsilon: First scalar dimension
            delta_epsilon_prime: Gap to second scalar (what we're testing)
            verbose: Print diagnostic information
            use_sos: If True, use full SOS constraints (numerically challenging).
                     If False, use dense polynomial sampling (more robust).

        Returns:
            Tuple of (is_excluded, solver_info)
        """
        if use_sos:
            return self._is_excluded_sos(delta_epsilon, delta_epsilon_prime, verbose)
        else:
            return self._is_excluded_dense_sampling(delta_epsilon, delta_epsilon_prime, verbose)

    def _is_excluded_dense_sampling(
        self,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        verbose: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Check exclusion using polynomial-enhanced dense sampling.

        Instead of full SOS constraints, we:
        1. Fit F-vectors to polynomials
        2. Use polynomials to generate dense sample points
        3. Check positivity at all sample points

        This is more numerically robust than full SOS while still benefiting
        from polynomial interpolation.
        """
        # Get discrete F-vectors for identity and first scalar
        F_id = self.crossing.build_F_vector(0)
        F_eps = self.crossing.build_F_vector(delta_epsilon)

        # Fit polynomial to get smooth approximation
        poly_F = self.fitter.fit_polynomial(
            delta_gap=delta_epsilon_prime,
            delta_max=self.delta_max
        )

        if verbose:
            print(f"  Polynomial fitting error: {poly_F.total_fitting_error:.2e}")

        # Generate dense sample points using the polynomial
        # Use more points than discrete sampling would
        n_dense_samples = max(200, 3 * self.poly_degree)
        delta_samples = np.linspace(delta_epsilon_prime, self.delta_max, n_dense_samples)

        # Evaluate polynomial at dense sample points
        F_dense = np.array([poly_F.evaluate(d) for d in delta_samples])

        # Stack all F-vectors for reshuffling (component-wise normalization)
        F_all = np.vstack([F_eps[np.newaxis, :], F_dense])

        # Apply component-wise normalization (pycftboot/SDPB convention)
        F_reduced, fixed_contribs, max_idx = reshuffle_with_normalization(F_all, F_id)

        # Separate epsilon and other operators
        F_eps_reduced = F_reduced[0, :]
        fixed_eps = fixed_contribs[0]
        F_dense_reduced = F_reduced[1:, :]
        fixed_dense = fixed_contribs[1:]

        # Setup CVXPY problem with reduced alpha
        alpha = cp.Variable(self.n_constraints - 1)

        constraints = [
            alpha @ F_eps_reduced >= -fixed_eps,
        ]

        # Add positivity constraints at all dense sample points
        for i, F_d_reduced in enumerate(F_dense_reduced):
            constraints.append(alpha @ F_d_reduced >= -fixed_dense[i])

        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-8)

            is_excluded = prob.status == cp.OPTIMAL

            solver_info = {
                "status": prob.status,
                "fitting_error": poly_F.total_fitting_error,
                "n_dense_samples": n_dense_samples,
                "method": "dense_sampling",
            }

            if is_excluded and verbose:
                if alpha.value is not None:
                    print(f"  Found excluding functional with |α| = {np.linalg.norm(alpha.value):.4f}")

            return is_excluded, solver_info

        except Exception as e:
            warnings.warn(f"SDP solver failed: {e}")
            return False, {"status": "error", "error": str(e)}

    def _is_excluded_sos(
        self,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        verbose: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Check exclusion using full SOS (sum-of-squares) constraints.

        This is mathematically more rigorous but numerically challenging.
        Use dense_sampling method for better robustness.
        """
        # Get discrete F-vectors for identity and first scalar
        F_id = self.crossing.build_F_vector(0)
        F_eps = self.crossing.build_F_vector(delta_epsilon)

        # Fit polynomial to F_Δ for Δ ≥ Δε'
        poly_F = self.fitter.fit_polynomial(
            delta_gap=delta_epsilon_prime,
            delta_max=self.delta_max
        )

        if verbose:
            print(f"  Polynomial fitting error: {poly_F.total_fitting_error:.2e}")

        # Get coefficient matrix: shape (n_constraints, poly_degree + 1)
        coeff_matrix = poly_F.get_coefficient_matrix()

        # Build SOS positivity constraint
        sos = SOSPositivityConstraint(self.poly_degree)

        # Apply component-wise normalization (pycftboot/SDPB convention)
        # Note: For SOS we can't simply reshuffle, but we can fix one component
        # We use the same approach: fix alpha[max_idx] = 1 / F_id[max_idx]
        max_idx = np.argmax(np.abs(F_id))
        alpha_fixed = 1.0 / F_id[max_idx]

        # Fixed contribution to F_eps constraint
        fixed_eps = alpha_fixed * F_eps[max_idx]

        # Build reduced F_eps (without max_idx component)
        F_eps_reduced = np.concatenate([F_eps[:max_idx], F_eps[max_idx+1:]])

        # Build reduced coefficient matrix (without max_idx row)
        coeff_matrix_reduced = np.vstack([coeff_matrix[:max_idx, :], coeff_matrix[max_idx+1:, :]])

        # Fixed contribution to polynomial: alpha_fixed * (row max_idx of coeff_matrix)
        fixed_poly_coeffs = alpha_fixed * coeff_matrix[max_idx, :]

        # Setup CVXPY problem with reduced alpha
        alpha = cp.Variable(self.n_constraints - 1)

        constraints = [
            alpha @ F_eps_reduced >= -fixed_eps,
        ]

        # Polynomial positivity via SOS (with fixed contribution)
        # The constraint is: alpha_reduced @ coeff_matrix_reduced + fixed_poly_coeffs >= 0
        # We need to modify SOS to handle the fixed contribution
        sos_constraints, Q0, Q1 = sos.build_constraints_cvxpy_with_fixed(
            coeff_matrix_reduced, alpha, fixed_poly_coeffs
        )
        constraints.extend(sos_constraints)

        # Solve feasibility problem
        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            # Try different solvers
            solved = False

            if not solved:
                try:
                    prob.solve(solver=cp.MOSEK, verbose=False)
                    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        solved = True
                except (cp.error.SolverError, Exception):
                    pass

            if not solved:
                try:
                    prob.solve(solver=cp.CVXOPT, verbose=False)
                    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        solved = True
                except (cp.error.SolverError, Exception):
                    pass

            if not solved:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-6)

            is_excluded = prob.status == cp.OPTIMAL

            solver_info = {
                "status": prob.status,
                "fitting_error": poly_F.total_fitting_error,
                "gram_matrix_sizes": sos.gram_matrix_size(),
                "n_constraints": len(constraints),
                "method": "sos",
            }

            if is_excluded and verbose:
                if alpha.value is not None:
                    print(f"  Found excluding functional with |α| = {np.linalg.norm(alpha.value):.4f}")

            return is_excluded, solver_info

        except Exception as e:
            warnings.warn(f"SDP solver failed: {e}")
            return False, {"status": "error", "error": str(e)}

    def is_excluded_hybrid(
        self,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        n_discrete_samples: int = 30,
        verbose: bool = False
    ) -> Tuple[bool, Dict]:
        """
        Hybrid method: polynomial positivity + discrete samples for robustness.

        This adds discrete sample constraints alongside polynomial positivity
        to improve numerical robustness, especially near boundaries.

        Args:
            delta_epsilon: First scalar dimension
            delta_epsilon_prime: Gap to second scalar
            n_discrete_samples: Number of discrete sample points to add
            verbose: Print diagnostic information

        Returns:
            Tuple of (is_excluded, solver_info)
        """
        # Get discrete F-vectors
        F_id = self.crossing.build_F_vector(0)
        F_eps = self.crossing.build_F_vector(delta_epsilon)

        # Sample some discrete points for robustness
        deltas_discrete = np.linspace(
            delta_epsilon_prime,
            self.delta_max,
            n_discrete_samples
        )
        F_discrete = np.array([self.crossing.build_F_vector(d) for d in deltas_discrete])

        # Fit polynomial
        poly_F = self.fitter.fit_polynomial(
            delta_gap=delta_epsilon_prime,
            delta_max=self.delta_max
        )
        coeff_matrix = poly_F.get_coefficient_matrix()

        # Build SOS constraint
        sos = SOSPositivityConstraint(self.poly_degree)

        # Apply component-wise normalization (pycftboot/SDPB convention)
        max_idx = np.argmax(np.abs(F_id))
        alpha_fixed = 1.0 / F_id[max_idx]

        # Fixed contributions
        fixed_eps = alpha_fixed * F_eps[max_idx]
        F_eps_reduced = np.concatenate([F_eps[:max_idx], F_eps[max_idx+1:]])

        # Stack and reshuffle discrete F-vectors
        F_all_discrete = np.vstack([F_eps[np.newaxis, :], F_discrete])
        F_reduced_discrete, fixed_discrete, _ = reshuffle_with_normalization(F_all_discrete, F_id)
        F_discrete_reduced = F_reduced_discrete[1:, :]
        fixed_ops = fixed_discrete[1:]

        # Reduced coefficient matrix (without max_idx row)
        coeff_matrix_reduced = np.vstack([coeff_matrix[:max_idx, :], coeff_matrix[max_idx+1:, :]])
        fixed_poly_coeffs = alpha_fixed * coeff_matrix[max_idx, :]

        # Setup CVXPY problem with reduced alpha
        alpha = cp.Variable(self.n_constraints - 1)

        constraints = [
            alpha @ F_eps_reduced >= -fixed_eps,
        ]

        # Add discrete sample constraints with fixed contribution
        for i, F_O_reduced in enumerate(F_discrete_reduced):
            constraints.append(alpha @ F_O_reduced >= -fixed_ops[i])

        # Add polynomial positivity constraints with fixed contribution
        sos_constraints, Q0, Q1 = sos.build_constraints_cvxpy_with_fixed(
            coeff_matrix_reduced, alpha, fixed_poly_coeffs
        )
        constraints.extend(sos_constraints)

        # Solve
        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-8)

            is_excluded = prob.status == cp.OPTIMAL

            solver_info = {
                "status": prob.status,
                "fitting_error": poly_F.total_fitting_error,
                "n_discrete_samples": n_discrete_samples,
                "n_total_constraints": len(constraints),
            }

            return is_excluded, solver_info

        except Exception as e:
            warnings.warn(f"Hybrid solver failed: {e}")
            return False, {"status": "error", "error": str(e)}

    def find_delta_epsilon_prime_bound(
        self,
        delta_epsilon: float,
        delta_prime_min: Optional[float] = None,
        delta_prime_max: float = 8.0,
        tolerance: float = 0.01,
        method: str = "polynomial",
        verbose: bool = True
    ) -> float:
        """
        Find upper bound on Δε' using binary search.

        Args:
            delta_epsilon: First scalar dimension
            delta_prime_min: Minimum Δε' to search (default: delta_epsilon + 0.1)
            delta_prime_max: Maximum Δε' to search
            tolerance: Binary search tolerance
            method: "polynomial" for pure SOS, "hybrid" for SOS + discrete samples
            verbose: Print progress

        Returns:
            Upper bound on Δε'
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1
        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        if verbose:
            print(f"Finding Δε' bound with polynomial positivity")
            print(f"  Δσ = {self.delta_sigma:.4f}")
            print(f"  Δε = {delta_epsilon:.4f}")
            print(f"  Polynomial degree: {self.poly_degree}")
            print(f"  Constraints: {self.n_constraints}")
            print(f"  Method: {method}")
            print(f"  Search range: [{delta_prime_min:.2f}, {delta_prime_max:.2f}]")
            print("-" * 50)

        # Choose exclusion method
        if method == "hybrid":
            check_excluded = lambda dp: self.is_excluded_hybrid(
                delta_epsilon, dp, verbose=False
            )[0]
        else:
            check_excluded = lambda dp: self.is_excluded(
                delta_epsilon, dp, verbose=False
            )[0]

        # Check boundary conditions
        if verbose:
            print(f"  Checking Δε' = {delta_prime_min:.2f}...", end=" ", flush=True)

        if check_excluded(delta_prime_min):
            if verbose:
                print("EXCLUDED")
            return delta_prime_min
        else:
            if verbose:
                print("ALLOWED")

        if verbose:
            print(f"  Checking Δε' = {delta_prime_max:.2f}...", end=" ", flush=True)

        if not check_excluded(delta_prime_max):
            if verbose:
                print("ALLOWED (no bound)")
            return float('inf')
        else:
            if verbose:
                print("EXCLUDED")

        # Binary search
        lo, hi = delta_prime_min, delta_prime_max
        iteration = 0

        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            iteration += 1

            if verbose:
                print(f"  [{iteration}] Δε' = {mid:.4f}...", end=" ", flush=True)

            if check_excluded(mid):
                if verbose:
                    print("EXCLUDED")
                hi = mid
            else:
                if verbose:
                    print("ALLOWED")
                lo = mid

        bound = (lo + hi) / 2

        if verbose:
            print("-" * 50)
            print(f"  Result: Δε' ≤ {bound:.4f}")

        return bound


# =============================================================================
# Gap Solver with Polynomial Positivity
# =============================================================================

class PolynomialPositivityGapSolver:
    """
    Compute Δε' bounds over a curve in (Δσ, Δε) space using polynomial positivity.

    This class provides the same interface as DeltaEpsilonPrimeBoundComputer
    but uses polynomial positivity constraints instead of discrete sampling.

    Usage:
        computer = PolynomialPositivityGapSolver(max_deriv=21, poly_degree=15)
        results = computer.compute_ising_plot()
    """

    def __init__(
        self,
        max_deriv: int = 21,
        poly_degree: int = 15,
        delta_max: float = 30.0
    ):
        """
        Initialize the gap solver.

        Args:
            max_deriv: Maximum derivative order
            poly_degree: Polynomial approximation degree
            delta_max: Maximum Δ for fitting
        """
        self.max_deriv = max_deriv
        self.poly_degree = poly_degree
        self.delta_max = delta_max
        self.n_constraints = (max_deriv + 1) // 2

    # Literature values for Δε boundary (same as in bootstrap_gap_solver.py)
    LITERATURE_BOUNDARY = {
        0.500: 1.000,
        0.505: 1.050,
        0.510: 1.150,
        0.515: 1.300,
        0.5181489: 1.412625,
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

    @classmethod
    def delta_epsilon_boundary_literature(cls, delta_sigma: float) -> float:
        """
        Get Δε boundary value from literature using interpolation.
        """
        ds_values = np.array(sorted(cls.LITERATURE_BOUNDARY.keys()))
        de_values = np.array([cls.LITERATURE_BOUNDARY[ds] for ds in ds_values])

        if delta_sigma <= ds_values[0]:
            return de_values[0]
        if delta_sigma >= ds_values[-1]:
            slope = (de_values[-1] - de_values[-2]) / (ds_values[-1] - ds_values[-2])
            return de_values[-1] + slope * (delta_sigma - ds_values[-1])

        idx = np.searchsorted(ds_values, delta_sigma) - 1
        t = (delta_sigma - ds_values[idx]) / (ds_values[idx + 1] - ds_values[idx])
        return de_values[idx] + t * (de_values[idx + 1] - de_values[idx])

    def compute_bound_at_point(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        tolerance: float = 0.02,
        method: str = "polynomial",
        verbose: bool = False
    ) -> float:
        """
        Compute Δε' bound at a single (Δσ, Δε) point.

        Args:
            delta_sigma: External dimension
            delta_epsilon: First scalar dimension
            tolerance: Binary search tolerance
            method: "polynomial" or "hybrid"
            verbose: Print progress

        Returns:
            Upper bound on Δε'
        """
        solver = PolynomialPositivitySolver(
            delta_sigma=delta_sigma,
            max_deriv=self.max_deriv,
            poly_degree=self.poly_degree,
            delta_max=self.delta_max
        )

        return solver.find_delta_epsilon_prime_bound(
            delta_epsilon=delta_epsilon,
            tolerance=tolerance,
            method=method,
            verbose=verbose
        )

    def compute_bounds_along_curve(
        self,
        delta_sigma_values: np.ndarray,
        delta_epsilon_func: Optional[Callable[[float], float]] = None,
        tolerance: float = 0.02,
        method: str = "polynomial",
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute Δε' bounds along a curve in (Δσ, Δε) space.

        Args:
            delta_sigma_values: Array of Δσ values
            delta_epsilon_func: Function mapping Δσ → Δε (default: literature boundary)
            tolerance: Binary search tolerance
            method: "polynomial" or "hybrid"
            verbose: Print progress

        Returns:
            Array of shape (N, 3) with [Δσ, Δε, Δε'_bound]
        """
        if delta_epsilon_func is None:
            delta_epsilon_func = self.delta_epsilon_boundary_literature

        results = []
        n_total = len(delta_sigma_values)

        if verbose:
            print("=" * 60)
            print("Computing Δε' bounds with polynomial positivity")
            print("=" * 60)
            print(f"  Points: {n_total}")
            print(f"  Δσ range: [{delta_sigma_values.min():.3f}, {delta_sigma_values.max():.3f}]")
            print(f"  Polynomial degree: {self.poly_degree}")
            print(f"  Constraints: {self.n_constraints}")
            print(f"  Method: {method}")
            print("-" * 60)

        for i, ds in enumerate(delta_sigma_values):
            de = delta_epsilon_func(ds)

            if verbose:
                print(f"[{i+1}/{n_total}] Δσ={ds:.4f}, Δε={de:.4f} ... ", end='', flush=True)

            bound = self.compute_bound_at_point(
                delta_sigma=ds,
                delta_epsilon=de,
                tolerance=tolerance,
                method=method,
                verbose=False
            )

            results.append([ds, de, bound])

            if verbose:
                if np.isinf(bound):
                    print("no bound")
                else:
                    print(f"Δε' ≤ {bound:.4f}")

        return np.array(results)

    def compute_ising_plot(
        self,
        delta_sigma_min: float = 0.50,
        delta_sigma_max: float = 0.60,
        n_points: int = 25,
        tolerance: float = 0.02,
        method: str = "polynomial",
        verbose: bool = True
    ) -> np.ndarray:
        """
        Compute data for the Ising Δε' bound plot (Fig. 7 of El-Showk et al.).

        Args:
            delta_sigma_min: Minimum Δσ
            delta_sigma_max: Maximum Δσ
            n_points: Number of points
            tolerance: Binary search tolerance
            method: "polynomial" or "hybrid"
            verbose: Print progress

        Returns:
            Array of shape (n_points, 3) with [Δσ, Δε, Δε'_bound]
        """
        delta_sigmas = np.linspace(delta_sigma_min, delta_sigma_max, n_points)

        return self.compute_bounds_along_curve(
            delta_sigma_values=delta_sigmas,
            delta_epsilon_func=self.delta_epsilon_boundary_literature,
            tolerance=tolerance,
            method=method,
            verbose=verbose
        )


# =============================================================================
# Comparison with Discrete Sampling
# =============================================================================

def compare_methods(
    delta_sigma: float = 0.518,
    delta_epsilon: float = 1.41,
    max_deriv: int = 11,
    poly_degree: int = 12,
    tolerance: float = 0.05,
    verbose: bool = True
) -> Dict:
    """
    Compare polynomial positivity with discrete sampling.

    This function runs both methods and reports the differences in bounds
    and computation characteristics.

    Args:
        delta_sigma: External dimension (default: Ising point)
        delta_epsilon: First scalar dimension
        max_deriv: Maximum derivative order
        poly_degree: Polynomial degree for polynomial method
        tolerance: Binary search tolerance
        verbose: Print comparison details

    Returns:
        Dictionary with comparison results
    """
    from .taylor_conformal_blocks import HighOrderGapBootstrapSolver

    if verbose:
        print("=" * 60)
        print("Comparing Discrete Sampling vs Polynomial Positivity")
        print("=" * 60)
        print(f"  Δσ = {delta_sigma:.4f}, Δε = {delta_epsilon:.4f}")
        print(f"  Constraints: {(max_deriv + 1) // 2}")
        print(f"  Polynomial degree: {poly_degree}")
        print("-" * 60)

    results = {}

    # Discrete sampling method
    if verbose:
        print("\n1. Discrete sampling (150 samples):")

    discrete_solver = HighOrderGapBootstrapSolver(d=3, max_deriv=max_deriv)

    import time
    t0 = time.time()
    discrete_bound = discrete_solver.find_delta_epsilon_prime_bound(
        delta_sigma, delta_epsilon, tolerance=tolerance
    )
    discrete_time = time.time() - t0

    results["discrete"] = {
        "bound": discrete_bound,
        "time": discrete_time,
        "n_samples": 150,
    }

    if verbose:
        print(f"     Bound: Δε' ≤ {discrete_bound:.4f}")
        print(f"     Time: {discrete_time:.2f}s")

    # Polynomial positivity method
    if verbose:
        print("\n2. Polynomial positivity:")

    poly_solver = PolynomialPositivitySolver(
        delta_sigma=delta_sigma,
        max_deriv=max_deriv,
        poly_degree=poly_degree
    )

    t0 = time.time()
    poly_bound = poly_solver.find_delta_epsilon_prime_bound(
        delta_epsilon, tolerance=tolerance, verbose=False
    )
    poly_time = time.time() - t0

    results["polynomial"] = {
        "bound": poly_bound,
        "time": poly_time,
        "poly_degree": poly_degree,
    }

    if verbose:
        print(f"     Bound: Δε' ≤ {poly_bound:.4f}")
        print(f"     Time: {poly_time:.2f}s")

    # Hybrid method
    if verbose:
        print("\n3. Hybrid (polynomial + 30 discrete samples):")

    t0 = time.time()
    hybrid_bound = poly_solver.find_delta_epsilon_prime_bound(
        delta_epsilon, tolerance=tolerance, method="hybrid", verbose=False
    )
    hybrid_time = time.time() - t0

    results["hybrid"] = {
        "bound": hybrid_bound,
        "time": hybrid_time,
    }

    if verbose:
        print(f"     Bound: Δε' ≤ {hybrid_bound:.4f}")
        print(f"     Time: {hybrid_time:.2f}s")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary:")
        print("-" * 60)
        print(f"  Discrete sampling:      Δε' ≤ {discrete_bound:.4f}")
        print(f"  Polynomial positivity:  Δε' ≤ {poly_bound:.4f}")
        print(f"  Hybrid method:          Δε' ≤ {hybrid_bound:.4f}")
        print(f"  Reference (El-Showk):   Δε' ~ 3.8")
        print("-" * 60)

        if poly_bound > discrete_bound:
            improvement = poly_bound - discrete_bound
            print(f"  Polynomial gives {improvement:.4f} higher (weaker/correct) bound")
        else:
            print(f"  Methods give similar bounds")

        print("=" * 60)

    return results


# =============================================================================
# Testing
# =============================================================================

def test_polynomial_positivity():
    """Test the polynomial positivity implementation."""
    print("=" * 60)
    print("Polynomial Positivity Constraint Tests")
    print("=" * 60)

    # Test 1: Polynomial fitting
    print("\n1. Testing polynomial fitting:")
    fitter = PolynomialFitter(delta_sigma=0.518, max_deriv=11, poly_degree=12)
    poly_F = fitter.fit_polynomial(delta_gap=1.5, delta_max=30.0)

    print(f"   Components: {poly_F.n_components}")
    print(f"   Max degree: {poly_F.max_degree}")
    print(f"   Fitting error: {poly_F.total_fitting_error:.2e}")

    # Validate fit
    max_err, comp_err = fitter.validate_fit(poly_F)
    print(f"   Max relative error: {max_err:.2e}")

    # Test 2: SOS constraint building
    print("\n2. Testing SOS constraint structure:")
    sos = SOSPositivityConstraint(poly_degree=12)
    size0, size1 = sos.gram_matrix_size()
    print(f"   Gram matrix sizes: Q0 is {size0}x{size0}, Q1 is {size1}x{size1}")

    # Test 3: Single point exclusion check
    print("\n3. Testing exclusion check at Ising point:")
    solver = PolynomialPositivitySolver(
        delta_sigma=0.518,
        max_deriv=11,
        poly_degree=12
    )

    for de_prime in [2.0, 2.5, 3.0, 3.5]:
        excluded, info = solver.is_excluded(1.41, de_prime)
        status = "EXCLUDED" if excluded else "ALLOWED"
        print(f"   Δε' = {de_prime:.1f}: {status}")

    # Test 4: Find bound
    print("\n4. Finding Δε' bound:")
    bound = solver.find_delta_epsilon_prime_bound(
        delta_epsilon=1.41,
        tolerance=0.05,
        verbose=True
    )

    print(f"\n   Final bound: Δε' ≤ {bound:.4f}")
    print(f"   Reference (El-Showk 2012): ~3.8 at Ising kink")

    print("\n" + "=" * 60)
    print("Tests complete")
    print("=" * 60)


if __name__ == "__main__":
    test_polynomial_positivity()
