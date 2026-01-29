"""
SDPB Interface for the Conformal Bootstrap.

This module provides integration with SDPB (Semidefinite Program solver for the Bootstrap),
the gold-standard solver for conformal bootstrap problems.

SDPB solves Polynomial Matrix Programs (PMP) of the form:

    maximize: b₀ + Σᵢ bᵢ yᵢ
    subject to: M_j^0(x) + Σᵢ yᵢ M_j^i(x) ≽ 0 for all x ≥ 0, j = 1, ..., J

where M_j^i(x) are symmetric matrices whose entries are polynomials in x.

For the conformal bootstrap, we encode:
- The crossing equation constraint: α·F_id = 1 (normalization)
- Positivity for each operator: α·F_Δ ≥ 0 for all Δ ≥ Δ_gap

The key is to approximate F_Δ as a polynomial in Δ, then enforce positivity
for all Δ ≥ Δ_gap using polynomial SDP constraints.

References:
    - D. Simmons-Duffin, "A Semidefinite Program Solver for the Conformal Bootstrap"
      JHEP 1506, 174 (2015) [arXiv:1502.02033]
    - W. Landry and D. Simmons-Duffin, "Scaling the semidefinite program solver SDPB"
      [arXiv:1909.09745]

Installation:
    SDPB must be installed separately. See https://github.com/davidsd/sdpb

    Docker (recommended for macOS/Windows):
        docker pull davidsd/sdpb:master

    Singularity (recommended for Linux/HPC):
        See https://github.com/davidsd/sdpb/blob/master/docs/Singularity.md

    From source:
        See https://github.com/davidsd/sdpb/blob/master/Install.md

Usage:
    >>> solver = SDPBSolver(sdpb_path="/usr/local/bin/sdpb")
    >>> bound = solver.find_bound(delta_sigma=0.518, delta_epsilon=1.41)
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.polynomial import chebyshev

# Import from other modules
try:
    from .taylor_conformal_blocks import TaylorCrossingVector, build_F_vector_taylor
except ImportError:
    from taylor_conformal_blocks import TaylorCrossingVector, build_F_vector_taylor

try:
    from .el_showk_basis import (
        ElShowkCrossingVector,
        ElShowkBootstrapSolver,
        count_coefficients,
        get_multiresolution_operators,
        get_simplified_multiresolution
    )
except ImportError:
    from el_showk_basis import (
        ElShowkCrossingVector,
        ElShowkBootstrapSolver,
        count_coefficients,
        get_multiresolution_operators,
        get_simplified_multiresolution
    )

# Import polynomial infrastructure
try:
    from .polynomial_bootstrap import (
        SymbolicPolynomialVector, BilinearBasis, PMPGenerator,
        R_CROSS, PREC, unitarity_bound as poly_unitarity_bound
    )
    HAVE_POLYNOMIAL_INFRASTRUCTURE = True
except ImportError:
    try:
        from polynomial_bootstrap import (
            SymbolicPolynomialVector, BilinearBasis, PMPGenerator,
            R_CROSS, PREC, unitarity_bound as poly_unitarity_bound
        )
        HAVE_POLYNOMIAL_INFRASTRUCTURE = True
    except ImportError:
        HAVE_POLYNOMIAL_INFRASTRUCTURE = False
        warnings.warn("polynomial_bootstrap not available")

# Import pycftboot bridge
try:
    from .pycftboot_bridge import (
        PycftbootBlockTable, generate_F_vectors_pycftboot,
        PYCFTBOOT_LOADED
    )
    HAVE_PYCFTBOOT_BRIDGE = PYCFTBOOT_LOADED
except ImportError:
    try:
        from pycftboot_bridge import (
            PycftbootBlockTable, generate_F_vectors_pycftboot,
            PYCFTBOOT_LOADED
        )
        HAVE_PYCFTBOOT_BRIDGE = PYCFTBOOT_LOADED
    except ImportError:
        HAVE_PYCFTBOOT_BRIDGE = False


from enum import Enum, auto


class SDPBExecutionMode(Enum):
    """Execution mode for SDPB."""
    BINARY = auto()      # Direct binary execution
    DOCKER = auto()      # Docker container
    SINGULARITY = auto() # Singularity container (for HPC clusters)


@dataclass
class DockerConfig:
    """Configuration for Docker-based SDPB execution."""
    image: str = "bootstrapcollaboration/sdpb:master"
    # Additional docker run options (e.g., memory limits)
    extra_options: List[str] = field(default_factory=list)


@dataclass
class SingularityConfig:
    """Configuration for Singularity-based SDPB execution (for HPC clusters like FASRC)."""
    # Image path - can be set via SDPB_SINGULARITY_IMAGE environment variable
    image_path: str = field(default_factory=lambda: os.environ.get(
        "SDPB_SINGULARITY_IMAGE", "${HOME}/singularity/sdpb_master.sif"
    ))
    # Use srun for MPI (FASRC hybrid model) - can be set via SDPB_USE_SRUN env var
    use_srun: bool = field(default_factory=lambda: os.environ.get(
        "SDPB_USE_SRUN", "true"
    ).lower() == "true")
    # MPI type for srun (pmix for FASRC) - can be set via SDPB_MPI_TYPE env var
    mpi_type: str = field(default_factory=lambda: os.environ.get(
        "SDPB_MPI_TYPE", "pmix"
    ))
    # Additional singularity exec options
    extra_options: List[str] = field(default_factory=list)


@dataclass
class SDPBConfig:
    """Configuration for SDPB solver."""

    # Paths to SDPB executables (used for BINARY mode)
    sdpb_path: str = "sdpb"
    pmp2sdp_path: str = "pmp2sdp"

    # Execution mode (auto-detected if None)
    execution_mode: Optional[SDPBExecutionMode] = None

    # Docker configuration (for DOCKER mode)
    docker: DockerConfig = field(default_factory=DockerConfig)

    # Singularity configuration (for SINGULARITY mode)
    singularity: SingularityConfig = field(default_factory=SingularityConfig)

    # Numerical precision (in bits)
    precision: int = 400  # ~120 decimal digits

    # Solver parameters
    max_iterations: int = 500
    duality_gap_threshold: float = 1e-30
    primal_error_threshold: float = 1e-30
    dual_error_threshold: float = 1e-30

    # Number of threads for MPI
    num_threads: int = 4

    # Working directory for temporary files
    work_dir: Optional[str] = None

    # Verbosity level
    verbosity: str = "regular"  # "silent", "regular", or "debug"

    # Keep temporary files for debugging
    keep_temp_files: bool = False


@dataclass
class PolynomialVector:
    """
    A vector of polynomials, representing a constraint in the bootstrap problem.

    Each component is a polynomial in x (the dimension variable shifted so x ≥ 0
    corresponds to Δ ≥ Δ_gap).
    """
    # Polynomial coefficients: polynomials[i] = [c_0, c_1, ..., c_d] for i-th component
    polynomials: List[np.ndarray] = field(default_factory=list)

    # Damped rational prefactor parameters (for positivity measure)
    # The prefactor is: base^x / prod(x - poles)
    prefactor_constant: float = 1.0
    prefactor_base: float = np.exp(-1)  # e^{-x} default
    prefactor_poles: List[float] = field(default_factory=list)

    @property
    def dimension(self) -> int:
        """Number of components (constraint dimension)."""
        return len(self.polynomials)

    @property
    def max_degree(self) -> int:
        """Maximum polynomial degree across all components."""
        if not self.polynomials:
            return 0
        return max(len(p) - 1 for p in self.polynomials)


class PolynomialApproximator:
    """
    Approximates F-vectors as polynomials in Δ for SDPB input.

    The key challenge is that F_Δ(constraint values) are not polynomials in Δ.
    We approximate them using:
    1. Chebyshev interpolation on a shifted domain
    2. The approximation is exact at sample points
    3. Error is controlled by polynomial degree
    """

    def __init__(
        self,
        delta_sigma: float,
        max_deriv: int = 21,
        poly_degree: int = 20,
    ):
        """
        Initialize the polynomial approximator.

        Args:
            delta_sigma: External operator dimension
            max_deriv: Maximum derivative order (odd integers 1, 3, ...)
            poly_degree: Degree of polynomial approximation
        """
        self.delta_sigma = delta_sigma
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2
        self.poly_degree = poly_degree

        # Crossing vector computer
        self.crossing = TaylorCrossingVector(delta_sigma, max_deriv)

    def approximate_F_as_polynomial(
        self,
        delta_min: float,
        delta_max: float,
        include_prefactor: bool = True
    ) -> PolynomialVector:
        """
        Approximate F_Δ as a polynomial in x = Δ - delta_min.

        We fit F_Δ on the interval [delta_min, delta_max] using Chebyshev
        interpolation, then convert to a polynomial in x = Δ - delta_min.

        Args:
            delta_min: Minimum dimension (gap)
            delta_max: Maximum dimension for fitting
            include_prefactor: Include damped rational prefactor

        Returns:
            PolynomialVector with polynomial approximation
        """
        # Sample points using Chebyshev nodes (better interpolation)
        n_points = self.poly_degree + 1

        # Chebyshev nodes on [-1, 1]
        cheb_nodes = np.cos(np.pi * (2 * np.arange(n_points) + 1) / (2 * n_points))

        # Map to [delta_min, delta_max]
        delta_samples = 0.5 * (delta_max - delta_min) * (cheb_nodes + 1) + delta_min

        # Compute F-vectors at sample points
        F_samples = np.array([self.crossing.build_F_vector(d) for d in delta_samples])

        # Fit polynomials for each constraint component
        # We want polynomials in x = Δ - delta_min
        x_samples = delta_samples - delta_min

        polynomials = []
        for i in range(self.n_constraints):
            # Values of this component at sample points
            y_values = F_samples[:, i]

            # Fit polynomial using least squares (more stable than Vandermonde)
            coeffs = np.polynomial.polynomial.polyfit(
                x_samples, y_values, self.poly_degree
            )
            polynomials.append(coeffs)

        # Create polynomial vector
        result = PolynomialVector(polynomials=polynomials)

        if include_prefactor:
            # Damped rational prefactor for positivity measure
            # Use R_CROSS = 3 - 2*sqrt(2) ≈ 0.172 to match pycftboot conventions
            # This is the crossing-symmetric radial coordinate
            R_CROSS = 3 - 2 * np.sqrt(2)  # ≈ 0.17157
            result.prefactor_constant = 1.0
            result.prefactor_base = R_CROSS
            result.prefactor_poles = []

        return result

    def build_polynomial_matrix_program(
        self,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        delta_max: float = 40.0,
    ) -> Dict:
        """
        Build the Polynomial Matrix Program for SDPB.

        The PMP encodes:
        - Normalization: α·F_id = 1
        - First scalar positivity: α·F_ε ≥ 0
        - Gap positivity: α·F_Δ ≥ 0 for all Δ ≥ Δε'

        Args:
            delta_epsilon: First scalar dimension
            delta_epsilon_prime: Gap to second scalar
            delta_max: Maximum dimension for polynomial fitting

        Returns:
            Dictionary with PMP data in SDPB JSON format
        """
        # Get F-vectors for identity and first scalar
        F_id = self.crossing.build_F_vector(0)
        F_eps = self.crossing.build_F_vector(delta_epsilon)

        # Approximate F_Δ for Δ ≥ Δε' as polynomial in x = Δ - Δε'
        F_poly = self.approximate_F_as_polynomial(
            delta_epsilon_prime, delta_max
        )

        # Build PMP structure
        # The problem is:
        #   maximize 0  (feasibility)
        #   subject to: α·F_id = 1 (via normalization)
        #               α·F_ε ≥ 0 (discrete constraint)
        #               α·F_Δ ≥ 0 for all Δ ≥ Δε' (polynomial constraint)

        pmp = {
            "objective": self._format_vector(np.zeros(self.n_constraints)),
            "normalization": self._format_vector(F_id),
            "PositiveMatrixWithPrefactorArray": []
        }

        # Add discrete constraint for first scalar
        # This is a 1x1 matrix constraint: [α·F_ε] ≽ 0
        # Each optimization variable i contributes a constant polynomial [F_eps[i]]
        pmp["PositiveMatrixWithPrefactorArray"].append({
            "DampedRational": {
                "constant": "1",
                "base": "0.5",  # Must be in (0, 1) for SDPB
                "poles": []
            },
            "polynomials": [[[
                [v] for v in self._format_vector(F_eps)
            ]]]
        })

        # Add polynomial constraint for gap
        # This is the key SDPB feature: enforce α·F_Δ ≥ 0 for all Δ ≥ Δε'
        poly_matrix = self._build_polynomial_matrix(F_poly)
        pmp["PositiveMatrixWithPrefactorArray"].append({
            "DampedRational": {
                "constant": str(F_poly.prefactor_constant),
                "base": str(F_poly.prefactor_base),
                "poles": [str(p) for p in F_poly.prefactor_poles]
            },
            "polynomials": poly_matrix
        })

        return pmp

    def _format_vector(self, vec: np.ndarray) -> List[str]:
        """Format numpy array as list of string numbers for JSON."""
        return [f"{x:.15e}" for x in vec]

    def _format_polynomial(self, coeffs: np.ndarray) -> List[str]:
        """Format polynomial coefficients as list of string numbers."""
        return [f"{c:.15e}" for c in coeffs]

    def _build_polynomial_matrix(self, F_poly: PolynomialVector) -> List:
        """
        Build the polynomial matrix for SDPB.

        SDPB PMP format for a 1x1 matrix with n optimization variables:
            polynomials[row][col] = [poly_var_0, poly_var_1, ...]
        where each poly_var_i = [coeff_deg0, coeff_deg1, ...] is the
        polynomial contributed by optimization variable i.
        """
        # Each optimization variable i has polynomial F_poly.polynomials[i]
        # already stored as [coeff_deg0, coeff_deg1, ...]
        poly_list = []
        for i in range(self.n_constraints):
            coeffs = F_poly.polynomials[i]
            poly_list.append([f"{c:.15e}" for c in coeffs])

        # 1x1 matrix: polynomials[0][0] = list of per-variable polynomials
        return [[poly_list]]


class ElShowkPolynomialApproximator:
    """
    Polynomial approximator for El-Showk F-vectors with spinning operators.

    This extends PolynomialApproximator to work with the El-Showk (a,b) derivative
    basis and spinning operators, enabling high-precision SDPB bounds.

    The El-Showk basis uses mixed derivatives:
        ∂_a^m ∂_b^n F(a,b)|_{a=0,b=0}  for m odd, m + 2n ≤ 2*nmax + 1

    This gives more constraints than the standard Taylor approach.

    Reference:
        El-Showk et al., "Solving the 3D Ising Model with the Conformal Bootstrap"
        arXiv:1203.6064 (2012)
    """

    def __init__(
        self,
        delta_sigma: float,
        nmax: int = 10,
        max_spin: int = 50,
        poly_degree: int = 20,
    ):
        """
        Initialize the El-Showk polynomial approximator.

        Args:
            delta_sigma: External operator dimension
            nmax: El-Showk derivative order (nmax=10 gives 66 coefficients)
            max_spin: Maximum spin for spinning operators
            poly_degree: Degree of polynomial approximation
        """
        self.delta_sigma = delta_sigma
        self.nmax = nmax
        self.max_spin = max_spin
        self.n_constraints = count_coefficients(nmax)
        self.poly_degree = poly_degree

        # El-Showk crossing vector computer (high_precision for numerical stability)
        # High precision uses mpmath which avoids boundary issues in Richardson extrapolation
        self.crossing = ElShowkCrossingVector(delta_sigma, nmax, high_precision=True)

    def unitarity_bound(self, ell: int, d: int = 3) -> float:
        """Unitarity bound for spin ell operators in d dimensions."""
        if ell == 0:
            return (d - 2) / 2  # 0.5 for d=3
        return ell + d - 2  # ell + 1 for d=3

    def approximate_F_as_polynomial(
        self,
        delta_min: float,
        delta_max: float,
        spin: int = 0,
        include_prefactor: bool = True
    ) -> PolynomialVector:
        """
        Approximate F_Δ as a polynomial in x = Δ - delta_min for given spin.

        Uses Chebyshev interpolation for numerical stability.

        Args:
            delta_min: Minimum dimension (gap for scalars, unitarity bound for spinning)
            delta_max: Maximum dimension for fitting
            spin: Operator spin (0, 2, 4, ...)
            include_prefactor: Include damped rational prefactor for SDPB

        Returns:
            PolynomialVector with polynomial approximation
        """
        # Sample points using Chebyshev nodes
        n_points = self.poly_degree + 1
        cheb_nodes = np.cos(np.pi * (2 * np.arange(n_points) + 1) / (2 * n_points))
        delta_samples = 0.5 * (delta_max - delta_min) * (cheb_nodes + 1) + delta_min

        # Compute F-vectors at sample points
        if spin == 0:
            F_samples = np.array([self.crossing.build_F_vector(d) for d in delta_samples])
        else:
            F_samples = np.array([self.crossing.build_F_vector_spinning(d, spin)
                                  for d in delta_samples])

        # Fit polynomials in x = Δ - delta_min
        x_samples = delta_samples - delta_min

        polynomials = []
        for i in range(self.n_constraints):
            y_values = F_samples[:, i]
            coeffs = np.polynomial.polynomial.polyfit(
                x_samples, y_values, self.poly_degree
            )
            polynomials.append(coeffs)

        result = PolynomialVector(polynomials=polynomials)

        if include_prefactor:
            # Use R_CROSS from pycftboot conventions (not exp(-1)!)
            # R_CROSS = 3 - 2*sqrt(2) ≈ 0.172 is the crossing-symmetric radial coordinate
            # This is critical for matching pycftboot/SDPB results
            R_CROSS = 3 - 2 * np.sqrt(2)  # ≈ 0.17157
            result.prefactor_constant = 1.0
            result.prefactor_base = R_CROSS
            # For numerical polynomial fitting, we don't have explicit poles
            # pycftboot uses Zamolodchikov recursion to compute exact poles
            # Without poles, we rely on the polynomial approximation quality
            result.prefactor_poles = []

        return result

    def build_polynomial_matrix_program(
        self,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        delta_max: float = 40.0,
        include_spinning: bool = True,
        use_multiresolution: bool = False,
    ) -> Dict:
        """
        Build the Polynomial Matrix Program for SDPB with El-Showk basis.

        Encodes:
        - Normalization: α·F_id = 1
        - First scalar positivity: α·F_ε ≥ 0
        - Scalar gap positivity: α·F_Δ ≥ 0 for all Δ ≥ Δε'
        - Spinning positivity: α·F_{Δ,ℓ} ≥ 0 for all spins ℓ and Δ ≥ unitarity bound

        Args:
            delta_epsilon: First scalar dimension
            delta_epsilon_prime: Gap to second scalar
            delta_max: Maximum dimension for polynomial fitting
            include_spinning: Include spinning operators
            use_multiresolution: Use T1-T5 style discretization for spins

        Returns:
            Dictionary with PMP data in SDPB JSON format
        """
        # Get F-vectors for identity and first scalar
        F_id = self.crossing.build_F_vector(0)
        F_eps = self.crossing.build_F_vector(delta_epsilon)

        # Build PMP structure
        pmp = {
            "objective": self._format_vector(np.zeros(self.n_constraints)),
            "normalization": self._format_vector(F_id),
            "PositiveMatrixWithPrefactorArray": []
        }

        # Add discrete constraint for first scalar
        # Each optimization variable i contributes a constant polynomial [F_eps[i]]
        pmp["PositiveMatrixWithPrefactorArray"].append({
            "DampedRational": {
                "constant": "1",
                "base": "0.5",  # Must be in (0, 1) for SDPB
                "poles": []
            },
            "polynomials": [[[
                [v] for v in self._format_vector(F_eps)
            ]]]
        })

        # Add polynomial constraint for scalar gap
        F_poly_scalar = self.approximate_F_as_polynomial(
            delta_epsilon_prime, delta_max, spin=0
        )
        poly_matrix = self._build_polynomial_matrix(F_poly_scalar)
        pmp["PositiveMatrixWithPrefactorArray"].append({
            "DampedRational": {
                "constant": str(F_poly_scalar.prefactor_constant),
                "base": str(F_poly_scalar.prefactor_base),
                "poles": [str(p) for p in F_poly_scalar.prefactor_poles]
            },
            "polynomials": poly_matrix
        })

        # Add polynomial constraints for spinning operators
        if include_spinning and self.max_spin >= 2:
            for ell in range(2, self.max_spin + 1, 2):
                delta_min_spin = self.unitarity_bound(ell)
                F_poly_spin = self.approximate_F_as_polynomial(
                    delta_min_spin, delta_max, spin=ell
                )
                poly_matrix_spin = self._build_polynomial_matrix(F_poly_spin)
                pmp["PositiveMatrixWithPrefactorArray"].append({
                    "DampedRational": {
                        "constant": str(F_poly_spin.prefactor_constant),
                        "base": str(F_poly_spin.prefactor_base),
                        "poles": [str(p) for p in F_poly_spin.prefactor_poles]
                    },
                    "polynomials": poly_matrix_spin
                })

        return pmp

    def _format_vector(self, vec: np.ndarray) -> List[str]:
        """Format numpy array as list of string numbers for JSON."""
        return [f"{x:.15e}" for x in vec]

    def _build_polynomial_matrix(self, F_poly: PolynomialVector) -> List:
        """
        Build polynomial matrix structure for SDPB.

        SDPB PMP format for a 1x1 matrix with n optimization variables:
            polynomials[row][col] = [poly_var_0, poly_var_1, ...]
        where each poly_var_i = [coeff_deg0, coeff_deg1, ...] is the
        polynomial contributed by optimization variable i.
        """
        poly_list = []
        for i in range(self.n_constraints):
            coeffs = F_poly.polynomials[i]
            poly_list.append([f"{c:.15e}" for c in coeffs])

        # 1x1 matrix: polynomials[0][0] = list of per-variable polynomials
        return [[poly_list]]


class SymbolicPolynomialApproximator:
    """
    Polynomial approximator using pycftboot's symbolic conformal blocks.

    This is the "correct" approach that produces accurate bootstrap bounds.
    Instead of fitting polynomials numerically to sampled F-vectors, this class
    uses pycftboot's Zamolodchikov recursion to compute F-vectors as true
    symbolic polynomials in delta with explicit poles.

    The key advantages:
    1. Exact polynomial structure (not approximated)
    2. Proper pole handling with damped rational prefactor
    3. Orthogonal bilinear basis for numerical stability
    4. Matches pycftboot/SDPB reference implementation

    This should produce bounds matching El-Showk et al. (2012) Figure 6.
    """

    def __init__(
        self,
        dim: float = 3.0,
        k_max: int = 20,
        l_max: int = 50,
        m_max: int = 10,
        n_max: int = 10,
    ):
        """
        Initialize the symbolic polynomial approximator.

        Args:
            dim: Spatial dimension
            k_max: Recursion depth (controls accuracy)
            l_max: Maximum spin for spinning operators
            m_max: Maximum 'a' derivatives
            n_max: Maximum 'b' derivatives
        """
        if not HAVE_PYCFTBOOT_BRIDGE:
            raise RuntimeError(
                "pycftboot bridge not available. Install symengine and ensure "
                "reference_implementations/pycftboot is present."
            )

        self.dim = dim
        self.k_max = k_max
        self.l_max = l_max
        self.m_max = m_max
        self.n_max = n_max

        # Build conformal block table
        self.block_table = PycftbootBlockTable(
            dim=dim,
            k_max=k_max,
            l_max=l_max,
            m_max=m_max,
            n_max=n_max,
            delta_12=0.0,  # Identical external scalars
            delta_34=0.0
        )
        self._table_built = False

    def build_table(self, verbose: bool = True) -> bool:
        """Build the conformal block table."""
        if self._table_built:
            return True
        self._table_built = self.block_table.build(verbose=verbose)
        return self._table_built

    def unitarity_bound(self, spin: int) -> float:
        """Unitarity bound for given spin in d dimensions."""
        if spin == 0:
            return (self.dim - 2) / 2  # 0.5 for d=3
        return spin + self.dim - 2  # spin + 1 for d=3

    def get_polynomial_vectors(self) -> List[SymbolicPolynomialVector]:
        """Get symbolic polynomial vectors for all spins."""
        if not self._table_built:
            self.build_table()
        return self.block_table.get_polynomial_vectors()

    def build_pmp_for_sdpb(
        self,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        include_spinning: bool = True,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Build Polynomial Matrix Program for SDPB.

        This generates the PMP in the format expected by SDPB, with:
        - Proper polynomial structure from Zamolodchikov recursion
        - Damped rational prefactors with conformal block poles
        - Orthogonal bilinear basis

        Args:
            delta_epsilon: First scalar dimension
            delta_epsilon_prime: Gap to second scalar (what we're bounding)
            include_spinning: Include spin-2, 4, ... operators
            output_dir: Optional directory to write JSON files
            verbose: Print progress

        Returns:
            Dictionary with PMP data (also writes to output_dir if specified)
        """
        if not self._table_built:
            self.build_table(verbose=verbose)

        vectors = self.get_polynomial_vectors()

        if verbose:
            print(f"Building PMP for SDPB:")
            print(f"  Δε = {delta_epsilon}, Δε' gap = {delta_epsilon_prime}")
            print(f"  {len(vectors)} spin channels available")

        # Define bounds: scalars have gap at delta_epsilon_prime,
        # spinning operators at unitarity bound
        bounds = {0: delta_epsilon_prime}
        for spin in range(2, self.l_max + 1, 2):
            bounds[spin] = self.unitarity_bound(spin)

        # Number of constraint components
        n_constraints = len(vectors[0].vector) if vectors else 0

        # Normalization vector: F_id (identity contribution)
        # For scalar: evaluate at delta=0
        if vectors:
            F_id = vectors[0].evaluate_polynomial_only(0.0)
        else:
            F_id = np.zeros(n_constraints)

        # Objective: zeros for feasibility problem
        objective = [0.0] * n_constraints

        # Build PMP structure
        pmp = {
            "objective": self._format_vector(objective),
            "normalization": self._format_vector(F_id),
            "PositiveMatrixWithPrefactorArray": []
        }

        # Add constraint for first scalar (at delta_epsilon, discrete)
        if vectors:
            F_eps = vectors[0].evaluate_polynomial_only(delta_epsilon)
            pmp["PositiveMatrixWithPrefactorArray"].append({
                "DampedRational": {
                    "constant": "1",
                    "base": "1",
                    "poles": []
                },
                "polynomials": [[[self._format_vector(F_eps)]]]
            })

        # Add polynomial constraints for each spin channel
        for vec in vectors:
            spin = vec.spin
            if spin > 0 and not include_spinning:
                continue

            delta_min = bounds.get(spin, self.unitarity_bound(spin))

            # Get poles shifted by delta_min
            shifted_poles = [p - delta_min for p in vec.poles]

            # Build polynomial matrix with proper prefactor
            poly_matrix = self._build_symbolic_polynomial_matrix(vec, delta_min)

            pmp["PositiveMatrixWithPrefactorArray"].append({
                "DampedRational": {
                    "constant": "1",
                    "base": str(R_CROSS),
                    "poles": [str(p) for p in shifted_poles]
                },
                "polynomials": poly_matrix
            })

        # Write to files if output_dir specified
        if output_dir is not None:
            self._write_pmp_json(pmp, output_dir, verbose=verbose)

        return pmp

    def _format_vector(self, vec) -> List[str]:
        """Format vector as list of string numbers."""
        if isinstance(vec, np.ndarray):
            return [f"{x:.15e}" for x in vec]
        return [f"{x:.15e}" for x in vec]

    def _build_symbolic_polynomial_matrix(
        self,
        poly_vec: SymbolicPolynomialVector,
        delta_min: float
    ) -> List:
        """
        Build polynomial matrix from symbolic polynomial vector.

        The matrix entry is the polynomial shifted so x = Δ - delta_min >= 0.
        """
        n_constraints = len(poly_vec.vector)

        # Get maximum degree
        max_deg = poly_vec.max_degree + 1

        # Build matrix entry: for each degree, collect coefficients across constraints
        matrix_entry = []
        for d in range(max_deg):
            coeff_vec = []
            for i in range(n_constraints):
                # Get coefficient at degree d for constraint i
                # This requires evaluating the polynomial structure
                try:
                    from polynomial_bootstrap import coefficients_from_polynomial
                    coeffs = coefficients_from_polynomial(poly_vec.vector[i])
                    if d < len(coeffs):
                        coeff_vec.append(f"{coeffs[d]:.15e}")
                    else:
                        coeff_vec.append("0")
                except Exception:
                    coeff_vec.append("0")
            matrix_entry.append(coeff_vec)

        return [[[matrix_entry]]]

    def _write_pmp_json(self, pmp: Dict, output_dir: str, verbose: bool = True):
        """Write PMP to JSON files for SDPB."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Write main PMP file
        with open(output_path / "pmp.json", 'w') as f:
            json.dump(pmp, f, indent=2)

        if verbose:
            print(f"  Wrote PMP to {output_dir}")

    def compute_bilinear_basis(
        self,
        spin: int,
        delta_min: float,
        max_degree: int
    ) -> BilinearBasis:
        """
        Compute orthogonal bilinear basis for a spin channel.

        This is needed for high-precision SDPB runs.
        """
        if not HAVE_POLYNOMIAL_INFRASTRUCTURE:
            raise RuntimeError("polynomial_bootstrap not available")

        vec = self.block_table.get_spin_vector(spin)
        if vec is None:
            raise ValueError(f"No vector for spin {spin}")

        return BilinearBasis(
            poles=vec.poles,
            delta_min=delta_min,
            max_degree=max_degree
        )


class SDPBSolver:
    """
    SDPB-based solver for conformal bootstrap bounds.

    This class provides a high-level interface to SDPB for computing
    rigorous bounds on operator dimensions.

    Supports multiple execution modes:
    - BINARY: Direct binary execution (SDPB in PATH)
    - DOCKER: Docker container (for local development)
    - SINGULARITY: Singularity container (for HPC clusters like Harvard FASRC)
    """

    def __init__(self, config: Optional[SDPBConfig] = None):
        """
        Initialize the SDPB solver.

        Args:
            config: SDPB configuration (uses defaults if None)
        """
        self.config = config or SDPBConfig()

        # Auto-detect execution mode if not specified
        self._execution_mode = self._detect_execution_mode()

        # Check if SDPB is available
        self._sdpb_available = self._execution_mode is not None

        if not self._sdpb_available:
            warnings.warn(
                "SDPB not found. Install SDPB from https://github.com/davidsd/sdpb\n"
                "Docker: docker pull bootstrapcollaboration/sdpb:master\n"
                "The solver will fall back to CVXPY if available."
            )
        else:
            mode_name = self._execution_mode.name if self._execution_mode else "NONE"
            # Only print in non-silent mode
            if self.config.verbosity != "silent":
                print(f"SDPB available via {mode_name}")

    def _detect_execution_mode(self) -> Optional[SDPBExecutionMode]:
        """Auto-detect the best available execution mode."""
        # If user specified a mode, try that first
        if self.config.execution_mode is not None:
            if self._check_mode_available(self.config.execution_mode):
                return self.config.execution_mode
            else:
                warnings.warn(
                    f"Requested execution mode {self.config.execution_mode.name} "
                    f"is not available, trying alternatives..."
                )

        # Try modes in order of preference
        for mode in [SDPBExecutionMode.BINARY, SDPBExecutionMode.DOCKER, SDPBExecutionMode.SINGULARITY]:
            if self._check_mode_available(mode):
                return mode

        return None

    def _check_mode_available(self, mode: SDPBExecutionMode) -> bool:
        """Check if a specific execution mode is available."""
        if mode == SDPBExecutionMode.BINARY:
            return self._check_binary_available()
        elif mode == SDPBExecutionMode.DOCKER:
            return self._check_docker_available()
        elif mode == SDPBExecutionMode.SINGULARITY:
            return self._check_singularity_available()
        return False

    def _check_binary_available(self) -> bool:
        """Check if SDPB binary is available in PATH."""
        try:
            result = subprocess.run(
                [self.config.sdpb_path, "--help"],
                capture_output=True,
                timeout=5
            )
            sdpb_ok = result.returncode == 0 or b"SDPB" in result.stdout or b"SDPB" in result.stderr

            result = subprocess.run(
                [self.config.pmp2sdp_path, "--help"],
                capture_output=True,
                timeout=5
            )
            pmp2sdp_ok = result.returncode == 0 or b"pmp2sdp" in result.stdout or b"pmp2sdp" in result.stderr

            return sdpb_ok and pmp2sdp_ok
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _check_docker_available(self) -> bool:
        """Check if SDPB Docker image is available."""
        try:
            # Check if docker is available
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return False

            # Check if SDPB image exists
            image_name = self.config.docker.image.split(":")[0]
            for line in result.stdout.splitlines():
                if image_name in line:
                    return True

            return False
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _check_singularity_available(self) -> bool:
        """Check if SDPB Singularity image is available."""
        try:
            # Expand environment variables in path
            image_path = os.path.expandvars(self.config.singularity.image_path)
            image_path = os.path.expanduser(image_path)

            # Check if singularity command exists
            result = subprocess.run(
                ["singularity", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return False

            # Check if image file exists
            return os.path.exists(image_path)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _run_command(
        self,
        cmd: List[str],
        work_dir: Path,
        timeout: int = 3600,
        use_mpi: bool = False
    ) -> subprocess.CompletedProcess:
        """
        Run a command using the appropriate execution mode.

        Args:
            cmd: Command and arguments to run (e.g., ["pmp2sdp", "--precision=400", ...])
            work_dir: Working directory (mounted as volume for containers)
            timeout: Command timeout in seconds
            use_mpi: Whether to wrap with MPI (mpirun or srun)

        Returns:
            CompletedProcess result
        """
        if self._execution_mode == SDPBExecutionMode.BINARY:
            return self._run_binary(cmd, work_dir, timeout, use_mpi)
        elif self._execution_mode == SDPBExecutionMode.DOCKER:
            return self._run_docker(cmd, work_dir, timeout, use_mpi)
        elif self._execution_mode == SDPBExecutionMode.SINGULARITY:
            return self._run_singularity(cmd, work_dir, timeout, use_mpi)
        else:
            raise RuntimeError("No SDPB execution mode available")

    def _run_binary(
        self,
        cmd: List[str],
        work_dir: Path,
        timeout: int,
        use_mpi: bool
    ) -> subprocess.CompletedProcess:
        """Run command as direct binary."""
        if use_mpi:
            full_cmd = ["mpirun", "-n", str(self.config.num_threads)] + cmd
        else:
            full_cmd = cmd

        return subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir)
        )

    def _run_docker(
        self,
        cmd: List[str],
        work_dir: Path,
        timeout: int,
        use_mpi: bool
    ) -> subprocess.CompletedProcess:
        """Run command in Docker container."""
        # Build docker command
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{work_dir}:/data",
            "-w", "/data",
        ]

        # Add extra options
        docker_cmd.extend(self.config.docker.extra_options)

        # Add image name
        docker_cmd.append(self.config.docker.image)

        # Add MPI wrapper if needed
        if use_mpi:
            docker_cmd.extend(["mpirun", "-n", str(self.config.num_threads)])

        # Translate paths in command to container paths
        translated_cmd = self._translate_paths_for_container(cmd, work_dir, "/data")
        docker_cmd.extend(translated_cmd)

        return subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

    def _run_singularity(
        self,
        cmd: List[str],
        work_dir: Path,
        timeout: int,
        use_mpi: bool
    ) -> subprocess.CompletedProcess:
        """Run command in Singularity container (for HPC clusters)."""
        # Expand image path
        image_path = os.path.expandvars(self.config.singularity.image_path)
        image_path = os.path.expanduser(image_path)

        # Build command based on MPI mode
        if use_mpi and self.config.singularity.use_srun:
            # FASRC hybrid model: srun wraps singularity
            full_cmd = [
                "srun", "-n", str(self.config.num_threads),
                f"--mpi={self.config.singularity.mpi_type}",
                "singularity", "exec",
            ]
        elif use_mpi:
            # Standard mpirun wrapping singularity
            full_cmd = [
                "mpirun", "-n", str(self.config.num_threads),
                "singularity", "exec",
            ]
        else:
            full_cmd = ["singularity", "exec"]

        # Add extra options
        full_cmd.extend(self.config.singularity.extra_options)

        # Bind work directory so all MPI ranks can access it
        full_cmd.extend(["--bind", str(work_dir)])

        # Add image and command
        full_cmd.append(image_path)
        full_cmd.extend(cmd)

        return subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir)
        )

    def _translate_paths_for_container(
        self,
        cmd: List[str],
        host_dir: Path,
        container_dir: str
    ) -> List[str]:
        """Translate host paths to container paths in command arguments."""
        host_dir_str = str(host_dir)
        translated = []
        for arg in cmd:
            # Handle --option=path format
            if "=" in arg:
                parts = arg.split("=", 1)
                if host_dir_str in parts[1]:
                    parts[1] = parts[1].replace(host_dir_str, container_dir)
                translated.append("=".join(parts))
            elif host_dir_str in arg:
                translated.append(arg.replace(host_dir_str, container_dir))
            else:
                translated.append(arg)
        return translated

    def _check_sdpb_available(self) -> bool:
        """Check if SDPB executables are available (legacy method for compatibility)."""
        return self._execution_mode is not None

    @property
    def is_available(self) -> bool:
        """Check if SDPB solver is available."""
        return self._sdpb_available

    def is_excluded_sdpb(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        max_deriv: int = 21,
        poly_degree: int = 20,
        delta_max: float = 40.0,
        approx: Optional[Union["PolynomialApproximator", "ElShowkPolynomialApproximator"]] = None,
    ) -> Tuple[bool, Dict]:
        """
        Check if a point is excluded using SDPB.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: First scalar dimension
            delta_epsilon_prime: Gap to second scalar (what we're testing)
            max_deriv: Maximum derivative order
            poly_degree: Polynomial approximation degree
            delta_max: Maximum dimension for fitting
            approx: Optional pre-built polynomial approximator (e.g., ElShowkPolynomialApproximator).
                    If None, uses the basic PolynomialApproximator.

        Returns:
            Tuple of (is_excluded, solver_info)
        """
        if not self._sdpb_available:
            raise RuntimeError("SDPB is not available. Install from https://github.com/davidsd/sdpb")

        # Build polynomial matrix program (use provided approx or create default)
        if approx is None:
            approx = PolynomialApproximator(delta_sigma, max_deriv, poly_degree)
        pmp = approx.build_polynomial_matrix_program(
            delta_epsilon, delta_epsilon_prime, delta_max
        )

        # Create temporary directory for SDPB files on shared filesystem.
        # /tmp is node-local on FASRC; srun dispatches sdpb across nodes
        # so the work dir must be visible to all nodes.
        if self.config.work_dir:
            base_dir = self.config.work_dir
        else:
            # Try shared paths: /scratch, then this script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            for candidate in ["/scratch", script_dir]:
                if os.path.isdir(candidate):
                    base_dir = candidate
                    break
            else:
                base_dir = "/tmp"
        work_dir = tempfile.mkdtemp(prefix="sdpb_", dir=base_dir)
        work_path = Path(work_dir)

        try:
            # Write PMP to JSON file
            pmp_file = work_path / "pmp.json"
            with open(pmp_file, 'w') as f:
                json.dump(pmp, f, indent=2)

            # Define output directories
            sdp_dir = work_path / "sdp"
            out_dir = work_path / "out"
            checkpoint_dir = work_path / "ck"

            # Convert PMP to SDP format using pmp2sdp
            pmp2sdp_cmd = [
                "pmp2sdp",
                f"--precision={self.config.precision}",
                f"--input={pmp_file}",
                f"--output={sdp_dir}",
                f"--verbosity={self.config.verbosity}"
            ]

            result = self._run_command(pmp2sdp_cmd, work_path, timeout=300, use_mpi=False)

            if result.returncode != 0:
                raise RuntimeError(f"pmp2sdp failed: {result.stderr}")

            # Run SDPB
            sdpb_cmd = [
                "sdpb",
                f"--precision={self.config.precision}",
                f"--sdpDir={sdp_dir}",
                f"--outDir={out_dir}",
                f"--checkpointDir={checkpoint_dir}",
                f"--maxIterations={self.config.max_iterations}",
                f"--dualityGapThreshold={self.config.duality_gap_threshold}",
                f"--primalErrorThreshold={self.config.primal_error_threshold}",
                f"--dualErrorThreshold={self.config.dual_error_threshold}",
                f"--verbosity={self.config.verbosity}"
            ]

            # Only use MPI if num_threads > 1
            use_mpi = self.config.num_threads > 1
            result = self._run_command(sdpb_cmd, work_path, timeout=3600, use_mpi=use_mpi)

            if result.returncode != 0:
                raise RuntimeError(
                    f"sdpb failed (rc={result.returncode}):\n"
                    f"stderr: {result.stderr}\n"
                    f"stdout: {result.stdout}"
                )

            # Parse output
            out_file = out_dir / "out.txt"
            if out_file.exists():
                with open(out_file, 'r') as f:
                    output = f.read()

                # Check termination reason
                is_excluded = "primalFeasible = true" in output and "dualFeasible = true" in output

                solver_info = {
                    "status": "optimal" if is_excluded else "infeasible",
                    "output": output,
                    "sdpb_return_code": result.returncode,
                    "execution_mode": self._execution_mode.name if self._execution_mode else "NONE"
                }
            else:
                raise RuntimeError(
                    f"sdpb returned rc=0 but no out.txt found at {out_file}.\n"
                    f"stderr: {result.stderr}\n"
                    f"stdout: {result.stdout}\n"
                    f"work_dir contents: {list(work_path.rglob('*'))}"
                )

            return is_excluded, solver_info

        finally:
            # Clean up temporary files unless debugging
            if not self.config.keep_temp_files:
                shutil.rmtree(work_path, ignore_errors=True)

    def find_bound(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: Optional[float] = None,
        delta_prime_max: float = 8.0,
        tolerance: float = 0.01,
        max_deriv: int = 21,
        poly_degree: int = 20,
        verbose: bool = True,
        approx: Optional[Union["PolynomialApproximator", "ElShowkPolynomialApproximator"]] = None,
    ) -> float:
        """
        Find the upper bound on Δε' using binary search with SDPB.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: First scalar dimension
            delta_prime_min: Minimum Δε' to consider (defaults to delta_epsilon + 0.1)
            delta_prime_max: Maximum Δε' to consider
            tolerance: Binary search tolerance
            max_deriv: Maximum derivative order
            poly_degree: Polynomial approximation degree
            verbose: Print progress
            approx: Optional pre-built polynomial approximator (e.g., ElShowkPolynomialApproximator).
                    If None, uses the basic PolynomialApproximator.

        Returns:
            Upper bound on Δε'
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        # Report constraint count based on approximator
        if approx is not None and hasattr(approx, 'n_constraints'):
            n_constraints = approx.n_constraints
        else:
            n_constraints = (max_deriv + 1) // 2

        if verbose:
            print(f"Finding Δε' bound with SDPB")
            print(f"  Δσ = {delta_sigma:.4f}, Δε = {delta_epsilon:.4f}")
            print(f"  Search range: [{delta_prime_min:.2f}, {delta_prime_max:.2f}]")
            print(f"  Constraints: {n_constraints}")
            print(f"  Polynomial degree: {poly_degree}")

        # Check boundary conditions
        if verbose:
            print(f"  Checking Δε' = {delta_prime_min:.2f}...", end=" ", flush=True)

        excluded, _ = self.is_excluded_sdpb(
            delta_sigma, delta_epsilon, delta_prime_min,
            max_deriv, poly_degree, approx=approx
        )

        if verbose:
            print("EXCLUDED" if excluded else "ALLOWED")

        if excluded:
            return delta_prime_min

        if verbose:
            print(f"  Checking Δε' = {delta_prime_max:.2f}...", end=" ", flush=True)

        excluded, _ = self.is_excluded_sdpb(
            delta_sigma, delta_epsilon, delta_prime_max,
            max_deriv, poly_degree, approx=approx
        )

        if verbose:
            print("EXCLUDED" if excluded else "ALLOWED")

        if not excluded:
            return float('inf')

        # Binary search
        lo, hi = delta_prime_min, delta_prime_max
        iteration = 0

        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            iteration += 1

            if verbose:
                print(f"  [{iteration}] Testing Δε' = {mid:.4f}...", end=" ", flush=True)

            excluded, _ = self.is_excluded_sdpb(
                delta_sigma, delta_epsilon, mid,
                max_deriv, poly_degree, approx=approx
            )

            if verbose:
                print("EXCLUDED" if excluded else "ALLOWED")

            if excluded:
                hi = mid
            else:
                lo = mid

        bound = (lo + hi) / 2

        if verbose:
            print(f"  Result: Δε' ≤ {bound:.4f}")

        return bound

    def is_excluded_symbolic(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        k_max: int = 20,
        l_max: int = 50,
        m_max: int = 10,
        n_max: int = 10,
    ) -> Tuple[bool, Dict]:
        """
        Check if a point is excluded using symbolic polynomial approximator.

        This uses pycftboot's Zamolodchikov recursion to compute exact polynomial
        F-vectors, which should produce more accurate bounds than the Chebyshev
        fitting approach.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: First scalar dimension
            delta_epsilon_prime: Gap to second scalar (what we're testing)
            k_max: Recursion depth for conformal blocks
            l_max: Maximum spin
            m_max: Maximum 'a' derivatives
            n_max: Maximum 'b' derivatives

        Returns:
            Tuple of (is_excluded, solver_info)
        """
        if not HAVE_PYCFTBOOT_BRIDGE:
            raise RuntimeError("pycftboot bridge not available")

        if not self._sdpb_available:
            raise RuntimeError("SDPB is not available")

        # Build symbolic polynomial approximator
        approx = SymbolicPolynomialApproximator(
            dim=3.0,  # 3D Ising
            k_max=k_max,
            l_max=l_max,
            m_max=m_max,
            n_max=n_max
        )

        # Create temporary directory on shared filesystem (same as is_excluded_sdpb)
        if self.config.work_dir:
            base_dir = self.config.work_dir
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            for candidate in ["/scratch", script_dir]:
                if os.path.isdir(candidate):
                    base_dir = candidate
                    break
            else:
                base_dir = "/tmp"
        work_dir = tempfile.mkdtemp(prefix="sdpb_symbolic_", dir=base_dir)
        work_path = Path(work_dir)

        try:
            # Build PMP using symbolic polynomials
            pmp = approx.build_pmp_for_sdpb(
                delta_epsilon=delta_epsilon,
                delta_epsilon_prime=delta_epsilon_prime,
                include_spinning=True,
                output_dir=str(work_path),
                verbose=False
            )

            # Convert PMP to SDP using pmp2sdp
            pmp_file = work_path / "pmp.json"
            sdp_dir = work_path / "sdp"

            pmp2sdp_cmd = [
                self.config.pmp2sdp_path,
                f"--precision={self.config.precision}",
                f"--input={pmp_file}",
                f"--output={sdp_dir}",
                f"--verbosity={self.config.verbosity}"
            ]

            result = subprocess.run(
                pmp2sdp_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return False, {"status": "pmp2sdp_failed", "stderr": result.stderr}

            # Run SDPB
            out_dir = work_path / "out"
            checkpoint_dir = work_path / "ck"

            sdpb_cmd = [
                "mpirun", "-n", str(self.config.num_threads),
                self.config.sdpb_path,
                f"--precision={self.config.precision}",
                f"--sdpDir={sdp_dir}",
                f"--outDir={out_dir}",
                f"--checkpointDir={checkpoint_dir}",
                f"--maxIterations={self.config.max_iterations}",
                f"--dualityGapThreshold={self.config.duality_gap_threshold}",
                f"--primalErrorThreshold={self.config.primal_error_threshold}",
                f"--dualErrorThreshold={self.config.dual_error_threshold}",
                f"--verbosity={self.config.verbosity}"
            ]

            result = subprocess.run(
                sdpb_cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )

            # Parse output
            out_file = out_dir / "out.txt"
            if out_file.exists():
                with open(out_file, 'r') as f:
                    output = f.read()

                is_excluded = "primalFeasible = true" in output and "dualFeasible = true" in output

                solver_info = {
                    "status": "optimal" if is_excluded else "infeasible",
                    "output": output,
                    "method": "symbolic_polynomial"
                }
            else:
                solver_info = {
                    "status": "error",
                    "stderr": result.stderr
                }
                is_excluded = False

            return is_excluded, solver_info

        finally:
            if not self.config.keep_temp_files:
                shutil.rmtree(work_path, ignore_errors=True)

    def find_bound_symbolic(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: Optional[float] = None,
        delta_prime_max: float = 8.0,
        tolerance: float = 0.01,
        k_max: int = 20,
        l_max: int = 50,
        m_max: int = 10,
        n_max: int = 10,
        verbose: bool = True
    ) -> float:
        """
        Find upper bound on Δε' using symbolic polynomial method.

        This method uses pycftboot's exact polynomial computation for
        conformal blocks, which should produce bounds matching the
        El-Showk et al. (2012) reference.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: First scalar dimension
            delta_prime_min: Minimum Δε' to consider
            delta_prime_max: Maximum Δε' to consider
            tolerance: Binary search tolerance
            k_max: Recursion depth
            l_max: Maximum spin
            m_max: Maximum 'a' derivatives
            n_max: Maximum 'b' derivatives
            verbose: Print progress

        Returns:
            Upper bound on Δε'
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        if verbose:
            print(f"Finding Δε' bound with SDPB (symbolic polynomial method)")
            print(f"  Δσ = {delta_sigma:.4f}, Δε = {delta_epsilon:.4f}")
            print(f"  Search range: [{delta_prime_min:.2f}, {delta_prime_max:.2f}]")
            print(f"  k_max={k_max}, l_max={l_max}, m_max={m_max}, n_max={n_max}")

        # Check boundary conditions
        if verbose:
            print(f"  Checking Δε' = {delta_prime_min:.2f}...", end=" ", flush=True)

        excluded, _ = self.is_excluded_symbolic(
            delta_sigma, delta_epsilon, delta_prime_min,
            k_max, l_max, m_max, n_max
        )

        if verbose:
            print("EXCLUDED" if excluded else "ALLOWED")

        if excluded:
            return delta_prime_min

        if verbose:
            print(f"  Checking Δε' = {delta_prime_max:.2f}...", end=" ", flush=True)

        excluded, _ = self.is_excluded_symbolic(
            delta_sigma, delta_epsilon, delta_prime_max,
            k_max, l_max, m_max, n_max
        )

        if verbose:
            print("EXCLUDED" if excluded else "ALLOWED")

        if not excluded:
            return float('inf')

        # Binary search
        lo, hi = delta_prime_min, delta_prime_max
        iteration = 0

        while hi - lo > tolerance:
            mid = (lo + hi) / 2
            iteration += 1

            if verbose:
                print(f"  [{iteration}] Testing Δε' = {mid:.4f}...", end=" ", flush=True)

            excluded, _ = self.is_excluded_symbolic(
                delta_sigma, delta_epsilon, mid,
                k_max, l_max, m_max, n_max
            )

            if verbose:
                print("EXCLUDED" if excluded else "ALLOWED")

            if excluded:
                hi = mid
            else:
                lo = mid

        bound = (lo + hi) / 2

        if verbose:
            print(f"  Result: Δε' ≤ {bound:.4f}")
            print(f"  (Reference El-Showk 2012: ~3.8 at Ising point)")

        return bound


class FallbackSDPBSolver:
    """
    Fallback solver using CVXPY when SDPB is not available.

    This provides the same interface as SDPBSolver but uses CVXPY's
    SDP solver, which is less precise but doesn't require external installation.
    """

    def __init__(self, max_deriv: int = 21):
        """Initialize the fallback solver."""
        self.max_deriv = max_deriv
        self.n_constraints = (max_deriv + 1) // 2

        try:
            import cvxpy as cp
            self._has_cvxpy = True
        except ImportError:
            self._has_cvxpy = False
            warnings.warn("CVXPY not available. Install with: pip install cvxpy")

    @property
    def is_available(self) -> bool:
        """Check if fallback solver is available."""
        return self._has_cvxpy

    def is_excluded(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        delta_max: float = 30.0,
        n_samples: int = 150
    ) -> bool:
        """Check if point is excluded using CVXPY SDP."""
        if not self._has_cvxpy:
            raise RuntimeError("CVXPY not available")

        import cvxpy as cp

        cross = TaylorCrossingVector(delta_sigma, self.max_deriv)

        # Build F-vectors
        F_id = cross.build_F_vector(0)
        F_eps = cross.build_F_vector(delta_epsilon)

        # Sample operators above the gap
        deltas = np.linspace(delta_epsilon_prime, delta_max, n_samples)
        F_ops = np.array([cross.build_F_vector(d) for d in deltas])

        # SDP: find α such that α·F_id = 1 and α·F_Δ ≥ 0 for all Δ
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
            return prob.status == cp.OPTIMAL
        except Exception as e:
            warnings.warn(f"SDP solver failed: {e}")
            return False

    def find_bound(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: Optional[float] = None,
        delta_prime_max: float = 8.0,
        tolerance: float = 0.01,
        verbose: bool = True
    ) -> float:
        """Find bound using CVXPY fallback."""
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        if verbose:
            print(f"Finding Δε' bound with CVXPY (fallback)")
            print(f"  Δσ = {delta_sigma:.4f}, Δε = {delta_epsilon:.4f}")

        # Check boundaries
        if self.is_excluded(delta_sigma, delta_epsilon, delta_prime_min):
            return delta_prime_min

        if not self.is_excluded(delta_sigma, delta_epsilon, delta_prime_max):
            return float('inf')

        # Binary search
        lo, hi = delta_prime_min, delta_prime_max

        while hi - lo > tolerance:
            mid = (lo + hi) / 2

            if verbose:
                print(f"  Testing Δε' = {mid:.4f}...", end=" ", flush=True)

            if self.is_excluded(delta_sigma, delta_epsilon, mid):
                if verbose:
                    print("EXCLUDED")
                hi = mid
            else:
                if verbose:
                    print("ALLOWED")
                lo = mid

        return (lo + hi) / 2

    def is_excluded_symbolic(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_epsilon_prime: float,
        delta_max: float = 30.0,
        n_samples: int = 200,
        k_max: int = 15,
        l_max: int = 30,
        m_max: int = 6,
        n_max: int = 6
    ) -> bool:
        """
        Check if point is excluded using symbolic polynomial F-vectors with CVXPY.

        This uses the pycftboot bridge to compute exact polynomial F-vectors,
        then samples them densely for the CVXPY constraint.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: First scalar dimension
            delta_epsilon_prime: Gap to second scalar
            delta_max: Maximum dimension to sample
            n_samples: Number of sample points above gap
            k_max: Recursion depth
            l_max: Maximum spin
            m_max: Max 'a' derivatives
            n_max: Max 'b' derivatives

        Returns:
            True if point is excluded (no allowed functional exists)
        """
        if not self._has_cvxpy:
            raise RuntimeError("CVXPY not available")

        if not HAVE_PYCFTBOOT_BRIDGE:
            warnings.warn("pycftboot bridge not available, using Taylor approach")
            return self.is_excluded(delta_sigma, delta_epsilon, delta_epsilon_prime,
                                   delta_max, n_samples)

        import cvxpy as cp

        # Build symbolic polynomial vectors using pycftboot
        try:
            approx = SymbolicPolynomialApproximator(
                dim=3.0,
                k_max=k_max,
                l_max=l_max,
                m_max=m_max,
                n_max=n_max
            )
            approx.build_table(verbose=False)
            vectors = approx.get_polynomial_vectors()
        except Exception as e:
            warnings.warn(f"Failed to build symbolic vectors: {e}, using Taylor")
            return self.is_excluded(delta_sigma, delta_epsilon, delta_epsilon_prime,
                                   delta_max, n_samples)

        if not vectors:
            warnings.warn("No vectors from symbolic approximator, using Taylor")
            return self.is_excluded(delta_sigma, delta_epsilon, delta_epsilon_prime,
                                   delta_max, n_samples)

        # Get symbols for substitution
        from pycftboot_bridge import _pycftboot_namespace
        delta_ext_sym = _pycftboot_namespace.get('delta_ext')
        delta_sym = _pycftboot_namespace.get('delta')
        RealMPFR = _pycftboot_namespace.get('RealMPFR')

        def evaluate_F_vector(vec, delta_val):
            """Evaluate F-vector with both delta_ext and delta substitutions."""
            result = np.zeros(len(vec.vector))
            for i, poly in enumerate(vec.vector):
                # First substitute delta_ext = delta_sigma
                poly_sub = poly.subs(delta_ext_sym, RealMPFR(str(delta_sigma), 200))
                # Then substitute delta = delta_val
                val = float(poly_sub.subs(delta_sym, delta_val))
                result[i] = val
            return result

        # Get F-vectors for identity and first scalar
        scalar_vec = vectors[0]  # spin-0 channel
        n_constraints = len(scalar_vec.vector)

        F_id = evaluate_F_vector(scalar_vec, 0.0)
        F_eps = evaluate_F_vector(scalar_vec, delta_epsilon)

        # Sample scalar operators above the gap
        scalar_deltas = np.linspace(delta_epsilon_prime, delta_max, n_samples)
        F_scalar_ops = np.array([evaluate_F_vector(scalar_vec, d)
                                 for d in scalar_deltas])

        # Also sample spinning operators
        F_spin_ops = []
        for vec in vectors[1:]:  # Skip spin-0
            spin = vec.spin
            delta_min_spin = spin + 1  # Unitarity bound for d=3
            spin_deltas = np.linspace(delta_min_spin, delta_max, n_samples // 2)
            for d in spin_deltas:
                try:
                    F_spin_ops.append(evaluate_F_vector(vec, d))
                except:
                    pass

        # Setup CVXPY problem
        alpha = cp.Variable(n_constraints)

        constraints = [
            alpha @ F_id == 1,
            alpha @ F_eps >= 0,
        ]

        # Scalar constraints
        for F_O in F_scalar_ops:
            constraints.append(alpha @ F_O >= 0)

        # Spinning constraints
        for F_O in F_spin_ops:
            constraints.append(alpha @ F_O >= 0)

        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-9)
            return prob.status == cp.OPTIMAL
        except Exception as e:
            warnings.warn(f"CVXPY solver failed: {e}")
            return False

    def find_bound_symbolic(
        self,
        delta_sigma: float,
        delta_epsilon: float,
        delta_prime_min: Optional[float] = None,
        delta_prime_max: float = 8.0,
        tolerance: float = 0.01,
        k_max: int = 15,
        l_max: int = 30,
        m_max: int = 6,
        n_max: int = 6,
        verbose: bool = True
    ) -> float:
        """
        Find bound using symbolic polynomial method with CVXPY.

        This method uses exact polynomial F-vectors from pycftboot,
        which should produce more accurate bounds.

        Args:
            delta_sigma: External operator dimension
            delta_epsilon: First scalar dimension
            delta_prime_min: Minimum gap to test
            delta_prime_max: Maximum gap to test
            tolerance: Binary search tolerance
            k_max: Recursion depth
            l_max: Maximum spin
            m_max: Max 'a' derivatives
            n_max: Max 'b' derivatives
            verbose: Print progress

        Returns:
            Upper bound on Δε'
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        if verbose:
            print(f"Finding Δε' bound with CVXPY (symbolic polynomial method)")
            print(f"  Δσ = {delta_sigma:.4f}, Δε = {delta_epsilon:.4f}")
            print(f"  k_max={k_max}, l_max={l_max}, m_max={m_max}, n_max={n_max}")

        # Check boundaries
        if verbose:
            print(f"  Checking Δε' = {delta_prime_min:.2f}...", end=" ", flush=True)

        excluded = self.is_excluded_symbolic(
            delta_sigma, delta_epsilon, delta_prime_min,
            k_max=k_max, l_max=l_max, m_max=m_max, n_max=n_max
        )

        if verbose:
            print("EXCLUDED" if excluded else "ALLOWED")

        if excluded:
            return delta_prime_min

        if verbose:
            print(f"  Checking Δε' = {delta_prime_max:.2f}...", end=" ", flush=True)

        excluded = self.is_excluded_symbolic(
            delta_sigma, delta_epsilon, delta_prime_max,
            k_max=k_max, l_max=l_max, m_max=m_max, n_max=n_max
        )

        if verbose:
            print("EXCLUDED" if excluded else "ALLOWED")

        if not excluded:
            return float('inf')

        # Binary search
        lo, hi = delta_prime_min, delta_prime_max

        while hi - lo > tolerance:
            mid = (lo + hi) / 2

            if verbose:
                print(f"  Testing Δε' = {mid:.4f}...", end=" ", flush=True)

            excluded = self.is_excluded_symbolic(
                delta_sigma, delta_epsilon, mid,
                k_max=k_max, l_max=l_max, m_max=m_max, n_max=n_max
            )

            if verbose:
                print("EXCLUDED" if excluded else "ALLOWED")

            if excluded:
                hi = mid
            else:
                lo = mid

        bound = (lo + hi) / 2

        if verbose:
            print(f"  Result: Δε' ≤ {bound:.4f}")
            print(f"  (Reference El-Showk 2012: ~3.8 at Ising point)")

        return bound


def get_best_solver(
    config: Optional[SDPBConfig] = None,
    max_deriv: int = 21
) -> Union[SDPBSolver, FallbackSDPBSolver]:
    """
    Get the best available solver.

    Returns SDPB solver if available, otherwise falls back to CVXPY.

    Args:
        config: SDPB configuration
        max_deriv: Maximum derivative order (for fallback)

    Returns:
        Best available solver instance
    """
    sdpb_solver = SDPBSolver(config)

    if sdpb_solver.is_available:
        return sdpb_solver
    else:
        print("SDPB not found, using CVXPY fallback solver")
        return FallbackSDPBSolver(max_deriv)


# =============================================================================
# Convenience functions
# =============================================================================

def compute_bound_with_sdpb(
    delta_sigma: float,
    delta_epsilon: float,
    max_deriv: int = 21,
    tolerance: float = 0.01,
    verbose: bool = True,
    config: Optional[SDPBConfig] = None
) -> float:
    """
    Compute Δε' bound using the best available solver.

    This is the main entry point for computing bootstrap bounds.

    Args:
        delta_sigma: External operator dimension
        delta_epsilon: First scalar dimension
        max_deriv: Maximum derivative order
        tolerance: Binary search tolerance
        verbose: Print progress
        config: SDPB configuration

    Returns:
        Upper bound on Δε'

    Example:
        >>> bound = compute_bound_with_sdpb(0.518, 1.41)
        >>> print(f"Δε' ≤ {bound:.4f}")
    """
    solver = get_best_solver(config, max_deriv)

    if isinstance(solver, SDPBSolver):
        return solver.find_bound(
            delta_sigma, delta_epsilon,
            tolerance=tolerance,
            max_deriv=max_deriv,
            verbose=verbose
        )
    else:
        return solver.find_bound(
            delta_sigma, delta_epsilon,
            tolerance=tolerance,
            verbose=verbose
        )


def compute_bound_symbolic(
    delta_sigma: float,
    delta_epsilon: float,
    tolerance: float = 0.01,
    k_max: int = 15,
    l_max: int = 30,
    m_max: int = 6,
    n_max: int = 6,
    verbose: bool = True,
    config: Optional[SDPBConfig] = None
) -> float:
    """
    Compute Δε' bound using the symbolic polynomial method.

    This is the RECOMMENDED method for accurate bootstrap bounds.
    It uses pycftboot's exact polynomial computation instead of
    numerical approximation.

    Args:
        delta_sigma: External operator dimension
        delta_epsilon: First scalar dimension
        tolerance: Binary search tolerance
        k_max: Recursion depth for conformal blocks
        l_max: Maximum spin
        m_max: Maximum 'a' derivatives
        n_max: Maximum 'b' derivatives
        verbose: Print progress
        config: SDPB configuration

    Returns:
        Upper bound on Δε'

    Example:
        >>> bound = compute_bound_symbolic(0.518, 1.41)
        >>> print(f"Δε' ≤ {bound:.4f}")  # Should be ~3.8
    """
    # Try SDPB first if available
    sdpb_solver = SDPBSolver(config)
    if sdpb_solver.is_available and HAVE_PYCFTBOOT_BRIDGE:
        return sdpb_solver.find_bound_symbolic(
            delta_sigma, delta_epsilon,
            tolerance=tolerance,
            k_max=k_max, l_max=l_max, m_max=m_max, n_max=n_max,
            verbose=verbose
        )

    # Fall back to CVXPY
    fallback = FallbackSDPBSolver()
    if fallback.is_available:
        return fallback.find_bound_symbolic(
            delta_sigma, delta_epsilon,
            tolerance=tolerance,
            k_max=k_max, l_max=l_max, m_max=m_max, n_max=n_max,
            verbose=verbose
        )

    raise RuntimeError("No solver available (need SDPB or CVXPY)")


# =============================================================================
# Testing
# =============================================================================

def test_sdpb_interface():
    """Test the SDPB interface."""
    print("=" * 60)
    print("SDPB Interface Test")
    print("=" * 60)

    # Test polynomial approximation
    print("\n1. Testing polynomial approximation (Chebyshev):")
    approx = PolynomialApproximator(delta_sigma=0.518, max_deriv=11, poly_degree=15)

    F_poly = approx.approximate_F_as_polynomial(1.5, 30.0)
    print(f"   Dimension: {F_poly.dimension}")
    print(f"   Max degree: {F_poly.max_degree}")
    print(f"   First component coeffs (first 5): {F_poly.polynomials[0][:5]}")

    # Test PMP generation
    print("\n2. Testing PMP generation (Chebyshev):")
    pmp = approx.build_polynomial_matrix_program(1.41, 2.0, 30.0)
    print(f"   Objective dimension: {len(pmp['objective'])}")
    print(f"   Normalization dimension: {len(pmp['normalization'])}")
    print(f"   Number of matrix constraints: {len(pmp['PositiveMatrixWithPrefactorArray'])}")

    # Test symbolic polynomial infrastructure
    print("\n3. Testing symbolic polynomial infrastructure:")
    print(f"   Have polynomial infrastructure: {HAVE_POLYNOMIAL_INFRASTRUCTURE}")
    print(f"   Have pycftboot bridge: {HAVE_PYCFTBOOT_BRIDGE}")

    if HAVE_PYCFTBOOT_BRIDGE:
        try:
            sym_approx = SymbolicPolynomialApproximator(
                dim=3.0, k_max=10, l_max=10, m_max=3, n_max=0
            )
            if sym_approx.build_table(verbose=False):
                vectors = sym_approx.get_polynomial_vectors()
                print(f"   Built {len(vectors)} spin channels")
                if vectors:
                    print(f"   Spin-0 vector: {len(vectors[0].vector)} components")
                    print(f"   Spin-0 poles: {len(vectors[0].poles)}")
        except Exception as e:
            print(f"   Error building symbolic table: {e}")

    # Test solver availability
    print("\n4. Checking solver availability:")
    sdpb_solver = SDPBSolver()
    print(f"   SDPB available: {sdpb_solver.is_available}")
    if sdpb_solver._execution_mode:
        print(f"   Execution mode: {sdpb_solver._execution_mode.name}")

    fallback = FallbackSDPBSolver(max_deriv=11)
    print(f"   CVXPY fallback available: {fallback.is_available}")

    # Test bound computation with Chebyshev (original method)
    print("\n5. Computing test bound at Ising point (Chebyshev method):")
    solver = get_best_solver(max_deriv=11)

    if solver.is_available:
        bound = compute_bound_with_sdpb(
            delta_sigma=0.518,
            delta_epsilon=1.41,
            max_deriv=11,
            tolerance=0.1,
            verbose=True
        )
        print(f"\n   Final bound (Chebyshev): Δε' ≤ {bound:.4f}")
        print(f"   Reference (El-Showk 2012): ~3.8")
        print(f"   NOTE: Chebyshev method typically gives ~2.6 (too low)")
    else:
        print("   No solver available for testing")

    # Test symbolic method
    print("\n6. Computing test bound at Ising point (Symbolic method):")
    if HAVE_PYCFTBOOT_BRIDGE and fallback.is_available:
        try:
            bound_sym = fallback.find_bound_symbolic(
                delta_sigma=0.518,
                delta_epsilon=1.41,
                tolerance=0.1,
                k_max=10,
                l_max=20,
                m_max=4,
                n_max=2,
                verbose=True
            )
            print(f"\n   Final bound (Symbolic): Δε' ≤ {bound_sym:.4f}")
            print(f"   Reference (El-Showk 2012): ~3.8")
            print(f"   Improvement: {bound_sym - bound:.4f}" if 'bound' in dir() else "")
        except Exception as e:
            print(f"   Error computing symbolic bound: {e}")
    else:
        print("   Symbolic method not available (need pycftboot + CVXPY)")

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


def test_symbolic_bound_quick():
    """Quick test of the symbolic bound computation."""
    print("=" * 60)
    print("Quick Symbolic Bound Test at Ising Point")
    print("=" * 60)
    print(f"Δσ = 0.518, Δε = 1.41")
    print(f"Expected bound (El-Showk 2012): ~3.8")
    print()

    if not HAVE_PYCFTBOOT_BRIDGE:
        print("ERROR: pycftboot bridge not available")
        return None

    fallback = FallbackSDPBSolver()
    if not fallback.is_available:
        print("ERROR: CVXPY not available")
        return None

    try:
        bound = fallback.find_bound_symbolic(
            delta_sigma=0.518,
            delta_epsilon=1.41,
            tolerance=0.05,
            k_max=12,
            l_max=25,
            m_max=5,
            n_max=3,
            verbose=True
        )
        print(f"\nResult: Δε' ≤ {bound:.4f}")
        return bound
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_sdpb_availability() -> Dict[str, any]:
    """
    Check SDPB availability and return detailed information.

    This function is useful for environment checking (check_env.py).

    Returns:
        Dictionary with:
            - available: bool
            - mode: str (BINARY, DOCKER, SINGULARITY, or NONE)
            - details: str (description of how SDPB will be run)
            - docker_image: str | None (Docker image if DOCKER mode)
            - singularity_image: str | None (Singularity image if SINGULARITY mode)
    """
    config = SDPBConfig()

    result = {
        "available": False,
        "mode": "NONE",
        "details": "SDPB not found",
        "docker_image": None,
        "singularity_image": None,
    }

    # Check binary first
    try:
        proc = subprocess.run(["sdpb", "--help"], capture_output=True, timeout=5)
        if proc.returncode == 0 or b"SDPB" in proc.stdout or b"SDPB" in proc.stderr:
            result["available"] = True
            result["mode"] = "BINARY"
            result["details"] = "SDPB binary found in PATH"
            return result
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Check Docker
    try:
        proc = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode == 0:
            image_name = config.docker.image.split(":")[0]
            for line in proc.stdout.splitlines():
                if image_name in line:
                    result["available"] = True
                    result["mode"] = "DOCKER"
                    result["details"] = f"Docker image: {line}"
                    result["docker_image"] = line.strip()
                    return result
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Check Singularity
    try:
        image_path = os.path.expandvars(config.singularity.image_path)
        image_path = os.path.expanduser(image_path)

        proc = subprocess.run(["singularity", "--version"], capture_output=True, timeout=5)
        if proc.returncode == 0 and os.path.exists(image_path):
            result["available"] = True
            result["mode"] = "SINGULARITY"
            result["details"] = f"Singularity image: {image_path}"
            result["singularity_image"] = image_path
            return result
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--symbolic":
        test_symbolic_bound_quick()
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Quick availability check
        info = check_sdpb_availability()
        print(f"SDPB Available: {info['available']}")
        print(f"Execution Mode: {info['mode']}")
        print(f"Details: {info['details']}")
    else:
        test_sdpb_interface()
