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

    On macOS with Homebrew:
        brew tap davidsd/sdpb
        brew install sdpb

    On Linux:
        Use Docker/Singularity or build from source (see SDPB docs)

Usage:
    >>> solver = SDPBSolver(sdpb_path="/usr/local/bin/sdpb")
    >>> bound = solver.find_bound(delta_sigma=0.518, delta_epsilon=1.41)
"""

import json
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


@dataclass
class SDPBConfig:
    """Configuration for SDPB solver."""

    # Paths to SDPB executables
    sdpb_path: str = "sdpb"
    pmp2sdp_path: str = "pmp2sdp"

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
            # Using e^{-x} ensures convergence of the integral
            result.prefactor_constant = 1.0
            result.prefactor_base = np.exp(-1)
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
        pmp["PositiveMatrixWithPrefactorArray"].append({
            "DampedRational": {
                "constant": "1",
                "base": "1",
                "poles": []
            },
            "polynomials": [[
                [self._format_vector(F_eps)]  # 1x1 matrix with polynomial [[F_ε]]
            ]]
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

        For a single scalar constraint, this is a 1x1 matrix:
        M(x) = [[p(x)]] where p(x) = α·F_{Δ_gap + x}

        The matrix entry is a polynomial whose coefficients depend on α.
        """
        # For scalar bootstrap: 1x1 matrix
        # Entry [0][0] is a polynomial in x with n_constraints coefficients
        # Each coefficient at degree d is: Σᵢ αᵢ * F_poly.polynomials[i][d]

        n_deg = F_poly.max_degree + 1

        # Build matrix where entry [0][0][d] = vector of coefficients for degree d
        # This is: [F_poly.polynomials[0][d], F_poly.polynomials[1][d], ...]
        matrix_entry = []
        for d in range(n_deg):
            # Coefficient at degree d for each constraint
            coeff_vec = []
            for i in range(self.n_constraints):
                if d < len(F_poly.polynomials[i]):
                    coeff_vec.append(f"{F_poly.polynomials[i][d]:.15e}")
                else:
                    coeff_vec.append("0")
            matrix_entry.append(coeff_vec)

        # Wrap in matrix structure: [[[entry]]] for 1x1 matrix
        return [[[matrix_entry]]]


class SDPBSolver:
    """
    SDPB-based solver for conformal bootstrap bounds.

    This class provides a high-level interface to SDPB for computing
    rigorous bounds on operator dimensions.
    """

    def __init__(self, config: Optional[SDPBConfig] = None):
        """
        Initialize the SDPB solver.

        Args:
            config: SDPB configuration (uses defaults if None)
        """
        self.config = config or SDPBConfig()

        # Check if SDPB is available
        self._sdpb_available = self._check_sdpb_available()

        if not self._sdpb_available:
            warnings.warn(
                "SDPB not found. Install SDPB from https://github.com/davidsd/sdpb\n"
                "On macOS: brew tap davidsd/sdpb && brew install sdpb\n"
                "The solver will fall back to CVXPY if available."
            )

    def _check_sdpb_available(self) -> bool:
        """Check if SDPB executables are available."""
        try:
            # Check sdpb
            result = subprocess.run(
                [self.config.sdpb_path, "--help"],
                capture_output=True,
                timeout=5
            )
            sdpb_ok = result.returncode == 0 or b"SDPB" in result.stdout or b"SDPB" in result.stderr

            # Check pmp2sdp
            result = subprocess.run(
                [self.config.pmp2sdp_path, "--help"],
                capture_output=True,
                timeout=5
            )
            pmp2sdp_ok = result.returncode == 0 or b"pmp2sdp" in result.stdout or b"pmp2sdp" in result.stderr

            return sdpb_ok and pmp2sdp_ok
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

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

        Returns:
            Tuple of (is_excluded, solver_info)
        """
        if not self._sdpb_available:
            raise RuntimeError("SDPB is not available. Install from https://github.com/davidsd/sdpb")

        # Build polynomial matrix program
        approx = PolynomialApproximator(delta_sigma, max_deriv, poly_degree)
        pmp = approx.build_polynomial_matrix_program(
            delta_epsilon, delta_epsilon_prime, delta_max
        )

        # Create temporary directory for SDPB files
        work_dir = self.config.work_dir or tempfile.mkdtemp(prefix="sdpb_")
        work_path = Path(work_dir)

        try:
            # Write PMP to JSON file
            pmp_file = work_path / "pmp.json"
            with open(pmp_file, 'w') as f:
                json.dump(pmp, f, indent=2)

            # Convert PMP to SDP format using pmp2sdp
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
                raise RuntimeError(f"pmp2sdp failed: {result.stderr}")

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
                timeout=3600  # 1 hour timeout
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
                    "sdpb_return_code": result.returncode
                }
            else:
                solver_info = {
                    "status": "error",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }
                is_excluded = False

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
        verbose: bool = True
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

        Returns:
            Upper bound on Δε'
        """
        if delta_prime_min is None:
            delta_prime_min = delta_epsilon + 0.1

        delta_prime_min = max(delta_prime_min, delta_epsilon + 0.05)

        if verbose:
            print(f"Finding Δε' bound with SDPB")
            print(f"  Δσ = {delta_sigma:.4f}, Δε = {delta_epsilon:.4f}")
            print(f"  Search range: [{delta_prime_min:.2f}, {delta_prime_max:.2f}]")
            print(f"  Constraints: {(max_deriv + 1) // 2}")
            print(f"  Polynomial degree: {poly_degree}")

        # Check boundary conditions
        if verbose:
            print(f"  Checking Δε' = {delta_prime_min:.2f}...", end=" ", flush=True)

        excluded, _ = self.is_excluded_sdpb(
            delta_sigma, delta_epsilon, delta_prime_min,
            max_deriv, poly_degree
        )

        if verbose:
            print("EXCLUDED" if excluded else "ALLOWED")

        if excluded:
            return delta_prime_min

        if verbose:
            print(f"  Checking Δε' = {delta_prime_max:.2f}...", end=" ", flush=True)

        excluded, _ = self.is_excluded_sdpb(
            delta_sigma, delta_epsilon, delta_prime_max,
            max_deriv, poly_degree
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
                max_deriv, poly_degree
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


# =============================================================================
# Testing
# =============================================================================

def test_sdpb_interface():
    """Test the SDPB interface."""
    print("=" * 60)
    print("SDPB Interface Test")
    print("=" * 60)

    # Test polynomial approximation
    print("\n1. Testing polynomial approximation:")
    approx = PolynomialApproximator(delta_sigma=0.518, max_deriv=11, poly_degree=15)

    F_poly = approx.approximate_F_as_polynomial(1.5, 30.0)
    print(f"   Dimension: {F_poly.dimension}")
    print(f"   Max degree: {F_poly.max_degree}")
    print(f"   First component coeffs (first 5): {F_poly.polynomials[0][:5]}")

    # Test PMP generation
    print("\n2. Testing PMP generation:")
    pmp = approx.build_polynomial_matrix_program(1.41, 2.0, 30.0)
    print(f"   Objective dimension: {len(pmp['objective'])}")
    print(f"   Normalization dimension: {len(pmp['normalization'])}")
    print(f"   Number of matrix constraints: {len(pmp['PositiveMatrixWithPrefactorArray'])}")

    # Test solver availability
    print("\n3. Checking solver availability:")
    sdpb_solver = SDPBSolver()
    print(f"   SDPB available: {sdpb_solver.is_available}")

    fallback = FallbackSDPBSolver(max_deriv=11)
    print(f"   CVXPY fallback available: {fallback.is_available}")

    # Test bound computation
    print("\n4. Computing test bound at Ising point:")
    solver = get_best_solver(max_deriv=11)

    if solver.is_available:
        bound = compute_bound_with_sdpb(
            delta_sigma=0.518,
            delta_epsilon=1.41,
            max_deriv=11,
            tolerance=0.1,
            verbose=True
        )
        print(f"\n   Final bound: Δε' ≤ {bound:.4f}")
        print(f"   Reference (El-Showk 2012): ~3.8")
    else:
        print("   No solver available for testing")

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_sdpb_interface()
