"""
CFT Bootstrap Package

A Python implementation of the conformal bootstrap for 3D conformal field theories,
focused on computing bounds on operator dimensions.

Main Modules
------------
bootstrap_solver
    Core bootstrap solver with conformal blocks and crossing equations.

bootstrap_gap_solver
    Gap-based Δε' bounds (discrete sampling approach).

taylor_conformal_blocks
    High-order derivatives via Taylor series expansion.

spinning_conformal_blocks
    Spinning operator conformal blocks using radial expansion.

sdpb_interface
    Integration with SDPB (external high-precision solver).

polynomial_positivity
    Polynomial positivity constraints via sum-of-squares (SOS) decomposition.
    Provides tighter bounds than discrete sampling.

mixed_correlator_bootstrap
    Mixed correlator bootstrap using multiple four-point functions.
    Provides the strongest constraints for the 3D Ising model.

mixed_correlator_blocks
    F-vectors for mixed correlators (ssss, ssee, eeee).

Example Usage
-------------
>>> from cft_bootstrap import PolynomialPositivitySolver
>>> solver = PolynomialPositivitySolver(delta_sigma=0.518, max_deriv=21, poly_degree=15)
>>> bound = solver.find_delta_epsilon_prime_bound(delta_epsilon=1.41)
>>> print(f"Δε' ≤ {bound:.4f}")

References
----------
- El-Showk et al. (2012): "Solving the 3D Ising Model with the Conformal Bootstrap"
- Simmons-Duffin (2015): "A Semidefinite Program Solver for the Conformal Bootstrap"
"""

# Core solvers
from .bootstrap_solver import (
    ConformalBlock3D,
    CrossingVector,
    BootstrapSolver,
    BootstrapBoundComputer,
)

# Gap-based bounds
from .bootstrap_gap_solver import (
    GapBootstrapSolver,
    DeltaEpsilonPrimeBoundComputer,
)

# Taylor series for high-order derivatives
from .taylor_conformal_blocks import (
    TaylorCrossingVector,
    HighOrderGapBootstrapSolver,
    build_F_vector_taylor,
)

# Polynomial positivity (main new feature)
from .polynomial_positivity import (
    PolynomialFitter,
    PolynomialFVector,
    FittedPolynomial,
    SOSPositivityConstraint,
    PolynomialPositivitySolver,
    PolynomialPositivityGapSolver,
    compare_methods,
)

# SDPB interface
from .sdpb_interface import (
    SDPBConfig,
    SDPBSolver,
    FallbackSDPBSolver,
    PolynomialApproximator,
    get_best_solver,
    compute_bound_with_sdpb,
)

# Mixed correlator bootstrap
from .mixed_correlator_blocks import (
    MixedCrossingVector,
)

from .mixed_correlator_bootstrap import (
    TwoCorrelatorBootstrapSolver,
    MixedCorrelatorBootstrapSolver,
    compare_single_vs_mixed,
)

__version__ = "0.3.0"
__all__ = [
    # Core
    "ConformalBlock3D",
    "CrossingVector",
    "BootstrapSolver",
    "BootstrapBoundComputer",
    # Gap bounds
    "GapBootstrapSolver",
    "DeltaEpsilonPrimeBoundComputer",
    # Taylor series
    "TaylorCrossingVector",
    "HighOrderGapBootstrapSolver",
    "build_F_vector_taylor",
    # Polynomial positivity
    "PolynomialFitter",
    "PolynomialFVector",
    "FittedPolynomial",
    "SOSPositivityConstraint",
    "PolynomialPositivitySolver",
    "PolynomialPositivityGapSolver",
    "compare_methods",
    # SDPB
    "SDPBConfig",
    "SDPBSolver",
    "FallbackSDPBSolver",
    "PolynomialApproximator",
    "get_best_solver",
    "compute_bound_with_sdpb",
    # Mixed correlator bootstrap
    "MixedCrossingVector",
    "TwoCorrelatorBootstrapSolver",
    "MixedCorrelatorBootstrapSolver",
    "compare_single_vs_mixed",
]
