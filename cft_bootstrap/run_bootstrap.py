#!/usr/bin/env python3
"""
Main script to run CFT bootstrap computations.

This script is designed to be run on a cluster with various options:
- Single point evaluation
- Grid scan over delta_sigma
- Parallel execution using multiprocessing or MPI
- SDPB integration for high-precision bounds
- Polynomial positivity constraints (NEW)

Usage:
    # Single point (LP method)
    python run_bootstrap.py --delta-sigma 0.518 --method lp

    # Single point (SDP method with CVXPY)
    python run_bootstrap.py --delta-sigma 0.518 --method sdp

    # Single point (SDPB - high precision)
    python run_bootstrap.py --delta-sigma 0.518 --method sdpb --max-deriv 21

    # Δε' bounds with gap assumption
    python run_bootstrap.py --gap-bound --delta-sigma 0.518 --delta-epsilon 1.41

    # Δε' bounds with polynomial positivity (recommended)
    python run_bootstrap.py --gap-bound --method polynomial --max-deriv 21 --poly-degree 15

    # Hybrid method: polynomial positivity + discrete samples
    python run_bootstrap.py --gap-bound --method hybrid --max-deriv 21

    # Grid scan
    python run_bootstrap.py --grid --sigma-min 0.50 --sigma-max 0.65 --n-points 50

    # Ising plot (Δε' bounds along the boundary)
    python run_bootstrap.py --ising-plot --method polynomial --n-points 25

    # Parallel (local)
    python run_bootstrap.py --grid --parallel --n-workers 8

    # For cluster submission, use the SLURM script
"""

import argparse
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional
import sys

from bootstrap_solver import BootstrapSolver, BootstrapBoundComputer
from taylor_conformal_blocks import HighOrderGapBootstrapSolver
from sdpb_interface import (
    SDPBSolver, SDPBConfig, FallbackSDPBSolver,
    get_best_solver, compute_bound_with_sdpb,
    ElShowkPolynomialApproximator
)
from polynomial_positivity import (
    PolynomialPositivitySolver,
    PolynomialPositivityGapSolver,
    compare_methods
)
from mixed_correlator_bootstrap import (
    TwoCorrelatorBootstrapSolver,
    MixedCorrelatorBootstrapSolver,
    compare_single_vs_mixed
)
from el_showk_basis import (
    ElShowkBootstrapSolver,
    ElShowkCrossingVector,
    count_coefficients,
    get_derivative_indices
)

# Silence the CVXPY warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='bootstrap_solver')


def run_single_point(delta_sigma: float, method: str = 'lp',
                     max_deriv: int = 5, tolerance: float = 0.01,
                     output_file: Optional[str] = None):
    """Run bootstrap for a single delta_sigma value."""
    print(f"Computing bootstrap bound for Δσ = {delta_sigma}")
    print(f"  Method: {method}")
    print(f"  Max derivative order: {max_deriv}")
    print(f"  Tolerance: {tolerance}")

    solver = BootstrapSolver(d=3, max_deriv=max_deriv)

    t0 = time.time()
    bound = solver.find_bound(delta_sigma, method=method, tolerance=tolerance)
    t1 = time.time()

    result = {
        'delta_sigma': delta_sigma,
        'delta_epsilon_bound': bound,
        'method': method,
        'max_derivative_order': max_deriv,
        'tolerance': tolerance,
        'compute_time_seconds': t1 - t0
    }

    print(f"\nResult: Δε ≤ {bound:.6f}")
    print(f"Compute time: {t1-t0:.2f}s")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_file}")

    return result


def run_gap_bound(delta_sigma: float, delta_epsilon: float,
                  method: str = 'sdpb', max_deriv: int = 21,
                  poly_degree: int = 20, tolerance: float = 0.01,
                  sdpb_threads: int = 4,
                  # El-Showk specific parameters
                  max_spin: int = 50,
                  use_multiresolution: bool = False,
                  el_showk_solver: str = 'auto',
                  nmax: Optional[int] = None,
                  # High-precision parameters
                  high_precision: bool = False,
                  precision: int = 100,
                  output_file: Optional[str] = None):
    """
    Compute Δε' bound with gap assumption using various methods.

    This computes the upper bound on the second Z2-even scalar dimension,
    assuming the first scalar is at Δε.

    Args:
        delta_sigma: External operator dimension
        delta_epsilon: First scalar dimension (assumed)
        method: Solver method:
            - 'polynomial': Polynomial positivity with SOS constraints (recommended)
            - 'hybrid': Polynomial + discrete samples for robustness
            - 'two-correlator': Two-correlator bootstrap (ssss + eeee)
            - 'mixed-correlator': Full mixed correlator with matrix SDP
            - 'el-showk': El-Showk (2012) full derivative basis with spinning operators
            - 'sdpb': SDPB solver (requires installation)
            - 'cvxpy': Discrete sampling with CVXPY
            - 'discrete': Alias for 'cvxpy'
        max_deriv: Maximum derivative order
        poly_degree: Polynomial approximation degree
        tolerance: Binary search tolerance
        sdpb_threads: Number of threads for SDPB
        max_spin: Maximum spin for El-Showk spinning operators (default: 50)
        use_multiresolution: Use T1-T5 multi-resolution discretization for El-Showk
        el_showk_solver: Solver backend for El-Showk (auto/scs/ecos/clarabel/mosek)
        nmax: El-Showk nmax parameter (overrides max_deriv//2 if provided)
        high_precision: Use full mpmath arbitrary-precision arithmetic (for nmax=10)
        precision: Decimal places for high-precision mode (default: 100)
        output_file: Output file path

    Returns:
        Dictionary with results
    """
    print(f"Computing Δε' bound with gap assumption")
    print(f"  Δσ = {delta_sigma}")
    print(f"  Δε = {delta_epsilon} (assumed first scalar)")
    print(f"  Method: {method}")
    print(f"  Max derivative order: {max_deriv}")
    print(f"  Number of constraints: {(max_deriv + 1) // 2}")
    if method in ['polynomial', 'hybrid']:
        print(f"  Polynomial degree: {poly_degree}")
    print(f"  Tolerance: {tolerance}")

    t0 = time.time()

    if method == 'polynomial':
        # Polynomial positivity with SOS constraints
        print("  Using polynomial positivity (SOS) constraints")
        solver = PolynomialPositivitySolver(
            delta_sigma=delta_sigma,
            max_deriv=max_deriv,
            poly_degree=poly_degree
        )
        bound = solver.find_delta_epsilon_prime_bound(
            delta_epsilon=delta_epsilon,
            tolerance=tolerance,
            method="polynomial",
            verbose=True
        )
    elif method == 'hybrid':
        # Hybrid: polynomial + discrete samples
        print("  Using hybrid method (polynomial + discrete samples)")
        solver = PolynomialPositivitySolver(
            delta_sigma=delta_sigma,
            max_deriv=max_deriv,
            poly_degree=poly_degree
        )
        bound = solver.find_delta_epsilon_prime_bound(
            delta_epsilon=delta_epsilon,
            tolerance=tolerance,
            method="hybrid",
            verbose=True
        )
    elif method == 'two-correlator':
        # Two-correlator bootstrap (ssss + eeee)
        print("  Using two-correlator bootstrap (ssss + eeee)")
        solver = TwoCorrelatorBootstrapSolver(d=3, max_deriv=max_deriv)
        bound = solver.find_delta_epsilon_prime_bound(
            delta_sigma, delta_epsilon,
            tolerance=tolerance,
            verbose=True
        )
    elif method == 'mixed-correlator':
        # Full mixed correlator bootstrap with matrix SDP
        print("  Using full mixed correlator bootstrap (matrix SDP)")
        solver = MixedCorrelatorBootstrapSolver(d=3, max_deriv=max_deriv)
        bound = solver.find_delta_epsilon_prime_bound(
            delta_sigma, delta_epsilon,
            tolerance=tolerance,
            verbose=True
        )
    elif method == 'el-showk':
        # El-Showk et al. (2012) full derivative basis with spinning operators
        # Use explicit nmax if provided, otherwise derive from max_deriv
        actual_nmax = nmax if nmax is not None else max_deriv // 2
        n_coeffs = count_coefficients(actual_nmax)

        print(f"  Using El-Showk derivative basis:")
        print(f"    nmax = {actual_nmax} ({n_coeffs} coefficients)")
        print(f"    max_spin = {max_spin}")
        print(f"    multiresolution = {use_multiresolution}")
        print(f"    solver = {el_showk_solver}")
        if high_precision:
            print(f"    HIGH-PRECISION MODE: {precision} decimal places")

        solver = ElShowkBootstrapSolver(
            d=3,
            nmax=actual_nmax,
            max_spin=max_spin,
            solver=el_showk_solver,
            high_precision=high_precision,
            precision=precision
        )
        bound = solver.find_delta_epsilon_prime_bound(
            delta_sigma, delta_epsilon,
            tolerance=tolerance,
            include_spinning=(max_spin > 0),
            use_multiresolution=use_multiresolution,
            verbose=True
        )
    elif method == 'el-showk-sdpb':
        # El-Showk basis with SDPB high-precision solver
        actual_nmax = nmax if nmax is not None else max_deriv // 2
        n_coeffs = count_coefficients(actual_nmax)

        print(f"  Using El-Showk derivative basis with SDPB:")
        print(f"    nmax = {actual_nmax} ({n_coeffs} coefficients)")
        print(f"    max_spin = {max_spin}")
        print(f"    poly_degree = {poly_degree}")

        # Build polynomial approximator
        approx = ElShowkPolynomialApproximator(
            delta_sigma=delta_sigma,
            nmax=actual_nmax,
            max_spin=max_spin,
            poly_degree=poly_degree
        )

        # Build PMP and run SDPB
        config = SDPBConfig(num_threads=sdpb_threads)
        solver = get_best_solver(config, max_deriv)

        if isinstance(solver, SDPBSolver):
            print("  Using SDPB solver")
            # Use SDPB's binary search to find the bound
            # Pass the ElShowkPolynomialApproximator so SDPB uses the correct constraint count
            bound = solver.find_bound(
                delta_sigma, delta_epsilon,
                delta_prime_min=delta_epsilon + 0.1,
                delta_prime_max=8.0,
                tolerance=tolerance,
                max_deriv=max_deriv,
                poly_degree=poly_degree,
                verbose=True,
                approx=approx,  # Use El-Showk approximator instead of basic PolynomialApproximator
            )
        else:
            print("  SDPB not available, falling back to El-Showk with CVXPY")
            solver = ElShowkBootstrapSolver(
                d=3,
                nmax=actual_nmax,
                max_spin=max_spin,
                solver=el_showk_solver
            )
            bound = solver.find_delta_epsilon_prime_bound(
                delta_sigma, delta_epsilon,
                tolerance=tolerance,
                include_spinning=(max_spin > 0),
                use_multiresolution=use_multiresolution,
                verbose=True
            )
    elif method == 'sdpb':
        # Try SDPB, fall back to CVXPY if not available
        config = SDPBConfig(num_threads=sdpb_threads)
        solver = get_best_solver(config, max_deriv)

        if isinstance(solver, SDPBSolver):
            print("  Using SDPB solver")
            bound = solver.find_bound(
                delta_sigma, delta_epsilon,
                tolerance=tolerance,
                max_deriv=max_deriv,
                poly_degree=poly_degree,
                verbose=True
            )
        else:
            print("  SDPB not available, using CVXPY fallback")
            bound = solver.find_bound(
                delta_sigma, delta_epsilon,
                tolerance=tolerance,
                verbose=True
            )
    else:
        # Use high-order solver with Taylor series (discrete sampling)
        print("  Using CVXPY with discrete sampling")
        solver = HighOrderGapBootstrapSolver(d=3, max_deriv=max_deriv)
        bound = solver.find_delta_epsilon_prime_bound(
            delta_sigma, delta_epsilon,
            tolerance=tolerance
        )

    t1 = time.time()

    result = {
        'delta_sigma': delta_sigma,
        'delta_epsilon': delta_epsilon,
        'delta_epsilon_prime_bound': bound,
        'method': method,
        'max_derivative_order': max_deriv,
        'n_constraints': (max_deriv + 1) // 2,
        'poly_degree': poly_degree if method in ['polynomial', 'hybrid'] else None,
        'tolerance': tolerance,
        'compute_time_seconds': t1 - t0
    }

    print(f"\nResult: Δε' ≤ {bound:.6f}")
    print(f"Compute time: {t1-t0:.2f}s")
    print(f"Reference (El-Showk 2012 at Ising point): ~3.8")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {output_file}")

    return result


def run_grid(sigma_min: float, sigma_max: float, n_points: int,
             method: str = 'lp', max_deriv: int = 5, tolerance: float = 0.01,
             parallel: bool = False, n_workers: int = 4,
             output_file: Optional[str] = None):
    """Run bootstrap over a grid of delta_sigma values."""
    delta_sigmas = np.linspace(sigma_min, sigma_max, n_points)

    print(f"Computing bootstrap bounds for {n_points} points")
    print(f"  Δσ range: [{sigma_min}, {sigma_max}]")
    print(f"  Method: {method}")
    print(f"  Max derivative order: {max_deriv}")
    print(f"  Parallel: {parallel} (workers: {n_workers})")

    t0 = time.time()

    if parallel:
        results = run_parallel(delta_sigmas, method, max_deriv, tolerance, n_workers)
    else:
        computer = BootstrapBoundComputer(d=3, max_deriv=max_deriv)
        results = computer.compute_bound_grid(delta_sigmas, method=method, tolerance=tolerance)

    t1 = time.time()

    output = {
        'delta_sigma_values': results[:, 0].tolist(),
        'delta_epsilon_bounds': results[:, 1].tolist(),
        'method': method,
        'max_derivative_order': max_deriv,
        'tolerance': tolerance,
        'total_compute_time_seconds': t1 - t0,
        'n_points': n_points
    }

    print(f"\nTotal compute time: {t1-t0:.1f}s ({(t1-t0)/n_points:.2f}s per point)")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {output_file}")

    # Also save as numpy for easy plotting
    np_file = output_file.replace('.json', '.npy') if output_file else 'bootstrap_results.npy'
    np.save(np_file, results)
    print(f"Saved numpy array to {np_file}")

    return results


def run_parallel(delta_sigmas: np.ndarray, method: str, max_deriv: int,
                 tolerance: float, n_workers: int) -> np.ndarray:
    """Run grid computation in parallel using multiprocessing."""
    from multiprocessing import Pool
    from functools import partial

    def compute_single(delta_sigma, method, max_deriv, tolerance):
        solver = BootstrapSolver(d=3, max_deriv=max_deriv)
        bound = solver.find_bound(delta_sigma, method=method, tolerance=tolerance)
        return [delta_sigma, bound]

    func = partial(compute_single, method=method, max_deriv=max_deriv, tolerance=tolerance)

    with Pool(n_workers) as pool:
        results = pool.map(func, delta_sigmas)

    return np.array(results)


def run_array_job(job_index: int, n_jobs: int, sigma_min: float, sigma_max: float,
                  method: str = 'lp', max_deriv: int = 5, tolerance: float = 0.01,
                  poly_degree: int = 15, output_dir: str = 'results',
                  gap_bound: bool = False,
                  # El-Showk specific parameters
                  max_spin: int = 50, use_multiresolution: bool = False,
                  el_showk_solver: str = 'auto', nmax: Optional[int] = None,
                  # High-precision parameters
                  high_precision: bool = False, precision: int = 100):
    """
    Run a single point as part of an array job (for SLURM).

    Each job computes one delta_sigma value. In gap_bound mode, this performs
    the two-stage El-Showk (2012) Figure 6 computation:
      1. Find the Δε boundary at this Δσ
      2. Compute Δε' bound with Δε fixed to the boundary value

    Args:
        job_index: SLURM array task ID (0-indexed)
        n_jobs: Total number of jobs in the array
        sigma_min, sigma_max: Range of Δσ values
        method: Solver method
        max_deriv: Maximum derivative order
        tolerance: Binary search tolerance
        poly_degree: Polynomial degree (for polynomial/hybrid methods)
        output_dir: Output directory for results
        gap_bound: If True, compute Δε' gap bounds (Figure 6 mode)
        max_spin: Maximum spin for El-Showk (default: 50)
        use_multiresolution: Enable T1-T5 discretization
        el_showk_solver: Solver backend for El-Showk
        nmax: El-Showk nmax parameter (overrides max_deriv//2)
        high_precision: Use full mpmath arbitrary-precision (REQUIRED for nmax=10)
        precision: Decimal places for high-precision mode

    Returns:
        Dictionary with results
    """
    delta_sigmas = np.linspace(sigma_min, sigma_max, n_jobs)
    delta_sigma = delta_sigmas[job_index]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if gap_bound:
        # Two-stage Figure 6 computation:
        # Stage 1: Find Δε boundary at this Δσ
        # Stage 2: Compute Δε' bound with gap assumption
        output_file = f"{output_dir}/gap_bound_{job_index:04d}.json"

        print("=" * 60)
        print(f"Array Job {job_index}/{n_jobs}: Δσ = {delta_sigma:.6f}")
        print(f"Computing Figure 6 style Δε' bound (two-stage)")
        print("=" * 60)

        # Stage 1: Find Δε boundary using El-Showk solver with spinning operators
        # This is the single-correlator bound on Δε, now with proper spinning!
        print("\n--- Stage 1: Finding Δε boundary (with spinning operators) ---")
        t0 = time.time()

        # Use ElShowkBootstrapSolver for Stage 1 (includes spinning operators)
        # This gives a more accurate Δε boundary
        stage1_nmax = nmax if nmax is not None else max_deriv // 2
        stage1_solver = ElShowkBootstrapSolver(
            d=3,
            nmax=stage1_nmax,
            max_spin=max_spin,
            solver=el_showk_solver,
            high_precision=high_precision,
            precision=precision
        )
        delta_epsilon = stage1_solver.find_delta_epsilon_bound(
            delta_sigma,
            delta_min=0.5,
            delta_max=3.0,
            tolerance=tolerance,
            include_spinning=(max_spin > 0),
            use_multiresolution=use_multiresolution,
            verbose=True
        )
        print(f"  Found Δε boundary: {delta_epsilon:.6f}")

        t1 = time.time()
        print(f"Stage 1 complete: Δε = {delta_epsilon:.6f} ({t1-t0:.1f}s)")

        # Stage 2: Compute Δε' bound with gap assumption
        print(f"\n--- Stage 2: Computing Δε' bound (Δε = {delta_epsilon:.6f}) ---")
        t2 = time.time()

        result = run_gap_bound(
            delta_sigma=delta_sigma,
            delta_epsilon=delta_epsilon,
            method=method,
            max_deriv=max_deriv,
            poly_degree=poly_degree,
            tolerance=tolerance,
            max_spin=max_spin,
            use_multiresolution=use_multiresolution,
            el_showk_solver=el_showk_solver,
            nmax=nmax,
            high_precision=high_precision,
            precision=precision,
            output_file=None  # We'll save combined result
        )

        t3 = time.time()

        # Combine results
        combined_result = {
            'job_index': job_index,
            'delta_sigma': delta_sigma,
            'delta_epsilon_boundary': delta_epsilon,
            'delta_epsilon_prime_bound': result['delta_epsilon_prime_bound'],
            'method': method,
            'max_deriv': max_deriv,
            'nmax': nmax if nmax is not None else max_deriv // 2,
            'max_spin': max_spin,
            'use_multiresolution': use_multiresolution,
            'tolerance': tolerance,
            'stage1_time_seconds': t1 - t0,
            'stage2_time_seconds': t3 - t2,
            'total_time_seconds': t3 - t0
        }

        print(f"\n{'=' * 60}")
        print(f"Result: Δσ = {delta_sigma:.6f}")
        print(f"        Δε = {delta_epsilon:.6f} (boundary)")
        print(f"        Δε' ≤ {combined_result['delta_epsilon_prime_bound']:.6f}")
        print(f"Total time: {t3-t0:.1f}s")
        print("=" * 60)

        with open(output_file, 'w') as f:
            json.dump(combined_result, f, indent=2)
        print(f"Saved to {output_file}")

        return combined_result
    else:
        # Simple single-point Δε bound
        output_file = f"{output_dir}/bound_{job_index:04d}.json"
        return run_single_point(delta_sigma, method, max_deriv, tolerance, output_file)


def collect_array_results(output_dir: str = 'results', output_file: str = 'combined_results.json'):
    """
    Collect results from array job into a single file.

    Automatically detects whether results are from:
    - Simple Δε bound computation (bound_*.json files)
    - Gap bound / Figure 6 computation (gap_bound_*.json files)
    """
    import glob

    # Check for gap bound results first
    gap_files = sorted(glob.glob(f"{output_dir}/gap_bound_*.json"))
    bound_files = sorted(glob.glob(f"{output_dir}/bound_*.json"))

    if gap_files:
        # Gap bound mode (Figure 6 style)
        print(f"Found {len(gap_files)} gap bound results")
        results = []
        for f in gap_files:
            with open(f) as fp:
                results.append(json.load(fp))

        delta_sigmas = [r['delta_sigma'] for r in results]
        delta_epsilons = [r['delta_epsilon_boundary'] for r in results]
        delta_epsilon_primes = [r['delta_epsilon_prime_bound'] for r in results]

        combined = {
            'type': 'gap_bound',
            'delta_sigma': delta_sigmas,
            'delta_epsilon_boundary': delta_epsilons,
            'delta_epsilon_prime_bound': delta_epsilon_primes,
            'method': results[0].get('method', 'unknown') if results else 'unknown',
            'nmax': results[0].get('nmax') if results else None,
            'max_spin': results[0].get('max_spin') if results else None,
            'n_points': len(results),
            'source_files': [str(f) for f in gap_files]
        }

        with open(output_file, 'w') as f:
            json.dump(combined, f, indent=2)

        # Save as numpy array: [Δσ, Δε, Δε']
        np_data = np.array(list(zip(delta_sigmas, delta_epsilons, delta_epsilon_primes)))
        np.save(output_file.replace('.json', '.npy'), np_data)

        print(f"Combined {len(gap_files)} gap bound results into {output_file}")
        print(f"  Δσ range: [{min(delta_sigmas):.4f}, {max(delta_sigmas):.4f}]")
        print(f"  Δε range: [{min(delta_epsilons):.4f}, {max(delta_epsilons):.4f}]")
        print(f"  Δε' range: [{min(delta_epsilon_primes):.4f}, {max(delta_epsilon_primes):.4f}]")

    elif bound_files:
        # Simple Δε bound mode
        print(f"Found {len(bound_files)} bound results")
        results = []
        for f in bound_files:
            with open(f) as fp:
                results.append(json.load(fp))

        delta_sigmas = [r['delta_sigma'] for r in results]
        bounds = [r['delta_epsilon_bound'] for r in results]

        combined = {
            'type': 'single_bound',
            'delta_sigma_values': delta_sigmas,
            'delta_epsilon_bounds': bounds,
            'n_points': len(results),
            'source_files': [str(f) for f in bound_files]
        }

        with open(output_file, 'w') as f:
            json.dump(combined, f, indent=2)

        np.save(output_file.replace('.json', '.npy'),
                np.array(list(zip(delta_sigmas, bounds))))

        print(f"Combined {len(bound_files)} results into {output_file}")

    else:
        print(f"No result files found in {output_dir}")
        print("  Expected: bound_*.json or gap_bound_*.json")


def run_two_stage_scan(max_deriv: int = 11, tolerance: float = 0.02,
                       sigma_min: float = 0.50, sigma_max: float = 0.60,
                       n_points: int = 25, output_file: Optional[str] = None):
    """
    Run two-stage scan following El-Showk et al. (2012) Figure 6 protocol.

    This is the CORRECT way to reproduce the paper's Δε' bounds:
    1. Stage 1: Compute Δε boundary for each Δσ
    2. Stage 2: Compute Δε' bound with Δε fixed to boundary

    Args:
        max_deriv: Maximum derivative order (used for both stages)
        tolerance: Binary search tolerance
        sigma_min: Minimum Δσ value
        sigma_max: Maximum Δσ value
        n_points: Number of points to compute
        output_file: Output file path

    Returns:
        Array of shape (n_points, 3) with [Δσ, Δε_max, Δε'_bound]
    """
    from bootstrap_gap_solver import DeltaEpsilonPrimeBoundComputer

    t0 = time.time()

    computer = DeltaEpsilonPrimeBoundComputer(d=3, max_deriv=max_deriv)
    results = computer.compute_two_stage_scan(
        delta_sigma_min=sigma_min,
        delta_sigma_max=sigma_max,
        n_points=n_points,
        tolerance_stage1=tolerance,
        tolerance_stage2=tolerance,
        verbose=True
    )

    t1 = time.time()

    print(f"\nTotal compute time: {t1-t0:.1f}s ({(t1-t0)/n_points:.2f}s per point)")

    # Save results
    output = {
        'delta_sigma': results[:, 0].tolist(),
        'delta_epsilon_max': results[:, 1].tolist(),
        'delta_epsilon_prime_bound': results[:, 2].tolist(),
        'method': 'two_stage',
        'max_deriv': max_deriv,
        'n_constraints': (max_deriv + 1) // 2,
        'tolerance': tolerance,
        'total_time_seconds': t1 - t0,
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {output_file}")

        # Also save as numpy
        np_file = output_file.replace('.json', '.npy')
        np.save(np_file, results)
        print(f"Saved numpy array to {np_file}")

    return results


def run_ising_plot(method: str = 'polynomial', max_deriv: int = 21,
                   poly_degree: int = 15, tolerance: float = 0.02,
                   sigma_min: float = 0.50, sigma_max: float = 0.60,
                   n_points: int = 25, output_file: Optional[str] = None):
    """
    Compute Δε' bounds along the Δε boundary curve (Ising plot).

    This reproduces the plot from El-Showk et al. (2012) Figure 6.

    Args:
        method: 'polynomial', 'hybrid', or 'discrete'
        max_deriv: Maximum derivative order
        poly_degree: Polynomial approximation degree
        tolerance: Binary search tolerance
        sigma_min: Minimum Δσ value
        sigma_max: Maximum Δσ value
        n_points: Number of points along curve
        output_file: Output file path

    Returns:
        Array of shape (n_points, 3) with [Δσ, Δε, Δε'_bound]
    """
    print("=" * 60)
    print("Computing Ising Plot (Δε' bounds along boundary)")
    print("=" * 60)

    t0 = time.time()

    if method in ['polynomial', 'hybrid']:
        computer = PolynomialPositivityGapSolver(
            max_deriv=max_deriv,
            poly_degree=poly_degree
        )
        results = computer.compute_ising_plot(
            delta_sigma_min=sigma_min,
            delta_sigma_max=sigma_max,
            n_points=n_points,
            tolerance=tolerance,
            method=method,
            verbose=True
        )
    else:
        # Use discrete sampling from bootstrap_gap_solver
        from bootstrap_gap_solver import DeltaEpsilonPrimeBoundComputer
        computer = DeltaEpsilonPrimeBoundComputer(d=3, max_deriv=max_deriv)
        results = computer.compute_ising_plot(
            delta_sigma_min=sigma_min,
            delta_sigma_max=sigma_max,
            n_points=n_points,
            tolerance=tolerance,
            verbose=True,
            boundary_method='literature'
        )

    t1 = time.time()

    print("\n" + "=" * 60)
    print(f"Total compute time: {t1-t0:.1f}s ({(t1-t0)/n_points:.2f}s per point)")
    print("=" * 60)

    # Save results
    output = {
        'delta_sigma': results[:, 0].tolist(),
        'delta_epsilon': results[:, 1].tolist(),
        'delta_epsilon_prime_bound': results[:, 2].tolist(),
        'method': method,
        'max_deriv': max_deriv,
        'poly_degree': poly_degree if method in ['polynomial', 'hybrid'] else None,
        'n_constraints': (max_deriv + 1) // 2,
        'tolerance': tolerance,
        'total_time_seconds': t1 - t0,
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {output_file}")

        # Also save as numpy
        np_file = output_file.replace('.json', '.npy')
        np.save(np_file, results)
        print(f"Saved numpy array to {np_file}")

    return results


def run_compare_methods(delta_sigma: float = 0.518, delta_epsilon: float = 1.41,
                        max_deriv: int = 11, poly_degree: int = 12,
                        tolerance: float = 0.05, output_file: Optional[str] = None):
    """
    Compare polynomial positivity with discrete sampling.

    This runs both methods and reports the differences.

    Args:
        delta_sigma: External dimension
        delta_epsilon: First scalar dimension
        max_deriv: Maximum derivative order
        poly_degree: Polynomial degree
        tolerance: Binary search tolerance
        output_file: Output file path

    Returns:
        Dictionary with comparison results
    """
    results = compare_methods(
        delta_sigma=delta_sigma,
        delta_epsilon=delta_epsilon,
        max_deriv=max_deriv,
        poly_degree=poly_degree,
        tolerance=tolerance,
        verbose=True
    )

    if output_file:
        # Convert numpy types for JSON
        output = {}
        for method, data in results.items():
            output[method] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in data.items()}

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='CFT Bootstrap Computation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic Δε bound
    python run_bootstrap.py --delta-sigma 0.518 --method lp

    # Δε' bound with gap assumption (discrete sampling)
    python run_bootstrap.py --gap-bound --delta-sigma 0.518 --delta-epsilon 1.41

    # Δε' bound with polynomial positivity (recommended)
    python run_bootstrap.py --gap-bound --method polynomial --max-deriv 21 --poly-degree 15

    # Δε' bound with hybrid method (polynomial + discrete)
    python run_bootstrap.py --gap-bound --method hybrid --max-deriv 21

    # Two-correlator bootstrap (ssss + eeee, stronger than single correlator)
    python run_bootstrap.py --gap-bound --method two-correlator --max-deriv 11

    # Full mixed correlator bootstrap (matrix SDP, strongest constraints)
    python run_bootstrap.py --gap-bound --method mixed-correlator --max-deriv 11

    # Ising plot (Δε' bounds along the boundary curve)
    python run_bootstrap.py --ising-plot --method polynomial --n-points 25

    # Two-stage scan (El-Showk 2012 Fig. 6 protocol)
    python run_bootstrap.py --two-stage --sigma-min 0.50 --sigma-max 0.60 --n-points 25

    # Compare discrete vs polynomial methods
    python run_bootstrap.py --compare --max-deriv 11 --poly-degree 12

    # Grid scan
    python run_bootstrap.py --grid --sigma-min 0.50 --sigma-max 0.60 --n-points 20
        """
    )

    # Mode selection
    parser.add_argument('--grid', action='store_true', help='Run grid scan')
    parser.add_argument('--gap-bound', action='store_true',
                       help='Compute Δε\' bound with gap assumption')
    parser.add_argument('--ising-plot', action='store_true',
                       help='Compute Δε\' bounds along the Δε boundary (Fig. 6)')
    parser.add_argument('--two-stage', action='store_true',
                       help='Two-stage scan: compute Δε boundary, then Δε\' (El-Showk 2012 Fig. 6 protocol)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare polynomial positivity vs discrete sampling')
    parser.add_argument('--array-job', action='store_true', help='Run as SLURM array job')
    parser.add_argument('--collect', action='store_true', help='Collect array job results')

    # Single point options
    parser.add_argument('--delta-sigma', type=float, default=0.518,
                       help='External operator dimension (default: 0.518 for Ising)')
    parser.add_argument('--delta-epsilon', type=float, default=1.41,
                       help='First scalar dimension for gap bounds (default: 1.41 for Ising)')

    # Grid options
    parser.add_argument('--sigma-min', type=float, default=0.50)
    parser.add_argument('--sigma-max', type=float, default=0.65)
    parser.add_argument('--n-points', type=int, default=50)

    # Parallel options
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n-workers', type=int, default=4)

    # Array job options
    parser.add_argument('--job-index', type=int, help='SLURM array task ID')
    parser.add_argument('--n-jobs', type=int, help='Total number of jobs')

    # Solver options
    parser.add_argument('--method',
                       choices=['lp', 'sdp', 'sdpb', 'cvxpy', 'discrete', 'polynomial', 'hybrid',
                                'two-correlator', 'mixed-correlator', 'el-showk', 'el-showk-sdpb'],
                       default='lp',
                       help='Solver method: lp, sdp, sdpb, cvxpy/discrete, polynomial (SOS), hybrid, '
                            'two-correlator (ssss+eeee), mixed-correlator (full matrix SDP), '
                            'el-showk (full derivative basis), el-showk-sdpb (El-Showk with SDPB)')
    parser.add_argument('--max-deriv', type=int, default=5,
                       help='Max derivative order (default: 5, use 21 for high precision)')
    parser.add_argument('--poly-degree', type=int, default=15,
                       help='Polynomial approximation degree (default: 15)')
    parser.add_argument('--tolerance', type=float, default=0.01,
                       help='Binary search tolerance (default: 0.01)')

    # SDPB specific options
    parser.add_argument('--sdpb-threads', type=int, default=4,
                       help='Number of threads for SDPB (default: 4)')
    parser.add_argument('--sdpb-precision', type=int, default=400,
                       help='SDPB precision in bits (default: 400)')

    # El-Showk method specific options
    parser.add_argument('--max-spin', type=int, default=50,
                       help='Maximum spin for El-Showk solver spinning operators (default: 50, paper uses 100)')
    parser.add_argument('--use-multiresolution', action='store_true',
                       help='Use T1-T5 style multi-resolution discretization for operator sampling')
    parser.add_argument('--el-showk-solver',
                       choices=['auto', 'scs', 'ecos', 'clarabel', 'mosek'],
                       default='auto',
                       help='Solver backend for El-Showk method (default: auto)')
    parser.add_argument('--nmax', type=int, default=None,
                       help='El-Showk nmax parameter directly (overrides --max-deriv//2 for el-showk method)')
    parser.add_argument('--high-precision', action='store_true',
                       help='Use full mpmath arbitrary-precision arithmetic (REQUIRED for '
                            'accurate reproduction of El-Showk et al. 2012 at nmax=10). '
                            'Much slower but avoids all float64 precision loss.')
    parser.add_argument('--precision', type=int, default=100,
                       help='Decimal places for high-precision mode (default: 100, use 150+ for nmax=10)')

    # Output
    parser.add_argument('--output', '-o', type=str, help='Output file')
    parser.add_argument('--output-dir', type=str, default='results')

    args = parser.parse_args()

    if args.collect:
        collect_array_results(args.output_dir, args.output or 'combined_results.json')
    elif args.array_job:
        if args.job_index is None or args.n_jobs is None:
            print("Error: --job-index and --n-jobs required for array job")
            sys.exit(1)
        run_array_job(
            job_index=args.job_index,
            n_jobs=args.n_jobs,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            method=args.method,
            max_deriv=args.max_deriv,
            tolerance=args.tolerance,
            poly_degree=args.poly_degree,
            output_dir=args.output_dir,
            gap_bound=args.gap_bound,
            # El-Showk parameters
            max_spin=args.max_spin,
            use_multiresolution=args.use_multiresolution,
            el_showk_solver=args.el_showk_solver,
            nmax=args.nmax,
            # High-precision parameters
            high_precision=args.high_precision,
            precision=args.precision
        )
    elif args.compare:
        # Compare methods
        output = args.output or f'compare_methods_{args.delta_sigma}.json'
        run_compare_methods(
            delta_sigma=args.delta_sigma,
            delta_epsilon=args.delta_epsilon,
            max_deriv=args.max_deriv,
            poly_degree=args.poly_degree,
            tolerance=args.tolerance,
            output_file=output
        )
    elif args.ising_plot:
        # Compute Ising plot (Δε' bounds along boundary)
        output = args.output or f'ising_plot_{args.method}_{args.n_points}pts.json'
        run_ising_plot(
            method=args.method if args.method in ['polynomial', 'hybrid', 'discrete'] else 'polynomial',
            max_deriv=args.max_deriv,
            poly_degree=args.poly_degree,
            tolerance=args.tolerance,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            n_points=args.n_points,
            output_file=output
        )
    elif args.two_stage:
        # Two-stage scan (El-Showk 2012 Fig. 6 protocol)
        output = args.output or f'two_stage_scan_{args.n_points}pts.json'
        run_two_stage_scan(
            max_deriv=args.max_deriv,
            tolerance=args.tolerance,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            n_points=args.n_points,
            output_file=output
        )
    elif args.gap_bound:
        # Δε' bound with gap assumption
        output = args.output or f'gap_bound_{args.delta_sigma}_{args.delta_epsilon}.json'
        # Map method names
        method = args.method
        if method in ['cvxpy', 'discrete']:
            method = 'cvxpy'
        elif method in ['lp', 'sdp']:
            method = 'cvxpy'  # These don't apply to gap bounds, use cvxpy
        run_gap_bound(
            args.delta_sigma, args.delta_epsilon,
            method=method,
            max_deriv=args.max_deriv,
            poly_degree=args.poly_degree,
            tolerance=args.tolerance,
            sdpb_threads=args.sdpb_threads,
            # El-Showk specific parameters
            max_spin=args.max_spin,
            use_multiresolution=args.use_multiresolution,
            el_showk_solver=args.el_showk_solver,
            nmax=args.nmax,
            output_file=output
        )
    elif args.grid:
        output = args.output or f'bootstrap_grid_{args.sigma_min}_{args.sigma_max}_{args.n_points}.json'
        run_grid(args.sigma_min, args.sigma_max, args.n_points,
                args.method, args.max_deriv, args.tolerance,
                args.parallel, args.n_workers, output)
    else:
        output = args.output or f'bootstrap_single_{args.delta_sigma}.json'
        run_single_point(args.delta_sigma, args.method, args.max_deriv, args.tolerance, output)


if __name__ == "__main__":
    main()
