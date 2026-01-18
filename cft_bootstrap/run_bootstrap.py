#!/usr/bin/env python3
"""
Main script to run CFT bootstrap computations.

This script is designed to be run on a cluster with various options:
- Single point evaluation
- Grid scan over delta_sigma
- Parallel execution using multiprocessing or MPI

Usage:
    # Single point
    python run_bootstrap.py --delta-sigma 0.518 --method lp

    # Grid scan
    python run_bootstrap.py --grid --sigma-min 0.50 --sigma-max 0.65 --n-points 50

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
                  output_dir: str = 'results'):
    """
    Run a single point as part of an array job (for SLURM).

    Each job computes one delta_sigma value.
    """
    delta_sigmas = np.linspace(sigma_min, sigma_max, n_jobs)
    delta_sigma = delta_sigmas[job_index]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}/bound_{job_index:04d}.json"

    return run_single_point(delta_sigma, method, max_deriv, tolerance, output_file)


def collect_array_results(output_dir: str = 'results', output_file: str = 'combined_results.json'):
    """Collect results from array job into a single file."""
    import glob

    files = sorted(glob.glob(f"{output_dir}/bound_*.json"))
    results = []

    for f in files:
        with open(f) as fp:
            results.append(json.load(fp))

    delta_sigmas = [r['delta_sigma'] for r in results]
    bounds = [r['delta_epsilon_bound'] for r in results]

    combined = {
        'delta_sigma_values': delta_sigmas,
        'delta_epsilon_bounds': bounds,
        'n_points': len(results),
        'source_files': files
    }

    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2)

    np.save(output_file.replace('.json', '.npy'),
            np.array(list(zip(delta_sigmas, bounds))))

    print(f"Combined {len(files)} results into {output_file}")


def main():
    parser = argparse.ArgumentParser(description='CFT Bootstrap Computation')

    # Mode selection
    parser.add_argument('--grid', action='store_true', help='Run grid scan')
    parser.add_argument('--array-job', action='store_true', help='Run as SLURM array job')
    parser.add_argument('--collect', action='store_true', help='Collect array job results')

    # Single point options
    parser.add_argument('--delta-sigma', type=float, default=0.518,
                       help='External operator dimension')

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
    parser.add_argument('--method', choices=['lp', 'sdp'], default='lp')
    parser.add_argument('--max-deriv', type=int, default=5,
                       help='Max derivative order (use 5 or 7 for stability)')
    parser.add_argument('--tolerance', type=float, default=0.01)

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
        run_array_job(args.job_index, args.n_jobs, args.sigma_min, args.sigma_max,
                     args.method, args.max_deriv, args.tolerance, args.output_dir)
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
