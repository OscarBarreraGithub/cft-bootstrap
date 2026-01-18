#!/usr/bin/env python3
"""
Collect results from cluster jobs and create the bootstrap plot.

Usage:
    python collect_and_plot.py --results-dir results_0.500_0.650 --output ising_bootstrap.png
"""

import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Known CFT values for reference
KNOWN_CFTS = {
    '3D Ising': {'delta_sigma': 0.5181489, 'delta_epsilon': 1.412625, 'color': 'red', 'marker': '*', 'size': 200},
    'Free scalar': {'delta_sigma': 0.5, 'delta_epsilon': 1.0, 'color': 'blue', 'marker': 'o', 'size': 100},
    'O(2) model': {'delta_sigma': 0.5191, 'delta_epsilon': 1.5117, 'color': 'green', 'marker': 's', 'size': 100},
    'O(3) model': {'delta_sigma': 0.5189, 'delta_epsilon': 1.5957, 'color': 'purple', 'marker': '^', 'size': 100},
}


def collect_results(results_dir: str) -> dict:
    """Collect all JSON results from a directory."""
    files = sorted(glob.glob(f"{results_dir}/bound_*.json"))

    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    results = []
    for f in files:
        with open(f) as fp:
            results.append(json.load(fp))

    delta_sigmas = np.array([r['delta_sigma'] for r in results])
    bounds = np.array([r['delta_epsilon_bound'] for r in results])

    # Sort by delta_sigma
    idx = np.argsort(delta_sigmas)
    delta_sigmas = delta_sigmas[idx]
    bounds = bounds[idx]

    return {
        'delta_sigma': delta_sigmas,
        'delta_epsilon_bound': bounds,
        'n_points': len(results),
        'method': results[0].get('method', 'unknown'),
        'max_derivative_order': results[0].get('max_derivative_order', 'unknown')
    }


def plot_bootstrap_bound(results: dict, output_file: str = 'bootstrap_plot.png',
                         show_known: bool = True, title: str = None):
    """Create the bootstrap exclusion plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    delta_sigma = results['delta_sigma']
    bounds = results['delta_epsilon_bound']

    # Plot the bound curve
    ax.plot(delta_sigma, bounds, 'b-', linewidth=2, label='Bootstrap bound')

    # Fill the allowed region
    ax.fill_between(delta_sigma, 0, bounds, alpha=0.3, color='lightblue', label='Allowed region')

    # Fill the excluded region
    ax.fill_between(delta_sigma, bounds, max(bounds) * 1.2, alpha=0.3, color='lightcoral', label='Excluded region')

    # Plot known CFTs
    if show_known:
        for name, cft in KNOWN_CFTS.items():
            ds, de = cft['delta_sigma'], cft['delta_epsilon']
            # Only plot if in range
            if delta_sigma.min() <= ds <= delta_sigma.max():
                ax.scatter([ds], [de], c=cft['color'], marker=cft['marker'],
                          s=cft['size'], label=name, zorder=10, edgecolors='black')

    # Labels and styling
    ax.set_xlabel(r'$\Delta_\sigma$', fontsize=14)
    ax.set_ylabel(r'$\Delta_\epsilon$', fontsize=14)

    if title:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title(f'3D CFT Bootstrap Bound\n({results["n_points"]} points, '
                    f'method={results["method"]}, derivatives≤{results["max_derivative_order"]})',
                    fontsize=14)

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set reasonable axis limits
    ax.set_xlim(delta_sigma.min() - 0.01, delta_sigma.max() + 0.01)
    ax.set_ylim(0, min(max(bounds) * 1.1, 4.0))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")

    # Also save as PDF for publications
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Saved PDF to {pdf_file}")

    plt.show()


def plot_comparison(results_list: list, labels: list, output_file: str = 'bootstrap_comparison.png'):
    """Compare multiple bootstrap runs (e.g., different derivative orders)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_list)))

    for results, label, color in zip(results_list, labels, colors):
        ax.plot(results['delta_sigma'], results['delta_epsilon_bound'],
               linewidth=2, label=label, color=color)

    # Plot known CFTs
    for name, cft in KNOWN_CFTS.items():
        ax.scatter([cft['delta_sigma']], [cft['delta_epsilon']],
                  c=cft['color'], marker=cft['marker'], s=cft['size'],
                  label=name, zorder=10, edgecolors='black')

    ax.set_xlabel(r'$\Delta_\sigma$', fontsize=14)
    ax.set_ylabel(r'$\Delta_\epsilon$', fontsize=14)
    ax.set_title('CFT Bootstrap: Convergence with Derivative Order', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_file}")


def print_summary(results: dict):
    """Print a summary of the bootstrap results."""
    print("\n" + "="*60)
    print("BOOTSTRAP RESULTS SUMMARY")
    print("="*60)

    print(f"\nTotal points computed: {results['n_points']}")
    print(f"Method: {results['method']}")
    print(f"Max derivative order: {results['max_derivative_order']}")

    ds = results['delta_sigma']
    bounds = results['delta_epsilon_bound']

    print(f"\nDelta_sigma range: [{ds.min():.4f}, {ds.max():.4f}]")
    print(f"Delta_epsilon bound range: [{bounds.min():.4f}, {bounds.max():.4f}]")

    # Check if Ising point is in allowed region
    ising_ds = KNOWN_CFTS['3D Ising']['delta_sigma']
    ising_de = KNOWN_CFTS['3D Ising']['delta_epsilon']

    if ds.min() <= ising_ds <= ds.max():
        # Interpolate bound at Ising point
        bound_at_ising = np.interp(ising_ds, ds, bounds)
        status = "ALLOWED ✓" if ising_de < bound_at_ising else "EXCLUDED ✗"
        print(f"\n3D Ising model ({ising_ds:.4f}, {ising_de:.4f}):")
        print(f"  Bound at Δσ={ising_ds:.4f}: Δε ≤ {bound_at_ising:.4f}")
        print(f"  Status: {status}")


def main():
    parser = argparse.ArgumentParser(description='Collect and plot bootstrap results')

    parser.add_argument('--results-dir', '-r', type=str, required=True,
                       help='Directory containing result JSON files')
    parser.add_argument('--output', '-o', type=str, default='bootstrap_plot.png',
                       help='Output plot filename')
    parser.add_argument('--no-known', action='store_true',
                       help='Do not plot known CFT points')
    parser.add_argument('--title', type=str, help='Custom plot title')
    parser.add_argument('--save-combined', type=str,
                       help='Save combined results to JSON file')

    args = parser.parse_args()

    # Collect results
    print(f"Collecting results from {args.results_dir}...")
    results = collect_results(args.results_dir)

    # Print summary
    print_summary(results)

    # Save combined results
    if args.save_combined:
        combined = {
            'delta_sigma_values': results['delta_sigma'].tolist(),
            'delta_epsilon_bounds': results['delta_epsilon_bound'].tolist(),
            'method': results['method'],
            'max_derivative_order': results['max_derivative_order'],
            'n_points': results['n_points']
        }
        with open(args.save_combined, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved combined results to {args.save_combined}")

    # Create plot
    plot_bootstrap_bound(results, args.output, show_known=not args.no_known, title=args.title)


if __name__ == "__main__":
    main()
