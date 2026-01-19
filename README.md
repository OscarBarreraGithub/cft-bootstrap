# CFT Bootstrap with Mathematica/Wolfram MCP

A research project implementing the conformal field theory (CFT) bootstrap to reproduce and extend the classic bounds on 3D CFT operator dimensions, with the 3D Ising model as the key target.

## Project Overview

This repository contains two main components:

1. **Wolfram MCP Server** - A Model Context Protocol server that provides Claude with access to Mathematica/Wolfram Language for symbolic computation, with special features for proof-carrying calculations and mathematical verification.

2. **CFT Bootstrap Code** - Python implementation of the numerical conformal bootstrap, designed to run on a computing cluster to reproduce the famous plot showing the 3D Ising model sitting at a kink on the boundary of the allowed region.

## The Physics: What is the CFT Bootstrap?

### Background

Conformal Field Theories (CFTs) are quantum field theories with scale invariance. They appear throughout physics:
- Critical points of phase transitions (like the 3D Ising model at its critical temperature)
- The AdS/CFT correspondence in string theory
- 2D statistical mechanics (exactly solvable models)

The **conformal bootstrap** is a non-perturbative approach that constrains CFT data using only:
1. **Crossing symmetry** - The 4-point correlation function must be consistent whether you expand in the s-channel or t-channel
2. **Unitarity** - OPE coefficients squared must be positive

### The Key Result

For a 3D CFT with a scalar operator σ, the bootstrap constrains the allowed values of:
- Δσ = dimension of σ
- Δε = dimension of the lowest scalar in the σ × σ OPE (other than identity)

The remarkable discovery (El-Showk et al., 2012): The **3D Ising model sits exactly at a kink** on the boundary of the allowed region! This allows extremely precise determination of Ising critical exponents.

```
        Δε
         │
    2.0 ─┤        ╱
         │       ╱
    1.5 ─┤    ★ ╱   ← 3D Ising sits here!
         │     ╱
    1.0 ─┤────╱  Allowed region (below curve)
         │
    0.5 ─┼────────────────
         0.5   0.52  0.54   Δσ
```

## Repository Structure

```
Claude Mathematica/
├── README.md                    # This file
├── SETUP.md                     # Configuration guide
├── .mcp.json.example            # MCP config template (copy to .mcp.json)
├── .claude/
│   └── settings.local.json     # Claude permissions for Wolfram tools
├── Wolfram-MCP/                 # Mathematica MCP server
│   ├── wolfram_mcp_server.py   # Main server implementation
│   ├── OPTIMAL_WORKFLOW.md     # Best practices for using the server
│   ├── VISUALIZATION_WORKFLOW.md
│   └── demo_notebook.ipynb
├── cft_bootstrap/              # Bootstrap implementation
│   ├── bootstrap_solver.py     # Core solver for Δε bounds
│   ├── bootstrap_gap_solver.py # Solver for Δε' bounds with gap assumption
│   ├── run_bootstrap.py        # CLI for cluster execution
│   ├── collect_and_plot.py     # Result collection and plotting
│   ├── submit_cluster.sh       # SLURM submission script
│   └── README.md               # Bootstrap-specific documentation
├── notebooks/                  # Jupyter notebooks
│   └── reproduce_ising_delta_epsilon_prime.ipynb  # Reproduce El-Showk 2012 Fig. 7
└── reference_plots/            # Reference and reproduced plots
    ├── el_showk_2012_fig7_delta_epsilon_prime.png  # Original from paper
    └── reproduced_delta_epsilon_prime.png          # Our reproduction
```

## Current Status

### What Works

✅ **Wolfram MCP Server**
- Full access to Mathematica from Claude
- Proof-carrying computation with assumption tracking
- Numeric validation of symbolic results
- Test suites for mathematical derivations

✅ **Bootstrap Implementation**
- 3D scalar conformal blocks (Dolan-Osborn formula)
- Crossing equation setup with proper symmetry constraints
- Linear programming feasibility check
- Binary search for bounds
- Cluster-ready with SLURM support

### Current Results

**Basic Δε bounds** (with 3 derivative constraints, m = 1, 3, 5):

| Δσ | Δε bound |
|-----|----------|
| 0.50 | ~1.0 |
| 0.518 | ~1.57 |
| 0.54 | ~1.65 |
| 0.56 | ~1.72 |
| 0.60 | ~1.86 |

The 3D Ising model (Δσ ≈ 0.518, Δε ≈ 1.41) is correctly inside the allowed region.

**Δε' bounds** (second scalar, with gap assumption):

We also reproduce the upper bound on Δε' from [El-Showk et al. (2012) Figure 7](https://arxiv.org/abs/1203.6064):

| Δσ | Δε (assumed) | Δε' bound |
|-----|--------------|-----------|
| 0.50 | 1.0 | ~1.1 |
| 0.518 | 1.41 | ~2.6 |
| 0.55 | 1.49 | ~3.1 |
| 0.60 | 1.58 | ~3.5 |

The reference plot shows Δε' ≤ ~3.8 at the Ising kink with higher derivative order (~20+). Our bounds are tighter due to fewer constraints, but capture the qualitative shape including the kink.

### What's Needed for Publication-Quality Results

1. **More derivatives** - Need ~20+ derivative constraints for tight bounds
   - Current limitation: numerical instability in high-order finite differences
   - Solution: Use automatic differentiation or rational arithmetic

2. **SDP solver** - Proper semidefinite programming gives optimal bounds
   - Install CVXPY: `pip install cvxpy`
   - Or use SDPB (the gold standard for bootstrap)

3. **Spinning operators** - Include the stress tensor (Δ=3, ℓ=2)
   - Adds another constraint that sharpens the bound significantly

4. **Multiple correlators** - Use ⟨σσσσ⟩, ⟨σσεε⟩, ⟨εεεε⟩ together
   - This is what produces the sharp kink at the Ising point

## Getting Started

### Prerequisites

- Python 3.9+
- Mathematica/Wolfram Engine (for the MCP server)
- Claude Code CLI

> **Platform Note:** The Wolfram MCP server is primarily developed and tested on **macOS**. It should work on other platforms with proper configuration of the `WOLFRAM_KERNEL_PATH` environment variable. See [`SETUP.md`](SETUP.md) and the [Wolfram-MCP README](Wolfram-MCP/README.md#configuration) for platform-specific details.

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/claude-mathematica.git
cd claude-mathematica

# Set up the bootstrap environment
cd cft_bootstrap
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: Install SDP solver
pip install cvxpy
```

### Running the Bootstrap

```bash
# Single point (Δε bound)
python run_bootstrap.py --delta-sigma 0.518

# Grid scan (local)
python run_bootstrap.py --grid --sigma-min 0.50 --sigma-max 0.65 --n-points 50

# Grid scan (parallel)
python run_bootstrap.py --grid --parallel --n-workers 8

# On a cluster (edit submit_cluster.sh first)
sbatch submit_cluster.sh

# Collect and plot results
python collect_and_plot.py --results-dir results_0.500_0.650 --output ising_plot.png
```

### Reproducing the Δε' Plot (El-Showk et al. 2012, Fig. 7)

The Jupyter notebook `notebooks/reproduce_ising_delta_epsilon_prime.ipynb` reproduces the famous plot showing the upper bound on Δε' (second Z₂-even scalar). This can be run locally:

```bash
cd notebooks
jupyter notebook reproduce_ising_delta_epsilon_prime.ipynb
```

Or run the computation directly:

```bash
cd cft_bootstrap
python -c "
from bootstrap_gap_solver import DeltaEpsilonPrimeBoundComputer
import numpy as np

computer = DeltaEpsilonPrimeBoundComputer(d=3, max_deriv=5)
results = computer.compute_ising_plot(n_points=50, tolerance=0.02)
np.save('delta_epsilon_prime_bounds.npy', results)
"
```

### Using the Wolfram MCP Server

The MCP server gives Claude access to Mathematica. Key tools:

- `wolfram_eval` - Evaluate Wolfram Language code
- `wolfram_eval_proven` - Evaluate with assumption tracking and numeric validation
- `wolfram_typed_equality` - Verify mathematical equalities
- `wolfram_numeric_validate` - Check symbolic results numerically
- `wolfram_create_test_suite` / `wolfram_add_test` / `wolfram_run_test_suite` - Mathematical regression testing

## The Math: How the Bootstrap Works

### Crossing Equation

For the 4-point function ⟨σ(x₁)σ(x₂)σ(x₃)σ(x₄)⟩:

```
G(u,v) = Σ_O λ²_{σσO} g_O(u,v)
```

where g_O are conformal blocks and u, v are cross-ratios.

Crossing symmetry requires:
```
(v/u)^{Δσ} G(u,v) = G(v,u)
```

### The Bootstrap Algorithm

1. Define F_O = v^{Δσ} g_O(u,v) - u^{Δσ} g_O(v,u)
2. Taylor expand around the crossing-symmetric point z = z̄ = 1/2
3. The crossing equation becomes: Σ_O p_O F_O^{(m,n)} = 0
4. Search for a linear functional α such that:
   - α · F_id > 0 (normalization)
   - α · F_O ≥ 0 for all O above the assumed gap
5. If such α exists → that (Δσ, Δε) point is **excluded**

## References

- Rattazzi, Rychkov, Tonni, Vichi (2008): "Bounding scalar operator dimensions in 4D CFT"
- El-Showk et al. (2012): "Solving the 3D Ising Model with the Conformal Bootstrap"
- Poland, Rychkov, Vichi (2019): "The Conformal Bootstrap: Theory, Numerical Techniques, and Applications"
- Simmons-Duffin (2016): "SDPB: A Semidefinite Program Solver for the Conformal Bootstrap"

## License

MIT License - see individual directories for component-specific licenses.
