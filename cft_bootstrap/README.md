# CFT Bootstrap Implementation

Numerical implementation of the conformal bootstrap for 3D CFTs, targeting the famous Ising model bound.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test the solver
python bootstrap_solver.py

# Run a single point
python run_bootstrap.py --delta-sigma 0.518

# Run a grid
python run_bootstrap.py --grid --sigma-min 0.50 --sigma-max 0.65 --n-points 50
```

## Files

| File | Description |
|------|-------------|
| `bootstrap_solver.py` | Core implementation: conformal blocks, crossing equation, LP solver |
| `run_bootstrap.py` | Command-line interface for local and cluster execution |
| `collect_and_plot.py` | Collect cluster results and generate publication-quality plots |
| `submit_cluster.sh` | SLURM submission script for HPC clusters |
| `conformal_blocks.py` | Alternative conformal block implementation (reference) |
| `conformal_blocks_v2.py` | Experimental improved numerics |

## Theory Summary

### The Setup

We consider a 3D CFT with a scalar primary operator σ of dimension Δσ. The operator product expansion (OPE) is:

```
σ × σ ~ 1 + ε + ...
```

where ε is the lowest-dimension scalar (dimension Δε) and ... includes higher scalars and spinning operators.

### The Crossing Equation

The 4-point function ⟨σσσσ⟩ can be decomposed in conformal blocks:

```
G(z, z̄) = Σ_O λ²_O g_{Δ_O, ℓ_O}(z, z̄)
```

Crossing symmetry (invariance under x₁ ↔ x₃) gives:

```
v^{Δσ} G(u,v) = u^{Δσ} G(v,u)
```

where u = zz̄ and v = (1-z)(1-z̄).

### The Algorithm

1. **Define the crossing vector**: F_O = v^{Δσ} g_O(u,v) - u^{Δσ} g_O(v,u)

2. **Expand in derivatives**: Taylor expand around z = z̄ = 1/2 to get F_O^{(m,n)}

3. **Crossing constraint**: Σ_O p_O F_O^{(m,n)} = 0 for all (m,n) with p_O ≥ 0

4. **Feasibility check**: Can -F_id be written as a positive combination of F_O for Δ ≥ Δε?
   - Yes → (Δσ, Δε) is **allowed**
   - No → (Δσ, Δε) is **excluded**

5. **Find the bound**: Binary search for the largest allowed Δε

## Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--max-deriv` | Maximum derivative order | 5 | Use 5-7 for stability; need ~20 for tight bounds |
| `--tolerance` | Bound precision | 0.01 | Lower = more precise, more computation |
| `--method` | Solver type | `lp` | `lp` is fast; `sdp` is optimal (needs CVXPY) |
| `--n-samples` | Operators sampled | 100 | More = tighter bounds |

## Cluster Execution

For large-scale computation:

```bash
# 1. Edit submit_cluster.sh with your parameters
vim submit_cluster.sh

# 2. Submit array job
sbatch submit_cluster.sh

# 3. Monitor
squeue -u $USER

# 4. Collect results
python collect_and_plot.py --results-dir results_0.500_0.650
```

## Known Issues & Limitations

### Numerical Instability

High-order derivatives (m > 7) become unstable due to finite-difference errors. Solutions:
- Use automatic differentiation (JAX, PyTorch)
- Implement symbolic/rational arithmetic for conformal blocks
- Use Chebyshev approximation

### LP vs SDP

The current LP formulation checks if crossing can be satisfied by sampling operator dimensions. This is weaker than the proper SDP approach which finds the optimal functional.

For publication-quality bounds:
```bash
pip install cvxpy
python run_bootstrap.py --method sdp
```

Or use [SDPB](https://github.com/davidsd/sdpb), the gold standard.

### Missing Features

For the sharp Ising kink, you need:
1. **Spinning operators** - Especially the stress tensor (Δ=3, ℓ=2)
2. **Multiple correlators** - ⟨σσσσ⟩, ⟨σσεε⟩, ⟨εεεε⟩
3. **Gap assumptions** - Known bounds on other operators

## Results

### Current (3 derivatives)

```
Δσ = 0.500: Δε ≤ 1.00  (free scalar is on boundary)
Δσ = 0.518: Δε ≤ 1.57  (Ising at 1.41 is allowed ✓)
Δσ = 0.540: Δε ≤ 1.65
Δσ = 0.600: Δε ≤ 1.86
```

### Target (from literature, ~20 derivatives + spinning)

```
Δσ = 0.518: Δε ≤ 1.42  (tight bound!)
```

The Ising model sits at the kink where the bound is tightest.

## References

1. Rattazzi, Rychkov, Tonni, Vichi - "Bounding scalar operator dimensions in 4D CFT" (2008)
2. El-Showk et al. - "Solving the 3D Ising Model with the Conformal Bootstrap" (2012)
3. Poland, Rychkov, Vichi - "The Conformal Bootstrap" (2019 review)
4. Simmons-Duffin - "SDPB: A Semidefinite Program Solver for the Conformal Bootstrap" (2015)
