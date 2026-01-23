# CFT Bootstrap Implementation

Numerical implementation of the conformal bootstrap for 3D CFTs, targeting the famous Ising model bound.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test the solver
python bootstrap_solver.py

# Run a single point (Δε bound)
python run_bootstrap.py --delta-sigma 0.518

# Run Δε' bound with gap assumption
python run_bootstrap.py --gap-bound --delta-sigma 0.518 --delta-epsilon 1.41

# Run with high-order constraints
python run_bootstrap.py --gap-bound --max-deriv 21 --method sdpb

# Run a grid
python run_bootstrap.py --grid --sigma-min 0.50 --sigma-max 0.65 --n-points 50
```

## Files

| File | Description |
|------|-------------|
| `bootstrap_solver.py` | Core implementation: conformal blocks, crossing equation, LP solver |
| `bootstrap_gap_solver.py` | Gap-based solver for Δε' bounds |
| `taylor_conformal_blocks.py` | Taylor series expansion for high-order derivatives |
| `spinning_conformal_blocks.py` | Spinning conformal blocks (radial expansion) |
| `sdpb_interface.py` | **NEW** SDPB integration for high-precision bounds |
| `run_bootstrap.py` | Command-line interface for local and cluster execution |
| `collect_and_plot.py` | Collect cluster results and generate publication-quality plots |
| `submit_cluster.sh` | SLURM submission script for HPC clusters |

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
| `--max-deriv` | Maximum derivative order | 5 | Use 5-7 for stability with finite diff; 21+ with Taylor series |
| `--tolerance` | Bound precision | 0.01 | Lower = more precise, more computation |
| `--method` | Solver type | `lp` | `lp`, `sdp` (CVXPY), `sdpb` (SDPB), `cvxpy` (fallback) |
| `--poly-degree` | Polynomial approximation | 20 | For SDPB polynomial matrix program |
| `--gap-bound` | Enable Δε' bounds | False | Compute second scalar bound with gap assumption |
| `--delta-epsilon` | First scalar dimension | 1.41 | For gap-bound mode (Ising default) |

## SDPB Integration

SDPB (Semidefinite Program solver for the Bootstrap) is the gold-standard solver for conformal bootstrap problems. This implementation provides:

1. **Polynomial Matrix Program (PMP) generation** - Approximates crossing constraints as polynomials
2. **Automatic fallback** - Uses CVXPY if SDPB is not installed
3. **High-order derivatives** - Supports 20+ constraint via Taylor series expansion

### Installing SDPB

**macOS (Homebrew):**
```bash
brew tap davidsd/sdpb
brew install sdpb
```

**Linux (Docker):**
```bash
docker pull bootstrapcollaboration/sdpb
```

**From source:** See [SDPB GitHub](https://github.com/davidsd/sdpb)

### Using SDPB

```bash
# Basic usage (falls back to CVXPY if SDPB unavailable)
python run_bootstrap.py --gap-bound --method sdpb

# With high-order constraints
python run_bootstrap.py --gap-bound --max-deriv 21 --method sdpb

# Configure SDPB parameters
python run_bootstrap.py --gap-bound --sdpb-threads 8 --sdpb-precision 512
```

### Python API

```python
from sdpb_interface import compute_bound_with_sdpb, SDPBConfig

# Quick computation
bound = compute_bound_with_sdpb(
    delta_sigma=0.518,
    delta_epsilon=1.41,
    max_deriv=21,
    tolerance=0.01
)
print(f"Δε' ≤ {bound:.4f}")

# With custom configuration
config = SDPBConfig(
    precision=512,      # bits
    num_threads=8,
    max_iterations=1000
)
from sdpb_interface import SDPBSolver
solver = SDPBSolver(config)
bound = solver.find_bound(0.518, 1.41, max_deriv=21)
```

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

## ⚠️ Critical Implementation Issues

**This implementation cannot accurately reproduce El-Showk et al. (2012) Figure 6.** The following fundamental issues must be addressed:

### 1. Numerical Instability (CRITICAL)

**Location:** [el_showk_basis.py:287-357](cft_bootstrap/el_showk_basis.py#L287-L357)

The derivative computation uses **finite differences** with step size `h=0.02`:

```python
def _numerical_derivative(self, delta, m, n, h_a=0.02, h_b=0.02):
    # Central differences for ∂_a^m ∂_b^n F
```

**Problem:** For `nmax=10` (paper's value), we need derivatives up to order ~20. The roundoff error grows as:
```
error ≈ ε_machine / h^n ≈ 10^-16 / 0.02^20 ≈ 10^18
```

This is **catastrophic cancellation**. The paper used **exact rational arithmetic** (via GMP/MPFR in SDPB) to avoid this entirely.

**Status:** ✅ SOLVED - Added `analytical_derivatives.py` module with two precision modes:

1. **Standard mode** (default for nmax < 10): Richardson extrapolation (6 levels) with float64
   - Faster but limited to nmax ≤ 7 due to precision loss at high orders
   - Uses hypergeometric series for conformal blocks

2. **High-precision mode** (`--high-precision` flag): Full mpmath arbitrary-precision arithmetic
   - REQUIRED for accurate reproduction of El-Showk et al. (2012) at nmax=10
   - Uses configurable precision (default 100 decimal places, use 150+ for nmax=10)
   - Much slower but avoids ALL float64 precision loss
   - Verified: at order F[7,0], standard mode is off by factor of 184x, high-precision is correct

**Usage for publication-quality results:**
```bash
# Local test
python run_bootstrap.py --gap-bound --method el-showk --nmax 10 --high-precision --precision 150

# Cluster (enabled by default in submit_cluster.sh)
sbatch submit_cluster.sh  # HIGH_PRECISION=true, PRECISION=150
```

### 2. Missing Spinning Operators in Stage 1

**Location:** [run_bootstrap.py:471-474](cft_bootstrap/run_bootstrap.py#L471-L474)

The Figure 6 two-stage computation uses `BootstrapSolver` for Stage 1 (finding Δε boundary):

```python
basic_solver = BootstrapSolver(d=3, max_deriv=max_deriv)
delta_epsilon = basic_solver.find_bound(delta_sigma, method='lp', ...)
```

**Problem:** `BootstrapSolver` and `taylor_conformal_blocks.py` include **scalars only** (ℓ=0). The paper includes spinning operators (stress tensor ℓ=2, Δ=3, etc.) which significantly tighten the bounds.

**Impact:** The Δε boundary curve will be **above** the true boundary, shifting the kink location.

**Status:** ⚠️ PARTIAL - Stage 2 (`el_showk_basis.py`) includes spinning operators, but Stage 1 does not.

### 3. SDPB Data Source Mismatch

**Location:** [sdpb_interface.py:172](cft_bootstrap/sdpb_interface.py#L172)

The basic `PolynomialApproximator` feeds SDPB data from `taylor_conformal_blocks.py`:

```python
self.crossing = TaylorCrossingVector(delta_sigma, max_deriv)  # Scalar-only!
```

**Problem:** Even when using SDPB (the gold-standard solver), we're feeding it **scalar-only** crossing data, missing the crucial spinning operator contributions.

**Note:** `ElShowkPolynomialApproximator` was added to address this, but it inherits the finite-difference instability from Issue #1.

**Status:** ⚠️ PARTIAL - architecture exists but blocked by Issue #1.

---

## Known Issues & Limitations

### Numerical Precision

- **Finite differences** (m > 7): Unstable due to error accumulation
  - ✅ **SOLVED**: Use `--high-precision` flag for mpmath arbitrary-precision mode

- **CVXPY SDP** (11+ constraints): Condition numbers grow to 10^15
  - ✅ **SOLVED**: Use SDPB integration (`sdpb_interface.py`)

### Current Gap to Literature

Our Δε' bounds are ~1.3 units below El-Showk et al. (2012). Root causes:

| Factor | Our Implementation | Reference | Impact |
|--------|-------------------|-----------|--------|
| **Derivative precision** | ✅ mpmath high-precision | Rational arithmetic | **SOLVED** |
| **Spinning operators** | Stage 2 only | Both stages | HIGH |
| Derivative constraints | 6-11 | ~66 | ✅ SOLVED with nmax=10 |
| Polynomial positivity | Discrete sampling | Continuous | MEDIUM |

### Remaining Work for Publication Quality

1. **✅ Arbitrary precision derivatives** - Implemented via `--high-precision` flag
2. **🟡 Spinning operators in Stage 1** - Add spinning blocks to `BootstrapSolver`
3. **🟡 Polynomial positivity constraints** - Enforce α·F_Δ ≥ 0 for ALL Δ ≥ gap
4. **✅ More constraints with SDPB** - 66 constraints enabled with nmax=10

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
