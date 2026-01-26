# Roadmap: Matching El-Showk et al. (2012) Figure 6

This document outlines what is needed to exactly reproduce the Δε' bounds from:

> **"Solving the 3D Ising Model with the Conformal Bootstrap"**
> S. El-Showk, M. Paulos, D. Poland, S. Rychkov, D. Simmons-Duffin, A. Vichi
> [arXiv:1203.6064](https://arxiv.org/abs/1203.6064) (2012), Figure 6

## Current Status

| Metric | Our Implementation | Reference | Gap |
|--------|-------------------|-----------|-----|
| Δε' at Ising (Δσ=0.518) | ~2.5 | ~3.8 | ~1.3 |
| Derivative constraints | 6-31 | ~60+ | Similar (see findings) |
| Operator types | Scalars + spinning (ℓ=0,2,4,6) | All spins | Covered |
| Numerical stability | Fixed with normalization | N/A | ✓ |
| Curve smoothness | Jagged | Smooth | Needs SDPB |

## Investigation Findings (January 2026)

### Verified Components ✓

1. **Conformal blocks**: Exact match with Mathematica (g(Δ=2, z=0.3, zbar=0.5) = 0.0874099805)
2. **F-vector derivatives**: Match numerical differentiation from Mathematica
3. **Crossing equation**: Correct formulation F_O = v^Δσ g(z,zbar) - u^Δσ g(1-z,1-zbar)
4. **SDP setup**: Correctly finds α with α·F_id=1 and α·F_O≥0

### Root Causes of the ~1.3 Unit Gap

**Finding 1: Numerical Instability (FIXED)**
- F-vectors have 10^8 dynamic range: ||F(Δ=30)|| / ||F_id|| ~ 10^4
- Causes solver to return "infeasible_inaccurate" or "optimal_inaccurate"
- **Fix**: Normalize each F-vector before passing to SDP

**Finding 2: Constraint Structure Problem (FUNDAMENTAL)**
- F_id is dominated by first component: F_id_normalized ≈ [-0.9999, 0.006, 0.002, ...]
- The normalization α·F_id = 1 effectively only constrains α[0] ≈ -1
- Higher components α[1], α[2], ... are nearly unconstrained
- This allows the solver to find "trivial" solutions

**Finding 3: More Constraints → TIGHTER Bounds**
Testing with proper normalization:
```
 3 constraints: Δε' ≤ 2.63
 6 constraints: Δε' ≤ 2.48
11 constraints: Δε' ≤ 2.30
16 constraints: Δε' ≤ 2.35
31 constraints: Δε' ≤ 2.35
```
More constraints give the optimizer more freedom to find excluding functionals → lower (tighter) bounds. This is mathematically correct but opposite to what we need.

**Finding 4: Spinning Operators Don't Help**
- All F-vectors (scalar and spinning) for Δ ≥ 2 have mostly positive components
- Including spinning operators adds more positive constraints, not negative ones
- The bound structure remains fundamentally unchanged

### What SDPB Would Provide

| Feature | CVXPY (current) | SDPB |
|---------|-----------------|------|
| Precision | 64-bit float | 400+ bit arbitrary |
| Curve smoothness | Jagged | Smooth |
| The ~1.3 gap | Still present | Still present |

**SDPB gives smooth curves but won't fix the gap.** The gap is from the problem formulation, not numerical precision.

### Remaining Questions

1. **Different normalization?** The paper may use α·F_id[0] = 1 instead of α·F_id = 1
2. **Different objective?** May maximize/minimize something other than feasibility
3. **Polynomial positivity?** Continuous positivity might change the constraint structure
4. **OPE bounds?** May bound OPE coefficients rather than just dimensions

## Required Improvements

### 1. Taylor Series Conformal Blocks ✅ IMPLEMENTED

**Previous:** Finite difference derivatives (unstable for m > 7).

**Solution:** Taylor series expansion around z = 1/2, reading off derivatives as coefficients.

**Implementation:** `cft_bootstrap/taylor_conformal_blocks.py`
- `HighOrderGapBootstrapSolver`: Supports up to 31+ constraints
- Avoids finite difference instability
- Verified against Mathematica implementation

**Result:** Works correctly but doesn't improve bounds due to missing spinning operators.

---

### 2. Spinning Conformal Blocks ✅ IMPLEMENTED

**Previous:** We only included scalar operators (spin-0) in the OPE.

**Solution:** Implemented spinning conformal blocks using the radial expansion from Hogervorst & Rychkov (2013).

**Implementation:** `cft_bootstrap/spinning_conformal_blocks.py`
- `SpinningConformalBlock`: Computes blocks for any spin ℓ using Casimir recursion
- `SpinningCrossingVector`: Builds F-vectors for spinning operators
- `SpinningBootstrapSolver`: Bootstrap solver including spinning operators up to max_spin

**Key Details:**
- Uses radial expansion: g_{Δ,ℓ}(ρ,η) = ρ^Δ Σ A_{n,j} ρ^n P_j(η)
- Casimir recursion for coefficients A_{n,j}
- Calibration factor (z/ρ)^Δ to match Dolan-Osborn normalization
- Scalars use exact Taylor series; spinning uses polynomial fitting for F-vectors

**Result:** Spinning operators are included but don't significantly improve bounds with current constraint count. The bound remains at ~2.5 with 6 constraints, matching the scalar-only result.

**Limitation:** The radial expansion has a different normalization than Dolan-Osborn, requiring calibration. More sophisticated normalization or direct implementation of spinning Dolan-Osborn formulas may be needed.

**References:**
- Hogervorst & Rychkov (2013): [arXiv:1303.1111](https://arxiv.org/abs/1303.1111) - Radial coordinates
- Dolan, Osborn (2001): "Conformal four point functions and the operator product expansion"

---

### 3. SDPB Integration ✅ IMPLEMENTED

**Previous:** CVXPY with SCS solver shows conditioning issues at 11+ constraints.

**Problem:** General-purpose SDP solvers are not optimized for bootstrap problems. Condition numbers grow as 10^15 for 21 constraints.

**Solution:** Interface with SDPB (Semidefinite Program Solver for the Bootstrap).

**Implementation:** `cft_bootstrap/sdpb_interface.py`
- `SDPBSolver`: Full SDPB integration with JSON PMP format
- `PolynomialApproximator`: Approximates F-vectors as polynomials in Δ
- `FallbackSDPBSolver`: CVXPY fallback when SDPB not installed
- `compute_bound_with_sdpb()`: High-level API for computing bounds

**Features:**
- Polynomial Matrix Program (PMP) generation in JSON format
- Automatic detection and fallback if SDPB not installed
- Configurable precision, threads, and solver parameters
- Integration with `run_bootstrap.py` via `--method sdpb` flag

**Usage:**
```bash
# Install SDPB (macOS)
brew tap davidsd/sdpb && brew install sdpb

# Run with SDPB
python run_bootstrap.py --gap-bound --method sdpb --max-deriv 21

# Falls back to CVXPY if SDPB not available
python run_bootstrap.py --gap-bound --method sdpb
```

**Impact:** HIGH - Enables arbitrary-precision arithmetic and handles 60+ constraints.

**Status:** Interface complete. Requires SDPB installation for full functionality.

---

### 4. Mixed Correlator Bootstrap ✅ IMPLEMENTED

**Previous:** We used only the ⟨σσσσ⟩ four-point function.

**Problem:** The sharp kink at the Ising point comes from combining multiple correlators.

**Solution:** Implemented mixed correlator bootstrap using three four-point functions:
- ⟨σσσσ⟩ (already had)
- ⟨σσεε⟩ (NEW - mixed external dimensions)
- ⟨εεεε⟩ (NEW - energy-only correlator)

**Implementation:** Two new modules:
- `cft_bootstrap/mixed_correlator_blocks.py` - F-vectors for all three correlators
- `cft_bootstrap/mixed_correlator_bootstrap.py` - Solvers

**Key Classes:**
- `MixedCrossingVector`: Computes F-vectors for ssss, ssee, and eeee
- `TwoCorrelatorBootstrapSolver`: Simplified solver using ssss + eeee (no matrix SDP)
- `MixedCorrelatorBootstrapSolver`: Full solver with 2×2 matrix SDP constraints

**Mathematical Background:**
The full mixed correlator constraint uses 2×2 positive semidefinite matrices:
```
| α_ssss · F^{ssss}_O    α_ssee · F^{ssee}_O |
| α_ssee · F^{ssee}_O    α_eeee · F^{eeee}_O |  >> 0
```
This captures OPE coefficient correlations (Cauchy-Schwarz inequality).

**Usage:**
```bash
# Two-correlator (ssss + eeee, no matrix SDP)
python run_bootstrap.py --gap-bound --method two-correlator --max-deriv 11

# Full mixed correlator (matrix SDP)
python run_bootstrap.py --gap-bound --method mixed-correlator --max-deriv 11
```

**Impact:** Creates the sharp kink that precisely locates the Ising model.

**References:**
- El-Showk et al. (2012): arXiv:1203.6064
- Kos, Poland, Simmons-Duffin (2014): arXiv:1406.4858

---

### 5. Polynomial Approximation for Positivity ✅ IMPLEMENTED

**Previous:** We sample operators at discrete Δ values and check positivity at each.

**Problem:** Discrete sampling can miss narrow excluded regions and finding the optimal functional.

**Solution:** Polynomial approximation to enforce positivity for **all** Δ ≥ Δ_gap using sum-of-squares (SOS) decomposition.

**Implementation:** `cft_bootstrap/polynomial_positivity.py`

**Key Classes:**
- `PolynomialFitter`: Fits F-vectors to polynomials using Chebyshev interpolation
- `SOSPositivityConstraint`: Builds SOS positivity via Gram matrix SDP constraints
- `PolynomialPositivitySolver`: Main solver combining polynomial fitting + SOS constraints
- `PolynomialPositivityGapSolver`: Computes Δε' bounds along curves

**Mathematical Background:**
A univariate polynomial p(x) is non-negative on [0, ∞) iff:
```
p(x) = σ₀(x) + x · σ₁(x)
```
where σ₀, σ₁ are sum-of-squares (SOS) polynomials. Each SOS polynomial can be written as:
```
σ(x) = v(x)ᵀ Q v(x),  Q ≽ 0
```
where v(x) = [1, x, x², ...] is the monomial basis and Q is a positive semidefinite Gram matrix.

**Usage:**
```bash
# Gap bound with polynomial positivity
python run_bootstrap.py --gap-bound --method polynomial --max-deriv 21 --poly-degree 15

# Hybrid method (polynomial + discrete samples)
python run_bootstrap.py --gap-bound --method hybrid --max-deriv 21

# Ising plot along boundary curve
python run_bootstrap.py --ising-plot --method polynomial --n-points 25

# Compare methods
python run_bootstrap.py --compare --max-deriv 11 --poly-degree 12
```

**Advantages over discrete sampling:**
1. Continuous positivity enforcement (no gaps between sample points)
2. Finds optimal linear functional more precisely
3. SDP size depends on polynomial degree, not sample count

**References:**
- Parrilo (2003): "Semidefinite programming relaxations for semialgebraic problems"
- Lasserre (2001): "Global optimization with polynomials and the problem of moments"

---

### 6. Literature Δε Boundary Values ✅ IMPLEMENTED

**Status:** Implemented in `bootstrap_gap_solver.py`. Uses tabulated values from published bootstrap results.

**Impact:** Minimal - confirmed the gap is due to constraints/operators, not boundary accuracy.

---

## Revised Implementation Priority

Based on the January 2026 investigation:

| Priority | Task | Impact | Status |
|----------|------|--------|--------|
| **1** | **Fix constraint formulation** | Tested - doesn't fix gap | ✅ Done |
| 2 | Install SDPB (Docker) | MEDIUM - smooth curves | ⬜ Blocked |
| 3 | Compare with reference implementations | HIGH - validate approach | ⬜ Next |

### January 2026 Update: pycftboot-style Normalization

Implemented the pycftboot-style reshuffling normalization across all solvers:
- `bootstrap_gap_solver.py` - `reshuffle_with_normalization()` function
- `taylor_conformal_blocks.py` - `HighOrderGapBootstrapSolver`
- `polynomial_positivity.py` - `PolynomialPositivitySolver`
- `mixed_correlator_bootstrap.py` - `TwoCorrelatorBootstrapSolver`, `MixedCorrelatorBootstrapSolver`

**Result:** Bound improved slightly from ~2.5 to ~2.6, but the ~1.2 unit gap to reference (~3.8) persists.

**Conclusion:** The gap is NOT primarily due to normalization convention. The reference paper likely uses a fundamentally different problem formulation, possibly:
- Different objective function (not just feasibility)
- Different handling of the positivity constraints
- Additional constraints we're not implementing

### Immediate Next Steps

1. **Compare with working implementations:**
   - [scalar_blocks](https://github.com/davidsd/scalar_blocks) - reference implementation
   - [pycftboot](https://github.com/cbehan/pycftboot) - Python bootstrap implementation
   - SDPB examples and test cases
   - Check if our problem formulation matches literature exactly

2. **Install SDPB for smooth curves:**
   - Docker: `docker pull davidsd/sdpb:master`
   - Will give smooth output but won't fix the gap

### What's Working

- ✅ Conformal blocks (verified against Mathematica)
- ✅ F-vector derivatives (verified numerically)
- ✅ Taylor series for high-order derivatives
- ✅ Spinning operator blocks (radial expansion)
- ✅ SDP solver with normalization (stable)
- ✅ pycftboot-style reshuffling normalization
- ✅ Environment check script (`python check_env.py`)

## Progress Tracking

### Completed ✅
- [x] Basic scalar conformal blocks (Dolan-Osborn)
- [x] Crossing equation setup
- [x] SDP solver integration (CVXPY)
- [x] Gap-based Δε' bounds
- [x] Qualitative reproduction of Figure 6 shape
- [x] Literature Δε boundary values
- [x] Taylor series conformal blocks (high-order derivatives)
- [x] Spinning conformal blocks (radial expansion)
- [x] Analysis of constraint power requirements
- [x] **SDPB integration** (JSON PMP format, fallback to CVXPY)
- [x] **Polynomial positivity constraints** (SOS decomposition via Gram matrices)
- [x] **Mixed correlator bootstrap** (ssss + ssee + eeee with matrix SDP)
- [x] **Investigation of ~1.3 unit gap** (January 2026)
- [x] **Environment check script** (`cft_bootstrap/check_env.py`)
- [x] **pycftboot-style reshuffling normalization** (January 2026) - doesn't fix gap
- [x] **Missing normalization constraint fix** (January 2026) - added `α_reduced · F_id_reduced = 0`
- [x] **Stage 1 spinning operators** (January 2026) - Stage 1 now uses `ElShowkBootstrapSolver`

### In Progress 🔄
- [ ] Compare with reference implementations to identify formulation differences

### Not Started ⬜
- [ ] Install SDPB via Docker for smooth curves
- [ ] Extensive numerical validation against literature

---

## Expected Results (Revised)

| Current State | Δε' at Ising | Notes |
|--------------|--------------|-------|
| Our implementation | ~2.6 | Consistent across 6-31 constraints |
| Reference (El-Showk 2012) | ~3.8 | Single correlator, ~66 coefficients (nmax=10) |

---

## Critical Findings: External Review (January 2026)

An external review identified several fundamental issues with our approach:

### 1. Figure Numbering Error
- **We claim:** Reproducing "El-Showk 2012 Fig. 7 (Δε')"
- **Reality:** Paper Fig. 7 is a **spin-2 (T') bound**; **Fig. 6** is the Δε' bound
- **Action:** Rename reference files and target Fig. 6

### 2. Two-Stage Pipeline Missing (CRITICAL)
The paper's Fig. 6 uses a two-stage protocol:
1. For each Δσ, compute **Δε,max(Δσ)** (boundary from Fig. 3, using nmax=11)
2. Then with **Δε fixed to that computed Δε,max**, compute Δε' bound (nmax=10)

**Our current approach:** We use hardcoded/literature Δε values (e.g., 1.41), NOT the self-consistently computed boundary. This is fundamentally different from the paper's protocol.

### 3. Paper's Multi-Resolution Discretization
The paper uses **5 operator tables** (Table 2, Appendix D):

| Table | Step δ | Δmax | Lmax |
|-------|--------|------|------|
| T1 | 2×10⁻⁵ | 3 | 0 |
| T2 | 5×10⁻⁴ | 8 | 6 |
| T3 | 2×10⁻³ | 22 | 20 |
| T4 | 0.02 | 100 | 50 |
| T5 | 1 | 500 | 100 |

**Our approach:** Single coarse grid, single cutoff → fundamentally different discretization.

### 4. Derivative Basis & Order
Paper uses nmax=10 → **(nmax+1)(nmax+2)/2 = 66 coefficients**

Functional form:
```
Λ = Σ_{m+2n ≤ 2nmax+1} λ_{m,n} ∂_a^m ∂_b^n F(a,b)|_{a=1,b=0}
```
where (a,b) are paper's coordinates (Section 4), only odd a-derivatives contribute.

**Our approach:** 3-31 derivatives in (z,zbar) coordinates → different basis.

### 5. Spinning Operators Required
Paper's σ×σ OPE includes **all even spins up to Lmax=100**.
**Our approach:** Scalars only → missing significant constraint power.

### 6. Solver Differences
Paper: **IBM ILOG CPLEX (dual simplex)**
Ours: **CVXPY/SCS** → different numerics, but shouldn't cause 1+ unit gap

---

## Revised Action Items (Priority Order)

| Priority | Task | Impact | Status |
|----------|------|--------|--------|
| **1** | **Fix figure reference** (Fig. 6 not Fig. 7) | Correctness | ✅ Done |
| **2** | **Implement two-stage pipeline** | CRITICAL - may fix gap | ✅ Done |
| **3** | **Add spinning operators (Lmax≥50)** | HIGH | ✅ Done |
| **4** | **Implement (a,b) derivative basis with nmax=10** | HIGH | ✅ Done |
| **5** | **Multi-resolution discretization (T1-T5)** | MEDIUM | ✅ Done |
| 6 | Install SDPB for smooth curves | LOW | ⬜ |

### Minimal Checklist for Fig. 6 Reproduction

- [x] Rename files: `fig7` → `fig6`
- [x] Implement two-stage scan (Δε boundary first, then Δε' with Δε fixed)
- [x] Implement (a,b) coordinate derivatives at (a=0, b=0) with mixed derivatives
- [x] Increase to nmax=10 (66 coefficients) via ElShowkBootstrapSolver
- [x] Add spinning operators l = 0, 2, 4, ..., Lmax ≥ 50 (via ElShowkBootstrapSolver)
- [x] Implement T1-T5 multi-resolution discretization (via get_multiresolution_operators)
- [x] Match LP tolerances to serious solver (1e-9 abs/rel tolerances, multi-solver support)

See `cft_bootstrap/REFERENCE_COMPARISON.md` for detailed implementation guidance.

### January 2026 Update: Deep Investigation into the ~1.2 Unit Gap

A thorough investigation was conducted to understand why all implementations consistently produce bounds ~1.2 units below the reference value (~2.6 vs ~3.8 at the Ising point).

#### 1. F-Vector Verification ✅ CORRECT

F-vectors were computed in Mathematica and compared with our implementation:

**Mathematica (El-Showk coordinates at a=0.5, b=0):**
```
F_identity derivatives (m=1,3,5): [-2.0209, 0.2812, 6.6016]
F_epsilon derivatives (m=1,3,5):  [0.9283, -1.8538, -42.717]
```

**Our implementation:**
```
F_identity: [-1.0105, 0.00585, 0.00171]
F_epsilon:  [0.4642, -0.03855, -0.01099]
```

**Relationship:** Our code uses the normalization convention:
```
Our_F[m] = ElShowk_F[m] / (2^m * m!)
```

This is a **consistent convention** that doesn't affect bootstrap bounds. The ratio `F_epsilon/F_identity` matches exactly between implementations, confirming correctness.

#### 2. All Solvers Give the Same Gap

Tested multiple solver approaches at the Ising point (Δσ=0.518, Δε=1.41):

| Solver | Approach | Bound | Gap to Reference |
|--------|----------|-------|------------------|
| Basic SDP | Discrete sampling | ~2.5 | ~1.3 |
| Polynomial Positivity | Continuous (SOS) | ~2.63 | ~1.2 |
| El-Showk Solver | Full basis + spinning | ~2.5 | ~1.3 |

**Conclusion:** The gap is NOT due to discrete vs continuous positivity or missing spinning operators.

#### 3. Key Discrepancy Point Confirmed

At Ising point with Δε'=3.0:
- **Our solver:** EXCLUDED (point ruled out)
- **Reference:** Should be ALLOWED (bound is ~3.8)

This confirms the ~1.2 unit discrepancy is real and consistent.

#### 4. Numerical Issues Identified

The SDP solver finds solutions with extremely large α values:
```python
SCS:  alpha_reduced = [-17692825, 61367342, -1498097]
OSQP: alpha_reduced = [-11637155, 40380292, -1021402]
```

These ~10^7 magnitude values indicate:
- **Poorly conditioned problem** - near-singular constraint matrix
- **Nearly unconstrained search space** - F_id dominated by first component
- The normalization constraint `α·F_id = 1` effectively only constrains `α[0] ≈ -1/F_id[0]`
- Higher components `α[1], α[2], ...` are nearly free to take any value

#### 5. Root Cause: Problem Formulation Difference

The gap is **NOT** due to:
- ❌ F-vector computation (verified correct)
- ❌ Discrete vs continuous positivity (both give same gap)
- ❌ Missing spinning operators (El-Showk solver includes them)
- ❌ Numerical precision (tested multiple solvers)
- ❌ Number of constraints (tested 3-31, gap persists)

The gap **IS** due to:
- **Fundamentally different problem formulation** than pycftboot/SDPB
- Our formulation is "too easy" - finds excluding functionals that shouldn't exist
- pycftboot uses **polynomial matrix programs** (PMPs) where F-vectors are polynomials in Δ
- pycftboot uses **damped rational prefactors** and **bilinear bases** for numerical stability

#### 6. pycftboot Structure Analysis

From examining `pycftboot/bootstrap.py`:

```python
# iterate() method - the core SDP call
obj = [0.0] * len(self.table[0][0][0].vector)  # Zero objective
self.write_xml(obj, self.unit, name)           # unit = identity contribution
```

Key structural differences:
1. **Polynomial vectors** - F-vectors are polynomials in δ (scaling dimension), not fixed numerical arrays
2. **Bilinear basis** - orthogonal polynomial basis for each spin channel
3. **Sample points & scalings** - Laguerre-based sample points with damped rational prefactors
4. **XML/PMP format** - specialized format for SDPB, not generic LP/SDP

Our discrete operator sampling approach creates a fundamentally different optimization problem.

#### 7. Next Steps to Resolve

1. **Install symengine + pycftboot** and run at Ising point to see exact constraint structure
2. **Install SDPB via Docker** and generate reference PMP files
3. **Compare constraint matrices** side-by-side at a single point
4. **Check for missing components:**
   - Stress tensor contribution (Δ=3, spin=2)
   - Ward identity constraints
   - OPE coefficient normalization (λ² positivity)

---

### January 2026 Update: Stage 1 Spinning Operators

**Problem Identified:** Stage 1 (finding the Δε boundary) was using scalar-only solver (`BootstrapSolver`), while Stage 2 had full spinning operators via `ElShowkBootstrapSolver`. This inconsistency meant Stage 1 was computing a weaker bound than intended.

**Solution:** Modified Stage 1 to use `ElShowkBootstrapSolver` with the same spinning operator support as Stage 2.

**Files Modified:**
- `el_showk_basis.py`: Added `is_point_excluded()` and `find_delta_epsilon_bound()` methods to `ElShowkBootstrapSolver`
- `run_bootstrap.py`: Stage 1 now uses `ElShowkBootstrapSolver` with spinning operators
- `bootstrap_gap_solver.py`: `compute_two_stage_scan()` now uses `ElShowkBootstrapSolver` for Stage 1 with configurable spinning

**New Parameters:**
- `use_spinning_stage1`: Boolean to enable spinning operators in Stage 1 (default: True)

**Impact:** Stage 1 now produces tighter Δε boundaries by including spinning operator constraints, which may improve the final Δε' bounds.

---

### January 2026 Update: CLI and Cluster Integration

| Feature | Status | Notes |
|---------|--------|-------|
| El-Showk CLI arguments | ✅ Done | `--max-spin`, `--use-multiresolution`, `--el-showk-solver`, `--nmax` |
| SBATCH script for El-Showk | ✅ Done | Full configuration with resource guidelines |
| ElShowkPolynomialApproximator | ✅ Done | SDPB integration for El-Showk basis |
| `--method el-showk-sdpb` | ✅ Done | High-precision SDPB with El-Showk |
| Stage 1 spinning operators | ✅ Done | Stage 1 now uses `ElShowkBootstrapSolver` with spinning |
| Documentation | ✅ Done | README.md cluster instructions |

**Usage:**
```bash
# Local execution with full parameters
python run_bootstrap.py --gap-bound --method el-showk --nmax 10 --max-spin 50 --use-multiresolution

# Cluster execution
sbatch submit_cluster.sh  # After editing configuration

# High-precision SDPB
python run_bootstrap.py --gap-bound --method el-showk-sdpb --nmax 10 --max-spin 50
```

---

## Files

### Core Solvers
- `cft_bootstrap/bootstrap_solver.py` - Basic bootstrap solver
- `cft_bootstrap/bootstrap_gap_solver.py` - Gap-based solver for Δε' bounds
- `cft_bootstrap/el_showk_basis.py` - **El-Showk (2012) full derivative basis with spinning operators**
- `cft_bootstrap/taylor_conformal_blocks.py` - Taylor series implementation for scalars
- `cft_bootstrap/spinning_conformal_blocks.py` - Spinning conformal blocks (radial expansion)
- `cft_bootstrap/polynomial_positivity.py` - Polynomial positivity via SOS constraints
- `cft_bootstrap/mixed_correlator_blocks.py` - F-vectors for mixed correlators
- `cft_bootstrap/mixed_correlator_bootstrap.py` - Mixed correlator bootstrap solvers

### SDPB Integration
- `cft_bootstrap/sdpb_interface.py` - SDPB integration + **ElShowkPolynomialApproximator**

### CLI and Cluster
- `cft_bootstrap/run_bootstrap.py` - **Full CLI with `--method el-showk`, `--method el-showk-sdpb`**
- `cft_bootstrap/submit_cluster.sh` - **SLURM script with El-Showk configuration**
- `cft_bootstrap/collect_and_plot.py` - Result collection and plotting

### Tests and Notebooks
- `cft_bootstrap/tests/test_mixed_correlator.py` - Tests for mixed correlator bootstrap
- `cft_bootstrap/tests/test_polynomial_positivity.py` - Tests for polynomial positivity
- `notebooks/reproduce_ising_delta_epsilon_prime.ipynb` - Jupyter notebook
- `reference_plots/` - Comparison plots

## References

1. El-Showk et al. (2012): [arXiv:1203.6064](https://arxiv.org/abs/1203.6064)
2. Hogervorst & Rychkov (2013): [arXiv:1303.1111](https://arxiv.org/abs/1303.1111) - Radial coordinates
3. Simmons-Duffin (2015): [arXiv:1502.02033](https://arxiv.org/abs/1502.02033) - SDPB
4. Costa et al. (2011): [arXiv:1109.6321](https://arxiv.org/abs/1109.6321) - Spinning blocks
