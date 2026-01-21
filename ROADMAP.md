# Roadmap: Matching El-Showk et al. (2012) Figure 7

This document outlines what is needed to exactly reproduce the Δε' bounds from:

> **"Solving the 3D Ising Model with the Conformal Bootstrap"**
> S. El-Showk, M. Paulos, D. Poland, S. Rychkov, D. Simmons-Duffin, A. Vichi
> [arXiv:1203.6064](https://arxiv.org/abs/1203.6064) (2012), Figure 7

## Current Status

| Metric | Our Implementation | Reference | Gap |
|--------|-------------------|-----------|-----|
| Δε' at Ising (Δσ=0.518) | ~2.5 | ~3.8 | ~1.3 |
| Derivative constraints | 6-11 | ~60+ | ~6-10x fewer |
| Operator types | Scalars + spinning (ℓ=0,2,4,6) | All spins | Covered |
| Numerical stability | max_deriv ~ 21 | max_deriv ~ 60 | SDP conditioning issues |

## Key Finding: Spinning Operators are Critical

**Important discovery:** Adding more derivative constraints (via Taylor series) does NOT significantly improve bounds. Testing with 3, 6, and 11 constraints shows:

- 3 constraints → Δε' ~ 2.6 at Ising
- 6 constraints → Δε' ~ 2.2 at Ising
- 11 constraints → Δε' ~ 2.2 at Ising

More constraints actually make bounds **tighter** (lower), not higher! This is because we're missing the positive contribution from **spinning operators** (stress tensor Δ=3, ℓ=2, etc.). The bootstrap functional finds more ways to exclude configurations when we have more constraints but only scalar positivity.

**The ~1.5 unit gap to the reference is primarily due to missing spinning operators, not insufficient derivative constraints.**

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

### 3. SDPB Integration ⬜ NOT STARTED

**Current:** CVXPY with SCS solver shows conditioning issues at 11+ constraints.

**Problem:** General-purpose SDP solvers are not optimized for bootstrap problems. Condition numbers grow as 10^15 for 21 constraints.

**Solution:** Interface with SDPB (Semidefinite Program Solver for the Bootstrap).

**Impact:** MEDIUM - Better numerical precision, faster computation.

**Complexity:** MEDIUM - requires installing SDPB and writing an interface.

---

### 4. Mixed Correlator Bootstrap ⬜ NOT STARTED

**Current:** We use only the ⟨σσσσ⟩ four-point function.

**Problem:** The sharp kink at the Ising point comes from combining multiple correlators.

**Solution:** Include ⟨σσεε⟩ and ⟨εεεε⟩ correlators in the bootstrap system.

**Impact:** Creates the sharp kink that precisely locates the Ising model.

**Complexity:** HIGH - requires setting up a system of crossing equations.

---

### 5. Polynomial Approximation for Positivity ⬜ NOT STARTED

**Current:** We sample operators at discrete Δ values and check positivity at each.

**Problem:** Discrete sampling can miss narrow excluded regions.

**Solution:** Use polynomial approximation to enforce positivity for all Δ ≥ Δ_gap.

**Impact:** More rigorous bounds, matches the paper's methodology exactly.

**Complexity:** MEDIUM - requires polynomial fitting and matrix positivity constraints.

---

### 6. Literature Δε Boundary Values ✅ IMPLEMENTED

**Status:** Implemented in `bootstrap_gap_solver.py`. Uses tabulated values from published bootstrap results.

**Impact:** Minimal - confirmed the gap is due to constraints/operators, not boundary accuracy.

---

## Revised Implementation Priority

Based on current analysis showing the ~1.3 unit gap is likely due to insufficient constraint power and numerical issues:

| Priority | Improvement | Impact | Complexity |
|----------|-------------|--------|------------|
| **1** | **SDPB integration** | **HIGH** | MEDIUM |
| 2 | Polynomial approximation | HIGH | MEDIUM |
| 3 | More derivative constraints | MEDIUM | LOW |
| 4 | Mixed correlator bootstrap | HIGH | HIGH |

**Note:** Both Taylor series and spinning operators have been implemented. The remaining gap is likely due to:
1. Insufficient number of constraints (6 vs ~60 in reference)
2. SDP solver conditioning issues
3. Discrete sampling vs polynomial positivity

## Progress Tracking

### Completed ✅
- [x] Basic scalar conformal blocks (Dolan-Osborn)
- [x] Crossing equation setup
- [x] SDP solver integration (CVXPY)
- [x] Gap-based Δε' bounds
- [x] Qualitative reproduction of Figure 7 shape
- [x] Literature Δε boundary values
- [x] Taylor series conformal blocks (high-order derivatives)
- [x] Spinning conformal blocks (radial expansion)
- [x] Analysis of constraint power requirements

### In Progress 🔄
- [ ] None currently

### Not Started ⬜
- [ ] SDPB integration (highest priority for precision)
- [ ] Polynomial approximation for positivity
- [ ] Mixed correlator bootstrap

---

## Expected Results After Each Improvement

| After Implementing | Expected Δε' at Ising | Gap to Reference |
|--------------------|----------------------|------------------|
| Current (scalars + spinning, 6 constraints) | ~2.5 | ~1.3 |
| + SDPB + 20 constraints | ~3.0-3.3 | ~0.5-0.8 |
| + Polynomial positivity | ~3.5-3.7 | ~0.1-0.3 |
| + All improvements | ~3.8 | ~0 |

---

## Files

- `cft_bootstrap/bootstrap_solver.py` - Basic bootstrap solver
- `cft_bootstrap/bootstrap_gap_solver.py` - Gap-based solver for Δε' bounds
- `cft_bootstrap/taylor_conformal_blocks.py` - Taylor series implementation for scalars
- `cft_bootstrap/spinning_conformal_blocks.py` - Spinning conformal blocks (radial expansion)
- `notebooks/reproduce_ising_delta_epsilon_prime.ipynb` - Jupyter notebook
- `reference_plots/` - Comparison plots

## References

1. El-Showk et al. (2012): [arXiv:1203.6064](https://arxiv.org/abs/1203.6064)
2. Hogervorst & Rychkov (2013): [arXiv:1303.1111](https://arxiv.org/abs/1303.1111) - Radial coordinates
3. Simmons-Duffin (2015): [arXiv:1502.02033](https://arxiv.org/abs/1502.02033) - SDPB
4. Costa et al. (2011): [arXiv:1109.6321](https://arxiv.org/abs/1109.6321) - Spinning blocks
