# CFT Bootstrap: Status & Changelog

## Current Status (February 2, 2026)

### Summary

| Component | Status | Notes |
|-----------|--------|-------|
| SDPB Container | **Working** | Singularity on FASRC, completes in ~2s for small problems |
| CVXPY Baseline | **Working** | Δε' ≤ 2.53 at Ising point |
| El-Showk + SDPB | **Broken** | Numerical formulation error (Q diagonal = 0) |
| Large Parameter Tests | **Slow** | PMP build takes minutes due to mpmath |

### Key Finding (February 2, 2026)

**The 8-hour timeout was NOT caused by MPI hanging.**

It was caused by **slow PMP construction** with high-precision mpmath:
- With `nmax=5, max_spin=10, poly_degree=15`: ~98 F-vector computations per SDPB call
- Each high-precision F-vector takes seconds with mpmath
- Total PMP build time: **minutes per call** (before SDPB even runs)

**SDPB itself is fast** - the quick test completes in 1.9 seconds.

---

## Test Results

### Working Tests

| Test | Parameters | Result | Time |
|------|------------|--------|------|
| `test_sdpb_quick.sh` | nmax=3, max_spin=2, poly_degree=8 | SDPB runs, returns "ALLOWED" | 1.9s |
| `quick_test.sh` | CVXPY discrete | Δε' ≤ 2.53 | ~10s |

### Failing Tests

| Test | Parameters | Issue |
|------|------------|-------|
| `test_fixed_prefactor.sh` | nmax=5, max_spin=10 | PMP build too slow (hangs for hours) |
| `test_small_params.sh` | nmax=3, max_spin=4 | SDPB error: "Q diagonal should be 1" |

---

## Known Issues

### Issue 1: Slow PMP Construction (CRITICAL)

**Problem**: `ElShowkPolynomialApproximator.build_polynomial_matrix_program()` is extremely slow with large parameters.

**Why**: For each SDPB check, it computes:
- `(poly_degree + 1)` sample points × `(max_spin/2 + 1)` spin channels
- Each sample calls `crossing.build_F_vector()` with high-precision mpmath
- Example: nmax=5, max_spin=10, poly_degree=15 → **96 F-vector computations**

**Workaround**: Use small parameters (nmax≤3, max_spin≤4, poly_degree≤8)

**Fix needed**: Cache polynomial approximations or pre-compute outside binary search loop.

### Issue 2: SDPB Q-Matrix Error (HIGH)

**Problem**: SDPB fails with:
```
Assertion 'diff < eps' failed:
  Normalized Q should have ones on diagonal. For i = 0: Q_ii = 0
```

**Cause**: The PMP formulation produces degenerate constraint matrices.

**Related**: FIX_NOTES previously mentioned:
- Missing bilinear basis (pycftboot uses Cholesky orthogonalization)
- Missing pole structure in prefactor
- Numerical vs symbolic polynomial representation

### Issue 3: "ALLOWED" When Should Be "EXCLUDED"

The quick SDPB test returns "ALLOWED" for Δε'=6.0, which should be excluded.
This indicates the optimization problem isn't correctly formulated.

---

## Bug Fixes Applied

### 1. Prefactor Base (January 28, 2026)
- **Bug**: Used `exp(-1) ≈ 0.368` for damped rational prefactor
- **Fix**: Changed to `R_CROSS = 3 - 2*sqrt(2) ≈ 0.172`
- **File**: `sdpb_interface.py`

### 2. MPI Conditional (January 29, 2026)
- **Bug**: Always used MPI even for `num_threads=1`
- **Fix**: `use_mpi = num_threads > 1`
- **File**: `sdpb_interface.py`

### 3. Singularity MPI Bindings (February 2, 2026)
- **Addition**: Added `/tmp`, `/dev/shm` bindings and `--network=host` for MPI mode
- **File**: `sdpb_interface.py` (`_run_singularity()`)
- **Note**: This fix is correct but wasn't the cause of the original timeout

### 4. Conda Path Hardcoding (February 2, 2026)
- **Bug**: Hardcoded `/n/sw/Miniforge3-24.7.1-0/etc/profile.d/conda.sh`
- **Fix**: Try user installations first, then system
- **Files**: All shell scripts (`*.sh`)

---

## Architecture Notes

### Why CVXPY Works But SDPB Fails

CVXPY uses **discrete operator sampling** - it checks positivity at specific Δ values.
This works but produces weak bounds (Δε' ≤ 2.53 instead of ~3.8).

SDPB uses **polynomial positivity** - it requires F(Δ) ≥ 0 for ALL Δ in a range.
This is mathematically stronger but requires correct polynomial representation:
1. Exact symbolic polynomials (via Zamolodchikov recursion)
2. Orthogonal bilinear basis (via Cholesky decomposition)
3. Correct pole structure in damped rational prefactor

Our current `ElShowkPolynomialApproximator` uses numerical Chebyshev interpolation,
which doesn't satisfy these requirements.

### The Real Fix

Use `SymbolicPolynomialApproximator` with pycftboot's exact polynomial computation:
```bash
# This path exists but isn't fully integrated
python run_bootstrap.py --method symbolic-sdpb ...
```

Or fix `ElShowkPolynomialApproximator` to:
1. Use symbolic polynomials instead of numerical interpolation
2. Add proper bilinear basis orthogonalization
3. Include conformal block poles in prefactor

---

## File Status

### Test Scripts
| Script | Status | Use For |
|--------|--------|---------|
| `quick_test.sh` | Working | CVXPY baseline verification |
| `test_sdpb_quick.sh` | Working | SDPB container verification |
| `test_fixed_prefactor.sh` | Broken | Don't use until PMP speed fixed |
| `test_small_params.sh` | Broken | Demonstrates Q-matrix error |

### Key Source Files
| File | Status |
|------|--------|
| `sdpb_interface.py` | Has MPI bindings fix, but PMP formulation broken |
| `polynomial_bootstrap.py` | Has correct infrastructure, needs integration |
| `pycftboot_bridge.py` | Working, provides exact polynomials |
| `el_showk_basis.py` | Working for CVXPY, not for SDPB |

---

## Next Steps (Priority Order)

1. **Fix PMP formulation** - Either integrate `SymbolicPolynomialApproximator` or fix numerical approximation
2. **Add polynomial caching** - Don't recompute F-vectors for every binary search iteration
3. **Validate with small problem** - Get nmax=3, max_spin=2 working end-to-end
4. **Scale up parameters** - Only after small problems work correctly
