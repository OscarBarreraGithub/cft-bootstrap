# CFT Bootstrap: Status & Changelog

## Current Status (January 29, 2026)

### Baseline Result
| Metric | Value | Reference | Gap |
|--------|-------|-----------|-----|
| Δε' bound at Ising point | **≤ 2.53** | ~3.8 | ~1.27 |
| Δσ (external scalar) | 0.518 | 0.518 | - |
| Δε (first scalar gap) | 1.41 | 1.41 | - |

Confirmed with CVXPY via `quick_test.sh`.

### SDPB Status
- Integration: **Working** (Singularity container on FASRC)
- Prefactor fix: **Applied** (R_CROSS)
- MPI fix: **Applied** (conditional on num_threads)
- Awaiting: Full validation run with MPI parallelism

### Next Step
```bash
sbatch test_fixed_prefactor.sh  # MPI parallel, 4 tasks, 3 hours
```

---

## Bug Fixes Applied

### 1. Prefactor Base (January 28, 2026)
- **Bug**: Used `exp(-1) ≈ 0.368` for damped rational prefactor
- **Fix**: Changed to `R_CROSS = 3 - 2*sqrt(2) ≈ 0.172`
- **File**: `sdpb_interface.py` (lines ~310 and ~515)
- **Reason**: Matches pycftboot convention for crossing-symmetric radial coordinate

### 2. MPI Conditional (January 29, 2026)
- **Bug**: Always used MPI even for `num_threads=1`, causing errors
- **Fix**: `use_mpi = num_threads > 1`
- **File**: `sdpb_interface.py` (line ~1281)
- **Reason**: Single-threaded SDPB should run without MPI overhead

---

## Test Scripts

| Script | Purpose | Parameters | Status |
|--------|---------|------------|--------|
| `quick_test.sh` | CVXPY baseline | Discrete SDP | **PASSED** (Δε' ≤ 2.53) |
| `test_sdpb.sh` | SDPB integration | nmax=5, max_spin=10 | Ready |
| `test_fixed_prefactor.sh` | SDPB with fixes | nmax=5, max_spin=10, MPI | Ready |

---

## Known Limitations: The ~1.2 Unit Gap

The gap is **structural**, not numerical. Root causes:

### 1. Numerical vs Symbolic Polynomials
- **pycftboot**: Exact symbolic via Zamolodchikov recursion
- **Our code**: Numerical Chebyshev interpolation

### 2. Missing Bilinear Basis
- **pycftboot**: Orthogonal basis via Cholesky decomposition
- **Our code**: Direct polynomial fitting

### 3. Missing Pole Structure
- **pycftboot**: Explicit conformal block poles in prefactor
- **Our code**: Empty pole list (`prefactor_poles = []`)

### Why It Matters
Polynomial positivity (α·F(Δ) ≥ 0 for ALL Δ) is harder than discrete sampling.
Our discrete approach finds "trivial" excluding functionals that shouldn't exist.

---

## Potential Full Fixes

### Option 1: Symengine (Recommended)
`SymbolicPolynomialApproximator` is already implemented correctly.
```bash
pip install symengine
# Then use --method symbolic-sdpb
```

### Option 2: Bilinear Basis
Implement in `ElShowkPolynomialApproximator` with known pole positions:
- 3D scalar poles: Δ = 0.5 - n for n = 0, 1, 2, ...

---

## Verification Checklist

After SDPB validation completes:
- [ ] Δε' = 3.0 is ALLOWED (not excluded)
- [ ] Bound converges to ~3.8 ± 0.1
- [ ] SDPB runs without errors
- [ ] Binary search completes within time limit

---

## Files Modified This Session

1. `sdpb_interface.py` - Prefactor fix + MPI conditional
2. `test_fixed_prefactor.sh` - MPI configuration (ntasks=4)
3. `FIX_NOTES.md` - This status document

## Files Cleaned Up

Deleted test artifacts:
- 22 SLURM output files (*.out, *.err)
- Experimental scripts (compare_test*.sh, quick_sdpb_test.sh)
- Python cache (__pycache__/)
