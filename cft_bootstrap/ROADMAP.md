# Roadmap: Accurate CFT Bootstrap Reproduction

## Problem Summary

**Current State**: Δε' bounds are ~1.2 units too low (getting ~2.5-2.6 instead of reference ~3.8)

**Root Cause**: The implementation uses **discrete operator sampling** while pycftboot/SDPB use **Polynomial Matrix Programs (PMPs)**. This creates a fundamentally weaker optimization problem where the solver can find "loopholes" between sample points.

**What's Been Verified Correct**:
- F-vectors and conformal blocks (match Mathematica to 10+ digits)
- Crossing equation formulation
- Spinning operators, multi-resolution discretization
- High-precision numerical mode

**What Needs to Change**: Transition from discrete sampling to continuous polynomial representation.

---

## Implementation Progress

### Completed (Phase 1-2)

| Task | Status | Files |
|------|--------|-------|
| Add symengine/mpmath dependencies | ✅ Done | [requirements.txt](requirements.txt) |
| Create `SymbolicPolynomialVector` class | ✅ Done | [polynomial_bootstrap.py](polynomial_bootstrap.py) |
| Implement `ConformalBlockPoles` for pole computation | ✅ Done | [polynomial_bootstrap.py](polynomial_bootstrap.py) |
| Implement `BilinearBasis` with Gram matrix + Cholesky | ✅ Done | [polynomial_bootstrap.py](polynomial_bootstrap.py) |
| Implement damped rational prefactor | ✅ Done | [polynomial_bootstrap.py](polynomial_bootstrap.py) |
| Implement Laguerre sample points | ✅ Done | [polynomial_bootstrap.py](polynomial_bootstrap.py) |
| Implement `PMPGenerator` for SDPB JSON output | ✅ Done | [polynomial_bootstrap.py](polynomial_bootstrap.py) |
| Create validation test suite | ✅ Done | [tests/test_polynomial_bootstrap.py](tests/test_polynomial_bootstrap.py) |
| Install symengine with MPFR | ✅ Done | `pip install symengine` |
| Create pycftboot bridge | ✅ Done | [pycftboot_bridge.py](pycftboot_bridge.py) |

**All tests pass. pycftboot bridge successfully builds symbolic block tables with 200+ digit precision.**

### Remaining Work (Phase 3)

| Task | Status | Priority | Notes |
|------|--------|----------|-------|
| Update `sdpb_interface.py` | 🔄 In Progress | Medium | Connect polynomial infrastructure |
| Test Ising point bound | 🔲 Pending | High | Key validation: should get ~3.8 not 2.6 |
| Full Figure 6 reproduction | 🔲 Pending | Final | End goal |

---

## Architecture

### New Module: `polynomial_bootstrap.py`

```
polynomial_bootstrap.py
├── Constants
│   ├── PREC = 660 bits (~200 decimal digits)
│   └── R_CROSS = 3 - 2√2 ≈ 0.172
│
├── Utility Functions
│   ├── rising_factorial()      - Pochhammer symbol
│   ├── unitarity_bound()       - Δ bounds by spin
│   ├── gather_poles()          - Count multiplicities
│   └── coefficients_from_polynomial()
│
├── Core Classes
│   ├── SymbolicPolynomialVector  - F-vector as polynomial
│   ├── ConformalBlockPoles       - Pole computation
│   ├── SymbolicConformalBlockTable - Block derivatives (placeholder)
│   └── ConvolvedBlockTable       - Crossing convolution
│
├── Bilinear Basis
│   └── BilinearBasis
│       ├── _compute_gram_matrix()   - ⟨x^i, x^j⟩
│       ├── _compute_integral()      - Damped rational integral
│       ├── _compute_basis()         - Cholesky decomposition
│       └── transform_polynomial()   - To orthogonal basis
│
├── PMP Generation
│   └── PMPGenerator
│       ├── make_laguerre_points()   - Sample points
│       ├── shifted_prefactor()      - r_cross^(x+s) / ∏(x-p)
│       ├── reshuffle_with_normalization()
│       └── write_json()             - SDPB format output
│
└── High-Level Interface
    └── PolynomialBootstrapSolver
        ├── setup_problem()
        └── generate_pmp()
```

### Key Mathematical Concepts

**1. Polynomial F-vectors**
```
F_Δ = (polynomial in Δ) / (product of poles)
```
The poles come from conformal block recursion and are at specific values determined by spin and dimension.

**2. Bilinear Basis**
```
⟨p, q⟩ = ∫₀^∞ p(x) q(x) · (r_cross^x / ∏(x - pole_i)) dx
```
Orthogonalization via Cholesky: if G = L L^T, then B = L^{-1} gives orthonormal basis.

**3. PMP Format for SDPB**
- `control.json` - metadata
- `objectives.json` - objective vector
- `block_info_N.json` - matrix dimensions
- `block_data_N.json` - bilinear bases and constraints

---

## Next Steps (Recommended Order)

### 1. Install symengine (Required)

```bash
pip install symengine
```

If this fails, try with conda:
```bash
conda install -c conda-forge symengine
```

Or build from source with MPFR support:
```bash
git clone https://github.com/symengine/symengine.py
cd symengine.py
pip install .
```

### 2. Complete Zamolodchikov Recursion

The block computation in `SymbolicConformalBlockTable._build_table()` is currently a placeholder. Options:

**Option A (Recommended)**: Import pycftboot directly
```python
import sys
sys.path.insert(0, "reference_implementations/pycftboot")
from bootstrap import ConformalBlockTable
```

**Option B**: Port the full recursion from `blocks1.py` and `blocks2.py`. This is ~400 lines of complex code involving:
- `LeadingBlockVector` - leading block contribution
- `MeromorphicBlockVector` - meromorphic structure
- `ConformalBlockVector` - full block with poles
- `chain_rule_single/double` - derivative transformations

### 3. Connect to SDPB

Update `sdpb_interface.py` to use the new polynomial infrastructure:

```python
def compute_bound_polynomial(self, delta_sigma, delta_epsilon, ...):
    from polynomial_bootstrap import PolynomialBootstrapSolver

    solver = PolynomialBootstrapSolver(dim=3, k_max=20, ...)
    solver.setup_problem(delta_sigma=delta_sigma)
    solver.generate_pmp(output_dir=self.work_dir, bounds=bounds)

    # Call SDPB
    return self._run_sdpb()
```

### 4. Validation

Run at Ising point:
```bash
python run_bootstrap.py --polynomial --delta-sigma 0.518 --delta-epsilon 1.41
```

Expected: Δε' ≤ 3.8 (not 2.6)

---

## Reference Materials

- [pycftboot/bootstrap.py](reference_implementations/pycftboot/bootstrap.py) - Main reference (set_basis, write_json)
- [pycftboot/blocks1.py](reference_implementations/pycftboot/blocks1.py) - Zamolodchikov recursion
- [pycftboot/common.py](reference_implementations/pycftboot/common.py) - Constants and utilities
- [SDPB Documentation](https://github.com/davidsd/sdpb) - PMP format specification
- [El-Showk et al. (2012)](https://arxiv.org/abs/1203.6064) - Original paper
- [Kos, Poland, Simmons-Duffin (2014)](https://arxiv.org/abs/1406.4858) - Conformal block poles

---

## Summary

The infrastructure for polynomial bootstrap is now in place:
- ✅ Bilinear basis with Gram matrix and Cholesky decomposition
- ✅ Damped rational prefactor computation
- ✅ PMP file generation for SDPB
- ✅ Test suite with 15 passing tests

The main remaining work is:
1. **Install symengine** for full symbolic polynomial support
2. **Complete block computation** (either port recursion or import pycftboot)
3. **Connect to SDPB** and validate at Ising point

Once these are done, the ~1.2 unit gap should close, achieving accurate reproduction of El-Showk et al. (2012) Figure 6.
