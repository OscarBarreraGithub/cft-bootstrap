# Reference Implementation Comparison Guide

This document provides detailed instructions for comparing our conformal bootstrap implementation against reference implementations to identify why our Δε' bounds (~2.6) are ~1.2 units lower than the literature value (~3.8).

## Problem Statement

We are trying to reproduce Figure 7 from El-Showk et al. (2012) "Solving the 3D Ising Model with the Conformal Bootstrap" (arXiv:1203.6064).

**Expected result:** At the 3D Ising point (Δσ ≈ 0.518, Δε ≈ 1.41), the upper bound on Δε' should be approximately 3.8.

**Our result:** We get Δε' ≤ 2.6, consistently across different solvers and constraint counts.

**What we've ruled out:**
- Normalization convention (implemented pycftboot-style reshuffling)
- Number of constraints (tested 3-31 constraints, no improvement)
- Spinning operators (tested, no improvement)
- Numerical instability (fixed with proper normalization)

**What we suspect:** The problem formulation itself may differ from the literature.

---

## Reference Implementations to Compare

### 1. pycftboot (Python)
**Repository:** https://github.com/cbehan/pycftboot

**Key files to examine:**
- `cboot/scalar/bootstrap.py` - Main bootstrap solver
- `cboot/scalar/ope_scan.py` - How they set up the optimization problem
- `cboot/scalar/sdp.py` - SDP problem formulation

**Questions to answer:**
1. What is the exact objective function? (Minimize 0? Maximize something? Minimize gap?)
2. How is the normalization constraint handled? (α·F_id = 1? Single component? Something else?)
3. What constraints are included beyond positivity? (OPE bounds? Unitarity bounds?)
4. How are F-vectors computed and normalized?

### 2. scalar_blocks (Mathematica/C++)
**Repository:** https://github.com/davidsd/scalar_blocks

**Key files to examine:**
- Look for the SDP setup code
- Check how polynomial positivity is enforced
- Check the normalization convention

### 3. SDPB (Semidefinite Program Bootstrap)
**Repository:** https://github.com/davidsd/sdpb

**Key files to examine:**
- `docs/` - Documentation on input format
- `test/` - Test cases with known correct answers
- Look for example input files that match our problem

**Questions to answer:**
1. What is the standard PMP (Polynomial Matrix Program) format for gap bounds?
2. Are there example files for Δε' bounds specifically?
3. What normalization convention does SDPB expect?

---

## Our Implementation Files

### Core Solver: `cft_bootstrap/bootstrap_gap_solver.py`

**Location:** Lines 48-93 (`reshuffle_with_normalization`) and Lines 148-198 (`_is_excluded_sdp`)

**Current approach:**
```python
# We transform F-vectors to eliminate normalization:
# const_O = F_O[max_idx] / F_id[max_idx]
# F_O_transformed = F_O - const_O * F_id

# Then solve:
# Find alpha_reduced such that:
#   alpha_reduced @ F_eps_reduced >= -fixed_eps
#   alpha_reduced @ F_O_reduced >= -fixed_O  for all operators O
```

**Compare with references:**
- Is this the standard transformation?
- Do references use the same constraint structure?
- Do references include additional constraints?

### Conformal Blocks: `cft_bootstrap/bootstrap_solver.py`

**Location:** Lines 1-150 (`ConformalBlock3D` class and `CrossingVector` class)

**What it computes:**
- 3D scalar conformal blocks using Dolan-Osborn formula
- F-vectors: `F_O = v^{Δσ} g_O(z,zbar) - u^{Δσ} g_O(1-z, 1-zbar)`
- Derivatives at crossing-symmetric point z = zbar = 1/2

**Compare with references:**
- Is our Dolan-Osborn formula correct?
- Is our F-vector definition correct?
- Are we taking derivatives correctly?

### Taylor Series Blocks: `cft_bootstrap/taylor_conformal_blocks.py`

**Location:** Lines 275-296 (`build_F_vector_taylor`) and Lines 348-513 (`HighOrderGapBootstrapSolver`)

**What it computes:**
- High-order derivatives using Taylor series expansion
- Avoids numerical instability of finite differences

**Compare with references:**
- Do references use Taylor series or another method (Zamolodchikov recursion)?
- Is our Taylor coefficient extraction correct?

### Polynomial Positivity: `cft_bootstrap/polynomial_positivity.py`

**Location:** Lines 348-508 (`SOSPositivityConstraint` class)

**What it computes:**
- Sum-of-squares positivity constraints
- Polynomial approximation of F-vectors

**Compare with references:**
- Is our SOS formulation correct?
- Do references use polynomial positivity or discrete sampling?

---

## Specific Comparison Tasks

### Task 1: Compare SDP Problem Formulation

**Our formulation (in `bootstrap_gap_solver.py:148-198`):**
```
Find alpha such that:
  α · F_id = 1  (normalization, transformed via reshuffling)
  α · F_ε ≥ 0  (first scalar positivity)
  α · F_Δ ≥ 0  for all Δ ≥ Δε'  (gap positivity)

If feasible → point EXCLUDED → Δε' is too low
If infeasible → point ALLOWED → Δε' is valid
```

**Questions:**
1. Is this the "dual" or "primal" formulation?
2. Do references solve the same problem or a different one?
3. Is there an objective function we should be optimizing?

**Where to look in references:**
- pycftboot: `cboot/scalar/sdp.py` - Look for the Problem setup
- SDPB docs: Check the mathematical formulation section

### Task 2: Compare F-vector Computation

**Our computation (in `bootstrap_solver.py:85-120`):**
```python
def build_F_vector(self, delta, max_deriv):
    # F = v^{Δσ} g(z,zbar) - u^{Δσ} g(1-z,1-zbar)
    # Take odd derivatives at z = zbar = 1/2
```

**Questions:**
1. Is our F-vector definition correct?
2. Do references include factors we're missing?
3. Are the derivatives normalized correctly?

**Test:** Compute F_id and F_epsilon at Ising point and compare numerically.

### Task 3: Compare Normalization

**Our normalization:**
```python
# pycftboot-style reshuffling:
max_idx = argmax(|F_id|)
const_O = F_O[max_idx] / F_id[max_idx]
F_O_transformed = F_O - const_O * F_id
```

**Questions:**
1. Is this the standard convention?
2. Do references use a different normalization?
3. Should we normalize differently?

### Task 4: Check for Missing Constraints

**What we include:**
- Identity contribution (F_id)
- First scalar at Δε (F_eps)
- All scalars above Δε' (F_ops)

**What we might be missing:**
- OPE coefficient bounds (λ² > 0 constraints)?
- Unitarity bounds (Δ > d/2 - 1)?
- Stress tensor contribution (Δ = 3, spin = 2)?
- Current conservation constraints?

**Where to look:**
- pycftboot: Check what operators are included
- SDPB examples: Check full constraint lists

---

## Validation Tests

### Test 1: F-vector Numerical Comparison

Run this to get our F-vectors:
```bash
cd cft_bootstrap
source venv/bin/activate
python -c "
from bootstrap_solver import CrossingVector
cross = CrossingVector(0.518)  # Ising Δσ
F_id = cross.build_F_vector(0, 5)  # Identity
F_eps = cross.build_F_vector(1.41, 5)  # First scalar
print('F_identity:', F_id)
print('F_epsilon:', F_eps)
"
```

Compare these values with what pycftboot/scalar_blocks produce.

### Test 2: Single Point Exclusion

Run this to test exclusion at a specific point:
```bash
python -c "
from bootstrap_gap_solver import GapBootstrapSolver
solver = GapBootstrapSolver(d=3, max_deriv=5)
excluded = solver.is_excluded(0.518, 1.41, 3.0)  # Test Δε' = 3.0
print('Δε\\'=3.0 excluded:', excluded)
"
```

Our result: `excluded = True` (point is excluded)
Reference should give: `excluded = False` (point is allowed, since bound is ~3.8)

This is the key discrepancy to investigate!

### Test 3: Bound Computation

Run this to compute the bound:
```bash
python -c "
from bootstrap_gap_solver import GapBootstrapSolver
solver = GapBootstrapSolver(d=3, max_deriv=5)
bound = solver.find_delta_epsilon_prime_bound(0.518, 1.41, tolerance=0.05)
print('Bound: Δε\\' <=', bound)
"
```

Our result: ~2.6
Reference result: ~3.8

---

## Expected Outcome

After this comparison, we should know:

1. **Is our SDP formulation correct?**
   - If not, what's the correct formulation?

2. **Are our F-vectors correct?**
   - If not, what's wrong with them?

3. **Are we missing constraints?**
   - If so, which ones?

4. **Is there a fundamental difference in how the problem is posed?**
   - Primal vs dual?
   - Different objective?
   - Different normalization?

---

## Resources

- El-Showk et al. (2012): https://arxiv.org/abs/1203.6064 (original paper)
- pycftboot: https://github.com/cbehan/pycftboot
- scalar_blocks: https://github.com/davidsd/scalar_blocks
- SDPB: https://github.com/davidsd/sdpb
- Simmons-Duffin (2015) SDPB paper: https://arxiv.org/abs/1502.02033 (explains PMP format)

---

## Contact

If you find the discrepancy, please update:
1. `cft_bootstrap/bootstrap_gap_solver.py` with the fix
2. `ROADMAP.md` with the findings
3. This document with what was learned
