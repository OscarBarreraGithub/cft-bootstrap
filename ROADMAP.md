# Roadmap: Matching El-Showk et al. (2012) Figure 7

This document outlines what is needed to exactly reproduce the Δε' bounds from:

> **"Solving the 3D Ising Model with the Conformal Bootstrap"**
> S. El-Showk, M. Paulos, D. Poland, S. Rychkov, D. Simmons-Duffin, A. Vichi
> [arXiv:1203.6064](https://arxiv.org/abs/1203.6064) (2012), Figure 7

## Current Status

| Metric | Our Implementation | Reference | Gap |
|--------|-------------------|-----------|-----|
| Δε' at Ising (Δσ=0.518) | ~2.6 | ~3.8 | ~1.2 |
| Derivative constraints | 3 | ~60+ | ~20x fewer |
| Operator types | Scalars only | Scalars + spinning | Missing spin-2+ |
| Numerical stability | max_deriv ≤ 7 | max_deriv ~ 60 | Limited by finite diff |

## Required Improvements

### 1. Zamolodchikov Recursion for Conformal Blocks ⬜ NOT STARTED

**Current:** We use the Dolan-Osborn formula with finite difference derivatives.

**Problem:** Finite differences become numerically unstable for m > 7.

**Solution:** Implement the Zamolodchikov recursion relation, which computes conformal block derivatives analytically.

**Impact:** Would allow ~60+ derivative constraints, likely closing most of the gap.

**Complexity:** HIGH - requires implementing recursive computation of conformal blocks and their derivatives.

**References:**
- Zamolodchikov (1984): "Conformal symmetry in two dimensions"
- Kos, Poland, Simmons-Duffin (2014): "Bootstrapping the O(N) vector models"

---

### 2. Spinning Conformal Blocks ⬜ NOT STARTED

**Current:** We only include scalar operators (spin-0) in the OPE.

**Problem:** The stress tensor (spin-2, Δ=3) and other spinning operators contribute to crossing.

**Solution:** Implement spinning conformal blocks using the Dolan-Osborn formula for general spin.

**Impact:** Would sharpen bounds by ~10-20%, especially near the Ising point.

**Complexity:** MEDIUM - formulas exist, need careful implementation.

**References:**
- Dolan, Osborn (2001): "Conformal four point functions and the operator product expansion"
- Costa et al. (2011): "Spinning conformal blocks"

---

### 3. Mixed Correlator Bootstrap ⬜ NOT STARTED

**Current:** We use only the ⟨σσσσ⟩ four-point function.

**Problem:** The sharp kink at the Ising point comes from combining multiple correlators.

**Solution:** Include ⟨σσεε⟩ and ⟨εεεε⟩ correlators in the bootstrap system.

**Impact:** Creates the sharp kink that precisely locates the Ising model.

**Complexity:** HIGH - requires setting up a system of crossing equations.

**References:**
- Kos, Poland, Simmons-Duffin (2014): "Bootstrapping mixed correlators in the 3D Ising model"

---

### 4. SDPB Integration ⬜ NOT STARTED

**Current:** We use CVXPY with the SCS solver.

**Problem:** General-purpose SDP solvers are slower and less accurate for bootstrap problems.

**Solution:** Interface with SDPB (Semidefinite Program Solver for the Bootstrap).

**Impact:** Faster computation, higher precision, ability to handle more constraints.

**Complexity:** MEDIUM - requires installing SDPB and writing an interface.

**References:**
- Simmons-Duffin (2015): "A semidefinite program solver for the conformal bootstrap"
- [SDPB GitHub](https://github.com/davidsd/sdpb)

---

### 5. Polynomial Approximation for Positivity ⬜ NOT STARTED

**Current:** We sample operators at discrete Δ values and check positivity at each.

**Problem:** Discrete sampling can miss narrow excluded regions.

**Solution:** Use polynomial approximation to enforce positivity for all Δ ≥ Δ_gap.

**Impact:** More rigorous bounds, matches the paper's methodology exactly.

**Complexity:** MEDIUM - requires polynomial fitting and matrix positivity constraints.

---

### 6. Self-Consistent Δε Boundary ✅ IMPLEMENTED (with caveats)

**Current:** We use a hand-tuned piecewise linear approximation for Δε(Δσ).

**Problem:** The real boundary comes from the bootstrap itself.

**Solution Attempted:** Compute the Δε boundary self-consistently using our bootstrap.

**Result:** ⚠️ WORSE than hand-tuned! Our computed Δε bounds are too weak (Δε~1.58 vs actual~1.41), which leads to LOWER Δε' bounds.

**Lesson Learned:** With only 3 derivative constraints, our Δε bounds are not accurate enough for this approach. The hand-tuned curve using known Ising CFT values actually gives better Δε' bounds.

**Revised Solution:** Use literature values for Δε(Δσ) or implement Zamolodchikov recursion first.

**Status:** ✅ IMPLEMENTED but not used (hand-tuned is better with current constraints)

---

---

### 7. Literature Δε Boundary Values ✅ IMPLEMENTED

**Current:** Hand-tuned piecewise linear approximation.

**Problem:** Our approximation doesn't match the actual bootstrap boundary precisely.

**Solution:** Use tabulated Δε boundary values from published bootstrap results.

**Result:** Implemented but gives essentially identical results to approximate boundary (~0.01 difference). The ~1 unit gap to reference is entirely due to limited derivative constraints, not boundary approximation.

**Impact:** Minimal - the bottleneck is derivative constraints, not boundary accuracy.

**Status:** ✅ IMPLEMENTED (now default in `compute_ising_plot`)

---

## Implementation Priority

Based on impact vs complexity (REVISED after learning from Priority 1):

| Priority | Improvement | Impact | Complexity | Est. Time |
|----------|-------------|--------|------------|-----------|
| ~~1~~ | ~~Self-consistent Δε boundary~~ | ~~LOW~~ | ~~LOW~~ | ~~1 hour~~ |
| 1 | **Literature Δε boundary values** | LOW | LOW | 30 min |
| 2 | Spinning conformal blocks | MEDIUM | MEDIUM | 1-2 days |
| 3 | Zamolodchikov recursion | HIGH | HIGH | 1 week |
| 4 | SDPB integration | MEDIUM | MEDIUM | 2-3 days |
| 5 | Polynomial approximation | MEDIUM | MEDIUM | 2-3 days |
| 6 | Mixed correlator bootstrap | HIGH | HIGH | 1-2 weeks |

**Note:** Self-consistent Δε boundary was attempted but gives worse results with our current weak constraints. The lesson: better prior knowledge (hand-tuned or literature values) beats self-consistency when constraints are limited.

## Progress Tracking

### Completed ✅
- [x] Basic scalar conformal blocks (Dolan-Osborn)
- [x] Crossing equation setup
- [x] SDP solver integration (CVXPY)
- [x] Gap-based Δε' bounds
- [x] Qualitative reproduction of Figure 7 shape
- [x] Self-consistent Δε boundary (implemented but not used - gives worse results)
- [x] Literature Δε boundary values

### In Progress 🔄
- [ ] None currently

### Not Started ⬜
- [ ] Zamolodchikov recursion (highest impact)
- [ ] Spinning conformal blocks
- [ ] Mixed correlator bootstrap
- [ ] SDPB integration
- [ ] Polynomial approximation

---

## Expected Results After Each Improvement

| After Implementing | Expected Δε' at Ising | Gap to Reference |
|--------------------|----------------------|------------------|
| Current | ~2.6 | ~1.2 |
| + Self-consistent Δε | ~2.7 | ~1.1 |
| + Spinning blocks | ~3.0 | ~0.8 |
| + Zamolodchikov (20 derivs) | ~3.4 | ~0.4 |
| + Zamolodchikov (60 derivs) | ~3.7 | ~0.1 |
| + All improvements | ~3.8 | ~0 |

---

## How to Contribute

1. Pick an improvement from the list
2. Create a branch: `git checkout -b feature/improvement-name`
3. Implement and test
4. Update this roadmap
5. Submit a PR

For questions about the physics, see the references in each section.
