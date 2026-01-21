#!/usr/bin/env python3
"""
Test suite for CFT Bootstrap implementation.

Run with: python -m pytest tests/test_bootstrap.py -v
Or simply: python tests/test_bootstrap.py
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_sdpb_interface():
    """Test SDPB interface module."""
    print("\n" + "=" * 60)
    print("TEST: SDPB Interface Module")
    print("=" * 60)

    from sdpb_interface import (
        SDPBSolver, SDPBConfig, FallbackSDPBSolver,
        PolynomialApproximator, get_best_solver
    )

    # Test polynomial approximation
    print("\n1. Testing PolynomialApproximator...")
    approx = PolynomialApproximator(delta_sigma=0.518, max_deriv=11, poly_degree=15)
    F_poly = approx.approximate_F_as_polynomial(1.5, 30.0)
    assert F_poly.dimension == 6, f"Wrong dimension: {F_poly.dimension}"
    assert F_poly.max_degree == 15, f"Wrong max degree: {F_poly.max_degree}"
    print(f"   ✓ Dimension: {F_poly.dimension}, Max degree: {F_poly.max_degree}")

    # Test PMP generation
    print("\n2. Testing PMP generation...")
    pmp = approx.build_polynomial_matrix_program(1.41, 2.0, 30.0)
    assert "objective" in pmp, "Missing objective"
    assert "normalization" in pmp, "Missing normalization"
    assert "PositiveMatrixWithPrefactorArray" in pmp, "Missing constraints"
    print(f"   ✓ PMP has {len(pmp['PositiveMatrixWithPrefactorArray'])} matrix constraints")

    # Test solver availability
    print("\n3. Testing solver availability...")
    sdpb_solver = SDPBSolver()
    fallback = FallbackSDPBSolver(max_deriv=11)
    print(f"   SDPB available: {sdpb_solver.is_available}")
    print(f"   CVXPY fallback available: {fallback.is_available}")
    assert fallback.is_available, "CVXPY should be available"
    print("   ✓ At least one solver is available")

    # Test get_best_solver
    print("\n4. Testing get_best_solver...")
    solver = get_best_solver(max_deriv=11)
    print(f"   ✓ Best solver: {type(solver).__name__}")

    print("\n" + "=" * 60)
    print("SDPB Interface: ALL TESTS PASSED")
    print("=" * 60)
    return True


def test_taylor_conformal_blocks():
    """Test Taylor series conformal blocks."""
    print("\n" + "=" * 60)
    print("TEST: Taylor Series Conformal Blocks")
    print("=" * 60)

    from taylor_conformal_blocks import (
        TaylorCrossingVector, build_F_vector_taylor,
        g_diagonal_taylor_coeffs, HighOrderGapBootstrapSolver
    )

    # Test diagonal block coefficients
    print("\n1. Testing g_Δ(z,z) Taylor coefficients...")
    coeffs = g_diagonal_taylor_coeffs(1.5, 0.5, 5)
    assert len(coeffs) == 6, f"Wrong number of coefficients: {len(coeffs)}"
    assert coeffs[0] != 0, "First coefficient should be non-zero"
    print(f"   ✓ Coefficients: {coeffs[:4]}")

    # Test F-vector construction
    print("\n2. Testing F-vector construction...")
    cross = TaylorCrossingVector(0.518, max_deriv=11)
    F_id = cross.build_F_vector(0)
    F_15 = cross.build_F_vector(1.5)
    assert len(F_id) == 6, f"Wrong F-vector length: {len(F_id)}"
    assert F_id[0] != 0, "Identity should have non-zero first component"
    print(f"   ✓ F_identity shape: {F_id.shape}")
    print(f"   ✓ F_1.5 shape: {F_15.shape}")

    # Test high-order solver
    print("\n3. Testing HighOrderGapBootstrapSolver...")
    solver = HighOrderGapBootstrapSolver(d=3, max_deriv=11)
    assert solver.n_constraints == 6, f"Wrong constraint count: {solver.n_constraints}"
    print(f"   ✓ Number of constraints: {solver.n_constraints}")

    print("\n" + "=" * 60)
    print("Taylor Series: ALL TESTS PASSED")
    print("=" * 60)
    return True


def test_spinning_conformal_blocks():
    """Test spinning conformal blocks."""
    print("\n" + "=" * 60)
    print("TEST: Spinning Conformal Blocks")
    print("=" * 60)

    from spinning_conformal_blocks import (
        SpinningConformalBlock, SpinningCrossingVector, SpinningBootstrapSolver
    )

    # Test spinning block creation
    print("\n1. Testing SpinningConformalBlock...")
    block_spin2 = SpinningConformalBlock(delta=3.0, ell=2, n_max=15, d=3)
    val = block_spin2.evaluate(r=0.1, eta=0.5)
    assert not np.isnan(val), "Block evaluation returned NaN"
    print(f"   ✓ Stress tensor block at (r=0.1, η=0.5): {val:.6f}")

    # Test diagonal evaluation
    print("\n2. Testing diagonal evaluation...")
    val_diag = block_spin2.evaluate_diagonal(z=0.3)
    assert not np.isnan(val_diag), "Diagonal evaluation returned NaN"
    print(f"   ✓ Diagonal block at z=0.3: {val_diag:.6f}")

    # Test spinning crossing vector
    print("\n3. Testing SpinningCrossingVector...")
    cross = SpinningCrossingVector(delta_sigma=0.518, max_deriv=5)
    F_stress = cross.build_F_vector(delta=3.0, ell=2)
    assert len(F_stress) == cross.n_constraints, "Wrong F-vector length"
    print(f"   ✓ Stress tensor F-vector: {F_stress[:3]}")

    # Test spinning bootstrap solver
    print("\n4. Testing SpinningBootstrapSolver...")
    solver = SpinningBootstrapSolver(d=3, max_deriv=5, max_spin=2)
    assert solver.max_spin == 2, "Wrong max_spin"
    print(f"   ✓ Solver created with max_spin={solver.max_spin}")

    print("\n" + "=" * 60)
    print("Spinning Blocks: ALL TESTS PASSED")
    print("=" * 60)
    return True


def test_gap_bound_computation():
    """Test gap-bound computation (quick version)."""
    print("\n" + "=" * 60)
    print("TEST: Gap Bound Computation")
    print("=" * 60)

    from sdpb_interface import FallbackSDPBSolver

    # Quick test with low tolerance
    print("\n1. Testing gap-bound at Ising point...")
    solver = FallbackSDPBSolver(max_deriv=7)

    # Test is_excluded
    excluded = solver.is_excluded(
        delta_sigma=0.518,
        delta_epsilon=1.41,
        delta_epsilon_prime=4.0
    )
    print(f"   Δε' = 4.0: {'EXCLUDED' if excluded else 'ALLOWED'}")
    assert excluded, "Should be excluded at Δε' = 4.0"

    excluded = solver.is_excluded(
        delta_sigma=0.518,
        delta_epsilon=1.41,
        delta_epsilon_prime=1.5
    )
    print(f"   Δε' = 1.5: {'EXCLUDED' if excluded else 'ALLOWED'}")
    # Note: 1.5 might be allowed or excluded depending on constraints

    print("\n2. Testing bound computation (coarse)...")
    bound = solver.find_bound(
        delta_sigma=0.518,
        delta_epsilon=1.41,
        tolerance=0.5,
        verbose=False
    )
    print(f"   ✓ Bound: Δε' ≤ {bound:.2f}")
    assert 2.0 < bound < 5.0, f"Bound {bound} outside expected range"

    print("\n" + "=" * 60)
    print("Gap Bound: ALL TESTS PASSED")
    print("=" * 60)
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("#  CFT Bootstrap Test Suite")
    print("#" * 60)

    results = {}

    try:
        results["SDPB Interface"] = test_sdpb_interface()
    except Exception as e:
        print(f"FAILED: {e}")
        results["SDPB Interface"] = False

    try:
        results["Taylor Series"] = test_taylor_conformal_blocks()
    except Exception as e:
        print(f"FAILED: {e}")
        results["Taylor Series"] = False

    try:
        results["Spinning Blocks"] = test_spinning_conformal_blocks()
    except Exception as e:
        print(f"FAILED: {e}")
        results["Spinning Blocks"] = False

    try:
        results["Gap Bound"] = test_gap_bound_computation()
    except Exception as e:
        print(f"FAILED: {e}")
        results["Gap Bound"] = False

    # Summary
    print("\n" + "#" * 60)
    print("#  TEST SUMMARY")
    print("#" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("#" * 60)
    if all_passed:
        print("#  ALL TESTS PASSED")
    else:
        print("#  SOME TESTS FAILED")
    print("#" * 60)

    return all_passed


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    success = run_all_tests()
    sys.exit(0 if success else 1)
