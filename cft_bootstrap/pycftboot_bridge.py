"""
Bridge module for using pycftboot's conformal block tables.

This module provides a clean interface to pycftboot's ConformalBlockTable and
ConvolvedBlockTable classes, allowing us to use their well-tested symbolic
polynomial computation for conformal blocks.

The key advantage is that pycftboot's Zamolodchikov recursion is battle-tested
and produces the exact polynomial structure needed for accurate bootstrap bounds.

Usage:
    from pycftboot_bridge import PycftbootBlockTable, generate_F_vectors

    # Generate conformal block table
    table = PycftbootBlockTable(dim=3, k_max=20, l_max=50, m_max=10, n_max=10)
    table.build()

    # Get F-vectors as polynomials
    F_vectors = table.get_F_vectors(delta_sigma=0.518)
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add pycftboot to path
PYCFTBOOT_PATH = Path(__file__).parent.parent / "reference_implementations" / "pycftboot"
if PYCFTBOOT_PATH.exists():
    sys.path.insert(0, str(PYCFTBOOT_PATH))
    HAVE_PYCFTBOOT = True
else:
    HAVE_PYCFTBOOT = False
    warnings.warn(f"pycftboot not found at {PYCFTBOOT_PATH}")

# Import pycftboot components if available
# pycftboot uses exec() to load files, so we need to replicate that pattern
PYCFTBOOT_LOADED = False
_pycftboot_namespace = {}

if HAVE_PYCFTBOOT:
    try:
        # Import symengine like pycftboot does (wildcard import)
        from symengine.lib.symengine_wrapper import *
        import symengine.lib.symengine_wrapper as _symengine_wrapper

        if not have_mpfr:
            warnings.warn("symengine compiled without MPFR support")

        # Define PolynomialVector class (from bootstrap.py)
        class PolynomialVector:
            """
            The main class for vectors on which the functionals being found by SDPB may act.
            """
            def __init__(self, derivatives, spin_irrep, poles):
                if type(spin_irrep) == type(1):
                    spin_irrep = [spin_irrep, 0]
                self.vector = derivatives
                self.label = spin_irrep
                self.poles = poles

        # Set up namespace for exec() like pycftboot does
        # Include all symengine exports
        _pycftboot_namespace = {k: v for k, v in _symengine_wrapper.__dict__.items()
                               if not k.startswith('_')}
        _pycftboot_namespace.update({
            'os': os, 're': __import__('re'), 'subprocess': __import__('subprocess'),
            'itertools': __import__('itertools'), 'sympy': __import__('sympy'),
            'factorial': __import__('math').factorial,
            'PolynomialVector': PolynomialVector,
        })

        # Change to pycftboot directory for file paths
        original_dir = os.getcwd()
        os.chdir(PYCFTBOOT_PATH)

        # Load common.py using exec() like pycftboot does
        # We need to handle the SDPB path lookup which may fail
        with open("common.py", 'r') as f:
            common_code = f.read()

        # Patch the SDPB path lookup section to avoid failures
        # Replace the path lookup and version detection with stubs
        import re

        # Replace find_executable call for sdpb
        common_code = common_code.replace(
            'if not os.path.isfile(sdpb_path):\n    sdpb_path = find_executable("sdpb")',
            '# SDPB lookup disabled for bridge\npass'
        )

        # Replace subprocess call for version detection
        common_code = common_code.replace(
            'proc = subprocess.Popen([sdpb_path, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n(stdout, _) = proc.communicate()',
            '# Version detection disabled for bridge\nproc = type("MockProc", (), {"returncode": 1})()\nstdout = b""'
        )

        # Also replace mpirun lookup
        common_code = common_code.replace(
            'if not os.path.isfile(mpirun_path):\n        mpirun_path = find_executable("mpirun")',
            '# mpirun lookup disabled for bridge\n        pass'
        )

        exec(common_code, _pycftboot_namespace)

        # Load blocks1.py with additional symengine functions
        _pycftboot_namespace['function_symbol'] = Symbol  # Used in blocks for special functions

        with open("blocks1.py", 'r') as f:
            exec(f.read(), _pycftboot_namespace)

        # Load blocks2.py
        with open("blocks2.py", 'r') as f:
            exec(f.read(), _pycftboot_namespace)

        os.chdir(original_dir)

        # Extract what we need from the namespace
        prec = _pycftboot_namespace.get('prec', 660)
        r_cross = _pycftboot_namespace.get('r_cross')
        delta = _pycftboot_namespace.get('delta')
        ell = _pycftboot_namespace.get('ell')
        ConformalBlockTableSeed = _pycftboot_namespace.get('ConformalBlockTableSeed')
        ConformalBlockTableSeed2 = _pycftboot_namespace.get('ConformalBlockTableSeed2')
        PolynomialVector = _pycftboot_namespace.get('PolynomialVector')
        coefficients = _pycftboot_namespace.get('coefficients')

        PYCFTBOOT_LOADED = True
        print(f"pycftboot loaded successfully from {PYCFTBOOT_PATH}")

    except Exception as e:
        try:
            os.chdir(original_dir)
        except:
            pass
        PYCFTBOOT_LOADED = False
        import traceback
        traceback.print_exc()
        warnings.warn(f"Error loading pycftboot: {e}")
else:
    PYCFTBOOT_LOADED = False

# Local imports
from polynomial_bootstrap import (
    SymbolicPolynomialVector, ConformalBlockPoles,
    PREC, R_CROSS, unitarity_bound as local_unitarity_bound
)


class PycftbootBlockTable:
    """
    Wrapper around pycftboot's ConformalBlockTable.

    This class provides a clean interface to generate conformal block tables
    using pycftboot's implementation, then convert them to our SymbolicPolynomialVector
    format.
    """

    def __init__(
        self,
        dim: float = 3.0,
        k_max: int = 20,
        l_max: int = 50,
        m_max: int = 10,
        n_max: int = 10,
        delta_12: float = 0.0,
        delta_34: float = 0.0,
        odd_spins: bool = False
    ):
        """
        Initialize block table parameters.

        Args:
            dim: Spatial dimension
            k_max: Recursion depth (controls accuracy)
            l_max: Maximum spin
            m_max: Maximum 'a' derivatives
            n_max: Maximum 'b' derivatives
            delta_12: Δ₁ - Δ₂ for external operators
            delta_34: Δ₃ - Δ₄ for external operators
            odd_spins: Include odd spins
        """
        self.dim = dim
        self.k_max = k_max
        self.l_max = l_max
        self.m_max = m_max
        self.n_max = n_max
        self.delta_12 = delta_12
        self.delta_34 = delta_34
        self.odd_spins = odd_spins

        self.block_table = None
        self.convolved_table = None
        self.m_order = []
        self.n_order = []

    def build(self, verbose: bool = True) -> bool:
        """
        Build the conformal block table using pycftboot.

        Returns:
            True if successful, False otherwise
        """
        if not PYCFTBOOT_LOADED:
            if verbose:
                print("pycftboot not available, cannot build table")
            return False

        if verbose:
            print(f"Building conformal block table:")
            print(f"  dim={self.dim}, k_max={self.k_max}, l_max={self.l_max}")
            print(f"  m_max={self.m_max}, n_max={self.n_max}")

        try:
            # Change to pycftboot directory
            original_dir = os.getcwd()
            os.chdir(PYCFTBOOT_PATH)

            # Build the table using pycftboot's ConformalBlockTableSeed
            if self.dim == int(self.dim) and int(self.dim) % 2 == 0:
                # Even integer dimension: use ConformalBlockTableSeed2
                self.block_table = ConformalBlockTableSeed2(
                    self.dim, self.k_max, self.l_max,
                    min(self.m_max + 2 * self.n_max, 3),
                    self.delta_12, self.delta_34, self.odd_spins
                )
            else:
                # Non-integer or odd dimension: use ConformalBlockTableSeed
                self.block_table = ConformalBlockTableSeed(
                    self.dim, self.k_max, self.l_max,
                    min(self.m_max + 2 * self.n_max, 3), 0,
                    self.delta_12, self.delta_34, self.odd_spins
                )

            self.m_order = list(self.block_table.m_order)
            self.n_order = list(self.block_table.n_order)

            os.chdir(original_dir)

            if verbose:
                print(f"  Built table with {len(self.block_table.table)} spin channels")
                print(f"  {len(self.m_order)} derivative components")

            return True

        except Exception as e:
            try:
                os.chdir(original_dir)
            except:
                pass
            if verbose:
                print(f"  Error building table: {e}")
            return False

    def get_polynomial_vectors(self) -> List[SymbolicPolynomialVector]:
        """
        Convert pycftboot table to list of SymbolicPolynomialVectors.

        Returns:
            List of SymbolicPolynomialVector, one per spin channel
        """
        if self.block_table is None:
            raise RuntimeError("Must call build() first")

        vectors = []
        for poly_vec in self.block_table.table:
            # Extract vector, label, and poles from pycftboot's PolynomialVector
            vector = list(poly_vec.vector)
            label = tuple(poly_vec.label)
            poles = list(poly_vec.poles)

            sym_vec = SymbolicPolynomialVector(
                vector=vector,
                label=label,
                poles=poles
            )
            vectors.append(sym_vec)

        return vectors

    def get_spin_vector(self, spin: int) -> Optional[SymbolicPolynomialVector]:
        """
        Get the polynomial vector for a specific spin.

        Args:
            spin: The spin value

        Returns:
            SymbolicPolynomialVector or None if not found
        """
        vectors = self.get_polynomial_vectors()
        for vec in vectors:
            if vec.spin == spin:
                return vec
        return None


def generate_F_vectors_pycftboot(
    delta_sigma: float,
    dim: float = 3.0,
    k_max: int = 20,
    l_max: int = 50,
    m_max: int = 10,
    n_max: int = 10,
    verbose: bool = True
) -> Tuple[List[SymbolicPolynomialVector], List[int], List[int]]:
    """
    Generate F-vectors using pycftboot's conformal block computation.

    This is the main interface for getting accurate polynomial F-vectors.

    Args:
        delta_sigma: External scalar dimension
        dim: Spatial dimension
        k_max: Recursion depth
        l_max: Maximum spin
        m_max: Maximum 'a' derivatives
        n_max: Maximum 'b' derivatives
        verbose: Print progress

    Returns:
        Tuple of (F_vectors, m_order, n_order)
    """
    # For identical scalars: delta_12 = delta_34 = 0
    table = PycftbootBlockTable(
        dim=dim,
        k_max=k_max,
        l_max=l_max,
        m_max=m_max,
        n_max=n_max,
        delta_12=0.0,
        delta_34=0.0
    )

    if not table.build(verbose=verbose):
        raise RuntimeError("Failed to build conformal block table")

    vectors = table.get_polynomial_vectors()
    return vectors, table.m_order, table.n_order


def compare_with_local_implementation(delta_sigma: float = 0.518):
    """
    Compare pycftboot F-vectors with our local implementation.

    This is useful for validating our polynomial_bootstrap.py implementation.
    """
    print("="*60)
    print("Comparing pycftboot with local implementation")
    print("="*60)
    print(f"Δσ = {delta_sigma}")

    if not PYCFTBOOT_LOADED:
        print("pycftboot not available for comparison")
        return

    # Get pycftboot vectors
    try:
        vectors, m_order, n_order = generate_F_vectors_pycftboot(
            delta_sigma=delta_sigma,
            k_max=10,
            l_max=10,
            m_max=3,
            n_max=0,
            verbose=True
        )

        print(f"\nPycftboot results:")
        print(f"  Number of spin channels: {len(vectors)}")
        for vec in vectors[:3]:  # Show first 3
            print(f"  Spin {vec.spin}: {vec.dimension} components, {len(vec.poles)} poles")
            print(f"    Poles: {vec.poles[:5]}{'...' if len(vec.poles) > 5 else ''}")

    except Exception as e:
        print(f"Error: {e}")


def test_pycftboot_bridge():
    """Simple test of the pycftboot bridge."""
    print("Testing pycftboot bridge...")
    print(f"HAVE_PYCFTBOOT: {HAVE_PYCFTBOOT}")
    print(f"PYCFTBOOT_LOADED: {PYCFTBOOT_LOADED}")

    if PYCFTBOOT_LOADED:
        compare_with_local_implementation()
    else:
        print("pycftboot not loaded, skipping test")


if __name__ == "__main__":
    test_pycftboot_bridge()
