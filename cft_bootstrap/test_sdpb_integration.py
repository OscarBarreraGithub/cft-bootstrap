#!/usr/bin/env python
"""
Integration tests for SDPB Docker/Singularity support.

Run with: python test_sdpb_integration.py
"""

import os
import subprocess
import sys
import tempfile
import shutil
import unittest


class TestSDPBAvailability(unittest.TestCase):
    """Test SDPB availability detection."""

    def test_check_sdpb_availability_function(self):
        """Test the check_sdpb_availability helper function."""
        from sdpb_interface import check_sdpb_availability

        info = check_sdpb_availability()

        self.assertIn("available", info)
        self.assertIn("mode", info)
        self.assertIn("details", info)

        # Mode should be one of the valid values
        self.assertIn(info["mode"], ["BINARY", "DOCKER", "SINGULARITY", "NONE"])

    def test_sdpb_solver_initialization(self):
        """Test SDPBSolver initializes correctly."""
        from sdpb_interface import SDPBSolver, SDPBConfig

        config = SDPBConfig()
        config.verbosity = "silent"

        solver = SDPBSolver(config)

        # If execution mode is set, solver should be available
        # If execution mode is None, solver should not be available
        if solver._execution_mode is not None:
            self.assertTrue(solver.is_available)
        else:
            self.assertFalse(solver.is_available)


class TestDockerExecution(unittest.TestCase):
    """Test Docker-based SDPB execution."""

    @classmethod
    def setUpClass(cls):
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}"],
                capture_output=True, text=True, timeout=10
            )
            cls.docker_available = result.returncode == 0
            cls.sdpb_image_available = "sdpb" in result.stdout.lower() or "bootstrapcollaboration" in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            cls.docker_available = False
            cls.sdpb_image_available = False

    def test_docker_sdpb_help(self):
        """Test that SDPB runs in Docker."""
        if not self.docker_available or not self.sdpb_image_available:
            self.skipTest("Docker or SDPB image not available")

        result = subprocess.run(
            ["docker", "run", "--rm", "bootstrapcollaboration/sdpb:master", "sdpb", "--help"],
            capture_output=True, text=True, timeout=30
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("SDPB", result.stdout)

    def test_docker_pmp2sdp_help(self):
        """Test that pmp2sdp runs in Docker."""
        if not self.docker_available or not self.sdpb_image_available:
            self.skipTest("Docker or SDPB image not available")

        result = subprocess.run(
            ["docker", "run", "--rm", "bootstrapcollaboration/sdpb:master", "pmp2sdp", "--help"],
            capture_output=True, text=True, timeout=30
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("input", result.stdout)


class TestSDPBConfig(unittest.TestCase):
    """Test SDPB configuration."""

    def test_default_config(self):
        """Test default SDPBConfig values."""
        from sdpb_interface import SDPBConfig

        config = SDPBConfig()

        self.assertEqual(config.precision, 400)
        self.assertEqual(config.num_threads, 4)
        self.assertEqual(config.max_iterations, 500)

    def test_docker_config(self):
        """Test DockerConfig values."""
        from sdpb_interface import DockerConfig

        config = DockerConfig()

        self.assertEqual(config.image, "bootstrapcollaboration/sdpb:master")

    def test_singularity_config_env_vars(self):
        """Test SingularityConfig reads environment variables."""
        from sdpb_interface import SingularityConfig

        # Set environment variables
        os.environ["SDPB_SINGULARITY_IMAGE"] = "/custom/path/sdpb.sif"
        os.environ["SDPB_USE_SRUN"] = "false"
        os.environ["SDPB_MPI_TYPE"] = "pmi2"

        try:
            config = SingularityConfig()

            self.assertEqual(config.image_path, "/custom/path/sdpb.sif")
            self.assertFalse(config.use_srun)
            self.assertEqual(config.mpi_type, "pmi2")
        finally:
            # Clean up
            del os.environ["SDPB_SINGULARITY_IMAGE"]
            del os.environ["SDPB_USE_SRUN"]
            del os.environ["SDPB_MPI_TYPE"]


class TestPMPGeneration(unittest.TestCase):
    """Test PMP generation."""

    def test_polynomial_approximator(self):
        """Test PolynomialApproximator generates valid structure."""
        from sdpb_interface import PolynomialApproximator

        approx = PolynomialApproximator(
            delta_sigma=0.518,
            max_deriv=5,
            poly_degree=5
        )

        pmp = approx.build_polynomial_matrix_program(
            delta_epsilon=1.41,
            delta_epsilon_prime=2.0,
            delta_max=10.0
        )

        # Check PMP structure
        self.assertIn("objective", pmp)
        self.assertIn("normalization", pmp)
        self.assertIn("PositiveMatrixWithPrefactorArray", pmp)

        # Check array has entries
        self.assertGreater(len(pmp["PositiveMatrixWithPrefactorArray"]), 0)

        # Check damped rational structure
        for entry in pmp["PositiveMatrixWithPrefactorArray"]:
            self.assertIn("DampedRational", entry)
            self.assertIn("polynomials", entry)

            dr = entry["DampedRational"]
            self.assertIn("constant", dr)
            self.assertIn("base", dr)
            self.assertIn("poles", dr)

            # Base must be in (0, 1) for SDPB
            base = float(dr["base"])
            self.assertGreater(base, 0)
            self.assertLess(base, 1)


class TestPycftbootBridge(unittest.TestCase):
    """Test pycftboot bridge integration."""

    def test_pycftboot_loaded(self):
        """Test pycftboot is loadable."""
        from pycftboot_bridge import PYCFTBOOT_LOADED

        # This may or may not be True depending on environment
        self.assertIsInstance(PYCFTBOOT_LOADED, bool)

    def test_block_table_creation(self):
        """Test PycftbootBlockTable can be created."""
        from pycftboot_bridge import PycftbootBlockTable, PYCFTBOOT_LOADED

        if not PYCFTBOOT_LOADED:
            self.skipTest("pycftboot not available")

        table = PycftbootBlockTable(
            dim=3.0,
            k_max=5,
            l_max=5,
            m_max=2,
            n_max=0
        )

        self.assertIsNotNone(table)


class TestCheckEnv(unittest.TestCase):
    """Test environment checker."""

    def test_check_sdpb_function(self):
        """Test check_sdpb returns valid tuple."""
        from check_env import check_sdpb

        available, method, details = check_sdpb()

        self.assertIsInstance(available, bool)
        self.assertIsInstance(method, str)
        self.assertIsInstance(details, str)

    def test_check_pmp2sdp_function(self):
        """Test check_pmp2sdp returns valid tuple."""
        from check_env import check_pmp2sdp

        available, details = check_pmp2sdp()

        self.assertIsInstance(available, bool)
        self.assertIsInstance(details, str)


if __name__ == "__main__":
    print("SDPB Integration Tests")
    print("=" * 60)
    print()

    # Run tests
    unittest.main(verbosity=2)
