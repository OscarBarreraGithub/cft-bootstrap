#!/bin/bash
#============================================================================
# Quick SDPB Sanity Test
#============================================================================
#
# Fast validation that SDPB integration works correctly.
# Tests a definitely-EXCLUDED point which should return quickly.
#
# This test validates:
#   1. Singularity container access
#   2. pmp2sdp conversion works
#   3. SDPB execution works
#   4. Output parsing works
#
# Expected: Completes in <10 minutes with EXCLUDED result
#
# USAGE:
#   sbatch test_sdpb_quick.sh
#
#============================================================================

#SBATCH --job-name=sdpb_quick
#SBATCH --output=test_sdpb_quick_%j.out
#SBATCH --error=test_sdpb_quick_%j.err
#SBATCH --account=iaifi_lab
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=8G

set -e  # Exit on error

echo "=============================================="
echo "Quick SDPB Sanity Test"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""
echo "This test uses minimal parameters to validate SDPB works."
echo "Testing a definitely-EXCLUDED point (delta_epsilon_prime=6.0)"
echo "which should return quickly."
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

WORKDIR="${SCRATCH}/schwartz_lab/Everyone/${USER}"

echo "1. Activating conda environment..."
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    source /n/sw/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
fi
conda activate cft_bootstrap
echo "   Python: $(which python)"

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Ensure unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

# ============================================================================
# SDPB CONFIGURATION
# ============================================================================

echo ""
echo "2. Setting SDPB environment variables..."
export SDPB_SINGULARITY_IMAGE="$WORKDIR/singularity/sdpb_3.1.0.sif"
export SDPB_USE_SRUN="false"  # Single task, no MPI complexity
echo "   SDPB_SINGULARITY_IMAGE=$SDPB_SINGULARITY_IMAGE"
echo "   SDPB_USE_SRUN=$SDPB_USE_SRUN"

# ============================================================================
# VERIFY SDPB IS ACCESSIBLE
# ============================================================================

echo ""
echo "3. Verifying SDPB is accessible..."
if [[ ! -f "$SDPB_SINGULARITY_IMAGE" ]]; then
    echo "   ERROR: Singularity image not found at $SDPB_SINGULARITY_IMAGE"
    echo "   Run setup_fasrc.sh first to download the SDPB image"
    exit 1
fi

echo "   Image exists: $(du -h "$SDPB_SINGULARITY_IMAGE" | cut -f1)"
singularity exec "$SDPB_SINGULARITY_IMAGE" sdpb --version || {
    echo "   ERROR: sdpb command failed inside container"
    exit 1
}
echo "   SDPB container OK!"

# ============================================================================
# RUN QUICK TEST
# ============================================================================

echo ""
echo "4. Running quick SDPB test..."
echo "   Testing a single EXCLUDED point (delta_epsilon_prime=6.0)"
echo "   with minimal parameters (nmax=3, max_spin=2, poly_degree=8)"
echo ""

cd $WORKDIR/cft_bootstrap/cft_bootstrap

python -c "
import time
import sys
sys.path.insert(0, '.')

from sdpb_interface import SDPBSolver, SDPBConfig, PolynomialApproximator

print('Initializing SDPB solver...')
config = SDPBConfig(num_threads=1, verbosity='regular')
solver = SDPBSolver(config)

if not solver.is_available:
    print('ERROR: SDPB not available')
    sys.exit(1)

print(f'SDPB available via {solver._execution_mode.name}')

# Test with minimal polynomial approximator
print('')
print('Building minimal polynomial approximator...')
print('  max_deriv=5, poly_degree=8')
approx = PolynomialApproximator(delta_sigma=0.518, max_deriv=5, poly_degree=8)

# Test a single EXCLUDED point (should be fast)
print('')
print('Testing single EXCLUDED point (delta_epsilon_prime=6.0)...')
print('This point is far from the physical region and should be quickly excluded.')
print('')

t0 = time.time()
excluded, info = solver.is_excluded_sdpb(
    delta_sigma=0.518,
    delta_epsilon=1.41,
    delta_epsilon_prime=6.0,
    max_deriv=5,
    poly_degree=8,
    approx=approx,
    verbose_timing=True
)
t1 = time.time()

print('')
print(f'Result: {\"EXCLUDED\" if excluded else \"ALLOWED\"}')
print(f'Status: {info.get(\"status\", \"unknown\")}')
print(f'Total time: {t1-t0:.1f}s')

if 'timings' in info:
    print(f'Timing breakdown:')
    for stage, duration in info['timings'].items():
        print(f'  {stage}: {duration:.1f}s')

print('')
if excluded:
    print('SUCCESS: SDPB correctly identified the point as EXCLUDED')
    sys.exit(0)
else:
    print('WARNING: Expected EXCLUDED, got ALLOWED')
    print('This may indicate an issue with the problem formulation.')
    sys.exit(1)
"

EXIT_CODE=$?

echo ""
echo "=============================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "TEST PASSED"
else
    echo "TEST FAILED (exit code: $EXIT_CODE)"
fi
echo "=============================================="
echo "Completed at: $(date)"

exit $EXIT_CODE
