#!/bin/bash
#============================================================================
# SDPB Test Script for Harvard FASRC (Cannon Cluster)
#============================================================================
#
# This script runs a quick test to verify SDPB is working correctly.
# Submit BEFORE running full production jobs to catch configuration errors.
#
# USAGE:
#   sbatch test_sdpb.sh
#
# SUCCESS INDICATORS:
#   1. Output shows "Using SDPB solver" (not "Using CVXPY fallback")
#   2. Output files appear in $WORKDIR/test_run/
#   3. No "SDPB not found!" or MPI plugin errors
#
# IF IT FAILS:
#   - Check SDPB container exists: ls $WORKDIR/singularity/sdpb_3.1.0.sif
#   - Check conda env exists: conda env list | grep cft_bootstrap
#   - If MPI errors, try: export SDPB_MPI_TYPE="pmix_v3" or "pmi2"
#
#============================================================================

#SBATCH --job-name=test_sdpb
#SBATCH --output=test_sdpb_%j.out
#SBATCH --error=test_sdpb_%j.err
#SBATCH --account=iaifi_lab
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --mem=16G

set -e  # Exit on error

# ============================================================================
# WORKDIR: Your personal scratch space on FASRC
# $SCRATCH points to /n/netscratch (base), but your space is under lab/Everyone/user
# ============================================================================
WORKDIR="${SCRATCH}/schwartz_lab/Everyone/${USER}"

echo "=============================================="
echo "SDPB Test Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "1. Activating conda environment..."
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    # Fallback to FASRC Miniforge path
    source /n/sw/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
fi
conda activate cft_bootstrap
echo "   Python: $(which python)"
echo "   Version: $(python --version)"

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ensure unbuffered output for real-time logging
export PYTHONUNBUFFERED=1

# ============================================================================
# SDPB CONFIGURATION
# ============================================================================

echo ""
echo "2. Setting SDPB environment variables..."
export SDPB_SINGULARITY_IMAGE="$WORKDIR/singularity/sdpb_3.1.0.sif"
export SDPB_USE_SRUN="true"
# Note: SDPB_MPI_TYPE defaults to "pmix". If job fails with MPI plugin error, try:
#   export SDPB_MPI_TYPE="pmix_v3"  # or "pmi2"
echo "   SDPB_SINGULARITY_IMAGE=$SDPB_SINGULARITY_IMAGE"
echo "   SDPB_USE_SRUN=$SDPB_USE_SRUN"

# ============================================================================
# SDPB VISIBILITY CHECK
# ============================================================================

echo ""
echo "3. Verifying SDPB is accessible..."
if [[ ! -f "$SDPB_SINGULARITY_IMAGE" ]]; then
    echo "   ERROR: Singularity image not found at $SDPB_SINGULARITY_IMAGE"
    echo "   Run setup_fasrc.sh first to download the SDPB image"
    exit 1
fi

echo "   Image size: $(du -h "$SDPB_SINGULARITY_IMAGE" | cut -f1)"
singularity exec "$SDPB_SINGULARITY_IMAGE" sdpb --version || {
    echo "   ERROR: sdpb command failed inside container"
    exit 1
}
echo "   SDPB OK!"

# ============================================================================
# RUN TEST
# ============================================================================

echo ""
echo "4. Running bootstrap test..."
cd $WORKDIR/cft_bootstrap/cft_bootstrap

# Clean previous test output
rm -rf $WORKDIR/test_run

python run_bootstrap.py \
    --gap-bound \
    --method el-showk-sdpb \
    --nmax 5 \
    --max-spin 10 \
    --sdpb-threads 1 \
    --sigma-min 0.518 \
    --sigma-max 0.518 \
    --n-points 1 \
    --output-dir $WORKDIR/test_run

# ============================================================================
# VERIFY OUTPUT
# ============================================================================

echo ""
echo "5. Checking output files..."
if [[ -d "$WORKDIR/test_run" ]]; then
    echo "   Output directory exists:"
    ls -lh $WORKDIR/test_run | head -10
else
    echo "   WARNING: Output directory not found"
fi

echo ""
echo "=============================================="
echo "TEST COMPLETE"
echo "=============================================="
echo ""
echo "Check the output above for:"
echo "  - 'Using SDPB solver' (success)"
echo "  - 'Using CVXPY fallback' (SDPB not invoked - check config)"
echo "  - MPI errors (try different SDPB_MPI_TYPE)"
echo ""
