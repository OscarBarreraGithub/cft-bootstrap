#!/bin/bash
#============================================================================
# SDPB Setup for Harvard FASRC (Cannon Cluster)
#============================================================================
#
# This script sets up SDPB on the Harvard FASRC cluster using Singularity.
# Run this ONCE before submitting bootstrap jobs.
#
# USAGE:
#   1. SSH into the FASRC cluster
#   2. Request a compute node: salloc -p test -c 4 -t 01:00:00 --mem=8G
#   3. Run this script: bash setup_fasrc.sh
#
# The script will:
#   - Create ~/singularity directory for container images
#   - Pull the SDPB Docker image and convert to Singularity format
#   - Verify the installation works
#
# After setup, you can submit bootstrap jobs with:
#   sbatch submit_cluster.sh
#
# REFERENCES:
#   - FASRC Singularity docs: https://docs.rc.fas.harvard.edu/kb/singularity-on-the-cluster/
#   - SDPB GitHub: https://github.com/davidsd/sdpb
#
#============================================================================

set -e  # Exit on error

# Configuration
# WORKDIR: Your personal scratch space on FASRC
# $SCRATCH points to /n/netscratch (base), but your space is under lab/Everyone/user
WORKDIR="${SCRATCH}/schwartz_lab/Everyone/${USER}"
SINGULARITY_DIR="${WORKDIR}/singularity"
# Pin to specific version for reproducibility (not :master)
SDPB_IMAGE="sdpb_3.1.0.sif"
SDPB_DOCKER_IMAGE="bootstrapcollaboration/sdpb:3.1.0"

echo "=============================================="
echo "SDPB Setup for Harvard FASRC"
echo "=============================================="
echo ""

# Check if we're on a compute node (recommended for large operations)
if [[ -z "${SLURM_JOB_ID}" ]]; then
    echo "WARNING: Not running on a compute node."
    echo "For best results, run from a compute node:"
    echo "  salloc -p test -c 4 -t 01:00:00 --mem=8G"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Run 'salloc' first, then try again."
        exit 1
    fi
fi

# Check if Singularity is available
echo "1. Checking Singularity availability..."
if ! command -v singularity &> /dev/null; then
    echo "ERROR: Singularity not found."
    echo "Try loading the module: module load singularity"
    exit 1
fi
SINGULARITY_VERSION=$(singularity --version)
echo "   Found: ${SINGULARITY_VERSION}"

# Create directory for Singularity images
echo ""
echo "2. Creating Singularity image directory..."
mkdir -p "${SINGULARITY_DIR}"
echo "   Directory: ${SINGULARITY_DIR}"

# Check if image already exists
if [[ -f "${SINGULARITY_DIR}/${SDPB_IMAGE}" ]]; then
    echo ""
    echo "3. SDPB image already exists at ${SINGULARITY_DIR}/${SDPB_IMAGE}"
    read -p "   Overwrite existing image? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Keeping existing image."
    else
        rm -f "${SINGULARITY_DIR}/${SDPB_IMAGE}"
        NEED_PULL=true
    fi
else
    NEED_PULL=true
fi

# Pull SDPB Docker image and convert to Singularity
if [[ "${NEED_PULL}" == "true" ]]; then
    echo ""
    echo "3. Pulling SDPB Docker image (this may take several minutes)..."
    echo "   Source: docker://${SDPB_DOCKER_IMAGE}"
    echo "   Destination: ${SINGULARITY_DIR}/${SDPB_IMAGE}"
    echo ""

    cd "${SINGULARITY_DIR}"
    singularity pull "${SDPB_IMAGE}" "docker://${SDPB_DOCKER_IMAGE}"

    echo ""
    echo "   Image downloaded successfully!"
fi

# Verify the installation
echo ""
echo "4. Verifying SDPB installation..."

echo "   Testing sdpb..."
if singularity exec "${SINGULARITY_DIR}/${SDPB_IMAGE}" sdpb --help > /dev/null 2>&1; then
    echo "   [OK] sdpb works"
else
    echo "   [FAIL] sdpb failed"
    exit 1
fi

echo "   Testing pmp2sdp..."
if singularity exec "${SINGULARITY_DIR}/${SDPB_IMAGE}" pmp2sdp --help > /dev/null 2>&1; then
    echo "   [OK] pmp2sdp works"
else
    echo "   [FAIL] pmp2sdp failed"
    exit 1
fi

# Get SDPB version info
echo ""
echo "5. SDPB Version Information:"
singularity exec "${SINGULARITY_DIR}/${SDPB_IMAGE}" sdpb --version 2>&1 | head -5 || true

# Print summary
echo ""
echo "=============================================="
echo "SETUP COMPLETE"
echo "=============================================="
echo ""
echo "SDPB Singularity image installed at:"
echo "  ${SINGULARITY_DIR}/${SDPB_IMAGE}"
echo ""
echo "Image size: $(du -h "${SINGULARITY_DIR}/${SDPB_IMAGE}" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Verify SINGULARITY_IMAGE in submit_cluster.sh points to:"
echo "     \$SCRATCH/singularity/sdpb_3.1.0.sif"
echo "  2. Set up Python environment:"
echo "     mamba create -n cft_bootstrap -c conda-forge python=3.10 numpy scipy matplotlib mpmath cvxpy symengine -y"
echo "  3. Submit test job: sbatch test_sdpb.sh"
echo "  4. Submit production: sbatch submit_cluster.sh"
echo ""
echo "To test SDPB manually:"
echo "  singularity exec ${SINGULARITY_DIR}/${SDPB_IMAGE} sdpb --help"
echo ""
