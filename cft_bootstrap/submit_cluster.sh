#!/bin/bash
#============================================================================
# CFT Bootstrap SLURM Submission Script
#============================================================================
#
# This script submits array jobs to compute conformal bootstrap bounds on a
# HPC cluster using SLURM. It supports multiple methods including the El-Showk
# et al. (2012) full derivative basis with spinning operators.
#
# USAGE:
#   1. Edit configuration section below
#   2. Submit: sbatch submit_cluster.sh
#   3. Monitor: squeue -u $USER
#   4. Collect results: python run_bootstrap.py --collect --output-dir results/
#
# REFERENCE:
#   El-Showk et al., "Solving the 3D Ising Model with the Conformal Bootstrap"
#   arXiv:1203.6064 (2012)
#
#============================================================================

#SBATCH --job-name=cft_bootstrap
#SBATCH --output=logs/bootstrap_%A_%a.out
#SBATCH --error=logs/bootstrap_%A_%a.err
#SBATCH --array=0-19

# ============================================================================
# RESOURCE CONFIGURATION
# ============================================================================
# Adjust based on method:
#   - lp/sdp/cvxpy:  --mem=4G, --time=02:00:00, --cpus-per-task=1
#   - el-showk:      --mem=16G, --time=08:00:00, --cpus-per-task=4
#   - el-showk-sdpb: --mem=32G, --time=24:00:00, --cpus-per-task=8

#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# ============================================================================
# CONFIGURATION - MODIFY THESE FOR YOUR RUN
# ============================================================================

# ---------- Grid Range ----------
# Range of delta_sigma values to scan
SIGMA_MIN=0.500
SIGMA_MAX=0.550

# Number of points (should match --array above: 0 to N_POINTS-1)
N_POINTS=20

# ---------- Computation Mode ----------
# GAP_BOUND: Enable two-stage Figure 6 computation
#   - "true"  = Compute Δε' bounds along the Δε boundary (FIGURE 6)
#   - "false" = Compute simple Δε bounds only
#
# When GAP_BOUND=true, each job performs:
#   Stage 1: Find Δε boundary at this Δσ
#   Stage 2: Compute Δε' bound with Δε fixed to boundary
#
GAP_BOUND=true

# ---------- Method Selection ----------
# Available methods:
#   "lp"              - Linear programming (fast, basic)
#   "sdp"             - Semidefinite programming via CVXPY
#   "cvxpy"           - Discrete sampling with CVXPY
#   "polynomial"      - Polynomial positivity with SOS constraints
#   "hybrid"          - Polynomial + discrete samples
#   "two-correlator"  - Two-correlator bootstrap (ssss + eeee)
#   "mixed-correlator"- Full mixed correlator with matrix SDP
#   "el-showk"        - El-Showk (2012) full derivative basis [RECOMMENDED]
#   "el-showk-sdpb"   - El-Showk with SDPB high-precision solver

METHOD="el-showk"

# ---------- General Solver Parameters ----------
MAX_DERIV=20           # Derivative order (el-showk uses nmax = MAX_DERIV // 2)
TOLERANCE=0.01         # Binary search tolerance
POLY_DEGREE=15         # Polynomial degree (for polynomial/hybrid methods)

# ---------- El-Showk Method Parameters ----------
# These only apply when METHOD="el-showk" or "el-showk-sdpb"
#
# NMAX: Number of mixed derivatives (overrides MAX_DERIV // 2)
#   - nmax=10 gives 66 coefficients (paper recommendation)
#   - nmax=5 gives 21 coefficients (fast test)
NMAX=10

# MAX_SPIN: Maximum spin for spinning operators
#   - Paper uses Lmax=100, but Lmax=50 is often sufficient
#   - Set to 0 to disable spinning operators (scalars only)
MAX_SPIN=50

# USE_MULTIRESOLUTION: Enable T1-T5 style multi-resolution discretization
#   - "true" = fine grid near unitarity, coarse grid at high Delta
#   - "false" = uniform sampling (faster but less accurate)
USE_MULTIRESOLUTION=true

# EL_SHOWK_SOLVER: Backend solver for El-Showk method
#   - "auto" = try CLARABEL > ECOS > MOSEK > SCS (recommended)
#   - "scs"  = SCS solver (always available)
#   - "ecos" = ECOS solver (if installed)
#   - "clarabel" = CLARABEL solver (if installed, high precision)
#   - "mosek" = MOSEK solver (if licensed)
EL_SHOWK_SOLVER="auto"

# HIGH_PRECISION: Use full mpmath arbitrary-precision arithmetic
#   - "true"  = REQUIRED for accurate reproduction of El-Showk (2012) at nmax=10
#   - "false" = Use standard float64 (faster but less accurate)
# This mode uses mpmath for ALL derivative computation, avoiding float64
# precision loss that causes numerical instability at high orders.
# Much slower but produces publication-quality results.
HIGH_PRECISION=true

# PRECISION: Number of decimal places for high-precision mode
#   - 100 = sufficient for nmax ≤ 7
#   - 150 = recommended for nmax = 10 (66 coefficients)
#   - 200 = for extreme precision requirements
PRECISION=150

# ---------- SDPB Parameters ----------
# Only for METHOD="sdpb" or "el-showk-sdpb"
SDPB_THREADS=4         # Threads for SDPB
SDPB_PRECISION=400     # Bits of precision

# ---------- Container Configuration (for HPC clusters) ----------
# USE_SINGULARITY: Enable Singularity container execution
#   - "true"  = Use Singularity container (recommended for FASRC)
#   - "false" = Use direct binary (assumes SDPB in PATH)
USE_SINGULARITY=true

# SINGULARITY_IMAGE: Path to the SDPB Singularity image
#   - Run setup_fasrc.sh to download and set up the image
#   - Default location: ~/singularity/sdpb_master.sif
SINGULARITY_IMAGE="${HOME}/singularity/sdpb_master.sif"

# MPI_TYPE: MPI type for srun (FASRC uses pmix)
#   - "pmix" = Standard for FASRC (Cannon cluster)
#   - "pmi2" = Alternative for some systems
MPI_TYPE="pmix"

# ---------- Output ----------
OUTPUT_DIR="results_elshowk_${SIGMA_MIN}_${SIGMA_MAX}_nmax${NMAX}_spin${MAX_SPIN}"

# ============================================================================
# SETUP (usually no changes needed)
# ============================================================================

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Load modules (uncomment and adjust for your cluster)
# module load python/3.9
# module load scipy

# Activate virtual environment (uncomment and adjust path)
# source /path/to/venv/bin/activate

# ============================================================================
# BUILD COMMAND
# ============================================================================

echo "=============================================="
echo "CFT Bootstrap Job ${SLURM_ARRAY_TASK_ID} of ${N_POINTS}"
echo "=============================================="
echo "Method: ${METHOD}"
echo "Gap bound mode: ${GAP_BOUND}"
echo "Delta_sigma range: [${SIGMA_MIN}, ${SIGMA_MAX}]"
echo "Output: ${OUTPUT_DIR}"

# Build base command
CMD="python run_bootstrap.py \
    --array-job \
    --job-index ${SLURM_ARRAY_TASK_ID} \
    --n-jobs ${N_POINTS} \
    --sigma-min ${SIGMA_MIN} \
    --sigma-max ${SIGMA_MAX} \
    --method ${METHOD} \
    --max-deriv ${MAX_DERIV} \
    --tolerance ${TOLERANCE} \
    --poly-degree ${POLY_DEGREE} \
    --output-dir ${OUTPUT_DIR}"

# Add gap bound flag if enabled (Figure 6 mode)
if [[ "${GAP_BOUND}" == "true" ]]; then
    echo "  Computing Δε' bounds (Figure 6 mode)"
    CMD="${CMD} --gap-bound"
fi

# Add El-Showk specific flags if applicable
if [[ "${METHOD}" == "el-showk"* ]]; then
    echo "El-Showk parameters:"
    echo "  nmax = ${NMAX}"
    echo "  max_spin = ${MAX_SPIN}"
    echo "  multiresolution = ${USE_MULTIRESOLUTION}"
    echo "  solver = ${EL_SHOWK_SOLVER}"
    echo "  high_precision = ${HIGH_PRECISION}"
    echo "  precision = ${PRECISION}"

    CMD="${CMD} \
        --nmax ${NMAX} \
        --max-spin ${MAX_SPIN} \
        --el-showk-solver ${EL_SHOWK_SOLVER}"

    if [[ "${USE_MULTIRESOLUTION}" == "true" ]]; then
        CMD="${CMD} --use-multiresolution"
    fi

    if [[ "${HIGH_PRECISION}" == "true" ]]; then
        CMD="${CMD} --high-precision --precision ${PRECISION}"
    fi
fi

# Add SDPB flags if applicable
if [[ "${METHOD}" == *"sdpb"* ]]; then
    echo "SDPB parameters:"
    echo "  threads = ${SDPB_THREADS}"
    echo "  precision = ${SDPB_PRECISION} bits"
    echo "  use_singularity = ${USE_SINGULARITY}"

    CMD="${CMD} \
        --sdpb-threads ${SDPB_THREADS} \
        --sdpb-precision ${SDPB_PRECISION}"

    # Add Singularity configuration if enabled
    if [[ "${USE_SINGULARITY}" == "true" ]]; then
        if [[ ! -f "${SINGULARITY_IMAGE}" ]]; then
            echo "ERROR: Singularity image not found at ${SINGULARITY_IMAGE}"
            echo "Run setup_fasrc.sh first to download the SDPB image"
            exit 1
        fi
        echo "  singularity_image = ${SINGULARITY_IMAGE}"
        echo "  mpi_type = ${MPI_TYPE}"

        # Set environment variables for the Python script to detect
        export SDPB_SINGULARITY_IMAGE="${SINGULARITY_IMAGE}"
        export SDPB_USE_SRUN="true"
        export SDPB_MPI_TYPE="${MPI_TYPE}"
    fi
fi

echo "=============================================="
echo "Running command:"
echo "${CMD}"
echo "=============================================="

# Execute
${CMD}

EXIT_CODE=$?
echo "=============================================="
echo "Job ${SLURM_ARRAY_TASK_ID} completed with exit code ${EXIT_CODE}"
echo "=============================================="
exit ${EXIT_CODE}
