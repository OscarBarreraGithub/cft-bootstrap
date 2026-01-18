#!/bin/bash
#SBATCH --job-name=cft_bootstrap
#SBATCH --output=logs/bootstrap_%A_%a.out
#SBATCH --error=logs/bootstrap_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# CFT Bootstrap SLURM submission script
# Adjust the parameters below for your cluster

# ============================================
# CONFIGURATION - MODIFY THESE
# ============================================

# Range of delta_sigma to scan
SIGMA_MIN=0.500
SIGMA_MAX=0.650

# Number of points (should match --array above)
N_POINTS=100

# Solver settings
METHOD="lp"           # "lp" or "sdp"
MAX_DERIV=11          # More derivatives = tighter bounds, slower
TOLERANCE=0.005       # Precision of each bound

# Output directory
OUTPUT_DIR="results_${SIGMA_MIN}_${SIGMA_MAX}"

# ============================================
# SETUP
# ============================================

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Load modules (adjust for your cluster)
# module load python/3.9
# module load scipy

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# ============================================
# RUN
# ============================================

echo "Running bootstrap job ${SLURM_ARRAY_TASK_ID} of ${N_POINTS}"
echo "Delta_sigma range: [${SIGMA_MIN}, ${SIGMA_MAX}]"

python run_bootstrap.py \
    --array-job \
    --job-index ${SLURM_ARRAY_TASK_ID} \
    --n-jobs ${N_POINTS} \
    --sigma-min ${SIGMA_MIN} \
    --sigma-max ${SIGMA_MAX} \
    --method ${METHOD} \
    --max-deriv ${MAX_DERIV} \
    --tolerance ${TOLERANCE} \
    --output-dir ${OUTPUT_DIR}

echo "Job ${SLURM_ARRAY_TASK_ID} completed"
