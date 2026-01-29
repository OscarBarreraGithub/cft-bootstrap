#!/bin/bash
#SBATCH -p shared
#SBATCH --account=iaifi_lab
#SBATCH -t 06:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH -o test_fixed_%j.out
#SBATCH -e test_fixed_%j.err

echo "Testing SDPB with fixed R_CROSS prefactor"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

source /n/sw/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
conda activate cft_bootstrap

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# SDPB config
WORKDIR="${SCRATCH}/schwartz_lab/Everyone/${USER}"
export SDPB_SINGULARITY_IMAGE="$WORKDIR/singularity/sdpb_3.1.0.sif"
export SDPB_USE_SRUN="true"

echo ""
echo "1. Verifying R_CROSS fix..."
python -c "
import numpy as np
R_CROSS = 3 - 2 * np.sqrt(2)
print(f'R_CROSS = {R_CROSS:.10f}')
print(f'exp(-1) = {np.exp(-1):.10f}')
print(f'Ratio (old/new) = {np.exp(-1) / R_CROSS:.2f}')
"

echo ""
echo "2. Running SDPB test at Ising point..."
python run_bootstrap.py \
    --gap-bound \
    --method el-showk-sdpb \
    --nmax 5 \
    --max-spin 10 \
    --sdpb-threads 4 \
    --sigma-min 0.518 \
    --sigma-max 0.518 \
    --n-points 1 \
    --tolerance 0.05 \
    --output-dir $WORKDIR/test_fixed

echo ""
echo "Test completed at $(date)"
