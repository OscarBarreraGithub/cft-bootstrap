#!/bin/bash
#SBATCH -p test
#SBATCH --account=iaifi_lab
#SBATCH -t 15
#SBATCH --mem=4G
#SBATCH -o quick_test_%j.out
#SBATCH -e quick_test_%j.err

source /n/sw/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
conda activate cft_bootstrap

echo "Quick baseline test at Ising point"
echo "==================================="

python -c "
from bootstrap_gap_solver import GapBootstrapSolver
import time

print('Testing GapBootstrapSolver (discrete SDP)...')
solver = GapBootstrapSolver(d=3, max_deriv=7)

# Test exclusion at specific points
delta_sigma = 0.518
delta_epsilon = 1.41

for test_val in [2.0, 2.5, 3.0, 3.5, 4.0]:
    start = time.time()
    excluded = solver.is_excluded(delta_sigma, delta_epsilon, test_val)
    status = 'EXCLUDED' if excluded else 'ALLOWED'
    elapsed = time.time() - start
    print(f'  Δε\\'={test_val}: {status} ({elapsed:.2f}s)')

# Find bound
print()
print('Finding bound via binary search...')
start = time.time()
bound = solver.find_delta_epsilon_prime_bound(delta_sigma, delta_epsilon, tolerance=0.1)
elapsed = time.time() - start
print(f'  Bound: Δε\\' <= {bound:.2f} (took {elapsed:.2f}s)')
print(f'  Reference: ~3.8')
print(f'  Gap: {3.8 - bound:.2f}')
"
