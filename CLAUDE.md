# Claude Guidelines for cft_bootstrap

## FASRC Cluster Rules

### Testing small scripts (quick validation, debugging)
Always use:
```
--account iaifi_lab
--partition test  (or -p test)
--time 00:10:00 to 00:20:00
--ntasks 1
--nodes 1
```

Do NOT use `--partition shared` for quick tests — `test` has lower wait times.
Do NOT request more than 20 minutes for a test job.
Do NOT use multiple nodes for small/test problems.

Example salloc for interactive testing:
```bash
salloc --account iaifi_lab -p test --ntasks=1 --time=00:15:00 -- <command>
```

### Production runs (full bootstrap scans)
Use `--partition shared` with longer time limits (hours).
Multi-node (`--ntasks=4` or more) is appropriate for large problems (66+ constraints).
See `cft_bootstrap/submit_cluster.sh` for production configuration.

## SDPB on FASRC

- SDPB runs inside a Singularity container: `singularity/sdpb_3.1.0.sif`
- Work directories MUST be on shared storage (`/scratch` or `/n/netscratch/...`), NOT `/tmp` (node-local)
- Singularity needs `--bind` for the work directory to be visible across nodes
- For small test problems (< 10 constraints), use `--sdpb-threads 1` — MPI overhead dominates
- symengine warnings are harmless; the SDPB path uses numpy/scipy, not symengine

## Environment

```bash
conda activate cft_bootstrap
```
Python at: `/n/home09/obarrera/.conda/envs/cft_bootstrap/bin/python`

Key packages: numpy, scipy, mpmath, cvxpy (symengine install is incomplete — ignore warnings)

---

## Known Issues (February 2026)

### Issue 1: Large parameters cause PMP build to hang

**Symptom:** Job appears to hang at `[Upper bound] Checking Δε' = 8.00...`

**Cause:** NOT MPI issues. The `ElShowkPolynomialApproximator` recomputes ~98 F-vectors with high-precision mpmath for EVERY SDPB check. Each computation takes seconds.

**Workaround:** Use small parameters:
```bash
--nmax 3 --max-spin 4 --poly-degree 8   # Completes in seconds
# NOT:
--nmax 5 --max-spin 10 --poly-degree 15  # Takes hours
```

**Proper fix needed:** Cache polynomial approximations outside binary search loop.

### Issue 2: SDPB Q-matrix error

**Symptom:**
```
Assertion 'diff < eps' failed:
  Normalized Q should have ones on diagonal. For i = 0: Q_ii = 0
```

**Cause:** PMP formulation issue. Numerical Chebyshev interpolation produces degenerate constraint matrices.

**Status:** Unresolved. Need to use `SymbolicPolynomialApproximator` or add proper bilinear basis.

### Issue 3: SDPB returns wrong results

**Symptom:** Points that should be "EXCLUDED" return "ALLOWED" (e.g., Δε'=6.0 at Ising point).

**Cause:** Same as Issue 2 - PMP formulation is incorrect.

**Status:** Unresolved.

---

## Working Test Commands

```bash
# Verify SDPB container works (completes in ~2s)
sbatch -p test --account=iaifi_lab test_sdpb_quick.sh

# Verify CVXPY baseline (completes in ~10s)
sbatch -p test --account=iaifi_lab quick_test.sh
```

## Non-Working Test Commands

```bash
# These will hang or fail - DO NOT USE until issues fixed:
sbatch test_fixed_prefactor.sh  # Hangs (PMP build too slow)
sbatch test_small_params.sh     # Q-matrix error
```

---

## Documentation Files

| File | Purpose |
|------|---------|
| `cft_bootstrap/FIX_NOTES.md` | Current status and known issues (most up-to-date) |
| `ROADMAP.md` | Implementation history and debugging sessions |
| `cft_bootstrap/README.md` | User guide and quick start |

**Always check `FIX_NOTES.md` first for current status.**
