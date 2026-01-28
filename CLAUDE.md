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
