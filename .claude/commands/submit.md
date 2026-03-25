Help me submit SLURM jobs using the context below.

## Cluster Context

- Account is restricted to the **Discovery** cluster (USC CARC, SLURM scheduler)
- Username: `jc_905`
- CPU partition: `main`
- GPU partition: `gpu`
- SLURM logs (stdout/stderr): grep in `/scratch1/jc_905/`

## Key Paths

| What | Path |
|---|---|
| Input data (6 parquet files) | `all30min/` |
| ML SLURM template | `projects/ml/infra/slurm/submit_carc.slurm` |
| GPU SLURM templates | `projects/dl/infra/slurm/{submit_gpu,patchts_backtest,ae_ridge_backtest}.slurm` |
| ML submit script | `projects/ml/scripts/submit.py` |
| DL submit script (module) | `projects.dl.cli.submit` |
| ML executor (module) | `projects.ml.cli.executor` |
| DL executor (module) | `projects.dl.cli.gpu_executor` |

## SLURM Job Defaults

### ML (CPU)
- `--partition=main`, `--cpus-per-task=1`, `--mem=64G`, `--time=1:00:00`
- Default: 100 array chunks
- Module setup: `module purge && module load python`

### DL (GPU)
- `--partition=gpu`, `--cpus-per-task=8`, `--mem=128G`, `--gres=gpu:2`, `--time=6:00:00`
- GPU constraint: `a100|a40|v100|l40s`
- Conda setup: `conda activate project-cucuringu`
- Env: `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"`

### Chunk ID Conversion
SLURM array IDs are **1-based**. The executor expects **0-based**. The SLURM templates handle this conversion (`chunk_id = SLURM_ARRAY_TASK_ID - 1`).

## Submitting ML Jobs
```bash
python scripts/submit.py <mode> [flags]
```

**Modes:**
- `model_comparison` — compare models on HAR features (`--models ridge xgboost ...`)
- `feature_transforms` — compare feature types on one model (`--features har pca ae`)
- `individual_features` — test each feature in a subgroup (`--subgroup moments`)
- `subgroup_analysis` — cartesian product (`--models all --features all --subgroups all`)
- `naive` — baseline only
- `from-config` — load from YAML/JSON config file

**Common flags:** `--total-chunks 100`, `--train-window 500`, `--horizon H`, `--n-components 5`

## Submitting DL Jobs
```bash
python -m projects.dl.cli.submit \
    --experiment {patchts|ae_ridge} \
    --result-dir results_patchts \
    --total-chunks 10
```
Optional: `--gpu-count`, `--batch-size`, `--epochs`, `--learning-rate`

## Output Structure

```
results/<experiment_set>/
  exp_{ID}_{MODEL}_{FEATURES}_{NAME}/
    config.txt                          # experiment metadata
    metadata.json                       # git hash, timestamp, full config
    .submitted                          # marker: prevents re-submission
    results_chunk_1.csv ... _100.csv    # per-chunk backtest results
    results_chunk_1_h{H}.csv           # (multi-horizon only)
    results_chunk_1_coefs.npz          # (if --save-coefs)
  .needs_aggregation                    # marker: triggers aggregation
```

## Debugging

### Job Status
```bash
squeue -u jc_905                                    # running/pending jobs
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,Elapsed  # completed job details
```

### Finding Logs
SLURM stdout/stderr land in `/scratch1/jc_905/`. Grep there for job output:
```bash
find /scratch1/jc_905/ -name "slurm-<JOBID>*" -type f
grep -r "Error\|OOM\|CANCELLED\|TIMEOUT" /scratch1/jc_905/slurm-<JOBID>*
```

### Detecting Failed Chunks
Count result CSVs vs expected total-chunks to find gaps:
```bash
ls results/<experiment_set>/exp_*/results_chunk_*.csv | wc -l
```
Missing chunk IDs indicate failed array tasks.

### Common Failure Modes

**OOM kills** — Job exceeded memory limit.
- Symptom: `sacct` shows State=OUT_OF_MEMORY or ExitCode=0:125
- Fix: Increase `--mem` in the SLURM template (e.g., 64G → 96G for ML, 128G → 192G for GPU)

**Timeouts** — Job exceeded walltime.
- Symptom: `sacct` shows State=TIMEOUT
- Fix: Increase `--time` in the SLURM template, or reduce `--total-chunks` (fewer chunks = more work per task = longer runtime)

**Module/env issues** — Python or conda environment not found.
- Symptom: `ModuleNotFoundError` or `conda: command not found` in logs
- Fix: Verify module loads in the SLURM template match Discovery's available modules (`module avail python`). For GPU jobs, ensure conda source path is correct: `/apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh`

### Resubmitting Failed Tasks
Resubmit only the failed array indices:
```bash
# If tasks 5, 23, 78 failed (1-based SLURM IDs):
sbatch --array=5,23,78 projects/ml/infra/slurm/submit_carc.slurm
```
Ensure the same `--export` variables are passed as the original submission.
