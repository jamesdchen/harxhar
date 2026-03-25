# CLAUDE.md

> **When to read this file:** Only consult this document when the user asks to
> **submit SLURM jobs**, **aggregate experiment results**, or **debug HPC job failures/logs**.
> Do not load this context for general code editing, testing, or exploration tasks.

## Cluster Context

- Account is restricted to the **Discovery** cluster (USC CARC, SLURM scheduler)
- CPU partition: `main`
- GPU partition: `gpu`

## Key Paths

| What | Path |
|---|---|
| Input data (6 parquet files) | `all30min/` |
| Results root | `results/` |
| ML SLURM template | `projects/ml/infra/slurm/submit_carc.slurm` |
| GPU SLURM templates | `projects/dl/infra/slurm/{submit_gpu,patchts_backtest,ae_ridge_backtest}.slurm` |
| ML submit script | `projects/ml/scripts/submit.py` |
| DL submit script (module) | `projects.dl.cli.submit` |
| Aggregation script | `projects/ml/scripts/aggregate.py` |
| Comparison script | `projects/ml/scripts/compare.py` |
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
  global_results_summary.csv            # (after aggregation)
```

## Workflows

### Submitting ML Jobs
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

### Submitting DL Jobs
```bash
python -m projects.dl.cli.submit \
    --experiment {patchts|ae_ridge} \
    --result-dir results_patchts \
    --total-chunks 10
```
Optional: `--gpu-count`, `--batch-size`, `--epochs`, `--learning-rate`

### Aggregating Results
```bash
# Auto-discover all dirs with .needs_aggregation marker
python scripts/aggregate.py

# Explicit directory + eval mode
python scripts/aggregate.py --base-dir results/model_comparison --eval-mode global
```

**Eval modes:** `global` (default, all hours), `segments` (pre-segmented files), `filter_by_tod` (filter in memory)

Output: `global_results_summary.csv` or `segment_results_summary.csv` in the base dir.

### Debugging HPC Jobs
```bash
squeue -u $USER                          # check running/pending jobs
sacct -j <JOBID> --format=State,ExitCode # check completed job status
```

- Check `config.txt` in experiment dir for what was submitted
- Count result CSVs vs total-chunks to find failed array tasks
- SLURM stdout/stderr go to default location or `logs/` dir depending on template
- `.needs_aggregation` marker persists until aggregation succeeds
