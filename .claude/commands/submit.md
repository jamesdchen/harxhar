Help me submit SLURM experiments using the workflow below.

## Step 1: Clarify What to Run

Ask me (if not already clear) which **mode** to use:

| Mode | Purpose | Key flags |
|------|---------|-----------|
| `model_comparison` | Compare models on HAR features | `--models ridge xgboost lightgbm random_forest` |
| `feature_transforms` | Compare feature types (HAR/PCA/AE) on one model | `--model ridge --features har pca ae --subgroup all_features` |
| `individual_features` | Ablate features one-by-one within a subgroup | `--model ridge --subgroup moments` |
| `subgroup_analysis` | Cartesian product: models × features × subgroups | `--models all --features all --subgroups all` |
| `naive` | Baseline only | (none) |
| `from-config` | Load from YAML/JSON config | `<config_file>` |

### Available Models
`ridge`, `xgboost`, `lightgbm`, `random_forest`

### Available Feature Types
`har` (rolling means), `pca` (compression), `ae` (autoencoder)

### Available Subgroups
`baseline` (empty), `moments`, `liquidity`, `market_ew`, `market_vw`, `sentiment`, `implied_vol`, `vol_demand`, `all_features` (51 vars)

## Step 2: Pre-Flight Validation

Before submitting, verify:

1. **No duplicate submission** — check for existing results:
   ```bash
   ls -d results/<SET>/exp_*/.submitted 2>/dev/null
   ```
   If `.submitted` markers exist, the script will skip those experiments automatically.

2. **Naive baseline exists** (unless `--no-naive`):
   ```bash
   ls -d results/naive/exp_0_naive_baseline/ 2>/dev/null
   ```
   If missing, the submission will auto-submit a naive job (or run `python scripts/submit.py naive` first).

3. **Cluster availability:**
   ```bash
   squeue -u jc_905 --clusters=discovery  # check current job load
   ```

## Step 3: Dry Run (Recommended)

Preview what will be submitted without actually launching jobs:

```bash
python scripts/submit.py <mode> [flags] --backend dry-run
```

This prints the experiments, env vars, and array ranges without calling `sbatch`. Review the output before proceeding.

## Step 4: Submit

```bash
python scripts/submit.py <mode> [flags]
```

### Common Flag Reference

| Flag | Default | Notes |
|------|---------|-------|
| `--total-chunks` | 100 | Number of array tasks per experiment |
| `--train-window` | 500 | Rolling window size in days |
| `--horizon` | 1 | Multi-horizon: runs h=1..H |
| `--n-components` | 5 | PCA/AE latent dimensions |
| `--ae-alpha` | 0.5 | AE loss: alpha×recon + (1-alpha)×pred |
| `--ae-epochs` | 50 | AE training epochs per refit |
| `--ae-hidden` | 0 (auto) | AE hidden width; 0 = n_features//2 |
| `--ae-weights-path` | None | Pre-trained AE weights .pt file |
| `--no-naive` | false | Skip naive baseline submission |
| `--backend` | slurm | `slurm`, `sge`, or `dry-run` |
| `--result-dir` | mode-dependent | Override output directory |

### Example Commands

```bash
# Compare 4 models on HAR features (100 chunks each = 500 total array tasks)
python scripts/submit.py model_comparison --models ridge xgboost lightgbm random_forest

# PCA vs AE feature transforms on ridge, moments subgroup, 5 components
python scripts/submit.py feature_transforms --model ridge --features pca ae --subgroup moments --n-components 5

# Full grid: all models × all features × all subgroups
python scripts/submit.py subgroup_analysis --models all --features all --subgroups all

# Multi-horizon (h=1..4) model comparison
python scripts/submit.py model_comparison --models ridge xgboost --horizon 4

# From YAML config
python scripts/submit.py from-config experiments/example_subgroup_analysis.yaml
```

### YAML Config Format

Example at `experiments/example_subgroup_analysis.yaml`:
```yaml
name: my_experiment
mode: subgroup_analysis
models: [ridge, xgboost]
features: [har, pca]
subgroups: [moments, liquidity]
total_chunks: 100
horizon: 1
```

## Step 5: Monitor Jobs

After submission, track progress. **All SLURM commands need `--clusters=discovery`** since jobs run on the discovery cluster:

```bash
# Watch job queue
squeue -u jc_905 --clusters=discovery

# Check completed job details
sacct -j <JOBID> --clusters=discovery --format=JobID,State,ExitCode,MaxRSS,Elapsed

# Quick health check: count completed chunks
for d in results/<SET>/exp_*/; do
    n=$(ls "$d"/results_chunk_*.csv 2>/dev/null | wc -l)
    echo "$d: $n/100 chunks"
done
```

## Step 6: Handle Failures

If some array tasks fail:

1. **Identify failures** (must use `--clusters=discovery`):
   ```bash
   sacct -j <JOBID> --clusters=discovery --format=JobID%20,State,ExitCode,MaxRSS,Elapsed | grep -v COMPLETED
   ```

2. **Check logs** (logs are in `/scratch1/jc_905/logs/`, not `/scratch1/jc_905/`):
   ```bash
   grep -l "Error\|OOM\|CANCELLED\|TIMEOUT" /scratch1/jc_905/logs/slurm-<JOBID>_*.err
   ```

3. **Reconstruct env vars from metadata.json** in the experiment's result dir:
   ```bash
   cat results/<SET>/exp_<N>_<name>/metadata.json
   # Extract: model_type, feature_type, variables, extra_args
   cat results/<SET>/exp_<N>_<name>/config.txt
   ```

4. **Resubmit failed task IDs** (1-based) with full context:
   ```bash
   # ML experiments — must include --clusters, --account, and --export
   sbatch --clusters=discovery --array=5,23,78 \
     --account=pollok_1603 \
     --output=/scratch1/jc_905/logs/slurm-%A_%a.out \
     --error=/scratch1/jc_905/logs/slurm-%A_%a.err \
     --export=RESULT_DIR=results/<SET>/exp_<N>_<name>,TOTAL_CHUNKS=100,MODEL_TYPE=<model>,EXOG_COLS="<comma-separated vars or NONE>",EXTRA_ARGS="<extra args from config>" \
     projects/ml/infra/slurm/submit_carc.slurm

   # Add resource overrides as needed (see table below):
   #   --mem=96G          (for OOM fixes)
   #   --time=4:00:00     (for timeout fixes)
   ```

### Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| State=OUT_OF_MEMORY, ExitCode 0:125 | Exceeded `--mem` | Resubmit with `--mem=96G` (or higher) |
| State=TIMEOUT | Exceeded `--time` | Resubmit with `--time=4:00:00` (or higher) |
| `torch.OutOfMemoryError: CUDA out of memory` | GPU OOM | Reduce batch_size in `projects/dl/config.py` |
| ModuleNotFoundError | Python/conda not loaded | Check `module load python` in template |
| `.submitted` marker blocks resubmit | Previous partial run | Delete marker: `rm results/<SET>/exp_*/.submitted` |

### Model-Specific Resource Requirements

Some models need more resources than the SLURM template defaults (64GB, 1hr):

| Model | Memory | Time per chunk (100 chunks) | Notes |
|-------|--------|-----------------------------|-------|
| `ridge` | 64G | ~10 min | Default is fine |
| `xgboost` | **96G** | ~30 min | Some chunks OOM at 64G |
| `lightgbm` | **96G** | ~30 min | Some chunks OOM at 64G |
| `random_forest` | 64G | **~3-4 hrs** | Needs `--time=4:00:00` minimum |
| `ridge` + `vol_demand` subgroup | **96G** | ~15 min | High feature count causes OOM at 64G |

When submitting these models, override resources at sbatch time or update the template.

## Cluster Context

- **Cluster:** Discovery (USC CARC, SLURM)
- **Account:** `pollok_1603`
- **Username:** `jc_905`
- **CPU partition:** `main` (1 CPU, 64G, 1hr default)
- **GPU partition:** `gpu` (8 CPUs, 128G, 2 GPUs, 6hr default)
- **SLURM logs:** `/scratch1/jc_905/logs/`
- **Chunk ID conversion:** SLURM 1-based → executor 0-based (handled by template)

## Key Paths

| What | Path |
|------|------|
| Submit orchestrator | `projects/ml/scripts/submit.py` |
| Submit utilities | `projects/ml/cli/submit.py` |
| Executor | `projects/ml/cli/executor.py` |
| SLURM template | `projects/ml/infra/slurm/submit_carc.slurm` |
| SGE template | `projects/ml/infra/sge/submit_hoffman2.sh` |
| Feature definitions | `projects/ml/features/feature_groups.py` |
| Example configs | `projects/ml/experiments/` |
| HPC backends | `core/backends/` |

## After Submission

Once all jobs complete, aggregate results:
```bash
python scripts/aggregate.py
```
Or use the `/aggregate` command for the full validation + aggregation workflow.
