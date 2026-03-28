Help me submit SGE experiments on Hoffman2 via SSH using the workflow below.

All cluster commands run remotely via `ssh jamesdc1@hoffman2.idre.ucla.edu`.
Code is synced from the local machine before submission.

## Step 1: Clarify What to Run

Ask me (if not already clear) which **mode** to use:

| Mode | Purpose | Key flags |
|------|---------|-----------|
| `model_comparison` | Compare models on HAR features | `--models ridge xgboost lightgbm random_forest` |
| `feature_transforms` | Compare feature types (HAR/PCA/AE) on one model | `--model ridge --features har pca ae --subgroup all_features` |
| `individual_features` | Ablate features one-by-one within a subgroup | `--model ridge --subgroup moments` |
| `subgroup_analysis` | Cartesian product: models x features x subgroups | `--models all --features all --subgroups all` |
| `naive` | Baseline only | (none) |
| `from-config` | Load from YAML/JSON config | `<config_file>` |

### Available Models
`ridge`, `xgboost`, `lightgbm`, `random_forest`

### Available Feature Types
`har` (rolling means), `pca` (compression), `ae` (autoencoder)

### Available Subgroups
`baseline` (empty), `moments`, `liquidity`, `market_ew`, `market_vw`, `sentiment`, `implied_vol`, `vol_demand`, `all_features` (51 vars)

## Step 2: Sync Code to Cluster

Before anything else, push local code to Hoffman2:

```bash
cd /path/to/harxhar && python -c "from core.remote import rsync_push; r = rsync_push(); print(r.stdout or 'synced'); print(r.stderr)"
```

Or directly:
```bash
rsync -az --delete \
    --exclude='.git/' --exclude='results/' --exclude='__pycache__/' \
    --exclude='*.pyc' --exclude='.mypy_cache/' --exclude='all30min/' \
    --exclude='.claude/' \
    . jamesdc1@hoffman2.idre.ucla.edu:$HPC_REPO/
```

Verify the sync succeeded (exit code 0) before proceeding.

## Step 3: Pre-Flight Validation

Run these checks via SSH:

1. **Cluster job load:**
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "qstat -u jamesdc1"
   ```

2. **No duplicate submission** — check for existing results:
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "ls -d $HPC_REPO/results/<SET>/exp_*/.submitted 2>/dev/null"
   ```
   If `.submitted` markers exist, the script will skip those experiments automatically.

3. **Naive baseline exists** (unless `--no-naive`):
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "ls -d $HPC_REPO/results/naive/exp_0_naive_baseline/ 2>/dev/null"
   ```

## Step 4: Dry Run (Recommended)

Preview what will be submitted without actually launching jobs. Dry run runs locally:

```bash
python projects/ml/scripts/submit.py <mode> [flags] --backend dry-run
```

This prints the experiments, env vars, and array ranges without calling `qsub`.

## Step 5: Submit

### ML Experiments

Submit via SSH using the `sge-remote` backend (wraps qsub over SSH):

```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python projects/ml/scripts/submit.py <mode> [flags] --backend sge"
```

### DL Experiments

```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python -m projects.dl.cli.lifecycle submit --experiment <experiment> --total-chunks <N> [flags]"
```

| Experiment | Description |
|-----------|-------------|
| `patchts` | PatchTST transformer backtest |
| `ae_ridge` | Autoencoder + Ridge GPU backtest |

### Common Flag Reference

| Flag | Default | Notes |
|------|---------|-------|
| `--total-chunks` | 100 | Number of array tasks per experiment |
| `--train-window` | 500 | Rolling window size in days |
| `--horizon` | 1 | Multi-horizon: runs h=1..H |
| `--n-components` | 5 | PCA/AE latent dimensions |
| `--ae-alpha` | 0.5 | AE loss: alpha x recon + (1-alpha) x pred |
| `--ae-epochs` | 50 | AE training epochs per refit |
| `--ae-hidden` | 0 (auto) | AE hidden width; 0 = n_features//2 |
| `--ae-weights-path` | None | Pre-trained AE weights .pt file |
| `--no-naive` | false | Skip naive baseline submission |
| `--backend` | slurm | `slurm`, `sge`, `sge-remote`, or `dry-run` |
| `--result-dir` | mode-dependent | Override output directory |

### Example Commands

```bash
# Compare 4 models on HAR features
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python projects/ml/scripts/submit.py model_comparison --models ridge xgboost lightgbm random_forest --backend sge"

# PCA vs AE feature transforms
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python projects/ml/scripts/submit.py feature_transforms --model ridge --features pca ae --subgroup moments --n-components 5 --backend sge"

# Full grid
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python projects/ml/scripts/submit.py subgroup_analysis --models all --features all --subgroups all --backend sge"
```

## Step 6: Monitor Jobs

After submission, track progress via SSH:

```bash
# Watch job queue
ssh jamesdc1@hoffman2.idre.ucla.edu "qstat -u jamesdc1"

# Check completed job details
ssh jamesdc1@hoffman2.idre.ucla.edu "qacct -j <JOBID>"

# Quick health check: count completed chunks
ssh jamesdc1@hoffman2.idre.ucla.edu "for d in $HPC_REPO/results/<SET>/exp_*/; do n=\$(ls \"\$d\"/results_chunk_*.csv 2>/dev/null | wc -l); echo \"\$d: \$n/100 chunks\"; done"
```

## Step 7: Handle Failures

If some array tasks fail:

1. **Identify failures:**
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "qacct -j <JOBID>" | grep -E "taskid|failed|exit_status"
   ```

2. **Check logs:**
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "tail -100 $HPC_REPO/logs/<job_log_file>"
   ```

3. **Reconstruct env vars from metadata.json:**
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "cat $HPC_REPO/results/<SET>/exp_<N>_<name>/metadata.json"
   ssh jamesdc1@hoffman2.idre.ucla.edu "cat $HPC_REPO/results/<SET>/exp_<N>_<name>/config.txt"
   ```

4. **Resubmit failed task IDs:**
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && qsub -t <FAILED_IDS> \
       -N <job_name> -o logs -j y \
       -v RESULT_DIR=results/<SET>/exp_<N>_<name>,TOTAL_CHUNKS=100,MODEL_TYPE=<model>,EXOG_COLS='<vars or NONE>',EXTRA_ARGS='<from config>' \
       projects/ml/infra/sge/submit_hoffman2.sh"
   ```

### Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Eqw` state in qstat | Job error (check qacct) | Fix issue, `qmod -cj <JOBID>` or resubmit |
| Memory exceeded | Exceeded h_vmem | Resubmit with `-l h_vmem=96G` |
| Walltime exceeded | Exceeded h_rt | Resubmit with `-l h_rt=14400` (4hrs) |
| `torch.OutOfMemoryError` | GPU OOM | Reduce batch_size |
| ModuleNotFoundError | Python not loaded | Check `module load` in template |
| `.submitted` marker blocks resubmit | Previous partial run | `ssh ... "rm $HPC_REPO/results/<SET>/exp_*/.submitted"` |

---

## Cluster Context

- **Cluster:** Hoffman2 (UCLA IDRE, SGE)
- **Username:** `jamesdc1`
- **SSH target:** `jamesdc1@hoffman2.idre.ucla.edu`
- **Remote repo:** `$HPC_REPO` (env var, default `/u/project/project-cucuringu/harxhar`)
- **SGE logs:** `$HPC_REPO/logs/`
- **Chunk ID conversion:** SGE 1-based -> executor 0-based (handled by template)

## Key Paths (on cluster)

| What | Path |
|------|------|
| ML submit orchestrator | `projects/ml/scripts/submit.py` |
| ML submit utilities | `projects/ml/cli/submit.py` |
| ML executor | `projects/ml/cli/executor.py` |
| SGE template | `projects/ml/infra/sge/submit_hoffman2.sh` |
| Feature definitions | `projects/ml/features/feature_groups.py` |
| Example configs | `projects/ml/experiments/` |
| DL submit + status | `projects/dl/cli/lifecycle.py` |
| DL GPU executor | `projects/dl/cli/gpu_executor.py` |
| HPC backends | `core/backends/` |
| SSH/rsync utilities | `core/remote.py` |

## After Submission

Once all jobs complete, aggregate results:
```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python projects/ml/scripts/aggregate.py"
```
Or use the `/aggregate` command for the full remote aggregation + download workflow.
