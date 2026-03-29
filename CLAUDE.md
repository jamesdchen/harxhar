# Harxhar Project

ML (Ridge/XGBoost/LightGBM/RandomForest) and DL (PatchTST/AE+Ridge) backtesting pipelines for financial volatility forecasting.

## HPC Configuration

- **Cluster:** Hoffman2 (UCLA IDRE), SGE scheduler
- **Username:** `jamesdc1`
- **SSH target:** `jamesdc1@hoffman2.idre.ucla.edu`
- **Remote repo:** `$HPC_REPO` (default `/u/project/project-cucuringu/harxhar`)
- **ML SGE logs:** `$HPC_REPO/logs/`
- **DL SGE logs:** `/u/scratch/j/jamesdc1/` (`$SCRATCH`)
- **Results:** `results/`

### SGE Commands
| Action | Command |
|--------|---------|
| Check queue | `qstat -u jamesdc1` |
| Job accounting | `qacct -j <JOBID>` |
| Clear error state | `qmod -cj <JOBID>` |
| Submit | `qsub -t <range> -N <name> -o logs -j y -v <vars> <template>` |

### Rsync
Exclude list: `.git/ results/ __pycache__/ *.pyc .mypy_cache/ all30min/ .claude/`

Pull summaries:
```bash
rsync -az \
    --include='*/' --include='*_summary*.csv' --include='metadata.json' \
    --include='config.txt' --include='lifecycle.jsonl' --exclude='*' \
    jamesdc1@hoffman2.idre.ucla.edu:$HPC_REPO/results/ ./results/
```

## Front-Facing Commands

All scripts support `--help` for full flag/default documentation.

```bash
# ML submit (modes: model_comparison, feature_transforms, individual_features, subgroup_analysis, naive, from-config)
python projects/ml/scripts/submit.py <mode> [flags] --backend <slurm|sge|sge-remote|dry-run>

# DL submit (experiments: patchts, ae_ridge, scaling)
python -m projects.dl.cli.submit --experiment <name> --total-chunks <n> --backend <slurm|sge|dry-run>

# DL lifecycle (submit + status tracking)
python -m projects.dl.cli.lifecycle <submit|status> [flags]

# ML aggregation (auto-discovers dirs with .needs_aggregation marker)
python projects/ml/scripts/aggregate.py [--base-dir results/<SET>] [--eval-mode <global|segments|filter_by_tod>]

# DL aggregation
python projects/dl/scripts/aggregate.py [dirs...] [--auto] [--eval-mode <global|segments|filter_by_tod>]

# Compare results
python projects/ml/scripts/compare.py results/<SET> --metric qlike --sort asc --top 10
```

## Key Paths

| What | Path |
|------|------|
| ML submit orchestrator | `projects/ml/scripts/submit.py` |
| ML submit utilities | `projects/ml/cli/submit.py` |
| ML executor | `projects/ml/cli/executor.py` |
| ML SGE template | `projects/ml/infra/sge/submit_hoffman2.sh` |
| DL SGE template | `projects/dl/infra/sge/submit_gpu.sh` |
| DL SGE scaling template | `projects/dl/infra/sge/submit_scaling.sh` |
| DL lifecycle manager | `projects/dl/cli/lifecycle.py` |
| DL GPU executor | `projects/dl/cli/gpu_executor.py` |
| Feature definitions | `core/features/feature_groups.py` |
| ML aggregation script | `projects/ml/scripts/aggregate.py` |
| DL aggregation script | `projects/dl/scripts/aggregate.py` |
| Aggregation logic | `core/evaluation/aggregation.py` |
| Metrics | `core/evaluation/metrics.py` |
| Comparison script | `projects/ml/scripts/compare.py` |
| Example configs | `projects/ml/experiments/` |
| SSH/rsync utilities | `core/remote.py` |
| HPC backends | `core/backends/` |
