# Harxhar Project

ML (Ridge/XGBoost/LightGBM/RandomForest) and DL (PatchTST/AE+Ridge) backtesting pipelines for financial volatility forecasting.

## Architecture

`projects/ml` and `projects/dl` are independent of each other. Both depend on `core` only.

```
core/           Shared foundation (data, features, models, backtest, evaluation, backends)
projects/ml/    Traditional ML models and HPC submission
projects/dl/    Deep learning models and GPU backtest engines
```

## Discoverability

- **Module APIs**: every `__init__.py` has `__all__` and a descriptive docstring. Read the package `__init__.py` first to understand what a module provides.
- **CLI usage**: all scripts support `--help` with defaults shown.
- **Full architecture reference**: `README.md`

## Front-Facing Commands

```bash
# ML single-chunk backtest
python -m projects.ml.cli.executor --help

# ML batch submission (modes: model_comparison, feature_transforms, individual_features, subgroup_analysis, naive, from-config)
python projects/ml/scripts/submit.py <mode> --help

# DL GPU backtest
python -m projects.dl.cli.gpu_executor --help

# DL lifecycle (submit + status)
python -m projects.dl.cli.lifecycle --help

# ML aggregation
python projects/ml/scripts/aggregate.py --help

# DL aggregation
python projects/dl/scripts/aggregate.py --help

# Compare results
python projects/ml/scripts/compare.py --help
```

## Development

```bash
ruff check .                    # lint
ruff format .                   # format
mypy core/ projects/ml/ projects/dl/ --ignore-missing-imports
pytest core/tests/ projects/ml/tests/ -m "not slow and not gpu" --tb=short
```

## HPC Configuration

- **Cluster:** Hoffman2 (UCLA IDRE), SGE scheduler
- **Username:** `jamesdc1`
- **SSH target:** `jamesdc1@hoffman2.idre.ucla.edu`
- **Remote repo:** `$HPC_REPO` (default `/u/home/j/jamesdc1/project-cucuringu/harxhar`)
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
