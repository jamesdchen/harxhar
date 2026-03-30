# Harxhar Project

ML (Ridge/XGBoost/LightGBM/RandomForest) and DL (PatchTST/AE+Ridge) backtesting pipelines for financial volatility forecasting.

## Architecture

`projects/ml` and `projects/dl` are independent of each other. Both depend on `core` only.

```
core/           Shared foundation (data, features, models, backtest, evaluation)
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

All HPC infrastructure is provided by the `claude-hpc` package. No project-specific job templates.

- **Config files:** `project.yaml` (stages, cluster envs), `clusters.yaml` (in claude-hpc)
- **Templates:** Generic `cpu_array` / `gpu_array` from claude-hpc (`hpc.get_template_path()`)
- **Backends:** `hpc.backends.get_backend()` → SLURM, SGE, SGE-remote, Dry-run
- **Remote:** `hpc.remote.ssh_run()` / `rsync_push()` with host/user from config
- **Results:** `results/`

### Key APIs
```python
from hpc import get_template_path, load_clusters_config, load_project_config
from hpc.backends import get_backend
get_template_path("sge", "cpu_array")          # → path to claude-hpc template
get_backend("slurm", script=str(template))     # → HPCBackend instance
```

### SGE Commands
| Action | Command |
|--------|---------|
| Check queue | `qstat -u jamesdc1` |
| Job accounting | `qacct -j <JOBID>` |
| Clear error state | `qmod -cj <JOBID>` |
| Submit | `qsub -t <range> -N <name> -o logs -j y -v <vars> <template>` |
