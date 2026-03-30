# Harxhar Project

ML (Ridge/XGBoost/LightGBM/RandomForest) and DL (PatchTST/AE+Ridge) backtesting pipelines for financial volatility forecasting.

## Architecture

`projects/ml` and `projects/dl` are independent of each other. Both depend on `core` only.

```
core/           Shared foundation (data, features, models, backtest, evaluation)
projects/ml/    Traditional ML models and backtest executors
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

# DL GPU backtest
python -m projects.dl.cli.gpu_executor --help

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

All HPC infrastructure is provided by the `claude-hpc` package via the experiment manifest system.

- **Manifest:** `hpc.yaml` defines profiles (ml, dl), parameter grids, resources, and chunking
- **Submission:** `claude-hpc` reads `hpc.yaml`, expands the grid, and dispatches via `/submit`
- **Monitoring:** `/monitor` tracks per-grid-point completion
- **No project-specific submission code** — claude-hpc handles everything

### hpc.yaml Profiles

| Profile | Grid | Chunks | Resources |
|---------|------|--------|-----------|
| `ml` | model × features (4×3=12) | 100 per point | 1 CPU, 16G, 4h |
| `dl` | experiment (2) | 10 per point | 4 CPU, 2 GPU, 16G, 6h |
