# HARXHAR

Realized volatility forecasting system using HAR-family models with exogenous features. Supports rolling-window backtesting across Ridge, XGBoost, LightGBM, Random Forest, SARIMAX, and deep learning (PatchTSMixer, Autoencoder+Ridge) models on intraday 30-minute bar data.

## Architecture

```
src/
├── config.py              # Central configuration (lags, windows, segments)
├── log.py                 # Logging setup
├── data/
│   ├── transforms.py      # Diurnal adjustment, winsorization, data transforms
│   ├── loading.py         # Parquet loading, gridding, circuit-breaker handling
│   ├── pipeline.py        # Lag feature generation, horizon shifts, segmentation
│   └── rolling.py         # RollingBuffer, RollingRobustScaler, RollingMedian
├── features/
│   └── transforms.py      # HAR/Raw lag features, PCA, Autoencoder transforms
├── models/
│   ├── base.py            # BaseModel, RollingRegressionModel, NaiveBaseline
│   ├── sklearn_models.py  # Ridge, XGBoost, LightGBM, RandomForest wrappers
│   ├── sarimax.py         # SARIMAX with rolling window
│   ├── registry.py        # MODEL_REGISTRY and create_model() factory
│   └── deep_learning.py   # PatchTSMixer, LagAutoEncoder
├── backtest/
│   ├── engine.py          # CPU backtest loop, Duan smearing, result saving
│   ├── gpu_utils.py       # GPU parallelization, batched training utilities
│   ├── gpu_engine.py      # PatchTSMixer GPU backtest
│   ├── gpu_engine_ae.py   # AE+Ridge GPU backtest
│   └── gpu_kernels.py     # Compiled vmap training kernels
├── evaluation/
│   ├── metrics.py         # MSE, MAE, QLIKE, OOS R²
│   └── aggregation.py     # Chunk stitching, config parsing, experiment processing
└── cli/
    ├── executor.py        # CLI arg parsing, backtest orchestration
    └── submit.py          # SLURM job submission utilities
```

## Setup

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

Run a single-chunk backtest:

```bash
python -m src.cli.executor \
    --model ridge \
    --features har \
    --input-path all30min \
    --output-file results/results_chunk_1.csv \
    --chunk-id 1 \
    --total-chunks 100
```

## Running Tests

```bash
pytest
pytest -m "not slow and not gpu"  # skip slow/GPU tests
```

## SLURM Workflow

Submit experiment batches via the scripts in `scripts/`:

```bash
python scripts/submit_model_comparison.py --result-dir results_comparison
python scripts/aggregate.py              # aggregate after jobs complete
```

## Development

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [pre-commit](https://pre-commit.com/) for git hooks:

```bash
pip install ruff pre-commit
pre-commit install

ruff check .          # lint
ruff format .         # format
```

Type checking with mypy:

```bash
pip install mypy
mypy src/
```
