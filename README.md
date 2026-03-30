# HARXHAR

Realized volatility forecasting system using HAR-family models with exogenous features. Supports rolling-window backtesting across Ridge, XGBoost, LightGBM, Random Forest, SARIMAX, and deep learning (PatchTST, Autoencoder+Ridge) models on intraday 30-minute bar data.

The system takes raw parquet data, engineers lag-based features at multiple time scales (geometric base-5 HAR lags: 1, 5, 25, 125, 625, 3125 half-hour periods), and runs walk-forward backtests with online model updates. It is designed for large-scale distributed execution on HPC clusters (SLURM, SGE) and GPU-accelerated training on Google Colab.

## Architecture

```
core/                              # Shared foundation (no ML/DL deps)
├── core/                          # Config (lags, windows, segments), logging
├── data/                          # Loading, transforms, rolling buffers, pipeline
├── features/                      # HAR/Raw lag features, PCA, factory
├── models/                        # BaseModel ABC, RollingRegressionModel, NaiveBaseline
├── backtest/                      # CPU backtest engine, Duan smearing, chunk splitting
├── evaluation/                    # Metrics (MSE, MAE, QLIKE, R²), aggregation
└── tests/                         # Core unit tests

projects/
├── ml/                            # Traditional ML
│   ├── models/                    # Ridge, XGBoost, LightGBM, RF, SARIMAX, registry
│   ├── cli/                       # Executor, job submission, experiment config
│   ├── features/                  # Feature group definitions and subgroup registry
│   ├── evaluation/                # ML-specific aggregation utilities
│   ├── scripts/                   # submit.py, aggregate.py, compare.py
│   ├── experiments/               # YAML experiment configs
│   └── tests/                     # ML model and integration tests
│
└── dl/                            # Deep learning
    ├── models/                    # PatchTST, LagAutoEncoder, QLIKE loss
    ├── backtest/                  # Multi-GPU engine, vmap kernels, scaling experiments
    ├── features/                  # AE transform (DL-specific)
    ├── data/                      # Synthetic data (MovingBlockBootstrap)
    ├── visualization/             # Forecast, scatter, residual, loss plots
    ├── cli/                       # GPU executor, lifecycle manager, submission
    ├── scripts/                   # DL runner template, aggregate, scaling experiments
    └── notebooks/                 # Colab training and visualization notebooks

writeup/                           # LaTeX paper
├── main.tex                       # Main document
├── sections/                      # abstract, intro, methodology, data, results, etc.
├── references.bib                 # Bibliography
└── figures/                       # Paper figures
```

**ml and dl are independent of each other.** Both depend on core only.

## End-to-End Data Flow

```
Raw parquet (30-min bars)
    │
    ▼
load_and_clean_base_data()        ← Grid, filter market hours, robust_transform
    │
    ▼
generate_lag_features()           ← HAR rolling means or raw lags at geometric scales
    │
    ▼
apply_horizon_shift()             ← Align features at t with targets at t+h
    │
    ▼
get_chunk_indices_strided()       ← Split into N chunks for distributed execution
    │
    ▼
create_model() + feature_transform  ← Ridge/XGB/LGBM/RF/SARIMAX + optional PCA/AE
    │
    ▼
run_backtest_agnostic()           ← Walk-forward: initialize → predict → update loop
    │
    ▼
apply_duan_smearing()             ← Convert adjusted-space → raw-space forecasts
    │
    ▼
save_chunk_results()              ← Per-chunk CSV with true/pred (adjusted + raw)
    │
    ▼
aggregate.py                      ← Stitch chunks, compute MSE/MAE/QLIKE/R²
    │
    ▼
compare.py                        ← Cross-experiment comparison tables
```

## Data Pipeline

### Loading and Cleaning (`core/data/loading.py`)

`load_and_clean_base_data()` reads parquet files from `all30min/`, then:

1. **Gridding**: Creates a complete 30-min frequency grid from 2005-01-01 to max date, reindexes to fill gaps
2. **Market hours filtering**: Drops Friday night (>20:00), Saturday, and Sunday morning (<18:30)
3. **Exogenous variable handling**: Parses from hparams; special overnight NaN-filling for equal/value-weighted stock factors and volatility demand; VIX/VVIX special handling
4. **Transforms**: Always applies `robust_transform()` to the RV target; conditionally transforms exogenous columns based on hparams flags

### Transform Pipeline (`core/data/transforms.py`)

`robust_transform()` applies three stages per column:

1. **Diurnal adjustment** — Groups by time-of-day slot, divides by rolling mean (non-negative vars) or rolling std (signed vars). Removes intraday seasonality.
2. **Semantic data transform** — Chooses transform based on column name: `sqrt` for RV/bipow/turnover, `sign(x)·sqrt(|x|)` for autocovariance, `cbrt` for ret3, fourth-root for ret4, `log` as default. Stabilizes variance.
3. **Rolling winsorization** — Clips to rolling 5th/95th quantiles to limit outlier influence.

Each stage is controlled by flags (`use_diurnal`, `use_transform`, `winsor_window`). VIX, sentiment, hour, and DOW skip diurnal adjustment by default.

### Rolling Utilities (`core/data/rolling.py`)

Online data structures for streaming walk-forward evaluation:

- **`RollingRobustScaler`** — Dual-buffer design: chronological ring buffer + sorted transposed array. Numba JIT kernels provide O(W) updates via binary search + element shift, with O(1) access to (median, IQR) for robust scaling.
- **`RollingBuffer`** — Stores (X, y) pairs in a ring buffer. `get_ordered_view()` returns chronologically-ordered data, critical for SARIMAX.
- **`RollingMedian`** — Simple rolling median over a ring buffer.

### Synthetic Data (`projects/dl/data/synth_data.py`)

`MovingBlockBootstrap` generates synthetic time series by randomly sampling contiguous blocks (default 48 = one trading day) from source data. Preserves local temporal dependencies and diurnal patterns while breaking long-range dependence. Used for data augmentation in scaling-law experiments.

## Feature Engineering

### Lag Features (`core/features/`)

A class hierarchy rooted in `BaseFeatureTransform` with dual interfaces — sklearn-style `fit`/`transform` and pandas-level `generate_pandas`:

- **`HARFeatures`** — Rolling-mean lags at geometric scales (e.g., `rolling(5).mean().shift(1)`). The core HAR representation. Features named `har_ma_{lag}`. (in `core`)
- **`RawLagFeatures`** — Simple point-shift lags (`shift(lag)`). Features named `{col}_lag_{lag}`. (in `core`)
- **`PCATransform`** — Wraps sklearn PCA for dimensionality reduction in rolling pipelines. (in `core`)
- **`AETransform`** — Hybrid autoencoder transform: trains `LagAutoEncoder` with `alpha * MSE(reconstruction) + (1-alpha) * MSE(prediction)`, then uses encoder output as compressed features. Supports weight checkpointing and loss logging. (in `dl`: `projects/dl/features/transforms.py`)

`generate_lag_features()` in `core/features/pipeline.py` is the public API. `resolve_lags()` returns geometric base-5 sequences for HAR or consecutive lags for raw features. For tree models, DOW and hour features are added automatically.

Segmented mode (`generate_lag_features_segmented()`) supports intraday time-slicing with a `lag_scope` parameter: `'global'` computes lags on the full dataset then slices (prevents lookahead bias), `'intra'` computes within each segment independently.

### Feature Groups (`projects/ml/features/feature_groups.py`)

Central registry of ~50 available exogenous features organized into subgroups:

| Subgroup | Examples |
|----------|----------|
| `moments` | sumret, sumabsret, sumret3, sumret4, sumpret2, sumbipow, sumautocov |
| `liquidity` | sumvolume, numobs, turnover, effspread |
| `market_ew` | Equal-weight cross-sectional factors (returns, volume, spread) |
| `market_vw` | Value-weight cross-sectional factors |
| `sentiment` | stocktwits_attention, stocktwits_sentiment, stocktwits_sentcount |
| `implied_vol` | vix, vvix, vix3m |
| `vol_demand` | Volatility demand elasticity variants |
| `all_features` | Complete set |

## Models

### Walk-Forward Interface (`core/models/base.py`)

`BaseModel` (ABC) defines three methods: `initialize(X_init, y_init)`, `predict(x_t)`, `update(x_t, y_t)`.

`RollingRegressionModel` is the workhorse implementation. It wraps any sklearn-like estimator with a `RollingBuffer` for training data, `RollingRobustScaler` for online normalization, and an optional `feature_transform` (PCA or AE). Configurable `refit_frequency` controls how often the model refits (every step for Ridge, every 5 steps for trees).

`NaiveBaseline` returns a specific lag value as the forecast.

### Traditional Models (`projects/ml/models/sklearn_models.py`)

| Model | Underlying | Scaling | Refit Freq | Key Defaults |
|-------|-----------|---------|------------|--------------|
| `RidgeModel` | `sklearn.linear_model.Ridge` | Yes | 1 | alpha=1.0 |
| `XGBoostModel` | `xgboost.XGBRegressor` | No | 5 | n_estimators=100, max_depth=3, lr=0.1 |
| `LightGBMModel` | `lightgbm.LGBMRegressor` | No | 5 | n_estimators=100, max_depth=3, lr=0.1 |
| `RandomForestModel` | `sklearn.ensemble.RandomForestRegressor` | No | 5 | n_estimators=100, max_depth=3 |

Linear models use robust scaling; tree models don't (scale-invariant). Trees refit less often due to higher compute cost.

### SARIMAX (`projects/ml/models/sarimax.py`)

Wraps statsmodels SARIMAX with order `(2,0,1)`, seasonal `(1,0,0,48)`. Uses chronologically-ordered views (via `get_ordered_view()`) rather than circular buffers — essential for AR/MA components. Smaller fit window (480 = 10 trading days) since parametric models need less data. Gracefully degrades to naive baseline after 5 consecutive fit failures.

### Deep Learning (`projects/dl/models/deep_learning.py`)

**PatchTST**: Hugging Face transformer-based patch time series backbone with a linear prediction head. Configured with context_len=241, patch_len=47, stride=31.

**LagAutoEncoder**: Hybrid supervised/unsupervised architecture with shared encoder (n_features → hidden_dim → n_components), decoder (reconstruction), and prediction head (→ 1 scalar). The encoder output feeds into Ridge regression — combining nonlinear representation learning with linear prediction stability. Not used as a standalone predictor.

### Loss Functions (`projects/dl/models/losses.py`)

`functional_qlike_loss`: QLIKE (Quasi-Maximum Likelihood Error) in log-space — `L = σ²_true · exp(-h) + h`. Numerically stable via clamping. Preferred for volatility forecasting due to its asymmetric penalty structure.

### Model Registry (`projects/ml/models/registry.py`)

`MODEL_REGISTRY` maps model names to `{class, defaults}`. The `create_model()` factory handles special cases: naive baseline (no buffers), SARIMAX (no feature_transform, uses horizon), and standard models (accept feature_transform and refit_frequency).

## Backtesting

### CPU Engine (`core/backtest/engine.py`)

`run_backtest_agnostic()` implements the walk-forward loop: initialize model with burn-in history → for each test step: predict → update with realized value. Returns predictions and optional coefficient history.

`apply_duan_smearing()` converts adjusted-space forecasts to raw space: `pred_raw = (forecast² + smear_factor) × baseline`. Essential for evaluating volatility forecasts in original units.

`get_chunk_indices_strided()` splits test indices into N chunks for distributed HPC execution.

### GPU Engine (`projects/dl/backtest/gpu_engine.py`)

Two strategies with unified architecture:

**PatchTST** (`run_multigpu_backtest()`): Creates 3D strided windows via `torch.as_strided` (zero-copy). Per-GPU worker runs instance normalization → compiled training kernel → predict. Predictions converted from log-space to sqrt-space via `exp(h_pred / 2.0)`.

**AE+Ridge** (`run_ae_multigpu_backtest()`): Creates 2D strided windows. Per-GPU worker runs normalize → train AE → encode training data → solve Ridge via closed-form `(X'X + αI)⁻¹X'y` → predict.

### GPU Kernels (`projects/dl/backtest/gpu_kernels.py`)

PyTorch-compiled training loops using `torch.func.vmap` + `torch.func.grad` for vectorized batch training. AdamW optimizer with gradient clipping. Two kernel factories:

- `make_train_kernel()` — PatchTST with QLIKE loss
- `make_ae_train_kernel()` — AE with hybrid reconstruction + prediction loss

### GPU Utilities (`projects/dl/backtest/gpu_utils.py`)

Shared infrastructure: chunk normalization, batched parameter allocation with fan-in initialization, Adam state management, checkpointing for fault tolerance, and multiprocessing distribution across GPUs via `torch.multiprocessing.Pool`.

### Scaling Experiments (`projects/dl/backtest/gpu_engine_scaling.py`)

`run_scaling_experiment()` studies how synthetic data augmentation affects deep learning performance: augment training data via `MovingBlockBootstrap` at various multipliers → train PatchTST → evaluate on chronological holdout → report QLIKE, MSE, MAE.

## Evaluation

### Metrics (`core/evaluation/metrics.py`)

`calculate_global_metrics()` computes:
- **Adjusted scale**: MSE, MAE, and winsorized variants
- **Raw scale**: QLIKE = `(true/pred) - log(true/pred) - 1`, and winsorized variant

`calculate_baseline_deltas()` computes improvements over naive baseline: delta metrics and out-of-sample R² = `1 - mse/baseline_mse`. Supports grouping by (segment, horizon) for multi-horizon experiments.

### Aggregation (`core/evaluation/aggregation.py`)

`load_all_chunks()` stitches per-chunk CSVs. `process_single_experiment()` loads chunks → filters by time-of-day (optional) → computes per-horizon metrics → adds cross-horizon aggregates. Supports three evaluation modes: global, pre-segmented results, and time-of-day filtering.

### Visualization (`projects/dl/visualization/plots.py`)

- `plot_timeseries_forecast()` — True vs predicted RV time series
- `plot_diagnostic_scatter()` — Log-log scatter with 45° reference line
- `plot_residual_histogram()` — Distribution of prediction errors
- `plot_training_losses()` — Dual-panel: individual chunk loss curves + mean ± std aggregate

## Setup

```bash
pip install -e core
pip install -e projects/ml
pip install -e projects/dl
```

Or install all packages at once:

```bash
pip install -e .
```

Requires Python 3.10+. Key dependencies: numpy, pandas, scikit-learn, xgboost, lightgbm, statsmodels, torch (>=2.0), transformers (>=4.30), numba, pyarrow, [claude-hpc](https://github.com/jamesdchen/claude-hpc).

## Quick Start

Run a single-chunk backtest (explicit CLI args):

```bash
python -m projects.ml.cli.executor \
    --model ridge \
    --features har \
    --input-path all30min \
    --output-file results/results_chunk_1.csv \
    --chunk-id 1 \
    --total-chunks 100
```

Or using env vars (compatible with claude-hpc templates):

```bash
export CHUNK_ID=0 TOTAL_CHUNKS=100 RESULT_DIR=results/
python -m projects.ml.cli.executor --model ridge --features har
```

Key CLI options:
- `--model`: ridge, xgboost, lightgbm, random_forest, sarimax, naive
- `--features`: har (rolling means), pca (compressed), ae (autoencoder-compressed)
- `--train-window`: Rolling window size in days (default varies by model)
- `--horizon`: Forecast horizon, 1-48 steps ahead
- `--segment`: morning, midday, closing, overnight, or all
- `--n-components`, `--ae-alpha`, `--ae-epochs`: Feature transform configuration
- `--save-coefs`: Save model coefficients to `.npz`

Run a GPU deep learning backtest:

```bash
python -m projects.dl.cli.gpu_executor \
    --experiment patchts \
    --input-path all30min \
    --output results/results_dl.csv \
    --gpu-count 2 \
    --epochs 50 \
    --batch-size 32
```

## Running Tests

```bash
# All tests
pytest core/tests/ projects/ml/tests/

# Skip slow/GPU tests
pytest core/tests/ projects/ml/tests/ -m "not slow and not gpu"

# Per-package (run from package directory)
cd core && pytest
cd projects/ml && pytest
```

## HPC Workflow

### Submission

Submit experiment batches via `projects/ml/scripts/submit.py`, which supports six modes:

```bash
# Compare models on baseline HAR features
python projects/ml/scripts/submit.py model_comparison --models ridge xgboost lightgbm

# Compare feature engineering methods
python projects/ml/scripts/submit.py feature_transforms --model ridge --features har pca ae

# Feature ablation: one job per feature in a subgroup
python projects/ml/scripts/submit.py individual_features --model ridge --subgroup moments

# Cartesian product: subgroups × models × features
python projects/ml/scripts/submit.py subgroup_analysis --models all --features all --subgroups all

# Naive baseline (cached and symlinked to other experiments)
python projects/ml/scripts/submit.py naive

# From a declarative YAML config
python projects/ml/scripts/submit.py from-config projects/ml/experiments/example_model_comparison.yaml
```

Common options: `--result-dir`, `--total-chunks` (default 100), `--backend` (slurm, sge, dry-run), `--no-naive`.

Each submission creates an experiment directory with `config.txt` (human-readable), `metadata.json` (git hash, branch, dirty status, Python version, timestamp), and a `.needs_aggregation` marker.

### YAML Experiment Configs

Declarative experiment definitions in `projects/ml/experiments/`:

```yaml
name: ridge_vs_trees
mode: model_comparison
models: [ridge, xgboost, lightgbm]
features: [har]
train_window: 500
horizon: 1
total_chunks: 100
backend: slurm
notes: "Baseline model comparison using HAR features"
```

### Aggregation and Comparison

```bash
python projects/ml/scripts/aggregate.py              # stitch chunks, compute metrics, calculate baseline deltas
python projects/ml/scripts/compare.py results/model_comparison --metric qlike --sort asc
python projects/dl/scripts/run_scaling_experiment.py  # GPU scaling-law sweep (multipliers: 0, 1, 2, 5, 10, 50)
```

`aggregate.py` auto-discovers `.needs_aggregation` markers, stitches chunk CSVs, computes MSE/MAE/QLIKE/R², and calculates improvements over naive baseline.

### HPC Backends

Job submission uses the [`claude-hpc`](https://github.com/jamesdchen/claude-hpc) package for all HPC infrastructure. There are no project-specific job templates — harxhar uses claude-hpc's generic `cpu_array` and `gpu_array` templates, configured via environment variables.

**Architecture:**
- `project.yaml` — defines stages (ml_backtest, dl_backtest, scaling) with executor commands, template names, and per-cluster conda/module settings
- `config/clusters.yaml` (in claude-hpc) — cluster connection details (host, user, scheduler, conda_source, GPU types)
- `hpc.backends` — pluggable backends (SLURM, SGE, SGE-remote, Dry-run) from claude-hpc
- `hpc.remote` — SSH/rsync utilities from claude-hpc (explicit host/user from config)

**Available backends:** SLURM, SGE, SGE-remote (via SSH), Dry-run.

**Template convention:** Executors read `CHUNK_ID`, `TOTAL_CHUNKS`, and `RESULT_DIR` from environment variables when CLI args aren't provided. The generic templates set these automatically.

```bash
pip install -e /path/to/claude-hpc  # required dependency
```

## Notebooks

Jupyter/Colab notebooks for deep learning model training and visualization are in `projects/dl/notebooks/`:

- `patchts_colab.ipynb` / `patchts_viz.ipynb` — PatchTST training and results visualization
- `ae_ridge_colab.ipynb` / `ae_ridge_viz.ipynb` — Autoencoder+Ridge training and results visualization
- `scaling_law_colab.ipynb` / `scaling_law_viz.ipynb` — Scaling-law experiments and visualization
- `dl_runner.ipynb` — General deep learning runner with Drive-persisted status tracking

`projects/dl/notebook_utils.py` provides shared utilities: CUDA configuration (TF32), GPU monitoring via nvidia-smi, Drive-persisted status management with atomic writes, and results download helpers.

See `projects/dl/COWORK_DL_INSTRUCTIONS.md` for detailed deep learning workflow guidance.

## Key Design Decisions

- **Geometric lag scales** (1, 5, 25, 125, 625, 3125) capture multi-horizon temporal patterns efficiently — the core HAR insight
- **Diurnal adjustment before lag construction** removes intraday seasonality that would otherwise contaminate lag features
- **Dual-buffer RollingRobustScaler** with Numba JIT enables O(W) online updates without full re-sorting
- **SARIMAX uses chronologically-ordered views** while other models use circular buffers — respecting the parametric model's stationarity assumptions
- **AE as feature transform, not standalone predictor** — the encoder compresses features for Ridge, combining nonlinear representation learning with linear prediction stability
- **Chunk-based parallelism** (default 100 chunks) maps naturally to HPC array jobs, with each chunk independently processable
- **QLIKE loss for deep learning** rather than MSE — better suited for volatility's asymmetric error structure
- **vmap + functional_call** for GPU training avoids Python loops over batch elements, maximizing GPU utilization

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
mypy core/ projects/ml/ projects/dl/ --ignore-missing-imports
```

### CI Pipeline

GitHub Actions runs three jobs on push/PR:

1. **lint** — `ruff check .` + `ruff format --check .`
2. **typecheck** — `mypy core/ projects/ml/ projects/dl/ --ignore-missing-imports`
3. **test** — `pytest core/tests/ projects/ml/tests/ -m "not slow and not gpu" --tb=short`
