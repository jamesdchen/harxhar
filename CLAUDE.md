# CLAUDE.md — HARXHAR Project Guide

## Project Overview

**HARXHAR** is a research codebase for high-frequency intraday realized volatility forecasting. It extends the classical HAR (Heterogeneous AutoRegressive) model with eXogenous features (HARXHAR) and benchmarks a suite of ML models in a strict walk-forward backtesting framework.

- **Target variable**: `adj_RV` — diurnally adjusted, sqrt-transformed 30-minute realized variance
- **Frequency**: 30-minute bars (48 bars/trading day)
- **History**: 2005-01-01 to present
- **HPC platform**: USC CARC (Slurm)

---

## Repository Structure

```
harxhar/
├── harx.py                    # Entry point: global (continuous) forecasting
├── harx_tod.py                # Entry point: time-of-day segmented forecasting
├── aggregate.py               # Evaluate single-experiment results (intraday MSE/QLIKE)
├── aggregate_exp.py           # Multi-experiment summary table with baseline deltas
├── submit_subgroup_analysis.py# Batch Slurm submission: model × feature subgroup grid
├── submit_moments.py          # Batch Slurm submission: individual moment features
├── submit_carc.slurm          # Slurm job template (USC CARC)
├── run_best.slurm             # Run the best-known model configuration
├── all30min/                  # Input data (Parquet files, one per feature group)
│   ├── core_stats.parquet     # RV, return moments (sumret, sumabsret, etc.)
│   ├── ewstock_stats.parquet  # Equal-weight stock microstructure
│   ├── vwstock_stats.parquet  # Value-weight stock microstructure
│   ├── spy_and_sentiment.parquet # SPY turnover, StockTwits sentiment
│   ├── time_categories.parquet   # Hour, day-of-week categories
│   └── vix_and_voldemand.parquet # VIX, VVIX, VIX3M, vol demand indices
├── results_best_model/        # Output CSVs for the best model run
├── results_ridge_subgroups/   # Output CSVs for subgroup experiments
├── src/                       # Core library
│   ├── config.py              # Central configuration constants
│   ├── data_main.py           # Data loading, cleaning, diurnal transforms
│   ├── data.py                # Global (continuous) lag feature engineering
│   ├── data_tod.py            # Segmented (time-of-day) lag feature engineering
│   ├── data_helper.py         # Chunk index splitting and result saving
│   ├── models.py              # All model classes (BaseModel interface)
│   ├── rolling.py             # Numba-JIT rolling buffers and robust scaler
│   ├── autoencoder.py         # PyTorch hybrid autoencoder for lag compression
│   ├── backtest.py            # Walk-forward backtesting engine
│   ├── executor.py            # CLI argument parsing and model dispatch
│   ├── metrics.py             # MSE, MAE, QLIKE loss functions
│   └── eval_utils.py          # Multi-experiment result loading and evaluation
└── unused/                    # Deprecated variants (do not use)
```

---

## Core Concepts

### HAR Lags

All features are constructed at a geometric lag sequence defined in `src/config.py`:

```python
HAR_LAGS = [1, 5, 25, 125, 625, 3125]  # ~30min, 2.5h, half-day, 2.6d, 13d, 65d
```

Two feature modes (controlled by `--model` / `hparams['feature_type']`):
- **`har`** (HAR model only): rolling means over each lag window — `har_ma_{lag}`
- **`raw`** (all other models): individual point lags — `{col}_lag_{lag}`

### Data Pipeline (`src/data_main.py`)

`load_and_clean_base_data(hparams, input_path)` is the shared foundation for all pipelines:

1. **Load**: Merges all Parquet files in `all30min/` via outer join on `endbartime`
2. **Grid**: Reindex to a complete 30-minute grid from START_DATE to the last timestamp
3. **Filter**: Drop weekends and pre-start rows
4. **Circuit breakers**: Forward-fill RV=0 on March 2020 halt dates; optionally drop rows for non-moments exog columns
5. **Transform** (`robust_transform`): Per-column pipeline:
   - *Diurnal adjustment*: divide by per-time-slot rolling mean (non-negative) or std (signed)
   - *Data-driven transform*: sqrt for RV/ret2/turnover; cbrt for ret3; fourth-root for ret4; log default
   - *Winsorization*: rolling 5th–95th percentile clip (default window=240)
6. Returns `data` DataFrame and `cols_to_transform` list

### Feature Engineering

**Global mode** (`src/data.py` — used by `harx.py`):
- Computes lags on the full contiguous time series
- Ensures temporal continuity across all time-of-day boundaries

**Segmented mode** (`src/data_tod.py` — used by `harx_tod.py`):
- Splits data by time-of-day segments defined in `config.SEGMENT_DEFINITIONS`
- `lag_scope='global'` (default): lags computed on full series, then segmented
- `lag_scope='intra'`: lags computed within each segment independently

Segments (with overlapping boundaries):
```python
SEGMENT_DEFINITIONS = {
    'morning':   {'start': 510, 'end': 660},   # 08:30–11:00
    'midday':    {'start': 630, 'end': 870},   # 10:30–14:30
    'closing':   {'start': 840, 'end': 960},   # 14:00–16:00
    'overnight': {'start': 990, 'end': 510}    # 16:30–08:30 (wraps)
}
```

### Walk-Forward Backtesting (`src/backtest.py`)

`run_backtest_agnostic(model, indices, X, y, train_win_periods)`:
1. Calls `model.initialize(X_init, y_init)` with the burn-in window
2. Iterates over test indices; at each step:
   - `model.predict(x_t)` → forecast
   - `model.update(x_t, y_t)` → online update with realized value

### Model Interface (`src/models.py`)

All models implement `BaseModel`:
```python
class BaseModel:
    def initialize(self, X_init, y_init): ...  # Burn-in fit
    def predict(self, x_t): ...                # One-step-ahead forecast
    def update(self, x_t, y_t): ...            # Online update after realization
```

**`RollingRegressionModel`** — engine used by most models. Manages:
- `RollingBuffer`: circular ring buffer for training data
- `RollingRobustScaler`: Numba-JIT median/IQR scaler with O(W) sorted-array maintenance
- Configurable `refit_frequency` (Ridge=1, tree models=5)

### Available Models

| `--model` flag | Class | Notes |
|---|---|---|
| `har` | `RidgeModel` | HAR rolling-mean features, Ridge regression |
| `ridge` | `RidgeModel` | Raw individual lags, Ridge regression |
| `naive` | `NaiveBaseline` | Persistence: uses lag-125 as forecast |
| `xgboost` | `XGBoostModel` | No scaling, NaN-native, refit every 5 steps |
| `lightgbm` | `LightGBMModel` | No scaling, NaN-native, refit every 5 steps |
| `random_forest` | `RandomForestModel` | No scaling, refit every 5 steps |
| `sarimax` | `SARIMAXModel` | ARIMA(2,0,1)×(1,0,0,48), fit_window=480, refit every 48 steps |
| `pca_ridge` | `PCALagRidgeModel` | PCA compression of raw lags + Ridge |
| `ae_ridge` | `AutoEncoderLagRidgeModel` | Hybrid PyTorch autoencoder + Ridge, refit every 240 steps |

### Rolling Infrastructure (`src/rolling.py`)

- **`RollingRobustScaler`**: Maintains a sorted buffer alongside the chronological buffer. `_update_sorted_matrix` (Numba) updates both in O(W) per step. `get_scaler()` returns (median, IQR) in O(1).
- **`RollingBuffer`**: Simple ring buffer for regression training data. `get_ordered_view()` returns data in chronological order (needed by SARIMAX).
- **`RollingMedian`**: Simpler rolling median without the sorted optimization.

### Output Format

Each backtest chunk writes a CSV with columns:
```
date, true_adj, pred_adj, true_raw, pred_raw
```
- `*_adj`: sqrt-transformed, diurnally adjusted space
- `*_raw`: reconstructed raw RV space using Duan's smearing estimator:
  `pred_raw = (pred_adj² + smear) × baseline_RV`

---

## Evaluation

### Single Experiment (`aggregate.py`)
```bash
python aggregate.py --num_files 100 --file_pattern "results_best_model/results_chunk_{}.csv"
```
Reports intraday MSE (on logs) and QLIKE following Zhang et al. (2023).

### Multi-Experiment (`aggregate_exp.py`)
Loads all `exp_*/` directories under a results folder, computes per-segment metrics, and produces a summary table with:
- `mse`, `mae` (adjusted space)
- `qlike` (raw space, filtered)
- `delta_mse_raw`, `delta_mae_raw`, `delta_qlike` vs. naive baseline
- `oos_r2 = 1 - MSE_model / MSE_baseline`

Loss functions (`src/metrics.py`):
- **MSE**: on adjusted (sqrt) space
- **QLIKE**: `E[σ²/σ̂² - log(σ²/σ̂²) - 1]` — robust to outliers, standard in volatility literature

---

## Running Experiments

### Local Single Run
```bash
python harx.py \
    --model ridge \
    --output-file results/chunk_0.csv \
    --chunk-id 0 \
    --total-chunks 1 \
    --train-window 500 \
    --exog-cols "sumabsret|sumpret2|sumbipow"
```

### Time-of-Day Segmented Run
```bash
python harx_tod.py \
    --model har \
    --output-file results/chunk_0.csv \
    --chunk-id 0 \
    --total-chunks 1 \
    --train-window 500
```

### HPC Batch Submission (Slurm/CARC)

**Subgroup experiment grid**:
```bash
python submit_subgroup_analysis.py \
    --models ridge xgboost lightgbm \
    --subgroups moments liquidity all_features \
    --result-dir results_subgroups \
    --total-chunks 100
```

**Individual moments experiment**:
```bash
python submit_moments.py
```

**Best model run**:
```bash
sbatch --array=1-100 run_best.slurm
```

### Key CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Model type (see table above) |
| `--input-path` | `all30min` | Path to Parquet data directory |
| `--output-file` | required | Path for result CSV |
| `--chunk-id` | required | 0-indexed chunk number |
| `--total-chunks` | required | Total parallel chunks |
| `--train-window` | 500 | Training window in calendar days |
| `--exog-cols` | None | Pipe-separated exogenous column names |
| `--lag-scope` | `global` | `global` or `intra` lag computation |
| `--n-components` | 5 | Latent dim for `pca_ridge`/`ae_ridge` |
| `--ae-alpha` | 0.5 | AE loss weight: α·recon + (1-α)·pred |
| `--ae-epochs` | 50 | AE training epochs per refit |

---

## Feature Subgroups

Defined in `submit_subgroup_analysis.py`:

| Subgroup | Description |
|---|---|
| `baseline` | No exogenous features (HAR-only) |
| `moments` | `sum*` columns excluding `*stock*` and `*volume*` |
| `liquidity` | Volume, turnover, bid-ask spread features |
| `market_ew` | Equal-weight stock return moments |
| `market_vw` | Value-weight stock return moments |
| `sentiment` | StockTwits attention, sentiment, count |
| `implied_vol` | VIX, VVIX, VIX3M |
| `vol_demand` | Volatility demand indices (SPX/all, open/close) |
| `all_features` | All of the above combined |

---

## Key Conventions

### NaN Handling
- Tree models (`xgboost`, `lightgbm`): `allow_missing=True` — NaNs passed through natively
- All others: `allow_missing=False` — rows with any NaN dropped after lag burn-in

### Transform Toggle
- Tree models: `use_transform=False` (raw values preferred for trees)
- Others: `use_transform=True` (log/sqrt transforms applied)

### Diurnal Adjustment
- Applied to all features except `{hour, DOW, t, date, vix, sentiment}` columns
- Non-negative features: divide by per-slot rolling mean (window=20, min_periods=5)
- Signed features: divide by per-slot rolling std

### Circuit Breaker Dates
The four March 2020 market-halt days (`2020-03-09`, `2020-03-12`, `2020-03-16`, `2020-03-18`) are handled specially:
- For moments-only features: RV=0 slots are forward-filled
- For experiments with non-moments exogenous columns: those rows are dropped entirely; output filenames are suffixed `_cb_drop`

### Result File Naming
- Standard: `results_chunk_{N}.csv`
- Circuit-breaker drop: `results_chunk_{N}_cb_drop.csv`
- TOD segmented: `results_chunk_{N}_{segment}.csv`

### Chunking
The test period is split into `total_chunks` equal slices by `get_chunk_indices_strided()`. Slurm array tasks are 1-indexed; the SLURM template subtracts 1 (`$((SLURM_ARRAY_TASK_ID-1))`) before passing to Python.

### Seed
`np.random.seed(42)` is set at the top of each entry-point script.

---

## Dependencies

Key packages (no requirements.txt — install manually):
- `numpy`, `pandas`, `pyarrow` — data handling
- `scikit-learn` — Ridge, PCA, XGBRegressor wrapper, RandomForest
- `xgboost`, `lightgbm` — tree models
- `statsmodels` — SARIMAX
- `torch` — autoencoder
- `numba` — JIT-compiled rolling statistics
- `tqdm` — progress bar in backtester

---

## Adding a New Model

1. Implement the `BaseModel` interface in `src/models.py`
2. Add an `elif args.model == 'your_model':` block in `execute_chunk_backtest()` in `src/executor.py`
3. Add `'your_model'` to the `choices` list in `get_common_parser()`
4. Update `get_common_hparams()` if the model needs special `use_transform` or `allow_missing` settings
5. Add the model to `ALL_MODELS` in `submit_subgroup_analysis.py` if it should appear in batch sweeps

## Adding New Exogenous Features

1. Add raw column to the appropriate Parquet file in `all30min/`
2. Add the column name to `FULL_FEATURE_STRING` in `submit_subgroup_analysis.py` and/or `submit_moments.py`
3. If the column needs special overnight NaN-filling, add it to `OVERNIGHT_WINDOWS` in `src/data_main.py`
4. If the column should be excluded from diurnal adjustment, add it to `diurnal_excluded_cols` in `robust_transform()`
