# HARXHAR

Realized volatility forecasting system using HAR-family models with exogenous features. Supports rolling-window backtesting across Ridge, XGBoost, LightGBM, PCA+Ridge (PCR), and deep learning (PatchTST, Autoencoder+Ridge) models on intraday 30-minute bar data.

## Architecture

```
colab/
├── notebooks/
│   ├── pipeline/                  Shared pipeline stages (run in order)
│   │   ├── 01_loading.ipynb       Load parquets, grid, filter market hours
│   │   ├── 02_transforms.ipynb    Diurnal adjust, semantic transform, winsorize
│   │   └── 03_evaluation.ipynb    Duan smearing, QLIKE/MSE/MAE metrics
│   ├── ml_ridge.ipynb             Ridge regression (CPU)
│   ├── ml_xgboost.ipynb           XGBoost (CPU)
│   ├── ml_lightgbm.ipynb          LightGBM (CPU)
│   ├── ml_pcr.ipynb               PCA + Ridge (CPU)
│   ├── ml_baseline.ipynb          Naive baseline (CPU)
│   ├── dl_patchts.ipynb           PatchTST transformer (GPU)
│   └── dl_ae_ridge.ipynb          Autoencoder + Ridge (GPU)
│
└── src/                           Standalone executors (%%writefile output)
    ├── loading.py                 Data loading
    ├── transforms.py              Data transforms
    ├── evaluation.py              Metrics (login node)
    ├── ml_ridge.py                Ridge executor
    ├── ml_xgboost.py              XGBoost executor
    ├── ml_lightgbm.py             LightGBM executor
    ├── ml_pcr.py                  PCR executor
    ├── ml_baseline.py             Baseline executor
    ├── dl_patchts.py              PatchTST executor
    └── dl_ae_ridge.py             AE+Ridge executor

all30min/                          Input data (6 parquet files, ~66 MB)
writeup/                           LaTeX paper and figures
```

## End-to-End Data Flow

```
Raw parquet (30-min bars)
    │
    ▼
load_raw_data()                   ← Grid, filter market hours, NaN policy
    │
    ▼
robust_transform()                ← Diurnal adjust → semantic transform → winsorize
    │
    ▼
generate_har_features()           ← Rolling means at geometric base-5 lags [1,5,25,125,625,3125]
    │
    ▼
apply_horizon_shift()             ← Align features at t with targets at t+h
    │
    ▼
chunk index range                 ← --chunk-id / --total-chunks for HPC parallelism
    │
    ▼
walk-forward backtest             ← initialize → predict → update loop
    │
    ▼
apply_duan_smearing()             ← Convert adjusted-space → raw-space forecasts
    │
    ▼
save chunk CSV                    ← Per-chunk results_chunk_*.csv
    │
    ▼
calculate_metrics()               ← Login node: MSE, MAE, QLIKE (claude-hpc handles chunk stitching)
```

## Notebooks

Each notebook is the single human-readable source of truth for its matching module in `src/`. A cell whose first non-blank line is `# export` ships to `src/<name>.py` verbatim; every other cell is notebook-only (exploration, plots, tests). The final cell of every notebook calls `export_notebook(...)` from `notebooks/_exporter.py`, which concatenates the `# export` cells in notebook order and writes the module. No `%%writefile` cells, no hand-copied HEADER strings, no drift.

**Reviewing a module means reading the `# export` cells of its notebook, top to bottom, with the adjacent test cells as the proof.** The final `export_notebook(...)` call regenerates `src/<name>.py` exactly — any hand-edit to `src/` will be overwritten the next time the notebook is run.

**Pipeline notebooks** (run in order to understand the data):
- `01_loading` — raw parquets → merged 30-min grid → market hours filter → NaN handling
- `02_transforms` — diurnal adjustment → semantic transform (sqrt/log) → winsorization
- `03_evaluation` — Duan smearing, QLIKE loss, MSE/MAE
- `04_scaling` — rolling robust scaler + walk-forward backtest numba kernels

**Experiment notebooks** (each imports from `src/loading` and `src/transforms`):
- `ml_ridge` — Ridge with HAR features, robust scaling, refit every step
- `ml_xgboost` — XGBoost with HAR + DOW/hour features, NaN passthrough, refit every 5 steps
- `ml_lightgbm` — LightGBM, same pattern as XGBoost
- `ml_pcr` — PCA-compressed log-spaced lags + Ridge, PCA refit every 240 steps
- `ml_baseline` — Naive lag-1 (har_ma_125) baseline
- `ml_random_forest` — RandomForest with HAR features
- `tune_tree` — Optuna hyperparameter tuning for tree-based models
- `dl_patchts` — PatchTST transformer with QLIKE loss, GPU multi-worker
- `dl_ae_ridge` — Hybrid autoencoder + closed-form Ridge, GPU multi-worker

`.hpc/tasks.py` has no owning notebook. Its `FLAGS` dict and the
open-loop `_OPEN_LOOP_TASKS` chunk plan are baked by `.hpc/_build_tasks.py`
(probes the post-feature series length, runs `discover_runs` + `plan_tasks`);
re-run that script after a data-vintage, HAR-lag, or `run()`-signature change.

## Runtime Requirements

| Notebook | Runtime | Estimated Time |
|----------|---------|---------------|
| Pipeline (01-03) | CPU | < 5 min |
| ml_ridge | CPU | ~10 min |
| ml_xgboost | CPU | ~15 min |
| ml_lightgbm | CPU | ~15 min |
| ml_pcr | CPU | ~20 min |
| ml_baseline | CPU | < 5 min |
| dl_patchts | GPU (T4+) | ~30 min |
| dl_ae_ridge | GPU (T4+) | ~45 min |

## Executors

Each `colab/src/*.py` executor is a standalone CLI script:

```bash
# Run a single chunk (compute node)
python colab/src/ml_ridge.py \
    --data-path all30min \
    --horizon 1 \
    --train-window 500 \
    --chunk-id 0 \
    --total-chunks 100 \
    --output-file results/chunk_0.csv

# GPU experiment
python colab/src/dl_patchts.py \
    --data-path all30min \
    --horizon 1 \
    --gpu-count 2 \
    --chunk-id 0 \
    --total-chunks 10 \
    --output-file results/chunk_0.csv

# Evaluate on login node (after claude-hpc stitches chunks)
python -c "
from colab.src.evaluation import calculate_metrics
import pandas as pd
df = pd.read_csv('results/stitched.csv')
print(calculate_metrics(df))
"
```

Executors have **zero imports from `core/` or `projects/`** — only numpy, pandas, sklearn, torch, etc.

## HPC Workflow

All HPC infrastructure is handled by [`claude-hpc`](https://github.com/jamesdchen/claude-hpc). `hpc.yaml` defines per-experiment profiles:

| Profile | Executor | Chunks | Resources |
|---------|----------|--------|-----------|
| `ml_ridge` | `colab/src/ml_ridge.py` | 100 | 1 CPU, 16G, 4h |
| `ml_xgboost` | `colab/src/ml_xgboost.py` | 100 | 1 CPU, 16G, 4h |
| `ml_lightgbm` | `colab/src/ml_lightgbm.py` | 100 | 1 CPU, 16G, 4h |
| `ml_pcr` | `colab/src/ml_pcr.py` | 100 | 1 CPU, 16G, 4h |
| `ml_baseline` | `colab/src/ml_baseline.py` | 100 | 1 CPU, 8G, 1h |
| `dl_patchts` | `colab/src/dl_patchts.py` | 10 | 4 CPU, 2×A100, 16G, 6h |
| `dl_ae_ridge` | `colab/src/dl_ae_ridge.py` | 10 | 4 CPU, 2×A100, 16G, 6h |

## Key Invariants

Properties that must hold for results to be valid. Audit reviewers should verify each at the cited file:line.

**1. Duan smearing (log/sqrt -> raw transform with bias correction)**
- Formula: `pred_raw = (forecast^2 + smear) * baseline`, where `smear = mean((y_true - forecasts)^2)`.
- Why: forecasts live on the adjusted scale (sqrt-RV / log-RV after `robust_transform`). Naive squaring underestimates raw variance by Jensen's inequality; the smearing term restores it nonparametrically (Duan, 1983).
- Source: `src/evaluation.py:apply_duan_smearing` (lines 20-50).

**2. Dropna policy (intersection-N for cross-feature comparison)**
- Baseline / HAR-only runs drop on `RV` only: `df.dropna(subset=["RV"])` (`src/ml_xgboost.py:56`, `src/ml_lightgbm.py:56`, `src/ml_baseline.py:35`, `src/loading.py:204`).
- Exogenous-feature runs drop on `RV + exog_cols`: `df.dropna(subset=["RV"] + exog_cols)` (`src/ml_ridge.py:114`, `src/ml_random_forest.py:50`, `src/ml_pcr.py:164`).
- Consequence: baseline-vs-exog comparisons MUST be evaluated on the intersection sample. The 2026-04-23 audit found Ridge's apparent 6% liquidity gain disappeared once both arms were aligned on intersection-N — apparent gains can otherwise reflect a smaller, easier sample.

**3. Strict-causality feature lag (`shift(1)` everywhere)**
- All HAR rolling means, exogenous lag features, diurnal baselines, and winsorization quantiles are produced via `.shift(1)` so that any feature/quantile at time t depends only on data through t-1. No look-ahead.
- Source: `src/transforms.py:generate_har_features` (line 266), `diurnal_adjust` (lines 88, 92), `rolling_winsorize` (lines 171-172).

**4. Refit cadence**
- Ridge: `refit_frequency=1` (every step) — sklearn Ridge is closed-form and cheap (`src/ml_ridge.py:63`).
- XGBoost / LightGBM: CLI-tunable via `--refit-frequency`, default `1`; HPC profiles typically override (`src/ml_xgboost.py:32-36`, `src/ml_lightgbm.py:32-36`).
- RandomForest: hardcoded `refit_frequency=5` (`src/ml_random_forest.py:91`).
- PCR: hardcoded `refit_frequency=240` (PCA refit only; Ridge head refits every step) (`src/ml_pcr.py:115`).
- Loop logic: `src/scaling.py:run_backtest` (line 231) — refits when `(t - train_win + 1) % refit_frequency == 0`.

**5. Reproducibility seeds**
- DL executors: `SEED=42` pinned through `_seed_everything()` covering `random`, `numpy`, `torch`, `torch.cuda`, `cudnn.deterministic=True` (`src/dl_patchts.py:321-333`, `src/dl_ae_ridge.py:342-354`). Trial-level overrides via `--seed` (default 42).
- HPC dispatchers: per-trial seed varies via `--seed` CLI flag, default 42.
- sklearn / Ridge / PCR: closed-form solvers, no RNG. XGBoost / LightGBM rely on library defaults today (no explicit `random_state` set in `src/`); flag if reproducibility across XGB/LGBM trials is required.

## Key Design Decisions

- **Geometric lag scales** (1, 5, 25, 125, 625, 3125) capture multi-horizon temporal patterns — the core HAR insight
- **Diurnal adjustment before lag construction** removes intraday seasonality
- **Numba JIT `RollingRobustScaler`** enables O(W) online median/IQR scaling via sorted-array maintenance
- **QLIKE loss for deep learning** — asymmetric penalty suited for volatility forecasting
- **`vmap` + `functional_call`** for GPU training avoids Python loops, maximizing GPU utilization
- **Notebooks → `%%writefile` → `src/`** — human-readable pipeline inspection that produces machine-executable code
- **Executors don't evaluate** — chunk CSVs are stitched by `claude-hpc` and evaluated separately on the login node via `evaluation`

## Setup

```bash
pip install numpy pandas scikit-learn pyarrow numba tqdm xgboost lightgbm  # ML
pip install torch transformers                                              # DL
```

Requires Python 3.10+.

## Development

```bash
ruff check colab/src/     # lint
ruff format colab/src/    # format
```
