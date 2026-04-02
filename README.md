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
    ├── chunk_loader.py            Stitch chunk CSVs (login node)
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
load_and_stitch_chunks()          ← Login node: stitch all chunks
    │
    ▼
calculate_metrics()               ← Login node: MSE, MAE, QLIKE
```

## Notebooks

Each notebook walks through the pipeline stage-by-stage with data inspection cells, then `%%writefile`s a standalone Python module to `colab/src/`.

**Pipeline notebooks** (run in order to understand the data):
- `01_loading` — raw parquets → merged 30-min grid → market hours filter → NaN handling
- `02_transforms` — diurnal adjustment → semantic transform (sqrt/log) → winsorization
- `03_evaluation` — Duan smearing, QLIKE loss, MSE/MAE, chunk loading

**Experiment notebooks** (each imports from `src/loading` and `src/transforms`):
- `ml_ridge` — Ridge with HAR features, robust scaling, refit every step
- `ml_xgboost` — XGBoost with HAR + DOW/hour features, NaN passthrough, refit every 5 steps
- `ml_lightgbm` — LightGBM, same pattern as XGBoost
- `ml_pcr` — PCA-compressed log-spaced lags + Ridge, PCA refit every 240 steps
- `ml_baseline` — Naive lag-1 (har_ma_125) baseline
- `dl_patchts` — PatchTST transformer with QLIKE loss, GPU multi-worker
- `dl_ae_ridge` — Hybrid autoencoder + closed-form Ridge, GPU multi-worker

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

# Evaluate on login node (after all chunks finish)
python -c "
from colab.src.chunk_loader import load_and_stitch_chunks
from colab.src.evaluation import calculate_metrics
df = load_and_stitch_chunks('results/')
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

## Key Design Decisions

- **Geometric lag scales** (1, 5, 25, 125, 625, 3125) capture multi-horizon temporal patterns — the core HAR insight
- **Diurnal adjustment before lag construction** removes intraday seasonality
- **Numba JIT `RollingRobustScaler`** enables O(W) online median/IQR scaling via sorted-array maintenance
- **QLIKE loss for deep learning** — asymmetric penalty suited for volatility forecasting
- **`vmap` + `functional_call`** for GPU training avoids Python loops, maximizing GPU utilization
- **Notebooks → `%%writefile` → `src/`** — human-readable pipeline inspection that produces machine-executable code
- **Executors don't evaluate** — chunk CSVs are stitched and evaluated separately on the login node via `chunk_loader` + `evaluation`

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
