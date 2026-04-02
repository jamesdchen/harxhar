# HARXHAR Colab Notebooks

Interactive notebooks for inspecting the volatility forecasting pipeline stage-by-stage,
plus self-contained experiment executors that output chunk CSVs for HPC parallelization.

## Structure

```
notebooks/
  pipeline/                      Shared pipeline stages (run in order)
    01_loading.ipynb              Load parquets, grid, filter market hours
    02_transforms.ipynb           Diurnal adjust, semantic transform, winsorize
    03_evaluation.ipynb           Duan smearing, QLIKE/MSE/MAE metrics
  ml_ridge.ipynb                  Ridge regression (CPU)
  ml_xgboost.ipynb                XGBoost (CPU)
  ml_lightgbm.ipynb               LightGBM (CPU)
  ml_pcr.ipynb                    PCA + Ridge (CPU)
  ml_baseline.ipynb               Naive lag-1 baseline (CPU)
  dl_patchts.ipynb                PatchTST transformer (GPU)
  dl_ae_ridge.ipynb               Autoencoder + Ridge (GPU)

src/                             %%writefile output — standalone modules
  loading.py                      Data loading
  transforms.py                   Data transforms
  evaluation.py                   Metrics (login node)
  chunk_loader.py                 Stitch chunk CSVs (login node, migratable to claude-hpc)
  ml_ridge.py                     Ridge executor
  ml_xgboost.py                   XGBoost executor
  ml_lightgbm.py                  LightGBM executor
  ml_pcr.py                       PCR executor
  ml_baseline.py                  Baseline executor
  dl_patchts.py                   PatchTST executor
  dl_ae_ridge.py                  AE+Ridge executor
```

## Workflow

### 1. Explore the pipeline

Run the pipeline notebooks in order to understand what happens to the data at each stage:
- **01_loading** — raw parquets → merged 30-min grid → market hours filter → NaN handling
- **02_transforms** — diurnal adjustment → semantic transform (sqrt/log) → winsorization
- **03_evaluation** — Duan smearing, QLIKE loss, MSE/MAE

### 2. Run an experiment

Each experiment notebook walks through model-specific code, then `%%writefile`s a standalone
executor to `src/`. You can run the notebook end-to-end on Colab to inspect results interactively.

### 3. HPC execution

The `src/` executors accept `--chunk-id` and `--total-chunks` for parallelization:

```bash
# Compute nodes (parallelized)
python src/ml_ridge.py --data-path all30min --horizon 1 --chunk-id 0 --total-chunks 100 --output-file results/chunk_0.csv

# Login node (after all chunks finish)
python -c "
from src.chunk_loader import load_and_stitch_chunks
from src.evaluation import calculate_metrics
df = load_and_stitch_chunks('results/')
print(calculate_metrics(df))
"
```

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

## Dependencies

**ML notebooks (CPU):**
```
numpy pandas scikit-learn pyarrow numba tqdm xgboost lightgbm
```

**DL notebooks (GPU):**
```
torch transformers numpy pandas scikit-learn tqdm pyarrow
```
