# Harxhar Project

ML (Ridge/XGBoost/LightGBM/RandomForest) and DL (PatchTST/AE+Ridge) backtesting pipelines for financial volatility forecasting.

## HPC Configuration

### Cluster
- **Cluster:** Hoffman2 (UCLA IDRE)
- **Scheduler:** SGE
- **Username:** `jamesdc1`
- **SSH target:** `jamesdc1@hoffman2.idre.ucla.edu`
- **Remote repo:** `$HPC_REPO` (default `/u/project/project-cucuringu/harxhar`)
- **ML SGE logs:** `$HPC_REPO/logs/`
- **DL SGE logs:** `/u/scratch/j/jamesdc1/` (`$SCRATCH`)
- **Chunk ID conversion:** SGE 1-based -> executor 0-based (handled by template)

### Rsync Exclude List
```
.git/ results/ __pycache__/ *.pyc .mypy_cache/ all30min/ .claude/
```

### SGE Commands
| Action | Command |
|--------|---------|
| Check queue | `qstat -u jamesdc1` |
| Job accounting | `qacct -j <JOBID>` |
| Clear error state | `qmod -cj <JOBID>` |
| Submit | `qsub -t <range> -N <name> -o logs -j y -v <vars> <template>` |

## ML Experiments

### Submission Modes
| Mode | Purpose | Key flags |
|------|---------|-----------|
| `model_comparison` | Compare models on HAR features | `--models ridge xgboost lightgbm random_forest` |
| `feature_transforms` | Compare feature types (HAR/PCA/AE) on one model | `--model ridge --features har pca ae --subgroup all_features` |
| `individual_features` | Ablate features one-by-one within a subgroup | `--model ridge --subgroup moments` |
| `subgroup_analysis` | Cartesian product: models x features x subgroups | `--models all --features all --subgroups all` |
| `naive` | Baseline only | (none) |
| `from-config` | Load from YAML/JSON config | `<config_file>` |

### Available Models
`ridge`, `xgboost`, `lightgbm`, `random_forest`

### Available Feature Types
`har` (rolling means), `pca` (compression), `ae` (autoencoder)

### Available Subgroups
`baseline` (empty), `moments`, `liquidity`, `market_ew`, `market_vw`, `sentiment`, `implied_vol`, `vol_demand`, `all_features` (51 vars)

### ML CLI Flags
| Flag | Default | Notes |
|------|---------|-------|
| `--total-chunks` | 100 | Number of array tasks per experiment |
| `--train-window` | 500 | Rolling window size in days |
| `--horizon` | 1 | Multi-horizon: runs h=1..H |
| `--n-components` | 5 | PCA/AE latent dimensions |
| `--ae-alpha` | 0.5 | AE loss: alpha x recon + (1-alpha) x pred |
| `--ae-epochs` | 50 | AE training epochs per refit |
| `--ae-hidden` | 0 (auto) | AE hidden width; 0 = n_features//2 |
| `--ae-weights-path` | None | Pre-trained AE weights .pt file |
| `--no-naive` | false | Skip naive baseline submission |
| `--backend` | slurm | `slurm`, `sge`, `sge-remote`, or `dry-run` |
| `--result-dir` | mode-dependent | Override output directory |

### ML Submit Commands
```bash
# Dry run (local preview)
python projects/ml/scripts/submit.py <mode> [flags] --backend dry-run

# Submit on cluster
cd $HPC_REPO && python projects/ml/scripts/submit.py <mode> [flags] --backend sge
```

### ML Example Commands
```bash
# Compare 4 models on HAR features
python projects/ml/scripts/submit.py model_comparison --models ridge xgboost lightgbm random_forest --backend sge

# PCA vs AE feature transforms
python projects/ml/scripts/submit.py feature_transforms --model ridge --features pca ae --subgroup moments --n-components 5 --backend sge

# Full grid
python projects/ml/scripts/submit.py subgroup_analysis --models all --features all --subgroups all --backend sge
```

## DL Experiments

### Experiment Types
| Experiment | Description |
|-----------|-------------|
| `patchts` | PatchTST transformer backtest |
| `ae_ridge` | Autoencoder + Ridge GPU backtest |

### DL Submit Commands
```bash
# PatchTST
cd $HPC_REPO && python -m projects.dl.cli.submit --experiment patchts --total-chunks 10 --backend sge

# AE+Ridge
cd $HPC_REPO && python -m projects.dl.cli.submit --experiment ae_ridge --total-chunks 10 --backend sge

# Via lifecycle manager (tracks job IDs)
cd $HPC_REPO && python -m projects.dl.cli.lifecycle submit --experiment <name> --total-chunks <n> --backend sge
```

### DL Status Command
```bash
cd $HPC_REPO && python -m projects.dl.cli.lifecycle status --result-dir <dir> --job-ids <ids> --total-chunks <n>
```

### Hoffman2 GPU Reference
| GPU | Cards/node | SGE flag | Notes |
|-----|-----------|----------|-------|
| H200 | 4 | `-l gpu,H200,cuda={1,4}` | Best performance |
| A100 | 4 | `-l gpu,A100,cuda={1,4}` | Default in template |
| H100 | 1 | `-l gpu,H100,cuda=1` | Single GPU only |
| A6000 | 2 | `-l gpu,A6000,cuda={1,2}` | |
| V100 | 1 | `-l gpu,V100,cuda=1` | Single GPU only |
| RTX2080Ti | 2 | `-l gpu,RTX2080Ti,cuda={1,2}` | No TF32, 11GB VRAM |
| ~~P4~~ | 1 | -- | **Excluded:** 4GB VRAM too small |

### DL Venv Setup (prerequisite)
```bash
module load conda cuda/12.3
mamba create -n harxhar-dl python=3.11 --no-default-packages -y
conda activate harxhar-dl
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install --only-binary :all: transformers "numpy<2" "pandas<3" scikit-learn pyarrow numba tqdm statsmodels lightgbm xgboost
```

### DL Log Path
SGE DL logs go to `$SCRATCH` (`/u/scratch/j/jamesdc1/`):
- Pattern: `<jobname>.o<JOBID>.<TASKID>` (e.g., `dl_patchts.o12345678.1`)

## Aggregation

### Aggregation Commands
```bash
# Auto-discover all dirs with .needs_aggregation marker
cd $HPC_REPO && python projects/ml/scripts/aggregate.py

# Explicit directory + eval mode
cd $HPC_REPO && python projects/ml/scripts/aggregate.py --base-dir results/<SET> --eval-mode <MODE>

# Comparison across experiment sets
cd $HPC_REPO && python projects/ml/scripts/compare.py results/<SET> --metric qlike --sort asc --top 10
```

### Eval Modes
| Mode | When to use | Output file |
|------|-------------|-------------|
| `global` (default) | Standard -- all trading hours combined | `global_results_summary.csv` |
| `segments` | Pre-segmented runs | `segment_results_summary.csv` |
| `filter_by_tod` | Slice global data by time-of-day | `global_results_tod_filtered.csv` |

Decision tree:
- Executor ran with `--segment all`? -> `segments`
- Want TOD breakdown from global runs? -> `filter_by_tod`
- Otherwise -> `global`

### DL Aggregation
```python
# Run on cluster for flat DL result dirs (results/dl_*)
import pandas as pd
from pathlib import Path
from core.evaluation.metrics import calculate_global_metrics

for name in ['dl_patchts_ctx480', 'dl_patchts_overlap', 'dl_ae_ridge']:
    d = Path('results') / name
    chunks = sorted(d.glob('results_chunk_*.csv'))
    if not chunks: continue
    df = pd.concat([pd.read_csv(f) for f in chunks], ignore_index=True)
    # ... calculate_global_metrics(df) per horizon
```

### Metrics
| Metric | Meaning | Better |
|--------|---------|--------|
| `mse` | Mean squared error (adjusted scale) | Lower |
| `mae` | Mean absolute error (adjusted scale) | Lower |
| `qlike` | QLIKE loss (raw scale, volatility-specific) | Lower |
| `oos_r2` | Out-of-sample R-squared vs naive baseline | Higher (>0 beats naive) |
| `delta_*` | Metric minus baseline metric | Negative = improvement |

Primary metric: **QLIKE**. Flag any model with `oos_r2 < 0` (underperforms naive).

### Rsync Pull (summaries only)
```bash
rsync -az \
    --include='*/' --include='*_summary*.csv' --include='metadata.json' \
    --include='config.txt' --include='lifecycle.jsonl' --exclude='*' \
    jamesdc1@hoffman2.idre.ucla.edu:$HPC_REPO/results/ ./results/
```

Or: `python -c "from core.remote import rsync_pull_results; rsync_pull_results()"`

## Key Paths (on cluster)

| What | Path |
|------|------|
| ML submit orchestrator | `projects/ml/scripts/submit.py` |
| ML submit utilities | `projects/ml/cli/submit.py` |
| ML executor | `projects/ml/cli/executor.py` |
| ML SGE template | `projects/ml/infra/sge/submit_hoffman2.sh` |
| DL SGE template | `projects/dl/infra/sge/submit_gpu.sh` |
| DL lifecycle manager | `projects/dl/cli/lifecycle.py` |
| DL GPU executor | `projects/dl/cli/gpu_executor.py` |
| Feature definitions | `core/features/feature_groups.py` |
| Aggregation script | `projects/ml/scripts/aggregate.py` |
| Aggregation logic | `core/evaluation/aggregation.py` |
| Metrics | `core/evaluation/metrics.py` |
| Comparison script | `projects/ml/scripts/compare.py` |
| Example configs | `projects/ml/experiments/` |
| SSH/rsync utilities | `core/remote.py` |
| HPC backends | `core/backends/` |
