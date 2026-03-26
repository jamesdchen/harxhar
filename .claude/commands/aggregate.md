Help me aggregate, validate, and analyze experiment results using the workflow below.

## Step 1: Check Job Status

Before aggregating, confirm all SLURM jobs have finished. **All SLURM commands need `--clusters=discovery`**:

```bash
squeue -u jc_905 --clusters=discovery  # any still running/pending?
```

If jobs are still running, report which ones and wait. Do NOT aggregate partial results unless I explicitly ask.

## Step 2: Validate Chunk Completeness

For each experiment set that needs aggregation, verify all chunks landed:

```bash
# Find pending result dirs
ls -d results/*/.needs_aggregation results_*/.needs_aggregation 2>/dev/null

# For each experiment dir, count chunks vs expected (default 100)
for d in results/<SET>/exp_*/; do
    n=$(ls "$d"/results_chunk_*.csv 2>/dev/null | wc -l)
    echo "$d: $n chunks"
done
```

**If chunks are missing:**
1. Identify the missing 1-based SLURM task IDs
2. Check `sacct` for failure reason (must use `--clusters=discovery`):
   ```bash
   sacct -j <JOBID> --clusters=discovery --format=JobID,State,ExitCode,MaxRSS,Elapsed
   ```
3. Grep SLURM logs for the error (logs are in `/scratch1/jc_905/logs/`):
   ```bash
   grep -l "Error\|OOM\|CANCELLED\|TIMEOUT" /scratch1/jc_905/logs/slurm-<JOBID>_*.err
   ```
4. **Diagnose** — common causes:
   - `OUT_OF_MEMORY` / ExitCode 0:125 → resubmit with `--mem=96G`
   - `TIMEOUT` → resubmit with `--time=4:00:00` (or higher)
   - `torch.OutOfMemoryError: CUDA out of memory` → reduce batch_size in `projects/dl/config.py`
   - `ModuleNotFoundError` → check `module load python` in template
5. **Reconstruct env vars** from the experiment's `metadata.json` / `config.txt`:
   ```bash
   cat results/<SET>/exp_<N>_<name>/metadata.json
   ```
6. **Resubmit** only the failed task IDs with full context:
   ```bash
   sbatch --clusters=discovery --array=<FAILED_IDS> \
     --account=pollok_1603 \
     --output=/scratch1/jc_905/logs/slurm-%A_%a.out \
     --error=/scratch1/jc_905/logs/slurm-%A_%a.err \
     --export=RESULT_DIR=results/<SET>/exp_<N>_<name>,TOTAL_CHUNKS=100,MODEL_TYPE=<model>,EXOG_COLS="<vars or NONE>",EXTRA_ARGS="<from config>" \
     projects/ml/infra/slurm/submit_carc.slurm
   # Add --mem=96G or --time=4:00:00 as needed for the failure type
   ```
7. Wait for resubmitted jobs, then re-validate before aggregating.

**Partial aggregation:** By default, only proceed to Step 3 when all expected chunks are present. If the user explicitly asks to aggregate partial results (e.g., 97/100 chunks), proceed but note the missing chunk count and percentage in the output.

## Step 3: Aggregate

```bash
# Auto-discover all dirs with .needs_aggregation marker
python scripts/aggregate.py

# OR explicit directory + eval mode
python scripts/aggregate.py --base-dir results/<SET> --eval-mode <MODE>
```

### Eval Modes

| Mode | When to use | Output file |
|------|-------------|-------------|
| `global` (default) | Standard — all trading hours combined | `global_results_summary.csv` |
| `segments` | Pre-segmented runs (chunks have `_morning`, `_midday`, etc. suffixes) | `segment_results_summary.csv` |
| `filter_by_tod` | Slice global data by time-of-day in memory (no re-read) | `global_results_tod_filtered.csv` |

**Decision tree:**
- Did the executor run with `--segment all`? → use `segments`
- Want TOD breakdown from global runs? → use `filter_by_tod`
- Otherwise → use `global`

### TOD Time Boundaries (hardcoded)

| Segment | Start | End |
|---------|-------|-----|
| morning | 09:30 | 11:30 |
| midday | 11:30 | 14:00 |
| closing | 14:00 | 16:00 |
| overnight | 16:00 | 09:30 |

## Step 4: Interpret Results

After aggregation, read and interpret the summary CSV. Key metrics:

| Metric | Meaning | Better |
|--------|---------|--------|
| `mse` | Mean squared error (adjusted scale) | Lower |
| `mae` | Mean absolute error (adjusted scale) | Lower |
| `qlike` | QLIKE loss (raw scale, volatility-specific) | Lower |
| `oos_r2` | Out-of-sample R² vs naive baseline | Higher (>0 beats naive) |
| `delta_*` | Metric minus baseline metric | Negative = improvement |

**When reporting results:**
- Lead with the best-performing model/config by QLIKE (primary metric)
- Flag any model that underperforms naive (oos_r2 < 0)
- Note sample counts — low n_samples may indicate data issues
- If multi-horizon, highlight which horizons benefit most

## Step 4b: Aggregate DL Results

DL results (e.g., `results/dl_patchts_*`, `results/dl_ae_ridge/`) use a flat directory structure with `results_chunk_*.csv` files — no `exp_*` subdirs or `config.txt`. The ML `aggregate.py` script does not discover these automatically.

To aggregate DL results, concat the chunks and compute metrics directly:

```python
python3 -c "
import pandas as pd
from pathlib import Path
from core.evaluation.metrics import calculate_global_metrics

for name in ['dl_patchts_ctx480', 'dl_patchts_overlap', 'dl_ae_ridge']:
    d = Path('results') / name
    chunks = sorted(d.glob('results_chunk_*.csv'))
    if not chunks:
        print(f'{name}: no chunks found'); continue
    df = pd.concat([pd.read_csv(f) for f in chunks], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    print(f'\n=== {name} ({len(chunks)} chunks) ===')
    print(f'Samples: {len(df)} | Range: {df.index.min()} to {df.index.max()}')
    if 'horizon' in df.columns:
        for h in sorted(df['horizon'].unique()):
            m = calculate_global_metrics(df[df['horizon'] == h])
            print(f'  h={int(h)}: MSE={m[\"mse\"]:.4e}  MAE={m[\"mae\"]:.4e}  QLIKE={m[\"qlike\"]:.6f}  n={int(m[\"n_samples\"])}')
    else:
        m = calculate_global_metrics(df)
        print(f'  MSE={m[\"mse\"]:.4e}  MAE={m[\"mae\"]:.4e}  QLIKE={m[\"qlike\"]:.6f}  n={int(m[\"n_samples\"])}')
"
```

When reporting DL results, compare them against the ML baselines (especially naive and ridge) from existing `global_results_summary.csv` files. Flag any DL model that underperforms naive on QLIKE.

## Step 5: Compare Across Experiments

```bash
# Compare result directories (auto-finds summary CSVs)
python scripts/compare.py results/model_comparison results/subgroup_analysis

# Sort by specific metric
python scripts/compare.py results/<SET> --metric qlike --sort asc --top 10

# Compare specific files
python scripts/compare.py results/*/global_results_summary.csv
```

**Flags:**
- `--metric` — sort by: mse, mae, qlike, oos_r2, delta_mse, delta_mae, delta_qlike (default: qlike)
- `--sort` — asc (best-first for loss) or desc (best-first for R²)
- `--top N` — show only top N results

## Key Paths

| What | Path |
|------|------|
| Aggregation script | `projects/ml/scripts/aggregate.py` |
| Aggregation logic | `projects/ml/evaluation/aggregation.py` |
| Metrics & deltas | `core/evaluation/metrics.py` |
| Comparison script | `projects/ml/scripts/compare.py` |
| Results root (ML) | `results/` (uses `exp_*` subdirs with `config.txt`) |
| Results root (DL) | `results/dl_*` (flat dirs, no `exp_*` structure) |
| SLURM logs | `/scratch1/jc_905/logs/` |
