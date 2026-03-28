Help me aggregate, validate, and analyze experiment results using the workflow below.

All cluster operations run remotely via `ssh jamesdc1@hoffman2.idre.ucla.edu`.
Aggregation runs on the cluster (avoids transferring hundreds of chunk CSVs).
Only summary files are downloaded locally for interpretation.

## Step 1: Check Job Status

Before aggregating, confirm all SGE jobs have finished:

```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "qstat -u jamesdc1"
```

If jobs are still running, report which ones and wait. Do NOT aggregate partial results unless I explicitly ask.

## Step 2: Validate Chunk Completeness

For each experiment set that needs aggregation, verify all chunks landed:

```bash
# Find pending result dirs
ssh jamesdc1@hoffman2.idre.ucla.edu "ls -d $HPC_REPO/results/*/.needs_aggregation $HPC_REPO/results_*/.needs_aggregation 2>/dev/null"

# For each experiment dir, count chunks vs expected (default 100)
ssh jamesdc1@hoffman2.idre.ucla.edu "for d in $HPC_REPO/results/<SET>/exp_*/; do n=\$(ls \"\$d\"/results_chunk_*.csv 2>/dev/null | wc -l); echo \"\$d: \$n chunks\"; done"
```

**If chunks are missing:**
1. Identify the missing 1-based SGE task IDs
2. Check `qacct` for failure reason:
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "qacct -j <JOBID>"
   ```
3. Check SGE logs for the error:
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "grep -l 'Error\|OOM\|killed' $HPC_REPO/logs/<job_name>.o<JOBID>.*"
   ```
4. **Diagnose** — common causes:
   - Memory exceeded -> resubmit with `-l h_vmem=96G`
   - Walltime exceeded -> resubmit with `-l h_rt=14400`
   - `torch.OutOfMemoryError` -> reduce batch_size
   - `ModuleNotFoundError` -> check `module load` in template
5. **Reconstruct env vars** from the experiment's `metadata.json` / `config.txt`:
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "cat $HPC_REPO/results/<SET>/exp_<N>_<name>/metadata.json"
   ```
6. **Resubmit** only the failed task IDs:
   ```bash
   ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && qsub -t <FAILED_IDS> \
       -N <job_name> -o logs -j y \
       -v RESULT_DIR=results/<SET>/exp_<N>_<name>,TOTAL_CHUNKS=100,MODEL_TYPE=<model>,EXOG_COLS='<vars or NONE>',EXTRA_ARGS='<from config>' \
       projects/ml/infra/sge/submit_hoffman2.sh"
   ```
7. Wait for resubmitted jobs, then re-validate before aggregating.

**Partial aggregation:** By default, only proceed to Step 3 when all expected chunks are present. If the user explicitly asks to aggregate partial results (e.g., 97/100 chunks), proceed but note the missing chunk count and percentage in the output.

## Step 3: Aggregate (on cluster)

Run aggregation remotely to avoid downloading chunk CSVs:

```bash
# Auto-discover all dirs with .needs_aggregation marker
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python projects/ml/scripts/aggregate.py"

# OR explicit directory + eval mode
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python projects/ml/scripts/aggregate.py --base-dir results/<SET> --eval-mode <MODE>"
```

### Eval Modes

| Mode | When to use | Output file |
|------|-------------|-------------|
| `global` (default) | Standard — all trading hours combined | `global_results_summary.csv` |
| `segments` | Pre-segmented runs | `segment_results_summary.csv` |
| `filter_by_tod` | Slice global data by time-of-day | `global_results_tod_filtered.csv` |

**Decision tree:**
- Did the executor run with `--segment all`? -> use `segments`
- Want TOD breakdown from global runs? -> use `filter_by_tod`
- Otherwise -> use `global`

## Step 4: Download Summaries

After aggregation completes on the cluster, pull only the summary files locally:

```bash
rsync -az \
    --include='*/' \
    --include='*_summary*.csv' \
    --include='metadata.json' \
    --include='config.txt' \
    --include='lifecycle.jsonl' \
    --exclude='*' \
    jamesdc1@hoffman2.idre.ucla.edu:$HPC_REPO/results/ ./results/
```

Or via Python:
```bash
python -c "from core.remote import rsync_pull_results; r = rsync_pull_results(); print(r.stdout or 'done'); print(r.stderr)"
```

## Step 5: Interpret Results

After downloading, read and interpret the local summary CSVs. Key metrics:

| Metric | Meaning | Better |
|--------|---------|--------|
| `mse` | Mean squared error (adjusted scale) | Lower |
| `mae` | Mean absolute error (adjusted scale) | Lower |
| `qlike` | QLIKE loss (raw scale, volatility-specific) | Lower |
| `oos_r2` | Out-of-sample R-squared vs naive baseline | Higher (>0 beats naive) |
| `delta_*` | Metric minus baseline metric | Negative = improvement |

**When reporting results:**
- Lead with the best-performing model/config by QLIKE (primary metric)
- Flag any model that underperforms naive (oos_r2 < 0)
- Note sample counts — low n_samples may indicate data issues
- If multi-horizon, highlight which horizons benefit most

## Step 5b: Aggregate DL Results

DL results use a flat directory structure. Run aggregation on cluster:

```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python3 -c \"
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
    print(f'=== {name} ({len(chunks)} chunks) ===')
    print(f'Samples: {len(df)} | Range: {df.index.min()} to {df.index.max()}')
    if 'horizon' in df.columns:
        for h in sorted(df['horizon'].unique()):
            m = calculate_global_metrics(df[df['horizon'] == h])
            print(f'  h={int(h)}: MSE={m[\\\"mse\\\"]:.4e}  MAE={m[\\\"mae\\\"]:.4e}  QLIKE={m[\\\"qlike\\\"]:.6f}  n={int(m[\\\"n_samples\\\"])}')
    else:
        m = calculate_global_metrics(df)
        print(f'  MSE={m[\\\"mse\\\"]:.4e}  MAE={m[\\\"mae\\\"]:.4e}  QLIKE={m[\\\"qlike\\\"]:.6f}  n={int(m[\\\"n_samples\\\"])}')
\""
```

## Step 6: Compare Across Experiments

Run comparison on cluster, then read output:

```bash
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python projects/ml/scripts/compare.py results/model_comparison results/subgroup_analysis"

# Sort by specific metric
ssh jamesdc1@hoffman2.idre.ucla.edu "cd $HPC_REPO && python projects/ml/scripts/compare.py results/<SET> --metric qlike --sort asc --top 10"
```

**Flags:**
- `--metric` — sort by: mse, mae, qlike, oos_r2, delta_mse, delta_mae, delta_qlike (default: qlike)
- `--sort` — asc (best-first for loss) or desc (best-first for R-squared)
- `--top N` — show only top N results

## Key Paths (on cluster)

| What | Path |
|------|------|
| Aggregation script | `projects/ml/scripts/aggregate.py` |
| Aggregation logic | `projects/ml/evaluation/aggregation.py` |
| Metrics & deltas | `core/evaluation/metrics.py` |
| Comparison script | `projects/ml/scripts/compare.py` |
| Results root (ML) | `results/` (uses `exp_*` subdirs with `config.txt`) |
| Results root (DL) | `results/dl_*` (flat dirs, no `exp_*` structure) |
| SGE logs | `$HPC_REPO/logs/` |
