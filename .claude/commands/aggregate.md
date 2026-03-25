Help me aggregate, validate, and analyze experiment results using the workflow below.

## Step 1: Check Job Status

Before aggregating, confirm all SLURM jobs have finished:

```bash
squeue -u jc_905  # any still running/pending?
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
2. Check `sacct` for failure reason:
   ```bash
   sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,Elapsed
   ```
3. Grep SLURM logs for the error:
   ```bash
   grep -l "Error\|OOM\|CANCELLED\|TIMEOUT" /scratch1/jc_905/slurm-<JOBID>_*.out
   ```
4. **Diagnose** — common causes:
   - `OUT_OF_MEMORY` / ExitCode 0:125 → increase `--mem` in SLURM template
   - `TIMEOUT` → increase `--time` or reduce `--total-chunks`
   - `ModuleNotFoundError` → check `module load python` in template
5. **Resubmit** only the failed task IDs (pass same env vars as original):
   ```bash
   sbatch --array=<FAILED_IDS> projects/ml/infra/slurm/submit_carc.slurm
   ```
6. Wait for resubmitted jobs, then re-validate before aggregating.

**Only proceed to Step 3 when all expected chunks are present.**

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
| Results root | `results/` |
| SLURM logs | `/scratch1/jc_905/` |
