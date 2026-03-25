Help me aggregate and analyze experiment results using the context below.

## Cluster Context

- Discovery cluster (USC CARC), username: `jc_905`
- SLURM logs (stdout/stderr): grep in `/scratch1/jc_905/`

## Key Paths

| What | Path |
|---|---|
| Results root | `results/` |
| Aggregation script | `projects/ml/scripts/aggregate.py` |
| Comparison script | `projects/ml/scripts/compare.py` |

## Output Structure

```
results/<experiment_set>/
  exp_{ID}_{MODEL}_{FEATURES}_{NAME}/
    config.txt                          # experiment metadata
    results_chunk_1.csv ... _100.csv    # per-chunk backtest results
    results_chunk_1_h{H}.csv           # (multi-horizon only)
  .needs_aggregation                    # marker: triggers aggregation
  global_results_summary.csv            # (after aggregation)
  segment_results_summary.csv           # (if segmented eval)
```

## Aggregating Results

```bash
# Auto-discover all dirs with .needs_aggregation marker
python scripts/aggregate.py

# Explicit directory + eval mode
python scripts/aggregate.py --base-dir results/model_comparison --eval-mode global
```

**Eval modes:**
- `global` (default) — all data across all hours
- `segments` — process pre-segmented files separately (requires `_morning`, `_midday`, `_closing`, `_overnight` suffixes)
- `filter_by_tod` — load global data, filter by time-of-day in memory

**Output:** `global_results_summary.csv` or `segment_results_summary.csv` in the base dir.
The `.needs_aggregation` marker is removed on success.

## Comparing Results

```bash
# Compare result directories
python scripts/compare.py results/model_comparison results/subgroup_analysis

# Compare specific CSV files
python scripts/compare.py results/*/global_results_summary.csv

# Sort and filter
python scripts/compare.py results/model_comparison --metric qlike --sort asc --top 10
```

**Flags:**
- `--metric` — sort by: mse, mae, qlike, oos_r2, delta_mse, delta_mae, delta_qlike (default: qlike)
- `--sort` — asc or desc (default: asc)
- `--top` — show only top N results

## Debugging

### Pre-Aggregation Checks
Before aggregating, verify all chunks completed:
```bash
# Count result CSVs in an experiment dir
ls results/<experiment_set>/exp_*/results_chunk_*.csv | wc -l
```
Missing chunk IDs indicate failed SLURM array tasks that need resubmission.

### Finding SLURM Logs
Logs land in `/scratch1/jc_905/`. To diagnose failed chunks:
```bash
find /scratch1/jc_905/ -name "slurm-<JOBID>*" -type f
grep -r "Error\|OOM\|CANCELLED\|TIMEOUT" /scratch1/jc_905/slurm-<JOBID>*
```

### Common Failure Modes

**OOM kills** — Job exceeded memory limit.
- Symptom: `sacct -j <JOBID>` shows State=OUT_OF_MEMORY or ExitCode=0:125
- Fix: Increase `--mem` in the SLURM template, then resubmit failed tasks

**Timeouts** — Job exceeded walltime.
- Symptom: State=TIMEOUT in sacct
- Fix: Increase `--time` or reduce `--total-chunks`

**Module/env issues** — Python or conda not found.
- Symptom: `ModuleNotFoundError` in logs
- Fix: Check module loads in SLURM template. For GPU: conda source path is `/apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh`, env is `project-cucuringu`

### Resubmitting Failed Tasks
```bash
# Resubmit only failed array indices (1-based):
sbatch --array=5,23,78 projects/ml/infra/slurm/submit_carc.slurm
```
Ensure the same `--export` variables are passed. Then re-run aggregation.
