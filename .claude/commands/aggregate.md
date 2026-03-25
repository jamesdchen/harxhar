Help me aggregate and analyze experiment results using the context below.

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
