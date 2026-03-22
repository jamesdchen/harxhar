# HARXHAR Deep Learning — Cowork Instructions

## Overview

You manage PatchTSMixer and AE+Ridge experiments on Google Colab via the
`googlecolab/colab-mcp` connector. The notebook `dl_runner.ipynb` is your
interface to Colab. You execute cells individually and monitor results via
a Drive-persisted status JSON.

## Notebook Cell Map

The notebook has 8 cells (0-7):

| Cell | Tag            | Purpose                                    | When to run         |
|------|----------------|--------------------------------------------|---------------------|
| 0    | (markdown)     | Header — skip                              | Never               |
| 1    | setup          | Install deps, mount Drive, clone repo      | Once per runtime    |
| 2    | parameters     | Experiment config — **you edit this**       | Before each run     |
| 3    | validate       | Check config + GPU availability            | After editing params|
| 4    | run            | Execute the DL experiment                  | Once per run        |
| 5    | collect        | Copy results from Colab to Drive           | After run succeeds  |
| 6    | eval           | Quick QLIKE/MSE/MAE on collected results   | After collect       |
| 7    | status_check   | Read status file + GPU utilization          | Anytime (polling)   |

## Status JSON

The coordination mechanism is a JSON file at
`Drive/MyDrive/harxhar_status/dl_runner.json`. Each cell writes status
transitions. You read it via Cell 7.

Status values:
- `validated` — config checked, ready to run
- `running` — experiment in progress
- `finished_run` — run complete, proceed to collect
- `failed` — error occurred, read `error` and `traceback` fields
- `collected` — results saved to Drive
- `evaluated` — metrics computed, experiment done

## Launch Sequence

When asked to run a DL experiment:

1. Open `dl_runner.ipynb` in Colab (should already be in Drive at
   `Drive/MyDrive/harxhar_notebooks/`)
2. Ensure the runtime is GPU-enabled (T4 for free tier, A100 if Pro+)
3. Execute Cell 1 (setup) — verify "Setup complete" in output
4. **Edit Cell 2** to set experiment parameters:
   - `EXPERIMENT`: `'patchts'` or `'ae_ridge'`
   - `HORIZON`: 1-48
   - `TRAIN_WINDOW`: default `None` (uses config default)
   - `GPU_COUNT`: 1 (Colab free) or match available GPUs
   - `TIMEOUT_HOURS`: 2.0 default
5. Execute Cell 3 (validate) — must print "Config OK"
6. Execute Cell 4 (run) — this is the long-running cell
7. Execute Cell 5 (collect) after run completes
8. Execute Cell 6 (eval) for metrics summary

## Babysitting Protocol

While Cell 4 (run) is executing:

- Execute Cell 7 (status_check) periodically to read the status JSON
- Status values and what to do:
  - `validated` — run hasn't started yet, execute Cell 4
  - `running` — experiment in progress, check back later
  - `finished_run` — run complete, proceed to Cell 5 (collect)
  - `failed` — read the error message, diagnose, fix, re-run
  - `collected` — results on Drive, proceed to Cell 6 (eval)
  - `evaluated` — everything complete, report metrics to user

## Failure Diagnosis

When status is `failed`:

1. Read the `error` and `traceback` fields from Cell 7 output
2. Cross-reference against local source files:
   - PatchTSMixer issues → check `src/models/deep_learning.py`, `src/backtest/gpu_engine.py`
   - AE+Ridge issues → check `src/models/deep_learning.py`, `src/backtest/gpu_engine.py`
   - Data issues → check `src/data/pipeline.py`, `src/data/transforms.py`
   - GPU/CUDA errors → check `src/backtest/gpu_kernels.py`

3. Known failure modes:
   - `RuntimeError: CUDA out of memory` → Reduce `BATCH_SIZE` in Cell 2, or reduce `TRAIN_WINDOW`
   - `ValueError: shape mismatch` → `context_length` or `patch_length` don't divide evenly, check PatchTSMixer config in `src/core/config.py`
   - `_WorkerError` → GPU worker crashed, usually OOM — check `gpu_engine.py`
   - `KeyError: column not found` → feature group mismatch, check `src/features/feature_groups.py`
   - `ConnectionError` or `Drive unmounted` → re-execute Cell 1 (setup)
   - `timeout` → exceeded `TIMEOUT_HOURS`, increase it or reduce `TRAIN_WINDOW`

4. Apply the fix:
   - Parameter issue → edit Cell 2 and re-run from Cell 3
   - Code issue → report to user what to fix in the repo
   - Colab resource issue → note it and suggest alternatives

5. If the same error persists after one retry, stop and report with:
   - Full error message
   - Which source file is involved
   - Your diagnosis
   - Suggested fix

## Scheduling Multiple Experiments

When asked to run a sweep (e.g., "run PatchTSMixer across horizons 1, 6, 12, 48"):

- Run them sequentially in the same notebook, editing Cell 2 between runs
- After each run, collect results (Cell 5) before starting the next
- Keep a running log of which configs succeeded/failed
- At the end, run Cell 6 for evaluation
- Report a summary table: experiment x horizon → QLIKE, MSE, MAE

## Model-Specific Notes

### PatchTSMixer
- `EXPERIMENT = "patchts"`
- HuggingFace transformer: context=241, patch=47, stride=31
- Memory-hungry — T4 (16GB) may struggle with large windows
- If OOM: try `BATCH_SIZE = 25` or `TRAIN_WINDOW = 25000`
- Config defaults in `src/core/config.py` → `DL_CONFIG`

### AE+Ridge
- `EXPERIMENT = "ae_ridge"`
- Hybrid: autoencoder with mixed loss (0.5x reconstruction + 0.5x prediction)
- More memory-efficient than PatchTSMixer
- Sets `np.random.seed(42)` for reproducibility
- Config defaults in `src/core/config.py` → `AE_RIDGE_GPU_CONFIG`

## File Locations

- Notebook: `Drive/MyDrive/harxhar_notebooks/dl_runner.ipynb`
- Status: `Drive/MyDrive/harxhar_status/dl_runner.json`
- Results: `Drive/MyDrive/harxhar_results/<experiment>/*.csv`
- Data: `Drive/MyDrive/harxhar_data/all30min/*.parquet`
- Repo (on Colab): `/content/harxhar/`
- Repo (local): this project folder

## What NOT To Do

- Don't "Run All" — execute cells individually so you can monitor each step
- Don't skip validation (Cell 3) — catches config errors before a long run
- Don't re-run Cell 1 (setup) unless Drive unmounts or you need fresh deps
- Don't ignore GPU quota warnings — if "GPU quota exceeded", stop and report
