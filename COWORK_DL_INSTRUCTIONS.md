# HARXHAR Deep Learning — Cowork Instructions

## Overview

You manage PatchTSMixer and AE+Ridge experiments on Google Colab via the
`googlecolab/colab-mcp` connector. The notebook `dl_runner.ipynb` is your
interface to Colab. You execute cells individually and monitor results via
a Drive-persisted status JSON.

## Setup

No manual Google Drive data upload is needed. The training data lives in the
repo at `all30min/` (6 parquet files). Cell 1 clones the repo to
`/content/harxhar` and `chdir`s into it, so the default `DATA_PATH = "all30min"`
resolves to `/content/harxhar/all30min/` automatically.

Drive is still mounted for two things:
- **Status JSON** — written to `Drive/MyDrive/harxhar_status/dl_runner.json`
- **Results persistence** — Cell 5 copies CSVs to `Drive/MyDrive/harxhar_results/`

## Notebook Cell Map

The notebook has 8 cells (0-7):

| Cell | Tag            | Purpose                                    | When to run         |
|------|----------------|--------------------------------------------|---------------------|
| 0    | (markdown)     | Header — skip                              | Never               |
| 1    | setup          | Install deps, mount Drive, clone repo      | Once per runtime    |
| 2    | parameters     | Experiment config — **you edit this**       | Before each run     |
| 3    | validate       | Check config + GPU availability            | After editing params|
| 4    | run            | Launch experiment (background process)     | Once per run        |
| 5    | collect        | Copy results from Colab to Drive           | After run finishes  |
| 6    | eval           | Quick QLIKE/MSE/MAE on collected results   | After collect       |
| 7    | status_check   | Process liveness + status + GPU + log tail | Anytime (polling)   |

## Status JSON

The coordination mechanism is a JSON file at
`Drive/MyDrive/harxhar_status/dl_runner.json`. The background training process
(`gpu_executor.py --write-status`) writes status transitions. You read it
via Cell 7.

Status values:
- `validated` — config checked, ready to run
- `running` — experiment in progress (includes `pid`)
- `finished_run` — run complete, proceed to collect
- `failed` — error occurred, read `error` and `traceback` fields
- `collected` — results saved to Drive
- `evaluated` — metrics computed, experiment done

## Launch Sequence

When asked to run a DL experiment:

1. Open `dl_runner.ipynb` in Colab
2. Ensure the runtime is GPU-enabled (T4 for free tier, A100 if Pro+)
3. Execute Cell 1 (setup) — verify "Setup complete" in output
4. **Edit Cell 2** to set experiment parameters:
   - `EXPERIMENT`: `'patchts'` or `'ae_ridge'`
   - `HORIZON`: 1-48
   - `TRAIN_WINDOW`: default `None` (uses config default)
   - `GPU_COUNT`: 1 (Colab free) or match available GPUs
   - `TIMEOUT_HOURS`: 2.0 default
5. Execute Cell 3 (validate) — must print "Config OK"
6. Execute Cell 4 (run) — **returns immediately** (training runs in background)
7. Poll Cell 7 (status_check) to monitor progress
8. When Cell 7 shows `finished_run`, execute Cell 5 (collect)
9. Execute Cell 6 (eval) for metrics summary

## Babysitting Protocol

Cell 4 launches training as a background process and returns immediately.
The kernel stays free for polling.

- Execute Cell 7 (status_check) periodically — it shows:
  - **Process liveness**: whether the PID is still running or finished
  - **Status JSON**: current status + metadata from the training process
  - **GPU utilization**: memory, compute %, temperature
  - **Log tail**: last 30 lines of training output

- What to do based on Cell 7 output:
  - Process RUNNING + status `running` → training in progress, check back later
  - Process FINISHED + status `finished_run` → proceed to Cell 5 (collect)
  - Process FINISHED + status `failed` → read error, diagnose, fix, re-run
  - Process FINISHED + no status update → check log tail for crash details
  - `collected` → results on Drive, proceed to Cell 6 (eval)
  - `evaluated` → everything complete, report metrics to user

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

- Notebook source: `notebooks/dl_runner.ipynb` (in repo)
- Notebook template: `scripts/dl_runner_template.py` (canonical source)
- Status: `Drive/MyDrive/harxhar_status/dl_runner.json`
- Results: `Drive/MyDrive/harxhar_results/<experiment>/*.csv`
- Data: `all30min/*.parquet` (in repo, cloned to Colab automatically)
- Repo (on Colab): `/content/harxhar/`
- Repo (local): this project folder

## What NOT To Do

- Don't "Run All" — execute cells individually so you can monitor each step
- Don't skip validation (Cell 3) — catches config errors before a long run
- Don't re-run Cell 1 (setup) unless Drive unmounts or you need fresh deps
- Don't ignore GPU quota warnings — if "GPU quota exceeded", stop and report
- Don't run Cell 5 (collect) before Cell 7 confirms `finished_run`
