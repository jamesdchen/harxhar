# HARXHAR Deep Learning — Cowork Instructions

## Project Overview

HARXHAR is a realized volatility forecasting system. It predicts intraday
30-minute volatility using HAR-family models with exogenous features on
rolling-window backtests.

## Cowork Desktop Setup

### Local Repo Access

Cowork on Desktop has **full access to this local repo**. This means:
- It can read and edit source files (e.g., fix bugs in `src/`)
- It can run `git commit` and `git push` to sync changes to GitHub
- Cell 1 on Colab runs `git pull --ff-only`, so pushed fixes are
  automatically picked up on the next Colab setup

**Important:** Never modify source code directly on Colab — those changes are
ephemeral and lost on runtime disconnect. Always fix code locally, push to
GitHub, then re-run Cell 1 on Colab to pull the fix.

### Notebook Persistence

Cell 2 parameter edits happen **on the Colab side via MCP** and are
ephemeral — they do not need to be saved or pushed back to the repo. The
canonical notebook is generated from `scripts/dl_runner_template.py`. If
the notebook structure needs to change, edit the template locally and run
`python scripts/dl_runner_template.py` to regenerate
`notebooks/dl_runner.ipynb`, then commit and push.

## Data Setup

No manual Google Drive data upload is needed. The training data lives in the
repo at `all30min/` (6 parquet files). Cell 1 clones the repo to
`/content/harxhar` and `chdir`s into it, so the default `DATA_PATH = "all30min"`
resolves to `/content/harxhar/all30min/` automatically.

Drive is still mounted for two things:
- **Status JSON** — written to `Drive/MyDrive/harxhar_status/dl_runner.json`
- **Results persistence** — Cell 5 copies CSVs to `Drive/MyDrive/harxhar_results/`

## MCP Tool Usage

You interact with the Colab notebook `dl_runner.ipynb` through the
`googlecolab/colab-mcp` MCP tools. Key operations:

- **Execute a cell:** Use the MCP tool to run a specific cell by index.
  Always read the cell output after execution to check for errors.
- **Edit a cell:** Use the MCP tool to modify cell contents (e.g., changing
  parameters in Cell 2 before each experiment run).
- **Read cell output:** Use the MCP tool to read the output of a previously
  executed cell to check results or errors.

**Rules:**
- Always read cell output after executing — never assume success.
- Execute cells one at a time, in order. Never use "Run All."
- The notebook should already be open in Colab at
  `Drive/MyDrive/harxhar_notebooks/dl_runner.ipynb`.

## Cell Reference

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

1. **Sync local repo to GitHub** — commit any pending changes and run
   `git push` so that Colab picks up the latest code via Cell 1's
   `git pull --ff-only`.
2. Open `dl_runner.ipynb` in Colab
3. Ensure the runtime is GPU-enabled (T4 for free tier, A100 if Pro+)
4. Execute Cell 1 (setup) — verify "Setup complete" in output
5. **Edit Cell 2** to set experiment parameters:
   - `EXPERIMENT`: `'patchts'` or `'ae_ridge'`
   - `HORIZON`: 1-48
   - `TRAIN_WINDOW`: default `None` (uses config default)
   - `GPU_COUNT`: 1 (Colab free) or match available GPUs
   - `TIMEOUT_HOURS`: 2.0 default
6. Execute Cell 3 (validate) — must print "Config OK"
7. Execute Cell 4 (run) — **returns immediately** (training runs in background)
8. Poll Cell 7 (status_check) to monitor progress
9. When Cell 7 shows `finished_run`, execute Cell 5 (collect)
10. Execute Cell 6 (eval) for metrics summary

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

1. Read the `error` and `traceback` fields from Cell 7 output.
2. Cross-reference against local source files:
   - PatchTSMixer/AE+Ridge issues → `src/models/deep_learning.py`, `src/backtest/gpu_engine.py`
   - Data issues → `src/data/pipeline.py`, `src/data/transforms.py`
   - GPU/CUDA errors → `src/backtest/gpu_kernels.py`

3. Known failure modes:
   - `RuntimeError: CUDA out of memory` → Reduce `BATCH_SIZE` in Cell 2, or reduce `TRAIN_WINDOW`
   - `ValueError: shape mismatch` → `context_length` or `patch_length` don't divide evenly; check `src/core/config.py`
   - `_WorkerError` → GPU worker crashed (usually OOM); check `gpu_engine.py`
   - `KeyError: column not found` → feature group mismatch; check `src/features/feature_groups.py`
   - `ConnectionError` or `Drive unmounted` → re-execute Cell 1 (setup)
   - `timeout` → exceeded `TIMEOUT_HOURS`; increase it or reduce `TRAIN_WINDOW`

4. Apply the fix:
   - Parameter issue → edit Cell 2 via MCP, re-run from Cell 3
   - Code issue → fix the source file locally, `git commit` and `git push`,
     then re-run Cell 1 on Colab (which does `git pull`) to pick up the fix
   - Colab resource issue → note it and suggest alternatives

5. If the same error persists after one retry, stop and report:
   - Full error message
   - Which source file is involved
   - Your diagnosis
   - Suggested fix

## Crash Recovery (Checkpoint Resume)

If a Colab runtime disconnects mid-run and `CHECKPOINT_DIR` was set:

1. Reconnect to Colab, re-run Cell 1 (setup).
2. Edit Cell 2 with the same parameters + `CHECKPOINT_DIR` pointing to the
   same Drive path (e.g., `"/content/drive/MyDrive/harxhar_checkpoints"`).
3. Re-run from Cell 3. The engine resumes from the last saved checkpoint.

If `CHECKPOINT_DIR` was not set, the run must restart from scratch.

**Recommendation:** For runs expected to take >1 hour, set
`CHECKPOINT_DIR = "/content/drive/MyDrive/harxhar_checkpoints"` in Cell 2.

## Model-Specific Notes

### PatchTSMixer
- `EXPERIMENT = "patchts"`
- HuggingFace transformer: context=241, patch=47, stride=31
- Defaults: epochs=150, lr=1e-4, batch_size=50, train_window=50000
- Memory-hungry — T4 (16GB) may struggle with large windows
- If OOM: try `BATCH_SIZE = 25` or `TRAIN_WINDOW = 25000`

### AE+Ridge
- `EXPERIMENT = "ae_ridge"`
- Hybrid: autoencoder with mixed loss (0.5× reconstruction + 0.5× prediction) + Ridge
- Defaults: epochs=50, lr=1e-3, batch_size=4, train_window=24000, n_components=5
- More memory-efficient than PatchTSMixer
- Uses `np.random.seed(42)` for reproducibility

## Scheduling Multiple Experiments

When running a sweep (e.g., "PatchTSMixer across horizons 1, 6, 12, 48"):

- Run sequentially in the same notebook, editing Cell 2 between runs.
- After each run, collect results (Cell 5) before starting the next.
- Keep a running log of which configs succeeded/failed.
- At the end, run Cell 6 for evaluation.
- Report a summary table to the user.

## Reporting Results

After completing experiments, report to the user with:

- A summary table: experiment × horizon → QLIKE, MSE, MAE
- GPU type used (T4, A100, etc.) and total runtime
- Any anomalies (OOM retries, timeouts, unexpected metric values)
- Comparison against prior results if available

## Autonomy Guidelines

**Do without asking:**
- Execute cells in the standard sequence
- Poll status during long runs
- Retry on OOM by reducing `BATCH_SIZE` (halve it) or `TRAIN_WINDOW`
- Collect and evaluate results
- Run sequential experiment sweeps as requested
- Fix clear bugs in local source code, commit, and push (Cell 1 pulls on Colab)

**Ask the user before:**
- Changing experiment type or model architecture
- Making non-trivial source code changes (refactors, architecture changes)
- Deciding whether to continue after 2+ consecutive failures
- Changing configs beyond simple OOM mitigation

## Rules

- Never use "Run All" — execute cells individually.
- Never skip Cell 3 (validate) — it catches config errors before long runs.
- Never modify source code on Colab — changes are ephemeral. Fix locally and `git push`.
- Never run overlapping experiments on the same Colab runtime.
- Always collect results (Cell 5) before starting a new experiment.
- Always read cell output after execution — never assume success.
- Don't re-run Cell 1 (setup) unless Drive unmounts or you need fresh deps.
- Don't modify cells other than Cell 2 — other cells are generated from
  `scripts/dl_runner_template.py`.
- Don't start a new experiment while one is still `running`.
- Don't assume the Colab runtime persists between sessions — always re-run
  Cell 1 on a fresh connection.
- Don't ignore GPU quota warnings — if "GPU quota exceeded", stop and report.
- Don't run Cell 5 (collect) before Cell 7 confirms `finished_run`.

## File Locations

- Notebook source: `notebooks/dl_runner.ipynb` (in repo)
- Notebook template: `scripts/dl_runner_template.py` (canonical source)
- Status: `Drive/MyDrive/harxhar_status/dl_runner.json`
- Results: `Drive/MyDrive/harxhar_results/<experiment>/*.csv`
- Data: `all30min/*.parquet` (in repo, cloned to Colab automatically)
- Repo (on Colab): `/content/harxhar/`
- Repo (local): this project folder
