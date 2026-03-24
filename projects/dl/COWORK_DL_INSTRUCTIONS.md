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

**Cell indexing convention:** Cell indices are 0-based throughout this
document and match the MCP `execute_cell(index=N)` parameter. Cell 0 is the
markdown header, Cell 1 is setup, etc.

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

## Cell 2 Parameter Reference

All experiment parameters are set in Cell 2. This is the single canonical
list — do not look elsewhere for parameter semantics.

| Parameter         | Type    | Default         | Valid range / values        | Applies to    | Notes                                                    |
|-------------------|---------|-----------------|-----------------------------|---------------|----------------------------------------------------------|
| `EXPERIMENT`      | str     | —               | `"patchts"`, `"ae_ridge"`   | All           | Required. Selects model type.                            |
| `HORIZON`         | int     | —               | 1–48                        | All           | Required. Forecast horizon in 30-min slots.              |
| `TRAIN_WINDOW`    | int     | `None`          | Any positive int or `None`  | All           | `None` uses config default (50000 patchts, 24000 ae_ridge). Reduce to save memory. |
| `BATCH_SIZE`      | int     | Model default   | Any positive int            | All           | Default: 50 (patchts), 4 (ae_ridge). Halve on OOM.      |
| `GPU_COUNT`       | int     | 1               | 1–N (match available GPUs)  | All           | 1 for Colab free tier.                                   |
| `TIMEOUT_HOURS`   | float   | 2.0             | > 0                         | All           | Increase for large windows / slow GPUs.                  |
| `CHECKPOINT_DIR`  | str     | `None`          | Drive path or `None`        | All           | Set for runs > 1 hr. Enables crash recovery resume.      |
| `EPOCHS`          | int     | Model default   | Any positive int            | All           | Default: 150 (patchts), 50 (ae_ridge).                   |
| `LR`              | float   | Model default   | > 0                         | All           | Default: 1e-4 (patchts), 1e-3 (ae_ridge).               |
| `N_COMPONENTS`    | int     | 5               | ≥ 1                         | ae_ridge only | Autoencoder latent dimension.                            |

## Status JSON & State Transitions

The coordination mechanism is a JSON file at
`Drive/MyDrive/harxhar_status/dl_runner.json`. The background training process
(`gpu_executor.py --write-status`) writes status transitions. You read it
via Cell 7.

### Valid transitions

```
validated ──→ running ──→ finished_run ──→ collected ──→ evaluated
                │
                └──→ failed
```

- **`validated`** — Cell 3 passed; config checked, ready to run.
  Re-running Cell 3 resets status back to `validated`.
- **`running`** — Cell 4 launched training; `pid` field is set.
- **`finished_run`** — Run complete. Proceed to Cell 5 (collect).
- **`failed`** — Error occurred. Read `error` and `traceback` fields.
  After fixing, re-run from Cell 3 to reset to `validated`.
- **`collected`** — Cell 5 saved results to Drive.
- **`evaluated`** — Cell 6 computed metrics. Experiment done.

## Launch Sequence

When asked to run a DL experiment:

1. **Sync local repo to GitHub** — commit any pending changes and run
   `git push` so that Colab picks up the latest code via Cell 1's
   `git pull --ff-only`.
2. **Handle MCP authorization** *(browser automation only)* — after the
   `colab-mcp` server connects, take a screenshot to check for a
   permission/authorization dialog in the browser. If one appears, click
   "Allow" / "Authorize" to grant access. Repeat the screenshot→click
   cycle until the dialog is dismissed and the Colab notebook is visible.
   Skip this step if not using Claude-in-Chrome.
3. Open `dl_runner.ipynb` in Colab.
4. Ensure the runtime is GPU-enabled (T4 for free tier, A100 if Pro+).
5. Execute Cell 1 (setup) — verify "Setup complete" in output.
6. **Edit Cell 2** with experiment parameters (see §Cell 2 Parameter
   Reference for the full schema).
7. Execute Cell 3 (validate) — must print "Config OK".
8. Execute Cell 4 (run) — **returns immediately** (training runs in background).
9. Poll Cell 7 (status_check) to monitor progress (see §Babysitting Protocol).
10. When Cell 7 shows `finished_run`, execute Cell 5 (collect).
11. Execute Cell 6 (eval) for metrics summary.

## Babysitting Protocol

Cell 4 launches training as a background process and returns immediately.
The kernel stays free for polling.

### Adaptive polling (naps)

Cell 7 now displays a **Training Progress** section with chunk-level ETA
and a **Recommended Nap** duration. Follow the recommended nap time
instead of using fixed intervals:

| Situation | Nap duration | Why |
|-----------|-------------|-----|
| Early training (< 10% done) | **3 min** | Pace is still settling; catch early failures fast. |
| Finishing soon (ETA < 10 min) | **3 min** | Don't miss completion. |
| Moderate ETA (10–30 min), stable pace | **10 min** | Pace deviation < 20% — safe to sleep longer. |
| Moderate ETA (10–30 min), variable pace | **5 min** | Something may be off. |
| Long ETA (> 30 min), stable pace | **15 min** | Training is cruising — take a long nap. |
| Long ETA (> 30 min), variable pace | **10 min** | Pace fluctuations warrant closer watch. |

**Stability** is determined by comparing the recent chunk pace (last 10
chunks) against the overall average. A deviation < 20% is "STABLE";
otherwise "VARIABLE".

If no progress data is shown yet (training is still loading data or
compiling kernels), poll again in **3 minutes**.

### Cell 7 output

Cell 7 shows:
- **Process liveness**: whether the PID is still running or finished
- **Status JSON**: current status + metadata from the training process
- **Training Progress**: chunks done/total, elapsed time, ETA, pace
  stability indicator, and recommended nap duration
- **GPU utilization**: memory, compute %, temperature
- **Log tail**: last 30 lines of training output

### Decision table

| Process state | Status JSON      | Action                                          |
|---------------|------------------|-------------------------------------------------|
| RUNNING       | `running`        | Training in progress — check back later.        |
| FINISHED      | `finished_run`   | Proceed to Cell 5 (collect).                    |
| FINISHED      | `failed`         | Read error, diagnose, fix, re-run (see §Failure Diagnosis). |
| FINISHED      | No status update | Check log tail for crash details; check full log at §File Locations. |
| —             | `collected`      | Results on Drive — proceed to Cell 6 (eval).    |
| —             | `evaluated`      | Everything complete — report metrics to user.   |

## OOM Handling

All OOM guidance is consolidated here. Other sections cross-reference this one.

When you see `RuntimeError: CUDA out of memory`:

1. **Halve `BATCH_SIZE`** in Cell 2. This is the most effective single knob.
2. If still OOM after halving, **reduce `TRAIN_WINDOW`** (e.g., 50000 → 25000
   for patchts, 24000 → 12000 for ae_ridge).
3. If both fail, note the GPU type and memory ceiling and report to the user.

**Model-specific OOM context:**
- **PatchTSMixer** is memory-hungry. T4 (16 GB) may struggle with the default
  window. Start with `BATCH_SIZE = 25` on T4 if preempting OOM.
- **AE+Ridge** is more memory-efficient. OOM is rare at default settings.

Retry once autonomously after adjusting parameters. If OOM persists after
the adjustment, stop and report.

## Failure Diagnosis

When status is `failed`:

1. Read the `error` and `traceback` fields from Cell 7 output.
2. If the last 30 lines from Cell 7 are insufficient, read more of the full
   log at `/content/harxhar_train.log` (e.g., `!tail -200 /content/harxhar_train.log`
   or `!cat /content/harxhar_train.log` for the complete output).
3. Cross-reference against local source files:
   - PatchTSMixer/AE+Ridge issues → `projects/dl/models/deep_learning.py`, `projects/dl/backtest/gpu_engine.py`
   - Data issues → `core/data/pipeline.py`, `core/data/transforms.py`
   - GPU/CUDA errors → `projects/dl/backtest/gpu_kernels.py`

4. Known failure modes:

   | Error                                | Cause                        | Fix                                                        |
   |--------------------------------------|------------------------------|------------------------------------------------------------|
   | `RuntimeError: CUDA out of memory`   | GPU memory exceeded          | See §OOM Handling.                                         |
   | `ValueError: shape mismatch`         | `context_length` / `patch_length` don't divide evenly | Check `projects/dl/config.py`. |
   | `_WorkerError`                       | GPU worker crashed (usually OOM) | See §OOM Handling; also check `gpu_engine.py`.         |
   | `KeyError: column not found`         | Feature group mismatch       | Check `core/features/feature_groups.py`.                    |
   | `ConnectionError` / Drive unmounted  | Colab lost Drive mount       | Re-execute Cell 1 (setup).                                 |
   | `timeout`                            | Exceeded `TIMEOUT_HOURS`     | Increase it or reduce `TRAIN_WINDOW`.                      |
   | `git pull --ff-only` fails in Cell 1 | Branch diverged or dirty tree on Colab | See §Git Pull Failure Recovery.                |

5. Apply the fix:
   - **Parameter issue** → edit Cell 2 via MCP, re-run from Cell 3.
   - **Code issue** → fix the source file locally, `git commit` and `git push`,
     then re-run Cell 1 on Colab (which does `git pull`) to pick up the fix.
   - **Colab resource issue** → note it and suggest alternatives.

6. If the same error persists after one retry, stop and report:
   - Full error message
   - Which source file is involved
   - Your diagnosis
   - Suggested fix

## Git Pull Failure Recovery

If `git pull --ff-only` in Cell 1 fails (e.g., due to diverged branches or
local modifications on Colab from a prior session):

1. Edit Cell 1 or run a one-off bash command in Colab to execute:
   ```bash
   cd /content/harxhar && git reset --hard origin/main && git pull --ff-only
   ```
2. This discards any Colab-side changes (which are ephemeral anyway) and
   forces the Colab clone to match the remote.
3. Do **not** make this the permanent Cell 1 behavior — `--ff-only` is the
   normal safe default. Use `reset --hard` only as recovery.

## Crash Recovery (Checkpoint Resume)

If a Colab runtime disconnects mid-run and `CHECKPOINT_DIR` was set:

1. Reconnect to Colab, re-run Cell 1 (setup).
2. Edit Cell 2 with the same parameters + `CHECKPOINT_DIR` pointing to the
   same Drive path (e.g., `"/content/drive/MyDrive/harxhar_checkpoints"`).
3. Re-run from Cell 3. The engine resumes from the last saved checkpoint.

If `CHECKPOINT_DIR` was not set, the run must restart from scratch.

**Recommendation:** For runs expected to take > 1 hour, always set
`CHECKPOINT_DIR = "/content/drive/MyDrive/harxhar_checkpoints"` in Cell 2.

## Model-Specific Notes

### PatchTSMixer
- `EXPERIMENT = "patchts"`
- HuggingFace transformer: context=241, patch=47, stride=31
- Defaults: epochs=150, lr=1e-4, batch_size=50, train_window=50000
- See §OOM Handling for memory guidance on T4.

### AE+Ridge
- `EXPERIMENT = "ae_ridge"`
- Hybrid: autoencoder with mixed loss (0.5× reconstruction + 0.5× prediction) + Ridge
- Defaults: epochs=50, lr=1e-3, batch_size=4, train_window=24000, n_components=5
- More memory-efficient than PatchTSMixer
- Uses `np.random.seed(42)` for reproducibility

## Scheduling Multiple Experiments

When running a sweep (e.g., "PatchTSMixer across horizons 1, 6, 12, 48"):

- Run sequentially in the same notebook, editing Cell 2 between runs.
- After each run, **always collect results (Cell 5) before starting the next.**
  Results are namespaced by experiment config, but failing to collect before
  the next run risks overwriting intermediate outputs on the Colab local
  filesystem.
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

**Do without asking (babysitting scope):**
- Execute cells in the standard sequence
- Poll status during long runs (see §Babysitting Protocol for cadence)
- Retry on OOM by adjusting Cell 2 parameters per §OOM Handling
- Collect and evaluate results
- Run sequential experiment sweeps as requested
- Fix runtime errors that block experiment completion — including crashes, OOM,
  shape mismatches, missing imports, data loading failures, and other errors
  that prevent the current experiment design from running as intended
- Adjust Cell 2 parameters to work around resource constraints (memory, timeout)
- Commit and push source-code bug fixes so Colab picks them up via Cell 1
- Recover from git pull failures per §Git Pull Failure Recovery

**Scope restriction — only touch what's needed to keep experiments running:**
- Every edit must be the minimum change required to fix an error or resource
  issue. Do not refactor, reorganize, or "improve" surrounding code.
- Do NOT change model architectures, loss functions, training algorithms,
  feature engineering logic, evaluation methodology, or any design decisions.
- Do NOT add new features, new parameters, new abstractions, or new files.
- Do NOT rename functions/classes, restructure modules, or change APIs.
- If you're unsure whether a fix crosses into design-change territory,
  ask the user before editing.

**Ask the user before:**
- Changing experiment type or model architecture
- Any edit that goes beyond fixing an error (refactors, design changes,
  new features, algorithmic changes)
- Deciding whether to continue after 2+ consecutive failures on the same error
- Changing Cell 2 configs for reasons other than resource constraints

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

| What                  | Path                                                      |
|-----------------------|-----------------------------------------------------------|
| Notebook source       | `notebooks/dl_runner.ipynb` (in repo)                     |
| Notebook template     | `scripts/dl_runner_template.py` (canonical source)        |
| Status JSON           | `Drive/MyDrive/harxhar_status/dl_runner.json`             |
| Results (Drive)       | `Drive/MyDrive/harxhar_results/<experiment>/*.csv`        |
| Training data         | `all30min/*.parquet` (in repo, cloned to Colab)           |
| Repo (on Colab)       | `/content/harxhar/`                                       |
| Repo (local)          | This project folder                                       |
| Full training log     | `/content/harxhar_train.log` (hardcoded in Cells 4 & 7)  |
| Training progress     | `/content/harxhar_progress.json` (live ETA + pace stats)  |
| Checkpoints (if set)  | Value of `CHECKPOINT_DIR` (typically on Drive)             |
