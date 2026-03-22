# HARXHAR Deep Learning — Cowork Instructions

## Project Overview

HARXHAR is a realized volatility forecasting system. It predicts intraday
30-minute volatility using HAR-family models with exogenous features on
rolling-window backtests.

**Models:**
- CPU: Ridge, XGBoost, LightGBM, Random Forest, SARIMAX
- GPU (Colab): PatchTSMixer (HuggingFace transformer), AE+Ridge (autoencoder + Ridge hybrid)

**Data:** Parquet files in `all30min/`, one per ticker, with 30-min OHLCV bars
(48 periods per trading day).

**Your role:** Drive deep learning experiments on Google Colab via the
`googlecolab/colab-mcp` MCP connector. You execute notebook cells individually,
monitor progress through a Drive-persisted status JSON, and report results.

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

## Notebook Cell Map

| Cell | Tag            | Purpose                                    | When to run          |
|------|----------------|--------------------------------------------|----------------------|
| 0    | (markdown)     | Header — skip                              | Never                |
| 1    | setup          | Install deps, mount Drive, clone repo      | Once per runtime     |
| 2    | parameters     | Experiment config — **you edit this**       | Before each run      |
| 3    | validate       | Check config + GPU availability            | After editing params |
| 4    | run            | Execute the DL experiment                  | Once per run         |
| 5    | collect        | Copy results from Colab to Drive           | After run succeeds   |
| 6    | eval           | Quick QLIKE/MSE/MAE on collected results   | After collect        |
| 7    | status_check   | Read status file + GPU utilization          | Anytime (polling)    |

## Status JSON

Coordination file: `Drive/MyDrive/harxhar_status/dl_runner.json`

Each cell writes status transitions. Read it via Cell 7.

| Status         | Meaning                                |
|----------------|----------------------------------------|
| `validated`    | Config checked, ready to run           |
| `running`      | Experiment in progress                 |
| `finished_run` | Run complete, proceed to collect       |
| `failed`       | Error occurred — read `error` field    |
| `collected`    | Results saved to Drive                 |
| `evaluated`    | Metrics computed, experiment done      |

## Execution Flow

```
Cell 1 (setup) → output says "Setup complete"?
├─ No  → read error, retry Cell 1
└─ Yes → edit Cell 2 (set params) → Cell 3 (validate) → "Config OK"?
         ├─ No  → fix Cell 2, retry Cell 3
         └─ Yes → Cell 4 (run) → poll Cell 7 every 2-3 min
                  ├─ status=running     → wait, poll again
                  ├─ status=finished_run → Cell 5 (collect) → Cell 6 (eval) → report results
                  └─ status=failed      → diagnose (see Failure Diagnosis below)
```

## Babysitting Protocol

While Cell 4 (run) is executing:

- Execute Cell 7 every 2-3 minutes to poll the status JSON.
- Check GPU utilization and memory in the Cell 7 output.
- Act on the status value:
  - `running` — experiment in progress, poll again later.
  - `finished_run` — proceed to Cell 5 (collect), then Cell 6 (eval).
  - `failed` — read the error message, diagnose, and handle.

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
   - Parameter issue → edit Cell 2, re-run from Cell 3
   - Code issue → report to user what needs fixing in the repo
   - Colab resource issue → note it and suggest alternatives

5. If the same error persists after one retry, stop and report:
   - Full error message
   - Which source file is involved
   - Your diagnosis
   - Suggested fix

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

**Ask the user before:**
- Changing experiment type or model architecture
- Modifying source code in the repo
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
- Don't ignore GPU quota warnings — if "GPU quota exceeded", stop and report.

## File Locations

- Notebook: `Drive/MyDrive/harxhar_notebooks/dl_runner.ipynb`
- Status: `Drive/MyDrive/harxhar_status/dl_runner.json`
- Results: `Drive/MyDrive/harxhar_results/<experiment>/*.csv`
- Data: `Drive/MyDrive/harxhar_data/all30min/*.parquet`
- Repo (on Colab): `/content/harxhar/`
- Repo (local): this project folder
