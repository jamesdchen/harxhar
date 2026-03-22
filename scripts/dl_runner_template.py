"""Generate notebooks/dl_runner.ipynb from cell definitions.

Re-run this script whenever you change the cell structure so the
notebook stays in sync with the canonical definitions here.

Usage:
    python scripts/dl_runner_template.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Cell definitions (8 cells: 1 markdown header + 7 code cells)
# ---------------------------------------------------------------------------

CELL_0_MARKDOWN = """\
# HARXHAR Deep Learning Runner

Unified notebook for PatchTSMixer and AE+Ridge GPU backtests.
Designed for **cell-by-cell execution** via the `googlecolab/colab-mcp`
connector — do **not** use *Runtime > Run All*.

| Cell | Tag | Purpose | When to run |
|------|-----|---------|-------------|
| 1 | setup | Mount Drive, clone repo, install deps | Once per runtime |
| 2 | parameters | Experiment config — **edit before each run** | Before each run |
| 3 | validate | Check config + GPU, write status `"validated"` | After editing params |
| 4 | run | Execute the DL experiment | Once per run |
| 5 | collect | Copy results to Drive | After run succeeds |
| 6 | eval | QLIKE / MSE / MAE on results | After collect |
| 7 | status_check | Read status file + GPU utilization | Anytime (polling) |
"""

CELL_1_SETUP = """\
# --- Cell 1: Setup ---
import os

# Mount Google Drive (required for status tracking and result persistence)
from google.colab import drive
drive.mount("/content/drive")

REPO_URL = "https://github.com/jamesdchen/harxhar.git"
REPO_DIR = "/content/harxhar"

if not os.path.isdir(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull --ff-only

os.chdir(REPO_DIR)
!pip install -q torch transformers numpy pandas scikit-learn tqdm pyarrow

from src.notebook_utils import verify_gpu, clear_status

verify_gpu()
clear_status()
print("Setup complete. Drive mounted. Status cleared.")
"""

CELL_2_PARAMETERS = """\
# --- Cell 2: Parameters (edit before each run) ---

EXPERIMENT = "patchts"       # "patchts" or "ae_ridge"
HORIZON = 1                  # forecast horizon (1-48)
TRAIN_WINDOW = None          # None = use default from config
GPU_COUNT = 1                # Colab typically has 1
BATCH_SIZE = None            # None = use default
EPOCHS = None                # None = use default
LEARNING_RATE = None         # None = use default
DATA_PATH = "all30min"
RESULTS_DIR = "results_dl"
TIMEOUT_HOURS = 2.0          # max runtime before auto-fail
CHECKPOINT_DIR = None        # set to enable crash recovery
LOSS_LOG_PATH = None         # set to save per-epoch training losses
"""

CELL_3_VALIDATE = """\
# --- Cell 3: Validate config + GPU ---
import copy
from src.notebook_utils import configure_cuda, write_status, get_gpu_utilization
from src.core.config import DL_CONFIG, AE_RIDGE_GPU_CONFIG

configure_cuda()
gpu_info = get_gpu_utilization()

if EXPERIMENT == "patchts":
    config = copy.deepcopy(DL_CONFIG)
elif EXPERIMENT == "ae_ridge":
    config = copy.deepcopy(AE_RIDGE_GPU_CONFIG)
else:
    raise ValueError(f"Unknown experiment: {EXPERIMENT!r}. Use 'patchts' or 'ae_ridge'.")

config["data_path"] = DATA_PATH
config["output_path"] = f"{RESULTS_DIR}/{EXPERIMENT}_h{HORIZON}_results.csv"
config["gpu_count"] = GPU_COUNT
if TRAIN_WINDOW is not None:
    config["train_window"] = TRAIN_WINDOW
if BATCH_SIZE is not None:
    config["train"]["batch_size"] = BATCH_SIZE
if EPOCHS is not None:
    config["train"]["num_epochs"] = EPOCHS
if LEARNING_RATE is not None:
    config["train"]["learning_rate"] = LEARNING_RATE
if CHECKPOINT_DIR is not None:
    config["checkpoint_dir"] = CHECKPOINT_DIR
if LOSS_LOG_PATH is not None:
    config["loss_log_path"] = LOSS_LOG_PATH

write_status(
    "validated",
    experiment=EXPERIMENT,
    horizon=HORIZON,
    gpu_name=gpu_info.get("gpu_name", "unknown"),
    config=config,
)
print(f"Config OK — {EXPERIMENT}, horizon={HORIZON}, GPU: {gpu_info.get('gpu_name', 'N/A')}")
"""

CELL_4_RUN = """\
# --- Cell 4: Run experiment ---
import signal
import time
import traceback
from src.data import load_and_prep_data_strided
from src.notebook_utils import write_status

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout(f"Run exceeded {TIMEOUT_HOURS}h timeout")

signal.signal(signal.SIGALRM, _alarm)
signal.alarm(int(TIMEOUT_HOURS * 3600))
_t0 = time.time()

try:
    write_status("running")

    if EXPERIMENT == "patchts":
        from src.backtest.gpu_engine import run_multigpu_backtest

        hparams = {
            "exog_cols": "none",
            "use_transform_exog": False,
            "use_diurnal": True,
            "allow_missing": False,
            "use_winsor": False,
        }
        X_np, y_np, dates, baselines, features = load_and_prep_data_strided(
            hparams, config["data_path"], lag=config["model"]["context_len"]
        )
        print(f"Data: {X_np.shape[0]:,} samples, {X_np.shape[1]} features")
        results = run_multigpu_backtest(
            X_np, y_np, dates, baselines, config,
            model_module="src.models.deep_learning",
        )

    elif EXPERIMENT == "ae_ridge":
        import numpy as np
        np.random.seed(42)
        from src.backtest.gpu_engine import run_ae_multigpu_backtest

        hparams = {
            "exog_cols": "none",
            "use_transform_exog": False,
            "use_diurnal": True,
            "allow_missing": False,
            "use_winsor": True,
            "feature_type": "raw",
            "lag_scope": "global",
        }
        X_np, y_np, dates, baselines, features = load_and_prep_data_strided(
            hparams, config["data_path"]
        )
        config["model"]["n_features"] = X_np.shape[1]
        print(f"Data: {X_np.shape[0]:,} samples, {X_np.shape[1]} features")
        results = run_ae_multigpu_backtest(X_np, y_np, dates, baselines, config)

    elapsed_min = (time.time() - _t0) / 60
    n = results.shape[0] if hasattr(results, "shape") else len(results)
    write_status("finished_run", n_results=n, elapsed_minutes=round(elapsed_min, 1))
    print(f"Run complete — {n:,} results in {elapsed_min:.1f} min")

except _Timeout:
    write_status("failed", error="timeout", timeout_hours=TIMEOUT_HOURS)
    print("FAILED: timeout exceeded")
    raise
except Exception as exc:
    write_status("failed", error=str(exc), traceback=traceback.format_exc()[-1000:])
    print(f"FAILED: {exc}")
    raise
finally:
    signal.alarm(0)
"""

CELL_5_COLLECT = """\
# --- Cell 5: Collect results to Drive ---
import os
import shutil
from src.notebook_utils import save_results, write_status

os.makedirs(RESULTS_DIR, exist_ok=True)
fname = f"{EXPERIMENT}_h{HORIZON}_results.csv"
csv_path = save_results(results, RESULTS_DIR, fname)

drive_results_dir = f"/content/drive/MyDrive/harxhar_results/{EXPERIMENT}"
os.makedirs(drive_results_dir, exist_ok=True)
drive_csv = shutil.copy2(csv_path, drive_results_dir)

write_status("collected", local_csv=csv_path, drive_csv=drive_csv)
print(f"Results saved to Drive: {drive_csv}")
"""

CELL_6_EVAL = """\
# --- Cell 6: Evaluate metrics ---
from src.evaluation.metrics import calculate_global_metrics
from src.notebook_utils import print_metrics, write_status

metrics = calculate_global_metrics(results)
print_metrics(metrics)

write_status("evaluated", metrics={k: round(v, 6) for k, v in metrics.items()})
"""

CELL_7_STATUS = """\
# --- Cell 7: Status check (safe to run anytime) ---
import json
from src.notebook_utils import read_status, get_gpu_utilization

status = read_status()
if status:
    print(json.dumps(status, indent=2, default=str))
else:
    print("No status file found.")

gpu = get_gpu_utilization()
print(f"\\nGPU: {gpu.get('gpu_name', 'N/A')}")
print(f"Utilization: {gpu.get('gpu_util_pct', 'N/A')}%")
print(f"Memory: {gpu.get('mem_used_mb', 'N/A')}/{gpu.get('mem_total_mb', 'N/A')} MB")
print(f"Temperature: {gpu.get('temp_c', 'N/A')}°C")
"""

# ---------------------------------------------------------------------------
# Notebook generation
# ---------------------------------------------------------------------------

CELLS = [
    {"cell_type": "markdown", "source": CELL_0_MARKDOWN},
    {"cell_type": "code", "source": CELL_1_SETUP},
    {"cell_type": "code", "source": CELL_2_PARAMETERS},
    {"cell_type": "code", "source": CELL_3_VALIDATE},
    {"cell_type": "code", "source": CELL_4_RUN},
    {"cell_type": "code", "source": CELL_5_COLLECT},
    {"cell_type": "code", "source": CELL_6_EVAL},
    {"cell_type": "code", "source": CELL_7_STATUS},
]


def _make_cell(cell_type: str, source: str) -> dict:
    """Build a single nbformat-v4 cell dict."""
    cell = {
        "cell_type": cell_type,
        "source": source.split("\n"),
        "metadata": {},
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def generate(output_path: str = "notebooks/dl_runner.ipynb") -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": "dl_runner.ipynb", "provenance": []},
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "accelerator": "GPU",
        },
        "cells": [_make_cell(c["cell_type"], c["source"]) for c in CELLS],
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated {out}")


if __name__ == "__main__":
    generate()
