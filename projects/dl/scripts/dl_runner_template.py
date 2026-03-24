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
| 4 | run | Launch DL experiment (background) | Once per run |
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

from projects.dl.notebook_utils import verify_gpu, clear_status

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
from projects.dl.notebook_utils import configure_cuda, write_status, get_gpu_utilization
from projects.dl.config import DL_CONFIG, AE_RIDGE_GPU_CONFIG

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
# --- Cell 4: Launch experiment (background process) ---
import subprocess, os, signal

PIDFILE = '/content/harxhar_train.pid'
LOGFILE = '/content/harxhar_train.log'

# Build CLI command with all parameters from Cell 2/3
extra_args = []
if BATCH_SIZE is not None:
    extra_args += ['--batch-size', str(BATCH_SIZE)]
if EPOCHS is not None:
    extra_args += ['--epochs', str(EPOCHS)]
if LEARNING_RATE is not None:
    extra_args += ['--learning-rate', str(LEARNING_RATE)]
if TRAIN_WINDOW is not None:
    extra_args += ['--train-window', str(TRAIN_WINDOW)]
if CHECKPOINT_DIR is not None:
    extra_args += ['--checkpoint-dir', CHECKPOINT_DIR]
if LOSS_LOG_PATH is not None:
    extra_args += ['--loss-log-path', LOSS_LOG_PATH]

output_csv = f"{RESULTS_DIR}/{EXPERIMENT}_h{HORIZON}_results.csv"
os.makedirs(RESULTS_DIR, exist_ok=True)

PROGRESS_FILE = '/content/harxhar_progress.json'

cmd = [
    'python', '-m', 'projects.dl.cli.gpu_executor',
    '--experiment', EXPERIMENT,
    '--input-path', DATA_PATH,
    '--output', output_csv,
    '--gpu-count', str(GPU_COUNT),
    '--horizon', str(HORIZON),
    '--timeout-hours', str(TIMEOUT_HOURS),
    '--write-status',
    '--progress-path', PROGRESS_FILE,
] + extra_args

# Launch in background — kernel stays free for status polling
with open(LOGFILE, 'w') as log_fh:
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh, stderr=subprocess.STDOUT,
        cwd='/content/harxhar',
        preexec_fn=os.setsid,
    )

with open(PIDFILE, 'w') as f:
    f.write(str(proc.pid))

print(f'Training launched as PID {proc.pid}')
print(f'Output: {output_csv}')
print(f'Log:    {LOGFILE}')
print(f'Kernel is free — use Cell 7 to check progress.')
"""

CELL_5_COLLECT = """\
# --- Cell 5: Collect results to Drive ---
import os
import shutil
import pandas as pd
from projects.dl.notebook_utils import write_status

csv_path = f"{RESULTS_DIR}/{EXPERIMENT}_h{HORIZON}_results.csv"
assert os.path.exists(csv_path), f"Results CSV not found: {csv_path}. Is training finished?"

results = pd.read_csv(csv_path)
print(f"Loaded {len(results):,} rows from {csv_path}")

drive_results_dir = f"/content/drive/MyDrive/harxhar_results/{EXPERIMENT}"
os.makedirs(drive_results_dir, exist_ok=True)
drive_csv = shutil.copy2(csv_path, drive_results_dir)

write_status("collected", local_csv=csv_path, drive_csv=drive_csv, n_results=len(results))
print(f"Results copied to Drive: {drive_csv}")
"""

CELL_6_EVAL = """\
# --- Cell 6: Evaluate metrics ---
from core.evaluation.metrics import calculate_global_metrics
from projects.dl.notebook_utils import print_metrics, write_status

metrics = calculate_global_metrics(results)
print_metrics(metrics)

write_status("evaluated", metrics={k: round(v, 6) for k, v in metrics.items()})
"""

CELL_7_STATUS = """\
# --- Cell 7: Status check (safe to run anytime) ---
import json, os, subprocess
from projects.dl.notebook_utils import read_status, get_gpu_utilization, read_progress, format_eta, recommend_nap

PIDFILE = '/content/harxhar_train.pid'
LOGFILE = '/content/harxhar_train.log'

# --- Process liveness ---
if os.path.exists(PIDFILE):
    pid = int(open(PIDFILE).read().strip())
    try:
        os.kill(pid, 0)  # signal 0 = check existence only
        print(f'Process {pid}: RUNNING')
    except ProcessLookupError:
        print(f'Process {pid}: FINISHED')
    except PermissionError:
        print(f'Process {pid}: RUNNING (owned by another user)')
else:
    print('No PID file — training not launched yet.')

# --- Status JSON ---
status = read_status()
if status:
    print('\\n=== Status ===')
    print(json.dumps(status, indent=2, default=str))
else:
    print('\\nNo status file found.')

# --- Training progress & ETA ---
progress = read_progress()
if progress:
    pct = progress.get('pct_complete', 0)
    done = progress.get('chunks_done', 0)
    total = progress.get('chunks_total', 0)
    eta = progress.get('eta_sec', 0)
    avg = progress.get('avg_chunk_sec', 0)
    recent = progress.get('recent_avg_chunk_sec', avg)
    wall = progress.get('wall_elapsed_sec', 0)

    pace_dev = abs(recent - avg) / avg * 100 if avg > 0 else 0
    stability = 'STABLE' if pace_dev < 20 else 'VARIABLE'

    print(f'\\n=== Training Progress ===')
    print(f'Chunks: {done}/{total} ({pct:.1f}%)')
    print(f'Elapsed: {format_eta(wall)}')
    print(f'ETA: {format_eta(eta)}')
    print(f'Pace: {avg:.1f}s/chunk (recent: {recent:.1f}s/chunk) — {stability} ({pace_dev:.0f}% deviation)')

    nap_sec, nap_reason = recommend_nap(progress)
    print(f'\\n=== Recommended Nap ===')
    print(f'Next check in: {nap_sec // 60} minutes ({nap_reason})')
else:
    print('\\nNo progress data yet (training may still be loading data).')

# --- GPU utilization ---
gpu = get_gpu_utilization()
print(f"\\nGPU: {gpu.get('gpu_name', 'N/A')}")
print(f"Utilization: {gpu.get('gpu_util_pct', 'N/A')}%")
print(f"Memory: {gpu.get('mem_used_mb', 'N/A')}/{gpu.get('mem_total_mb', 'N/A')} MB")
print(f"Temperature: {gpu.get('temp_c', 'N/A')}°C")

# --- Log tail ---
if os.path.exists(LOGFILE):
    print('\\n=== Last 30 lines of log ===')
    !tail -30 {LOGFILE}
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
    cell: dict[str, object] = {
        "cell_type": cell_type,
        "source": source.splitlines(True),
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
