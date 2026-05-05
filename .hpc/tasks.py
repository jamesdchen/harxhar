"""harxhar tasks declaration — generated for the new compute(args) contract.

Per-executor FLAGS dict declares the CLI shape; the matching .hpc/cli.py
dispatcher reads it at runtime and parses argv for whichever executor
module is invoked. resolve(task_id)/total() control task fan-out per
/submit-hpc submission.
"""

from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path as _Path

from claude_hpc.executor_cli import flag, generic_args, gpu_args

# Time-of-day segments. Canonical source: src/transforms.py:SEGMENT_CHOICES
# (defined in notebooks/pipeline/02_transforms.ipynb). Mirrored here as a
# constant rather than imported because tasks.py is loaded at /submit-hpc
# time before the experiment's deps are necessarily on path; an import
# would force pandas/numpy to resolve before scaffolding can finish. Keep
# in sync with src/transforms.py if those values ever change.
_SEGMENT_CHOICES = ("all", "morning", "midday", "closing", "overnight")

# Shared CPU-executor flag set used by ml_ridge / ml_xgboost / ml_lightgbm /
# ml_random_forest / ml_pcr / ml_baseline. Method-specific extras (e.g.
# --n-components for ml_pcr) are appended per-key below.
_CPU_BASE = [
    *generic_args(),
    flag("horizon", int, default=1),
    flag("train_window", int, default=500, help="training window in days"),
    flag(
        "refit_frequency",
        int,
        default=None,
        help="how often to refit during walk-forward; None falls back to per-method default",
    ),
    flag("exog_cols", str, default=None, help="pipe-separated exog columns, e.g. vix|sentiment"),
    flag("params_file", str, default=None, help="JSON file with tuned hyperparams"),
    flag("segment", str, default=None, choices=_SEGMENT_CHOICES, help="time-of-day segment"),
    flag(
        "lag_scope",
        str,
        default="global",
        choices=("global", "intra"),
        help="compute lags on full dataset or per-segment",
    ),
]

# Shared DL-executor flag set used by dl_patchts / dl_ae_ridge.
_DL_BASE = [
    *generic_args(),
    *gpu_args(),
    flag("horizon", int, default=1),
]


FLAGS: dict[str, list] = {
    "src.ml_ridge": _CPU_BASE,
    "src.ml_xgboost": _CPU_BASE,
    "src.ml_lightgbm": _CPU_BASE,
    "src.ml_random_forest": _CPU_BASE,
    "src.ml_baseline": _CPU_BASE,
    "src.ml_pcr": [
        *_CPU_BASE,
        flag("n_components", int, default=5, help="PCA components for PCR"),
    ],
    "src.dl_patchts": _DL_BASE,
    "src.dl_ae_ridge": [
        *_DL_BASE,
        flag(
            "n_components",
            int,
            default=None,
            help="autoencoder bottleneck dimension; None = method default",
        ),
    ],
}

# ─── Tasks ────────────────────────────────────────────────────────────────
#
# Open-loop default: one sanity task. Used when /submit-hpc is invoked
# without --campaign-id (HPC_CAMPAIGN_ID unset).
#
# Closed-loop (campaign): when HPC_CAMPAIGN_ID is set, this module asks
# Optuna for a batch of `_BATCH` trials per /submit-hpc iteration and
# materializes one JSON params file per trial under
# `params/<cid>/iter_<N>/`. resolve(task_id) returns the path so the
# executor receives `--params-file params/.../trial_K.json` — no
# changes to FLAGS or src/ml_xgboost.py needed. The campaign driver
# (.hpc/campaigns/<cid>/score_iter.py) reads each iteration's manifest.json
# and the per-task qlike.json after the array job lands, then calls
# study.tell() to push results back into the Optuna study.
_CAMPAIGN_ID = _os.environ.get("HPC_CAMPAIGN_ID")
_TUNE_BATCH = 10  # trials per /submit-hpc iteration
_TUNE_BUDGET = 100  # total trials before campaign stops
_OPTUNA_STORAGE = ".hpc/optuna.db"
_TUNE_MODEL = "xgb"


def _build_xgb_optuna_batch() -> list[dict]:
    """Ask Optuna for the next batch of XGB trials (idempotent on disk).

    Re-running the same iteration (e.g. on resume) re-uses the existing
    manifest.json + trial_*.json instead of asking Optuna again, which
    would orphan trial numbers in the study.
    """
    from claude_hpc.mapreduce.reduce.history import prior as _prior

    from src.tune_tree import _get_search_space, _load_or_create_study

    n_done = len(_prior(".", _CAMPAIGN_ID)) * _TUNE_BATCH
    if n_done >= _TUNE_BUDGET:
        return []

    n_iter = n_done // _TUNE_BATCH
    n_this_iter = min(_TUNE_BATCH, _TUNE_BUDGET - n_done)
    iter_dir = _Path(f"params/{_CAMPAIGN_ID}/iter_{n_iter:03d}")
    manifest_file = iter_dir / "manifest.json"

    if not manifest_file.exists():
        iter_dir.mkdir(parents=True, exist_ok=True)
        study = _load_or_create_study(_TUNE_MODEL, storage_path=_OPTUNA_STORAGE)
        trials_info = []
        for i in range(n_this_iter):
            trial = study.ask(fixed_distributions=_get_search_space(_TUNE_MODEL))
            (iter_dir / f"trial_{i}.json").write_text(_json.dumps(trial.params, indent=2))
            trials_info.append({"id": i, "file": f"trial_{i}.json", "optuna_number": trial.number})
        manifest_file.write_text(
            _json.dumps(
                {
                    "model": _TUNE_MODEL,
                    "study_name": study.study_name,
                    "batch_size": n_this_iter,
                    "trials": trials_info,
                },
                indent=2,
            )
        )

    return [{"params_file": (iter_dir / f"trial_{i}.json").as_posix()} for i in range(n_this_iter)]


# ─── Open-loop backtest chunking ──────────────────────────────────────────
#
# When HPC_CAMPAIGN_ID is unset, split the walk-forward backtest into
# `_TOTAL_CHUNKS` chunks. Each task gets a (start, end) row-index slice
# that includes a `_TRAIN_OVERLAP` warm-up prefix plus its share of OOS
# predictions; the executor trains on the first overlap rows and predicts
# the rest. Bucket and model are NOT axes here — they're baked into
# per-(model, bucket) run sidecars at submit time, so 18 array submissions
# (3 models × 6 buckets) all share this same chunked tasks.py.
#
# Constants mirrored from the experiment (cannot be imported because
# tasks.py runs in the framework env without pandas/numpy):
#   _TOTAL_ROWS    : src.hpc_backtest_shim.get_total_rows("all30min", 1)
#   _TRAIN_OVERLAP : default --train-window (500) × src.transforms.PERIODS_PER_DAY (48)
# Re-probe and update if the data vintage or HAR-lag set changes.
_TOTAL_ROWS = 242934
_TRAIN_OVERLAP = 24000
_TOTAL_CHUNKS = 100


def _range_split_overlap(total_rows: int, total_chunks: int, chunk_id: int, overlap: int) -> tuple[int, int]:
    """Mirror of src.hpc_backtest_shim.range_split_overlap (stdlib-only)."""
    oos_rows = total_rows - overlap
    base = oos_rows // total_chunks
    rem = oos_rows % total_chunks
    oos_start = overlap + base * chunk_id + min(chunk_id, rem)
    oos_end = oos_start + base + (1 if chunk_id < rem else 0)
    return oos_start - overlap, oos_end


def _build_chunk_tasks() -> list[dict]:
    out: list[dict] = []
    for c in range(_TOTAL_CHUNKS):
        start, end = _range_split_overlap(_TOTAL_ROWS, _TOTAL_CHUNKS, c, _TRAIN_OVERLAP)
        out.append({"start": start, "end": end})
    return out


_TASKS: list[dict] = _build_xgb_optuna_batch() if _CAMPAIGN_ID else _build_chunk_tasks()


def total() -> int:
    return len(_TASKS)


def resolve(task_id: int) -> dict:
    return _TASKS[task_id]
