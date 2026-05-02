"""harxhar tasks declaration — generated for the new compute(args) contract.

Per-executor FLAGS dict declares the CLI shape; the matching .hpc/cli.py
dispatcher reads it at runtime and parses argv for whichever executor
module is invoked. resolve(task_id)/total() control task fan-out per
/submit-hpc submission.
"""

from __future__ import annotations

from hpc_mapreduce.executor_cli import flag, generic_args, gpu_args

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

# ─── Tasks: starter sanity (a single task; replace with real grid axes) ────
_TASKS: list[dict] = [
    {"horizon": 1, "train_window": 500, "seed": 42},
]


def total() -> int:
    return len(_TASKS)


def resolve(task_id: int) -> dict:
    return _TASKS[task_id]
