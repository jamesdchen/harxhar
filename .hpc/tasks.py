"""harxhar tasks declaration — generated for the new compute(args) contract.

Per-executor FLAGS dict declares the CLI shape; the matching .hpc/cli.py
dispatcher reads it at runtime and parses argv for whichever executor
module is invoked. resolve(task_id)/total() control task fan-out per
/submit-hpc submission.

This is a starter scaffold for the migration proof-of-concept (ml_ridge
only). Add other executors' FLAGS entries as their notebooks get
migrated; add real grid axes to ``_TASKS`` when running an actual
parameter sweep.
"""

from __future__ import annotations

from hpc_mapreduce.executor_cli import flag, generic_args, gpu_args

# Time-of-day segments accepted by ml_ridge / src.executor.SEGMENT_CHOICES.
# Ridge is the only method that varies by segment today; the choice list
# stays here (in tasks.py) rather than the executor file because the
# allowed values are a parallelization concern, not a substance concern.
_SEGMENT_CHOICES = ("am", "pm", "all", "morning_close", "afternoon_close")


FLAGS: dict[str, list] = {
    "src.ml_ridge": [
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
    ],
    # Phase 2 will add: src.ml_xgboost, src.ml_lightgbm, src.ml_pcr,
    # src.ml_random_forest, src.ml_baseline, src.dl_patchts, src.dl_ae_ridge,
    # src.tune_tree.
}

# ─── Tasks: starter no-op (a single sanity task; replace with real grid) ───
_TASKS: list[dict] = [
    {"horizon": 1, "train_window": 500, "seed": 42},
]


def total() -> int:
    return len(_TASKS)


def resolve(task_id: int) -> dict:
    return _TASKS[task_id]
