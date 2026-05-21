"""harxhar tasks declaration — the ``total()`` / ``resolve()`` contract.

Per-executor ``FLAGS`` declares each executor's CLI shape; the matching
``.hpc/cli.py`` dispatcher reads it at runtime and parses argv for
whichever executor module is invoked. ``resolve(task_id)`` / ``total()``
control task fan-out per ``/submit-hpc`` submission.

**Generated, not hand-edited (the two ``# <build:...>`` regions).**
``FLAGS`` and the open-loop ``_OPEN_LOOP_TASKS`` literal are baked by
``.hpc/_build_tasks.py`` — run it after a data-vintage, HAR-lag, or
``run()``-signature change. Baking keeps *this* module stdlib-only: it
imports no pandas/numpy and no ``hpc_agent.template`` planner, so it is
safe to import on a stdlib-only cluster runtime. The only third-party
import is ``flag`` (the ``Flag`` constructor) for the ``FLAGS`` values.
"""

from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path as _Path

# ``flag`` builds the per-executor CLI ``Flag`` specs in FLAGS. On the
# cluster the deployed runtime is ``claude_hpc``; locally / in CI it is
# ``hpc_agent``. Both ship an identical, stdlib-only ``executor_cli``.
try:  # pragma: no cover - import shim, exercised by whichever env is live
    from claude_hpc.executor_cli import flag
except ModuleNotFoundError:  # pragma: no cover
    from hpc_agent.executor_cli import flag

# ─── FLAGS ────────────────────────────────────────────────────────────────
#
# Per-executor CLI flag set. BAKED by .hpc/_build_tasks.py from an AST
# walk of notebooks/executors (hpc_agent.template.discover_runs) — do
# not hand-edit between the build markers. Each list is the executor's
# full CLI surface: generic_args() (minus signature collisions) + the
# planner --halo flag + the run() signature flags.
# <build:FLAGS>
FLAGS: dict[str, list] = {
    "src.ml_baseline": [
        flag("seed", int, default=42, help="Random seed for deterministic reproduction."),
        flag("start", int, default=0, help="Window start index/offset (interpretation is executor-specific)."),
        flag("end", int, default=-1, help="Window end index/offset (-1 = to end)."),
        flag(
            "halo", int, default=0, help="Warm-up rows replayed before the emit range (planner-set; 0 = whole series)."
        ),
        flag("horizon", int, default=1),
        flag("train_window", int, default=500),
        flag("data_path", str, default="all30min"),
        flag("output_file", str, default="results/baseline/run.json"),
    ],
    "src.ml_lightgbm": [
        flag("start", int, default=0, help="Window start index/offset (interpretation is executor-specific)."),
        flag("end", int, default=-1, help="Window end index/offset (-1 = to end)."),
        flag(
            "halo", int, default=0, help="Warm-up rows replayed before the emit range (planner-set; 0 = whole series)."
        ),
        flag("horizon", int, default=1),
        flag("train_window", int, default=500),
        flag("refit_frequency", int),
        flag("exog_cols", str, default=""),
        flag("seed", int, default=42),
        flag("data_path", str, default="all30min"),
        flag("output_file", str, default="results/lgbm/run.json"),
        flag("params_file", str, default=""),
    ],
    "src.ml_pcr": [
        flag("start", int, default=0, help="Window start index/offset (interpretation is executor-specific)."),
        flag("end", int, default=-1, help="Window end index/offset (-1 = to end)."),
        flag(
            "halo", int, default=0, help="Warm-up rows replayed before the emit range (planner-set; 0 = whole series)."
        ),
        flag("horizon", int, default=1),
        flag("train_window", int, default=500),
        flag("n_components", int, default=5),
        flag("exog_cols", str, default=""),
        flag("seed", int, default=42),
        flag("data_path", str, default="all30min"),
        flag("output_file", str, default="results/pcr/run.json"),
    ],
    "src.ml_random_forest": [
        flag("start", int, default=0, help="Window start index/offset (interpretation is executor-specific)."),
        flag("end", int, default=-1, help="Window end index/offset (-1 = to end)."),
        flag(
            "halo", int, default=0, help="Warm-up rows replayed before the emit range (planner-set; 0 = whole series)."
        ),
        flag("horizon", int, default=1),
        flag("train_window", int, default=500),
        flag("refit_frequency", int),
        flag("exog_cols", str, default=""),
        flag("seed", int, default=42),
        flag("data_path", str, default="all30min"),
        flag("output_file", str, default="results/rf/run.json"),
        flag("params_file", str, default=""),
    ],
    "src.ml_ridge": [
        flag("start", int, default=0, help="Window start index/offset (interpretation is executor-specific)."),
        flag("end", int, default=-1, help="Window end index/offset (-1 = to end)."),
        flag(
            "halo", int, default=0, help="Warm-up rows replayed before the emit range (planner-set; 0 = whole series)."
        ),
        flag("horizon", int, default=1),
        flag("train_window", int, default=500),
        flag("refit_frequency", int),
        flag("exog_cols", str, default=""),
        flag("segment", str, default=""),
        flag("lag_scope", str, default="global"),
        flag("alpha", float, default=1.0),
        flag("seed", int, default=42),
        flag("data_path", str, default="all30min"),
        flag("output_file", str, default="results/ridge/run.json"),
        flag("params_file", str, default=""),
    ],
    "src.ml_xgboost": [
        flag("start", int, default=0, help="Window start index/offset (interpretation is executor-specific)."),
        flag("end", int, default=-1, help="Window end index/offset (-1 = to end)."),
        flag(
            "halo", int, default=0, help="Warm-up rows replayed before the emit range (planner-set; 0 = whole series)."
        ),
        flag("horizon", int, default=1),
        flag("train_window", int, default=500),
        flag("refit_frequency", int),
        flag("exog_cols", str, default=""),
        flag("seed", int, default=42),
        flag("data_path", str, default="all30min"),
        flag("output_file", str, default="results/xgb/run.json"),
        flag("params_file", str, default=""),
    ],
}
# </build:FLAGS>

# ─── Tasks ────────────────────────────────────────────────────────────────
#
# Open-loop default: 100-chunk walk-forward split. Used when /submit-hpc
# is invoked without --campaign-id (HPC_CAMPAIGN_ID unset).
#
# Closed-loop (campaign): when HPC_CAMPAIGN_ID is set, this module asks
# Optuna for a batch of trials per /submit-hpc iteration and
# materializes one JSON params file per trial under
# `params/<cid>/iter_<N>/`. resolve(task_id) returns the path so the
# executor receives `--params-file params/.../trial_K.json`. The
# campaign driver (.hpc/campaigns/<cid>/score_iter.py) reads each
# iteration's manifest.json and the per-task qlike.json after the array
# job lands, then calls study.tell() to push results into the study.
_CAMPAIGN_ID = _os.environ.get("HPC_CAMPAIGN_ID")
_TUNE_BATCH = 10  # legacy: trials per /submit-hpc iteration for xgb_optuna_2026_05
_TUNE_BUDGET = 100  # legacy
_OPTUNA_STORAGE = ".hpc/optuna.db"
_TUNE_MODEL = "xgb"  # legacy

# tune_<model>_<bucket> campaign defaults — sequential trials (K=1 per iteration)
# so Optuna's TPE sees each completed trial's QLIKE before suggesting the next.
# H2 throttles parallelism at ~100 concurrent slots/user anyway, so batched
# trials run mostly sequentially in wall-clock terms but waste TPE feedback.
_TUNE_BATCH_CHUNKED = 1  # trials per iteration
_TUNE_BUDGET_CHUNKED = 10  # total trials per (model, bucket) campaign → 10 iterations

# Bucket columns mirrored from src.loading.SUBGROUPS (cannot import — framework env lacks pandas).
# Used by tune_<model>_<bucket> campaigns to set --exog-cols per task.
_MOMENTS = ("sumret", "sumabsret", "sumret3", "sumret4", "sumpret2", "sumbipow", "sumautocov")
_LIQUIDITY = (
    "sumvolume",
    "numobs",
    "turnover_ewstock",
    "buyturnover_ewstock",
    "sellturnover_ewstock",
    "effspread_ewstock",
    "spread_ewstock",
    "turnover_vwstock",
    "buyturnover_vwstock",
    "sellturnover_vwstock",
    "effspread_vwstock",
    "spread_vwstock",
)
_MARKET_EW = (
    "sumret2_ewstock",
    "sumret3_ewstock",
    "sumret4_ewstock",
    "sumabsret_ewstock",
    "sumbipow_ewstock",
    "sumpret2_ewstock",
)
_MARKET_VW = (
    "sumret2_vwstock",
    "sumret3_vwstock",
    "sumret4_vwstock",
    "sumabsret_vwstock",
    "sumbipow_vwstock",
    "sumpret2_vwstock",
)
_VOL_DEMAND = (
    "voldemand_spx_open_and_close",
    "voldemand_spx_open_only",
    "voldemand_all_open_and_close",
    "voldemand_all_open_only",
)


# ─── Open-loop backtest chunking ──────────────────────────────────────────
#
# When HPC_CAMPAIGN_ID is unset, the walk-forward backtest is split into
# 100 chunks. Each task carries a SliceSpec triple — `start`, `end`,
# `halo` (the fields hpc_agent.template's SliceSpec / executor.py's
# `_backtest_and_save` consume): `[start, end)` is the OOS emit range
# and `halo` is the warm-up prefix replayed before it, so the executor
# processes X[start - halo : end] and trains on the first `halo` rows
# without emitting predictions. Bucket and model are NOT axes here —
# they're baked into per-(model, bucket) run sidecars at submit time, so
# 18 array submissions (3 models × 6 buckets) all share this tasks.py.
#
# BAKED by .hpc/_build_tasks.py: it probes the post-feature series
# length (only the pandas pipeline can), runs hpc_agent.template's
# plan_tasks over the OOS-length span, and shifts each chunk into
# absolute X coordinates (start/end += overlap, halo = overlap). The
# halo is a constant 24000 (= default --train-window 500 ×
# PERIODS_PER_DAY 48). Re-run the builder if the data vintage or
# HAR-lag set changes. Do not hand-edit between the build markers.
# <build:TASKS>
_OPEN_LOOP_TASKS: list[dict] = [
    {"start": 24000, "end": 26190, "halo": 24000},
    {"start": 26190, "end": 28380, "halo": 24000},
    {"start": 28380, "end": 30570, "halo": 24000},
    {"start": 30570, "end": 32760, "halo": 24000},
    {"start": 32760, "end": 34950, "halo": 24000},
    {"start": 34950, "end": 37140, "halo": 24000},
    {"start": 37140, "end": 39330, "halo": 24000},
    {"start": 39330, "end": 41520, "halo": 24000},
    {"start": 41520, "end": 43710, "halo": 24000},
    {"start": 43710, "end": 45900, "halo": 24000},
    {"start": 45900, "end": 48090, "halo": 24000},
    {"start": 48090, "end": 50280, "halo": 24000},
    {"start": 50280, "end": 52470, "halo": 24000},
    {"start": 52470, "end": 54660, "halo": 24000},
    {"start": 54660, "end": 56850, "halo": 24000},
    {"start": 56850, "end": 59040, "halo": 24000},
    {"start": 59040, "end": 61230, "halo": 24000},
    {"start": 61230, "end": 63420, "halo": 24000},
    {"start": 63420, "end": 65610, "halo": 24000},
    {"start": 65610, "end": 67800, "halo": 24000},
    {"start": 67800, "end": 69990, "halo": 24000},
    {"start": 69990, "end": 72180, "halo": 24000},
    {"start": 72180, "end": 74370, "halo": 24000},
    {"start": 74370, "end": 76560, "halo": 24000},
    {"start": 76560, "end": 78750, "halo": 24000},
    {"start": 78750, "end": 80940, "halo": 24000},
    {"start": 80940, "end": 83130, "halo": 24000},
    {"start": 83130, "end": 85320, "halo": 24000},
    {"start": 85320, "end": 87510, "halo": 24000},
    {"start": 87510, "end": 89700, "halo": 24000},
    {"start": 89700, "end": 91890, "halo": 24000},
    {"start": 91890, "end": 94080, "halo": 24000},
    {"start": 94080, "end": 96270, "halo": 24000},
    {"start": 96270, "end": 98460, "halo": 24000},
    {"start": 98460, "end": 100649, "halo": 24000},
    {"start": 100649, "end": 102838, "halo": 24000},
    {"start": 102838, "end": 105027, "halo": 24000},
    {"start": 105027, "end": 107216, "halo": 24000},
    {"start": 107216, "end": 109405, "halo": 24000},
    {"start": 109405, "end": 111594, "halo": 24000},
    {"start": 111594, "end": 113783, "halo": 24000},
    {"start": 113783, "end": 115972, "halo": 24000},
    {"start": 115972, "end": 118161, "halo": 24000},
    {"start": 118161, "end": 120350, "halo": 24000},
    {"start": 120350, "end": 122539, "halo": 24000},
    {"start": 122539, "end": 124728, "halo": 24000},
    {"start": 124728, "end": 126917, "halo": 24000},
    {"start": 126917, "end": 129106, "halo": 24000},
    {"start": 129106, "end": 131295, "halo": 24000},
    {"start": 131295, "end": 133484, "halo": 24000},
    {"start": 133484, "end": 135673, "halo": 24000},
    {"start": 135673, "end": 137862, "halo": 24000},
    {"start": 137862, "end": 140051, "halo": 24000},
    {"start": 140051, "end": 142240, "halo": 24000},
    {"start": 142240, "end": 144429, "halo": 24000},
    {"start": 144429, "end": 146618, "halo": 24000},
    {"start": 146618, "end": 148807, "halo": 24000},
    {"start": 148807, "end": 150996, "halo": 24000},
    {"start": 150996, "end": 153185, "halo": 24000},
    {"start": 153185, "end": 155374, "halo": 24000},
    {"start": 155374, "end": 157563, "halo": 24000},
    {"start": 157563, "end": 159752, "halo": 24000},
    {"start": 159752, "end": 161941, "halo": 24000},
    {"start": 161941, "end": 164130, "halo": 24000},
    {"start": 164130, "end": 166319, "halo": 24000},
    {"start": 166319, "end": 168508, "halo": 24000},
    {"start": 168508, "end": 170697, "halo": 24000},
    {"start": 170697, "end": 172886, "halo": 24000},
    {"start": 172886, "end": 175075, "halo": 24000},
    {"start": 175075, "end": 177264, "halo": 24000},
    {"start": 177264, "end": 179453, "halo": 24000},
    {"start": 179453, "end": 181642, "halo": 24000},
    {"start": 181642, "end": 183831, "halo": 24000},
    {"start": 183831, "end": 186020, "halo": 24000},
    {"start": 186020, "end": 188209, "halo": 24000},
    {"start": 188209, "end": 190398, "halo": 24000},
    {"start": 190398, "end": 192587, "halo": 24000},
    {"start": 192587, "end": 194776, "halo": 24000},
    {"start": 194776, "end": 196965, "halo": 24000},
    {"start": 196965, "end": 199154, "halo": 24000},
    {"start": 199154, "end": 201343, "halo": 24000},
    {"start": 201343, "end": 203532, "halo": 24000},
    {"start": 203532, "end": 205721, "halo": 24000},
    {"start": 205721, "end": 207910, "halo": 24000},
    {"start": 207910, "end": 210099, "halo": 24000},
    {"start": 210099, "end": 212288, "halo": 24000},
    {"start": 212288, "end": 214477, "halo": 24000},
    {"start": 214477, "end": 216666, "halo": 24000},
    {"start": 216666, "end": 218855, "halo": 24000},
    {"start": 218855, "end": 221044, "halo": 24000},
    {"start": 221044, "end": 223233, "halo": 24000},
    {"start": 223233, "end": 225422, "halo": 24000},
    {"start": 225422, "end": 227611, "halo": 24000},
    {"start": 227611, "end": 229800, "halo": 24000},
    {"start": 229800, "end": 231989, "halo": 24000},
    {"start": 231989, "end": 234178, "halo": 24000},
    {"start": 234178, "end": 236367, "halo": 24000},
    {"start": 236367, "end": 238556, "halo": 24000},
    {"start": 238556, "end": 240745, "halo": 24000},
    {"start": 240745, "end": 242934, "halo": 24000},
]
# </build:TASKS>


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


def _parse_tune_campaign(cid: str | None) -> tuple[str, str] | None:
    """Map ``tune_<model>_<bucket>`` -> (model, bucket); else None.

    Bucket strings can themselves contain underscores (``market_ew``,
    ``vol_demand``), so the parse pivots on a known model prefix rather
    than a naive split.
    """
    if not cid or not cid.startswith("tune_"):
        return None
    rest = cid[len("tune_") :]
    for m in ("xgb", "lgbm", "ridge", "rf", "pcr"):
        if rest.startswith(m + "_"):
            return m, rest[len(m) + 1 :]
    return None


_BUCKET_COLS_LOOKUP: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "moments": _MOMENTS,
    "liquidity": _LIQUIDITY,
    "market_ew": _MARKET_EW,
    "market_vw": _MARKET_VW,
    "vol_demand": _VOL_DEMAND,
}


def _resolve_iter_idx() -> int:
    """Pick the iteration index for this `tune_*` invocation.

    Cluster-side: ``$HPC_ITER_IDX`` is set by the submitter (one value per
    qsub/sbatch), so every task in the array sees the same iter and reads
    the matching pre-materialized ``params/<cid>/iter_<N>/`` directory.
    Importing ``claude_hpc.mapreduce.reduce.history.prior`` is unnecessary
    (and would crash — that submodule isn't shipped by ``deploy_runtime``).

    Local submit-time: ``HPC_ITER_IDX`` is unset, so count existing
    on-disk ``iter_*`` dirs to pick the next one. No claude_hpc deps.
    """
    explicit = _os.environ.get("HPC_ITER_IDX")
    if explicit is not None and explicit != "":
        return int(explicit)
    base = _Path(f"params/{_CAMPAIGN_ID}")
    if not base.exists():
        return 0
    # max(existing iter)+1 — robust to gaps from manual cleanup. Counting
    # would silently collide when an intermediate iter_N has been removed.
    indices = [
        int(d.name[len("iter_") :])
        for d in base.iterdir()
        if d.is_dir() and d.name.startswith("iter_") and d.name[len("iter_") :].isdigit()
    ]
    return max(indices) + 1 if indices else 0


def _build_chunked_tune_batch(model: str, bucket: str) -> list[dict]:
    """K trials × open-loop chunks for one tune_<model>_<bucket> iteration."""
    n_iter = _resolve_iter_idx()
    n_done_trials = n_iter * _TUNE_BATCH_CHUNKED
    if n_done_trials >= _TUNE_BUDGET_CHUNKED:
        return []
    n_this = min(_TUNE_BATCH_CHUNKED, _TUNE_BUDGET_CHUNKED - n_done_trials)
    iter_dir = _Path(f"params/{_CAMPAIGN_ID}/iter_{n_iter:03d}")
    manifest_file = iter_dir / "manifest.json"

    if not manifest_file.exists():
        # Submit-time path only — asks Optuna for trials, writes manifest.
        # On-cluster path always sees an existing manifest (pushed before submit).
        from src.tune_tree import _get_search_space, _load_or_create_study, _study_name

        iter_dir.mkdir(parents=True, exist_ok=True)
        study_name = _study_name(model, bucket)
        study = _load_or_create_study(model, storage_path=_OPTUNA_STORAGE, study_name=study_name)
        trials_info = []
        for i in range(n_this):
            trial = study.ask(fixed_distributions=_get_search_space(model))
            (iter_dir / f"trial_{i}.json").write_text(_json.dumps(trial.params, indent=2))
            trials_info.append({"id": i, "file": f"trial_{i}.json", "optuna_number": trial.number})
        manifest_file.write_text(
            _json.dumps(
                {
                    "model": model,
                    "bucket": bucket,
                    "study_name": study_name,
                    "batch_size": n_this,
                    "trials": trials_info,
                },
                indent=2,
            )
        )

    bucket_cols = _BUCKET_COLS_LOOKUP[bucket]
    exog_str = "|".join(bucket_cols)
    out: list[dict] = []
    for trial_idx in range(n_this):
        params_file = (iter_dir / f"trial_{trial_idx}.json").as_posix()
        for chunk_id, chunk in enumerate(_OPEN_LOOP_TASKS):
            out.append(
                {
                    "params_file": params_file,
                    "exog_cols": exog_str,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "halo": chunk["halo"],
                    "trial_idx": trial_idx,
                    "chunk_id": chunk_id,
                    "iter_idx": n_iter,
                }
            )
    return out


_tune_parsed = _parse_tune_campaign(_CAMPAIGN_ID)
if _tune_parsed is not None:
    _TASKS: list[dict] = _build_chunked_tune_batch(*_tune_parsed)
elif _CAMPAIGN_ID and _CAMPAIGN_ID.startswith("xgb_optuna"):
    _TASKS = _build_xgb_optuna_batch()  # legacy xgb_optuna_2026_05 path
else:
    # CAMPAIGN_ID unset OR a pure tracking tag (e.g. exog_buckets_full) — same chunk axis.
    _TASKS = list(_OPEN_LOOP_TASKS)


def total() -> int:
    return len(_TASKS)


def resolve(task_id: int) -> dict:
    return dict(_TASKS[task_id])
