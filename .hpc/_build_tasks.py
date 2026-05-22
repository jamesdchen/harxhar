"""Build-time generator for ``.hpc/tasks.py`` — planner-mode scaffold.

``tasks.py`` must import cleanly on a stdlib-only cluster runtime (no
pandas, no numpy, no ``hpc_agent.template``). Two things it would
otherwise need a heavy environment for are pre-computed *here*, at
submit/scaffold time, and baked into ``tasks.py`` as plain literals:

1. **FLAGS** — the per-executor CLI flag set. Derived by an AST walk of
   the experiment notebooks (:func:`hpc_agent.template.discover_runs`),
   which never imports torch/pandas. ``discover_runs`` keys off the
   ``@register_run`` decorator; an *exported* ``src/ml_*.py`` inlines
   the runtime and drops the ``hpc_agent.template`` import, so the
   notebook is the source of truth — we scan ``notebooks/executors``.

2. **The open-loop ``_TASKS`` literal** — the 100-chunk walk-forward
   split. Built with :func:`hpc_agent.template.plan_tasks` over the
   *OOS-length* span, then shifted into absolute X coordinates. The
   series ``DataAxis`` (a bounded-halo look-back) is the
   interview-classified axis in ``.hpc/axes.yaml`` — read here via
   :func:`load_data_axis`, not hard-coded. The series length is the
   post-feature row count, which only the pandas pipeline can produce —
   so it is probed here.

Run from the repo root after the data vintage, the HAR-lag set, or any
``run()`` signature changes::

    python .hpc/_build_tasks.py

It rewrites the ``# <build:FLAGS>`` and ``# <build:TASKS>`` regions of
``.hpc/tasks.py`` in place. ``tasks.py`` is hand-/build-maintained — it
has no owning notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hpc_agent.template.axis import DataAxis

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TASKS_PY = _REPO_ROOT / ".hpc" / "tasks.py"
_NOTEBOOK_DIR = _REPO_ROOT / "notebooks" / "executors"

# Open-loop chunking constants. The series DataAxis — a bounded look-back
# halo — is NOT hard-coded here: it is the interview-classified axis in
# .hpc/axes.yaml (executors.run.data_axis), loaded by load_data_axis().
# _TRAIN_WINDOW_DAYS is the experiment's default train window — the sweep
# point the halo expression is evaluated at — not the axis itself.
_TRAIN_WINDOW_DAYS = 500
_TOTAL_CHUNKS = 100


# ── series-length probe ──────────────────────────────────────────────────────


def probe_total_rows(data_path: str = "all30min", horizon: int = 1) -> int:
    """Compute post-transform row count for the harxhar backtest pipeline.

    Runs the same transforms as the executors (load → robust_transform →
    HAR features → calendar features → lag trim → horizon shift) and
    returns ``len(X)`` — the number of rows an executor iterates over.
    Needs the experiment's pandas/numpy environment; only ever called at
    build time, never from ``tasks.py``.
    """
    import numpy as np

    sys.path.insert(0, str(_REPO_ROOT))
    from src.loading import load_raw_data
    from src.transforms import (
        add_calendar_features,
        apply_horizon_shift,
        generate_har_features,
        resolve_har_lags,
        robust_transform,
    )

    df = load_raw_data(data_path, allow_missing=True)
    df = df.dropna(subset=["RV"]).reset_index(drop=True)
    adj_rv, _baseline = robust_transform(
        df,
        "RV",
        is_target=True,
        use_diurnal=True,
        winsor_window=240,
    )
    df["adj_RV"] = adj_rv

    df, har_names = generate_har_features(df, target_col="adj_RV")
    cal_names = add_calendar_features(df)

    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    x = df[har_names + cal_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64) if "baseline" in df else np.zeros(len(df))

    x, y, dates, baselines = apply_horizon_shift(x, y, dates, baselines, horizon)
    return len(x)


# ── FLAGS reconstruction ─────────────────────────────────────────────────────


def build_flags_source() -> str:
    """Return the ``FLAGS = {...}`` literal as Python source.

    ``discover_runs`` gives the *signature* flags only; the dispatcher
    needs the full executor CLI surface. Mirror
    :func:`hpc_agent.template.flags_for_run`: prepend ``generic_args``
    (and ``gpu_args`` for GPU runs), add the planner ``--halo`` flag,
    and let a signature flag win any name collision.
    """
    from hpc_agent.executor_cli import Flag, flag, generic_args, gpu_args
    from hpc_agent.template import discover_runs

    runs = discover_runs(_NOTEBOOK_DIR)
    if not runs:
        raise SystemExit(f"discover_runs found no @register_run executors under {_NOTEBOOK_DIR}")

    halo_flag = flag(
        "halo",
        int,
        default=0,
        help="Warm-up rows replayed before the emit range (planner-set; 0 = whole series).",
    )

    entries: list[str] = []
    for run in sorted(runs, key=lambda r: r.path.stem):
        module = f"src.{run.path.stem}"
        sig_flags = list(run.flags)
        sig_names = {f.name for f in sig_flags}
        base = [f for f in generic_args() if f.name not in sig_names]
        if run.gpu:
            base += [f for f in gpu_args() if f.name not in sig_names]
        if "halo" not in sig_names:
            base.append(halo_flag)
        full = base + sig_flags
        flag_lines = ",\n".join("        " + _flag_repr(f) for f in full)
        entries.append(f'    "{module}": [\n{flag_lines},\n    ],')

    _ = Flag  # imported for the type contract documented above
    return "FLAGS: dict[str, list] = {\n" + "\n".join(entries) + "\n}"


def _flag_repr(f: object) -> str:
    """Render one :class:`Flag` as a ``flag(...)`` constructor call."""
    from hpc_agent.executor_cli import Flag

    assert isinstance(f, Flag)
    parts = [repr(f.name)]
    if f.type is not None:
        parts.append(f.type.__name__)
    else:
        parts.append("None")
    if f.default is not None:
        parts.append(f"default={f.default!r}")
    if f.required:
        parts.append("required=True")
    if f.choices is not None:
        parts.append(f"choices={f.choices!r}")
    if f.help:
        parts.append(f"help={f.help!r}")
    if f.nargs is not None:
        parts.append(f"nargs={f.nargs!r}")
    if f.action is not None:
        parts.append(f"action={f.action!r}")
    return "flag(" + ", ".join(parts) + ")"


# ── open-loop task plan ──────────────────────────────────────────────────────


def load_data_axis() -> DataAxis:
    """Load the interview-classified series ``DataAxis`` from ``.hpc/axes.yaml``.

    The axis — a bounded look-back halo of ``train_window`` days × 48
    periods/day — is recorded in ``axes.yaml``'s ``executors.run`` block
    (all six ``ml_*`` executors share one ``@register_run def run`` axis),
    not hard-coded here. See hpc-agent's ``classify-axis`` primitive and
    :mod:`hpc_agent.template.axis_config`.
    """
    from hpc_agent.planning.axes import read_axes
    from hpc_agent.template.axis_config import data_axis_from_config

    cfg = read_axes(_REPO_ROOT)
    executors = (cfg or {}).get("executors") or {}
    if "run" not in executors:
        raise SystemExit(
            f"no classified DataAxis for run() in {_REPO_ROOT / '.hpc' / 'axes.yaml'} "
            "— record it with `hpc-agent classify-axis`"
        )
    return data_axis_from_config(executors["run"]["data_axis"])


def build_open_loop_tasks_source(total_rows: int) -> str:
    """Return the ``_OPEN_LOOP_TASKS = [...]`` literal as Python source.

    ``plan_tasks`` tiles ``[0, series_length)``; the executor can only
    emit predictions for the OOS region ``[overlap, total_rows)`` and
    every OOS chunk has a full ``overlap``-row warm-up behind it. So we
    tile the *OOS-length* span and shift each chunk into absolute X
    coordinates: ``start/end += overlap``, ``halo = overlap`` (constant —
    plan_tasks's clamped halo is wrong here because our chunked series is
    a suffix of X, not the whole of it).

    ``overlap`` is the classified halo evaluated at the default train
    window — the ``DataAxis`` comes from ``axes.yaml`` (see
    :func:`load_data_axis`), not a hard-coded constant.
    """
    from hpc_agent.template import BoundedHalo, plan_tasks

    data_axis = load_data_axis()
    if not isinstance(data_axis, BoundedHalo):
        raise SystemExit(
            "open-loop chunking expects a bounded_halo DataAxis; "
            f".hpc/axes.yaml classifies run() as {type(data_axis).__name__}"
        )
    overlap = data_axis.halo_fn({"train_window": _TRAIN_WINDOW_DAYS})

    plan = plan_tasks(
        [{"train_window": _TRAIN_WINDOW_DAYS}],
        data_axis,
        chunks=_TOTAL_CHUNKS,
        series_length=total_rows - overlap,
    )

    lines: list[str] = []
    for i in range(plan.total()):
        t = plan.resolve(i)
        start = t["start"] + overlap
        end = t["end"] + overlap
        lines.append(f'    {{"start": {start}, "end": {end}, "halo": {overlap}}},')

    return "_OPEN_LOOP_TASKS: list[dict] = [\n" + "\n".join(lines) + "\n]"


# ── region rewrite ───────────────────────────────────────────────────────────


def _replace_region(text: str, tag: str, body: str) -> str:
    """Replace the source between ``# <build:TAG>`` and ``# </build:TAG>``."""
    open_marker = f"# <build:{tag}>"
    close_marker = f"# </build:{tag}>"
    i = text.index(open_marker)
    j = text.index(close_marker)
    head = text[: i + len(open_marker)]
    tail = text[j:]
    return f"{head}\n{body}\n{tail}"


def main() -> None:
    total_rows = probe_total_rows()
    print(f"[build_tasks] total_rows = {total_rows}")

    flags_src = build_flags_source()
    tasks_src = build_open_loop_tasks_source(total_rows)

    text = _TASKS_PY.read_text(encoding="utf-8")
    text = _replace_region(text, "FLAGS", flags_src)
    text = _replace_region(text, "TASKS", tasks_src)
    _TASKS_PY.write_text(text, encoding="utf-8")
    print(f"[build_tasks] rewrote FLAGS + _OPEN_LOOP_TASKS in {_TASKS_PY}")

    # The generated FLAGS literal emits one-line ``flag(...)`` calls that
    # can exceed the line-length limit; normalise the whole file so the
    # baked output always satisfies ``ruff format --check``.
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "ruff", "format", str(_TASKS_PY)],
        check=True,
        cwd=_REPO_ROOT,
    )
    print(f"[build_tasks] ruff-formatted {_TASKS_PY}")


if __name__ == "__main__":
    main()
