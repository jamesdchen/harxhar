"""Minimal dispatcher for `make repro` / the CI audit gate.

Equivalent to ``python -m cli src.ml_<method> ...`` but without depending on
``claude_hpc.executor_cli`` at *dispatch* time (it isn't installed on a
stdlib-only cluster). It imports the named executor module and calls its
injected ``compute(args)``.

Usage::

    python scripts/_repro_dispatch.py src.ml_ridge --output-file ... [--end 1000]

The argparse parser is **derived**, not hand-mirrored: it is built from
``hpc_agent.template.discover_runs`` over ``notebooks/executors`` — the same
source of truth ``.hpc/_build_tasks.py`` uses. An *exported* ``src/ml_*.py``
inlines the runtime and drops the ``hpc_agent.template`` import the
decorator-resolver keys off, so ``discover_runs('src')`` finds nothing; the
notebooks are scanned instead. On top of each executor's signature flags the
parser adds the framework slice flags ``--start`` / ``--end`` / ``--halo``,
which the injected ``compute`` reads off ``args`` to build the active
``SliceSpec``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NOTEBOOK_DIR = _REPO_ROOT / "notebooks" / "executors"


def _flags_by_module() -> dict[str, list]:
    """Return ``{module_path: [Flag, ...]}`` for every ``@register_run`` executor.

    Mirrors ``.hpc/_build_tasks.py``: scan ``notebooks/executors`` (the
    exported ``src/ml_*.py`` drop the ``hpc_agent.template`` import the
    AST decorator-resolver needs, so the notebook is the source of truth).
    Each module's signature flags are augmented with the framework slice
    flags ``--start`` / ``--end`` / ``--halo``.
    """
    from hpc_agent.executor_cli import flag
    from hpc_agent.template import discover_runs

    runs = discover_runs(_NOTEBOOK_DIR)
    if not runs:
        raise SystemExit(f"discover_runs found no @register_run executors under {_NOTEBOOK_DIR}")

    slice_flags = [
        flag("start", int, default=0, help="Emit-range start index (planner-set)."),
        flag("end", int, default=-1, help="Emit-range end index (-1 = to end)."),
        flag("halo", int, default=0, help="Warm-up rows replayed before the emit range (planner-set)."),
    ]

    out: dict[str, list] = {}
    for run in runs:
        module = f"src.{run.path.stem}"
        sig_names = {f.name for f in run.flags}
        extras = [f for f in slice_flags if f.name not in sig_names]
        out[module] = list(run.flags) + extras
    return out


def _add_flag(p: argparse.ArgumentParser, f: object) -> None:
    """Register one ``hpc_agent`` ``Flag`` on the argparse parser."""
    cli_name = "--" + f.name.replace("_", "-")  # type: ignore[attr-defined]
    kwargs: dict = {}
    if f.action is not None:  # type: ignore[attr-defined]
        kwargs["action"] = f.action  # type: ignore[attr-defined]
    else:
        if f.type is not None:  # type: ignore[attr-defined]
            kwargs["type"] = f.type  # type: ignore[attr-defined]
        if f.nargs is not None:  # type: ignore[attr-defined]
            kwargs["nargs"] = f.nargs  # type: ignore[attr-defined]
    if f.required:  # type: ignore[attr-defined]
        kwargs["required"] = True
    else:
        kwargs["default"] = f.default  # type: ignore[attr-defined]
    if f.choices is not None:  # type: ignore[attr-defined]
        kwargs["choices"] = f.choices  # type: ignore[attr-defined]
    if f.help:  # type: ignore[attr-defined]
        kwargs["help"] = f.help  # type: ignore[attr-defined]
    p.add_argument(cli_name, **kwargs)


def _build_parser(flags_by_module: dict[str, list]) -> argparse.ArgumentParser:
    """Build the dispatcher parser.

    ``module`` is positional; ``--output-file`` and the per-executor flags
    are the union across every discovered executor (the chosen module only
    consumes the ones it declares — argparse tolerates the unused rest at
    their defaults).
    """
    p = argparse.ArgumentParser(description="Dispatch a harxhar executor's compute(args).")
    p.add_argument("module", choices=sorted(flags_by_module), help="Executor module, e.g. src.ml_ridge")

    seen: set[str] = set()
    for flags in flags_by_module.values():
        for f in flags:
            if f.name in seen:
                continue
            seen.add(f.name)
            _add_flag(p, f)
    return p


def main() -> int:
    flags_by_module = _flags_by_module()
    args = _build_parser(flags_by_module).parse_args()
    module = importlib.import_module(args.module)
    if not hasattr(module, "compute"):
        print(f"ERROR: module {args.module} has no `compute(args)` function", file=sys.stderr)
        return 2
    module.compute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
