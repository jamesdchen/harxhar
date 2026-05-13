"""Minimal stdlib-only dispatcher for `make repro` / CI audit gate.

Equivalent to ``python -m cli src.ml_<method> ...`` but without depending on
``claude_hpc.executor_cli`` (which isn't installed in CI). Recreates the CLI
flag set in ``.hpc/tasks.py`` FLAGS using vanilla argparse, imports the
named executor module, and calls its ``compute(args)``.

Usage:
    python scripts/_repro_dispatch.py src.ml_ridge --output-file ... [--n-components N]

The flag list mirrors ``_CPU_BASE`` from ``.hpc/tasks.py`` plus the PCR-only
``--n-components`` extra. If FLAGS in tasks.py change, update this script.
"""

from __future__ import annotations

import argparse
import importlib
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("module", help="Importable module path, e.g. src.ml_ridge")
    # Mirrors `_CPU_BASE` in .hpc/tasks.py.
    p.add_argument("--output-file", required=True)
    p.add_argument("--data-path", default="all30min")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--train-window", type=int, default=500, help="training window in days")
    p.add_argument("--refit-frequency", type=int, default=None)
    p.add_argument("--exog-cols", default=None, help="pipe-separated exog column names")
    p.add_argument("--params-file", default=None)
    p.add_argument(
        "--segment",
        default=None,
        choices=("all", "morning", "midday", "closing", "overnight"),
    )
    p.add_argument("--lag-scope", default="global", choices=("global", "intra"))
    p.add_argument("--seed", type=int, default=42)
    # PCR-only extras.
    p.add_argument("--n-components", type=int, default=5)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    module = importlib.import_module(args.module)
    if not hasattr(module, "compute"):
        print(f"ERROR: module {args.module} has no `compute(args)` function", file=sys.stderr)
        return 2
    module.compute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
