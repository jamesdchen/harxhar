#!/usr/bin/env python3
"""Scaffold a new experiment notebook: notebooks/<name>.ipynb.

The skeleton has one @register_run-decorated `run()` plus a smoke-test cell.
The experimenter fills in run(); `make export` turns it into src/<name>.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _code(*lines: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(lines)}


def _md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def scaffold(name: str) -> int:
    nb_path = REPO / "notebooks" / f"{name}.ipynb"
    if nb_path.exists():
        print(f"refusing to overwrite existing {nb_path.relative_to(REPO)}", file=sys.stderr)
        return 1

    cells = [
        _md(
            f"# Experiment: {name}\n",
            "\n",
            "Fill in `run()`. Smoke-test in the cells below — only structural\n",
            "definitions (imports, defs, classes, UPPERCASE constants) are exported.",
        ),
        _code("# imports\n", "from hpc_agent.template import register_run, save_artifact, load_series\n"),
        _code(
            "@register_run\n",
            "def run(seed: int = 42) -> dict:\n",
            '    """One experiment run. Returns a JSON-serializable metrics dict."""\n',
            "    # series = load_series('your_data.parquet')\n",
            "    # ... your experiment ...\n",
            "    # save_artifact('predictions.parquet', preds_df)  # for large outputs\n",
            '    return {"metric": 0.0}\n',
        ),
        _md("Smoke test — not exported (not a structural definition)."),
        _code("run(seed=0)\n"),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"created {nb_path.relative_to(REPO)}")

    data_dir = REPO / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / ".gitkeep").touch()
    print(f"  drop your dataset into {data_dir.relative_to(REPO)}/ and reference it from load_series()")
    print(f"  then: make export   (notebooks/{name}.ipynb -> src/{name}.py)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/_scaffold_experiment.py <name>", file=sys.stderr)
        sys.exit(2)
    sys.exit(scaffold(sys.argv[1]))
