#!/usr/bin/env python3
"""Pre-commit hook: re-export staged notebooks and stage the resulting .py.

Mirrors the Makefile mapping (pipeline -> src/, executors -> src/, scripts ->
scripts/). Fails the commit if anything in src/ or scripts/ ends up modified,
forcing the contributor to `git add` the exported file alongside the notebook.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
from _exporter import export_notebook  # type: ignore[import-not-found]

PIPELINE_MAP = {
    "01_loading": "src/loading.py",
    "02_transforms": "src/transforms.py",
    "03_evaluation": "src/evaluation.py",
    "04_scaling": "src/scaling.py",
    "05_executor": "src/executor.py",
    "05b_dl_executor": "src/dl_executor.py",
    "06_strategy_eval": "src/strategy_eval.py",
    "07_tune_tree": "src/tune_tree.py",
}


def out_path_for(nb: Path) -> Path | None:
    parent = nb.parent.name
    stem = nb.stem
    if parent == "pipeline":
        rel = PIPELINE_MAP.get(stem)
        return REPO / rel if rel else None
    if parent == "executors":
        return REPO / "src" / f"{stem}.py"
    if parent == "scripts":
        return REPO / "scripts" / f"{stem}.py"
    return None


def main(argv: list[str]) -> int:
    exported: list[Path] = []
    for arg in argv:
        nb = (REPO / arg).resolve() if not Path(arg).is_absolute() else Path(arg)
        out = out_path_for(nb)
        if out is None:
            print(f"skip (no mapping): {nb}")
            continue
        export_notebook(nb, out)
        exported.append(out)
        print(f"exported {nb.relative_to(REPO)} -> {out.relative_to(REPO)}")

    if not exported:
        return 0

    # Stage the regenerated files so the commit includes them.
    subprocess.run(["git", "add", "--", *[str(p) for p in exported]], check=True, cwd=REPO)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
