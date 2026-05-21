"""Export one executor notebook to ``src/<name>.py``.

A notebook that imports ``hpc_agent.template`` (a ``@register_run``
experiment) is exported by the upstream strict-AST exporter
(:func:`hpc_agent.template.export_notebook`) — self-contained output with
the hpc_agent runtime inlined, so the executor needs no ``hpc_agent``
import on the cluster. A notebook that does not is exported by the legacy
``# export``-marker exporter (:mod:`notebooks._exporter`) — this is the
path for executors not yet migrated to the template (the DL executors).

The strict-AST exporter is not guaranteed isort-/ruff-format-clean, so the
output is post-processed with ``ruff check --fix`` + ``ruff format`` to
keep the committed ``src/<name>.py`` past the repo's ``ruff check .`` /
``ruff format --check .`` lint gate and ``make export`` idempotent.

Usage:
    python notebooks/_export_executor.py <notebook.ipynb> <out.py>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "notebooks"))


def _is_template_notebook(nb_path: str) -> bool:
    """True if any code cell imports ``hpc_agent.template`` (a @register_run experiment)."""
    data = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if "hpc_agent.template" in src:
            return True
    return False


def _ruff_normalize(out: str) -> None:
    """Apply ``ruff check --fix`` + ``ruff format`` to *out* in place.

    No-op (with a warning) when ruff is unavailable, so the export still
    works in a stdlib-only environment.
    """
    try:
        subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", "--quiet", out], check=False)
        subprocess.run([sys.executable, "-m", "ruff", "format", "--quiet", out], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[warn] ruff not available; {out} left un-normalized ({exc})", file=sys.stderr)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python notebooks/_export_executor.py <notebook.ipynb> <out.py>", file=sys.stderr)
        return 2
    nb, out = sys.argv[1], sys.argv[2]

    if _is_template_notebook(nb):
        from hpc_agent.template import export_notebook

        export_notebook(nb, out)
        _ruff_normalize(out)
        print(f"[template] {nb} -> {out}")
    else:
        from _exporter import export_notebook as marker_export

        marker_export(nb, out)
        print(f"[marker]   {nb} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
