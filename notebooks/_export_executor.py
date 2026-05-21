"""Export one executor notebook to ``src/<name>.py``.

Bridges the incremental ``@register_run`` migration (Phase B): a notebook
that carries a ``@register_run`` entrypoint is exported by the strict-AST
template exporter (:func:`src._template.export_notebook`); a notebook that
does not is exported by the legacy ``# export``-marker exporter
(:mod:`notebooks._exporter`). Once every executor notebook is converted the
fallback can be dropped and ``make executors-export`` can call the template
exporter directly.

The strict-AST exporter emits one blank line between every top-level node,
which is neither ``isort``- nor ``ruff-format``-clean. The export is post-
processed with ``ruff check --fix`` + ``ruff format`` so the committed
``src/<name>.py`` survives the repo's ``ruff check .`` / ``ruff format
--check .`` lint gate and ``make export`` stays idempotent.

Usage:
    python notebooks/_export_executor.py <notebook.ipynb> <out.py>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "notebooks"))


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

    from src._template import export_notebook as template_export

    try:
        template_export(nb, out)
        _ruff_normalize(out)
        print(f"[template] {nb} -> {out}")
    except ValueError:
        # No @register_run entrypoint — notebook not yet migrated.
        from _exporter import export_notebook as marker_export

        marker_export(nb, out)
        print(f"[marker]   {nb} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
