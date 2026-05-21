"""Convention-driven notebook -> package build.

Local stand-in for hpc-agent's future ``export-package`` primitive. Replaces
the Makefile's ``pipeline-export`` / ``executors-export`` / ``scripts-export``
targets and ``notebooks/_export_executor.py``.

It globs ``notebooks/{pipeline,executors,scripts}/*.ipynb``, derives each
output path by convention (strip a leading ``\\d+[a-z]?_`` ordering prefix),
auto-detects the exporter, and content-hash-caches so unchanged notebooks are
skipped. ``notebooks/audits/`` is intentionally not an export source.

  notebooks/pipeline/01_loading.ipynb     -> src/loading.py
  notebooks/pipeline/05b_dl_executor.ipynb-> src/dl_executor.py
  notebooks/executors/ml_ridge.ipynb      -> src/ml_ridge.py
  notebooks/scripts/audit_check.ipynb     -> scripts/audit_check.py

Exporter is picked per notebook: a notebook that *applies* ``@register_run``
(a runnable experiment) goes through the strict-AST template exporter that
inlines the runtime; everything else — including pipeline *library* notebooks
that merely import ``hpc_agent.template`` for the runtime seam — through the
``# export``-marker exporter (``notebooks/_exporter.py``).

When hpc-agent ships the ``export-package`` primitive, delete this file and
point the Makefile / conftest / CI at ``hpc-agent export-package``.

Usage:
    python notebooks/_build_package.py [--force]
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "notebooks"))

# Strip a leading ordering prefix: 01_, 02_, 05b_ ...
_PREFIX_RE = re.compile(r"^\d+[a-z]?_")
_CACHE = _REPO / ".hpc" / ".build-cache.json"

# (notebook subdir, output dir). audits/ is deliberately absent — those
# notebooks are executed, not exported.
_GROUPS: tuple[tuple[str, str], ...] = (
    ("pipeline", "src"),
    ("executors", "src"),
    ("scripts", "scripts"),
)


def _out_stem(nb: Path) -> str:
    """Output module stem for a notebook (ordering prefix stripped)."""
    return _PREFIX_RE.sub("", nb.stem)


# A notebook ships through the template exporter iff it *applies* @register_run
# (a runnable experiment). Detecting merely the ``hpc_agent.template`` import is
# too broad — pipeline *library* notebooks (05_executor, 07_tune_tree) import it
# for the runtime seam (current_slice / load_series) without being experiments,
# and the template exporter's strict-AST allowlist would mangle them.
_REGISTER_RUN_RE = re.compile(r"^\s*@(\w+\.)*register_run\b", re.MULTILINE)


def _is_template_notebook(nb: Path) -> bool:
    """True if a code cell applies ``@register_run`` (a template experiment)."""
    data = json.loads(nb.read_text(encoding="utf-8"))
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else src
        if _REGISTER_RUN_RE.search(text):
            return True
    return False


def _hash(nb: Path) -> str:
    return hashlib.sha256(nb.read_bytes()).hexdigest()


def _ruff_normalize(out: Path) -> None:
    """``ruff check --fix`` + ``ruff format`` *out* in place.

    The strict-AST template exporter is not guaranteed ruff-clean. No-op
    (with a warning) when ruff is unavailable, so the build still works in
    a stdlib-only environment.
    """
    try:
        subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", "--quiet", str(out)], check=False)
        subprocess.run([sys.executable, "-m", "ruff", "format", "--quiet", str(out)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[warn] ruff unavailable; {out} left un-normalized ({exc})", file=sys.stderr)


def _export(nb: Path, out: Path) -> str:
    """Export one notebook to *out*; return the exporter kind used."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if _is_template_notebook(nb):
        from hpc_agent.template import export_notebook

        export_notebook(str(nb), str(out))
        _ruff_normalize(out)
        return "template"
    from _exporter import export_notebook as marker_export

    marker_export(str(nb), str(out))
    return "marker"


def main() -> int:
    force = "--force" in sys.argv[1:]

    cache: dict[str, str] = {}
    if _CACHE.is_file() and not force:
        try:
            cache = json.loads(_CACHE.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}

    # Ensure the package is importable even before the first module lands.
    (_REPO / "src").mkdir(exist_ok=True)
    (_REPO / "src" / "__init__.py").touch()

    new_cache: dict[str, str] = {}
    built = skipped = 0
    for subdir, outdir in _GROUPS:
        nb_dir = _REPO / "notebooks" / subdir
        if not nb_dir.is_dir():
            continue
        for nb in sorted(nb_dir.glob("*.ipynb")):
            out = _REPO / outdir / f"{_out_stem(nb)}.py"
            key = str(nb.relative_to(_REPO)).replace("\\", "/")
            digest = _hash(nb)
            new_cache[key] = digest
            if cache.get(key) == digest and out.is_file():
                skipped += 1
                continue
            kind = _export(nb, out)
            rel_out = str(out.relative_to(_REPO)).replace("\\", "/")
            print(f"[{kind:8}] {key} -> {rel_out}")
            built += 1

    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE.write_text(json.dumps(new_cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"built {built}, skipped {skipped} (cache hit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
