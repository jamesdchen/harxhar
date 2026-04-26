"""Assemble notebooks/pipeline/06_strategy_eval.ipynb from staging files.

Reads Python source fragments from _strategy_staging/ and a markdown JSON
manifest, then emits a single nbformat 4.5 .ipynb file that interleaves
markdown and code cells in the order specified by the task.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT: Path = Path(r"C:\Users\james\CC Allowed\harxhar")
STAGING_DIR: Path = REPO_ROOT / "_strategy_staging"
MARKDOWN_JSON: Path = STAGING_DIR / "07_markdown.json"
OUTPUT_NB: Path = REPO_ROOT / "notebooks" / "pipeline" / "06_strategy_eval.ipynb"

# (staging filename, after_section key) pairs in cell order. None means the
# leading markdown cell that has after_section: null.
SECTIONS: list[tuple[str, str]] = [
    ("01_module_header.py", "module_header"),
    ("02_bs_gamma.py", "bs_gamma"),
    ("03_trade_date_and_underlying.py", "trade_date_and_underlying"),
    ("04_filter.py", "filter"),
    ("05_pnl.py", "pnl"),
    ("06_smoke_tests.py", "smoke_tests"),
]


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _to_source_lines(text: str) -> list[str]:
    """Split text into nbformat-style source lines.

    Each line keeps its trailing newline except possibly the last. An empty
    string yields an empty list (nbformat accepts this).
    """
    if text == "":
        return []
    lines = text.splitlines(keepends=True)
    return lines


def _md_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": _new_id(),
        "metadata": {},
        "source": _to_source_lines(text),
    }


def _code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": _new_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _to_source_lines(text),
    }


def _load_markdown_map(path: Path) -> dict[Any, str]:
    """Return mapping {after_section_value: content}.

    Uses None as the key for the leading markdown cell (after_section: null).
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print(f"ERROR: {path} must be a JSON list, got {type(raw).__name__}")
        sys.exit(1)

    mapping: dict[Any, str] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            print(f"ERROR: {path} entries must be objects, got {type(entry).__name__}")
            sys.exit(1)
        if "after_section" not in entry or "content" not in entry:
            print(f"ERROR: {path} entry missing required keys (need 'after_section' and 'content'): {entry!r}")
            sys.exit(1)
        mapping[entry["after_section"]] = entry["content"]
    return mapping


def _require_md(mapping: dict[Any, str], key: Any, path: Path) -> str:
    if key not in mapping:
        printable = "null" if key is None else repr(key)
        print(f"ERROR: {path} is missing markdown entry for after_section={printable}")
        sys.exit(1)
    return mapping[key]


def main() -> None:
    # Verify staging files exist.
    missing: list[Path] = []
    for fname, _ in SECTIONS:
        p = STAGING_DIR / fname
        if not p.is_file():
            missing.append(p)
    if not MARKDOWN_JSON.is_file():
        print(f"ERROR: missing markdown manifest: {MARKDOWN_JSON}")
        sys.exit(1)
    if missing:
        for p in missing:
            print(f"ERROR: missing staging file: {p}")
        sys.exit(1)

    md_map = _load_markdown_map(MARKDOWN_JSON)

    cells: list[dict[str, Any]] = []

    # 1. Leading markdown cell (after_section: null).
    cells.append(_md_cell(_require_md(md_map, None, MARKDOWN_JSON)))

    # 2..13. For each section: md cell, then code cell with file contents.
    for fname, section_key in SECTIONS:
        md_text = _require_md(md_map, section_key, MARKDOWN_JSON)
        cells.append(_md_cell(md_text))
        code_text = (STAGING_DIR / fname).read_text(encoding="utf-8")
        cells.append(_code_cell(code_text))

    # 14. Final code cell that invokes run_all_smoke().
    cells.append(_code_cell("run_all_smoke()\n"))

    notebook: dict[str, Any] = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUTPUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NB.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    total_lines = sum(len(c["source"]) for c in cells)
    print(f"wrote {OUTPUT_NB}, {len(cells)} cells, {total_lines} total source lines")


if __name__ == "__main__":
    main()
