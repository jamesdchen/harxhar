"""Export `# export` cells from a notebook to a .py file. Stdlib only."""
from __future__ import annotations

import json
from pathlib import Path

EXPORT_MARKER = "# export"
AUTOGEN_HEADER = "# Auto-generated from notebooks/{src}. Do not edit by hand.\n"


def export_notebook(nb_path: str | Path, out_path: str | Path) -> Path:
    """Concatenate every `# export`-marked cell of ``nb_path`` into ``out_path``.

    A cell ships iff its first non-blank line, stripped, starts with
    ``# export``. The marker line is dropped; everything after it is
    written verbatim. Cells are concatenated in notebook order with one
    blank line between them.
    """
    nb = json.loads(Path(nb_path).read_text())
    chunks: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        first = next((ln for ln in source.splitlines() if ln.strip()), "")
        if not first.strip().startswith(EXPORT_MARKER):
            continue
        body = source.split("\n", 1)[1] if "\n" in source else ""
        chunks.append(body.rstrip() + "\n")

    rel_name = Path(nb_path).name
    header = AUTOGEN_HEADER.format(src=rel_name)
    body = "\n\n".join(chunks)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n" + body)
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python _exporter.py <notebook.ipynb> <out.py>", file=sys.stderr)
        sys.exit(2)
    dest = export_notebook(sys.argv[1], sys.argv[2])
    print(f"wrote {dest}")
