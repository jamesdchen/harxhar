"""Root conftest — repo root on sys.path; builds the generated src/ package on demand.

src/ is a build artifact generated from notebooks/ (not committed — see
.gitignore). If it is absent — a fresh clone, a bare `pytest`, an IDE run —
build it before collection so imports resolve with no manual `make export`.
`make export` handles staleness (content-hash cache); this hook only covers
the absent case.

notebooks/_build_package.py is the local convention-driven stand-in for
hpc-agent's `export-package` primitive.
"""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

if not (_ROOT / "src" / "ml_ridge.py").exists():
    subprocess.run([sys.executable, str(_ROOT / "notebooks" / "_build_package.py")], check=True)
