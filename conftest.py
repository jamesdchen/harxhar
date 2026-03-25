"""Root conftest — ensures repo root is on sys.path for CI and bare environments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
