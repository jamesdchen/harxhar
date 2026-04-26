# Auto-generated from notebooks/audit_check.ipynb. Do not edit by hand.

"""Fast, comprehensive audit-readiness gate for the harxhar repo.

Run this first to confirm the repo is in a coherent state. Read-only.

Checks (in order; first failure halts):
    1. Manifest validity (results/MANIFEST.json schema + results_dir existence)
    2. Master table coherence (row count + column schema; --quick skips re-build diff)
    3. Diagnostics integrity (7 files per bundle + mz_stats.json keys + recompute match)
    4. src.evaluation exports (10 expected symbols importable)
    5. plot_mz_scatter draws both MZ fit and 45° reference (literal-string check)
    6. DL seeding pinned (seed_everything + --seed in dl_patchts and dl_ae_ridge)
    7. README invariants documented (## Key Invariants section + 4 keywords)

Usage:
    python scripts/audit_check.py            # run all checks
    python scripts/audit_check.py --quick    # skip slow re-build verification
    python scripts/audit_check.py --verbose  # detailed per-check output
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ----- output helpers ---------------------------------------------------------

VERBOSE = False


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _v(msg: str) -> None:
    if VERBOSE:
        print(f"       {msg}")


# ----- check 1: manifest validity ---------------------------------------------


def check_manifest() -> tuple[bool, dict | None]:
    path = REPO / "results" / "MANIFEST.json"
    if not path.exists():
        _fail(f"results/MANIFEST.json missing at {path}")
        return False, None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _fail(f"MANIFEST.json failed to parse: {e}")
        return False, None

    for key in ("version", "entries", "notes"):
        if key not in manifest:
            _fail(f"MANIFEST.json missing top-level key: {key!r}")
            return False, None

    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        _fail("MANIFEST.json 'entries' is not a non-empty list")
        return False, None

    required_entry_keys = ("method", "feature_set", "config", "results_dir")
    for i, entry in enumerate(entries):
        for k in required_entry_keys:
            if k not in entry:
                _fail(f"entry {i} missing key {k!r}: {entry}")
                return False, None
        rdir = REPO / entry["results_dir"]
        if not rdir.exists():
            _fail(f"entry {i} references missing results_dir: {rdir}")
            return False, None
        _v(f"entry {i}: {entry['method']}/{entry['feature_set']}/{entry['config']} -> {entry['results_dir']}")

    _ok(f"Manifest valid ({len(entries)} entries, {len(manifest['notes'])} notes)")
    return True, manifest


# ----- check 2: master table coherence ----------------------------------------


def check_master_table(manifest: dict, quick: bool) -> bool:
    from scripts.build_master_table import MASTER_COLUMNS  # type: ignore

    path = REPO / "results" / "master_table.csv"
    if not path.exists():
        _fail(f"results/master_table.csv missing at {path}")
        return False

    import pandas as pd

    try:
        df = pd.read_csv(path)
    except Exception as e:  # pragma: no cover
        _fail(f"master_table.csv failed to parse: {e}")
        return False

    actual_cols = list(df.columns)
    if actual_cols != MASTER_COLUMNS:
        _fail(
            "master_table.csv columns mismatch MASTER_COLUMNS\n"
            f"        expected: {MASTER_COLUMNS}\n"
            f"        actual:   {actual_cols}"
        )
        return False

    n_entries = len(manifest["entries"])
    if len(df) > n_entries:
        _fail(f"master_table.csv has {len(df)} rows but only {n_entries} manifest entries")
        return False
    _v(f"row count={len(df)}, manifest entries={n_entries} (skipped={n_entries - len(df)})")

    if quick:
        _ok(f"master_table.csv coherent ({len(df)} rows, expected schema) [quick]")
        return True

    # Slow path: re-run build_master_table.py to a temp file and diff.
    with tempfile.TemporaryDirectory() as td:
        tmp_csv = Path(td) / "master_table.csv"
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "build_master_table.py"),
            "--no-tex",
            "--out-csv",
            str(tmp_csv),
        ]
        _v(f"re-running build: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            _fail("build_master_table.py timed out after 180s during re-build verify")
            return False
        if res.returncode != 0:
            _fail(f"build_master_table.py failed during re-build verify\n        stderr: {res.stderr.strip()[-500:]}")
            return False
        if not tmp_csv.exists():
            _fail("build_master_table.py did not produce the expected CSV")
            return False

        # Compare row-by-row, value-by-value, with NaN-tolerant equality.
        df_rebuilt = pd.read_csv(tmp_csv)
        if list(df_rebuilt.columns) != MASTER_COLUMNS:
            _fail(f"re-built master_table columns drifted: {list(df_rebuilt.columns)}")
            return False
        if len(df_rebuilt) != len(df):
            _fail(f"re-built master_table has {len(df_rebuilt)} rows, on-disk has {len(df)}")
            return False

        # Use pandas' assert_frame_equal-like semantics with float tolerance.
        try:
            pd.testing.assert_frame_equal(
                df.reset_index(drop=True),
                df_rebuilt.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-12,
            )
        except AssertionError as e:
            _fail(f"master_table.csv on disk drifted from a fresh re-build\n        diff: {str(e).splitlines()[0]}")
            return False

    _ok(f"master_table.csv coherent ({len(df)} rows, expected schema)")
    return True


# ----- check 3: diagnostics integrity -----------------------------------------

EXPECTED_DIAG_FILES = (
    "mz_scatter.png",
    "timeseries.png",
    "crash_2008.png",
    "crash_2020.png",
    "qlike_by_slot.png",
    "qlike_by_slot.csv",
    "mz_stats.json",
)

EXPECTED_MZ_KEYS = (
    "alpha",
    "beta",
    "alpha_se",
    "beta_se",
    "r2",
    "n",
    "t_beta_eq_1",
    "t_alpha_eq_0",
)


def _entries_with_predictions(manifest: dict) -> list[dict]:
    return [e for e in manifest["entries"] if e.get("predictions_glob")]


def _entry_id(entry: dict) -> str:
    return f"{entry['method']}_{entry['feature_set']}_{entry['config']}"


def check_diagnostics(manifest: dict) -> tuple[bool, bool]:
    """Returns (files_ok, mz_recompute_ok)."""
    diag_root = REPO / "results" / "diagnostics"
    if not diag_root.exists():
        _fail(f"results/diagnostics/ missing at {diag_root}")
        return False, False

    pred_entries = _entries_with_predictions(manifest)
    if not pred_entries:
        _fail("manifest has no entries with predictions_glob; nothing to validate")
        return False, False

    # 3a. file presence + json keys
    for entry in pred_entries:
        eid = _entry_id(entry)
        bundle = diag_root / eid
        if not bundle.exists():
            _fail(f"diagnostics bundle missing: {bundle}")
            return False, False
        for f in EXPECTED_DIAG_FILES:
            p = bundle / f
            if not p.exists():
                _fail(f"diagnostics file missing: {p}")
                return False, False
        try:
            stats = json.loads((bundle / "mz_stats.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            _fail(f"{bundle / 'mz_stats.json'} failed to parse: {e}")
            return False, False
        for k in EXPECTED_MZ_KEYS:
            if k not in stats:
                _fail(f"{bundle / 'mz_stats.json'} missing key: {k!r}")
                return False, False
        _v(f"{eid}: 7 files present, mz_stats.json schema ok")

    n = len(pred_entries)
    _ok(f"diagnostics bundles intact ({n}/{n} entries with predictions)")

    # 3b. MZ regression-test: recompute and compare.
    import numpy as np
    import pandas as pd

    from src.evaluation import mz_regression

    n_match = 0
    for entry in pred_entries:
        eid = _entry_id(entry)
        bundle = diag_root / eid
        cached = json.loads((bundle / "mz_stats.json").read_text(encoding="utf-8"))

        rdir = REPO / entry["results_dir"]
        paths = sorted(rdir.glob(entry["predictions_glob"]))
        if not paths:
            _fail(f"{eid}: predictions_glob matched no files at {rdir}")
            return True, False
        frames = []
        for p in paths:
            try:
                frames.append(pd.read_csv(p, usecols=["true_raw", "pred_raw"]))
            except (ValueError, KeyError) as exc:
                _fail(f"{eid}: cannot read {p} ({exc})")
                return True, False
        df = pd.concat(frames, ignore_index=True)
        y = df["true_raw"].to_numpy()
        yhat = df["pred_raw"].to_numpy()
        m = np.isfinite(y) & np.isfinite(yhat) & (y > 0) & (yhat > 0)
        y, yhat = y[m], yhat[m]
        recomputed = mz_regression(y, yhat)

        for k in EXPECTED_MZ_KEYS:
            cv, rv = cached[k], recomputed[k]
            if abs(float(cv) - float(rv)) > 1e-9:
                _fail(
                    f"{eid}: mz_stats.{k} drift  cached={cv}  recomputed={rv}  |diff|={abs(float(cv) - float(rv)):.3e}"
                )
                return True, False
        _v(f"{eid}: mz_stats matches recomputed within 1e-9")
        n_match += 1

    _ok(f"MZ stats match recomputed values ({n_match}/{n} within 1e-9)")
    return True, True


# ----- check 4: src.evaluation exports ----------------------------------------

EXPECTED_EXPORTS = (
    "mz_regression",
    "qlike_by_slot",
    "plot_mz_scatter",
    "plot_y_yhat_timeseries",
    "plot_crash_window",
    "plot_qlike_by_slot",
    "apply_duan_smearing",
    "calculate_metrics",
    "save_chunk_reduce",
    "build_results_dataframe",
)


def check_evaluation_exports() -> bool:
    try:
        import src.evaluation as ev  # type: ignore
    except Exception as e:
        _fail(f"src.evaluation failed to import: {e}")
        return False
    missing = [name for name in EXPECTED_EXPORTS if not hasattr(ev, name)]
    if missing:
        _fail(f"src.evaluation missing exports: {missing}")
        return False
    for name in EXPECTED_EXPORTS:
        _v(f"src.evaluation.{name} importable")
    _ok(f"src.evaluation exports complete ({len(EXPECTED_EXPORTS)}/{len(EXPECTED_EXPORTS)})")
    return True


# ----- check 5: plot_mz_scatter literal-string sanity -------------------------


def check_mz_scatter_strings() -> bool:
    import inspect

    from src.evaluation import plot_mz_scatter

    src = inspect.getsource(plot_mz_scatter)
    missing = [s for s in ("MZ fit", "45°") if s not in src]
    if missing:
        _fail(f"plot_mz_scatter source missing label strings: {missing}")
        return False
    _v("plot_mz_scatter source contains both 'MZ fit' and '45°'")
    _ok("plot_mz_scatter draws both lines (literal-string check)")
    return True


# ----- check 6: DL seeding ----------------------------------------------------


def check_dl_seeding() -> bool:
    failures = []
    # Per-DL-file invariants: import + call site
    for name in ("dl_patchts", "dl_ae_ridge"):
        path = REPO / "src" / f"{name}.py"
        if not path.exists():
            failures.append(f"src/{name}.py missing")
            continue
        text = path.read_text(encoding="utf-8")
        has_import = "from src.dl_executor import" in text and "seed_everything" in text
        if not has_import:
            failures.append(f"src/{name}.py missing 'from src.dl_executor import ... seed_everything'")
        if "seed_everything(args.seed)" not in text:
            failures.append(f"src/{name}.py missing 'seed_everything(args.seed)' call site")
        if has_import and "seed_everything(args.seed)" in text:
            _v(f"src/{name}.py: import + seed_everything(args.seed) present")

    # Shared definition + RNG pins + --seed flag live in dl_executor
    exec_path = REPO / "src" / "dl_executor.py"
    if not exec_path.exists():
        failures.append("src/dl_executor.py missing")
    else:
        exec_text = exec_path.read_text(encoding="utf-8")
        required = (
            "def seed_everything(",
            "np.random.seed(",
            "torch.manual_seed(",
            "torch.cuda.manual_seed_all(",
            "torch.backends.cudnn",
            "PYTHONHASHSEED",
            "--seed",
        )
        missing = [r for r in required if r not in exec_text]
        if missing:
            failures.append(f"src/dl_executor.py missing seeding primitives: {missing}")
        else:
            _v(
                "src/dl_executor.py: seed_everything pins numpy + torch + cuda + cudnn + PYTHONHASHSEED; "
                "--seed in build_dl_parser"
            )

    if failures:
        for f in failures:
            _fail(f)
        return False
    _ok("DL seeding pinned (dl_executor.seed_everything; dl_patchts, dl_ae_ridge)")
    return True


# ----- check 7: README invariants ---------------------------------------------

README_KEYWORDS = ("Duan smearing", "shift(1)", "intersection-N", "SEED=42")


def check_readme_invariants() -> bool:
    path = REPO / "README.md"
    if not path.exists():
        _fail(f"README.md missing at {path}")
        return False
    text = path.read_text(encoding="utf-8")
    if "## Key Invariants" not in text:
        _fail("README.md missing '## Key Invariants' section header")
        return False
    missing = [kw for kw in README_KEYWORDS if kw not in text]
    if missing:
        _fail(f"README.md missing invariant keywords: {missing}")
        return False
    for kw in README_KEYWORDS:
        _v(f"README contains: {kw!r}")
    _ok(f"README invariants documented ({len(README_KEYWORDS)}/{len(README_KEYWORDS)} keywords)")
    return True


# ----- driver -----------------------------------------------------------------


def main() -> int:
    global VERBOSE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Skip slow re-build verification of master_table.csv.")
    parser.add_argument("--verbose", action="store_true", help="Print per-check details, not just the summary line.")
    args = parser.parse_args()
    VERBOSE = args.verbose

    print(f"== harxhar audit check ({date.today().isoformat()}) ==")

    ok, manifest = check_manifest()
    if not ok or manifest is None:
        return 1

    if not check_master_table(manifest, args.quick):
        return 1

    files_ok, mz_ok = check_diagnostics(manifest)
    if not files_ok or not mz_ok:
        return 1

    if not check_evaluation_exports():
        return 1

    if not check_mz_scatter_strings():
        return 1

    if not check_dl_seeding():
        return 1

    if not check_readme_invariants():
        return 1

    print("\n== ALL CHECKS PASSED ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
