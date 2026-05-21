"""Chunk-fungibility gate — a chunked walk-forward run must elide to the serial one.

Each chunk-fungible executor is driven the production way (the injected
``compute(args)`` wrapper, with ``start`` / ``end`` / ``halo`` on the args
namespace, which the inlined runtime turns into the active ``SliceSpec``):

* **whole** — ``start=0, end=E, halo=0`` — the serial walk-forward run.
* **3 chunks** — the OOS region ``[TRAIN_WIN, E)`` split at the emit
  boundaries in :data:`CHUNKS`, each with ``halo=TRAIN_WIN``.

The concatenated chunk outputs must tile the whole run row-for-row. This is
the open-loop-chunking correctness claim B3/B4 rest on: a 100-way HPC split
of one backtest must produce the same predictions as running it serially.

Coverage
--------
Three executors are gated, covering the three distinct slicing code paths:
``baseline`` (the inline backtest), ``ridge`` (the shared ``run_executor``
scaffold), and ``pcr`` (the bespoke ``run_pcr_backtest`` helper). xgboost /
lightgbm / random_forest add no chunk-slicing code of their own — they share
ridge's ``run_executor`` path verbatim — so the ridge result covers them;
exercising them here would only add CPU-bound tree backtests with no extra
code path tested.

Correctness facts baked into this gate
---------------------------------------
* **pcr is NOT chunk-fungible.** ``run_pcr_backtest`` keeps an in-loop
  ``RollingRobustScaler`` whose look-back cannot be relocated bit-identically
  to a whole-series transform; a chunk boundary desyncs it. pcr is classified
  *Sequential* (run whole, never chunked) and is therefore excluded from the
  chunked-vs-whole assertion — it is exercised whole-only, as a smoke check.
* **Fungible executors:** ridge, baseline (and, by sharing ridge's path,
  xgboost / lightgbm / random_forest). Ridge because B3 moved the rolling
  feature scaler into a whole-series data-prep transform; baseline because it
  has no fitted model. Both are bit-exact (tol 0).
* **``pred_raw`` is excluded** from the row comparison: its Duan-smearing
  factor is a mean over each run's own OOS rows, so it is chunk-local by
  construction. ``pred_adj`` / ``true_adj`` / ``true_raw`` / ``date`` are
  per-row and must tile exactly.
* **Refit-boundary alignment.** A chunk's emit boundary must land on a
  refit-cadence multiple or the refit schedule desyncs across chunks. ridge
  accepts ``--refit-frequency``; this gate pins it to :data:`REFIT_FREQ` and
  makes every :data:`CHUNKS` boundary a multiple of it.

The fixture exercises ≥3 chunks, a horizon shift (``HORIZON > 1``), an
in-chunk refit (each chunk spans several refit cadences), seeded RNG (the
ridge solver takes ``seed``), and an aggregated metric (each ``run()``
returns a metrics dict reduced from its per-row table).

Each test runs a real backtest four times (whole + 3 chunks) over the
``all30min`` dataset, so the module is marked ``slow`` (~90 s) and excluded
from the default ``pytest -m "not slow"`` set; the CI ``fungibility`` job
runs it explicitly with ``-m slow``.
"""

from __future__ import annotations

import importlib
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── fixture geometry ─────────────────────────────────────────────────────────
TW_DAYS = 10
PERIODS_PER_DAY = 48
TRAIN_WIN = TW_DAYS * PERIODS_PER_DAY  # 480-period warm-up halo
E = 900  # small --end so the tree backtests stay CI-fast
HORIZON = 2  # >1 — exercises the horizon shift
REFIT_FREQ = 20  # pinned for every executor; CHUNKS boundaries are multiples of it
# Emit ranges tiling the OOS region [TRAIN_WIN, E). Every boundary is a
# multiple of REFIT_FREQ and >= TRAIN_WIN, so the refit schedule stays in
# sync across chunks. 3 chunks, each spanning several refit cadences.
CHUNKS = [(480, 620), (620, 760), (760, 900)]

PER_ROW = ["pred_adj", "true_adj", "true_raw"]  # chunk-fungible columns


def _run(mod, tmp_path: Path, tag: str, *, start: int, end: int, halo: int) -> pd.DataFrame:
    """Drive an executor's compute(args) over one slice; return its results.csv."""
    out = tmp_path / tag / "run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    ns = types.SimpleNamespace(
        train_window=TW_DAYS,
        data_path="all30min",
        horizon=HORIZON,
        output_file=str(out),
        start=start,
        end=end,
        halo=halo,
        seed=42,
        # Pinned cadence — every fungible executor accepts it; baseline/pcr
        # have no such run() param and compute() drops the unused attr.
        refit_frequency=REFIT_FREQ,
    )
    mod.compute(ns)
    # run() returns a metrics dict; compute() writes it to output_file and the
    # per-row table to results.csv alongside. Assert the aggregated metric is
    # present and well-formed before comparing per-row outputs.
    metrics = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(metrics, dict) and "mse" in metrics, f"{tag}: no aggregated metric in run.json"
    return pd.read_csv(out.with_name("results.csv"))


def _maxdiff(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> float:
    return max(float(np.abs(a[c].to_numpy() - b[c].to_numpy()).max()) for c in cols)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["baseline", "ridge"])
def test_chunked_run_equals_serial(method: str, tmp_path: Path) -> None:
    """A 3-chunk walk-forward run tiles the serial run row-for-row (bit-exact)."""
    mod = importlib.import_module(f"src.ml_{method}")

    whole = _run(mod, tmp_path, "whole", start=0, end=E, halo=0)
    parts = [_run(mod, tmp_path, f"c{j}", start=s, end=e, halo=TRAIN_WIN) for j, (s, e) in enumerate(CHUNKS)]
    combined = pd.concat(parts, ignore_index=True)

    assert len(combined) == len(whole), f"chunked rows={len(combined)} != whole rows={len(whole)}"
    assert (combined["date"].to_numpy() == whole["date"].to_numpy()).all(), "chunk dates desynced from whole run"

    diff = _maxdiff(combined, whole, PER_ROW)
    assert diff == 0.0, f"{method}: fungibility maxdiff={diff:.3e} (expected bit-exact)"


@pytest.mark.slow
def test_pcr_is_sequential_not_chunked(tmp_path: Path) -> None:
    """pcr is Sequential — exercised whole-only.

    ``run_pcr_backtest``'s in-loop scaler is not chunk-relocatable, so pcr is
    never chunked. This is a smoke check that the whole-series run still
    produces a per-row table and an aggregated metric; pcr is deliberately
    absent from :func:`test_chunked_run_equals_serial`.
    """
    mod = importlib.import_module("src.ml_pcr")
    whole = _run(mod, tmp_path, "pcr_whole", start=0, end=E, halo=0)
    assert len(whole) > 0
    assert list(PER_ROW) == [c for c in PER_ROW if c in whole.columns]
