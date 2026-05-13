# Auto-generated from notebooks/05_executor.ipynb. Do not edit by hand.

"""Shared executor scaffold for ML walk-forward volatility backtests.

Lifts the ~90% duplicated `main()` body of the per-method executors
(`src/ml_xgboost.py`, `src/ml_lightgbm.py`, `src/ml_random_forest.py`,
`src/ml_ridge.py`) into one module. Per-method scripts supply only a
`fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams)`
callable plus a default-hyperparam dict; everything else — CLI parsing,
loading, transforms, horizon shift, chunk slicing, smearing, reduce
JSON — is shared.

The CLI contract is now owned by ``.hpc/cli.py`` (the auto-generated
dispatcher) and ``.hpc/tasks.py`` FLAGS dict — the per-executor flag
list lives there, not in this module. ``run_executor`` and the
backtest plumbing remain here. The flags expected on ``args`` match
those declared in ``.hpc/tasks.py`` FLAGS for the calling executor's
module key (e.g. ``FLAGS["src/tune_tree.py`` (cmd_evaluate) passes via subprocess:
``--params-file --output-file --start --end --data-path --train-window``
``--horizon --exog-cols`` plus the Ridge-only ``--segment`` and
``--lag-scope`` and the optional ``--refit-frequency`` / ``--seed``.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.evaluation import apply_duan_smearing, save_chunk_reduce
from src.loading import apply_overnight_fills, load_raw_data
from src.transforms import (
    PERIODS_PER_DAY,
    SEGMENT_DEFINITIONS,
    add_calendar_features,
    apply_horizon_shift,
    compute_segment_train_window,
    generate_har_features,
    resolve_har_lags,
    robust_transform,
    slice_to_segment,
)

FitPredict = Callable[[np.ndarray, np.ndarray, int, dict], np.ndarray]


from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class ExecutorConfig:
    """Per-method backtest invariants. Drift here was the proximate cause
    of the 2026-04-23 alignment audit (intersection-N regression). Each
    per-method module (``src/ml_<method>.py``) defines its own
    ``CONFIG = ExecutorConfig(...)`` constant; the CONFIGS registry
    below imports them at runtime for the drift-check inside
    ``run_executor``.
    """
    method: str
    add_calendar: bool
    target_use_diurnal: bool
    target_winsor_window: int | None
    dropna_with_exog: bool
    refit_frequency: int
    # Calendar encoding: "raw" for tree models (int DOW, int hour, binary
    # is_overnight — trees split natively); "rich" for ridge (5 weekday
    # dummies + sin/cos hour + binary is_overnight, so the linear model can
    # represent the volatility U-shape and per-DOW intercepts).
    calendar_encoding: str = "raw"

    def as_data_prep_kwargs(self) -> dict:
        # NOTE: `calendar_encoding` is intentionally NOT included here —
        # it's looked up directly from CONFIGS inside run_executor, not
        # passed through the CLI. This keeps the CLI contract stable.
        return {
            "add_calendar": self.add_calendar,
            "target_use_diurnal": self.target_use_diurnal,
            "target_winsor_window": self.target_winsor_window,
            "dropna_with_exog": self.dropna_with_exog,
        }


def _load_configs() -> dict[str, "ExecutorConfig"]:
    """Lazy registry — imports CONFIG from each method module so per-method
    specs live next to per-method model code (not centralized here).

    Lazy because of the circular import: ``src.ml_<method>`` imports
    ``ExecutorConfig`` from this module, so we can't import them at
    module-import time. ``run_executor`` calls this once when needed.

    Excluded by design: dl_ae_ridge, dl_patchts (separate config shape);
    tune_tree (no run_executor call); ml_baseline (no exog path).
    """
    from src.ml_ridge import CONFIG as _ridge
    from src.ml_xgboost import CONFIG as _xgb
    from src.ml_lightgbm import CONFIG as _lgbm
    from src.ml_random_forest import CONFIG as _rf
    from src.ml_pcr import CONFIG as _pcr
    return {c.method: c for c in (_ridge, _xgb, _lgbm, _rf, _pcr)}


# Lazy singleton — populated on first access via _get_configs(); CONFIGS
# is exported as a module-level name so existing call sites that import
# CONFIGS keep working unchanged.
_CONFIGS_CACHE: dict[str, "ExecutorConfig"] | None = None


def _get_configs() -> dict[str, "ExecutorConfig"]:
    global _CONFIGS_CACHE
    if _CONFIGS_CACHE is None:
        _CONFIGS_CACHE = _load_configs()
    return _CONFIGS_CACHE


class _ConfigsProxy:
    """Dict-like proxy that delegates to the lazy-loaded registry."""
    def __getitem__(self, k):
        return _get_configs()[k]
    def __contains__(self, k):
        return k in _get_configs()
    def __iter__(self):
        return iter(_get_configs())
    def keys(self):
        return _get_configs().keys()
    def items(self):
        return _get_configs().items()
    def values(self):
        return _get_configs().values()


CONFIGS = _ConfigsProxy()


def _backtest_and_save(
    df: pd.DataFrame,
    feature_names: list[str],
    fit_predict: FitPredict,
    hyperparams: dict,
    train_win_periods: int,
    horizon: int,
    start: int,
    end: int,
    output_file: str,
) -> None:
    """Run a prepared DataFrame through the walk-forward backtest and save.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``adj_RV``, ``baseline``, ``t``, plus all
        ``feature_names`` columns. The HAR-burn-in (``max_lag``) rows
        are dropped here before any extraction.
    feature_names : list[str]
        Column names in ``df`` that form the feature matrix.
    fit_predict : FitPredict
        Callable ``(X_chunk, y_chunk, train_win_periods, hyperparams)
        -> preds`` of shape ``(len(X_chunk) - train_win_periods,)``.
    hyperparams : dict
        Method-specific hyperparameters (typically a merge of method
        defaults and tuned overrides from ``--params-file``).
    train_win_periods : int
        Training window in 30-min periods.
    horizon : int
        Forecast horizon in 30-min periods.
    start, end : int
        Inclusive/exclusive chunk bounds in *post horizon-shift* index
        space. ``end == -1`` means "to the end".
    output_file : str
        Output CSV path. ``<basename>_reduce.json`` is written
        alongside.
    """
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, horizon)

    actual_end = len(X) if end == -1 else end
    X_chunk = X[start:actual_end]
    y_chunk = y[start:actual_end]
    dates_chunk = dates.iloc[start:actual_end].reset_index(drop=True)
    baselines_chunk = baselines[start:actual_end]

    if train_win_periods >= len(X_chunk):
        raise ValueError(f"train_window ({train_win_periods} periods) >= chunk size ({len(X_chunk)})")

    preds = fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams)

    oos_start = train_win_periods
    y_oos = y_chunk[oos_start:]
    dates_oos = dates_chunk.iloc[oos_start:].values
    baselines_oos = baselines_chunk[oos_start:]

    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

    results = pd.DataFrame(
        {
            "date": dates_oos,
            "horizon": horizon,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results.to_csv(output_file, index=False)
    save_chunk_reduce(results, output_file)
    print(f"Saved {len(results)} rows -> {output_file}")


def _build_har_and_calendar(df, exog_cols, add_calendar, calendar_encoding="raw"):
    df, har_names = generate_har_features(df, target_col="adj_RV", exog_cols=exog_cols)
    if add_calendar:
        feature_names = har_names + add_calendar_features(df, encoding=calendar_encoding)
    else:
        feature_names = har_names
    return df, feature_names


def load_and_transform(
    data_path: str,
    exog_cols: list[str],
    *,
    target_use_diurnal: bool,
    target_winsor_window: int | None,
    dropna_with_exog: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Load raw data, apply RV + exog robust transforms, return (df, adj_exog).

    Parameters
    ----------
    data_path : str
        Forwarded to ``load_raw_data``.
    exog_cols : list[str]
        Raw exog column names (pre-transform). May be empty.
    target_use_diurnal : bool
        If True, apply diurnal adjustment to the RV target transform.
        Tree methods: True. Ridge: False.
    target_winsor_window : int | None
        Winsorization window for the RV target. Tree methods: 240.
        Ridge: None (no winsorization beyond ``robust_transform``
        defaults).
    dropna_with_exog : bool
        Exog columns are forward-filled before this branch, so the only
        rows that can still have NaN exog are at the leading edge of the
        dataset (before any valid value). If True, drop those rows.
        Ridge: True (sklearn Ridge requires no NaN). Tree methods: False
        (trees handle NaN natively).
    """
    df = load_raw_data(data_path, allow_missing=True)
    if exog_cols:
        apply_overnight_fills(df, exog_cols)
        df[exog_cols] = df[exog_cols].ffill()
        if dropna_with_exog:
            df = df.dropna(subset=["RV"] + exog_cols).reset_index(drop=True)
        else:
            df = df.dropna(subset=["RV"]).reset_index(drop=True)

    transform_kwargs: dict = {"is_target": True}
    if target_use_diurnal:
        transform_kwargs["use_diurnal"] = True
    if target_winsor_window is not None:
        transform_kwargs["winsor_window"] = target_winsor_window
    adj_rv, baseline = robust_transform(df, "RV", **transform_kwargs)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    adj_exog_cols: list[str] = []
    for col in exog_cols:
        adj_col = f"adj_{col}"
        adj_series, _ = robust_transform(df, col, use_transform=True, use_diurnal=True)
        df[adj_col] = adj_series
        adj_exog_cols.append(adj_col)

    return df, adj_exog_cols


def _iter_TOD_segment(
    df, *, segment, lag_scope, train_window, output_file, exog_cols, add_calendar,
    calendar_encoding="raw",
):
    """Yield (seg_name, job_df, feature_names, train_win_periods, job_output_file)
    for each time-of-day segment we need to backtest. ``seg_name`` is None when
    no segmentation is requested."""
    if segment is None:
        df, feature_names = _build_har_and_calendar(df, exog_cols, add_calendar, calendar_encoding)
        yield None, df, feature_names, train_window * PERIODS_PER_DAY, output_file
        return

    segments = list(SEGMENT_DEFINITIONS) if segment == "all" else [segment]
    base, ext = os.path.splitext(output_file)

    if lag_scope == "global":
        df, feature_names = _build_har_and_calendar(df, exog_cols, add_calendar, calendar_encoding)

    for seg_name in segments:
        seg_df = slice_to_segment(df, seg_name)
        if seg_df.empty:
            print(f"No data for segment '{seg_name}'. Skipping.")
            continue
        if lag_scope == "intra":
            seg_df, feature_names = _build_har_and_calendar(seg_df, exog_cols, add_calendar, calendar_encoding)
        train_win_periods = compute_segment_train_window(seg_df["t"], train_window)
        yield seg_name, seg_df, feature_names, train_win_periods, f"{base}_{seg_name}{ext}"


def run_executor(
    method_name: str,
    fit_predict: FitPredict,
    hyperparams: dict,
    *,
    data_path: str,
    output_file: str,
    horizon: int,
    train_window: int,
    start: int,
    end: int,
    exog_cols: list[str],
    segment: str | None,
    lag_scope: str,
    add_calendar: bool,
    target_use_diurnal: bool,
    target_winsor_window: int | None,
    dropna_with_exog: bool,
    seed: int = 42,
) -> None:
    """Top-level scaffold. Loads data, builds features, dispatches backtest.

    Per-method scripts (`ml_xgboost.py`, etc.) call this from their
    ``main()`` after parsing the canonical CLI and assembling the
    method-specific ``fit_predict`` closure + merged hyperparams.

    Notes
    -----
    * ``segment`` is None for tree methods (XGB/LGBM/RF). Ridge is the
      only method that uses segments.
    * ``add_calendar`` is True for tree methods, False for Ridge.
    * ``method_name`` is informational — used in log lines only.
    """
    del seed  # reserved; per-method scripts can wire seed into model_fn directly

    if method_name in CONFIGS:
        expected = CONFIGS[method_name].as_data_prep_kwargs()
        actual = {
            "add_calendar": add_calendar,
            "target_use_diurnal": target_use_diurnal,
            "target_winsor_window": target_winsor_window,
            "dropna_with_exog": dropna_with_exog,
        }
        if actual != expected:
            raise ValueError(
                f"data-prep drift for {method_name}: {actual} != {expected}"
            )

    df, adj_exog_cols = load_and_transform(
        data_path,
        exog_cols,
        target_use_diurnal=target_use_diurnal,
        target_winsor_window=target_winsor_window,
        dropna_with_exog=dropna_with_exog,
    )

    calendar_encoding = CONFIGS[method_name].calendar_encoding if method_name in CONFIGS else "raw"
    for seg_name, job_df, feature_names, train_win, out_file in _iter_TOD_segment(
        df,
        segment=segment, lag_scope=lag_scope,
        train_window=train_window, output_file=output_file,
        exog_cols=adj_exog_cols, add_calendar=add_calendar,
        calendar_encoding=calendar_encoding,
    ):
        if seg_name is not None:
            print(f"{'=' * 20} {method_name.upper()} SEGMENT: {seg_name.upper()} {'=' * 20}")
            print(f"Window: {train_win} periods ({train_window} days)")
        _backtest_and_save(
            job_df, feature_names, fit_predict, hyperparams,
            train_win, horizon, start, end, out_file,
        )
