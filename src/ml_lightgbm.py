# Auto-generated from ml_lightgbm.ipynb. Do not edit by hand.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src._template import register_run
from src.evaluation import calculate_metrics
from src.executor import run_executor
from src.loading import parse_exog_cols
from src.scaling import run_backtest

DEFAULT_LGBM_PARAMS: dict = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=-1,
    verbose=-1,
)


def fit_predict_lgbm(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Walk-forward backtest with LGBMRegressor. Returns OOS predictions.

    Wraps :func:`src.scaling.run_backtest` (no feature scaling;
    refit cadence from ``hyperparams['_refit_frequency']``). Internal control
    keys (``_*``) are stripped before forwarding to the model constructor.
    """
    refit_frequency = int(hyperparams.get("_refit_frequency", 1))
    model_kwargs = {k: v for k, v in hyperparams.items() if not k.startswith("_")}
    model_kwargs.setdefault("random_state", 42)

    def model_fn():
        return LGBMRegressor(**model_kwargs)

    return run_backtest(
        model_fn,
        X_chunk,
        y_chunk,
        train_win=train_win_periods,
        refit_frequency=refit_frequency,
        use_scaling=False,
    )


@register_run
def run(
    horizon: int = 1,
    train_window: int = 500,
    refit_frequency: int | None = None,
    exog_cols: str = "",
    seed: int = 42,
    data_path: str = "all30min",
    output_file: str = "results/lgbm/run.json",
    params_file: str = "",
    start: int = 0,
    end: int = -1,
) -> dict:
    """# LightGBM Volatility Backtest

    Walk-forward volatility backtest with LightGBM. Defined as a single
    `@register_run def run()` experiment entrypoint. -- one task.

        Returns a metrics dict (the run's value; the coordinator reduces it across
        chunks). The per-row prediction table is written next to ``output_file`` as
        ``results.csv`` by the shared backtest scaffold. Data-prep invariants
        (formerly ``ExecutorConfig``) are inline literals below.
    """
    hyperparams: dict = dict(DEFAULT_LGBM_PARAMS)
    if params_file:
        with open(params_file) as fh:
            hyperparams.update(json.load(fh))
    hyperparams["random_state"] = seed
    hyperparams["_refit_frequency"] = refit_frequency if refit_frequency is not None else 1

    results_csv = str(Path(output_file).with_name("results.csv"))
    run_executor(
        method_name="lightgbm",
        fit_predict=fit_predict_lgbm,
        hyperparams=hyperparams,
        data_path=data_path,
        output_file=results_csv,
        horizon=horizon,
        train_window=train_window,
        start=start,
        end=end,
        exog_cols=parse_exog_cols(exog_cols or None),
        segment=None,
        lag_scope="global",
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        seed=seed,
    )
    metrics = calculate_metrics(pd.read_csv(results_csv))
    return {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
