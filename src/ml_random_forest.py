# Auto-generated from notebooks/ml_random_forest.ipynb. Do not edit by hand.

"""Random Forest volatility backtest executor.

Method-specific glue around the shared scaffold in :mod:`src.executor`.
Only the model factory + default hyperparams + ``main()`` wrapper live
here; everything else (CLI parsing, loading, features, backtest loop,
smearing, reduce JSON) is owned by ``src.executor``.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.executor import parse_executor_args, run_executor
from src.loading import parse_exog_cols
from src.scaling import run_backtest

# Per-method default hyperparams. Tuned overrides from --params-file are
# merged on top via dict.update().
DEFAULT_RF_PARAMS: dict = dict(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=5,
    n_jobs=-1,
)

# Method-specific refit-frequency default. The CLI ``--refit-frequency``
# sentinel of None falls back to this; the original ml_random_forest.py
# used 5 as its argparse default.
DEFAULT_RF_REFIT_FREQUENCY: int = 5


def fit_predict_rf(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Walk-forward backtest with RandomForest. Returns OOS predictions.

    Wraps :func:`src.scaling.run_backtest` with the RF-specific settings
    (``use_scaling=False``; refit frequency from
    ``hyperparams['_refit_frequency']``). ``random_state`` defaults to
    42 and can be overridden via ``hyperparams['random_state']``.
    """
    refit_frequency = int(hyperparams.get("_refit_frequency", DEFAULT_RF_REFIT_FREQUENCY))
    # Strip our internal control key before passing to RandomForestRegressor.
    model_kwargs = {k: v for k, v in hyperparams.items() if not k.startswith("_")}
    model_kwargs.setdefault("random_state", 42)

    def model_fn():
        return RandomForestRegressor(**model_kwargs)

    return run_backtest(
        model_fn,
        X_chunk,
        y_chunk,
        train_win=train_win_periods,
        refit_frequency=refit_frequency,
        use_scaling=False,
    )


def main() -> None:
    args = parse_executor_args("Random Forest walk-forward backtest")

    tuned_params: dict = {}
    if args.params_file:
        with open(args.params_file) as f:
            tuned_params = json.load(f)

    # Method defaults <- tuned overrides; refit-frequency from CLI sentinel.
    hyperparams = dict(DEFAULT_RF_PARAMS)
    hyperparams.update(tuned_params)
    # Wire seed through to RandomForestRegressor's random_state.
    hyperparams.setdefault("random_state", args.seed)
    refit_frequency = args.refit_frequency if args.refit_frequency is not None else DEFAULT_RF_REFIT_FREQUENCY
    hyperparams["_refit_frequency"] = refit_frequency

    exog_cols = parse_exog_cols(args.exog_cols)

    run_executor(
        method_name="random_forest",
        fit_predict=fit_predict_rf,
        hyperparams=hyperparams,
        data_path=args.data_path,
        output_file=args.output_file,
        horizon=args.horizon,
        train_window=args.train_window,
        start=args.start,
        end=args.end,
        exog_cols=exog_cols,
        segment=args.segment,
        lag_scope=args.lag_scope,
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=True,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
