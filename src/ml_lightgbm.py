# Auto-generated from notebooks/ml_lightgbm.ipynb. Do not edit by hand.

"""LightGBM volatility backtest executor.

Method-specific glue around the shared scaffold in :mod:`src.executor`.
Only the model factory + default hyperparams + ``main()`` wrapper live
here; everything else (CLI parsing, loading, features, backtest loop,
smearing, reduce JSON) is owned by ``src.executor``.
"""

from __future__ import annotations

import json

import numpy as np
from lightgbm import LGBMRegressor

from src.executor import parse_executor_args, run_executor
from src.loading import parse_exog_cols
from src.scaling import run_backtest

# Per-method default hyperparams. Tuned overrides from --params-file are
# merged on top via dict.update().
DEFAULT_LGBM_PARAMS: dict = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=-1,
    verbose=-1,
)

# Method-specific refit-frequency default. The CLI ``--refit-frequency``
# sentinel of None falls back to this; the original ml_lightgbm.py used
# 1 as its argparse default.
DEFAULT_LGBM_REFIT_FREQUENCY: int = 1


def fit_predict_lgbm(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Walk-forward backtest with LightGBM. Returns OOS predictions.

    Wraps :func:`src.scaling.run_backtest` with the LGBM-specific
    settings (``use_scaling=False``; refit frequency from
    ``hyperparams['_refit_frequency']``). The model is constructed
    with ``random_state=hyperparams.get("random_state", 42)`` for
    reproducibility.
    """
    refit_frequency = int(hyperparams.get("_refit_frequency", DEFAULT_LGBM_REFIT_FREQUENCY))
    # Strip our internal control key before passing to LGBMRegressor.
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


def main() -> None:
    args = parse_executor_args("LightGBM walk-forward backtest")

    tuned_params: dict = {}
    if args.params_file:
        with open(args.params_file) as f:
            tuned_params = json.load(f)

    # Method defaults <- tuned overrides; refit-frequency from CLI sentinel.
    hyperparams = dict(DEFAULT_LGBM_PARAMS)
    hyperparams.update(tuned_params)
    hyperparams["random_state"] = args.seed
    refit_frequency = args.refit_frequency if args.refit_frequency is not None else DEFAULT_LGBM_REFIT_FREQUENCY
    hyperparams["_refit_frequency"] = refit_frequency

    exog_cols = parse_exog_cols(args.exog_cols)

    run_executor(
        method_name="lightgbm",
        fit_predict=fit_predict_lgbm,
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
        dropna_with_exog=False,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
