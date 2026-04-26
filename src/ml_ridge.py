# Auto-generated from notebooks/ml_ridge.ipynb. Do not edit by hand.

"""Ridge regression volatility backtest executor.

Method-specific glue around the shared scaffold in :mod:`src.executor`.
Only the model factory + default hyperparams + ``main()`` wrapper live
here; everything else (CLI parsing, loading, features, backtest loop,
smearing, reduce JSON, segment dispatch) is owned by ``src.executor``.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import Ridge

from src.executor import parse_executor_args, run_executor
from src.loading import parse_exog_cols
from src.scaling import run_backtest

# Per-method default hyperparams. Tuned overrides from --params-file are
# merged on top via dict.update().
DEFAULT_RIDGE_PARAMS: dict = dict(alpha=1.0)

# Method-specific refit-frequency default. The CLI ``--refit-frequency``
# sentinel of None falls back to this; the original ml_ridge.py used
# 1 as its argparse default (closed-form solver -> cheap to refit).
DEFAULT_RIDGE_REFIT_FREQUENCY: int = 1


def fit_predict_ridge(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    train_win_periods: int,
    hyperparams: dict,
) -> np.ndarray:
    """Walk-forward backtest with Ridge. Returns OOS predictions.

    Wraps :func:`src.scaling.run_backtest` with the Ridge-specific
    settings (``use_scaling=True`` -- linear models need feature
    standardization; refit frequency from
    ``hyperparams['_refit_frequency']``).

    Notes
    -----
    Ridge's default solver is closed-form, so ``random_state`` is
    irrelevant for reproducibility. We strip any internal control keys
    (``_*``) before forwarding to ``Ridge(**...)``.
    """
    refit_frequency = int(hyperparams.get("_refit_frequency", DEFAULT_RIDGE_REFIT_FREQUENCY))
    # Strip our internal control keys before passing to Ridge.
    model_kwargs = {k: v for k, v in hyperparams.items() if not k.startswith("_")}

    def model_fn():
        return Ridge(**model_kwargs)

    return run_backtest(
        model_fn,
        X_chunk,
        y_chunk,
        train_win=train_win_periods,
        refit_frequency=refit_frequency,
        use_scaling=True,
    )


def main() -> None:
    args = parse_executor_args("Ridge walk-forward backtest")

    tuned_params: dict = {}
    if args.params_file:
        with open(args.params_file) as f:
            tuned_params = json.load(f)

    # Method defaults <- tuned overrides; refit-frequency from CLI sentinel.
    hyperparams = dict(DEFAULT_RIDGE_PARAMS)
    hyperparams.update(tuned_params)
    refit_frequency = args.refit_frequency if args.refit_frequency is not None else DEFAULT_RIDGE_REFIT_FREQUENCY
    hyperparams["_refit_frequency"] = refit_frequency

    exog_cols = parse_exog_cols(args.exog_cols)

    run_executor(
        method_name="ridge",
        fit_predict=fit_predict_ridge,
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
        add_calendar=False,
        target_use_diurnal=False,
        target_winsor_window=None,
        dropna_with_exog=True,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
