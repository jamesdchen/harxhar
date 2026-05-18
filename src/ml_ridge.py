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

from src.executor import ExecutorConfig, run_executor
from src.loading import parse_exog_cols
from src.scaling import run_backtest

# Method-specific data-prep config. Lives here (not in src.executor) so the
# ridge spec sits next to the ridge model code. src.executor.CONFIGS imports
# this at runtime to keep a method-name → config registry for drift checks.
CONFIG = ExecutorConfig(
    method="ridge",
    add_calendar=True,
    target_use_diurnal=True,
    target_winsor_window=240,
    dropna_with_exog=True,
    refit_frequency=1,
)

# Per-method default hyperparams. Tuned overrides from --params-file are
# merged on top via dict.update().
DEFAULT_RIDGE_PARAMS: dict = dict(alpha=1.0)


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
    refit_frequency = int(hyperparams.get("_refit_frequency", CONFIG.refit_frequency))
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


def compute(args) -> None:

    tuned_params: dict = {}
    if args.params_file:
        with open(args.params_file) as f:
            tuned_params = json.load(f)

    # Method defaults <- tuned overrides; refit-frequency from CLI sentinel.
    hyperparams = dict(DEFAULT_RIDGE_PARAMS)
    hyperparams.update(tuned_params)
    refit_frequency = args.refit_frequency if args.refit_frequency is not None else CONFIG.refit_frequency
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
        **CONFIG.as_data_prep_kwargs(),
        seed=args.seed,
    )
