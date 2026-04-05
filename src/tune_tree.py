"""Optuna hyperparameter tuning for tree-based volatility models.

Tunes on horizon=1 (one-step-ahead), outputs best params as JSON.
Supports distributed Optuna via JournalFileStorage for HPC parallelism.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import optuna
import pandas as pd

from evaluation import apply_duan_smearing
from src.loading import load_raw_data
from src.ml_random_forest import (
    PERIODS_PER_DAY,
    add_calendar_features,
    apply_horizon_shift,
    generate_har_features,
    resolve_har_lags,
    run_backtest,
)
from src.transforms import robust_transform

# ── Search spaces ────────────────────────────────────────────────────────────


def _suggest_rf(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        "n_jobs": -1,
    }


def _suggest_xgb(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "tree_method": "hist",
        "n_jobs": -1,
    }


def _suggest_lgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_jobs": -1,
        "verbose": -1,
    }


_SUGGEST_FNS = {"rf": _suggest_rf, "xgb": _suggest_xgb, "lgbm": _suggest_lgbm}


# ── Lazy model imports ───────────────────────────────────────────────────────


def _get_model_class(model_name: str):
    if model_name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor
    elif model_name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor
    elif model_name == "lgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Data preparation ─────────────────────────────────────────────────────────


def prepare_data(data_path: str, train_window_days: int) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray, int]:
    """Run the standard pipeline and return arrays ready for backtesting.

    Returns (X, y, dates, baselines, train_win).
    """
    # 1. Load
    df = load_raw_data(data_path, allow_missing=True)

    # 2. Robust transform on RV
    adj_rv, baseline = robust_transform(df, "RV", is_target=True, use_diurnal=True, winsor_window=240)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    # 3. HAR features
    df, har_names = generate_har_features(df, target_col="adj_RV")

    # 4. Calendar features
    cal_names = add_calendar_features(df)

    feature_names = har_names + cal_names

    # 5. Drop initial NaN rows from HAR lag computation
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    # 6. Horizon shift (horizon=1)
    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, horizon=1)

    train_win = train_window_days * PERIODS_PER_DAY

    return X, y, dates, baselines, train_win


# ── Objective factory ────────────────────────────────────────────────────────


def make_objective(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    baselines: np.ndarray,
    train_win: int,
    refit_frequency: int,
):
    """Return an Optuna objective function (minimises QLIKE)."""
    suggest_fn = _SUGGEST_FNS[model_name]
    model_cls = _get_model_class(model_name)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        model_fn = lambda: model_cls(**params)  # noqa: E731

        # Walk-forward backtest
        preds = run_backtest(model_fn, X, y, train_win, refit_frequency)

        # OOS slice
        y_oos = y[train_win:]
        baselines_oos = baselines[train_win:]

        # Duan smearing → raw scale
        pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

        # QLIKE: mean(ratio - log(ratio) - 1), mask where both > 0
        mask = (true_raw > 0) & (pred_raw > 0)
        ratio = true_raw[mask] / pred_raw[mask]
        qlike = float(np.mean(ratio - np.log(ratio) - 1))

        return qlike

    return objective


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for tree-based volatility models")
    parser.add_argument("--model", required=True, choices=["rf", "xgb", "lgbm"])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--data-path", default="all30min")
    parser.add_argument("--train-window", type=int, default=500, help="training window in days")
    parser.add_argument("--refit-frequency", type=int, default=5)
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna journal file path for distributed tuning",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name (required for distributed tuning)",
    )
    parser.add_argument("--output-file", required=True, help="Output JSON file for best params")
    args = parser.parse_args()

    # Prepare data
    X, y, dates, baselines, train_win = prepare_data(args.data_path, args.train_window)

    # Build objective
    objective = make_objective(args.model, X, y, dates, baselines, train_win, args.refit_frequency)

    # Storage
    storage = None
    if args.storage is not None:
        storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(args.storage))

    study_name = args.study_name if args.study_name is not None else f"tune_{args.model}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    # Save best params
    best_params = study.best_trial.params
    out_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Best QLIKE: {study.best_trial.value:.6f}")
    print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()
