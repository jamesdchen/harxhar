"""Optuna hyperparameter tuning for tree-based volatility models.

Three subcommands for HPC-parallel tuning:
  suggest  — generate a batch of candidate param sets (shotgun TPE)
  evaluate — thin wrapper that delegates one trial to the right ml_*.py executor
  score    — read executor results, compute QLIKE, report back to Optuna

Legacy single-machine mode:
  run      — full Optuna loop in-process (original behavior)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.evaluation import apply_duan_smearing
from src.loading import load_raw_data
from src.scaling import run_backtest
from src.transforms import (
    PERIODS_PER_DAY,
    add_calendar_features,
    apply_horizon_shift,
    generate_har_features,
    resolve_har_lags,
    robust_transform,
)

# ── Search spaces ────────────────────────────────────────────────────────────

_MODELS = ["rf", "xgb", "lgbm"]

_EXECUTOR_SCRIPTS = {
    "rf": "src/ml_random_forest.py",
    "xgb": "src/ml_xgboost.py",
    "lgbm": "src/ml_lightgbm.py",
}


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


# ── Shared helpers ───────────────────────────────────────────────────────────


def _make_storage(path: str | None) -> optuna.storages.BaseStorage | None:
    if path is None:
        return None
    return optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(path))


def _load_or_create_study(
    model: str,
    storage_path: str | None,
    study_name: str | None = None,
) -> optuna.Study:
    storage = _make_storage(storage_path)
    name = study_name or f"tune_{model}"
    return optuna.create_study(
        study_name=name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(constant_liar=True),
        direction="minimize",
        load_if_exists=True,
    )


def _compute_qlike(results_df: pd.DataFrame) -> float:
    true_raw = results_df["true_raw"].values
    pred_raw = results_df["pred_raw"].values
    mask = (true_raw > 0) & (pred_raw > 0)
    ratio = true_raw[mask] / pred_raw[mask]
    return float(np.mean(ratio - np.log(ratio) - 1))


# ── Data preparation (for legacy run mode) ───────────────────────────────────


def prepare_data(data_path: str, train_window_days: int) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray, int]:
    """Run the standard pipeline and return arrays ready for backtesting."""
    df = load_raw_data(data_path, allow_missing=True)

    adj_rv, baseline = robust_transform(df, "RV", is_target=True, use_diurnal=True, winsor_window=240)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    df, har_names = generate_har_features(df, target_col="adj_RV")
    cal_names = add_calendar_features(df)
    feature_names = har_names + cal_names

    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, horizon=1)
    train_win = train_window_days * PERIODS_PER_DAY

    return X, y, dates, baselines, train_win


# ── Objective factory (for legacy run mode) ──────────────────────────────────


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

        preds = run_backtest(model_fn, X, y, train_win, refit_frequency, use_scaling=False)

        y_oos = y[train_win:]
        baselines_oos = baselines[train_win:]

        pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

        mask = (true_raw > 0) & (pred_raw > 0)
        ratio = true_raw[mask] / pred_raw[mask]
        return float(np.mean(ratio - np.log(ratio) - 1))

    return objective


# ── Subcommand: suggest ──────────────────────────────────────────────────────


def suggest_batch(
    model: str,
    batch_size: int,
    storage_path: str | None,
    output_dir: str,
    study_name: str | None = None,
) -> dict:
    """Generate a batch of candidate param sets via shotgun TPE.

    Returns the manifest dict (also written to output_dir/manifest.json).
    """
    study = _load_or_create_study(model, storage_path, study_name)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trials_info = []
    for i in range(batch_size):
        trial = study.ask(search_space=_get_search_space(model))

        fname = f"trial_{i}.json"
        with open(out / fname, "w") as f:
            json.dump(trial.params, f, indent=2)

        trials_info.append(
            {
                "id": i,
                "file": fname,
                "optuna_number": trial.number,
            }
        )
        print(f"  Trial {i} (optuna #{trial.number}): {trial.params}")

    manifest = {
        "model": model,
        "study_name": study.study_name,
        "batch_size": batch_size,
        "trials": trials_info,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {batch_size} param files + manifest.json -> {output_dir}")
    return manifest


def _get_search_space(model: str) -> dict:
    """Return Optuna search space distributions for study.ask()."""
    if model == "rf":
        return {
            "n_estimators": optuna.distributions.IntDistribution(100, 1000, step=100),
            "max_depth": optuna.distributions.IntDistribution(3, 20),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 50, log=True),
            "min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "max_features": optuna.distributions.FloatDistribution(0.3, 1.0),
            "max_samples": optuna.distributions.FloatDistribution(0.5, 1.0),
        }
    elif model == "xgb":
        return {
            "n_estimators": optuna.distributions.IntDistribution(100, 1000, step=100),
            "max_depth": optuna.distributions.IntDistribution(3, 12),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            "min_child_weight": optuna.distributions.IntDistribution(1, 20),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.3, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "reg_lambda": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
        }
    elif model == "lgbm":
        return {
            "n_estimators": optuna.distributions.IntDistribution(100, 1000, step=100),
            "max_depth": optuna.distributions.IntDistribution(3, 12),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            "num_leaves": optuna.distributions.IntDistribution(15, 127),
            "min_child_samples": optuna.distributions.IntDistribution(5, 100),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.3, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "reg_lambda": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
        }
    else:
        raise ValueError(f"Unknown model: {model}")


def cmd_suggest(args: argparse.Namespace) -> None:
    suggest_batch(
        model=args.model,
        batch_size=args.batch_size,
        storage_path=args.storage,
        output_dir=args.output_dir,
        study_name=args.study_name,
    )


# ── Subcommand: evaluate ─────────────────────────────────────────────────────


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Delegate one trial evaluation to the appropriate ml_*.py executor."""
    params_file = os.path.join(args.params_dir, f"trial_{args.trial_id}.json")
    if not os.path.isfile(params_file):
        print(f"ERROR: params file not found: {params_file}", file=sys.stderr)
        sys.exit(1)

    executor = _EXECUTOR_SCRIPTS[args.model]
    cmd = [
        sys.executable,
        executor,
        "--params-file",
        params_file,
        "--output-file",
        args.output_file,
    ]
    if args.start is not None:
        cmd += ["--start", str(args.start)]
    if args.end is not None:
        cmd += ["--end", str(args.end)]
    if args.data_path:
        cmd += ["--data-path", args.data_path]
    if args.train_window:
        cmd += ["--train-window", str(args.train_window)]
    if args.horizon:
        cmd += ["--horizon", str(args.horizon)]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ── Subcommand: score ────────────────────────────────────────────────────────


def score_trials(
    model: str,
    storage_path: str | None,
    params_dir: str,
    results_dir: str,
    output_file: str,
    study_name: str | None = None,
) -> dict:
    """Score completed trials and report to Optuna. Returns best params."""
    manifest_path = os.path.join(params_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    study = _load_or_create_study(model, storage_path, study_name)

    scored = 0
    for trial_info in manifest["trials"]:
        tid = trial_info["id"]
        optuna_num = trial_info["optuna_number"]

        trial_dir = os.path.join(results_dir, f"trial_{tid}")
        if not os.path.isdir(trial_dir):
            print(f"  Trial {tid}: no results dir, skipping")
            continue

        # Concatenate all chunk CSVs
        csvs = sorted(Path(trial_dir).glob("*.csv"))
        if not csvs:
            print(f"  Trial {tid}: no CSVs found, skipping")
            continue

        chunks = [pd.read_csv(p) for p in csvs]
        results_df = pd.concat(chunks, ignore_index=True)

        qlike = _compute_qlike(results_df)
        study.tell(optuna_num, qlike)
        scored += 1
        print(f"  Trial {tid} (optuna #{optuna_num}): QLIKE = {qlike:.6f}")

    if scored == 0:
        print("WARNING: No trials scored")
        return {}

    best = study.best_trial
    print(f"\nBest QLIKE: {best.value:.6f}")
    print(f"Best params: {best.params}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"Saved -> {output_file}")

    return dict(best.params)


def cmd_score(args: argparse.Namespace) -> None:
    score_trials(
        model=args.model,
        storage_path=args.storage,
        params_dir=args.params_dir,
        results_dir=args.results_dir,
        output_file=args.output_file,
        study_name=args.study_name,
    )


# ── Subcommand: run (legacy) ────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """Original single-machine Optuna loop."""
    X, y, dates, baselines, train_win = prepare_data(args.data_path, args.train_window)
    objective = make_objective(args.model, X, y, dates, baselines, train_win, args.refit_frequency)

    storage = _make_storage(args.storage)
    study_name = args.study_name or f"tune_{args.model}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    best_params = study.best_trial.params
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Best QLIKE: {study.best_trial.value:.6f}")
    print(f"Best params: {best_params}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for tree-based volatility models")
    sub = parser.add_subparsers(dest="command")

    # ── suggest ──
    p_suggest = sub.add_parser("suggest", help="Generate batch of candidate param sets")
    p_suggest.add_argument("--model", required=True, choices=_MODELS)
    p_suggest.add_argument("--batch-size", type=int, default=10)
    p_suggest.add_argument("--storage", default=None)
    p_suggest.add_argument("--study-name", default=None)
    p_suggest.add_argument("--output-dir", required=True)
    p_suggest.set_defaults(func=cmd_suggest)

    # ── evaluate ──
    p_eval = sub.add_parser("evaluate", help="Run one trial via ml_*.py executor")
    p_eval.add_argument("--model", required=True, choices=_MODELS)
    p_eval.add_argument("--trial-id", type=int, required=True)
    p_eval.add_argument("--params-dir", required=True)
    p_eval.add_argument("--output-file", required=True)
    p_eval.add_argument("--start", type=int, default=None)
    p_eval.add_argument("--end", type=int, default=None)
    p_eval.add_argument("--data-path", default=None)
    p_eval.add_argument("--train-window", type=int, default=None)
    p_eval.add_argument("--horizon", type=int, default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    # ── score ──
    p_score = sub.add_parser("score", help="Score trials and report to Optuna")
    p_score.add_argument("--model", required=True, choices=_MODELS)
    p_score.add_argument("--storage", default=None)
    p_score.add_argument("--study-name", default=None)
    p_score.add_argument("--params-dir", required=True)
    p_score.add_argument("--results-dir", required=True)
    p_score.add_argument("--output-file", required=True)
    p_score.set_defaults(func=cmd_score)

    # ── run (legacy) ──
    p_run = sub.add_parser("run", help="Full Optuna loop in-process (legacy)")
    p_run.add_argument("--model", required=True, choices=_MODELS)
    p_run.add_argument("--n-trials", type=int, default=50)
    p_run.add_argument("--data-path", default="all30min")
    p_run.add_argument("--train-window", type=int, default=500)
    p_run.add_argument("--refit-frequency", type=int, default=5)
    p_run.add_argument("--storage", default=None)
    p_run.add_argument("--study-name", default=None)
    p_run.add_argument("--output-file", required=True)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
