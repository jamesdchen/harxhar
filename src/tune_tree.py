# Auto-generated from notebooks/tune_tree.ipynb. Do not edit by hand.

"""Optuna hyperparameter tuning for tree-based volatility models.

Subcommands (HPC-parallel workflow):
  suggest  — generate a batch of candidate param sets (shotgun TPE)
  evaluate — thin wrapper that delegates one trial to the right ml_*.py executor
  reduce   — cluster-side: per-trial CSV concat -> QLIKE -> qlike.json
  score    — read qlike.json (or CSVs), compute QLIKE, report back to Optuna
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Search spaces ────────────────────────────────────────────────────────────

_MODELS = ["rf", "xgb", "lgbm"]

_EXECUTOR_SCRIPTS = {
    "rf": "src/ml_random_forest.py",
    "xgb": "src/ml_xgboost.py",
    "lgbm": "src/ml_lightgbm.py",
}


# ── Shared helpers ───────────────────────────────────────────────────────────


def _make_storage(path: str | None):
    import optuna

    if path is None:
        return None
    if path.endswith(".db"):
        return f"sqlite:///{path}"
    return optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(path))


def _load_or_create_study(
    model: str,
    storage_path: str | None,
    study_name: str | None = None,
):
    import optuna

    storage = _make_storage(storage_path)
    name = study_name or f"tune_{model}"
    return optuna.create_study(
        study_name=name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(
            constant_liar=True,
            n_ei_candidates=96,
            gamma=lambda n: min(int(np.ceil(0.05 * n)), 15),
        ),
        direction="minimize",
        load_if_exists=True,
    )


def _compute_qlike(results_df: pd.DataFrame) -> float:
    true_raw = results_df["true_raw"].values
    pred_raw = results_df["pred_raw"].values
    mask = (true_raw > 0) & (pred_raw > 0)
    ratio = true_raw[mask] / pred_raw[mask]
    return float(np.mean(ratio - np.log(ratio) - 1))


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
        trial = study.ask(fixed_distributions=_get_search_space(model))

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
    import optuna

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
            "n_estimators": optuna.distributions.IntDistribution(100, 2000, step=100),
            "max_depth": optuna.distributions.IntDistribution(3, 12),
            "learning_rate": optuna.distributions.FloatDistribution(0.005, 0.5, log=True),
            "min_child_weight": optuna.distributions.IntDistribution(1, 50),
            "subsample": optuna.distributions.FloatDistribution(0.3, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.3, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "reg_lambda": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "gamma": optuna.distributions.FloatDistribution(0.0, 5.0),
        }
    elif model == "lgbm":
        return {
            "n_estimators": optuna.distributions.IntDistribution(100, 1000, step=100),
            "max_depth": optuna.distributions.IntDistribution(3, 12),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3, log=True),
            "num_leaves": optuna.distributions.IntDistribution(15, 60),
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
    # Look for model-specific params subdir first, then flat layout
    params_file = os.path.join(args.params_dir, args.model, f"trial_{args.trial_id}.json")
    if not os.path.isfile(params_file):
        params_file = os.path.join(args.params_dir, f"trial_{args.trial_id}.json")
    if not os.path.isfile(params_file):
        print(f"ERROR: params file not found: {params_file}", file=sys.stderr)
        sys.exit(1)

    # Default output-file from HPC dispatch env vars
    output_file = args.output_file
    if output_file is None:
        result_dir = os.environ.get("RESULT_DIR", ".")
        task_id = os.environ.get("TASK_ID", "0")
        output_file = os.path.join(result_dir, f"results_chunk_{task_id}.csv")

    executor = _EXECUTOR_SCRIPTS[args.model]
    cmd = [
        sys.executable,
        executor,
        "--params-file",
        params_file,
        "--output-file",
        output_file,
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


def _compute_trial_qlike(trial_dir: str, require_chunks: int | None = None) -> float | None:
    """Trial QLIKE from per-chunk partial reduce JSONs; fall back to CSV concat.

    Executors write ``*_reduce.json`` next to each chunk CSV via
    ``evaluation.save_chunk_reduce``. Aggregating the partials is O(chunks) of
    tiny JSON reads vs O(chunks * rows) for CSV parsing.
    """
    partials = sorted(Path(trial_dir).glob("*_reduce.json"))
    if partials and (require_chunks is None or len(partials) >= require_chunks):
        total_count = 0
        total_sum = 0.0
        for p in partials:
            with open(p) as f:
                d = json.load(f)
            total_count += d["qlike_count"]
            total_sum += d["qlike_sum"]
        if total_count == 0:
            return None
        return total_sum / total_count

    # Fallback: concatenate CSVs (slow, for legacy trial dirs without partials)
    csvs = sorted(Path(trial_dir).glob("*.csv"))
    if not csvs or (require_chunks is not None and len(csvs) < require_chunks):
        return None
    chunks = [pd.read_csv(p) for p in csvs]
    results_df = pd.concat(chunks, ignore_index=True)
    return _compute_qlike(results_df)


def reduce_trials(
    model: str,
    results_dir: str,
    require_chunks: int | None = 100,
    force: bool = False,
) -> int:
    """Cluster-side reduce: compute QLIKE per trial dir, write qlike.json.

    Idempotent: skips dirs that already have qlike.json unless force=True.
    Returns count of trials reduced.
    """
    reduced = 0
    for trial_dir in sorted(Path(results_dir).glob(f"{model}_*")):
        if not trial_dir.is_dir():
            continue
        out = trial_dir / "qlike.json"
        if out.exists() and not force:
            continue
        tid_str = trial_dir.name[len(f"{model}_") :]
        try:
            tid = int(tid_str)
        except ValueError:
            continue
        qlike = _compute_trial_qlike(str(trial_dir), require_chunks)
        if qlike is None:
            print(f"  Trial {tid}: insufficient chunks, skipping")
            continue
        with open(out, "w") as f:
            json.dump({"trial_id": tid, "qlike": qlike}, f)
        reduced += 1
        print(f"  Trial {tid}: QLIKE = {qlike:.6f} -> {out}")
    return reduced


def cmd_reduce(args: argparse.Namespace) -> None:
    n = reduce_trials(
        model=args.model,
        results_dir=args.results_dir,
        require_chunks=args.require_chunks,
        force=args.force,
    )
    print(f"\nReduced {n} trials for model={args.model}")


def score_trials(
    model: str,
    storage_path: str | None,
    params_dir: str,
    results_dir: str,
    output_file: str,
    study_name: str | None = None,
) -> dict:
    """Score completed trials and report to Optuna. Returns best params.

    Prefers pre-computed qlike.json (from cluster-side reduce); falls back to
    concatenating CSVs locally if qlike.json is missing.
    """
    # Look for model-specific manifest first, then flat layout
    manifest_path = os.path.join(params_dir, model, "manifest.json")
    if not os.path.isfile(manifest_path):
        manifest_path = os.path.join(params_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    study = _load_or_create_study(model, storage_path, study_name)

    scored = 0
    for trial_info in manifest["trials"]:
        tid = trial_info["id"]
        optuna_num = trial_info["optuna_number"]

        # Try model-specific dir (from grid run_id), then flat layout
        trial_dir = os.path.join(results_dir, f"{model}_{tid}")
        if not os.path.isdir(trial_dir):
            trial_dir = os.path.join(results_dir, f"trial_{tid}")
        if not os.path.isdir(trial_dir):
            print(f"  Trial {tid}: no results dir, skipping")
            continue

        qlike_path = os.path.join(trial_dir, "qlike.json")
        if os.path.isfile(qlike_path):
            with open(qlike_path) as f:
                qlike = json.load(f)["qlike"]
        else:
            qlike = _compute_trial_qlike(trial_dir)
            if qlike is None:
                print(f"  Trial {tid}: no qlike.json and no CSVs, skipping")
                continue

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
    p_eval.add_argument("--output-file", default=None)
    p_eval.add_argument("--start", type=int, default=None)
    p_eval.add_argument("--end", type=int, default=None)
    p_eval.add_argument("--data-path", default=None)
    p_eval.add_argument("--train-window", type=int, default=None)
    p_eval.add_argument("--horizon", type=int, default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    # ── score ──
    p_reduce = sub.add_parser("reduce", help="Cluster-side: compute per-trial QLIKE into qlike.json")
    p_reduce.add_argument("--model", required=True, choices=_MODELS)
    p_reduce.add_argument("--results-dir", required=True)
    p_reduce.add_argument("--require-chunks", type=int, default=100)
    p_reduce.add_argument("--force", action="store_true")
    p_reduce.set_defaults(func=cmd_reduce)

    p_score = sub.add_parser("score", help="Score trials and report to Optuna")
    p_score.add_argument("--model", required=True, choices=_MODELS)
    p_score.add_argument("--storage", default=None)
    p_score.add_argument("--study-name", default=None)
    p_score.add_argument("--params-dir", required=True)
    p_score.add_argument("--results-dir", required=True)
    p_score.add_argument("--output-file", required=True)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
