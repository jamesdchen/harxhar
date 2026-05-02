# Auto-generated from notebooks/07_tune_tree.ipynb. Do not edit by hand.

"""Optuna helpers for tree-model hyperparameter tuning.

Library module — used by .hpc/tasks.py inside a /campaign-hpc-style
closed loop. The CLI subcommands (suggest/evaluate/score) that this
module used to ship were retired 2026-05-02; their orchestration role
now lives in /campaign-hpc + a campaign-aware tasks.py that reads
prior(experiment_dir, campaign_id) to drive Optuna ask/tell.

Public surface:
    suggest_batch(model, batch_size, ...)        — Optuna ask + persist trials
    score_trials(model, storage_path, ...)        — read per-trial outputs, tell()
    _get_search_space(model)                       — TPE search-space dict
    _compute_qlike(df)                             — QLIKE on a one-trial result frame
    reduce_trials(...)                             — aggregate per-trial QLIKEs
    _make_storage / _study_name / _load_or_create_study  — Optuna study mgmt

A campaign-aware tasks.py example for tuning ml_xgboost over 100 trials
in batches of 10:

    from src.tune_tree import suggest_batch, _load_or_create_study
    from hpc_mapreduce.reduce.history import prior
    import os
    _PRIOR = prior(".", os.environ["HPC_CAMPAIGN_ID"])
    if len(_PRIOR) >= 10:                           # 10 batches done -> stop
        _TASKS = []
    else:
        study = _load_or_create_study("xgboost", storage_path=".hpc/optuna.db")
        _TASKS = [study.ask().params for _ in range(10)]
"""

from __future__ import annotations

import json
import os
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
    if path.startswith(("sqlite:", "postgresql:", "mysql:")):
        return path
    if path.endswith(".db"):
        return f"sqlite:///{path}"
    return optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(path))


def _study_name(model: str, exog_bucket: str | None) -> str:
    return f"tune_{model}_{exog_bucket}" if exog_bucket else f"tune_{model}"


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


def score_trials(
    model: str,
    storage_path: str | None,
    params_dir: str,
    results_dir: str,
    output_file: str,
    study_name: str | None = None,
    exog_bucket: str | None = None,
    min_chunks: int = 100,
) -> dict:
    """Score completed trials and report to Optuna. Returns best params.

    Prefers pre-computed qlike.json (from cluster-side reduce); falls back to
    concatenating CSVs locally if qlike.json is missing.

    Skips any trial whose result dir contains fewer than `min_chunks`
    `results_chunk_*.csv` files — partial-data QLIKEs are biased and would
    contaminate the optuna study if reported.
    """
    # Try bucket-scoped manifest (exog runs), then model-specific, then flat
    manifest_path = None
    if exog_bucket:
        cand = os.path.join(params_dir, f"{model}_{exog_bucket}", "manifest.json")
        if os.path.isfile(cand):
            manifest_path = cand
    if manifest_path is None:
        cand = os.path.join(params_dir, model, "manifest.json")
        if os.path.isfile(cand):
            manifest_path = cand
    if manifest_path is None:
        manifest_path = os.path.join(params_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    study = _load_or_create_study(model, storage_path, study_name)

    scored = 0
    for trial_info in manifest["trials"]:
        tid = trial_info["id"]
        optuna_num = trial_info["optuna_number"]

        # Try bucket-scoped dir (exog runs), then model-specific, then flat
        trial_dir = None
        if exog_bucket:
            cand = os.path.join(results_dir, f"{model}_{exog_bucket}_{tid}")
            if os.path.isdir(cand):
                trial_dir = cand
        if trial_dir is None:
            trial_dir = os.path.join(results_dir, f"{model}_{tid}")
            if not os.path.isdir(trial_dir):
                trial_dir = os.path.join(results_dir, f"trial_{tid}")
        if not os.path.isdir(trial_dir):
            print(f"  Trial {tid}: no results dir, skipping")
            continue

        n_chunks = sum(1 for f in os.listdir(trial_dir) if f.startswith("results_chunk_") and f.endswith(".csv"))
        if n_chunks < min_chunks:
            print(f"  Trial {tid}: incomplete ({n_chunks}/{min_chunks} chunks), skipping")
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

        # Idempotent: skip if this trial was already told previously.
        existing = next((t for t in study.trials if t.number == optuna_num), None)
        if existing is not None and existing.state.is_finished():
            scored += 1
            print(f"  Trial {tid} (optuna #{optuna_num}): QLIKE = {existing.value:.6f}  (already-told)")
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
    trial_prefix: str | None = None,
) -> int:
    """Cluster-side reduce: compute QLIKE per trial dir, write qlike.json.

    Trial dirs are any subdirs of ``results_dir`` matching ``{trial_prefix}*``
    (default ``{model}_``). The suffix is kept as a string, so arbitrary
    naming like ``lgbm_jitter_rf1_0`` or ``lgbm_replay_libdef`` works.

    Idempotent: skips dirs that already have qlike.json unless force=True.
    Returns count of trials reduced.
    """
    prefix = trial_prefix if trial_prefix is not None else f"{model}_"
    reduced = 0
    for trial_dir in sorted(Path(results_dir).glob(f"{prefix}*")):
        if not trial_dir.is_dir():
            continue
        out = trial_dir / "qlike.json"
        if out.exists() and not force:
            continue
        tid = trial_dir.name[len(prefix):]
        qlike = _compute_trial_qlike(str(trial_dir), require_chunks)
        if qlike is None:
            print(f"  Trial {tid}: insufficient chunks, skipping")
            continue
        with open(out, "w") as f:
            json.dump({"trial_id": tid, "qlike": qlike}, f)
        reduced += 1
        print(f"  Trial {tid}: QLIKE = {qlike:.6f} -> {out}")
    return reduced
