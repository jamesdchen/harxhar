"""Generic per-iteration scorer for `tune_<model>_<bucket>` campaigns.

Closes the loop after one iteration of a tune campaign lands: aggregates the
iteration's per-trial chunked outputs into one QLIKE/trial via
``src.tune_tree.score_trials``, then ``study.tell``s the right Optuna study.

Usage::

    python .hpc/campaigns/_score_iter.py <campaign_id> <iter_index>

Reads (relative to repo root)::

    params/<cid>/iter_<N>/manifest.json
    params/<cid>/iter_<N>/trial_<id>.json
    results/tune/<cid>/iter_<N>/<model>_<bucket>_<trial_id>/results_chunk_*.csv

Writes::

    .hpc/optuna.db                                          # study.tell
    params_archive/<cid>/best_after_iter_<N>.json           # snapshot

Idempotent — already-told trials are skipped by Optuna.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
OPTUNA_STORAGE = ".hpc/optuna.db"

_MODEL_PREFIXES = ("xgb", "lgbm", "ridge", "rf", "pcr")


def _parse_campaign(campaign_id: str) -> tuple[str, str]:
    """Parse ``tune_<model>_<bucket>`` -> (model, bucket).

    Mirrors ``.hpc/tasks.py:_parse_tune_campaign`` — pivot on a known model
    prefix so bucket strings with embedded underscores (``market_ew``,
    ``vol_demand``) parse correctly.
    """
    if not campaign_id.startswith("tune_"):
        raise ValueError(f"campaign_id must start with 'tune_': {campaign_id!r}")
    rest = campaign_id[len("tune_") :]
    for m in _MODEL_PREFIXES:
        if rest.startswith(m + "_"):
            return m, rest[len(m) + 1 :]
    raise ValueError(f"unknown model prefix in campaign_id: {campaign_id!r}")


def score_iter(campaign_id: str, iter_index: int, *, min_chunks: int = 100) -> dict:
    sys.path.insert(0, str(EXPERIMENT_DIR))
    from src.tune_tree import _study_name, score_trials

    model, bucket = _parse_campaign(campaign_id)
    iter_dir = EXPERIMENT_DIR / f"params/{campaign_id}/iter_{iter_index:03d}"
    results_dir = EXPERIMENT_DIR / f"results/tune/{campaign_id}/iter_{iter_index:03d}"
    out_file = EXPERIMENT_DIR / f"params_archive/{campaign_id}/best_after_iter_{iter_index:03d}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    return score_trials(
        model=model,
        storage_path=str(EXPERIMENT_DIR / OPTUNA_STORAGE),
        params_dir=str(iter_dir),
        results_dir=str(results_dir),
        output_file=str(out_file),
        study_name=_study_name(model, bucket),
        exog_bucket=bucket,
        min_chunks=min_chunks,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python _score_iter.py <campaign_id> <iter_index>")
    result = score_iter(sys.argv[1], int(sys.argv[2]))
    print(json.dumps(result, indent=2, default=str))
