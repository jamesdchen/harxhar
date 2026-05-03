"""Push QLIKE results from a landed iteration into the Optuna study.

The campaign loop is just repeated `/submit-hpc campaign_id=xgb_optuna_2026_05`
invocations; this helper closes the feedback loop for the strategy half by
mapping each iteration's per-trial result dirs back to their Optuna trial
numbers and calling `study.tell(...)` on the QLIKE.

Run between iterations (or after the last one):

    python .hpc/campaigns/xgb_optuna_2026_05/score_iter.py <iter_index>

Idempotent — already-told trials are skipped, so re-running is safe.

Inputs (relative to repo root):
  - params/xgb_optuna_2026_05/iter_<N>/manifest.json     (trial_id -> optuna_number)
  - results/tune/xgb_optuna_2026_05/iter_<N>/            (per-trial qlike.json)

Outputs:
  - .hpc/optuna.db                                        (study.tell per trial)
  - params_archive/xgb_optuna_2026_05/best_after_iter_<N>.json
"""

from __future__ import annotations

import sys
from pathlib import Path

CAMPAIGN_ID = "xgb_optuna_2026_05"
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
OPTUNA_STORAGE = ".hpc/optuna.db"
TUNE_MODEL = "xgb"


def score_iter(iter_index: int) -> dict:
    sys.path.insert(0, str(EXPERIMENT_DIR))
    from src.tune_tree import score_trials

    iter_dir = EXPERIMENT_DIR / f"params/{CAMPAIGN_ID}/iter_{iter_index:03d}"
    results_dir = EXPERIMENT_DIR / f"results/tune/{CAMPAIGN_ID}/iter_{iter_index:03d}"
    out_file = EXPERIMENT_DIR / f"params_archive/{CAMPAIGN_ID}/best_after_iter_{iter_index:03d}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    return score_trials(
        model=TUNE_MODEL,
        storage_path=str(EXPERIMENT_DIR / OPTUNA_STORAGE),
        params_dir=str(iter_dir),
        results_dir=str(results_dir),
        output_file=str(out_file),
        min_chunks=0,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python score_iter.py <iter_index>")
    score_iter(int(sys.argv[1]))
