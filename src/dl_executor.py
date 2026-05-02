# Auto-generated from notebooks/05b_dl_executor.ipynb. Do not edit by hand.

"""Shared DL-executor boilerplate.

Pulled out of `src/dl_ae_ridge.py` and `src/dl_patchts.py`; both
were duplicating the same RNG seeding, the same CLI flag set, and
the same smearing/save block. This module owns those three things;
the model+training+backtest logic stays in the per-method files.
"""

from __future__ import annotations

import json
import logging
import os
import random

import numpy as np
import pandas as pd
import torch

from src.evaluation import apply_duan_smearing, calculate_metrics

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Pin RNGs for reproducibility (numpy, torch, cuda, cudnn).

    Call at the top of main() BEFORE any data loading or model construction.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_dl_results(
    preds,
    y_chunk,
    dates_chunk,
    baselines_chunk,
    train_window: int,
    horizon: int,
    output_file: str,
) -> None:
    """Apply Duan smearing, build result DataFrame, write CSV + metrics.json.

    Slices ``y_chunk``, ``dates_chunk``, ``baselines_chunk`` to the OOS
    window (``train_window : train_window + len(preds)``), applies Duan
    smearing via :func:`apply_duan_smearing`, builds the canonical
    6-column result frame, writes the CSV, computes metrics with
    :func:`calculate_metrics`, and writes ``metrics.json`` next to the
    output file.
    """
    num_windows = len(preds)
    y_oos = y_chunk[train_window : train_window + num_windows]
    dates_oos = dates_chunk.iloc[train_window : train_window + num_windows].values
    baselines_oos = baselines_chunk[train_window : train_window + num_windows]

    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

    results = pd.DataFrame(
        {
            "date": dates_oos,
            "horizon": horizon,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )

    out_dir = os.path.dirname(output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(output_file, index=False)

    metrics = calculate_metrics(results)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    logger.info(f"Saved {len(results)} rows -> {output_file}")
