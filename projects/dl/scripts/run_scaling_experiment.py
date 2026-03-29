"""Run MBB synthetic data augmentation scaling law experiments.

Usage:
    python -m scripts.run_scaling_experiment
    python -m scripts.run_scaling_experiment --multipliers 0 1 2 5 10 50
    python -m scripts.run_scaling_experiment --results-dir results_scaling_laws --repeats 5

SGE array job usage:
    qsub -t 1-18 ... -- python -m scripts.run_scaling_experiment --task-id \$SGE_TASK_ID --total-tasks 18
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Any

import pandas as pd
import torch

from core.core.log import get_logger
from core.data import load_and_prep_data_strided
from projects.dl.backtest.gpu_engine_scaling import run_scaling_experiment
from projects.dl.config import DL_CONFIG

logger = get_logger(__name__)


def _setup_cuda_env() -> None:
    """Configure CUDA environment for optimal GPU performance."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scaling law experiments with MBB synthetic data augmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-path", type=str, default="all30min", help="Data directory.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_scaling_laws",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--multipliers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 5, 10, 50],
        help="Synthetic data multipliers to test.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per multiplier.")
    parser.add_argument(
        "--block-size",
        type=int,
        default=48,
        help="MBB block size (one trading day of 30-min bars).",
    )
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train fraction.")
    parser.add_argument("--batch-size", type=int, default=None, help="Windows per batch.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="SGE array task ID (0-based). Maps to (multiplier, repeat) pair via divmod.",
    )
    parser.add_argument(
        "--total-tasks",
        type=int,
        default=None,
        help="Total number of SGE array tasks (len(multipliers) * repeats).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _setup_cuda_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    config: dict[str, Any] = copy.deepcopy(DL_CONFIG)
    config["data_path"] = args.input_path

    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.epochs:
        config["train"]["num_epochs"] = args.epochs
    if args.learning_rate:
        config["train"]["learning_rate"] = args.learning_rate

    hparams = {
        "exog_cols": "none",
        "use_transform_exog": False,
        "use_diurnal": True,
        "allow_missing": False,
        "use_winsor": False,
    }

    # Load data
    logger.info("Loading data from '%s'", args.input_path)
    X_np, y_np, dates, baselines, features = load_and_prep_data_strided(
        hparams, config["data_path"], lag=config["model"]["context_len"]
    )
    logger.info("Data: %d samples, %d features", X_np.shape[0], X_np.shape[1])
    logger.info("Target range: [%.4f, %.4f]", y_np.min(), y_np.max())

    # Chronological train/test split
    split_idx = int(len(X_np) * args.train_frac)
    X_train, y_train = X_np[:split_idx], y_np[:split_idx]
    X_test, y_test = X_np[split_idx:], y_np[split_idx:]
    baselines_test = baselines[split_idx:]
    dates_test = dates.iloc[split_idx:]

    logger.info("Train: %d samples (%.0f%%)", len(X_train), args.train_frac * 100)
    logger.info("Test:  %d samples (%.0f%%)", len(X_test), (1 - args.train_frac) * 100)
    logger.info("Test date range: %s -> %s", dates_test.iloc[0], dates_test.iloc[-1])

    # Run experiments with incremental saving for fault tolerance
    os.makedirs(args.results_dir, exist_ok=True)

    # SGE array job mode: run a single (multiplier, repeat) pair
    if args.task_id is not None:
        n_multipliers = len(args.multipliers)
        expected_tasks = n_multipliers * args.repeats
        if args.total_tasks is not None and args.total_tasks != expected_tasks:
            logger.warning(
                "--total-tasks=%d does not match len(multipliers)*repeats=%d",
                args.total_tasks,
                expected_tasks,
            )
        if args.task_id < 0 or args.task_id >= expected_tasks:
            raise ValueError(
                f"task_id={args.task_id} out of range [0, {expected_tasks})"
            )

        mult_idx, rep = divmod(args.task_id, args.repeats)
        mult = args.multipliers[mult_idx]
        seed = mult * 1000 + rep

        logger.info("SGE array task %d: multiplier=%d, repeat=%d, seed=%d", args.task_id, mult, rep, seed)

        result = run_scaling_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            baselines_test=baselines_test,
            model_config=config["model"],
            train_config=config["train"],
            multiplier=mult,
            block_size=args.block_size,
            seed=seed,
            device=device,
        )
        result["repeat"] = rep

        csv_path = os.path.join(args.results_dir, f"scaling_result_{args.task_id}.csv")
        pd.DataFrame([result]).drop(columns=["epoch_losses"], errors="ignore").to_csv(csv_path, index=False)
        logger.info("Task %d result saved to %s (QLIKE=%.6f)", args.task_id, csv_path, result["qlike"])
        return

    # Sequential mode: run all multiplier x repeat combinations
    csv_path = os.path.join(args.results_dir, "scaling_results.csv")

    all_results: list[dict] = []
    done_keys: set[tuple[int, int]] = set()

    if os.path.exists(csv_path):
        prev_df = pd.read_csv(csv_path)
        done_keys = set(zip(prev_df["multiplier"], prev_df["repeat"], strict=False))
        all_results = prev_df.to_dict("records")
        logger.info("Resuming: %d runs already completed", len(done_keys))

    for mult in args.multipliers:
        for rep in range(args.repeats):
            if (mult, rep) in done_keys:
                logger.info("Skipping multiplier=%d, repeat=%d (already done)", mult, rep)
                continue

            seed = mult * 1000 + rep
            logger.info("=" * 60)
            logger.info("Multiplier=%d, Repeat=%d, Seed=%d", mult, rep, seed)
            logger.info("=" * 60)

            result = run_scaling_experiment(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                baselines_test=baselines_test,
                model_config=config["model"],
                train_config=config["train"],
                multiplier=mult,
                block_size=args.block_size,
                seed=seed,
                device=device,
            )
            result["repeat"] = rep
            all_results.append(result)

            # Save incrementally
            pd.DataFrame(all_results).drop(columns=["epoch_losses"], errors="ignore").to_csv(csv_path, index=False)
            logger.info("  -> QLIKE=%.6f, n_windows=%d", result["qlike"], result["n_train_windows"])

    logger.info("All results saved to %s", csv_path)

    # Print summary
    df = pd.read_csv(csv_path)
    summary = (
        df.groupby("multiplier")
        .agg(
            qlike_mean=("qlike", "mean"),
            qlike_std=("qlike", "std"),
            mse_mean=("mse", "mean"),
            mae_mean=("mae", "mean"),
            n_windows=("n_train_windows", "first"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
