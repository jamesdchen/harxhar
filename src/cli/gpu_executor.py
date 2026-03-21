"""CLI entry point for GPU-based backtests (PatchTSMixer and AE+Ridge).

Usage:
    python -m src.cli.gpu_executor --experiment patchts
    python -m src.cli.gpu_executor --experiment ae_ridge --gpu-count 4
    python -m src.cli.gpu_executor --experiment patchts --output results_dl.csv
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Any

import torch

from src.core.config import AE_RIDGE_GPU_CONFIG, DL_CONFIG
from src.core.log import get_logger
from src.data import load_and_prep_data_strided

logger = get_logger(__name__)


def _setup_cuda_env() -> None:
    """Configure CUDA environment for optimal GPU performance."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def _run_patchts(args: argparse.Namespace) -> None:
    """Run PatchTSMixer GPU backtest."""
    from src.backtest.gpu_engine import run_multigpu_backtest

    config: dict[str, Any] = copy.deepcopy(DL_CONFIG)
    config["data_path"] = args.input_path

    # Apply CLI overrides
    if args.output:
        config["output_path"] = args.output
    if args.gpu_count:
        config["gpu_count"] = args.gpu_count
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.epochs:
        config["train"]["num_epochs"] = args.epochs
    if args.learning_rate:
        config["train"]["learning_rate"] = args.learning_rate
    if args.train_window:
        config["train_window"] = args.train_window

    hparams = {
        "exog_cols": "none",
        "use_transform_exog": False,
        "use_diurnal": True,
        "allow_missing": False,
        "use_winsor": False,
    }

    logger.info("Loading data from '%s'", args.input_path)
    X_np, y_np, dates, baselines, features = load_and_prep_data_strided(
        hparams, config["data_path"], lag=config["model"]["context_len"]
    )
    logger.info("Data: %d samples, %d features", X_np.shape[0], X_np.shape[1])

    results = run_multigpu_backtest(X_np, y_np, dates, baselines, config, model_module="src.models.deep_learning")
    shape = results.shape if hasattr(results, "shape") else len(results)
    logger.info("PatchTSMixer backtest complete. Results shape: %s", shape)


def _run_ae_ridge(args: argparse.Namespace) -> None:
    """Run AE+Ridge GPU backtest."""
    from src.backtest.gpu_engine import run_ae_multigpu_backtest

    config: dict[str, Any] = copy.deepcopy(AE_RIDGE_GPU_CONFIG)
    config["data_path"] = args.input_path

    # Apply CLI overrides
    if args.output:
        config["output_path"] = args.output
    if args.gpu_count:
        config["gpu_count"] = args.gpu_count
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.epochs:
        config["train"]["num_epochs"] = args.epochs
    if args.learning_rate:
        config["train"]["learning_rate"] = args.learning_rate
    if args.train_window:
        config["train_window"] = args.train_window
    if args.weights_dir:
        config["weights_dir"] = args.weights_dir

    hparams = {
        "exog_cols": "none",
        "use_transform_exog": False,
        "use_diurnal": True,
        "allow_missing": False,
        "use_winsor": True,
        "feature_type": "raw",
        "lag_scope": "global",
    }

    logger.info("Loading data from '%s'", args.input_path)
    X_np, y_np, dates, baselines, features = load_and_prep_data_strided(hparams, config["data_path"])
    config["model"]["n_features"] = X_np.shape[1]
    logger.info("Data: %d samples, %d features", X_np.shape[0], X_np.shape[1])

    if config.get("weights_dir"):
        os.makedirs(config["weights_dir"], exist_ok=True)

    results = run_ae_multigpu_backtest(X_np, y_np, dates, baselines, config)
    shape = results.shape if hasattr(results, "shape") else len(results)
    logger.info("AE+Ridge backtest complete. Results shape: %s", shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU backtest executor for deep learning models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["patchts", "ae_ridge"],
        required=True,
        help="Which GPU experiment to run.",
    )
    parser.add_argument("--input-path", type=str, default="all30min", help="Data directory.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path.")
    parser.add_argument("--gpu-count", type=int, default=None, help="Number of GPUs to use.")
    parser.add_argument("--batch-size", type=int, default=None, help="Windows per batch.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--train-window", type=int, default=None, help="Training window size (periods).")
    parser.add_argument("--weights-dir", type=str, default=None, help="Directory to save AE weights (ae_ridge only).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _setup_cuda_env()

    n_gpus = torch.cuda.device_count()
    logger.info("CUDA available: %s (%d GPUs)", torch.cuda.is_available(), n_gpus)

    if args.experiment == "patchts":
        _run_patchts(args)
    elif args.experiment == "ae_ridge":
        _run_ae_ridge(args)


if __name__ == "__main__":
    main()
