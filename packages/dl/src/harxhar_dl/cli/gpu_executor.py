"""CLI entry point for GPU-based backtests (PatchTSMixer and AE+Ridge).

Usage:
    python -m harxhar_dl.cli.gpu_executor --experiment patchts
    python -m harxhar_dl.cli.gpu_executor --experiment ae_ridge --gpu-count 4
    python -m harxhar_dl.cli.gpu_executor --experiment patchts --output results_dl.csv
"""

from __future__ import annotations

import argparse
import copy
import os
import signal
import time
import traceback
from typing import Any

import torch
from harxhar_core.core.log import get_logger
from harxhar_core.data import load_and_prep_data_strided

from harxhar_dl.config import AE_RIDGE_GPU_CONFIG, DL_CONFIG

logger = get_logger(__name__)


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout("Run exceeded timeout")


def _setup_cuda_env() -> None:
    """Configure CUDA environment for optimal GPU performance."""
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def _run_patchts(args: argparse.Namespace) -> None:
    """Run PatchTSMixer GPU backtest."""
    from harxhar_dl.backtest.gpu_engine import run_multigpu_backtest

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

    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
        config["checkpoint_every"] = args.checkpoint_every
    if args.loss_log_path:
        config["loss_log_path"] = args.loss_log_path
    if args.progress_path:
        config["progress_path"] = args.progress_path

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

    results = run_multigpu_backtest(
        X_np, y_np, dates, baselines, config, model_module="harxhar_dl.models.deep_learning"
    )
    shape = results.shape if hasattr(results, "shape") else len(results)
    logger.info("PatchTSMixer backtest complete. Results shape: %s", shape)


def _run_ae_ridge(args: argparse.Namespace) -> None:
    """Run AE+Ridge GPU backtest."""
    from harxhar_dl.backtest.gpu_engine import run_ae_multigpu_backtest

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
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
        config["checkpoint_every"] = args.checkpoint_every
    if args.loss_log_path:
        config["loss_log_path"] = args.loss_log_path
    if args.progress_path:
        config["progress_path"] = args.progress_path

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
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Directory for training checkpoints (enables crash recovery)."
    )
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N chunks (0=disabled).")
    parser.add_argument("--loss-log-path", type=str, default=None, help="CSV path to save per-epoch training losses.")
    parser.add_argument("--progress-path", type=str, default=None, help="JSON path to write live training progress.")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon.")
    parser.add_argument("--timeout-hours", type=float, default=0, help="Max runtime in hours (0=no limit).")
    parser.add_argument("--write-status", action="store_true", help="Write Drive status JSON for MCP monitoring.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Conditional import — write_status is only needed when --write-status is set
    if args.write_status:
        from harxhar_dl.notebook_utils import write_status
    else:

        def write_status(status: str, **kw: Any) -> dict:
            return {}  # no-op when status tracking is disabled

    _setup_cuda_env()

    n_gpus = torch.cuda.device_count()
    logger.info("CUDA available: %s (%d GPUs)", torch.cuda.is_available(), n_gpus)

    # Timeout guard
    if args.timeout_hours > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(args.timeout_hours * 3600))

    t0 = time.time()
    try:
        write_status("running", experiment=args.experiment, horizon=args.horizon, pid=os.getpid())

        if args.experiment == "patchts":
            _run_patchts(args)
        elif args.experiment == "ae_ridge":
            _run_ae_ridge(args)

        elapsed_min = (time.time() - t0) / 60
        write_status("finished_run", elapsed_minutes=round(elapsed_min, 1))
        logger.info("Done in %.1f min", elapsed_min)

    except _Timeout:
        write_status("failed", error="timeout", timeout_hours=args.timeout_hours)
        logger.error("FAILED: timeout exceeded (%.1fh)", args.timeout_hours)
        raise SystemExit(1) from None
    except Exception as exc:
        write_status("failed", error=str(exc), traceback=traceback.format_exc()[-1000:])
        logger.error("FAILED: %s", exc)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
