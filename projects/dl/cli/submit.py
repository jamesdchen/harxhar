"""
DL experiment submission via shared HPC backend.

Submits GPU backtest array jobs using the same backend infrastructure as ML.

Usage:
    python -m projects.dl.cli.submit --experiment patchts --total-chunks 10
    python -m projects.dl.cli.submit --experiment ae_ridge --auto-chunks
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from core.backends import get_backend
from core.core.log import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TASKS_PER_ARRAY = 100
DL_SLURM_SCRIPT = str(PROJECT_ROOT / "projects" / "dl" / "infra" / "slurm" / "submit_gpu.slurm")
DL_SGE_SCRIPT = str(PROJECT_ROOT / "projects" / "dl" / "infra" / "sge" / "submit_gpu.sh")
DL_SGE_PASS_ENV_KEYS = (
    "EXPERIMENT", "RESULT_DIR", "TOTAL_CHUNKS", "INPUT_PATH", "GPU_COUNT",
    "BATCH_SIZE", "EPOCHS", "LEARNING_RATE", "TRAIN_WINDOW",
    "CONTEXT_LEN", "PATCH_LEN", "STRIDE", "WEIGHTS_DIR",
)


def build_job_env(
    experiment: str,
    result_dir: str,
    total_chunks: int,
    *,
    input_path: str = "all30min",
    gpu_count: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    train_window: int | None = None,
    context_len: int | None = None,
    patch_len: int | None = None,
    stride: int | None = None,
    weights_dir: str | None = None,
) -> dict[str, str]:
    """Build the env dict that submit_gpu.slurm expects."""
    env = os.environ.copy()
    env["EXPERIMENT"] = experiment
    env["RESULT_DIR"] = result_dir
    env["TOTAL_CHUNKS"] = str(total_chunks)
    env["INPUT_PATH"] = input_path
    if gpu_count is not None:
        env["GPU_COUNT"] = str(gpu_count)
    if batch_size is not None:
        env["BATCH_SIZE"] = str(batch_size)
    if epochs is not None:
        env["EPOCHS"] = str(epochs)
    if learning_rate is not None:
        env["LEARNING_RATE"] = str(learning_rate)
    if train_window is not None:
        env["TRAIN_WINDOW"] = str(train_window)
    if context_len is not None:
        env["CONTEXT_LEN"] = str(context_len)
    if patch_len is not None:
        env["PATCH_LEN"] = str(patch_len)
    if stride is not None:
        env["STRIDE"] = str(stride)
    if weights_dir is not None:
        env["WEIGHTS_DIR"] = weights_dir
    return env


def get_sample_count(input_path: str) -> int:
    """Get post-filter sample count by running the data cleaning pipeline."""
    from core.data.loading import load_and_clean_base_data

    hparams = {
        "use_transform_exog": False,
        "use_diurnal": True,
        "allow_missing": False,
        "use_winsor": False,
    }
    df, _ = load_and_clean_base_data(hparams, input_path)
    return len(df)


def estimate_total_chunks(
    experiment: str,
    input_path: str = "all30min",
    walltime: int | None = None,
    train_window: int | None = None,
) -> int:
    """Auto-calculate optimal chunk count from data size and experiment type.

    Targets per-task computation ≈ 2× startup overhead, balancing parallelism
    against wasted GPU-hours on repeated startup.  Result is clamped to [1, 100].
    """
    from projects.dl.config import AE_RIDGE_GPU_CONFIG, CHUNK_SIZING, DL_CONFIG

    sizing = CHUNK_SIZING[experiment]

    cfg: dict = DL_CONFIG if experiment == "patchts" else AE_RIDGE_GPU_CONFIG
    tw = train_window or cfg["train_window"]
    pred_len: int = cfg.get("model", {}).get("prediction_length", 1)

    total_samples = get_sample_count(input_path)
    num_windows = total_samples - tw - (pred_len - 1)
    total_compute = num_windows * sizing["per_window_seconds"]
    startup = sizing["startup_overhead"]

    # Target: each task's compute time ≈ 2× startup → ~67% efficiency
    target_task_compute = 2 * startup
    total_chunks = max(1, min(100, math.ceil(total_compute / target_task_compute)))

    logger.info(
        "Auto-chunks: %d samples, %d windows, %.1fs total compute, "
        "%ds startup → %d chunks (%.0fs/task, %.0f%% efficiency)",
        total_samples,
        num_windows,
        total_compute,
        startup,
        total_chunks,
        total_compute / total_chunks,
        100 * total_compute / (total_compute + total_chunks * startup),
    )
    return total_chunks


def submit_dl_experiment(
    experiment: str,
    result_dir: str,
    total_chunks: int,
    tasks_per_array: int = DEFAULT_TASKS_PER_ARRAY,
    backend_name: str = "slurm",
    **env_kwargs,
) -> str:
    """Submit a DL GPU backtest as a SLURM array job."""
    result_path = Path(result_dir).resolve()
    result_path.mkdir(parents=True, exist_ok=True)

    job_name = f"dl_{experiment}"
    job_env = build_job_env(experiment, str(result_path), total_chunks, **env_kwargs)

    logger.info(
        "Submitting %s: %d chunks to %s",
        experiment.upper(),
        total_chunks,
        result_path,
    )

    if backend_name in ("sge", "sge-remote"):
        backend = get_backend(backend_name, script=DL_SGE_SCRIPT, pass_env_keys=DL_SGE_PASS_ENV_KEYS)
    else:
        backend = get_backend(backend_name, script=DL_SLURM_SCRIPT)
    backend.submit_array(job_name, total_chunks, tasks_per_array, job_env)

    logger.info("Submitted %s (%d array tasks).", experiment, total_chunks)
    return str(result_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit DL GPU backtest as SLURM array job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["patchts", "ae_ridge"],
        required=True,
        help="Which DL experiment to run.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Output directory for chunk results. Defaults to results/dl_<experiment>.",
    )
    chunk_group = parser.add_mutually_exclusive_group()
    chunk_group.add_argument("--total-chunks", type=int, default=None, help="Number of array tasks / chunks.")
    chunk_group.add_argument(
        "--auto-chunks", action="store_true", help="Auto-calculate chunk count from data size and experiment."
    )
    parser.add_argument(
        "--walltime", type=int, default=None, help="Walltime in seconds (default: 6h). Used with --auto-chunks."
    )
    parser.add_argument("--tasks-per-array", type=int, default=DEFAULT_TASKS_PER_ARRAY, help="Max tasks per sbatch.")
    parser.add_argument("--backend", type=str, default="slurm", help="HPC backend (slurm, sge, dry-run).")
    parser.add_argument("--input-path", type=str, default="all30min", help="Data directory.")
    parser.add_argument("--gpu-count", type=int, default=None, help="GPUs per task.")
    parser.add_argument("--batch-size", type=int, default=None, help="Windows per batch.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--train-window", type=int, default=None, help="Training window size.")
    parser.add_argument("--context-len", type=int, default=None, help="Context length (patchts only).")
    parser.add_argument("--patch-len", type=int, default=None, help="Patch length (patchts only).")
    parser.add_argument("--stride", type=int, default=None, help="Stride between patches (patchts only).")
    parser.add_argument("--weights-dir", type=str, default=None, help="AE weights directory (ae_ridge only).")

    args = parser.parse_args()

    env_kwargs = {}
    for key in (
        "input_path",
        "gpu_count",
        "batch_size",
        "epochs",
        "learning_rate",
        "train_window",
        "context_len",
        "patch_len",
        "stride",
        "weights_dir",
    ):
        val = getattr(args, key)
        if val is not None:
            env_kwargs[key] = val

    # Resolve total chunks
    if args.auto_chunks:
        total_chunks = estimate_total_chunks(
            experiment=args.experiment,
            input_path=args.input_path,
            walltime=args.walltime,
            train_window=env_kwargs.get("train_window"),
        )
    elif args.total_chunks is not None:
        total_chunks = args.total_chunks
    else:
        total_chunks = 10  # legacy default

    result_dir = args.result_dir or str(PROJECT_ROOT / "results" / f"dl_{args.experiment}")

    submit_dl_experiment(
        experiment=args.experiment,
        result_dir=result_dir,
        total_chunks=total_chunks,
        tasks_per_array=args.tasks_per_array,
        backend_name=args.backend,
        **env_kwargs,
    )


if __name__ == "__main__":
    main()
