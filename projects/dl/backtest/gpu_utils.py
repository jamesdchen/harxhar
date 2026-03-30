"""Shared utilities for GPU backtest pipelines."""

import csv
import gc
import importlib
import json
import logging
import math
import os
import tempfile
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from hpc.chunking import chunk_context

from core.backtest.engine import build_results_dataframe, extract_subset, save_chunk_results
from core.core.log import get_logger
from projects.dl import config as cfg

logger = get_logger(__name__)


def normalize_chunks(
    X_chunk: torch.Tensor,
    X_test_chunk: torch.Tensor,
    dim: int,
    use_train_stats_for_test: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize training and test chunks using mean/std normalization.

    Parameters
    ----------
    dim : int
        Reduction dimension for mean/std.
    use_train_stats_for_test : bool
        If True, normalize test data with training statistics.
        If False, normalize test data with its own statistics.
    """
    mean = X_chunk.mean(dim=dim, keepdim=True)
    std = X_chunk.std(dim=dim, keepdim=True) + cfg.NORM_EPS
    X_chunk = (X_chunk - mean) / std

    if use_train_stats_for_test:
        X_test_chunk = (X_test_chunk - mean) / std
    else:
        t_mean = X_test_chunk.mean(dim=dim, keepdim=True)
        t_std = X_test_chunk.std(dim=dim, keepdim=True) + cfg.NORM_EPS
        X_test_chunk = (X_test_chunk - t_mean) / t_std

    return X_chunk, X_test_chunk


def log_to_file(message: str) -> None:
    """Log message to the GPU worker log file via a FileHandler."""
    file_logger = logging.getLogger("gpu_worker")
    if not file_logger.handlers:
        handler = logging.FileHandler(cfg.GPU_WORKER_LOG)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
        file_logger.addHandler(handler)
        file_logger.setLevel(logging.INFO)
    file_logger.info(message)


def setup_device(gpu_id: int) -> torch.device:
    """Create CUDA device and configure precision settings."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    return device


def load_model(model_module: str, factory_fn: str, model_config: dict, device: torch.device) -> tuple:
    """
    Load a model via importlib and create train/eval instances.

    Parameters
    ----------
    model_module : str
        Dotted module path (e.g. 'src.dl_models').
    factory_fn : str
        Name of the factory function in the module (e.g. 'get_model').
    model_config : dict
        Config dict passed to the factory function.
    device : torch.device

    Returns
    -------
    base_model_train, base_model_eval, buffers, param_keys
    """
    mod = importlib.import_module(model_module)
    factory = getattr(mod, factory_fn)

    base_model_train = factory(model_config).to(device)
    base_model_eval = factory(model_config).to(device)
    base_model_eval.eval()

    buffers = {n: b for n, b in base_model_train.named_buffers()}
    param_keys = [n for n, _ in base_model_train.named_parameters()]

    return base_model_train, base_model_eval, buffers, param_keys


def allocate_params(base_model: torch.nn.Module, max_batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Pre-allocate batched parameter tensors for vmapped training."""
    param_shapes = {n: p.shape for n, p in base_model.named_parameters()}
    params_store = {}
    for name, shape in param_shapes.items():
        if len(shape) > 1:
            p = torch.empty((max_batch_size, *shape), device=device)
        else:
            p = torch.zeros((max_batch_size, *shape), device=device)
        params_store[name] = p.requires_grad_(True)
    return params_store


def init_params(params_store: dict[str, torch.Tensor], curr_bs: int, max_batch_size: int) -> dict[str, torch.Tensor]:
    """
    Re-initialize parameters with fan-in uniform for weights, zero for biases.
    Returns current_params (sliced if curr_bs < max_batch_size).
    """
    for _name, p_tensor in params_store.items():
        fan_in = p_tensor.shape[-1] if len(p_tensor.shape) > 2 else 1
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.01

        if len(p_tensor.shape) > 2:
            p_tensor.data.uniform_(-bound, bound)
        else:
            p_tensor.data.zero_()

    if curr_bs < max_batch_size:
        return {k: v[:curr_bs] for k, v in params_store.items()}
    return params_store


def init_adam_state(current_params, device):
    """Create zeroed Adam optimizer state dicts."""
    exp_avgs = {}
    exp_avg_sqs = {}
    step_tensors = {}
    for k, p in current_params.items():
        exp_avgs[k] = torch.zeros_like(p)
        exp_avg_sqs[k] = torch.zeros_like(p)
        step_tensors[k] = torch.tensor(0.0, device=device)
    return exp_avgs, exp_avg_sqs, step_tensors


def save_checkpoint(checkpoint_dir, gpu_id, chunk_idx, params_store, results_so_far):
    """Save training checkpoint: params + completed chunk results."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_gpu{gpu_id}.pt")
    state = {
        "chunk_idx": chunk_idx,
        "params_store": {k: v.cpu() for k, v in params_store.items()},
        "results": results_so_far,
    }
    torch.save(state, path)
    log_to_file(f"Worker {gpu_id}: Saved checkpoint at chunk {chunk_idx}")


def load_checkpoint(checkpoint_dir, gpu_id, device):
    """Load checkpoint if it exists. Returns (start_chunk_idx, params_state, results) or None."""
    path = os.path.join(checkpoint_dir, f"ckpt_gpu{gpu_id}.pt")
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location=device, weights_only=False)
    log_to_file(f"Worker {gpu_id}: Resuming from chunk {state['chunk_idx']}")
    return state


def save_loss_log(loss_log_path, chunk_idx, epoch_losses_cpu):
    """Append per-epoch losses for a chunk to a CSV file.

    Parameters
    ----------
    loss_log_path : str
        Path to the output CSV.
    chunk_idx : int
        Window/chunk index for this training run.
    epoch_losses_cpu : np.ndarray
        1-D array of per-epoch mean losses.
    """
    write_header = not os.path.exists(loss_log_path)
    with open(loss_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["chunk", "epoch", "loss"])
        for epoch, loss_val in enumerate(epoch_losses_cpu, start=1):
            writer.writerow([chunk_idx, epoch, f"{loss_val:.6g}"])


def run_kernel_and_detach(kernel, params, buffers, exp_avgs, exp_avg_sqs, step_tensors, X, y):
    """Run compiled training kernel and detach the resulting parameters.

    Returns
    -------
    params : dict[str, Tensor]
        Detached trained parameters.
    epoch_losses : Tensor
        1-D tensor of per-epoch mean losses.
    """
    final_params, epoch_losses = kernel(params, buffers, exp_avgs, exp_avg_sqs, step_tensors, X, y)
    detached = {k: v.detach().requires_grad_(True) for k, v in final_params.items()}
    return detached, epoch_losses.detach()


def aggregate_predictions(results_nested: list[list[dict]], num_windows: int) -> np.ndarray:
    """Flatten, sort, concatenate predictions from all GPU workers.

    Returns
    -------
    preds : np.ndarray
        Shape (num_windows,) for single-horizon or (num_windows, H) for multi-horizon.
    """
    flat_results = [item for sublist in results_nested for item in sublist]
    flat_results.sort(key=lambda x: x["chunk_index"])
    preds = torch.cat([r["predictions"] for r in flat_results]).numpy()

    # Multi-horizon: preds may be (N, H) — check first dim
    n_rows = preds.shape[0] if preds.ndim == 1 else preds.shape[0]
    if n_rows != num_windows:
        logger.warning("Prediction shape mismatch. Expected %d, got %d.", num_windows, n_rows)
        preds = preds[:num_windows]

    return preds


def build_results_df(preds, test_indices, y_np, dates, baselines, output_file=None):
    """
    Apply Duan's smearing, optionally save, and return results DataFrame(s).

    For multi-horizon predictions (preds.ndim == 2, shape (N, H)), saves
    separate CSV per horizon and returns a dict {horizon: DataFrame}.
    """
    if preds.ndim == 2 and preds.shape[1] > 1:
        H = preds.shape[1]
        results = {}
        for h in range(H):
            # For horizon h+1, targets are shifted by h from test_indices
            h_indices = test_indices[: len(test_indices) - h] if h > 0 else test_indices
            h_target_indices = h_indices + h
            # Ensure target indices don't exceed data length
            valid = h_target_indices < len(y_np)
            h_indices = h_indices[valid]
            h_target_indices = h_target_indices[valid]
            h_preds = preds[: len(h_indices), h]

            if output_file is not None:
                import os

                base, ext = os.path.splitext(output_file)
                h_file = f"{base}_h{h + 1}{ext}"
                save_chunk_results(
                    output_file=h_file,
                    forecasts=h_preds,
                    indices=h_target_indices,
                    train_window=test_indices[0],
                    y_true=y_np,
                    dates=dates,
                    baselines=baselines,
                    horizon=h + 1,
                )

            dates_subset = extract_subset(dates, h_indices)
            y_subset = y_np[h_target_indices]
            base_subset = baselines[h_target_indices]
            results[h + 1] = build_results_dataframe(h_preds, y_subset, dates_subset, base_subset, horizon=h + 1)

        return results

    # Single-horizon path (backward compatible)
    if output_file is not None:
        save_chunk_results(
            output_file=output_file,
            forecasts=preds,
            indices=test_indices,
            train_window=test_indices[0],
            y_true=y_np,
            dates=dates,
            baselines=baselines,
        )

    dates_subset = extract_subset(dates, test_indices)
    y_subset = y_np[test_indices]
    base_subset = baselines[test_indices]
    return build_results_dataframe(preds, y_subset, dates_subset, base_subset)


class _WorkerError:
    """Lightweight picklable sentinel returned when a worker fails.

    Using this instead of raising across the ``spawn`` boundary avoids
    the ``MaybeEncodingError: cannot pickle 'cell' object`` that occurs
    when multiprocessing tries to pickle exception tracebacks whose
    frames reference closures (e.g. ``setup_fn`` / ``chunk_fn``).
    """

    def __init__(self, message: str):
        self.message = message


def _safe_worker(worker_fn, *args):
    """Call *worker_fn* and return a ``_WorkerError`` on failure.

    This is a **module-level** function (no closure cells) so that the
    ``_WorkerError`` it returns is always picklable.
    """
    try:
        return worker_fn(*args)
    except Exception:
        import traceback

        return _WorkerError(traceback.format_exc())


def distribute_and_run(worker_fn, worker_args_fn, gpu_count, num_windows, chunk_size):
    """
    Distribute chunks across GPUs and run workers via multiprocessing.

    Parameters
    ----------
    worker_fn : callable
        The GPU worker function.
    worker_args_fn : callable(gpu_id, chunk_indices) -> tuple
        Returns the full args tuple for worker_fn given gpu_id and its chunks.
    gpu_count : int
    num_windows : int
    chunk_size : int

    Returns
    -------
    results_nested : list of lists (one per GPU)
    """
    num_chunks = math.ceil(num_windows / chunk_size)
    chunk_indices = list(range(num_chunks))
    chunks_per_gpu = [chunk_indices[i::gpu_count] for i in range(gpu_count)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=gpu_count) as pool:
        args = [worker_args_fn(gpu_id, chunks_per_gpu[gpu_id]) for gpu_id in range(gpu_count)]
        safe_args = [(worker_fn,) + a for a in args]
        results_nested = pool.starmap(_safe_worker, safe_args)

    # Re-raise worker errors in the main process (no closures here).
    for r in results_nested:
        if isinstance(r, _WorkerError):
            raise RuntimeError(r.message)

    return results_nested


def save_losses_csv(all_losses, loss_log_path):
    """Write collected chunk losses to a single CSV.

    Parameters
    ----------
    all_losses : list[dict]
        Each dict has keys 'chunk_index' and 'epoch_losses' (np.ndarray).
    loss_log_path : str
    """
    all_losses.sort(key=lambda x: x["chunk_index"])
    with open(loss_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "epoch", "loss"])
        for entry in all_losses:
            for epoch, loss_val in enumerate(entry["epoch_losses"], start=1):
                writer.writerow([entry["chunk_index"], epoch, f"{loss_val:.6g}"])
    logger.info("Saved training losses to %s", loss_log_path)


def _write_progress(progress_path, data):
    """Atomically write progress JSON so readers never see partial writes."""
    dir_name = os.path.dirname(progress_path) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp", prefix="prog_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp_path, progress_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def run_worker(
    gpu_id,
    chunk_indices,
    shared_X,
    shared_y,
    shared_test_X,
    chunk_size,
    total_windows,
    setup_fn,
    chunk_fn,
    checkpoint_dir=None,
    checkpoint_every=0,
    progress_path=None,
):
    """
    Generic per-GPU worker loop with optional checkpointing.

    Parameters
    ----------
    setup_fn : callable(device) -> (params_store, ctx_dict)
        Initialise model, compile kernels, return pre-allocated params_store
        and an arbitrary context dict passed through to chunk_fn.
    chunk_fn : callable(ctx, current_params, X_chunk, y_chunk, X_test_chunk, curr_bs, idx)
        Model-specific logic: normalise, train, predict.
        Must return a 1-D predictions tensor on the same device.
    checkpoint_dir : str | None
        If set, save checkpoints to this directory.
    checkpoint_every : int
        Save a checkpoint every N chunks (0 = disabled).
    """
    results = []
    start_from = 0
    chunk_times: list[float] = []
    try:
        device = setup_device(gpu_id)
        params_store, ctx = setup_fn(device)

        # Resume from checkpoint if available
        if checkpoint_dir:
            ckpt = load_checkpoint(checkpoint_dir, gpu_id, device)
            if ckpt is not None:
                # Restore params
                for k, v in ckpt["params_store"].items():
                    params_store[k].data.copy_(v.to(device))
                results = ckpt["results"]
                # Skip already-completed chunks
                completed = {r["chunk_index"] for r in results}
                start_from = 0
                for j, idx in enumerate(chunk_indices):
                    if idx not in completed:
                        start_from = j
                        break
                else:
                    start_from = len(chunk_indices)
                log_to_file(f"Worker {gpu_id}: Skipping {start_from} already-completed chunks")

        total_chunks = len(chunk_indices)
        worker_t0 = time.monotonic()

        for i, idx in enumerate(chunk_indices[start_from:], start=start_from):
            chunk_t0 = time.monotonic()

            start = idx * chunk_size
            end = min(start + chunk_size, total_windows)

            X_chunk = shared_X[start:end].to(device, non_blocking=True)
            y_chunk = shared_y[start:end].to(device, non_blocking=True)
            X_test_chunk = shared_test_X[start:end].to(device, non_blocking=True)

            curr_bs = X_chunk.shape[0]
            if curr_bs == 0:
                continue

            current_params = init_params(params_store, curr_bs, chunk_size)
            preds = chunk_fn(ctx, current_params, X_chunk, y_chunk, X_test_chunk, curr_bs, idx)

            # Flatten per-window predictions, preserving horizon dim if present
            if preds.dim() > 1 and preds.shape[-1] > 1:
                # Multi-horizon: shape (batch, H) → keep as-is
                preds_cpu = preds.cpu()
            else:
                preds_cpu = preds.reshape(-1).cpu()

            # Free GPU memory from this chunk before the next iteration
            del X_chunk, y_chunk, X_test_chunk, preds, current_params

            chunk_elapsed = time.monotonic() - chunk_t0
            chunk_times.append(chunk_elapsed)

            results.append(
                {
                    "chunk_index": idx,
                    "predictions": preds_cpu,
                }
            )

            chunks_done = i + 1
            chunks_remaining = total_chunks - chunks_done
            avg_chunk_sec = sum(chunk_times) / len(chunk_times)
            eta_sec = chunks_remaining * avg_chunk_sec
            wall_elapsed = time.monotonic() - worker_t0

            if i % 10 == 0:
                log_to_file(
                    f"Worker {gpu_id}: Chunk {idx} done "
                    f"({chunks_done}/{total_chunks}, "
                    f"avg {avg_chunk_sec:.1f}s/chunk, "
                    f"ETA {eta_sec / 60:.1f}min)"
                )

            # Write progress file (only from gpu 0 to avoid contention)
            if progress_path and gpu_id == 0:
                # Use last 10 chunk times for recent pace
                recent = chunk_times[-10:]
                recent_avg = sum(recent) / len(recent)
                _write_progress(
                    progress_path,
                    {
                        "chunks_done": chunks_done,
                        "chunks_total": total_chunks,
                        "avg_chunk_sec": round(avg_chunk_sec, 2),
                        "recent_avg_chunk_sec": round(recent_avg, 2),
                        "eta_sec": round(eta_sec, 1),
                        "wall_elapsed_sec": round(wall_elapsed, 1),
                        "last_chunk_sec": round(chunk_elapsed, 2),
                        "pct_complete": round(100 * chunks_done / total_chunks, 1),
                    },
                )

            # Periodic checkpoint
            if checkpoint_dir and checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
                save_checkpoint(checkpoint_dir, gpu_id, idx, params_store, results)

        return results

    except Exception:
        import traceback

        tb_str = traceback.format_exc()

        # Save emergency checkpoint on crash
        if checkpoint_dir and results:
            try:
                save_checkpoint(checkpoint_dir, gpu_id, chunk_indices[start_from], params_store, results)
                log_to_file(f"Worker {gpu_id}: Emergency checkpoint saved before crash")
            except Exception:
                pass

        log_to_file(f"Worker {gpu_id} CRASHED:\n{tb_str}")
        # Re-raise so _safe_worker (a closure-free top-level function)
        # can catch this and return a picklable _WorkerError value.
        raise


def run_backtest(
    X_np,
    y_np,
    dates,
    baselines,
    config,
    worker_fn,
    make_windows_fn,
    make_worker_args_fn,
    label="GPU Backtest",
    default_output="results.csv",
):
    """
    Generic GPU-parallel backtest orchestrator.

    Parameters
    ----------
    make_windows_fn : callable(X_tensor, y_tensor, config) ->
        (all_train_X, all_train_y, all_test_X, num_windows)
    make_worker_args_fn : callable(gpu_id, chunks, config, all_train_X,
        all_train_y, all_test_X, chunk_size, num_windows) -> tuple
        Build the positional args tuple for worker_fn.
    """
    gpu_count = config["gpu_count"]
    chunk_size = config["train"]["batch_size"]

    logger.info("Starting %s on %d GPUs", label, gpu_count)

    X_tensor = torch.tensor(X_np, dtype=torch.float32).pin_memory()
    y_tensor = torch.tensor(y_np, dtype=torch.float32).pin_memory()

    all_train_X, all_train_y, all_test_X, num_windows = make_windows_fn(X_tensor, y_tensor, config)

    # HPC chunking: slice windows to process only this chunk's subset
    ctx = chunk_context()
    if ctx.total_chunks > 1:
        window_range = ctx.split(num_windows)
        if len(window_range) == 0:
            logger.warning("Chunk %d has no windows to process. Exiting.", ctx.chunk_id)
            return None
        w_start = window_range.start
        w_end = window_range.stop
        all_train_X = all_train_X[w_start:w_end]
        all_train_y = all_train_y[w_start:w_end]
        all_test_X = all_test_X[w_start:w_end]
        num_windows = w_end - w_start
        logger.info(
            "HPC chunk %d/%d: processing windows %d-%d (%d windows)",
            ctx.chunk_id,
            ctx.total_chunks,
            w_start,
            w_end - 1,
            num_windows,
        )

    logger.info("Windows: %d", num_windows)
    logger.info("Train X Shape: %s", all_train_X.shape)
    logger.info("Test X Shape:  %s", all_test_X.shape)

    def worker_args(gpu_id, chunks):
        return make_worker_args_fn(
            gpu_id, chunks, config, all_train_X, all_train_y, all_test_X, chunk_size, num_windows
        )

    results_nested = distribute_and_run(worker_fn, worker_args, gpu_count, num_windows, chunk_size)

    # Free large tensors before aggregation to reduce peak memory
    all_train_X = all_train_y = all_test_X = X_tensor = y_tensor = None  # noqa: F841
    torch.cuda.empty_cache()
    gc.collect()

    logger.info("Workers finished. Aggregating results...")
    preds = aggregate_predictions(results_nested, num_windows)

    train_window = config["train_window"]
    if ctx.total_chunks > 1:
        # Offset test_indices by the chunk's starting window
        test_indices = np.arange(train_window + w_start, train_window + w_end)
    else:
        test_indices = np.arange(train_window, train_window + num_windows)
    output_file = config.get("output_path", default_output)
    logger.info("Saving %d results to %s", len(test_indices), output_file)

    return build_results_df(preds, test_indices, y_np, dates, baselines, output_file=output_file)
