"""Shared utilities for GPU backtest pipelines."""

import importlib
import math
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from src.backtest_helper import save_chunk_results
from src import config as cfg


def log_to_file(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    with open(cfg.GPU_WORKER_LOG, 'a') as f:
        f.write(f'{timestamp} - {message}\n')


def setup_device(gpu_id):
    """Create CUDA device and configure precision settings."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    return device


def load_model(model_module, factory_fn, model_config, device):
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


def allocate_params(base_model, max_batch_size, device):
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


def init_params(params_store, curr_bs, max_batch_size):
    """
    Re-initialize parameters with fan-in uniform for weights, zero for biases.
    Returns current_params (sliced if curr_bs < max_batch_size).
    """
    for name, p_tensor in params_store.items():
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


def run_kernel_and_detach(kernel, params, buffers, exp_avgs, exp_avg_sqs,
                          step_tensors, X, y):
    """Run compiled training kernel and detach the resulting parameters."""
    final_params = kernel(params, buffers, exp_avgs, exp_avg_sqs,
                          step_tensors, X, y)
    return {k: v.detach().requires_grad_(True) for k, v in final_params.items()}


def aggregate_predictions(results_nested, num_windows):
    """Flatten, sort, concatenate predictions from all GPU workers."""
    flat_results = [item for sublist in results_nested for item in sublist]
    flat_results.sort(key=lambda x: x['chunk_index'])
    preds = torch.cat([r['predictions'] for r in flat_results]).numpy()

    if len(preds) != num_windows:
        print(f'Warning: Prediction shape mismatch. '
              f'Expected {num_windows}, got {len(preds)}.')
        ratio = len(preds) / num_windows
        if ratio == int(ratio) and ratio > 1:
            preds = preds.reshape(num_windows, int(ratio))[:, 0]
        else:
            preds = preds[:num_windows]

    return preds


def build_results_df(preds, test_indices, y_np, dates, baselines,
                     output_file=None):
    """
    Apply Duan's smearing, optionally save, and return results DataFrame.
    """
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

    dates_subset = (dates.iloc[test_indices].values
                    if hasattr(dates, 'iloc') else dates[test_indices])
    y_subset = y_np[test_indices]
    base_subset = baselines[test_indices]
    smear = np.mean((y_subset - preds) ** 2)
    pred_raw = (preds ** 2 + smear) * base_subset
    true_raw = (y_subset ** 2) * base_subset

    return pd.DataFrame({
        'date': dates_subset,
        'true_adj': y_subset,
        'pred_adj': preds,
        'true_raw': true_raw,
        'pred_raw': pred_raw,
    })


def distribute_and_run(worker_fn, worker_args_fn, gpu_count, num_windows,
                       chunk_size):
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

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=gpu_count) as pool:
        args = [worker_args_fn(gpu_id, chunks_per_gpu[gpu_id])
                for gpu_id in range(gpu_count)]
        results_nested = pool.starmap(worker_fn, args)

    return results_nested


def run_worker(gpu_id, chunk_indices, shared_X, shared_y, shared_test_X,
               chunk_size, total_windows, setup_fn, chunk_fn):
    """
    Generic per-GPU worker loop.

    Parameters
    ----------
    setup_fn : callable(device) -> (params_store, ctx_dict)
        Initialise model, compile kernels, return pre-allocated params_store
        and an arbitrary context dict passed through to chunk_fn.
    chunk_fn : callable(ctx, X_chunk, y_chunk, X_test_chunk, curr_bs) -> Tensor
        Model-specific logic: normalise, train, predict.
        Must return a 1-D predictions tensor on the same device.
    """
    try:
        device = setup_device(gpu_id)
        params_store, ctx = setup_fn(device)

        results = []
        for i, idx in enumerate(chunk_indices):
            start = idx * chunk_size
            end = min(start + chunk_size, total_windows)

            X_chunk = shared_X[start:end].to(device, non_blocking=True)
            y_chunk = shared_y[start:end].to(device, non_blocking=True)
            X_test_chunk = shared_test_X[start:end].to(device, non_blocking=True)

            curr_bs = X_chunk.shape[0]
            if curr_bs == 0:
                continue

            current_params = init_params(params_store, curr_bs, chunk_size)
            preds = chunk_fn(ctx, current_params, X_chunk, y_chunk,
                             X_test_chunk, curr_bs, idx)

            results.append({
                'chunk_index': idx,
                'predictions': preds.view(-1).cpu(),
            })
            if i % 10 == 0:
                log_to_file(f'Worker {gpu_id}: Chunk {idx} done')

        return results

    except Exception as e:
        import traceback
        log_to_file(f'Worker {gpu_id} CRASHED:\n{traceback.format_exc()}')
        raise e


def run_backtest(X_np, y_np, dates, baselines, config, worker_fn,
                 make_windows_fn, make_worker_args_fn, label='GPU Backtest',
                 default_output='results.csv'):
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
    gpu_count = config['gpu_count']
    chunk_size = config['train']['batch_size']

    print(f'Starting {label} on {gpu_count} GPUs')

    X_tensor = torch.tensor(X_np, dtype=torch.float32).share_memory_().pin_memory()
    y_tensor = torch.tensor(y_np, dtype=torch.float32).share_memory_().pin_memory()

    all_train_X, all_train_y, all_test_X, num_windows = make_windows_fn(
        X_tensor, y_tensor, config)

    print(f'Windows: {num_windows}')
    print(f'Train X Shape: {all_train_X.shape}')
    print(f'Test X Shape:  {all_test_X.shape}')

    def worker_args(gpu_id, chunks):
        return make_worker_args_fn(gpu_id, chunks, config, all_train_X,
                                   all_train_y, all_test_X, chunk_size,
                                   num_windows)

    results_nested = distribute_and_run(
        worker_fn, worker_args, gpu_count, num_windows, chunk_size)

    print('Workers finished. Aggregating results...')
    preds = aggregate_predictions(results_nested, num_windows)

    train_window = config['train_window']
    test_indices = np.arange(train_window, train_window + num_windows)
    output_file = config.get('output_path', default_output)
    print(f'Saving {len(test_indices)} results to {output_file}...')

    return build_results_df(preds, test_indices, y_np, dates, baselines,
                            output_file=output_file)
