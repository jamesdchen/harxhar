import importlib
import math
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from torch.func import vmap, functional_call
from src.backtest_helper import save_chunk_results
from src.gpu_kernels import make_train_kernel
from src import config as cfg


def log_to_file(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    with open(cfg.GPU_WORKER_LOG, 'a') as f:
        f.write(f'{timestamp} - {message}\n')


def gpu_worker(gpu_id, chunk_indices, model_module, model_config, train_config,
               shared_X, shared_y, shared_test_X, chunk_size, total_windows):
    """
    Per-GPU worker. Model-agnostic: loads model via importlib from model_module.

    Any module that exports get_model(cfg) -> PreTrainedModel can be used.
    """
    try:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision('high')

        mod = importlib.import_module(model_module)
        model_factory = lambda: mod.get_model(model_config)

        base_model_train = model_factory().to(device)
        base_model_eval = model_factory().to(device)
        base_model_eval.eval()

        buffers = {n: b for n, b in base_model_train.named_buffers()}
        param_keys = [n for n, p in base_model_train.named_parameters()]

        steady_epochs = train_config['num_epochs']
        lr = train_config['learning_rate']

        log_to_file(f'Worker {gpu_id}: Compiling...')
        raw_steady = make_train_kernel(base_model_train, param_keys, steady_epochs, lr)
        steady_kernel = torch.compile(raw_steady, mode='default')

        # Prediction kernel: model outputs h = log(sigma^2)
        def stateless_fwd(p, b, x):
            x_input = x.unsqueeze(-1)
            h_pred = functional_call(base_model_eval, (p, b), args=(x_input,), kwargs={})
            return h_pred.squeeze(-1)

        predict_kernel = vmap(stateless_fwd, in_dims=(0, None, 0), out_dims=0)

        results = []
        param_shapes = {n: p.shape for n, p in base_model_train.named_parameters()}
        max_batch_size = chunk_size

        # Pre-allocate parameter tensors
        params_store = {}
        for name, shape in param_shapes.items():
            if len(shape) > 1:
                p = torch.empty((max_batch_size, *shape), device=device)
            else:
                p = torch.zeros((max_batch_size, *shape), device=device)
            params_store[name] = p.requires_grad_(True)

        for i, idx in enumerate(chunk_indices):
            start = idx * chunk_size
            end = min(start + chunk_size, total_windows)

            X_chunk = shared_X[start:end].to(device, non_blocking=True)
            y_chunk = shared_y[start:end].to(device, non_blocking=True)
            X_test_chunk = shared_test_X[start:end].to(device, non_blocking=True)

            curr_bs = X_chunk.shape[0]
            if curr_bs == 0:
                continue

            # --- INIT PARAMETERS ---
            for name, p_tensor in params_store.items():
                fan_in = p_tensor.shape[-1] if len(p_tensor.shape) > 2 else 1
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.01

                if len(p_tensor.shape) > 1:
                    p_tensor.data.uniform_(-bound, bound)
                else:
                    p_tensor.data.zero_()

            current_params = params_store
            if curr_bs < max_batch_size:
                current_params = {k: v[:curr_bs] for k, v in params_store.items()}

            # --- INSTANCE NORMALIZATION (same concept as RollingRobustScaler) ---
            mean = X_chunk.mean(dim=2, keepdim=True)
            std = X_chunk.std(dim=2, keepdim=True) + cfg.NORM_EPS
            X_chunk = (X_chunk - mean) / std

            t_mean = X_test_chunk.mean(dim=2, keepdim=True)
            t_std = X_test_chunk.std(dim=2, keepdim=True) + cfg.NORM_EPS
            X_test_chunk = (X_test_chunk - t_mean) / t_std

            # --- TRAIN ---
            exp_avgs = {}
            exp_avg_sqs = {}
            step_tensors = {}

            for k, p in current_params.items():
                exp_avgs[k] = torch.zeros_like(p)
                exp_avg_sqs[k] = torch.zeros_like(p)
                step_tensors[k] = torch.tensor(0.0, device=device)

            final_params = steady_kernel(
                current_params, buffers, exp_avgs, exp_avg_sqs, step_tensors,
                X_chunk, y_chunk
            )

            detached_update = {
                k: v.detach().requires_grad_(True)
                for k, v in final_params.items()
            }

            # --- PREDICT ---
            with torch.no_grad():
                h_pred = predict_kernel(detached_update, buffers, X_test_chunk)
                # Convert from log-space to sqrt-space for Duan's smearing
                pred_sqrt = torch.exp(h_pred / 2.0)

            results.append({'chunk_index': idx, 'predictions': pred_sqrt.view(-1).cpu()})
            if i % 10 == 0:
                log_to_file(f'Worker {gpu_id}: Chunk {idx} done')

            del X_chunk, y_chunk, X_test_chunk, mean, std
            del final_params, detached_update, exp_avgs, exp_avg_sqs
            del h_pred, pred_sqrt

        return results

    except Exception as e:
        import traceback
        log_to_file(f'Worker {gpu_id} CRASHED:\n{traceback.format_exc()}')
        raise e


def run_multigpu_backtest(X_np, y_np, dates, baselines, config, model_module='src.dl_models'):
    """
    GPU-parallel backtest with model-agnostic architecture.

    Parameters
    ----------
    X_np : np.ndarray
        Feature matrix (n_samples, n_features).
    y_np : np.ndarray
        Target array (n_samples,) in sqrt-space (adj_RV).
    dates : pd.Series or np.ndarray
        Timestamps for each sample.
    baselines : np.ndarray
        Diurnal baseline for raw-space reconstruction.
    config : dict
        Must contain 'train_window', 'gpu_count', 'output_path',
        'model' (dict), and 'train' (dict with 'batch_size').
    model_module : str
        Dotted module path exporting get_model(cfg). Default: 'src.dl_models'.

    Returns
    -------
    results_df : pd.DataFrame
        Columns: date, true_adj, pred_adj, true_raw, pred_raw.
    """
    gpu_count = config['gpu_count']
    train_window = config['train_window']
    context_len = config['model']['context_len']
    stride_step = config['model']['prediction_length']
    chunk_size = config['train']['batch_size']

    print(f'Starting GPU Backtest on {gpu_count} GPUs (model: {model_module})')

    total_samples = X_np.shape[0]
    num_windows = total_samples - train_window
    samples_per_window = train_window // context_len

    # Convert to shared-memory tensors for GPU
    X_tensor = torch.tensor(X_np, dtype=torch.float32).share_memory_().pin_memory()
    y_tensor = torch.tensor(y_np, dtype=torch.float32).share_memory_().pin_memory()

    print(f'Windows: {num_windows} | Samples/Window: {samples_per_window} | Context: {context_len}')

    # --- Create Training Windows (3D) ---
    window_shape_X = (num_windows, samples_per_window, context_len)
    strides_X = (
        X_tensor.stride(0) * stride_step,
        X_tensor.stride(0) * context_len,
        X_tensor.stride(0)
    )
    all_train_X = torch.as_strided(X_tensor, size=window_shape_X, stride=strides_X)

    # --- Align Targets ---
    y_offset = y_tensor[context_len:]
    window_shape_y = (num_windows, samples_per_window, 1)
    strides_y = (
        y_offset.stride(0) * stride_step,
        y_offset.stride(0) * context_len,
        y_offset.stride(1)
    )
    all_train_y = torch.as_strided(y_offset, size=window_shape_y, stride=strides_y)

    # --- Create Test Tensor (OOS) ---
    X_test_start = X_tensor[train_window - context_len:]
    window_shape_test = (num_windows, 1, context_len)
    strides_test = (
        X_test_start.stride(0) * stride_step,
        X_test_start.stride(0),
        X_test_start.stride(0)
    )
    all_test_X = torch.as_strided(X_test_start, size=window_shape_test, stride=strides_test)

    print(f'Train X Shape: {all_train_X.shape}')
    print(f'Test X Shape:  {all_test_X.shape}')

    # --- Distribute chunks across GPUs ---
    num_chunks = math.ceil(num_windows / chunk_size)
    chunk_indices = list(range(num_chunks))
    chunks_per_gpu = [chunk_indices[i::gpu_count] for i in range(gpu_count)]

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=gpu_count) as pool:
        args = []
        for gpu_id in range(gpu_count):
            args.append((
                gpu_id, chunks_per_gpu[gpu_id], model_module,
                config['model'], config['train'],
                all_train_X, all_train_y, all_test_X, chunk_size, num_windows
            ))
        results_nested = pool.starmap(gpu_worker, args)

    # --- Aggregation ---
    print('Workers finished. Aggregating results...')
    flat_results = [item for sublist in results_nested for item in sublist]
    flat_results.sort(key=lambda x: x['chunk_index'])

    preds_sqrt = torch.cat([r['predictions'] for r in flat_results]).numpy()

    # Shape correction
    if len(preds_sqrt) != num_windows:
        print(f'Warning: Prediction shape mismatch. Expected {num_windows}, got {len(preds_sqrt)}.')
        ratio = len(preds_sqrt) / num_windows
        if ratio == int(ratio) and ratio > 1:
            preds_sqrt = preds_sqrt.reshape(num_windows, int(ratio))[:, 0]
        else:
            preds_sqrt = preds_sqrt[:num_windows]

    # --- Save results using save_chunk_results (Duan's smearing + codebase convention) ---
    test_indices = np.arange(train_window, train_window + num_windows)
    output_file = config.get('output_path', 'results.csv')

    print(f'Saving {len(test_indices)} results to {output_file}...')
    save_chunk_results(
        output_file=output_file,
        forecasts=preds_sqrt,
        indices=test_indices,
        train_window=train_window,
        y_true=y_np,
        dates=dates,
        baselines=baselines
    )

    # Also return the DataFrame for in-notebook evaluation
    dates_subset = dates.iloc[test_indices].values if hasattr(dates, 'iloc') else dates[test_indices]
    y_subset = y_np[test_indices]
    base_subset = baselines[test_indices]
    smear = np.mean((y_subset - preds_sqrt) ** 2)
    pred_raw = (preds_sqrt ** 2 + smear) * base_subset
    true_raw = (y_subset ** 2) * base_subset

    results_df = pd.DataFrame({
        'date': dates_subset,
        'true_adj': y_subset,
        'pred_adj': preds_sqrt,
        'true_raw': true_raw,
        'pred_raw': pred_raw,
    })

    return results_df
