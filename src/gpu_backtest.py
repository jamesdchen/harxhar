import math
import torch
import numpy as np
from torch.func import vmap, functional_call
from src.gpu_kernels import make_train_kernel
from src import config as cfg
from src.gpu_utils import (
    log_to_file, setup_device, load_model, allocate_params, init_params,
    init_adam_state, run_kernel_and_detach, aggregate_predictions,
    build_results_df, distribute_and_run,
)


def gpu_worker(gpu_id, chunk_indices, model_module, model_config, train_config,
               shared_X, shared_y, shared_test_X, chunk_size, total_windows):
    """
    Per-GPU worker. Model-agnostic: loads model via importlib from model_module.

    Any module that exports get_model(cfg) -> PreTrainedModel can be used.
    """
    try:
        device = setup_device(gpu_id)

        base_model_train, base_model_eval, buffers, param_keys = load_model(
            model_module, 'get_model', model_config, device)

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
        params_store = allocate_params(base_model_train, chunk_size, device)

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

            # --- INSTANCE NORMALIZATION (same concept as RollingRobustScaler) ---
            mean = X_chunk.mean(dim=2, keepdim=True)
            std = X_chunk.std(dim=2, keepdim=True) + cfg.NORM_EPS
            X_chunk = (X_chunk - mean) / std

            t_mean = X_test_chunk.mean(dim=2, keepdim=True)
            t_std = X_test_chunk.std(dim=2, keepdim=True) + cfg.NORM_EPS
            X_test_chunk = (X_test_chunk - t_mean) / t_std

            # --- TRAIN ---
            exp_avgs, exp_avg_sqs, step_tensors = init_adam_state(
                current_params, device)
            detached_update = run_kernel_and_detach(
                steady_kernel, current_params, buffers, exp_avgs, exp_avg_sqs,
                step_tensors, X_chunk, y_chunk)

            # --- PREDICT ---
            with torch.no_grad():
                h_pred = predict_kernel(detached_update, buffers, X_test_chunk)
                # Convert from log-space to sqrt-space for Duan's smearing
                pred_sqrt = torch.exp(h_pred / 2.0)

            results.append({'chunk_index': idx, 'predictions': pred_sqrt.view(-1).cpu()})
            if i % 10 == 0:
                log_to_file(f'Worker {gpu_id}: Chunk {idx} done')

            del X_chunk, y_chunk, X_test_chunk, mean, std
            del detached_update, exp_avgs, exp_avg_sqs
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

    # --- Distribute and run ---
    def worker_args(gpu_id, chunks):
        return (gpu_id, chunks, model_module, config['model'], config['train'],
                all_train_X, all_train_y, all_test_X, chunk_size, num_windows)

    results_nested = distribute_and_run(
        gpu_worker, worker_args, gpu_count, num_windows, chunk_size)

    # --- Aggregation ---
    print('Workers finished. Aggregating results...')
    preds_sqrt = aggregate_predictions(results_nested, num_windows)

    test_indices = np.arange(train_window, train_window + num_windows)
    output_file = config.get('output_path', 'results.csv')
    print(f'Saving {len(test_indices)} results to {output_file}...')

    return build_results_df(preds_sqrt, test_indices, y_np, dates, baselines,
                            output_file=output_file)
