"""GPU-parallel backtest pipeline for AE+Ridge models."""

import os
import torch
import numpy as np
from torch.func import vmap, functional_call
from src.gpu_kernels import make_ae_train_kernel
from src import config as cfg
from src.gpu_utils import (
    log_to_file, setup_device, load_model, allocate_params, init_params,
    init_adam_state, run_kernel_and_detach, aggregate_predictions,
    build_results_df, distribute_and_run,
)


def ae_gpu_worker(gpu_id, chunk_indices, model_config, train_config,
                  shared_X, shared_y, shared_test_X, chunk_size,
                  total_windows, weights_dir):
    """
    Per-GPU worker for AE+Ridge backtest.

    For each batch of windows:
      1. Train AE with hybrid MSE loss (vmap across windows)
      2. Encode training features via trained encoder
      3. Solve Ridge in closed form on GPU
      4. Encode test point and predict via Ridge weights

    Parameters
    ----------
    shared_X :      (num_windows, train_win, n_features) strided view
    shared_y :      (num_windows, train_win) targets
    shared_test_X : (num_windows, 1, n_features) OOS test points
    weights_dir :   str or None. If set, save AE state dicts per chunk.
    """
    try:
        device = setup_device(gpu_id)

        base_model_train, base_model_eval, buffers, param_keys = load_model(
            'src.dl_models', 'get_ae_model', model_config, device)

        num_epochs = train_config['num_epochs']
        lr = train_config['learning_rate']
        alpha_recon = model_config['alpha_recon']
        alpha_ridge = model_config['alpha_ridge']
        n_components = model_config['n_components']

        log_to_file(f'AE Worker {gpu_id}: Compiling...')
        raw_kernel = make_ae_train_kernel(
            base_model_train, param_keys, num_epochs, lr, alpha_recon)
        train_kernel = torch.compile(raw_kernel, mode='default')

        # Encode kernel: run full forward, keep only z
        def stateless_encode(p, b, x):
            _recon, z, _pred = functional_call(
                base_model_eval, (p, b), args=(x,), kwargs={})
            return z

        batch_encode = vmap(stateless_encode, in_dims=(0, None, 0), out_dims=0)

        results = []
        params_store = allocate_params(base_model_train, chunk_size, device)
        ridge_eye = alpha_ridge * torch.eye(n_components, device=device)

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

            # --- NORMALIZE (use training stats for test point) ---
            mean = X_chunk.mean(dim=1, keepdim=True)   # (batch, 1, n_features)
            std = X_chunk.std(dim=1, keepdim=True) + cfg.NORM_EPS
            X_chunk = (X_chunk - mean) / std
            X_test_chunk = (X_test_chunk - mean) / std

            # --- TRAIN AE ---
            exp_avgs, exp_avg_sqs, step_tensors = init_adam_state(
                current_params, device)
            trained_params = run_kernel_and_detach(
                train_kernel, current_params, buffers, exp_avgs, exp_avg_sqs,
                step_tensors, X_chunk, y_chunk)

            # --- ENCODE + RIDGE SOLVE + PREDICT ---
            with torch.no_grad():
                # Encode training features: (batch, train_win, n_components)
                Z_train = batch_encode(trained_params, buffers, X_chunk)

                # Batched Ridge solve: w = (Z^T Z + alpha*I)^{-1} Z^T y
                ZtZ = torch.bmm(Z_train.transpose(1, 2), Z_train)  # (bs, K, K)
                reg = ridge_eye[:n_components, :n_components].unsqueeze(0)
                Zty = torch.bmm(
                    Z_train.transpose(1, 2),
                    y_chunk.unsqueeze(-1))  # (bs, K, 1)
                w = torch.linalg.solve(ZtZ + reg, Zty)  # (bs, K, 1)

                # Encode test point: (batch, 1, n_components)
                z_test = batch_encode(trained_params, buffers, X_test_chunk)

                # Predict: (batch,)
                pred = torch.bmm(z_test, w).squeeze()

            results.append({
                'chunk_index': idx,
                'predictions': pred.view(-1).cpu(),
            })

            # Save AE weights if requested
            if weights_dir is not None:
                _save_chunk_weights(trained_params, param_keys, idx,
                                    weights_dir)

            if i % 10 == 0:
                log_to_file(f'AE Worker {gpu_id}: Chunk {idx} done')

            del X_chunk, y_chunk, X_test_chunk, mean, std
            del trained_params, exp_avgs, exp_avg_sqs
            del Z_train, ZtZ, Zty, w, z_test, pred

        return results

    except Exception as e:
        import traceback
        log_to_file(f'AE Worker {gpu_id} CRASHED:\n{traceback.format_exc()}')
        raise e


def _save_chunk_weights(trained_params, param_keys, chunk_idx, weights_dir):
    """Save the mean AE weights across the batch for a given chunk."""
    os.makedirs(weights_dir, exist_ok=True)
    state_dict = {}
    for k in param_keys:
        # Average across the batch dimension (dim 0)
        state_dict[k] = trained_params[k].mean(dim=0).cpu()
    torch.save(state_dict, os.path.join(weights_dir,
                                        f'ae_weights_{chunk_idx:04d}.pt'))


def run_ae_multigpu_backtest(X_np, y_np, dates, baselines, config):
    """
    GPU-parallel AE+Ridge backtest.

    Parameters
    ----------
    X_np : np.ndarray, shape (n_samples, n_features)
    y_np : np.ndarray, shape (n_samples,)
    dates : pd.Series or np.ndarray
    baselines : np.ndarray
    config : dict matching AE_RIDGE_GPU_CONFIG structure

    Returns
    -------
    results_df : pd.DataFrame
        Columns: date, true_adj, pred_adj, true_raw, pred_raw.
    """
    gpu_count = config['gpu_count']
    train_window = config['train_window']
    chunk_size = config['train']['batch_size']
    weights_dir = config.get('weights_dir', None)

    # Set n_features from data
    n_features = X_np.shape[1]
    model_config = {**config['model'], 'n_features': n_features}

    print(f'Starting AE+Ridge GPU Backtest on {gpu_count} GPUs')

    total_samples = X_np.shape[0]
    num_windows = total_samples - train_window

    # Convert to shared-memory tensors
    X_tensor = torch.tensor(X_np, dtype=torch.float32).share_memory_().pin_memory()
    y_tensor = torch.tensor(y_np, dtype=torch.float32).share_memory_().pin_memory()

    print(f'Windows: {num_windows} | Train Window: {train_window} | '
          f'Features: {n_features}')

    # --- Create 2D sliding windows via as_strided ---
    # Training: (num_windows, train_window, n_features)
    all_train_X = torch.as_strided(
        X_tensor,
        size=(num_windows, train_window, n_features),
        stride=(X_tensor.stride(0), X_tensor.stride(0), X_tensor.stride(1)),
    )

    # Targets: (num_windows, train_window)
    all_train_y = torch.as_strided(
        y_tensor,
        size=(num_windows, train_window),
        stride=(y_tensor.stride(0), y_tensor.stride(0)),
    )

    # Test points: (num_windows, 1, n_features)
    all_test_X = X_tensor[train_window:].unsqueeze(1)

    print(f'Train X Shape: {all_train_X.shape}')
    print(f'Test X Shape:  {all_test_X.shape}')

    # --- Distribute and run ---
    def worker_args(gpu_id, chunks):
        return (gpu_id, chunks, model_config, config['train'],
                all_train_X, all_train_y, all_test_X, chunk_size,
                num_windows, weights_dir)

    results_nested = distribute_and_run(
        ae_gpu_worker, worker_args, gpu_count, num_windows, chunk_size)

    # --- Aggregation ---
    print('Workers finished. Aggregating results...')
    preds = aggregate_predictions(results_nested, num_windows)

    test_indices = np.arange(train_window, train_window + num_windows)
    output_file = config.get('output_path', 'ae_ridge_results.csv')
    print(f'Saving {len(test_indices)} results to {output_file}...')

    return build_results_df(preds, test_indices, y_np, dates, baselines,
                            output_file=output_file)
