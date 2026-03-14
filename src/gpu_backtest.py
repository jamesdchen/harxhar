import importlib
import math
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from datetime import datetime
from torch.func import vmap, functional_call, grad
from torch.amp import autocast
from torch.optim.adamw import adamw
from src.data_helper import save_chunk_results


def log_to_file(message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    with open('worker_log.txt', 'a') as f:
        f.write(f'{timestamp} - {message}\n')


# --- Log-Space Parameterized QLIKE Loss ---
def functional_qlike_loss(h_pred, target_sqrt):
    """
    QLIKE parameterized in log-space for numerical stability.

    h_pred: model output = log(sigma^2_pred), unconstrained real number
    target_sqrt: adj_RV (sqrt-space target from codebase pipeline)

    L = sigma^2_true * exp(-h_pred) + h_pred
    dL/dh = -sigma^2_true * exp(-h) + 1   (always bounded, no log(0) or div-by-zero)
    """
    target_sq = target_sqrt.squeeze().float() ** 2
    h = h_pred.squeeze().float()
    h = torch.clamp(h, min=-30.0, max=30.0)
    return target_sq * torch.exp(-h) + h


# --- COMPILED TRAINING KERNEL ---
def make_train_kernel(base_model, param_keys, num_epochs, base_lr):

    def compute_loss_stateless(params, buffers, x, y):
        x_in = x.unsqueeze(-1)
        h_pred = functional_call(base_model, (params, buffers), args=(x_in,), kwargs={})
        return functional_qlike_loss(h_pred, y)

    batch_loss_fn = vmap(compute_loss_stateless, in_dims=(0, None, 0, 0), randomness='different')

    def train_loop(params, buffers, exp_avgs, exp_avg_sqs, step_tensors, X, y):

        for i in range(1, num_epochs + 1):
            def mean_loss(p):
                with autocast('cuda'):
                    losses = batch_loss_fn(p, buffers, X, y)
                    return losses.mean()

            grads_dict = grad(mean_loss)(params)

            grad_list = []
            found_inf = torch.tensor(False, device=X.device)

            for k in param_keys:
                g = grads_dict[k]
                g = torch.clamp(g, min=-5.0, max=5.0)
                grad_list.append(g)
                if not torch.isfinite(g).all():
                    found_inf = torch.tensor(True, device=X.device)

            if not found_inf:
                mutable_params = [params[k].clone() for k in param_keys]
                mutable_exp_avgs = [exp_avgs[k] for k in param_keys]
                mutable_exp_avg_sqs = [exp_avg_sqs[k] for k in param_keys]
                mutable_steps = [step_tensors[k] for k in param_keys]

                adamw(
                    params=mutable_params,
                    grads=grad_list,
                    exp_avgs=mutable_exp_avgs,
                    exp_avg_sqs=mutable_exp_avg_sqs,
                    max_exp_avg_sqs=[],
                    state_steps=mutable_steps,
                    amsgrad=False,
                    beta1=0.9,
                    beta2=0.999,
                    lr=base_lr,
                    weight_decay=0.01,
                    eps=1e-8,
                    maximize=False,
                    foreach=False,
                    capturable=True
                )
                params = {k: mutable_params[idx] for idx, k in enumerate(param_keys)}

        return params

    return train_loop


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
            std = X_chunk.std(dim=2, keepdim=True) + 1e-8
            X_chunk = (X_chunk - mean) / std

            t_mean = X_test_chunk.mean(dim=2, keepdim=True)
            t_std = X_test_chunk.std(dim=2, keepdim=True) + 1e-8
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
