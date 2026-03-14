import torch
from torch.func import vmap, functional_call
from src.gpu_kernels import make_train_kernel
from src.gpu_utils import (
    log_to_file, load_model, allocate_params,
    init_adam_state, run_kernel_and_detach,
    normalize_chunks, run_worker, run_backtest,
)


def gpu_worker(gpu_id, chunk_indices, model_module, model_config, train_config,
               shared_X, shared_y, shared_test_X, chunk_size, total_windows):
    """Per-GPU worker for standard DL backtest."""

    def setup_fn(device):
        base_model_train, base_model_eval, buffers, param_keys = load_model(
            model_module, 'get_model', model_config, device)

        log_to_file(f'Worker {gpu_id}: Compiling...')
        raw_kernel = make_train_kernel(
            base_model_train, param_keys,
            train_config['num_epochs'], train_config['learning_rate'])
        train_kernel = torch.compile(raw_kernel, mode='default')

        def stateless_fwd(p, b, x):
            x_input = x.unsqueeze(-1)
            h_pred = functional_call(
                base_model_eval, (p, b), args=(x_input,), kwargs={})
            return h_pred.squeeze(-1)

        predict_kernel = vmap(stateless_fwd, in_dims=(0, None, 0), out_dims=0)

        params_store = allocate_params(base_model_train, chunk_size, device)
        ctx = dict(buffers=buffers, train_kernel=train_kernel,
                   predict_kernel=predict_kernel)
        return params_store, ctx

    def chunk_fn(ctx, current_params, X_chunk, y_chunk, X_test_chunk,
                 curr_bs, idx):
        # Instance normalization (each window normalized independently)
        X_chunk, X_test_chunk = normalize_chunks(
            X_chunk, X_test_chunk, dim=2, use_train_stats_for_test=False)

        # Train
        exp_avgs, exp_avg_sqs, step_tensors = init_adam_state(
            current_params, X_chunk.device)
        trained = run_kernel_and_detach(
            ctx['train_kernel'], current_params, ctx['buffers'],
            exp_avgs, exp_avg_sqs, step_tensors, X_chunk, y_chunk)

        # Predict: convert from log-space to sqrt-space
        # h_pred shape: (batch, channels, prediction_length)
        with torch.no_grad():
            h_pred = ctx['predict_kernel'](trained, ctx['buffers'], X_test_chunk)
            return torch.exp(h_pred / 2.0)

    return run_worker(gpu_id, chunk_indices, shared_X, shared_y, shared_test_X,
                      chunk_size, total_windows, setup_fn, chunk_fn)


def run_multigpu_backtest(X_np, y_np, dates, baselines, config,
                          model_module='src.dl_models'):
    """GPU-parallel backtest with model-agnostic architecture."""

    def make_windows_fn(X_tensor, y_tensor, config):
        train_window = config['train_window']
        context_len = config['model']['context_len']
        prediction_length = config['model'].get('prediction_length', 1)
        stride_step = 1  # always step by 1 for walk-forward

        total_samples = X_tensor.shape[0]
        # Reduce available windows to account for multi-step targets
        num_windows = total_samples - train_window - (prediction_length - 1)
        samples_per_window = train_window // context_len

        # 3D training windows
        window_shape_X = (num_windows, samples_per_window, context_len)
        strides_X = (
            X_tensor.stride(0) * stride_step,
            X_tensor.stride(0) * context_len,
            X_tensor.stride(0),
        )
        all_train_X = torch.as_strided(X_tensor, size=window_shape_X,
                                       stride=strides_X)

        # Aligned targets — multi-step: shape (num_windows, samples_per_window, prediction_length)
        y_offset = y_tensor[context_len:]
        window_shape_y = (num_windows, samples_per_window, prediction_length)
        strides_y = (
            y_offset.stride(0) * stride_step,
            y_offset.stride(0) * context_len,
            y_offset.stride(0),  # consecutive future steps
        )
        all_train_y = torch.as_strided(y_offset, size=window_shape_y,
                                       stride=strides_y)

        # OOS test tensor
        X_test_start = X_tensor[train_window - context_len:]
        window_shape_test = (num_windows, 1, context_len)
        strides_test = (
            X_test_start.stride(0) * stride_step,
            X_test_start.stride(0),
            X_test_start.stride(0),
        )
        all_test_X = torch.as_strided(X_test_start, size=window_shape_test,
                                      stride=strides_test)

        return all_train_X, all_train_y, all_test_X, num_windows

    def make_worker_args(gpu_id, chunks, config, all_train_X, all_train_y,
                         all_test_X, chunk_size, num_windows):
        return (gpu_id, chunks, model_module, config['model'], config['train'],
                all_train_X, all_train_y, all_test_X, chunk_size, num_windows)

    return run_backtest(X_np, y_np, dates, baselines, config,
                        gpu_worker, make_windows_fn, make_worker_args,
                        label=f'GPU Backtest (model: {model_module})')
