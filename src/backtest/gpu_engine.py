"""Unified GPU-parallel backtest engines for PatchTSMixer and AE+Ridge models."""

import os

import torch
from torch.func import functional_call, vmap

from src.backtest.gpu_kernels import make_ae_train_kernel, make_train_kernel
from src.backtest.gpu_utils import (
    allocate_params,
    init_adam_state,
    load_model,
    log_to_file,
    normalize_chunks,
    run_backtest,
    run_kernel_and_detach,
    run_worker,
    save_loss_log,
)

# ---------------------------------------------------------------------------
# PatchTSMixer strategy
# ---------------------------------------------------------------------------


def _patchts_worker(
    gpu_id,
    chunk_indices,
    model_module,
    model_config,
    train_config,
    shared_X,
    shared_y,
    shared_test_X,
    chunk_size,
    total_windows,
    checkpoint_dir=None,
    checkpoint_every=0,
    loss_log_path=None,
):
    """Per-GPU worker for standard DL backtest."""

    def setup_fn(device):
        base_model_train, base_model_eval, buffers, param_keys = load_model(
            model_module, "get_model", model_config, device
        )

        log_to_file(f"Worker {gpu_id}: Compiling...")
        raw_kernel = make_train_kernel(
            base_model_train, param_keys, train_config["num_epochs"], train_config["learning_rate"]
        )
        train_kernel = torch.compile(raw_kernel, mode="default")

        def stateless_fwd(p, b, x):
            x_input = x.unsqueeze(-1)
            h_pred = functional_call(base_model_eval, (p, b), args=(x_input,), kwargs={})
            return h_pred.squeeze(-1)

        predict_kernel = vmap(stateless_fwd, in_dims=(0, None, 0), out_dims=0)

        params_store = allocate_params(base_model_train, chunk_size, device)
        ctx = dict(
            buffers=buffers,
            train_kernel=train_kernel,
            predict_kernel=predict_kernel,
            loss_log_path=loss_log_path,
        )
        return params_store, ctx

    def chunk_fn(ctx, current_params, X_chunk, y_chunk, X_test_chunk, curr_bs, idx):
        # Instance normalization (each window normalized independently)
        X_chunk, X_test_chunk = normalize_chunks(X_chunk, X_test_chunk, dim=2, use_train_stats_for_test=False)

        # Train
        exp_avgs, exp_avg_sqs, step_tensors = init_adam_state(current_params, X_chunk.device)
        trained, epoch_losses = run_kernel_and_detach(
            ctx["train_kernel"], current_params, ctx["buffers"], exp_avgs, exp_avg_sqs, step_tensors, X_chunk, y_chunk
        )

        # Save per-epoch losses for this chunk
        if ctx["loss_log_path"]:
            save_loss_log(ctx["loss_log_path"], idx, epoch_losses.cpu().numpy())

        # Predict: convert from log-space to sqrt-space
        # h_pred shape: (batch, channels, prediction_length)
        with torch.no_grad():
            h_pred = ctx["predict_kernel"](trained, ctx["buffers"], X_test_chunk)
            return torch.exp(h_pred / 2.0)

    return run_worker(
        gpu_id,
        chunk_indices,
        shared_X,
        shared_y,
        shared_test_X,
        chunk_size,
        total_windows,
        setup_fn,
        chunk_fn,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
    )


def _patchts_make_windows(X_tensor, y_tensor, config):
    """Create 3D strided windows for PatchTSMixer (samples_per_window x context_len)."""
    train_window = config["train_window"]
    context_len = config["model"]["context_len"]
    prediction_length = config["model"].get("prediction_length", 1)
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
    all_train_X = torch.as_strided(X_tensor, size=window_shape_X, stride=strides_X)

    # Aligned targets — multi-step: shape (num_windows, samples_per_window, prediction_length)
    y_offset = y_tensor[context_len:]
    window_shape_y = (num_windows, samples_per_window, prediction_length)
    strides_y = (
        y_offset.stride(0) * stride_step,
        y_offset.stride(0) * context_len,
        y_offset.stride(0),  # consecutive future steps
    )
    all_train_y = torch.as_strided(y_offset, size=window_shape_y, stride=strides_y)

    # OOS test tensor
    X_test_start = X_tensor[train_window - context_len :]
    window_shape_test = (num_windows, 1, context_len)
    strides_test = (
        X_test_start.stride(0) * stride_step,
        X_test_start.stride(0),
        X_test_start.stride(0),
    )
    all_test_X = torch.as_strided(X_test_start, size=window_shape_test, stride=strides_test)

    return all_train_X, all_train_y, all_test_X, num_windows


# ---------------------------------------------------------------------------
# AE+Ridge strategy
# ---------------------------------------------------------------------------


def _ae_ridge_worker(
    gpu_id,
    chunk_indices,
    model_config,
    train_config,
    shared_X,
    shared_y,
    shared_test_X,
    chunk_size,
    total_windows,
    weights_dir,
    checkpoint_dir=None,
    checkpoint_every=0,
    loss_log_path=None,
):
    """Per-GPU worker for AE+Ridge backtest."""

    def setup_fn(device):
        base_model_train, base_model_eval, buffers, param_keys = load_model(
            "src.models.deep_learning", "get_ae_model", model_config, device
        )

        alpha_recon = model_config["alpha_recon"]
        n_components = model_config["n_components"]
        alpha_ridge = model_config["alpha_ridge"]

        log_to_file(f"AE Worker {gpu_id}: Compiling...")
        raw_kernel = make_ae_train_kernel(
            base_model_train, param_keys, train_config["num_epochs"], train_config["learning_rate"], alpha_recon
        )
        train_kernel = torch.compile(raw_kernel, mode="default")

        def stateless_encode(p, b, x):
            _recon, z, _pred = functional_call(base_model_eval, (p, b), args=(x,), kwargs={})
            return z

        batch_encode = vmap(stateless_encode, in_dims=(0, None, 0), out_dims=0)

        ridge_eye = alpha_ridge * torch.eye(n_components, device=device)

        params_store = allocate_params(base_model_train, chunk_size, device)
        ctx = dict(
            buffers=buffers,
            train_kernel=train_kernel,
            batch_encode=batch_encode,
            ridge_eye=ridge_eye,
            n_components=n_components,
            param_keys=param_keys,
            weights_dir=weights_dir,
            loss_log_path=loss_log_path,
        )
        return params_store, ctx

    def chunk_fn(ctx, current_params, X_chunk, y_chunk, X_test_chunk, curr_bs, idx):
        # Normalize (use training stats for test point)
        X_chunk, X_test_chunk = normalize_chunks(X_chunk, X_test_chunk, dim=1, use_train_stats_for_test=True)

        # Train AE
        exp_avgs, exp_avg_sqs, step_tensors = init_adam_state(current_params, X_chunk.device)
        trained_params, epoch_losses = run_kernel_and_detach(
            ctx["train_kernel"], current_params, ctx["buffers"], exp_avgs, exp_avg_sqs, step_tensors, X_chunk, y_chunk
        )

        # Save per-epoch losses for this chunk
        if ctx["loss_log_path"]:
            save_loss_log(ctx["loss_log_path"], idx, epoch_losses.cpu().numpy())

        # Encode + Ridge solve + Predict
        with torch.no_grad():
            Z_train = ctx["batch_encode"](trained_params, ctx["buffers"], X_chunk)
            ZtZ = torch.bmm(Z_train.transpose(1, 2), Z_train)
            reg = ctx["ridge_eye"].unsqueeze(0)

            # Multi-output Ridge: y_chunk may be (batch, T) or (batch, T, H)
            if y_chunk.dim() == 3:
                # Multi-horizon: solve for H targets simultaneously
                # y_chunk: (batch, T, H), Z_train: (batch, T, n_components)
                Zty = torch.bmm(Z_train.transpose(1, 2), y_chunk)  # (batch, n, H)
            else:
                Zty = torch.bmm(Z_train.transpose(1, 2), y_chunk.unsqueeze(-1))  # (batch, n, 1)
            w = torch.linalg.solve(ZtZ + reg, Zty)  # (batch, n, H) or (batch, n, 1)

            z_test = ctx["batch_encode"](trained_params, ctx["buffers"], X_test_chunk)
            pred = torch.bmm(z_test, w)  # (batch, 1, H) or (batch, 1, 1)
            if pred.shape[-1] > 1:
                pred = pred.squeeze(1)  # (batch, H)
            else:
                pred = pred.squeeze(-1).squeeze(-1)  # (batch,)

        if ctx["weights_dir"] is not None:
            _save_chunk_weights(trained_params, ctx["param_keys"], idx, ctx["weights_dir"])

        return pred

    return run_worker(
        gpu_id,
        chunk_indices,
        shared_X,
        shared_y,
        shared_test_X,
        chunk_size,
        total_windows,
        setup_fn,
        chunk_fn,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
    )


def _save_chunk_weights(trained_params, param_keys, chunk_idx, weights_dir):
    """Save the mean AE weights across the batch for a given chunk."""
    os.makedirs(weights_dir, exist_ok=True)
    state_dict = {}
    for k in param_keys:
        state_dict[k] = trained_params[k].mean(dim=0).cpu()
    torch.save(state_dict, os.path.join(weights_dir, f"ae_weights_{chunk_idx:04d}.pt"))


def _ae_ridge_make_windows(X_tensor, y_tensor, config):
    """Create 2D strided windows for AE+Ridge (train_window x n_features)."""
    train_window = config["train_window"]
    prediction_length = config["model"].get("prediction_length", 1)
    n_feat = X_tensor.shape[1]
    total_samples = X_tensor.shape[0]
    num_windows = total_samples - train_window - (prediction_length - 1)

    all_train_X = torch.as_strided(
        X_tensor,
        size=(num_windows, train_window, n_feat),
        stride=(X_tensor.stride(0), X_tensor.stride(0), X_tensor.stride(1)),
    )

    if prediction_length > 1:
        # Multi-step targets: shape (num_windows, train_window, prediction_length)
        all_train_y = torch.as_strided(
            y_tensor,
            size=(num_windows, train_window, prediction_length),
            stride=(y_tensor.stride(0), y_tensor.stride(0), y_tensor.stride(0)),
        )
    else:
        all_train_y = torch.as_strided(
            y_tensor,
            size=(num_windows, train_window),
            stride=(y_tensor.stride(0), y_tensor.stride(0)),
        )

    all_test_X = X_tensor[train_window : train_window + num_windows].unsqueeze(1)

    return all_train_X, all_train_y, all_test_X, num_windows


# ---------------------------------------------------------------------------
# Public API (backward-compatible entry points)
# ---------------------------------------------------------------------------


def run_multigpu_backtest(X_np, y_np, dates, baselines, config, model_module="src.models.deep_learning"):
    """GPU-parallel PatchTSMixer backtest."""
    checkpoint_dir = config.get("checkpoint_dir", None)
    checkpoint_every = config.get("checkpoint_every", 0)
    loss_log_path = config.get("loss_log_path", None)

    def make_worker_args(gpu_id, chunks, config, all_train_X, all_train_y, all_test_X, chunk_size, num_windows):
        per_gpu_loss_path = None
        if loss_log_path:
            base, ext = os.path.splitext(loss_log_path)
            per_gpu_loss_path = f"{base}_gpu{gpu_id}{ext}"
        return (
            gpu_id,
            chunks,
            model_module,
            config["model"],
            config["train"],
            all_train_X,
            all_train_y,
            all_test_X,
            chunk_size,
            num_windows,
            checkpoint_dir,
            checkpoint_every,
            per_gpu_loss_path,
        )

    return run_backtest(
        X_np,
        y_np,
        dates,
        baselines,
        config,
        _patchts_worker,
        _patchts_make_windows,
        make_worker_args,
        label=f"GPU Backtest (model: {model_module})",
    )


def run_ae_multigpu_backtest(X_np, y_np, dates, baselines, config):
    """GPU-parallel AE+Ridge backtest."""

    n_features = X_np.shape[1]
    model_config = {**config["model"], "n_features": n_features}
    # Patch config so worker gets enriched model_config
    config = {**config, "model": model_config}
    checkpoint_dir = config.get("checkpoint_dir", None)
    checkpoint_every = config.get("checkpoint_every", 0)
    loss_log_path = config.get("loss_log_path", None)

    def make_worker_args(gpu_id, chunks, config, all_train_X, all_train_y, all_test_X, chunk_size, num_windows):
        weights_dir = config.get("weights_dir", None)
        per_gpu_loss_path = None
        if loss_log_path:
            base, ext = os.path.splitext(loss_log_path)
            per_gpu_loss_path = f"{base}_gpu{gpu_id}{ext}"
        return (
            gpu_id,
            chunks,
            config["model"],
            config["train"],
            all_train_X,
            all_train_y,
            all_test_X,
            chunk_size,
            num_windows,
            weights_dir,
            checkpoint_dir,
            checkpoint_every,
            per_gpu_loss_path,
        )

    return run_backtest(
        X_np,
        y_np,
        dates,
        baselines,
        config,
        _ae_ridge_worker,
        _ae_ridge_make_windows,
        make_worker_args,
        label="AE+Ridge GPU Backtest",
        default_output="ae_ridge_results.csv",
    )
