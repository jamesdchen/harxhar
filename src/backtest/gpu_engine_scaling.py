"""GPU training engine for scaling-law experiments with synthetic data augmentation.

Adapts the existing GPU backtest infrastructure for a simpler use case: train
a single PatchTSMixer model on real + MBB-augmented data and evaluate on a
chronological holdout.  Reuses model factory, loss, normalization, Duan
smearing, and results construction from the existing codebase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import config as cfg
from src.backtest.engine import apply_duan_smearing
from src.backtest.gpu_kernels import functional_qlike_loss
from src.backtest.gpu_utils import normalize_chunks
from src.data.synth_data import MovingBlockBootstrap
from src.evaluation.metrics import calculate_global_metrics
from src.log import get_logger
from src.models.deep_learning import get_model

logger = get_logger(__name__)


def _windows_from_series(series: np.ndarray, context_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window (X, y) arrays from a 1-D adj_RV series.

    Mirrors the lag structure produced by ``load_and_prep_data_strided`` with
    ``feature_type='raw'`` and ``lag=context_len``.

    Parameters
    ----------
    series : 1-D array of adj_RV values (sqrt-space).
    context_len : number of consecutive lags per window.

    Returns
    -------
    X : (N, context_len) array of lagged features.
    y : (N,) array of targets (one step ahead).
    """
    n = len(series) - context_len
    X = np.lib.stride_tricks.sliding_window_view(series, context_len)[:n]
    y = series[context_len : context_len + n]
    return X.astype(np.float64), y.astype(np.float64)


def _build_augmented_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    multiplier: int,
    context_len: int,
    block_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine real training data with MBB-generated synthetic windows.

    The MBB operates on the 1-D target series (adj_RV) to preserve temporal
    structure, then sliding windows are extracted from the synthetic series
    using the same lag convention as the real data.
    """
    if multiplier <= 0:
        return X_train, y_train

    mbb = MovingBlockBootstrap(y_train, block_size=block_size)
    n_synth_raw = multiplier * len(y_train)
    synth_series = mbb.generate(n_synth_raw, seed=seed)
    X_synth, y_synth = _windows_from_series(synth_series, context_len)

    logger.info(
        "Augmentation: %d real + %d synthetic = %d total windows",
        len(X_train),
        len(X_synth),
        len(X_train) + len(X_synth),
    )
    return (
        np.vstack([X_train, X_synth]),
        np.concatenate([y_train, y_synth]),
    )


def _normalize_instance(X: torch.Tensor) -> torch.Tensor:
    """Per-sample instance normalization matching the backtest convention.

    Uses ``normalize_chunks`` from ``gpu_utils`` with a dummy test tensor,
    then discards the test output.
    """
    dummy_test = torch.zeros_like(X[:1])
    X_norm, _ = normalize_chunks(X, dummy_test, dim=1, use_train_stats_for_test=False)
    return X_norm


def train_model(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    train_config: dict,
    device: torch.device,
) -> list[float]:
    """Train a PatchTSMixer on the (possibly augmented) dataset.

    Uses the same loss (``functional_qlike_loss``), optimizer (AdamW), and
    gradient clipping (``GRAD_CLIP_BOUND``) as the existing GPU backtest engine.

    Returns per-epoch mean losses for monitoring.
    """
    num_epochs = train_config["num_epochs"]
    lr = train_config["learning_rate"]
    batch_size = train_config["batch_size"]

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(cfg.ADAMW_BETA1, cfg.ADAMW_BETA2),
        weight_decay=cfg.ADAMW_WEIGHT_DECAY,
    )
    model.train()
    epoch_losses: list[float] = []

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            # Model expects (batch, seq_len, channels) — add channel dim
            h_pred = model(x_batch.unsqueeze(-1))
            # Output shape: (batch, channels=1, prediction_length=1) → scalar
            h_pred = h_pred.squeeze(-1).squeeze(-1)
            loss = functional_qlike_loss(h_pred, y_batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_BOUND)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(epoch_loss)
        if epoch % 25 == 0 or epoch == 1:
            logger.info("  Epoch %3d/%d  loss=%.4f", epoch, num_epochs, epoch_loss)

    return epoch_losses


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    y_test_np: np.ndarray,
    baselines_test: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float]:
    """Evaluate on the chronological holdout set.

    Prediction conversion follows ``gpu_engine.py:67``:
    ``pred_sqrt = exp(h_pred / 2)`` (log-space → sqrt-space).

    Raw-space conversion uses ``apply_duan_smearing`` from ``engine.py``.
    Final metrics via ``calculate_global_metrics`` from the evaluation library.
    """
    model.eval()
    loader = DataLoader(TensorDataset(X_test), batch_size=batch_size, shuffle=False)

    preds_sqrt: list[np.ndarray] = []
    for (x_batch,) in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        h_pred = model(x_batch.unsqueeze(-1))
        h_pred = h_pred.squeeze(-1).squeeze(-1)
        # Convert from log-space to sqrt-space (matching gpu_engine.py:67)
        preds_sqrt.append(torch.exp(h_pred / 2.0).cpu().numpy())

    forecasts = np.concatenate(preds_sqrt)
    pred_raw, true_raw = apply_duan_smearing(forecasts, y_test_np, baselines_test)

    df = pd.DataFrame({"true_raw": true_raw, "pred_raw": pred_raw})
    return calculate_global_metrics(df)


def run_scaling_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baselines_test: np.ndarray,
    model_config: dict,
    train_config: dict,
    multiplier: int,
    block_size: int = 48,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    """Run a single scaling experiment for a given synthetic data multiplier.

    Parameters
    ----------
    X_train, y_train : real training arrays from ``load_and_prep_data_strided``.
    X_test, y_test : chronological holdout arrays.
    baselines_test : baseline RV for Duan smearing on the test set.
    model_config : model hyperparameters (passed to ``get_model``).
    train_config : training hyperparameters (epochs, lr, batch_size).
    multiplier : 0 = real only, k = real + k× synthetic.
    block_size : MBB block length (default 48 = one trading day).
    seed : random seed for reproducibility.
    device : torch device (auto-detected if None).

    Returns
    -------
    dict with keys: multiplier, seed, qlike, mse, mae, n_train_windows,
    epoch_losses (list).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    context_len = model_config["context_len"]

    # --- Augment ---
    X_aug, y_aug = _build_augmented_data(
        X_train, y_train, multiplier, context_len, block_size, seed
    )
    n_windows = len(X_aug)

    # --- Normalize & tensorize ---
    X_aug_t = torch.tensor(X_aug, dtype=torch.float32)
    X_aug_t = _normalize_instance(X_aug_t)
    y_aug_t = torch.tensor(y_aug, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    X_test_t = _normalize_instance(X_test_t)

    # --- Train ---
    model = get_model(model_config).to(device)
    logger.info(
        "Training: mult=%d, seed=%d, n_windows=%d, device=%s",
        multiplier, seed, n_windows, device,
    )
    epoch_losses = train_model(model, X_aug_t, y_aug_t, train_config, device)

    # --- Evaluate ---
    metrics = evaluate_model(model, X_test_t, y_test, baselines_test, device)
    logger.info(
        "Result: mult=%d, seed=%d, QLIKE=%.6f, MSE=%.4e, MAE=%.4e",
        multiplier, seed,
        metrics.get("qlike", float("nan")),
        metrics.get("mse", float("nan")),
        metrics.get("mae", float("nan")),
    )

    return {
        "multiplier": multiplier,
        "seed": seed,
        "qlike": metrics.get("qlike", float("nan")),
        "mse": metrics.get("mse", float("nan")),
        "mae": metrics.get("mae", float("nan")),
        "n_train_windows": n_windows,
        "epoch_losses": epoch_losses,
    }
