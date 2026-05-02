# Auto-generated from notebooks/dl_ae_ridge.ipynb. Do not edit by hand.

"""AE+Ridge GPU backtest executor for volatility forecasting.

Self-contained CLI: load -> transform -> AE encode -> Ridge predict -> save chunk CSV.
"""

import gc
import json
import logging
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

from src.dl_executor import save_dl_results, seed_everything
from src.executor import load_and_transform
from src.transforms import apply_horizon_shift

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_AE_CONFIG = {
    "train_window": 24000,
    "gpu_count": 1,
    "model": {
        "n_features": 0,  # set at runtime
        "n_components": 5,
        "hidden_dim": 0,  # 0 = auto
        "alpha_recon": 0.5,
        "alpha_ridge": 1.0,
    },
    "train": {
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 2,
    },
}


# ── LagAutoEncoder Model ────────────────────────────────────────────────


class LagAutoEncoder(nn.Module):
    def __init__(self, n_features, n_components=5, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(n_components, n_features // 2)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )
        self.head = nn.Linear(n_components, 1)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        pred_rv = self.head(z).squeeze(-1)
        return reconstructed, z, pred_rv

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)


# ── Raw lag feature generation ───────────────────────────────────────────


def generate_raw_lags(series, max_lag=100):
    """Generate consecutive shift lags [1..max_lag] for a 1D array.

    Parameters
    ----------
    series : np.ndarray (1D)
        The adjusted RV time series.
    max_lag : int
        Maximum lag to include.

    Returns
    -------
    X : np.ndarray (n_samples, max_lag)
        Lag features (NaN in burn-in rows).
    """
    n = len(series)
    X = np.full((n, max_lag), np.nan, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        X[lag:, lag - 1] = series[:-lag]
    return X


# ── Instance normalization ───────────────────────────────────────────────


def instance_norm_np(X, eps=1e-8):
    """Per-window zero-mean unit-variance normalization (numpy)."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (X - mean) / std, mean, std


# ── AE training ──────────────────────────────────────────────────────────


def train_ae_window(model, X_train, y_train, cfg, device):
    """Train AE on one walk-forward window.

    Loss = alpha * MSE(reconstructed, x) + (1 - alpha) * MSE(pred, y)
    """
    num_epochs = cfg["train"]["num_epochs"]
    lr = cfg["train"]["learning_rate"]
    batch_size = cfg["train"]["batch_size"]
    alpha = cfg["model"]["alpha_recon"]

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_samples = X_train.shape[0]

    for _epoch in range(num_epochs):
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = X_train[idx]
            y_batch = y_train[idx]

            reconstructed, z, pred_rv = model(x_batch)

            loss_recon = nn.functional.mse_loss(reconstructed, x_batch)
            loss_pred = nn.functional.mse_loss(pred_rv, y_batch)
            loss = alpha * loss_recon + (1.0 - alpha) * loss_pred

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


# ── Ridge solve via torch.linalg ─────────────────────────────────────────


def ridge_solve(Z, y, alpha_ridge=1.0):
    """Solve Ridge regression: w = (Z^T Z + alpha * I)^{-1} Z^T y.

    Parameters
    ----------
    Z : (n, k) tensor — latent embeddings
    y : (n,) tensor — targets
    alpha_ridge : float

    Returns
    -------
    w : (k,) tensor — Ridge coefficients
    """
    k = Z.shape[1]
    ZtZ = Z.T @ Z + alpha_ridge * torch.eye(k, device=Z.device, dtype=Z.dtype)
    Zty = Z.T @ y
    w = torch.linalg.solve(ZtZ, Zty)
    return w


# ── Single window AE+Ridge prediction ───────────────────────────────────


def _predict_window(X_train, y_train, X_test, cfg, device):
    """Train AE, encode, solve Ridge, predict test point.

    Parameters
    ----------
    X_train : (train_window, n_features) tensor on device
    y_train : (train_window,) tensor on device
    X_test  : (1, n_features) tensor on device
    cfg : dict
    device : torch.device

    Returns
    -------
    float : prediction for this window
    """
    n_features = X_train.shape[1]
    n_components = cfg["model"]["n_components"]
    hidden_dim = cfg["model"]["hidden_dim"]
    if hidden_dim == 0:
        hidden_dim = None
    alpha_ridge = cfg["model"]["alpha_ridge"]

    # Instance normalization
    t_mean = X_train.mean(dim=0, keepdim=True)
    t_std = X_train.std(dim=0, keepdim=True).clamp(min=1e-8)
    X_train_norm = (X_train - t_mean) / t_std
    X_test_norm = (X_test - t_mean) / t_std

    # Build fresh AE
    model = LagAutoEncoder(n_features, n_components, hidden_dim).to(device)

    # Train AE
    train_ae_window(model, X_train_norm, y_train, cfg, device)

    # Encode training data
    model.eval()
    with torch.no_grad():
        Z_train = model.encode(X_train_norm)  # (train_window, n_components)

    # Solve Ridge
    w = ridge_solve(Z_train, y_train, alpha_ridge)

    # Encode test point and predict
    with torch.no_grad():
        z_test = model.encode(X_test_norm)  # (1, n_components)
    pred = (z_test @ w).item()

    return pred


# ── GPU worker ───────────────────────────────────────────────────────────


def _gpu_worker(gpu_id, window_indices, X_all, y_all, config, result_dict):
    """Worker for a single GPU. Processes assigned windows sequentially."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    train_window = config["train_window"]

    X_device = torch.tensor(X_all, dtype=torch.float32, device=device)
    y_device = torch.tensor(y_all, dtype=torch.float32, device=device)

    predictions = {}
    for w_idx in window_indices:
        # Walk-forward: train on [w_idx : w_idx + train_window], predict w_idx + train_window
        t_start = w_idx
        t_end = w_idx + train_window
        test_idx = t_end

        X_train = X_device[t_start:t_end]
        y_train = y_device[t_start:t_end]
        X_test = X_device[test_idx : test_idx + 1]

        pred = _predict_window(X_train, y_train, X_test, config, device)
        predictions[w_idx] = pred

        if w_idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    result_dict[gpu_id] = predictions


# ── Multi-GPU distribution ───────────────────────────────────────────────


def run_ae_ridge_backtest(X, y, config):
    """Run AE+Ridge walk-forward backtest across GPUs.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
    y : np.ndarray (n_samples,)
    config : dict

    Returns
    -------
    np.ndarray : predictions of shape (num_windows,)
    """
    gpu_count = config.get("gpu_count", 1)
    available_gpus = torch.cuda.device_count()
    gpu_count = min(gpu_count, available_gpus)

    if gpu_count == 0:
        raise RuntimeError("No CUDA GPU available")

    train_window = config["train_window"]
    num_windows = len(X) - train_window
    logger.info(f"Using {gpu_count} GPU(s) for AE+Ridge backtest, {num_windows} windows")

    if gpu_count == 1:
        result_dict: dict[int, dict[int, float]] = {}
        _gpu_worker(
            0,
            list(range(num_windows)),
            X,
            y,
            config,
            result_dict,
        )
        predictions = np.array([result_dict[0][i] for i in range(num_windows)])
    else:
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        result_dict = manager.dict()

        # Distribute windows across GPUs
        window_splits: list[list[int]] = [[] for _ in range(gpu_count)]
        for i in range(num_windows):
            window_splits[i % gpu_count].append(i)

        processes = []
        for gpu_id in range(gpu_count):
            p = ctx.Process(
                target=_gpu_worker,
                args=(
                    gpu_id,
                    window_splits[gpu_id],
                    X,
                    y,
                    config,
                    result_dict,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        all_preds = {}
        for gpu_id in range(gpu_count):
            all_preds.update(result_dict[gpu_id])

        predictions = np.array([all_preds[i] for i in range(num_windows)])

    return predictions


# ── CLI ──────────────────────────────────────────────────────────────────


def compute(args) -> None:
    seed_everything(args.seed)

    config = json.loads(json.dumps(DEFAULT_AE_CONFIG))  # deep copy
    config["gpu_count"] = args.gpu_count
    if args.epochs is not None:
        config["train"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if args.n_components is not None:
        config["model"]["n_components"] = args.n_components

    # 1. Load + RV transform
    logger.info(f"Loading data from {args.data_path}")
    df, _ = load_and_transform(
        args.data_path,
        exog_cols=[],
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
    )

    # 2. Generate raw lag features
    adj_rv_arr = df["adj_RV"].values.astype(np.float64)
    max_lag = 100
    X_lags = generate_raw_lags(adj_rv_arr, max_lag=max_lag)

    # 3. Drop NaN burn-in rows
    valid_mask = ~np.isnan(X_lags).any(axis=1)
    first_valid = np.argmax(valid_mask)
    X = X_lags[first_valid:]
    y = adj_rv_arr[first_valid:]
    dates = df["t"].iloc[first_valid:].reset_index(drop=True)
    baselines = df["baseline"].values.astype(np.float64)[first_valid:]

    # Set n_features in config
    config["model"]["n_features"] = X.shape[1]
    logger.info(f"Feature matrix shape: {X.shape}")

    # 4. Horizon shift
    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, args.horizon)

    # 5. Slice
    start = args.start
    end = len(X) if args.end == -1 else args.end

    X_chunk = X[start:end]
    y_chunk = y[start:end]
    dates_chunk = dates.iloc[start:end].reset_index(drop=True)
    baselines_chunk = baselines[start:end]

    train_window = config["train_window"]
    if train_window >= len(X_chunk):
        raise ValueError(f"train_window ({train_window}) >= chunk size ({len(X_chunk)})")

    # 6. Run AE+Ridge backtest
    logger.info("Running AE+Ridge backtest")
    t0 = time.time()
    preds = run_ae_ridge_backtest(X_chunk, y_chunk, config)
    elapsed = time.time() - t0
    logger.info(f"Backtest complete in {elapsed:.1f}s")

    # 7. Save (Duan smearing + DataFrame + CSV + metrics.json + log)
    save_dl_results(preds, y_chunk, dates_chunk, baselines_chunk, train_window, args.horizon, args.output_file)
