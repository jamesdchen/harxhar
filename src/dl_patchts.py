# Auto-generated from notebooks/dl_patchts.ipynb. Do not edit by hand.

"""PatchTST GPU backtest executor for volatility forecasting.

Self-contained CLI: load -> transform -> PatchTST GPU backtest -> save chunk CSV.
"""

import argparse
import gc
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel, PreTrainedModel

from src.evaluation import apply_duan_smearing, calculate_metrics
from src.loading import load_raw_data
from src.transforms import robust_transform

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
PERIODS_PER_DAY = 48

DEFAULT_CONFIG = {
    "train_window": 24000,
    "gpu_count": 1,
    "model": {
        "context_len": 240,
        "num_input_channels": 1,
        "hidden_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 128,
        "dropout": 0.3,
        "patch_len": 48,
        "stride": 48,
        "prediction_length": 1,
    },
    "train": {
        "num_epochs": 50,
        "learning_rate": 5e-4,
        "batch_size": 32,
    },
}


# ── PatchTST Model ──────────────────────────────────────────────────────


class PatchTSTForecaster(PreTrainedModel):
    config_class = PatchTSTConfig

    def __init__(self, config):
        super().__init__(config)
        self.backbone = PatchTSTModel(config)
        dummy_input = torch.zeros(1, config.context_length, config.num_input_channels)
        with torch.no_grad():
            dummy_out = self.backbone(past_values=dummy_input).last_hidden_state
        self.num_patches = dummy_out.shape[2]
        self.flat_dim = self.num_patches * config.d_model
        self.head = nn.Linear(self.flat_dim, config.prediction_length)
        self.head.weight.data.normal_(0, 0.001)
        self.head.bias.data.fill_(0.0)
        self.post_init()

    def forward(self, past_values, future_values=None, output_attentions=False):
        outputs = self.backbone(past_values=past_values, output_attentions=output_attentions)
        last_hidden_state = outputs.last_hidden_state
        batch_size, num_channels, _, _ = last_hidden_state.shape
        flattened = last_hidden_state.view(batch_size, num_channels, -1)
        pred = self.head(flattened)
        if output_attentions:
            return pred, outputs.attentions
        return pred


# ── Model factory ────────────────────────────────────────────────────────


def get_model(cfg):
    config = PatchTSTConfig(
        context_length=cfg["context_len"],
        prediction_length=cfg.get("prediction_length", 1),
        num_input_channels=cfg["num_input_channels"],
        d_model=cfg["hidden_dim"],
        num_hidden_layers=cfg.get("num_layers", 4),
        num_attention_heads=cfg.get("num_heads", 2),
        ffn_dim=cfg.get("ffn_dim", cfg["hidden_dim"] * 4),
        attention_dropout=cfg["dropout"],
        ff_dropout=cfg["dropout"],
        path_dropout=cfg["dropout"],
        patch_length=cfg["patch_len"],
        patch_stride=cfg["stride"],
        norm_type="layernorm",
        scaling=None,
    )
    return PatchTSTForecaster(config)


# ── Strided window creation ─────────────────────────────────────────────


def make_patchts_windows(X_tensor, y_tensor, config):
    """Create strided windows for walk-forward PatchTST backtest."""
    train_window = config["train_window"]
    context_len = config["model"]["context_len"]
    total_samples = X_tensor.shape[0]
    num_windows = total_samples - train_window
    samples_per_window = train_window // context_len

    window_shape_X = (num_windows, samples_per_window, context_len)
    strides_X = (
        X_tensor.stride(0),
        X_tensor.stride(0) * context_len,
        X_tensor.stride(0),
    )
    all_train_X = torch.as_strided(X_tensor, size=window_shape_X, stride=strides_X)

    y_offset = y_tensor[context_len:]
    window_shape_y = (num_windows, samples_per_window, 1)
    strides_y = (
        y_offset.stride(0),
        y_offset.stride(0) * context_len,
        y_offset.stride(0),
    )
    all_train_y = torch.as_strided(y_offset, size=window_shape_y, stride=strides_y)

    X_test_start = X_tensor[train_window - context_len :]
    window_shape_test = (num_windows, 1, context_len)
    strides_test = (
        X_test_start.stride(0),
        X_test_start.stride(0),
        X_test_start.stride(0),
    )
    all_test_X = torch.as_strided(X_test_start, size=window_shape_test, stride=strides_test)

    return all_train_X, all_train_y, all_test_X, num_windows


# ── Instance normalization ───────────────────────────────────────────────


def instance_norm(x, eps=1e-8):
    """Per-window zero-mean unit-variance normalization."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=eps)
    return (x - mean) / std, mean, std


# ── QLIKE loss ───────────────────────────────────────────────────────────


def qlike_loss(pred, target, clamp_val=30.0):
    """QLIKE loss: exp(log_ratio) - log_ratio - 1."""
    log_ratio = torch.log(pred.clamp(min=1e-8)) - torch.log(target.clamp(min=1e-8))
    log_ratio = log_ratio.clamp(-clamp_val, clamp_val)
    return (torch.exp(log_ratio) - log_ratio - 1.0).mean()


# ── GPU training kernel ─────────────────────────────────────────────────


def _train_single_window(model, train_X, train_y, test_X, cfg, device):
    """Train PatchTST on one walk-forward window and return prediction."""
    num_epochs = cfg["train"]["num_epochs"]
    lr = cfg["train"]["learning_rate"]
    batch_size = cfg["train"]["batch_size"]

    train_flat = train_X.reshape(-1)
    t_mean = train_flat.mean()
    t_std = train_flat.std().clamp(min=1e-8)
    train_X_norm = (train_X - t_mean) / t_std
    test_X_norm = (test_X - t_mean) / t_std

    n_samples = train_X_norm.shape[0]
    train_X_3d = train_X_norm.unsqueeze(-1)
    test_X_3d = test_X_norm.unsqueeze(-1)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _epoch in range(num_epochs):
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            x_batch = train_X_3d[idx]
            y_batch = train_y[idx]

            pred = model(past_values=x_batch)
            pred_squeezed = pred.squeeze(-1).squeeze(-1)
            y_squeezed = y_batch.squeeze(-1)

            loss = qlike_loss(pred_squeezed, y_squeezed)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(past_values=test_X_3d)
        pred_val = pred_test.squeeze().item()

    return pred_val


# ── GPU worker ───────────────────────────────────────────────────────────


def _gpu_worker(gpu_id, window_indices, shared_data, config, result_dict):
    """Worker for a single GPU. Trains on assigned windows sequentially."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    all_train_X = shared_data["all_train_X"].to(device)
    all_train_y = shared_data["all_train_y"].to(device)
    all_test_X = shared_data["all_test_X"].to(device)

    predictions = {}
    for w_idx in window_indices:
        train_X = all_train_X[w_idx]
        train_y = all_train_y[w_idx]
        test_X = all_test_X[w_idx]

        model_fresh = get_model(config["model"]).to(device)
        pred = _train_single_window(
            model_fresh,
            train_X,
            train_y,
            test_X,
            config,
            device,
        )
        predictions[w_idx] = pred

        del model_fresh
        if w_idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    result_dict[gpu_id] = predictions


# ── Multi-GPU distribution ───────────────────────────────────────────────


def run_patchts_backtest(X_tensor, y_tensor, config):
    """Run PatchTST walk-forward backtest across GPUs."""
    gpu_count = config.get("gpu_count", 1)
    available_gpus = torch.cuda.device_count()
    gpu_count = min(gpu_count, available_gpus)

    if gpu_count == 0:
        raise RuntimeError("No CUDA GPU available")

    logger.info(f"Using {gpu_count} GPU(s) for PatchTST backtest")

    all_train_X, all_train_y, all_test_X, num_windows = make_patchts_windows(X_tensor, y_tensor, config)
    logger.info(f"Created {num_windows} walk-forward windows")

    shared_data = {
        "all_train_X": all_train_X,
        "all_train_y": all_train_y,
        "all_test_X": all_test_X,
    }

    if gpu_count == 1:
        result_dict: dict[int, dict[int, float]] = {}
        _gpu_worker(0, list(range(num_windows)), shared_data, config, result_dict)
        predictions = np.array([result_dict[0][i] for i in range(num_windows)])
    else:
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        result_dict = manager.dict()

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
                    shared_data,
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


def _seed_everything(seed: int = 42) -> None:
    """Pin RNGs for reproducibility (numpy, torch, cuda, cudnn).

    Call at the top of main() BEFORE any data loading or model construction.
    """
    import os
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PatchTST GPU walk-forward backtest")
    parser.add_argument("--data-path", default="all30min")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    _seed_everything(args.seed)

    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["gpu_count"] = args.gpu_count
    if args.epochs is not None:
        config["train"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate

    logger.info(f"Loading data from {args.data_path}")
    df = load_raw_data(args.data_path)

    adj_rv, baseline = robust_transform(
        df,
        "RV",
        use_diurnal=True,
        winsor_window=240,
        is_target=True,
    )
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    adj_rv_arr = df["adj_RV"].values.astype(np.float64)
    baseline_arr = df["baseline"].values.astype(np.float64)
    dates = df["t"]

    # Horizon shift (PatchTST uses 1D arrays, apply manually)
    if args.horizon > 1:
        shift = args.horizon - 1
        adj_rv_arr_X = adj_rv_arr[:-shift]
        adj_rv_arr_y = adj_rv_arr[shift:]
        dates = dates.iloc[:-shift].reset_index(drop=True)
        baseline_arr = baseline_arr[shift:]
    else:
        adj_rv_arr_X = adj_rv_arr
        adj_rv_arr_y = adj_rv_arr

    start = args.start
    end = len(adj_rv_arr_X) if args.end == -1 else args.end

    X_chunk = adj_rv_arr_X[start:end]
    y_chunk = adj_rv_arr_y[start:end]
    dates_chunk = dates.iloc[start:end].reset_index(drop=True)
    baselines_chunk = baseline_arr[start:end]

    train_window = config["train_window"]
    if train_window >= len(X_chunk):
        raise ValueError(f"train_window ({train_window}) >= chunk size ({len(X_chunk)})")

    X_tensor = torch.tensor(X_chunk, dtype=torch.float32)
    y_tensor = torch.tensor(y_chunk, dtype=torch.float32)

    t0 = time.time()
    preds = run_patchts_backtest(X_tensor, y_tensor, config)
    elapsed = time.time() - t0
    logger.info(f"Backtest complete in {elapsed:.1f}s")

    num_windows = len(preds)
    y_oos = y_chunk[train_window : train_window + num_windows]
    dates_oos = dates_chunk.iloc[train_window : train_window + num_windows].values
    baselines_oos = baselines_chunk[train_window : train_window + num_windows]

    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

    results = pd.DataFrame(
        {
            "date": dates_oos,
            "horizon": args.horizon,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )

    out_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(args.output_file, index=False)

    metrics = calculate_metrics(results)
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    logger.info(f"Saved {len(results)} rows -> {args.output_file}")


if __name__ == "__main__":
    main()
