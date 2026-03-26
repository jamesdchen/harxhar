"""Deep learning specific configuration."""

# --- GPU Training Constants ---
NORM_EPS = 1e-8
GRAD_CLIP_BOUND = 5.0
QLIKE_CLAMP_MIN = -30.0
QLIKE_CLAMP_MAX = 30.0
ADAMW_BETA1 = 0.9
ADAMW_BETA2 = 0.999
ADAMW_WEIGHT_DECAY = 0.01
GPU_WORKER_LOG = "worker_log.txt"
CHECKPOINT_EVERY_N_CHUNKS = 10  # save checkpoint every N chunks (0 = disabled)
AE_REFIT_FREQUENCY = 240  # steps between autoencoder refits

# --- DL (PatchTST) Configuration ---
DL_CONFIG = {
    "output_path": "results.csv",
    "train_window": 24000,  # ~500 trading days (~2 years)
    "gpu_count": 2,
    "model": {
        "context_len": 240,  # 5 trading days (5 × 48)
        "num_input_channels": 1,
        "hidden_dim": 32,
        "num_heads": 4,  # 8 dims per head
        "num_layers": 2,
        "ffn_dim": 128,  # 4x hidden_dim
        "dropout": 0.3,
        "patch_len": 48,  # 1 trading day
        "stride": 48,  # non-overlapping patches
        "prediction_length": 1,
    },
    "train": {
        "num_epochs": 50,
        "learning_rate": 5e-4,
        "batch_size": 32,  # reduced from 64 to avoid OOM on V100/A40
        "optimizer": "ADAMW",
        "loss_fn": "QLIKE",
    },
}

# --- AE+Ridge GPU Configuration ---
AE_RIDGE_GPU_CONFIG = {
    "output_path": "ae_ridge_results.csv",
    "train_window": 24000,  # 500 days * 48 periods
    "gpu_count": 2,
    "model": {
        "n_features": 0,  # set at runtime from X.shape[1]
        "n_components": 5,
        "hidden_dim": 0,  # 0 = auto (n_features // 2)
        "alpha_recon": 0.5,  # weight: alpha*recon + (1-alpha)*pred
        "alpha_ridge": 1.0,  # Ridge regularization strength
    },
    "train": {
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 2,  # windows per batch (reduced from 4 to avoid OOM on V100-32GB)
    },
}

# --- Auto Chunk Sizing ---
# Measured 2026-03-25 on 2×V100 (job 7580680): 221,820 windows in ~29 min
# computation across 2 GPUs, plus ~180s startup per task.
# Optimal range: 5-10 chunks (balances parallelism vs startup overhead).
CHUNK_SIZING = {
    "patchts": {
        "per_window_seconds": 0.016,  # ~0.5s per 32-window batch / 32 ≈ 16ms/window on 2×GPU
        "startup_overhead": 180,  # ~3 min: conda + data load + GPU init + torch compile
    },
    "ae_ridge": {
        "per_window_seconds": 0.064,  # ~4× patchts (batch_size=2, AE refit overhead)
        "startup_overhead": 180,
    },
}
DEFAULT_WALLTIME_SECONDS = 6 * 3600  # 6 hours, matches SLURM --time
