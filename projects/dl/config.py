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
        "batch_size": 64,
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
        "batch_size": 4,  # windows per batch (reduced from 10 to avoid OOM)
    },
}
