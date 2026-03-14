import numpy as np
import torch
import torch.nn as nn
from transformers import PatchTSMixerConfig, PatchTSMixerModel, PreTrainedModel


# --- PatchTSMixer ---

class PatchTSMixerForecaster(PreTrainedModel):
    config_class = PatchTSMixerConfig

    def __init__(self, config):
        super().__init__(config)
        self.backbone = PatchTSMixerModel(config)

        dummy_input = torch.zeros(1, config.context_length, config.num_input_channels)
        with torch.no_grad():
            dummy_out = self.backbone(past_values=dummy_input).last_hidden_state

        self.num_patches = dummy_out.shape[2]
        self.flat_dim = self.num_patches * config.d_model

        self.head = nn.Linear(self.flat_dim, config.prediction_length)
        self.head.weight.data.normal_(0, 0.001)
        self.head.bias.data.fill_(0.0)

        self.post_init()

    def forward(self, past_values, future_values=None):
        outputs = self.backbone(past_values=past_values)
        last_hidden_state = outputs.last_hidden_state
        batch_size, num_channels, _, _ = last_hidden_state.shape
        flattened = last_hidden_state.view(batch_size, num_channels, -1)
        return self.head(flattened)


def get_model(cfg):
    config = PatchTSMixerConfig(
        context_length=cfg['context_len'],
        prediction_length=1,
        num_input_channels=cfg['num_input_channels'],
        d_model=cfg['hidden_dim'],
        num_layers=cfg.get('num_layers', 4),
        dropout=cfg['dropout'],
        patch_length=cfg['patch_len'],
        patch_stride=cfg['stride'],
        gated_attn=False,
        norm_type='layernorm',
        scaling=None
    )
    return PatchTSMixerForecaster(config)


# --- Lag Autoencoder ---

class LagAutoEncoder(nn.Module):
    """
    Hybrid autoencoder for lag feature compression.

    Three branches sharing a single encoder:
      - Decoder: reconstructs the input (unsupervised, reconstruction loss)
      - Head:    single linear layer predicting the target (supervised, prediction loss)

    Training loss: L = alpha * MSE(reconstructed, x) + (1 - alpha) * MSE(pred_rv, y)

    At inference only encode() is called; the Ridge regressor on top of the
    latent z is what actually produces final forecasts.
    """

    def __init__(self, n_features: int, n_components: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(n_components, n_features // 2)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )

        # Supervised prediction head
        self.head = nn.Linear(n_components, 1)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        pred_rv = self.head(z).squeeze(-1)
        return reconstructed, z, pred_rv

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)


def train_autoencoder(
    model: LagAutoEncoder,
    X_scaled: np.ndarray,
    y: np.ndarray,
    alpha: float,
    epochs: int,
    lr: float,
    device: torch.device,
    loss_log: list | None = None,
) -> LagAutoEncoder:
    """
    Train the hybrid autoencoder on the current rolling window.

    Parameters
    ----------
    model       : LagAutoEncoder instance (trained in-place and returned)
    X_scaled    : (N, n_features) standardized lag features
    y           : (N,) target values
    alpha       : weight for reconstruction loss; (1-alpha) weights prediction loss
    epochs      : number of full passes over the training data
    lr          : Adam learning rate
    device      : torch device
    loss_log    : optional list to which per-epoch dicts are appended
                  each dict: {'recon': float, 'pred': float, 'total': float}

    Returns
    -------
    model (same object, trained in-place)
    """
    model = model.to(device)
    model.train()

    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    y_t = torch.tensor(y.ravel(), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed, _z, pred_rv = model(X_t)

        recon_loss = mse(reconstructed, X_t)
        pred_loss = mse(pred_rv, y_t)
        total_loss = alpha * recon_loss + (1.0 - alpha) * pred_loss

        total_loss.backward()
        optimizer.step()

        if loss_log is not None:
            loss_log.append({
                "recon": recon_loss.item(),
                "pred": pred_loss.item(),
                "total": total_loss.item(),
            })

    model.eval()
    return model
