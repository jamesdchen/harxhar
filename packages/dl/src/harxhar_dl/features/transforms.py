"""Autoencoder feature transform for deep learning-based feature compression."""

from __future__ import annotations

import os

import numpy as np
from harxhar_core.features.transforms import BaseFeatureTransform


class AETransform(BaseFeatureTransform):
    """Autoencoder feature transform with reconstruction + prediction loss."""

    def __init__(
        self,
        n_features: int,
        n_components: int,
        alpha: float = 0.5,
        hidden_dim: int | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        ae_loss_path: str | None = None,
        ae_weights_dir: str | None = None,
    ) -> None:
        import torch

        from harxhar_dl.models.deep_learning import LagAutoEncoder

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ae = LagAutoEncoder(n_features, n_components, hidden_dim).to(self.device)
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.ae_loss_path = ae_loss_path
        self.ae_weights_dir = ae_weights_dir
        self._loss_log: list[dict[str, float]] = []
        self._refit_count = 0
        self.frozen = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> AETransform:
        if self.frozen:
            return self

        from harxhar_dl.models.deep_learning import train_autoencoder

        y_flat = y.ravel() if y is not None else np.zeros(X.shape[0])
        train_autoencoder(
            self.ae,
            X,
            y_flat,
            alpha=self.alpha,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            loss_log=self._loss_log,
        )
        if self.ae_loss_path is not None:
            self._flush_loss_log()
        if self.ae_weights_dir is not None:
            self._save_weights()
        return self

    def load_weights(self, path: str) -> AETransform:
        """Load pre-trained AE weights and freeze (skip future fit calls)."""
        import torch

        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.ae.load_state_dict(state_dict)
        self.ae.eval()
        self.frozen = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        import torch

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.ae.encode(X_t).cpu().numpy()

    def _flush_loss_log(self) -> None:
        import csv

        if not self._loss_log or self.ae_loss_path is None:
            return
        write_header = not os.path.exists(self.ae_loss_path)
        with open(self.ae_loss_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["recon", "pred", "total"])
            if write_header:
                writer.writeheader()
            writer.writerows(self._loss_log)
        self._loss_log.clear()

    def _save_weights(self) -> None:
        import torch

        if self.ae_weights_dir is None:
            return
        os.makedirs(self.ae_weights_dir, exist_ok=True)
        path = os.path.join(self.ae_weights_dir, f"ae_weights_{self._refit_count:04d}.pt")
        torch.save(self.ae.state_dict(), path)
        self._refit_count += 1
