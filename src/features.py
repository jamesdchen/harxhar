import os
import csv
import numpy as np
import torch
from sklearn.decomposition import PCA
from src.autoencoder import LagAutoEncoder, train_autoencoder


def make_har_features(df, cols, lags, feature_type, target_col='adj_RV'):
    """
    Generate HAR or raw lag features for the given columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source data containing the columns to transform.
    cols : list[str]
        Column names to create lag features for.
    lags : list[int]
        Lag windows (e.g. [1, 5, 25, 125, 625, 3125]).
    feature_type : str
        'har' for rolling-mean aggregates, 'raw' for individual point lags.
    target_col : str
        Name of the target column (used for naming HAR features).

    Returns
    -------
    feature_dict : dict[str, pd.Series]
        Mapping of feature name → computed series.
    feature_names : list[str]
        Ordered list of feature names.
    """
    feature_dict = {}
    feature_names = []

    for col in cols:
        for lag in lags:
            if feature_type == 'har':
                name = f"har_ma_{lag}" if col == target_col else f"{col}_ma_{lag}"
                feature_dict[name] = df[col].rolling(window=lag, min_periods=1).mean().shift(1)
            else:  # 'raw'
                name = f"{col}_lag_{lag}"
                feature_dict[name] = df[col].shift(lag)
            feature_names.append(name)

    return feature_dict, feature_names


class PCATransform:
    """Sklearn-like PCA wrapper for use as a rolling feature transform."""

    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)


class AETransform:
    """Autoencoder feature transform with reconstruction + prediction loss."""

    def __init__(self, n_features, n_components, alpha=0.5, hidden_dim=None,
                 epochs=50, lr=1e-3, ae_loss_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ae = LagAutoEncoder(n_features, n_components, hidden_dim).to(self.device)
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.ae_loss_path = ae_loss_path
        self._loss_log = []

    def fit(self, X, y=None):
        y_flat = y.ravel() if y is not None else np.zeros(X.shape[0])
        train_autoencoder(
            self.ae, X, y_flat,
            alpha=self.alpha, epochs=self.epochs, lr=self.lr,
            device=self.device, loss_log=self._loss_log,
        )
        if self.ae_loss_path is not None:
            self._flush_loss_log()
        return self

    def transform(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        return self.ae.encode(X_t).cpu().numpy()

    def _flush_loss_log(self):
        if not self._loss_log:
            return
        write_header = not os.path.exists(self.ae_loss_path)
        with open(self.ae_loss_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["recon", "pred", "total"])
            if write_header:
                writer.writeheader()
            writer.writerows(self._loss_log)
        self._loss_log.clear()
