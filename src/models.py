import os
import csv
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
from src.rolling import RollingRobustScaler, RollingBuffer
from src.autoencoder import LagAutoEncoder, train_autoencoder

# --- 1. Top-Level Interface ---
class BaseModel:
    def initialize(self, X_init, y_init): pass
    def predict(self, x_t): pass
    def update(self, x_t, y_t): pass

# --- 2. The Engine (Handles all Rolling/Scaling Logic) ---
class RollingRegressionModel(BaseModel):
    def __init__(self, model, train_win_periods, n_features, use_scaling=True, refit_frequency=1):
        self.model = model
        self.train_win_periods = train_win_periods
        self.n_features = n_features
        self.use_scaling = use_scaling
        self.refit_frequency = refit_frequency
        self.steps_since_refit = 0
        
        self.buffer = RollingBuffer(train_win_periods, n_features, 1)
        
        if self.use_scaling:
            self.scaler = RollingRobustScaler(train_win_periods, n_features)
        
        self.mean_x = np.zeros(n_features)
        self.std_x = np.ones(n_features)
        
        self.hist_X = []
        self.hist_y = []

    def initialize(self, X_init, y_init):
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, 1)
            
        if self.use_scaling:
            self.scaler.initialize(X_init)
            self.mean_x, self.std_x = self.scaler.get_scaler()
            X_buffered = (X_init - self.mean_x) / self.std_x
        else:
            X_buffered = X_init

        self.buffer.X_buffer[:] = X_buffered
        self.buffer.y_buffer[:] = y_init
        
        self.hist_X = list(X_init)
        self.hist_y = list(y_init)
        
        X_tr, y_tr = self.buffer.get_view()
        self.model.fit(X_tr, y_tr)

    def predict(self, x_t):
        if self.use_scaling:
            x_input = (x_t - self.mean_x) / self.std_x
        else:
            x_input = x_t
            
        return self.model.predict(x_input.reshape(1, -1)).item()

    def update(self, x_t, y_t):
        # Update Scaler
        if self.use_scaling:
            self.scaler.update(x_t)
            self.mean_x, self.std_x = self.scaler.get_scaler()
            x_new = (x_t - self.mean_x) / self.std_x
        else:
            x_new = x_t

        # Add new to buffer and history
        self.buffer.add(x_new, y_t)
        self.hist_X.append(x_t)
        self.hist_y.append(y_t)

        # Conditionally Refit
        self.steps_since_refit += 1
        if self.steps_since_refit >= self.refit_frequency:
            X_tr, y_tr = self.buffer.get_view()
            self.model.fit(X_tr, y_tr)
            self.steps_since_refit = 0


# --- 3. The Specific Algorithms ---

class RidgeModel(RollingRegressionModel):
    def __init__(self, train_win_periods, n_features, use_scaling=True, **ridge_kwargs):
        # Initialize Ridge and pass it up to the parent engine
        model = Ridge(**ridge_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=1 # Ridge is fast, refit every step
        )

class XGBoostModel(RollingRegressionModel):
    def __init__(self, train_win_periods, n_features, use_scaling=False, refit_frequency=5, **xgb_kwargs):
        # Apply XGBoost speed tweaks
        if 'tree_method' not in xgb_kwargs:
            xgb_kwargs['tree_method'] = 'hist'
        if 'n_jobs' not in xgb_kwargs:
            xgb_kwargs['n_jobs'] = -1
            
        # Initialize XGBoost and pass it up to the parent engine
        model = XGBRegressor(**xgb_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency
        )

class LightGBMModel(RollingRegressionModel):
    def __init__(self, train_win_periods, n_features, use_scaling=False, refit_frequency=5, **lgbm_kwargs):
        # Apply default speed/system tweaks if not provided
        if 'n_jobs' not in lgbm_kwargs:
            lgbm_kwargs['n_jobs'] = -1
        # Suppress some common LightGBM verbosity by default
        if 'verbose' not in lgbm_kwargs:
            lgbm_kwargs['verbose'] = -1
            
        # Initialize LightGBM and pass it to the parent engine
        model = LGBMRegressor(**lgbm_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency
        )


class RandomForestModel(RollingRegressionModel):
    def __init__(self, train_win_periods, n_features, use_scaling=False, refit_frequency=5, **rf_kwargs):
        # Apply default speed tweaks if not provided
        if 'n_jobs' not in rf_kwargs:
            rf_kwargs['n_jobs'] = -1
            
        # Initialize Random Forest and pass it to the parent engine
        model = RandomForestRegressor(**rf_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency
        )

# --- 4. SARIMAX Model ---

class _SARIMAXEstimator:
    """
    Thin sklearn-style wrapper around statsmodels SARIMAX.

    Accepts fit(X, y) where X/y are in chronological order, and
    predict(X) for one-step-ahead forecasting with exogenous data.
    Silently retains the previous fitted result on any failure.
    """

    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order
        self._result = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        exog = np.asarray(X, dtype=np.float64) if X.shape[1] > 0 else None
        try:
            m = _SARIMAX(
                endog=y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._result = m.fit(disp=False, method="lbfgs", maxiter=100)
        except Exception:
            pass
        return self

    def predict(self, X):
        """Return np.array([scalar]) so .item() in RollingRegressionModel works."""
        if self._result is None:
            return np.array([0.0])
        exog = np.asarray(X, dtype=np.float64) if X.shape[1] > 0 else None
        try:
            fc = self._result.forecast(steps=1, exog=exog)
            return np.array([float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0])])
        except Exception:
            return np.array([0.0])


class SARIMAXModel(RollingRegressionModel):
    """
    SARIMAX baseline that inherits RollingRegressionModel for buffer/scaler reuse.

    Uses raw-lag exogenous features (same lag indices as HAR_LAGS but as individual
    point values) together with ARMA(p,q) and seasonal AR components. The rolling
    window is fit_window periods (default 480 = 10 trading days), much shorter than
    the outer train_win_periods, since SARIMAX is a parametric model.

    Parameters
    ----------
    train_win_periods : int
        Burn-in window passed in by the backtester (only used to slice X_init/y_init).
    n_features : int
        Number of exogenous features (raw-lag columns).
    fit_window : int
        Internal buffer / fitting window. Defaults to 480 (10 days of 30-min bars).
    refit_frequency : int
        Steps between refits. Defaults to 48 (once per simulated day).
    order : tuple
        ARIMA (p, d, q) non-seasonal order.
    seasonal_order : tuple
        (P, D, Q, s) seasonal order. s=48 for daily seasonality on 30-min bars.
    """

    def __init__(
        self,
        train_win_periods,
        n_features,
        fit_window=480,
        refit_frequency=48,
        order=(2, 0, 1),
        seasonal_order=(1, 0, 0, 48),
    ):
        estimator = _SARIMAXEstimator(order, seasonal_order)
        # Pass fit_window as train_win_periods so buffer/scaler size = fit_window
        super().__init__(
            model=estimator,
            train_win_periods=fit_window,
            n_features=n_features,
            use_scaling=True,
            refit_frequency=refit_frequency,
        )

    def initialize(self, X_init, y_init):
        """Override to slice X_init/y_init to fit_window and use ordered fit."""
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, 1)

        # Slice to the buffer size (fit_window rows)
        X_init = X_init[-self.train_win_periods:]
        y_init = y_init[-self.train_win_periods:]

        if self.use_scaling:
            self.scaler.initialize(X_init)
            self.mean_x, self.std_x = self.scaler.get_scaler()
            X_buffered = (X_init - self.mean_x) / self.std_x
        else:
            X_buffered = X_init

        # Fill buffer directly; ptr stays 0 → data is already chronological
        self.buffer.X_buffer[:] = X_buffered
        self.buffer.y_buffer[:] = y_init

        # Fit SARIMAX with chronologically ordered data
        X_tr, y_tr = self.buffer.get_ordered_view()
        self.model.fit(X_tr, y_tr)

    def update(self, x_t, y_t):
        """Override to use get_ordered_view() when refitting."""
        if self.use_scaling:
            self.scaler.update(x_t)
            self.mean_x, self.std_x = self.scaler.get_scaler()
            x_new = (x_t - self.mean_x) / self.std_x
        else:
            x_new = x_t

        self.buffer.add(x_new, y_t)

        self.steps_since_refit += 1
        if self.steps_since_refit >= self.refit_frequency:
            X_tr, y_tr = self.buffer.get_ordered_view()
            self.model.fit(X_tr, y_tr)
            self.steps_since_refit = 0


# --- 5. PCA + Ridge ---

class PCALagRidgeModel(BaseModel):
    """
    Compresses raw lag features with PCA (fit on the correlation matrix, i.e.
    robust-scaled features) then regresses with Ridge.

    Rolling refit matches Ridge cadence (every step by default).
    """

    def __init__(self, train_win_periods, n_features, n_components, alpha=1.0, refit_frequency=1):
        self.n_components = n_components
        self.refit_frequency = refit_frequency
        self.steps_since_refit = 0

        self.buffer = RollingBuffer(train_win_periods, n_features, 1)
        self.scaler = RollingRobustScaler(train_win_periods, n_features)

        self.pca = PCA(n_components=n_components)
        self.ridge = Ridge(alpha=alpha)

        self.mean_x = np.zeros(n_features)
        self.std_x = np.ones(n_features)
        self.is_fitted = False

    def _scale(self, x):
        return (x - self.mean_x) / self.std_x

    def initialize(self, X_init, y_init):
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, 1)

        self.scaler.initialize(X_init)
        self.mean_x, self.std_x = self.scaler.get_scaler()
        X_scaled = self._scale(X_init)

        self.buffer.X_buffer[:] = X_scaled.astype(np.float32)
        self.buffer.y_buffer[:] = y_init.astype(np.float32)

        self.pca.fit(X_scaled)
        X_pca = self.pca.transform(X_scaled)
        self.ridge.fit(X_pca, y_init.ravel())
        self.is_fitted = True

    def predict(self, x_t):
        if not self.is_fitted:
            return 0.0
        x_scaled = self._scale(x_t)
        x_pca = self.pca.transform(x_scaled.reshape(1, -1))
        return self.ridge.predict(x_pca).item()

    def update(self, x_t, y_t):
        self.scaler.update(x_t)
        self.mean_x, self.std_x = self.scaler.get_scaler()
        x_scaled = self._scale(x_t)
        self.buffer.add(x_scaled, y_t)

        self.steps_since_refit += 1
        if self.steps_since_refit >= self.refit_frequency:
            X_tr, y_tr = self.buffer.get_view()
            self.pca.fit(X_tr)
            X_pca = self.pca.transform(X_tr)
            self.ridge.fit(X_pca, y_tr.ravel())
            self.steps_since_refit = 0


# --- 6. Hybrid AutoEncoder + Ridge ---

class AutoEncoderLagRidgeModel(BaseModel):
    """
    Compresses raw lag features with a hybrid autoencoder (reconstruction +
    RV-prediction loss) then regresses with Ridge on the latent embedding.

    Neural-network refit is expensive, so refit_frequency defaults to 240
    steps (~5 trading days of 30-min bars).

    AE training losses are appended to loss_log (list of dicts) when provided,
    and flushed to a CSV at ae_loss_path after every refit so they are available
    for later plotting.
    """

    def __init__(
        self,
        train_win_periods,
        n_features,
        n_components,
        alpha=0.5,
        hidden_dim=None,
        epochs=50,
        lr=1e-3,
        refit_frequency=240,
        ae_loss_path=None,
    ):
        self.n_features = n_features
        self.n_components = n_components
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.refit_frequency = refit_frequency
        self.ae_loss_path = ae_loss_path
        self.steps_since_refit = 0

        self.buffer = RollingBuffer(train_win_periods, n_features, 1)
        self.scaler = RollingRobustScaler(train_win_periods, n_features)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ae = LagAutoEncoder(n_features, n_components, hidden_dim)
        self.ridge = Ridge(alpha=1.0)

        self.mean_x = np.zeros(n_features)
        self.std_x = np.ones(n_features)
        self.is_fitted = False

        # Accumulates per-epoch loss dicts across all refits
        self._loss_log = []

    def _scale(self, x):
        return (x - self.mean_x) / self.std_x

    def _encode_np(self, X_scaled_np):
        """Encode a numpy array, return numpy array of latents."""
        X_t = torch.tensor(X_scaled_np, dtype=torch.float32, device=self.device)
        z = self.ae.encode(X_t)
        return z.cpu().numpy()

    def _refit_ae_and_ridge(self, X_scaled_np, y_np):
        train_autoencoder(
            self.ae, X_scaled_np, y_np,
            alpha=self.alpha,
            epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            loss_log=self._loss_log,
        )
        z = self._encode_np(X_scaled_np)
        self.ridge.fit(z, y_np.ravel())

        if self.ae_loss_path is not None:
            self._flush_loss_log()

    def _flush_loss_log(self):
        """Write accumulated loss entries to CSV (append mode)."""
        if not self._loss_log:
            return
        write_header = not os.path.exists(self.ae_loss_path)
        with open(self.ae_loss_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["recon", "pred", "total"])
            if write_header:
                writer.writeheader()
            writer.writerows(self._loss_log)
        self._loss_log.clear()

    def initialize(self, X_init, y_init):
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, 1)

        self.scaler.initialize(X_init)
        self.mean_x, self.std_x = self.scaler.get_scaler()
        X_scaled = self._scale(X_init)

        self.buffer.X_buffer[:] = X_scaled.astype(np.float32)
        self.buffer.y_buffer[:] = y_init.astype(np.float32)

        self._refit_ae_and_ridge(X_scaled, y_init.ravel())
        self.is_fitted = True

    def predict(self, x_t):
        if not self.is_fitted:
            return 0.0
        x_scaled = self._scale(x_t)
        z_t = self._encode_np(x_scaled.reshape(1, -1))
        return self.ridge.predict(z_t).item()

    def update(self, x_t, y_t):
        self.scaler.update(x_t)
        self.mean_x, self.std_x = self.scaler.get_scaler()
        x_scaled = self._scale(x_t)
        self.buffer.add(x_scaled, y_t)

        self.steps_since_refit += 1
        if self.steps_since_refit >= self.refit_frequency:
            X_tr, y_tr = self.buffer.get_view()
            self._refit_ae_and_ridge(X_tr.astype(np.float64), y_tr.ravel().astype(np.float64))
            self.steps_since_refit = 0


# --- 7. The Baseline ---

class NaiveBaseline(BaseModel):
    # Notice this skips the RollingRegressionModel and inherits straight from BaseModel
    # because it doesn't need to waste memory on buffers or scaling!
    def __init__(self, lag_index=0):
        self.lag_index = lag_index

    def initialize(self, X_init, y_init):
        pass 

    def predict(self, x_t):
        return x_t[self.lag_index]

    def update(self, x_t, y_t):
        pass