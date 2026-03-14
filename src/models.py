import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
from src.rolling import RollingRobustScaler, RollingBuffer
from src import config as cfg

# --- 1. Top-Level Interface ---
class BaseModel:
    def initialize(self, X_init: np.ndarray, y_init: np.ndarray) -> None: pass
    def predict(self, x_t: np.ndarray) -> float: pass
    def update(self, x_t: np.ndarray, y_t: float) -> None: pass
    def get_coefs(self) -> np.ndarray | None: return None

# --- 2. The Engine (Handles all Rolling/Scaling Logic) ---
class RollingRegressionModel(BaseModel):
    def __init__(self, model, train_win_periods, n_features, use_scaling=True,
                 refit_frequency=1, feature_transform=None):
        self.model = model
        self.train_win_periods = train_win_periods
        self.n_features = n_features
        self.use_scaling = use_scaling
        self.refit_frequency = refit_frequency
        self.feature_transform = feature_transform
        self.steps_since_refit = 0

        self.buffer = RollingBuffer(train_win_periods, n_features, 1)

        if self.use_scaling:
            self.scaler = RollingRobustScaler(train_win_periods, n_features)

        self.mean_x = np.zeros(n_features)
        self.std_x = np.ones(n_features)

        self.hist_X = []
        self.hist_y = []

    def _fit_model(self, X, y):
        """Fit feature transform (if any) and model on the buffer data."""
        if self.feature_transform is not None:
            self.feature_transform.fit(X, y)
            X = self.feature_transform.transform(X)
        self.model.fit(X, y)

    def _transform_input(self, X):
        """Apply feature transform (if any) to input data."""
        if self.feature_transform is not None:
            return self.feature_transform.transform(X)
        return X

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
        self._fit_model(X_tr, y_tr)

    def predict(self, x_t):
        if self.use_scaling:
            x_input = (x_t - self.mean_x) / self.std_x
        else:
            x_input = x_t

        x_input = self._transform_input(x_input.reshape(1, -1))
        return self.model.predict(x_input).item()

    def get_coefs(self):
        if hasattr(self.model, 'coef_'):
            return self.model.coef_.ravel()
        return None

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
            self._fit_model(X_tr, y_tr)
            self.steps_since_refit = 0


# --- 3. The Specific Algorithms ---

class RidgeModel(RollingRegressionModel):
    def __init__(self, train_win_periods, n_features, use_scaling=True,
                 feature_transform=None, refit_frequency=1, **ridge_kwargs):
        model = Ridge(**ridge_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency,
            feature_transform=feature_transform,
        )

class XGBoostModel(RollingRegressionModel):
    def __init__(self, train_win_periods, n_features, use_scaling=False, refit_frequency=5,
                 feature_transform=None, **xgb_kwargs):
        if 'tree_method' not in xgb_kwargs:
            xgb_kwargs['tree_method'] = 'hist'
        if 'n_jobs' not in xgb_kwargs:
            xgb_kwargs['n_jobs'] = -1

        model = XGBRegressor(**xgb_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency,
            feature_transform=feature_transform,
        )

class LightGBMModel(RollingRegressionModel):
    def __init__(self, train_win_periods, n_features, use_scaling=False, refit_frequency=5,
                 feature_transform=None, **lgbm_kwargs):
        if 'n_jobs' not in lgbm_kwargs:
            lgbm_kwargs['n_jobs'] = -1
        if 'verbose' not in lgbm_kwargs:
            lgbm_kwargs['verbose'] = -1

        model = LGBMRegressor(**lgbm_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency,
            feature_transform=feature_transform,
        )


class RandomForestModel(RollingRegressionModel):
    def __init__(self, train_win_periods, n_features, use_scaling=False, refit_frequency=5,
                 feature_transform=None, **rf_kwargs):
        if 'n_jobs' not in rf_kwargs:
            rf_kwargs['n_jobs'] = -1

        model = RandomForestRegressor(**rf_kwargs)
        super().__init__(
            model=model,
            train_win_periods=train_win_periods,
            n_features=n_features,
            use_scaling=use_scaling,
            refit_frequency=refit_frequency,
            feature_transform=feature_transform,
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
        except (np.linalg.LinAlgError, ValueError) as e:
            import warnings
            warnings.warn(f"SARIMAX fit failed, retaining previous fit: {e}")
        return self

    def predict(self, X):
        """Return np.array([scalar]) so .item() in RollingRegressionModel works."""
        if self._result is None:
            return np.array([0.0])
        exog = np.asarray(X, dtype=np.float64) if X.shape[1] > 0 else None
        try:
            fc = self._result.forecast(steps=1, exog=exog)
            return np.array([float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0])])
        except (ValueError, IndexError) as e:
            import warnings
            warnings.warn(f"SARIMAX predict failed, returning 0.0: {e}")
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


# --- 5. The Baseline ---

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


# --- 6. Model Registry & Factory ---

MODEL_REGISTRY = {
    'ridge': {
        'class': RidgeModel,
        'defaults': {'use_scaling': True, 'alpha': 1.0},
    },
    'xgboost': {
        'class': XGBoostModel,
        'defaults': {'use_scaling': False, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'tree_method': 'hist'},
    },
    'lightgbm': {
        'class': LightGBMModel,
        'defaults': {'use_scaling': False, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
    },
    'random_forest': {
        'class': RandomForestModel,
        'defaults': {'use_scaling': False, 'n_estimators': 100, 'max_depth': 3},
    },
    'sarimax': {
        'class': SARIMAXModel,
        'defaults': {
            'order': cfg.SARIMAX_ORDER,
            'seasonal_order': cfg.SARIMAX_SEASONAL_ORDER,
            'fit_window': cfg.SARIMAX_FIT_WINDOW,
            'refit_frequency': cfg.SARIMAX_REFIT_FREQUENCY,
        },
    },
}


def create_model(model_name, train_win_periods, n_features,
                 feature_transform=None, refit_frequency=1, naive_lag_index=None, **overrides):
    """
    Factory function that creates a model instance from the registry.

    Parameters
    ----------
    model_name : str
        Key in MODEL_REGISTRY, or 'naive' for NaiveBaseline.
    naive_lag_index : int or None
        Required when model_name == 'naive'.
    **overrides
        Override any default hyperparameter from the registry.
    """
    if model_name == 'naive':
        return NaiveBaseline(lag_index=naive_lag_index)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_name}")

    entry = MODEL_REGISTRY[model_name]
    kwargs = {**entry['defaults'], **overrides}

    # SARIMAX uses its own refit_frequency and doesn't take feature_transform
    if model_name == 'sarimax':
        return entry['class'](
            train_win_periods=train_win_periods,
            n_features=n_features,
            **kwargs,
        )

    return entry['class'](
        train_win_periods=train_win_periods,
        n_features=n_features,
        feature_transform=feature_transform,
        refit_frequency=refit_frequency,
        **kwargs,
    )