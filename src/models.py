import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
from src.rolling import RollingRobustScaler, RollingBuffer

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

class SARIMAXModel(BaseModel):
    """
    Pure SARIMAX baseline — no exogenous features.

    Fits SARIMAX on the raw y series only. The model's AR/MA and seasonal
    AR components capture autoregressive structure via individual raw lags,
    as opposed to the aggregated HAR rolling means used by Ridge. HAR
    features passed through the pipeline interface are intentionally ignored.

    Uses RollingBuffer from rolling.py for the internal y window.

    Parameters
    ----------
    train_win_periods : int
        Burn-in window required by the backtester framework.
    order : tuple
        ARIMA (p, d, q) non-seasonal order.
    seasonal_order : tuple
        (P, D, Q, s) seasonal order. s=48 for daily seasonality on
        30-minute bars.
    fit_window : int
        Number of most-recent observations used when (re)fitting.
        Defaults to 480 (10 trading days of 30-min bars).
    refit_frequency : int
        Steps between model refits. Defaults to 48 (once per day).
    """

    def __init__(
        self,
        train_win_periods,
        order=(2, 0, 1),
        seasonal_order=(1, 0, 0, 48),
        fit_window=480,
        refit_frequency=48,
    ):
        self.train_win_periods = train_win_periods
        self.order = order
        self.seasonal_order = seasonal_order
        self.fit_window = fit_window
        self.refit_frequency = refit_frequency
        self.steps_since_refit = 0
        self._buf_full = False

        # Reuse RollingBuffer for the y window; n_features=0 since no exog
        self.buffer = RollingBuffer(fit_window, n_features=0, n_targets=1)
        self.result = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_y_ordered(self):
        """Return y in chronological order from the RollingBuffer ring."""
        y_flat = self.buffer.y_buffer[:, 0]
        ptr = self.buffer.ptr
        if self._buf_full:
            return np.concatenate([y_flat[ptr:], y_flat[:ptr]]).astype(np.float64)
        return y_flat[:ptr].astype(np.float64)

    def _fit(self, y):
        """Fit (or refit) SARIMAX; silently retains the previous result on failure."""
        try:
            m = _SARIMAX(
                endog=y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.result = m.fit(disp=False, method="lbfgs", maxiter=100)
        except Exception:
            pass  # keep previous result on failure

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def initialize(self, X_init, y_init):
        # X_init (HAR features) intentionally ignored
        if y_init.ndim == 2:
            y_init = y_init.ravel()

        # Seed the buffer with the most recent fit_window observations
        n = len(y_init)
        start = max(0, n - self.fit_window)
        for val in y_init[start:]:
            self.buffer.add(
                np.empty(0, dtype=np.float32),
                np.array([val], dtype=np.float32),
            )
        self._buf_full = n >= self.fit_window

        self._fit(self._get_y_ordered())

    def predict(self, x_t):
        # x_t (HAR features) intentionally ignored
        if self.result is None:
            ptr = self.buffer.ptr
            return float(self.buffer.y_buffer[(ptr - 1) % self.fit_window, 0])
        try:
            fc = self.result.forecast(steps=1)
            return float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0])
        except Exception:
            ptr = self.buffer.ptr
            return float(self.buffer.y_buffer[(ptr - 1) % self.fit_window, 0])

    def update(self, x_t, y_t):
        # x_t (HAR features) intentionally ignored
        self.buffer.add(
            np.empty(0, dtype=np.float32),
            np.array([float(y_t)], dtype=np.float32),
        )
        if not self._buf_full and self.buffer.ptr == 0:
            self._buf_full = True

        self.steps_since_refit += 1
        if self.steps_since_refit >= self.refit_frequency:
            self._fit(self._get_y_ordered())
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