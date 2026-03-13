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
    SARIMAX baseline using statsmodels.

    Uses HAR lag features as exogenous regressors and a short rolling
    window for efficient periodic refitting. The seasonal AR component
    (default s=48) captures intraday periodicity in 30-min data.

    Parameters
    ----------
    train_win_periods : int
        Burn-in window required by the backtester framework (unused
        internally; the model fits on the last `fit_window` observations).
    n_features : int
        Number of exogenous features (HAR lags + any extra columns).
    order : tuple
        ARIMA (p, d, q) non-seasonal order.
    seasonal_order : tuple
        (P, D, Q, s) seasonal order.  Set s=48 for daily seasonality
        on 30-minute bars.
    fit_window : int
        Number of most-recent observations used when (re)fitting.
        Defaults to 480 (10 trading days of 30-min bars).
    refit_frequency : int
        How many steps between model refits.  Defaults to 48
        (once per simulated trading day).
    """

    def __init__(
        self,
        train_win_periods,
        n_features,
        order=(2, 0, 1),
        seasonal_order=(1, 0, 0, 48),
        fit_window=480,
        refit_frequency=48,
    ):
        self.train_win_periods = train_win_periods
        self.n_features = n_features
        self.order = order
        self.seasonal_order = seasonal_order
        self.fit_window = fit_window
        self.refit_frequency = refit_frequency
        self.steps_since_refit = 0

        # Ring buffers for endog and exog
        self.y_buf = np.zeros(fit_window)
        self.X_buf = np.zeros((fit_window, n_features))
        self.buf_ptr = 0
        self.buf_full = False

        self.result = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ordered_buffers(self):
        """Return (y, X) in chronological order from the ring buffer."""
        if self.buf_full:
            p = self.buf_ptr
            y = np.concatenate([self.y_buf[p:], self.y_buf[:p]])
            X = np.vstack([self.X_buf[p:], self.X_buf[:p]])
        else:
            y = self.y_buf[: self.buf_ptr]
            X = self.X_buf[: self.buf_ptr]
        return y, X

    def _fit(self, y, X):
        """Fit (or refit) the SARIMAX model; silently retains the previous
        result if fitting fails."""
        try:
            m = _SARIMAX(
                endog=y,
                exog=X,
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
        if y_init.ndim == 2:
            y_init = y_init.ravel()

        # Store only the most recent fit_window observations
        n = len(y_init)
        if n >= self.fit_window:
            self.y_buf[:] = y_init[-self.fit_window :]
            self.X_buf[:] = X_init[-self.fit_window :]
            self.buf_ptr = 0
            self.buf_full = True
        else:
            self.y_buf[:n] = y_init
            self.X_buf[:n] = X_init
            self.buf_ptr = n
            self.buf_full = False

        y, X = self._ordered_buffers()
        self._fit(y, X)

    def predict(self, x_t):
        if self.result is None:
            # Fallback: repeat the most recent observed value
            last = (self.buf_ptr - 1) % self.fit_window
            return float(self.y_buf[last])
        try:
            fc = self.result.forecast(steps=1, exog=x_t.reshape(1, -1))
            return float(fc.iloc[0] if hasattr(fc, "iloc") else fc[0])
        except Exception:
            last = (self.buf_ptr - 1) % self.fit_window
            return float(self.y_buf[last])

    def update(self, x_t, y_t):
        # Write new observation into the ring buffer
        self.y_buf[self.buf_ptr] = float(y_t)
        self.X_buf[self.buf_ptr] = x_t
        self.buf_ptr = (self.buf_ptr + 1) % self.fit_window
        if not self.buf_full and self.buf_ptr == 0:
            self.buf_full = True

        self.steps_since_refit += 1
        if self.steps_since_refit >= self.refit_frequency:
            y, X = self._ordered_buffers()
            self._fit(y, X)
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