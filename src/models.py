import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
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

# --- 4. The Baseline ---

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