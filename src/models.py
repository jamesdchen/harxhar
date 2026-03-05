import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from src.rolling import RollingStandardScaler, RollingBuffer # Your custom classes

class BaseModel:
    def initialize(self, X_init, y_init): pass
    def predict(self, x_t): pass
    def update(self, x_t, y_t): pass

class XGBoostModel(BaseModel):
    def __init__(self, train_win_periods, n_features, use_scaling=False, refit_frequency=5, **xgb_kwargs):
        self.train_win_periods = train_win_periods
        self.n_features = n_features
        self.use_scaling = use_scaling
        self.refit_frequency = refit_frequency # <-- NEW: How often to retrain
        self.steps_since_refit = 0             # <-- NEW: Counter
        
        # Optimize XGBoost for speed
        if 'tree_method' not in xgb_kwargs:
            xgb_kwargs['tree_method'] = 'hist' # Much faster than exact greedy
        if 'n_jobs' not in xgb_kwargs:
            xgb_kwargs['n_jobs'] = -1          # Use all cores
        
        # Initialize XGBoost with any passed kwargs (e.g., n_estimators, max_depth)
        self.model = XGBRegressor(**xgb_kwargs)
        self.buffer = RollingBuffer(train_win_periods, n_features, 1)
        
        if self.use_scaling:
            self.scaler = RollingStandardScaler(n_features)
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
        # Pop oldest
        x_old = self.hist_X.pop(0)
        _ = self.hist_y.pop(0)

        # Update Scaler
        if self.use_scaling:
            self.scaler.update(x_t, x_old)
            self.mean_x, self.std_x = self.scaler.get_scaler()
            x_new = (x_t - self.mean_x) / self.std_x
        else:
            x_new = x_t

        # Add new to buffer
        self.buffer.add(x_new, y_t)
        
        # Maintain raw history
        self.hist_X.append(x_t)
        self.hist_y.append(y_t)

        # Conditionally Refit
        self.steps_since_refit += 1
        if self.steps_since_refit >= self.refit_frequency:
            X_tr, y_tr = self.buffer.get_view()
            self.model.fit(X_tr, y_tr)
            self.steps_since_refit = 0 # Reset counter

class RidgeModel(BaseModel):
    def __init__(self, train_win_periods, n_features, use_scaling=True, **ridge_kwargs):
        self.train_win_periods = train_win_periods
        self.n_features = n_features
        self.use_scaling = use_scaling
        
        self.model = Ridge(**ridge_kwargs)
        self.buffer = RollingBuffer(train_win_periods, n_features, 1)
        self.scaler = RollingStandardScaler(n_features)
        
        self.hist_X = []
        self.hist_y = []
        self.mean_x = np.zeros(n_features)
        self.std_x = np.ones(n_features)

    def initialize(self, X_init, y_init):
        if y_init.ndim == 1:
            y_init = y_init.reshape(-1, 1)
            
        if self.use_scaling:
            self.scaler.initialize(X_init)
            self.mean_x, self.std_x = self.scaler.get_scaler()

        self.buffer.X_buffer[:] = (X_init - self.mean_x) / self.std_x
        self.buffer.y_buffer[:] = y_init
        
        self.hist_X = list(X_init)
        self.hist_y = list(y_init)
        
        X_tr, y_tr = self.buffer.get_view()
        self.model.fit(X_tr, y_tr)

    def predict(self, x_t):
        x_scl = (x_t - self.mean_x) / self.std_x
        return self.model.predict(x_scl.reshape(1, -1)).item()

    def update(self, x_t, y_t):
        # Pop oldest
        x_old = self.hist_X.pop(0)
        _ = self.hist_y.pop(0)

        # Update Scaler
        if self.use_scaling:
            self.scaler.update(x_t, x_old)
            self.mean_x, self.std_x = self.scaler.get_scaler()

        # Add new to buffer
        x_new_scl = (x_t - self.mean_x) / self.std_x
        self.buffer.add(x_new_scl, y_t)
        
        # Maintain raw history
        self.hist_X.append(x_t)
        self.hist_y.append(y_t)

        # Refit
        X_tr, y_tr = self.buffer.get_view()
        self.model.fit(X_tr, y_tr)


class NaiveBaseline(BaseModel):
    def __init__(self, lag_index=0):
        # 0 = 1-period MA, 1 = 5-period MA, etc.
        self.lag_index = lag_index

    def initialize(self, X_init, y_init):
        pass # No training required

    def predict(self, x_t):
        return x_t[self.lag_index]

    def update(self, x_t, y_t):
        pass # No refitting required