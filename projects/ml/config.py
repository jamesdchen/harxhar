"""ML-specific configuration constants."""

from core.core.config import PERIODS_PER_DAY

# SARIMAX defaults
SARIMAX_ORDER = (2, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 0, 0, PERIODS_PER_DAY)
SARIMAX_FIT_WINDOW = 480  # 10 trading days
SARIMAX_REFIT_FREQUENCY = 48  # once per simulated day
SARIMAX_FIT_METHOD = "lbfgs"
SARIMAX_FIT_MAXITER = 100
