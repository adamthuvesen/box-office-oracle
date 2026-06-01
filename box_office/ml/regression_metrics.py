"""Regression metrics for log-transformed targets."""

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse_on_log_scale(y_true, y_pred):
    """RMSE on already-log-transformed inputs.

    Callers MUST pass ``log1p``-transformed targets and predictions.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
