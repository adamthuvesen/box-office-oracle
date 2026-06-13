"""Regression metrics for log-transformed targets."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


def rmse_on_log_scale(y_true, y_pred):
    """RMSE on already-log-transformed inputs.

    Callers MUST pass ``log1p``-transformed targets and predictions.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def spearman_rank_corr(y_true, y_pred) -> float:
    """Spearman rank correlation between truth and prediction.

    Rank-based, so monotonic transforms (e.g. ``expm1``) don't change it —
    measures how well the model *orders* films, the lens that survives the
    heavy-tailed revenue distribution where dollar-space R² is dominated by a
    handful of blockbusters. Returns NaN when undefined (n < 2 or constant
    input), so a degenerate fold never raises.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Undefined for fewer than two points or a constant input (zero spread);
    # short-circuit so scipy doesn't warn and we return a clean NaN.
    if y_true.size < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)
