import logging

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


class BoxOfficeXGBoostModel:
    """XGBoost model for box office prediction with time series cross-validation."""

    def __init__(
        self,
        n_estimators: int = 1500,
        learning_rate: float = 0.04,
        max_depth: int = 4,
        min_child_weight: int = 2,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.01,
        reg_lambda: float = 0.2,
        early_stopping_rounds: int | None = None,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            # Targets are log-transformed, so squared error on the log scale is
            # equivalent to RMSLE on the original (dollar) scale.
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_jobs": -1,
            "verbosity": 0,
        }

        if early_stopping_rounds is not None:
            self.params["early_stopping_rounds"] = early_stopping_rounds

        self.model = None
        self.is_fitted = False

    def create_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(**self.params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: list | None = None,
        verbose: bool = False,
    ) -> "BoxOfficeXGBoostModel":
        self.model = self.create_model()

        fit_params = {"verbose": verbose}
        if eval_set is not None:
            fit_params["eval_set"] = eval_set

        self.model.fit(X_train, y_train, **fit_params)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted to access feature importances")
        return self.model.feature_importances_

    @property
    def best_iteration(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted to access best iteration")
        return getattr(self.model, "best_iteration", self.params["n_estimators"])
