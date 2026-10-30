import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from box_office.config import MODEL_HYPERPARAMETERS as _HP
from box_office.config import ModelHyperparameters

logger = logging.getLogger(__name__)


class BoxOfficeXGBoostModel:
    """XGBoost model for box office prediction with time series cross-validation."""

    def __init__(
        self,
        n_estimators: int = _HP.n_estimators,
        learning_rate: float = _HP.learning_rate,
        max_depth: int = _HP.max_depth,
        min_child_weight: int = _HP.min_child_weight,
        subsample: float = _HP.subsample,
        colsample_bytree: float = _HP.colsample_bytree,
        reg_alpha: float = _HP.reg_alpha,
        reg_lambda: float = _HP.reg_lambda,
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
    ) -> BoxOfficeXGBoostModel:
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


def fit_final_model(
    X_scaled: pd.DataFrame,
    y_log: pd.Series,
    mean_best_iteration: float,
    hyperparameters: ModelHyperparameters = _HP,
) -> BoxOfficeXGBoostModel:
    """Fit the deployment model on all rows at the CV-mean best iteration.

    Early stopping needs a held-out set, so the final fit has none; the CV-mean
    best iteration stands in for it. The local and the SageMaker paths both
    call this function.
    """
    model = BoxOfficeXGBoostModel(
        **hyperparameters.model_dump(exclude={"n_estimators", "early_stopping_rounds"}),
        n_estimators=int(mean_best_iteration),
        random_state=42,
    )
    model.fit(X_scaled, y_log)
    return model
