"""Core numerical feature transformer."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from box_office.ml.feature_pipeline.constants import CORE_NUMERICAL_FEATURES

logger = logging.getLogger(__name__)


class CoreNumericalTransformer(BaseEstimator, TransformerMixin):
    """Pass-through type-coercion for the core numerical columns.

    Missing core columns are logged and filled with zero for the inference
    defaults defined by the feature contract.
    """

    def fit(self, X: pd.DataFrame, y=None) -> CoreNumericalTransformer:
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in CORE_NUMERICAL_FEATURES if c not in X.columns]
        if missing:
            for col in missing:
                logger.warning(
                    "Core feature column %r missing from input; filling with default 0.",
                    col,
                )

        out = X.copy()
        for col in CORE_NUMERICAL_FEATURES:
            if col in X.columns:
                if X[col].dtype == "object":
                    out[col] = pd.to_numeric(X[col], errors="coerce")
                else:
                    out[col] = X[col]
            else:
                out[col] = 0
        return out
