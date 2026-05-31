"""Core numerical feature transformer."""

from __future__ import annotations

import logging
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from box_office.ml.feature_pipeline.constants import CORE_NUMERICAL_FEATURES
from box_office.utils.feature_flags import (
    strict_features_enabled as _strict_features_enabled,
)

logger = logging.getLogger(__name__)


class CoreNumericalTransformer(BaseEstimator, TransformerMixin):
    """Pass-through type-coercion for the 7 core numerical columns.

    Strict mode (``ML_STRICT_FEATURES=true``) raises ``KeyError`` on any
    missing core column instead of silently filling with zero — keeps an
    upstream typo from masquerading as a real prediction.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "CoreNumericalTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in CORE_NUMERICAL_FEATURES if c not in X.columns]
        if missing:
            if _strict_features_enabled():
                raise KeyError(
                    f"ML_STRICT_FEATURES=true and core columns missing from input: {missing}"
                )
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
