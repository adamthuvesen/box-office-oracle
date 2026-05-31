"""Final column selection: drops pre-engineered/leakage-prone columns."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin

from box_office.ml.feature_pipeline.constants import RAW_INPUT_COLUMNS_TO_DROP


class _DropPreEngineered(BaseEstimator, TransformerMixin):
    """Strip any pre-existing engineered columns from input.

    The legacy raw schema carries offline-encoded columns (``mpaa_encoded``,
    ``release_type_encoded``, ``production_company_encoded``) that collide
    with what the downstream transformers produce. Concatenating the
    transformer's output back into ``X`` would yield duplicates and trip
    ``FeaturePreprocessorHigh``'s collision guard.
    """

    PRE_ENCODED_COLUMNS: tuple[str, ...] = (
        "MPAA_ENCODED",
        "RELEASE_TYPE_ENCODED",
        "PRODUCTION_COMPANY_ENCODED",
    )

    def fit(self, X: pd.DataFrame, y=None) -> "_DropPreEngineered":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.PRE_ENCODED_COLUMNS if c in X.columns]
        return X.drop(columns=present) if present else X


class _SelectEngineered(BaseEstimator, TransformerMixin):
    """Final step: drop raw input columns, then refuse non-numeric leftovers."""

    def __init__(self, raw_cols=RAW_INPUT_COLUMNS_TO_DROP) -> None:
        self.raw_cols = tuple(raw_cols)

    def fit(self, X: pd.DataFrame, y=None) -> "_SelectEngineered":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=[c for c in self.raw_cols if c in X.columns])
        # Index by position rather than label — duplicate column names would
        # make ``out[col]`` return a DataFrame and break the dtype check.
        non_numeric = [
            str(out.columns[i])
            for i in range(out.shape[1])
            if not is_numeric_dtype(out.dtypes.iloc[i])
        ]
        if non_numeric:
            raise ValueError(
                "Feature pipeline output contains non-numeric columns: "
                f"{non_numeric}. Add them to RAW_INPUT_COLUMNS_TO_DROP or "
                "have a transformer consume them; otherwise the downstream "
                "scaler will raise on the first non-coercible value."
            )
        return out
