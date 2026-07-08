"""Final column selection: drops pre-engineered/leakage-prone columns."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin

from box_office.ml.feature_pipeline.constants import (
    RAW_INPUT_COLUMNS_TO_DROP,
    SELECTED_FEATURES,
)


class _DropPreEngineered(BaseEstimator, TransformerMixin):
    """Strip any pre-existing engineered columns from input.

    Input snapshots can include offline-encoded columns (``mpaa_encoded``,
    ``release_type_encoded``, ``production_company_encoded``) that collide with
    the columns emitted by the pipeline. Dropping them here keeps the final
    feature contract unique and deterministic.
    """

    PRE_ENCODED_COLUMNS: tuple[str, ...] = (
        "MPAA_ENCODED",
        "RELEASE_TYPE_ENCODED",
        "PRODUCTION_COMPANY_ENCODED",
    )

    def fit(self, X: pd.DataFrame, y=None) -> _DropPreEngineered:
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.PRE_ENCODED_COLUMNS if c in X.columns]
        return X.drop(columns=present) if present else X


class _SelectEngineered(BaseEstimator, TransformerMixin):
    """Final step: drop raw input columns, then refuse non-numeric leftovers."""

    def __init__(self, raw_cols=RAW_INPUT_COLUMNS_TO_DROP) -> None:
        self.raw_cols = tuple(raw_cols)

    def fit(self, X: pd.DataFrame, y=None) -> _SelectEngineered:
        self.is_fitted_ = True
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


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Project onto the canonical ``SELECTED_FEATURES`` contract (names + order).

    The upstream transformers compute every engineered feature; this final step
    keeps only the curated, decorrelated subset and fixes its order, so the
    feature contract lives in exactly one place (``SELECTED_FEATURES``) rather
    than being smeared across the transformers. Computing-then-projecting also
    keeps intermediate dependencies intact (e.g. ``LOG_BUDGET_X_HORROR`` needs
    ``GENRE_horror``, which is not itself selected).
    """

    def __init__(self, selected=SELECTED_FEATURES) -> None:
        self.selected = tuple(selected)

    def fit(self, X: pd.DataFrame, y=None) -> FeatureSelector:
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Degenerate input: some transformers emit no rows-derived columns on an
        # empty frame. There are no values to be wrong, so project to the
        # contract shape rather than failing the missing-column guard.
        if len(X) == 0:
            return X.reindex(columns=list(self.selected))
        missing = [c for c in self.selected if c not in X.columns]
        if missing:
            raise ValueError(
                f"FeatureSelector: pipeline did not produce {missing}. Either a "
                "transformer that emits them was removed, or SELECTED_FEATURES is "
                "out of sync with the transformers."
            )
        return X.loc[:, list(self.selected)]
