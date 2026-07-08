"""Financial feature transformers (budget and inflation-adjusted dollars)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from box_office.ml.feature_pipeline._helpers import _column
from box_office.ml.feature_pipeline.cpi import CPI_ANCHOR_YEAR, CPI_BY_YEAR


class FinancialTransformer(BaseEstimator, TransformerMixin):
    """Budget-derived features: ratios, interactions, inflation adjustment."""

    def fit(self, X: pd.DataFrame, y=None) -> FinancialTransformer:
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new = pd.DataFrame(index=X.index)
        prod = _column(X, "PRODUCTION_BUDGET")
        year = _column(X, "RELEASE_YEAR")
        log_budget = np.log1p(prod.clip(lower=0))

        new["YEAR_TO_BUDGET_RATIO"] = year / (prod + 1000)
        new["LOG_PRODUCTION_BUDGET"] = log_budget

        anchor_cpi = CPI_BY_YEAR[CPI_ANCHOR_YEAR]
        years_int = year.fillna(CPI_ANCHOR_YEAR).astype(int)
        row_cpi = years_int.map(CPI_BY_YEAR).fillna(anchor_cpi)
        new["BUDGET_INFLATION_ADJUSTED"] = prod * (anchor_cpi / row_cpi)

        director_freq = _optional_column(X, "DIRECTOR_FREQ")
        company_freq = _optional_column(X, "COMPANY_FREQ")
        lead_actor_freq = _optional_column(X, "LEAD_ACTOR_FREQ")
        max_actor_freq = _optional_column(X, "MAX_ACTOR_FREQ")

        new["DIRECTOR_BUDGET_CONFIDENCE"] = director_freq * log_budget
        new["CREATIVE_FREQ_SCORE"] = (
            np.log1p(director_freq.clip(lower=0))
            + np.log1p(company_freq.clip(lower=0))
            + np.log1p(max_actor_freq.clip(lower=0))
        )
        new["LOG1P_LEAD_ACTOR_FREQ"] = np.log1p(lead_actor_freq.clip(lower=0))

        new["LOG_BUDGET_X_HORROR"] = log_budget * _optional_column(X, "GENRE_horror")
        new["LOG_BUDGET_X_ADVENTURE"] = log_budget * _optional_column(
            X, "GENRE_adventure"
        )
        new["LOG_BUDGET_X_COMEDY"] = log_budget * _optional_column(X, "GENRE_comedy")
        new["LOG_BUDGET_X_SUMMER"] = log_budget * _optional_column(
            X, "IS_SUMMER_RELEASE"
        )
        new["LOG_BUDGET_X_COMPANY_FREQ"] = log_budget * np.log1p(
            company_freq.clip(lower=0)
        )
        holiday_score = (
            _optional_column(X, "IS_BLOCKBUSTER_SEASON")
            + 2 * _optional_column(X, "IS_MEMORIAL_DAY_WEEKEND")
            + 2 * _optional_column(X, "IS_JULY_4TH_WEEKEND")
            + 1.5 * _optional_column(X, "IS_THANKSGIVING_WEEK")
            + 1.5 * _optional_column(X, "IS_CHRISTMAS_WEEK")
        )
        new["BLOCKBUSTER_BUDGET_MULTIPLIER"] = prod * holiday_score

        if "AVG_ACTOR_FREQ" in X.columns:
            new["BUDGET_PER_ACTOR_FREQ"] = prod / (_column(X, "AVG_ACTOR_FREQ") + 1)
        else:
            new["BUDGET_PER_ACTOR_FREQ"] = prod

        return pd.concat([X, new], axis=1)


def _optional_column(X: pd.DataFrame, col: str) -> pd.Series:
    if col not in X.columns:
        return pd.Series(0.0, index=X.index)
    return _column(X, col)
