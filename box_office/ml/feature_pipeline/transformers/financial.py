"""Financial feature transformers (budget and inflation-adjusted dollars)."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from box_office.ml.feature_pipeline._helpers import _column
from box_office.ml.feature_pipeline.cpi import CPI_ANCHOR_YEAR, CPI_BY_YEAR


class FinancialTransformer(BaseEstimator, TransformerMixin):
    """Budget-derived features: totals, ratios, inflation adjustment."""

    def fit(self, X: pd.DataFrame, y=None) -> "FinancialTransformer":
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new = pd.DataFrame(index=X.index)
        prod = _column(X, "PRODUCTION_BUDGET")
        ad = _column(X, "AD_BUDGET")
        year = _column(X, "RELEASE_YEAR")
        total_budget = prod + ad
        log_total_budget = np.log1p(total_budget.clip(lower=0))

        new["TOTAL_BUDGET"] = total_budget
        new["AD_TO_PROD_RATIO"] = ad / (prod + 1)
        new["YEAR_TO_BUDGET_RATIO"] = year / (total_budget + 1000)
        new["LOG_PRODUCTION_BUDGET"] = np.log1p(prod.clip(lower=0))

        anchor_cpi = CPI_BY_YEAR[CPI_ANCHOR_YEAR]
        years_int = year.fillna(CPI_ANCHOR_YEAR).astype(int)
        row_cpi = years_int.map(CPI_BY_YEAR).fillna(anchor_cpi)
        new["BUDGET_INFLATION_ADJUSTED"] = total_budget * (anchor_cpi / row_cpi)

        director_freq = _optional_column(X, "DIRECTOR_FREQ")
        company_freq = _optional_column(X, "COMPANY_FREQ")
        lead_actor_freq = _optional_column(X, "LEAD_ACTOR_FREQ")
        max_actor_freq = _optional_column(X, "MAX_ACTOR_FREQ")

        new["DIRECTOR_BUDGET_CONFIDENCE"] = director_freq * log_total_budget
        new["CREATIVE_FREQ_SCORE"] = (
            np.log1p(director_freq.clip(lower=0))
            + np.log1p(company_freq.clip(lower=0))
            + np.log1p(max_actor_freq.clip(lower=0))
        )
        new["LOG1P_LEAD_ACTOR_FREQ"] = np.log1p(lead_actor_freq.clip(lower=0))

        new["LOG_TOTAL_BUDGET_X_HORROR"] = log_total_budget * _optional_column(
            X, "GENRE_horror"
        )
        new["LOG_TOTAL_BUDGET_X_ADVENTURE"] = log_total_budget * _optional_column(
            X, "GENRE_adventure"
        )
        new["LOG_TOTAL_BUDGET_X_COMEDY"] = log_total_budget * _optional_column(
            X, "GENRE_comedy"
        )
        new["LOG_TOTAL_BUDGET_X_SUMMER"] = log_total_budget * _optional_column(
            X, "IS_SUMMER_RELEASE"
        )
        new["LOG_TOTAL_BUDGET_X_COMPANY_FREQ"] = log_total_budget * np.log1p(
            company_freq.clip(lower=0)
        )
        holiday_score = (
            _optional_column(X, "IS_BLOCKBUSTER_SEASON")
            + 2 * _optional_column(X, "IS_MEMORIAL_DAY_WEEKEND")
            + 2 * _optional_column(X, "IS_JULY_4TH_WEEKEND")
            + 1.5 * _optional_column(X, "IS_THANKSGIVING_WEEK")
            + 1.5 * _optional_column(X, "IS_CHRISTMAS_WEEK")
        )
        new["BLOCKBUSTER_BUDGET_MULTIPLIER"] = total_budget * holiday_score

        if "AVG_ACTOR_FREQ" in X.columns:
            new["BUDGET_PER_ACTOR_FREQ"] = total_budget / (
                _column(X, "AVG_ACTOR_FREQ") + 1
            )
        else:
            new["BUDGET_PER_ACTOR_FREQ"] = total_budget

        return pd.concat([X, new], axis=1)


def _optional_column(X: pd.DataFrame, col: str) -> pd.Series:
    if col not in X.columns:
        return pd.Series(0.0, index=X.index)
    return _column(X, col)
