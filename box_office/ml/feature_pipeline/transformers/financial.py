"""Financial feature transformers (budget and inflation-adjusted dollars)."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from box_office.ml.feature_pipeline._helpers import _column
from box_office.ml.feature_pipeline.cpi import CPI_ANCHOR_YEAR, CPI_BY_YEAR


class FinancialTransformer(BaseEstimator, TransformerMixin):
    """Budget-derived features: totals, ratios, inflation adjustment."""

    def fit(self, X: pd.DataFrame, y=None) -> "FinancialTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new = pd.DataFrame(index=X.index)
        prod = _column(X, "PRODUCTION_BUDGET")
        ad = _column(X, "AD_BUDGET")
        votes = _column(X, "VOTES")
        rating = _column(X, "RATING")
        year = _column(X, "RELEASE_YEAR")

        new["TOTAL_BUDGET"] = prod + ad
        new["BUDGET_TO_VOTES_RATIO"] = new["TOTAL_BUDGET"] / (votes + 1)
        new["AD_TO_PROD_RATIO"] = ad / (prod + 1)
        new["VOTES_PER_BUDGET"] = votes / (new["TOTAL_BUDGET"] + 1000)
        new["RATING_PER_BUDGET"] = rating / (new["TOTAL_BUDGET"] + 1000)
        new["RATING_VOTES_INTERACTION"] = rating * votes
        new["YEAR_TO_BUDGET_RATIO"] = year / (new["TOTAL_BUDGET"] + 1000)
        new["YEAR_TO_VOTES_RATIO"] = year / (votes + 1000)

        anchor_cpi = CPI_BY_YEAR[CPI_ANCHOR_YEAR]
        years_int = year.fillna(CPI_ANCHOR_YEAR).astype(int)
        row_cpi = years_int.map(CPI_BY_YEAR).fillna(anchor_cpi)
        new["BUDGET_INFLATION_ADJUSTED"] = new["TOTAL_BUDGET"] * (anchor_cpi / row_cpi)

        # VOTES_ERA_ADJUSTED is amortization, not inflation; the divisor is intentional.
        if "YEARS_SINCE_2000" in X.columns:
            new["VOTES_ERA_ADJUSTED"] = votes / (_column(X, "YEARS_SINCE_2000") + 1)
        else:
            new["VOTES_ERA_ADJUSTED"] = votes

        if "AVG_ACTOR_FREQ" in X.columns:
            new["BUDGET_PER_ACTOR_FREQ"] = new["TOTAL_BUDGET"] / (
                _column(X, "AVG_ACTOR_FREQ") + 1
            )
        else:
            new["BUDGET_PER_ACTOR_FREQ"] = new["TOTAL_BUDGET"]

        return pd.concat([X, new], axis=1)
