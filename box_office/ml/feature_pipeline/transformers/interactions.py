"""Cross-feature interaction transformers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from box_office.ml.feature_pipeline._helpers import _column
from box_office.ml.feature_pipeline.constants import (
    _INTERACTION_FILL_ZERO,
    _INTERACTION_KEEP_NAN,
)

logger = logging.getLogger(__name__)


class InteractionTransformer(BaseEstimator, TransformerMixin):
    """Cross-feature interactions: temporal × financial, genre × budget, etc."""

    def fit(self, X: pd.DataFrame, y=None) -> "InteractionTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new = pd.DataFrame(index=X.index)

        if "IS_BLOCKBUSTER_SEASON" in X.columns and "TOTAL_BUDGET" in X.columns:
            new["BUDGET_SEASONAL_BOOST"] = _column(X, "TOTAL_BUDGET") * _column(
                X, "IS_BLOCKBUSTER_SEASON"
            )
        if "IS_SUMMER_RELEASE" in X.columns and "TOTAL_BUDGET" in X.columns:
            new["SUMMER_BUDGET_INTERACTION"] = _column(X, "TOTAL_BUDGET") * _column(
                X, "IS_SUMMER_RELEASE"
            )
        if "IS_HOLIDAY_RELEASE" in X.columns and "TOTAL_BUDGET" in X.columns:
            new["HOLIDAY_BUDGET_INTERACTION"] = _column(X, "TOTAL_BUDGET") * _column(
                X, "IS_HOLIDAY_RELEASE"
            )
        if "IS_WEEKEND_RELEASE" in X.columns and "RATING" in X.columns:
            new["WEEKEND_RATING_BOOST"] = _column(X, "RATING") * _column(
                X, "IS_WEEKEND_RELEASE"
            )

        if "IS_COVID_ERA" in X.columns:
            covid = _column(X, "IS_COVID_ERA")
            if "TOTAL_BUDGET" in X.columns:
                new["COVID_BUDGET_IMPACT"] = _column(X, "TOTAL_BUDGET") * covid
            if "RATING" in X.columns:
                new["COVID_RATING_IMPACT"] = _column(X, "RATING") * covid
            if "VOTES" in X.columns:
                new["COVID_VOTES_IMPACT"] = _column(X, "VOTES") * covid

        if "FRANCHISE_RATING" in X.columns:
            franchise = _column(X, "FRANCHISE_RATING")
            if "VOTES" in X.columns:
                new["FRANCHISE_STRENGTH"] = franchise * _column(X, "VOTES")
            if "TOTAL_BUDGET" in X.columns:
                new["FRANCHISE_BUDGET_CONFIDENCE"] = franchise * _column(
                    X, "TOTAL_BUDGET"
                )

        needed = (
            "TOTAL_BUDGET",
            "IS_BLOCKBUSTER_SEASON",
            "IS_MEMORIAL_DAY_WEEKEND",
            "IS_JULY_4TH_WEEKEND",
            "IS_THANKSGIVING_WEEK",
            "IS_CHRISTMAS_WEEK",
        )
        if all(c in X.columns for c in needed):
            tot = _column(X, "TOTAL_BUDGET")
            new["BLOCKBUSTER_BUDGET_MULTIPLIER"] = tot * (
                _column(X, "IS_BLOCKBUSTER_SEASON")
                + _column(X, "IS_MEMORIAL_DAY_WEEKEND") * 2
                + _column(X, "IS_JULY_4TH_WEEKEND") * 2
                + _column(X, "IS_THANKSGIVING_WEEK") * 1.5
                + _column(X, "IS_CHRISTMAS_WEEK") * 1.5
            )

        if "TOTAL_BUDGET" in X.columns:
            tot = _column(X, "TOTAL_BUDGET")
            ga = _column(X, "GENRE_action") if "GENRE_action" in X.columns else 0
            gc = _column(X, "GENRE_comedy") if "GENRE_comedy" in X.columns else 0
            gh = _column(X, "GENRE_horror") if "GENRE_horror" in X.columns else 0
            new["ACTION_BUDGET_INTERACTION"] = ga * tot
            new["COMEDY_BUDGET_EFFICIENCY"] = (
                gc * _column(X, "AD_TO_PROD_RATIO")
                if "AD_TO_PROD_RATIO" in X.columns
                else 0
            )
            new["HORROR_LOW_BUDGET_ADVANTAGE"] = (
                gh / (_column(X, "PRODUCTION_BUDGET") + 1000)
                if "PRODUCTION_BUDGET" in X.columns
                else 0
            )

        if "DIRECTOR_FREQ" in X.columns and "TOTAL_BUDGET" in X.columns:
            new["DIRECTOR_BUDGET_CONFIDENCE"] = _column(X, "DIRECTOR_FREQ") * np.log1p(
                _column(X, "TOTAL_BUDGET")
            )
        if "MAX_ACTOR_FREQ" in X.columns and "AD_TO_PROD_RATIO" in X.columns:
            new["STAR_POWER_PREMIUM"] = _column(X, "MAX_ACTOR_FREQ") * _column(
                X, "AD_TO_PROD_RATIO"
            )

        for col in list(new.columns):
            nan_count = int(new[col].isna().sum())
            if nan_count == 0:
                continue
            if col in _INTERACTION_FILL_ZERO:
                new[col] = new[col].fillna(0)
                logger.warning(
                    "Interaction column %r had %d NaN rows; filled with 0 per policy.",
                    col,
                    nan_count,
                )
            elif col in _INTERACTION_KEEP_NAN:
                logger.warning(
                    "Interaction ratio %r has %d NaN rows; propagating NaN per policy.",
                    col,
                    nan_count,
                )
            else:
                logger.warning(
                    "Interaction column %r has %d NaN rows but no documented "
                    "fill policy; propagating NaN.",
                    col,
                    nan_count,
                )

        return pd.concat([X, new], axis=1)
