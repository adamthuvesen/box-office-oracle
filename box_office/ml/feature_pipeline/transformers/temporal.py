"""Temporal feature transformers derived from the release date."""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict

import pandas as pd
from pandas.tseries.holiday import USMemorialDay, USThanksgivingDay
from sklearn.base import BaseEstimator, TransformerMixin

from box_office.config import config

logger = logging.getLogger(__name__)


class TemporalTransformer(BaseEstimator, TransformerMixin):
    """Time-based features: seasonality, market timing, era indicators."""

    def fit(self, X: pd.DataFrame, y=None) -> "TemporalTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Vectorise the parse once; the row helper now reads pre-parsed
        # Timestamps and avoids per-row `pd.to_datetime` overhead.
        parsed = pd.to_datetime(X["RELEASE_DATE"], errors="coerce")
        feats = parsed.apply(self._extract_date_features)
        new_cols = pd.DataFrame(feats.tolist(), index=X.index)
        return pd.concat([X, new_cols], axis=1)

    def _extract_date_features(self, date_obj) -> Dict[str, Any]:
        if pd.isna(date_obj):
            return _DEFAULT_DATE_FEATURES.copy()

        month, day, weekday = date_obj.month, date_obj.day, date_obj.weekday()
        is_memorial = 1 if _is_in_holiday_weekend(date_obj, USMemorialDay) else 0
        is_thanks = 1 if _is_in_holiday_weekend(date_obj, USThanksgivingDay) else 0
        is_covid = (
            1
            if config.feature_engineering.enable_covid_features
            and date_obj >= pd.to_datetime("2020-03-01")
            else 0
        )

        return {
            "RELEASE_MONTH": month,
            "RELEASE_WEEK": date_obj.isocalendar().week,
            "RELEASE_QUARTER": (month - 1) // 3 + 1,
            "IS_SUMMER_RELEASE": 1 if month in (5, 6, 7, 8) else 0,
            "IS_HOLIDAY_RELEASE": (
                1 if month in (11, 12) or (month == 1 and day <= 15) else 0
            ),
            "IS_BLOCKBUSTER_SEASON": 1 if month in (5, 6, 7, 11, 12) else 0,
            "IS_WEEKEND_RELEASE": 1 if weekday >= 4 else 0,
            "IS_MEMORIAL_DAY_WEEKEND": is_memorial,
            "IS_JULY_4TH_WEEKEND": 1 if month == 7 and 1 <= day <= 7 else 0,
            "IS_THANKSGIVING_WEEK": is_thanks,
            "IS_CHRISTMAS_WEEK": 1 if month == 12 and 18 <= day <= 31 else 0,
            "IS_FIRST_WEEK_OF_MONTH": 1 if day <= 7 else 0,
            "IS_MID_MONTH": 1 if 10 <= day <= 20 else 0,
            "DAYS_FROM_MONTH_START": day,
            "YEARS_SINCE_2000": max(0, date_obj.year - 2000),
            "IS_PRE_STREAMING_ERA": 1 if date_obj.year < 2010 else 0,
            "IS_STREAMING_MATURE_ERA": 1 if date_obj.year >= 2015 else 0,
            "IS_COVID_ERA": is_covid,
        }


_DEFAULT_DATE_FEATURES: Dict[str, int] = {
    "RELEASE_MONTH": 0,
    "RELEASE_WEEK": 0,
    "RELEASE_QUARTER": 0,
    "IS_SUMMER_RELEASE": 0,
    "IS_HOLIDAY_RELEASE": 0,
    "IS_BLOCKBUSTER_SEASON": 0,
    "IS_WEEKEND_RELEASE": 0,
    "IS_MEMORIAL_DAY_WEEKEND": 0,
    "IS_JULY_4TH_WEEKEND": 0,
    "IS_THANKSGIVING_WEEK": 0,
    "IS_CHRISTMAS_WEEK": 0,
    "IS_FIRST_WEEK_OF_MONTH": 0,
    "IS_MID_MONTH": 0,
    "DAYS_FROM_MONTH_START": 0,
    "YEARS_SINCE_2000": 0,
    "IS_PRE_STREAMING_ERA": 0,
    "IS_STREAMING_MATURE_ERA": 0,
    "IS_COVID_ERA": 0,
}


@functools.lru_cache(maxsize=256)
def _holiday_for_year(holiday_rule, year: int):
    """Return the first holiday date in ``year`` per ``holiday_rule``, or None.

    Cached because every row in a training batch shares the same calendar; the
    rule dates lookup is otherwise re-run per row.
    """
    holiday_dates = holiday_rule.dates(f"{year}-01-01", f"{year}-12-31")
    if len(holiday_dates) == 0:
        return None
    return pd.Timestamp(holiday_dates[0])


def _is_in_holiday_weekend(date_obj: pd.Timestamp, holiday_rule) -> bool:
    """True if ``date_obj`` lies in the Fri-Sun window around the holiday."""
    holiday = _holiday_for_year(holiday_rule, date_obj.year)
    if holiday is None:
        return False
    delta = (pd.Timestamp(date_obj.year, date_obj.month, date_obj.day) - holiday).days
    if holiday.weekday() == 0:
        return -3 <= delta <= 0
    if holiday.weekday() == 3:
        return -1 <= delta <= 3
    return -3 <= delta <= 3
