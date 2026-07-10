"""Unit tests for the pure franchise-history features in
scripts/experiment_ip_features.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_ip_features import (  # noqa: E402
    compute_time_safe_franchise_features,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df


def test_synthetic_franchise_prior_history_only():
    frame = _frame(
        [
            # A three-film franchise, deliberately out of row order
            {
                "franchise_key": "Saga",
                "release_date": "2012-05-01",
                "worldwide_gross": 300.0,
            },
            {
                "franchise_key": "Saga",
                "release_date": "2010-06-15",
                "worldwide_gross": 100.0,
            },
            {
                "franchise_key": "Saga",
                "release_date": "2014-07-01",
                "worldwide_gross": 500.0,
            },
            # A standalone movie
            {
                "franchise_key": None,
                "release_date": "2011-01-01",
                "worldwide_gross": 999.0,
            },
        ]
    )
    out = compute_time_safe_franchise_features(frame)

    # First film of the franchise: zeros
    assert out.loc[1].tolist() == [0.0, 0.0, 0.0]
    # Second film sees only the first film's gross
    assert out.at[0, "PRIOR_FRANCHISE_GROSS_LOG"] == np.log1p(100.0)
    assert out.at[0, "PRIOR_FRANCHISE_FILM_COUNT"] == 1.0
    assert out.at[0, "IS_FRANCHISE_FOLLOWUP"] == 1.0
    # Third film sees the first two
    assert out.at[2, "PRIOR_FRANCHISE_GROSS_LOG"] == np.log1p(400.0)
    assert out.at[2, "PRIOR_FRANCHISE_FILM_COUNT"] == 2.0
    # Standalone movie: zeros
    assert out.loc[3].tolist() == [0.0, 0.0, 0.0]


def test_orders_by_date_not_year():
    # Two films in the same year: the March one must not see the November one.
    frame = _frame(
        [
            {
                "franchise_key": "Twins",
                "release_date": "2015-03-01",
                "worldwide_gross": 50.0,
            },
            {
                "franchise_key": "Twins",
                "release_date": "2015-11-01",
                "worldwide_gross": 80.0,
            },
        ]
    )
    out = compute_time_safe_franchise_features(frame)
    assert out.loc[0].tolist() == [0.0, 0.0, 0.0]
    assert out.at[1, "PRIOR_FRANCHISE_GROSS_LOG"] == np.log1p(50.0)
    assert out.at[1, "PRIOR_FRANCHISE_FILM_COUNT"] == 1.0


def test_same_day_releases_do_not_see_each_other():
    frame = _frame(
        [
            {
                "franchise_key": "Tie",
                "release_date": "2018-06-01",
                "worldwide_gross": 10.0,
            },
            {
                "franchise_key": "Tie",
                "release_date": "2018-06-01",
                "worldwide_gross": 20.0,
            },
        ]
    )
    out = compute_time_safe_franchise_features(frame)
    assert out["IS_FRANCHISE_FOLLOWUP"].tolist() == [0.0, 0.0]


def test_franchises_are_independent():
    frame = _frame(
        [
            {
                "franchise_key": "A",
                "release_date": "2010-01-01",
                "worldwide_gross": 100.0,
            },
            {
                "franchise_key": "B",
                "release_date": "2012-01-01",
                "worldwide_gross": 200.0,
            },
        ]
    )
    out = compute_time_safe_franchise_features(frame)
    # B's film is later in time but a different franchise: no history.
    assert out.loc[1].tolist() == [0.0, 0.0, 0.0]
