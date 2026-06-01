"""
Focused tests for temporal transformer business logic.

Tests complex temporal feature rules (COVID, seasons, holidays).
"""

import pandas as pd
import pytest


class TestTemporalTransformer:
    """Test TemporalTransformer temporal feature extraction."""

    def test_covid_era_detection(self):
        """COVID era starts 2020-03-01 and stays on for all later dates."""
        from box_office.ml.feature_pipeline import TemporalTransformer

        data = pd.DataFrame(
            {
                "RELEASE_DATE": pd.to_datetime(
                    [
                        "2020-02-28",  # day before COVID start
                        "2020-03-01",  # COVID start
                        "2020-06-15",
                        "2024-01-01",
                    ]
                )
            }
        )

        transformer = TemporalTransformer()
        result = transformer.transform(data)

        assert result["IS_COVID_ERA"].iloc[0] == 0, "Pre-COVID should be 0"
        assert result["IS_COVID_ERA"].iloc[1] == 1, "COVID start should be 1"
        assert result["IS_COVID_ERA"].iloc[2] == 1, "During COVID should be 1"
        assert result["IS_COVID_ERA"].iloc[3] == 1, "Post-COVID still in era"

    def test_summer_release_detection(self):
        """Summer covers May-August inclusive."""
        from box_office.ml.feature_pipeline import TemporalTransformer

        data = pd.DataFrame(
            {
                "RELEASE_DATE": pd.to_datetime(
                    [
                        "2023-04-30",
                        "2023-05-01",
                        "2023-06-15",
                        "2023-07-20",
                        "2023-08-31",
                        "2023-09-01",
                    ]
                )
            }
        )

        transformer = TemporalTransformer()
        result = transformer.transform(data)

        expected_summer = [0, 1, 1, 1, 1, 0]
        assert list(result["IS_SUMMER_RELEASE"]) == expected_summer

    def test_holiday_release_logic(self):
        """Holiday window covers Nov, Dec, and the first ~10 days of Jan."""
        from box_office.ml.feature_pipeline import TemporalTransformer

        data = pd.DataFrame(
            {
                "RELEASE_DATE": pd.to_datetime(
                    [
                        "2023-10-31",
                        "2023-11-01",
                        "2023-12-25",
                        "2024-01-10",
                        "2024-01-20",
                    ]
                )
            }
        )

        transformer = TemporalTransformer()
        result = transformer.transform(data)

        assert result["IS_HOLIDAY_RELEASE"].iloc[0] == 0, "October not holiday"
        assert result["IS_HOLIDAY_RELEASE"].iloc[1] == 1, "November is holiday"
        assert result["IS_HOLIDAY_RELEASE"].iloc[2] == 1, "December is holiday"
        assert result["IS_HOLIDAY_RELEASE"].iloc[3] == 1, "Early Jan is holiday"
        assert result["IS_HOLIDAY_RELEASE"].iloc[4] == 0, "Mid Jan not holiday"

    def test_blockbuster_season_detection(self):
        """Blockbuster season covers May, Jun, Jul, Nov, Dec."""
        from box_office.ml.feature_pipeline import TemporalTransformer

        data = pd.DataFrame(
            {
                "RELEASE_DATE": pd.to_datetime(
                    [
                        "2023-03-15",
                        "2023-05-15",
                        "2023-07-15",
                        "2023-09-15",
                        "2023-11-15",
                    ]
                )
            }
        )

        transformer = TemporalTransformer()
        result = transformer.transform(data)

        expected_blockbuster = [0, 1, 1, 0, 1]
        assert list(result["IS_BLOCKBUSTER_SEASON"]) == expected_blockbuster

    def test_streaming_era_features(self):
        """Pre-streaming = year < 2010; mature streaming = year >= 2015."""
        from box_office.ml.feature_pipeline import TemporalTransformer

        data = pd.DataFrame(
            {
                "RELEASE_DATE": pd.to_datetime(
                    [
                        "2009-06-15",
                        "2010-06-15",
                        "2014-12-31",
                        "2015-01-01",
                        "2023-06-15",
                    ]
                )
            }
        )

        transformer = TemporalTransformer()
        result = transformer.transform(data)

        assert result["IS_PRE_STREAMING_ERA"].iloc[0] == 1
        assert result["IS_PRE_STREAMING_ERA"].iloc[1] == 0

        assert result["IS_STREAMING_MATURE_ERA"].iloc[2] == 0
        assert result["IS_STREAMING_MATURE_ERA"].iloc[3] == 1
        assert result["IS_STREAMING_MATURE_ERA"].iloc[4] == 1

    @pytest.mark.parametrize("year", [2022, 2023, 2024, 2025])
    def test_special_weekend_detection(self, year):
        """Special weekend detection across multiple years.

        Expected dates come from the calendar, not a year-specific lookup,
        so a year-dependent regression in the holiday-window logic surfaces.
        """
        from box_office.ml.feature_pipeline import TemporalTransformer

        # Compute the Saturday of Memorial Day weekend (last Monday of May - 2 days).
        memorial_monday = pd.Timestamp(f"{year}-05-31")
        while memorial_monday.weekday() != 0:  # 0 = Monday
            memorial_monday -= pd.Timedelta(days=1)
        memorial_saturday = memorial_monday - pd.Timedelta(days=2)

        # Compute Thanksgiving (4th Thursday of November).
        nov_first = pd.Timestamp(f"{year}-11-01")
        # Day-of-month of the first Thursday.
        first_thursday_day = ((3 - nov_first.weekday()) % 7) + 1
        thanksgiving = pd.Timestamp(f"{year}-11-{first_thursday_day + 21:02d}")

        july_fourth = pd.Timestamp(f"{year}-07-04")
        christmas = pd.Timestamp(f"{year}-12-25")

        data = pd.DataFrame(
            {
                "RELEASE_DATE": pd.to_datetime(
                    [
                        memorial_saturday,
                        july_fourth,
                        thanksgiving,
                        christmas,
                    ]
                )
            }
        )

        transformer = TemporalTransformer()
        result = transformer.transform(data)

        assert (
            result["IS_MEMORIAL_DAY_WEEKEND"].iloc[0] == 1
        ), f"Memorial Day Saturday {memorial_saturday.date()} not flagged for year {year}"
        assert (
            result["IS_JULY_4TH_WEEKEND"].iloc[1] == 1
        ), f"July 4 not flagged for year {year}"
        assert (
            result["IS_THANKSGIVING_WEEK"].iloc[2] == 1
        ), f"Thanksgiving {thanksgiving.date()} not flagged for year {year}"
        assert (
            result["IS_CHRISTMAS_WEEK"].iloc[3] == 1
        ), f"Christmas not flagged for year {year}"

    def test_temporal_columns_present_after_transform(self):
        from box_office.ml.feature_pipeline import TemporalTransformer

        data = pd.DataFrame({"RELEASE_DATE": ["2020-06-15", "2024-01-01"]})
        out = TemporalTransformer().transform(data)
        for c in (
            "IS_SUMMER_RELEASE",
            "IS_COVID_ERA",
            "RELEASE_MONTH",
            "YEARS_SINCE_2000",
        ):
            assert c in out.columns
