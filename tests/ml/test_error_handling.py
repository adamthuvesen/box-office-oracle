"""
Test error conditions and edge cases.

Ensures graceful failure with clear error messages.
"""

import pandas as pd
import pytest


class TestTemporalTransformerErrors:
    """Test TemporalTransformer error handling."""

    def test_missing_release_date_column(self):
        from box_office.ml.feature_pipeline import TemporalTransformer

        data = pd.DataFrame({"OTHER_COLUMN": [1, 2, 3]})

        transformer = TemporalTransformer()

        with pytest.raises(KeyError):
            transformer.transform(data)

    def test_invalid_date_format(self):
        """Invalid dates fall back to default zero values rather than raising."""
        # pd.to_datetime with errors="coerce" turns bad dates into NaT; extracting
        # month/year from NaT yields 0 via fillna(0) — intentional fallback behavior.
        from box_office.ml.feature_pipeline import TemporalTransformer

        data = pd.DataFrame({"RELEASE_DATE": ["not-a-date", "invalid", "2023-13-45"]})

        transformer = TemporalTransformer()
        result = transformer.transform(data)

        assert len(result) == 3
        assert (result["RELEASE_MONTH"] == 0).all()
        assert (result["IS_SUMMER_RELEASE"] == 0).all()
        assert (result["YEARS_SINCE_2000"] == 0).all()


class TestFeaturePreprocessorErrors:
    """Test FeaturePreprocessorHigh error handling."""

    def test_empty_dataframe(self):
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [],
                "RELEASE_DATE": pd.to_datetime([]),
                "PRODUCTION_BUDGET": [],
                "RUNTIME": [],
                "DIRECTOR": [],
                "PRODUCTION_COMPANY": [],
                "ACTORS": [],
                "MPAA": [],
                "GENRES": [],
            }
        )

        preprocessor = FeaturePreprocessorHigh()

        # Empty DataFrame fit_transforms to empty without raising.
        result = preprocessor.fit_transform(data)
        assert len(result) == 0
        assert result.shape[0] == 0
        # ``fit_transform`` always returns a DataFrame, never None.
        assert isinstance(result, pd.DataFrame)

    def test_single_row_dataframe(self, expected_feature_count):
        """Test preprocessing with single-row DataFrame.

        Feature count comes from the live preprocessor so intentional schema
        changes fail this assertion alongside the rest of the suite.
        """
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        # Single row of data
        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [2023],
                "RELEASE_DATE": pd.to_datetime(["2023-06-15"]),
                "PRODUCTION_BUDGET": [1000000],
                "RUNTIME": [120],
                "DIRECTOR": ["Director A"],
                "PRODUCTION_COMPANY": ["Company A"],
                "ACTORS": ["Actor A, Actor B"],
                "MPAA": ["PG-13"],
                "GENRES": ["Action, Adventure"],
            }
        )

        preprocessor = FeaturePreprocessorHigh()
        result = preprocessor.fit_transform(data)

        # Should successfully process single row
        assert len(result) == 1
        assert result.shape[1] == expected_feature_count

    def test_mismatched_columns_train_val(self):
        """Preprocessor tolerates a missing column at transform time."""
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        X_train = pd.DataFrame(
            {
                "RELEASE_YEAR": [2020, 2021],
                "RELEASE_DATE": pd.to_datetime(["2020-01-01", "2021-01-01"]),
                "PRODUCTION_BUDGET": [1000000, 2000000],
                "RUNTIME": [120, 130],
                "DIRECTOR": ["Dir A", "Dir B"],
                "PRODUCTION_COMPANY": ["Company A", "Company B"],
                "ACTORS": ["Actor A", "Actor B"],
                "MPAA": ["PG-13", "R"],
                "GENRES": ["Action", "Comedy"],
            }
        )

        X_val = X_train.drop(columns=["RUNTIME"]).copy()

        preprocessor = FeaturePreprocessorHigh()
        preprocessor.fit(X_train)

        result = preprocessor.transform(X_val)
        # Transform still produces the full engineered feature matrix even with a
        # missing input column: same width as the fitted pipeline reports, so a
        # silent column drop would fail this rather than slip through.
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(X_val)
        assert result.shape[1] == len(preprocessor.get_feature_names())


class TestGenreTransformerErrors:
    """Test GenreTransformer error handling."""

    def test_empty_genres_string(self):
        from box_office.ml.feature_pipeline import GenreTransformer

        data = pd.DataFrame({"GENRES": ["Action, Adventure", "", None, "Comedy"]})

        transformer = GenreTransformer()
        transformer.fit(data)

        result = transformer.transform(data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        # SUPER_GENRE_ENCODED must be present for all rows, including empty/None genres.
        assert "SUPER_GENRE_ENCODED" in result.columns
        assert result["SUPER_GENRE_ENCODED"].notna().all()


class TestIndustryTransformerErrors:
    """Test IndustryTransformer error handling."""

    def test_transform_before_fit(self):
        from box_office.ml.feature_pipeline import IndustryTransformer

        data = pd.DataFrame(
            {
                "DIRECTOR": ["Dir A", "Dir B"],
                "PRODUCTION_COMPANY": ["Company A", "Company B"],
                "ACTORS": ["Actor A", "Actor B"],
                "MPAA": ["PG-13", "R"],
            }
        )

        transformer = IndustryTransformer()

        # frequency_maps not fitted -> AttributeError.
        with pytest.raises(AttributeError):
            transformer.transform(data)

    def test_new_unseen_values_in_transform(self):
        from box_office.ml.feature_pipeline import IndustryTransformer

        train_data = pd.DataFrame(
            {
                "DIRECTOR": ["Dir A", "Dir A", "Dir B"],
                "PRODUCTION_COMPANY": ["Company A", "Company A", "Company B"],
                "ACTORS": ["Actor A", "Actor B", "Actor C"],
                "MPAA": ["PG-13", "R", "PG-13"],
            }
        )

        test_data = pd.DataFrame(
            {
                "DIRECTOR": ["Dir A", "Dir C"],  # Dir C is unseen
                "PRODUCTION_COMPANY": ["Company A", "Company A"],
                "ACTORS": ["Actor A", "Actor D"],  # Actor D is unseen
                "MPAA": ["PG-13", "PG-13"],
            }
        )

        transformer = IndustryTransformer()
        transformer.fit(train_data)

        result = transformer.transform(test_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "DIRECTOR_FREQ" in result.columns
        assert "MPAA_ENCODED" in result.columns
        # Unseen values fall back to a non-zero default frequency.
        assert result["DIRECTOR_FREQ"].iloc[1] > 0


class TestDataValidation:
    """Test data validation utilities."""

    def test_negative_budget_values(self):
        """Negative budgets pass through the transformer without raising."""
        from box_office.ml.feature_pipeline import FinancialTransformer

        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [2020, 2021, 2022],
                "PRODUCTION_BUDGET": [5000000, 3000000, -1000000],
                "RUNTIME": [120, 130, 140],
                "IS_SUMMER_RELEASE": [1, 0, 1],
                "IS_HOLIDAY_RELEASE": [0, 1, 0],
                "IS_WEEKEND_RELEASE": [1, 1, 0],
                "IS_COVID_ERA": [0, 0, 1],
                "IS_STREAMING_MATURE_ERA": [1, 1, 1],
            }
        )

        transformer = FinancialTransformer()

        result = transformer.transform(data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3
        assert "LOG_PRODUCTION_BUDGET" in result.columns
