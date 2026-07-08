"""Tests for ML edge cases and critical path validation."""

import numpy as np
import pandas as pd
import pytest


class TestCrossValidationIndexHandling:
    """Test OOF index handling with various DataFrame index types."""

    def test_cv_with_non_range_index(self):
        """OOF works with non-contiguous/non-RangeIndex and every fold has data.

        Data is constructed deterministically with a fixed number of rows per
        year so every fold the splitter produces is non-empty.
        """
        from box_office.ml.model import BoxOfficeXGBoostModel, TimeSeriesCrossValidator

        np.random.seed(42)

        # 20 rows per year × 5 years = 100 rows. Each year is guaranteed to be
        # present in the right quantity, so the splitter cannot produce an
        # empty validation fold.
        years = [2018, 2019, 2020, 2021, 2022]
        rows_per_year = 20
        n_samples = len(years) * rows_per_year

        years_column = np.repeat(years, rows_per_year)
        # Non-contiguous, non-RangeIndex labels (still unique).
        index = np.random.choice(range(1000, 2000), n_samples, replace=False)

        df = pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.randn(n_samples),
                "feature3": np.random.randn(n_samples),
            },
            index=index,
        )

        X = df.copy()
        y = pd.Series(
            np.random.randn(n_samples) + 10, index=df.index
        )  # Log-transformed target (positive values)
        dates = pd.Series(years_column, index=df.index)

        # Initialize cross-validator with minimal settings
        cv = TimeSeriesCrossValidator(
            cv_folds=2, start_eval_year=2020, end_year=2022, early_stopping_rounds=5
        )

        # This should not raise or produce corrupted predictions
        cv_results = cv.cross_validate(
            BoxOfficeXGBoostModel, X, y, dates, n_estimators=10, max_depth=2
        )

        # Verify results are valid
        assert "mean_cv_mae" in cv_results
        assert "oof_predictions" in cv_results
        assert cv_results["mean_cv_mae"] > 0

    def test_oof_predictions_align_with_source_rows(self):
        """OOF predictions preserve their source-row alignment."""
        from box_office.ml.model import BoxOfficeXGBoostModel, TimeSeriesCrossValidator

        np.random.seed(7)
        n_samples = 60

        # Non-RangeIndex; permuted labels.
        index = np.random.choice(range(5000, 6000), n_samples, replace=False)

        X = pd.DataFrame(
            {
                "f1": np.random.randn(n_samples),
                "f2": np.random.randn(n_samples),
            },
            index=index,
        )
        y = pd.Series(np.random.randn(n_samples) + 10, index=index)
        dates = pd.Series(
            np.random.choice([2019, 2020, 2021, 2022], n_samples),
            index=index,
        )

        cv = TimeSeriesCrossValidator(
            cv_folds=2,
            start_eval_year=2020,
            end_year=2022,
            early_stopping_rounds=5,
        )
        cv_results = cv.cross_validate(
            BoxOfficeXGBoostModel,
            X,
            y,
            dates,
            n_estimators=10,
            max_depth=2,
        )

        # Indices stored in 'oof_predictions' are the post-reset positional
        # indices (0..N-1 over the reset-index frame). Resolving each to the
        # underlying date should yield a year >= start_eval_year — i.e. it
        # was ACTUALLY a validation row, not training data leaking through.
        dates_reset = dates.reset_index(drop=True)
        for idx_str in cv_results["oof_predictions"]:
            pos = int(idx_str)
            assert dates_reset.iloc[pos] >= 2020, (
                f"OOF prediction at positional idx {pos} maps to date "
                f"{dates_reset.iloc[pos]}, which is before start_eval_year=2020 — "
                "index alignment is broken."
            )


class TestFeatureAccumulationDuplicateColumns:
    """Test that duplicate columns don't corrupt feature values."""

    def test_get_column_takes_last_engineered(self):
        """``_column`` returns the LAST (engineered) column, not the FIRST (raw)."""
        from box_office.ml.feature_pipeline import _column

        df_with_duplicates = pd.DataFrame()
        # raw
        df_with_duplicates["LOG_PRODUCTION_BUDGET"] = [
            "not_a_number",
            "also_not",
        ]
        df_with_duplicates = pd.concat(
            [
                df_with_duplicates,
                pd.DataFrame(
                    {"LOG_PRODUCTION_BUDGET": [1000000.0, 2000000.0]}
                ),  # engineered (last)
            ],
            axis=1,
        )

        result = _column(df_with_duplicates, "LOG_PRODUCTION_BUDGET")
        assert result.dtype in [np.float64, np.int64]
        assert result.iloc[0] == 1000000.0

    def test_preprocessor_produces_numeric_features(self):
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [2020, 2021],
                "RELEASE_DATE": pd.to_datetime(["2020-06-15", "2021-12-20"]),
                "PRODUCTION_BUDGET": [1000000, 2000000],
                "RUNTIME": [120, 130],
                "DIRECTOR": ["Dir A", "Dir B"],
                "PRODUCTION_COMPANY": ["Company A", "Company B"],
                "ACTORS": ["Actor A", "Actor B"],
                "MPAA": ["PG-13", "R"],
                "GENRES": ["Action", "Comedy"],
            }
        )

        preprocessor = FeaturePreprocessorHigh()
        result = preprocessor.fit_transform(data)

        for col in result.columns:
            assert result[col].dtype in [
                np.float64,
                np.int64,
                np.int32,
                np.float32,
            ], f"Column {col} has non-numeric dtype: {result[col].dtype}"


class TestMPAAUnseenCategories:
    """Unseen MPAA ratings map to the 'unknown' bucket instead of raising."""

    def test_mpaa_unseen_maps_to_unknown_bucket(self):
        from box_office.ml.feature_pipeline import IndustryTransformer

        train_data = pd.DataFrame(
            {
                "DIRECTOR": ["Dir A", "Dir B"],
                "PRODUCTION_COMPANY": ["Company A", "Company B"],
                "ACTORS": ["Actor A", "Actor B"],
                "MPAA": ["PG-13", "R"],
            }
        )

        test_data = pd.DataFrame(
            {
                "DIRECTOR": ["Dir A"],
                "PRODUCTION_COMPANY": ["Company A"],
                "ACTORS": ["Actor A"],
                "MPAA": ["NC-17"],  # Unseen rating — must NOT raise.
            }
        )

        transformer = IndustryTransformer()
        transformer.fit(train_data)

        result = transformer.transform(test_data)

        unknown_code = transformer.mpaa_encoder.transform([transformer.UNKNOWN_MPAA])[0]
        assert result["MPAA_ENCODED"].iloc[0] == unknown_code

    def test_mpaa_known_rating_still_maps_correctly(self):
        from box_office.ml.feature_pipeline import IndustryTransformer

        train_data = pd.DataFrame(
            {
                "DIRECTOR": ["Dir A", "Dir B"],
                "PRODUCTION_COMPANY": ["Company A", "Company B"],
                "ACTORS": ["Actor A", "Actor B"],
                "MPAA": ["PG-13", "R"],
            }
        )

        transformer = IndustryTransformer()
        transformer.fit(train_data)

        # 'PG-13' is in training vocab; must NOT collapse to unknown bucket.
        result = transformer.transform(
            pd.DataFrame(
                {
                    "DIRECTOR": ["Dir A"],
                    "PRODUCTION_COMPANY": ["Company A"],
                    "ACTORS": ["Actor A"],
                    "MPAA": ["PG-13"],
                }
            )
        )
        unknown_code = transformer.mpaa_encoder.transform([transformer.UNKNOWN_MPAA])[0]
        pg13_code = transformer.mpaa_encoder.transform(["PG-13"])[0]
        assert result["MPAA_ENCODED"].iloc[0] == pg13_code
        assert result["MPAA_ENCODED"].iloc[0] != unknown_code


class TestTransformerSanityChecks:
    """Minimal sanity checks for all transformers."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame(
            {
                "RELEASE_YEAR": [2020, 2021],
                "RELEASE_DATE": pd.to_datetime(["2020-06-15", "2021-12-20"]),
                "PRODUCTION_BUDGET": [1000000, 2000000],
                "RUNTIME": [120, 130],
                "DIRECTOR": ["Dir A", "Dir B"],
                "PRODUCTION_COMPANY": ["Company A", "Company B"],
                "ACTORS": ["Actor A", "Actor B"],
                "MPAA": ["PG-13", "R"],
                "GENRES": ["Action", "Comedy"],
            }
        )

    def test_core_numerical_transformer(self, sample_data):
        from box_office.ml.feature_pipeline import CoreNumericalTransformer

        transformer = CoreNumericalTransformer()
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        assert len(result) == 2
        assert "RELEASE_YEAR" in result.columns
        assert "PRODUCTION_BUDGET" in result.columns

    def test_temporal_transformer(self, sample_data):
        from box_office.ml.feature_pipeline import TemporalTransformer

        transformer = TemporalTransformer()
        result = transformer.transform(sample_data)

        assert len(result) == 2
        assert "IS_SUMMER_RELEASE" in result.columns
        assert "RELEASE_MONTH" in result.columns

    def test_genre_transformer(self, sample_data):
        from box_office.ml.feature_pipeline import GenreTransformer

        transformer = GenreTransformer()
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        assert len(result) == 2
        assert "SUPER_GENRE_ENCODED" in result.columns

    def test_industry_transformer(self, sample_data):
        from box_office.ml.feature_pipeline import IndustryTransformer

        transformer = IndustryTransformer()
        transformer.fit(sample_data)
        result = transformer.transform(sample_data)

        assert len(result) == 2
        assert "DIRECTOR_FREQ" in result.columns
        assert "MPAA_ENCODED" in result.columns

    def test_financial_transformer_with_context(self, sample_data):
        """FinancialTransformer needs accumulated features from upstream transformers."""
        from box_office.ml.feature_pipeline import (
            CoreNumericalTransformer,
            FinancialTransformer,
            TemporalTransformer,
        )

        core = CoreNumericalTransformer()
        temporal = TemporalTransformer()
        financial = FinancialTransformer()

        core_features = core.fit_transform(sample_data)
        temporal_features = temporal.transform(sample_data)

        accumulated = pd.concat([sample_data, core_features, temporal_features], axis=1)

        result = financial.transform(accumulated)

        assert len(result) == 2
        assert "LOG_PRODUCTION_BUDGET" in result.columns


class TestSharedUtilities:
    """Shared text-parsing and column-extraction helpers."""

    def test_process_text_list_handles_various_inputs(self):
        from box_office.ml.text_utils import process_text_list

        assert process_text_list(None) == []
        assert process_text_list(float("nan")) == []
        assert process_text_list("[]") == []
        assert process_text_list(["Action", "Comedy"]) == ["action", "comedy"]
        assert process_text_list("['Action', 'Comedy']") == ["action", "comedy"]
        assert process_text_list("Action") == ["action"]

    def test_column_extractor_handles_single_column(self):
        from box_office.ml.feature_pipeline import _column

        df = pd.DataFrame({"col": [1, 2, 3]})
        result = _column(df, "col")

        assert isinstance(result, pd.Series)
        assert list(result) == [1, 2, 3]
