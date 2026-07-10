"""
Essential tests for feature preprocessor.
"""

import numpy as np
import pandas as pd
import pytest


class TestFeaturePreprocessor:
    """Test FeaturePreprocessorHigh essential functionality."""

    def test_preprocessor_initialization(self):
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        preprocessor = FeaturePreprocessorHigh()
        assert preprocessor is not None
        # Pre-engineered drop + five engineered transformers + raw-column strip.
        assert len(preprocessor.pipeline.named_steps) == 8

    def test_fit_transform_with_realistic_data(self, sample_movie_data):
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        X = sample_movie_data.drop(columns=["WORLDWIDE_GROSS"])

        preprocessor = FeaturePreprocessorHigh()
        X_transformed = preprocessor.fit_transform(X)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == 10
        assert not X_transformed.isnull().any().any(), (
            "Transformed data contains NaN values"
        )
        assert X_transformed.dtypes.apply(
            lambda x: np.issubdtype(x, np.number)
        ).all(), "Not all features are numeric"

    def test_produces_correct_feature_count(
        self, sample_movie_data, expected_feature_count
    ):
        """Feature count tracks the live preprocessor."""
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        X = sample_movie_data.drop(columns=["WORLDWIDE_GROSS"])

        preprocessor = FeaturePreprocessorHigh()
        X_transformed = preprocessor.fit_transform(X)

        actual_features = X_transformed.shape[1]

        assert actual_features == expected_feature_count, (
            f"Expected {expected_feature_count} features, got {actual_features}"
        )

        # Verify get_feature_names matches
        feature_names = preprocessor.get_feature_names()
        assert len(feature_names) == expected_feature_count

    def test_handles_missing_values(self):
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [2020, 2021, None, 2023],
                "RELEASE_DATE": pd.to_datetime(
                    ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]
                ),
                "PRODUCTION_BUDGET": [1000000, 2000000, 3000000, None],
                "RUNTIME": [120, 130, None, 150],
                "DIRECTOR": ["Dir A", "Dir B", "Dir C", None],
                "PRODUCTION_COMPANY": ["Company A", None, "Company C", "Company D"],
                "ACTORS": ["Actor A", "Actor B", None, "Actor D"],
                "MPAA": ["PG-13", "R", None, "PG"],
                "GENRES": ["Action", "Comedy", None, "Drama"],
            }
        )

        preprocessor = FeaturePreprocessorHigh()
        X_transformed = preprocessor.fit_transform(data)

        assert X_transformed is not None
        assert len(X_transformed) == 4

        # NaN must propagate so XGBoost models it natively; a fillna(0)
        # regression would silently mask data quality.
        assert X_transformed.isnull().any().any(), (
            "expected NaN to propagate through the pipeline; if it doesn't, "
            "the silent fillna(0) regressed and we're back to masking data quality"
        )

    def test_transform_without_fit_fails(self):
        """IndustryTransformer's fitted-state check propagates as an error."""
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        preprocessor = FeaturePreprocessorHigh()

        data = pd.DataFrame(
            {
                "RELEASE_YEAR": [2020],
                "RELEASE_DATE": pd.to_datetime(["2020-01-01"]),
                "PRODUCTION_BUDGET": [1000000],
                "RUNTIME": [120],
                "DIRECTOR": ["Dir A"],
                "PRODUCTION_COMPANY": ["Company A"],
                "ACTORS": ["Actor A"],
                "MPAA": ["PG-13"],
                "GENRES": ["Action"],
            }
        )

        with pytest.raises((AttributeError, ValueError)):
            preprocessor.transform(data)


class TestFeatureNameCollision:
    """Duplicate column names raise loudly instead of silently deduping."""

    def test_duplicate_columns_raise_collision_error(self):
        from sklearn.base import BaseEstimator, TransformerMixin

        from box_office.ml.feature_preprocessor import (
            FeatureNameCollisionError,
            FeaturePreprocessorHigh,
        )

        preprocessor = FeaturePreprocessorHigh()

        # Inject a step that re-emits a SELECTED feature ('PRODUCTION_BUDGET'). Insert before
        # the final 'feature_selector' projection so the duplicate survives into
        # fit_transform output and trips the collision guard.
        class CollidingTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                # Concat-with-duplicate-name forces a true duplicate column.
                dup = pd.DataFrame({"PRODUCTION_BUDGET": [0] * len(X)}, index=X.index)
                return pd.concat([X, dup], axis=1)

        selector_idx = next(
            i
            for i, (name, _) in enumerate(preprocessor.pipeline.steps)
            if name == "feature_selector"
        )
        preprocessor.pipeline.steps.insert(
            selector_idx, ("collider", CollidingTransformer())
        )

        df = pd.DataFrame(
            {
                "RELEASE_YEAR": [2020, 2021],
                "RELEASE_DATE": pd.to_datetime(["2020-06-01", "2021-06-01"]),
                "PRODUCTION_BUDGET": [1000000, 2000000],
                "RUNTIME": [120, 130],
                "DIRECTOR": ["Dir A", "Dir B"],
                "PRODUCTION_COMPANY": ["Co A", "Co B"],
                "ACTORS": ["Actor A", "Actor B"],
                "MPAA": ["PG-13", "R"],
                "GENRES": ["Action", "Drama"],
            }
        )

        with pytest.raises(FeatureNameCollisionError) as exc:
            preprocessor.fit_transform(df)

        assert "PRODUCTION_BUDGET" in str(exc.value)

    def test_normal_pipeline_columns_are_unique(self, sample_movie_data):
        from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

        preprocessor = FeaturePreprocessorHigh()
        result = preprocessor.fit_transform(sample_movie_data)

        assert result.columns.is_unique, (
            "Default pipeline emitted duplicates: "
            f"{result.columns[result.columns.duplicated()].tolist()}"
        )
