"""
Essential tests for orchestration data tasks.
"""

import pandas as pd
import numpy as np
import pytest


class TestTemporalDataSplit:
    """Test temporal data splitting functionality."""

    def test_temporal_split_logic(self):
        test_data = pd.DataFrame(
            {
                "release_date": pd.to_datetime(
                    ["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01"]
                ),
                "feature1": [1, 2, 3, 4],
                "target": [10, 20, 30, 40],
            }
        )

        split_date = pd.to_datetime("2021-01-01")
        train_mask = test_data["release_date"] < split_date
        val_mask = test_data["release_date"] >= split_date

        X_train = test_data[train_mask].drop(columns=["target"])
        X_val = test_data[val_mask].drop(columns=["target"])
        y_train = test_data[train_mask]["target"]
        y_val = test_data[val_mask]["target"]

        assert len(X_train) == 2
        assert len(X_val) == 2
        assert list(y_train) == [10, 20]
        assert list(y_val) == [30, 40]

    def test_split_data_by_year(self, sample_movie_data):
        target_column = "WORLDWIDE_GROSS"

        train_mask = sample_movie_data["RELEASE_YEAR"] < 2024
        val_mask = sample_movie_data["RELEASE_YEAR"] >= 2024

        train_df = sample_movie_data[train_mask]
        val_df = sample_movie_data[val_mask]

        X_train = train_df.drop(columns=[target_column])
        X_val = val_df.drop(columns=[target_column])

        assert X_train["RELEASE_YEAR"].max() < 2024
        assert len(X_train) == 9  # sample has 9 pre-2024 rows

        assert X_val["RELEASE_YEAR"].min() >= 2024
        assert len(X_val) == 1

        assert len(X_train) + len(X_val) == len(sample_movie_data)


class TestTargetTransformation:
    """Test target transformation functions."""

    def test_transform_targets(self):
        from box_office.ml.feature_preprocessor import TargetTransformer

        y_train = pd.Series([100, 1000, 10000, 100000])
        y_val = pd.Series([500, 5000])

        y_train_log, y_val_log = TargetTransformer.log_transform(y_train, y_val)

        assert isinstance(y_train_log, (np.ndarray, pd.Series))
        assert isinstance(y_val_log, (np.ndarray, pd.Series))

        assert len(y_train_log) == len(y_train)
        assert len(y_val_log) == len(y_val)

        expected_train = np.log1p(y_train)
        np.testing.assert_array_almost_equal(y_train_log, expected_train)

        assert np.all(np.isfinite(y_train_log))
        assert np.all(np.isfinite(y_val_log))


class TestFeatureScaling:
    """Test feature scaling functions."""

    def test_scale_features(self, sample_features_data, monkeypatch):
        from unittest.mock import MagicMock
        import box_office.orchestration.tasks.data_tasks as data_tasks
        from box_office.orchestration.tasks.data_tasks import scale_features

        monkeypatch.setattr(data_tasks, "get_run_logger", lambda: MagicMock())

        train_size = 8
        X_train = sample_features_data.iloc[:train_size]
        X_val = sample_features_data.iloc[train_size:]

        X_train_scaled, X_val_scaled, scaler = scale_features.fn(X_train, X_val)

        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape

        train_means = X_train_scaled.mean()
        train_stds = X_train_scaled.std()

        assert np.abs(train_means).max() < 0.1, "Scaled train set not centered"
        assert np.abs(train_stds - 1.0).max() < 0.1, "Scaled train set not standardized"

        assert list(X_train_scaled.columns) == list(X_train.columns)


class TestFeatureMetadata:
    def test_create_feature_metadata(self, monkeypatch):
        from unittest.mock import MagicMock
        import box_office.orchestration.tasks.data_tasks as data_tasks
        from box_office.orchestration.tasks.data_tasks import create_feature_metadata

        monkeypatch.setattr(data_tasks, "get_run_logger", lambda: MagicMock())

        feature_names = ["FEATURE_1", "FEATURE_2", "FEATURE_3"]
        processor_path = "/path/to/processor.pkl"
        scaler_path = "/path/to/scaler.pkl"

        metadata = create_feature_metadata.fn(
            feature_names, processor_path, scaler_path
        )

        assert len(metadata) == 3
        assert "FEATURE_NAME" in metadata.columns
        assert "FEATURE_INDEX" in metadata.columns
        assert "CREATED_AT" in metadata.columns
        assert "PROCESSOR_PATH" in metadata.columns
        assert "SCALER_PATH" in metadata.columns

        assert list(metadata["FEATURE_INDEX"]) == [0, 1, 2]

        assert metadata["PROCESSOR_PATH"].iloc[0] == processor_path
        assert metadata["SCALER_PATH"].iloc[0] == scaler_path


class TestDataSplitter:
    def test_data_splitter_imports(self):
        from box_office.ml.feature_preprocessor import DataSplitter

        assert DataSplitter is not None
        assert hasattr(DataSplitter, "split_data")

    def test_data_splitter_basic_split(self, sample_movie_data):
        """Test DataSplitter temporal split behavior."""
        from box_office.ml.feature_preprocessor import DataSplitter

        target_column = "WORLDWIDE_GROSS"

        X_train, X_val, y_train, y_val = DataSplitter.split_data(
            sample_movie_data,
            target_column,
            test_size=0.2,
        )

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_val, pd.Series)

        total_samples = len(sample_movie_data)
        train_samples = len(X_train)
        val_samples = len(X_val)

        assert train_samples + val_samples == total_samples
        assert val_samples / total_samples == pytest.approx(0.2, abs=0.1)

        assert target_column not in X_train.columns
        assert target_column not in X_val.columns

    def test_data_splitter_is_temporal(self, sample_movie_data):
        """Validation rows MUST post-date training rows (spec scenario)."""
        from box_office.ml.feature_preprocessor import DataSplitter

        X_train, X_val, _, _ = DataSplitter.split_data(
            sample_movie_data,
            target_column="WORLDWIDE_GROSS",
            test_size=0.2,
        )

        if len(X_train) and len(X_val):
            assert X_train["RELEASE_DATE"].max() <= X_val["RELEASE_DATE"].min()

    def test_data_splitter_falls_back_to_release_year(self):
        """If RELEASE_DATE is missing, falls back to RELEASE_YEAR."""
        from box_office.ml.feature_preprocessor import DataSplitter

        df = pd.DataFrame(
            {
                "RELEASE_YEAR": [
                    2018,
                    2019,
                    2020,
                    2021,
                    2022,
                    2023,
                    2024,
                    2025,
                    2026,
                    2027,
                ],
                "feature_a": range(10),
                "WORLDWIDE_GROSS": [
                    1e8,
                    2e8,
                    3e8,
                    1.5e8,
                    4e8,
                    2.5e8,
                    5e8,
                    3.5e8,
                    6e8,
                    4.5e8,
                ],
            }
        )

        X_train, X_val, _, _ = DataSplitter.split_data(
            df, target_column="WORLDWIDE_GROSS", test_size=0.3
        )

        assert X_train["RELEASE_YEAR"].max() <= X_val["RELEASE_YEAR"].min()
