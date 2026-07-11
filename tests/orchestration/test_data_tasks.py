"""
Essential tests for orchestration data tasks.
"""

import numpy as np
import pytest


@pytest.fixture
def dbt_task(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import box_office.orchestration.tasks.data_tasks as data_tasks

    project_dir = tmp_path / "transformations"
    project_dir.mkdir()
    (project_dir / "profiles.yml").write_text("box_office: {}\n")

    runner = MagicMock()
    monkeypatch.setattr(data_tasks, "get_run_logger", lambda: MagicMock())
    monkeypatch.setattr(
        data_tasks,
        "config",
        SimpleNamespace(
            paths=SimpleNamespace(
                project_root=str(tmp_path), transformations_dir="transformations"
            ),
            snowflake=SimpleNamespace(
                database="BOX_OFFICE",
                warehouse="COMPUTE_WH",
                schemas=SimpleNamespace(staging="STAGING"),
            ),
        ),
    )
    runner_factory = MagicMock(return_value=runner)
    monkeypatch.setattr(data_tasks, "PrefectDbtRunner", runner_factory)
    return data_tasks, runner, runner_factory


def test_dbt_task_builds_and_tests_staging(dbt_task):
    data_tasks, runner, runner_factory = dbt_task

    data_tasks.run_raw_to_staging_dbt_transformations.fn()

    assert [call.args[0] for call in runner.invoke.call_args_list] == [
        ["deps"],
        ["debug"],
        ["build", "--select", "staging"],
    ]
    assert runner_factory.call_args.kwargs["raise_on_failure"] is True


def test_dbt_task_propagates_build_failure(dbt_task):
    data_tasks, runner, _ = dbt_task
    runner.invoke.side_effect = [None, None, ValueError("dbt failed")]

    with pytest.raises(ValueError, match="dbt failed"):
        data_tasks.run_raw_to_staging_dbt_transformations.fn()


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
