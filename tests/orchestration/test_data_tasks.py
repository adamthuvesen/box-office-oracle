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

    data_tasks.run_raw_to_staging_dbt_transformations.fn(approved=True)

    assert [call.args[0] for call in runner.invoke.call_args_list] == [
        ["deps"],
        ["debug"],
        ["build", "--select", "staging"],
    ]
    assert runner_factory.call_args.kwargs["raise_on_failure"] is True


def test_dbt_task_logs_the_nodes_the_build_processed(tmp_path, monkeypatch, dbt_task):
    from unittest.mock import MagicMock

    from dbt.artifacts.schemas.run import RunExecutionResult
    from dbt.cli.main import dbtRunnerResult

    data_tasks, runner, _ = dbt_task
    logger = MagicMock()
    monkeypatch.setattr(data_tasks, "get_run_logger", lambda: logger)
    build = RunExecutionResult(results=[MagicMock(), MagicMock()], elapsed_time=1.0, args={})
    runner.invoke.side_effect = [None, None, dbtRunnerResult(success=True, result=build)]

    data_tasks.run_raw_to_staging_dbt_transformations.fn(approved=True)

    logger.info.assert_any_call("dbt build completed: 2 nodes processed")


def test_dbt_task_propagates_build_failure(dbt_task):
    data_tasks, runner, _ = dbt_task
    runner.invoke.side_effect = [None, None, ValueError("dbt failed")]

    with pytest.raises(ValueError, match="dbt failed"):
        data_tasks.run_raw_to_staging_dbt_transformations.fn(approved=True)


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

        X_train_scaled, X_val_scaled, _scaler = scale_features.fn(X_train, X_val)

        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape

        train_means = X_train_scaled.mean()
        train_stds = X_train_scaled.std()

        assert np.abs(train_means).max() < 0.1, "Scaled train set not centered"
        assert np.abs(train_stds - 1.0).max() < 0.1, "Scaled train set not standardized"

        assert list(X_train_scaled.columns) == list(X_train.columns)
