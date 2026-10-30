"""Every spending or production-writing function refuses without approval.

Each test checks two things: the refusal is ``ApprovalRequired`` and the fake
cloud client records zero calls, then the approved call reaches the fake.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry
from box_office.orchestration.tasks import data_tasks, training_tasks
from box_office.sagemaker import sagemaker_train_job
from box_office.utils.approval import ApprovalRequired

PACKAGE_ARN = "arn:aws:sagemaker:eu-north-1:123456789012:model-package/foo/1"


def _fake_sagemaker_client() -> SimpleNamespace:
    return SimpleNamespace(
        role="arn:aws:iam::123456789012:role/test",
        s3_bucket="bucket",
        s3_prefix="prefix",
        sagemaker_session=MagicMock(),
    )


def _registry() -> AWSModelRegistry:
    registry = AWSModelRegistry.__new__(AWSModelRegistry)
    registry.region_name = "eu-north-1"
    registry.sagemaker_client = MagicMock()
    registry.s3_client = MagicMock()
    return registry


@pytest.fixture
def task_logger(monkeypatch):
    monkeypatch.setattr(training_tasks, "get_run_logger", lambda: MagicMock())
    monkeypatch.setattr(data_tasks, "get_run_logger", lambda: MagicMock())


# --- SageMaker ---------------------------------------------------------------


HYPERPARAMETERS = {
    "n_estimators": 10,
    "learning_rate": 0.1,
    "cv_folds": 3,
    "start_eval_year": 2015,
    "end_year": 2023,
}


def test_training_job_refuses_without_approval():
    client = _fake_sagemaker_client()
    with patch.object(sagemaker_train_job, "XGBoost") as estimator_class:
        with pytest.raises(ApprovalRequired):
            sagemaker_train_job.train_xgboost_model_with_timeseries_cv_framework(
                client, "s3://train", (10, 3), HYPERPARAMETERS
            )
        estimator_class.assert_not_called()


def test_training_job_fits_when_approved():
    client = _fake_sagemaker_client()
    with patch.object(sagemaker_train_job, "XGBoost") as estimator_class:
        sagemaker_train_job.train_xgboost_model_with_timeseries_cv_framework(
            client, "s3://train", (10, 3), HYPERPARAMETERS, approved=True
        )
        estimator_class.return_value.fit.assert_called_once_with({"train": "s3://train"})


def test_training_upload_refuses_without_approval():
    client = _fake_sagemaker_client()
    frame = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ApprovalRequired):
        sagemaker_train_job.upload_processed_data_to_s3(client, frame, frame)
    client.sagemaker_session.upload_data.assert_not_called()


def test_training_upload_writes_when_approved():
    client = _fake_sagemaker_client()
    client.sagemaker_session.upload_data.return_value = "s3://bucket/prefix/train/x.parquet"
    X_train = pd.DataFrame({"a": [1, 2]})
    y_train = pd.DataFrame({"GROSS_LOG": [1.0, 2.0]})
    sagemaker_train_job.upload_processed_data_to_s3(client, X_train, y_train, approved=True)
    client.sagemaker_session.upload_data.assert_called_once()


def test_sagemaker_client_opens_no_session_without_approval():
    with (
        patch.object(sagemaker_train_job.boto3, "Session") as boto_session,
        patch.object(sagemaker_train_job.sagemaker, "Session") as sagemaker_session,
    ):
        with pytest.raises(ApprovalRequired):
            sagemaker_train_job.SageMakerClient("eu-north-1", "arn:role")
        boto_session.assert_not_called()
        sagemaker_session.assert_not_called()

        sagemaker_train_job.SageMakerClient("eu-north-1", "arn:role", approved=True)
        boto_session.assert_called_once()


def test_training_tasks_refuse_without_approval(task_logger):
    client = _fake_sagemaker_client()
    with patch.object(
        training_tasks.sagemaker_train_job, "train_xgboost_model_with_timeseries_cv_framework"
    ) as launch:
        with pytest.raises(ApprovalRequired):
            training_tasks.train_model.fn(client, "s3://train", (10, 3))
        launch.assert_not_called()

    with pytest.raises(ApprovalRequired):
        training_tasks.upload_preprocessing_artifacts_to_s3.fn(client, "p.pkl", "s.pkl")
    client.sagemaker_session.upload_data.assert_not_called()

    with patch.object(training_tasks.sagemaker_train_job, "upload_processed_data_to_s3") as upload:
        with pytest.raises(ApprovalRequired):
            training_tasks.upload_processed_data_to_s3.fn(client, None, None)
        upload.assert_not_called()


def test_preprocessing_artifacts_upload_when_approved(task_logger):
    client = _fake_sagemaker_client()
    client.sagemaker_session.upload_data.return_value = "s3://bucket/artifacts/x"
    training_tasks.upload_preprocessing_artifacts_to_s3.fn(client, "p.pkl", "s.pkl", approved=True)
    assert client.sagemaker_session.upload_data.call_count == 2


# --- Model registry ----------------------------------------------------------


def test_register_model_package_refuses_without_approval():
    registry = _registry()
    with pytest.raises(ApprovalRequired):
        registry.register_model_package("foo", "s3://bucket/key.tar.gz")
    assert registry.sagemaker_client.mock_calls == []
    assert registry.s3_client.mock_calls == []


def test_register_model_package_creates_when_approved():
    registry = _registry()
    registry.sagemaker_client.create_model_package.return_value = {"ModelPackageArn": PACKAGE_ARN}
    with patch(
        "box_office.ml.model_registry.aws_model_registry._compute_sha256_of_s3_object",
        return_value=("a" * 64, 1024),
    ):
        result = registry.register_model_package(
            "foo", "s3://bucket/key.tar.gz", inference_specification={}, approved=True
        )
    registry.sagemaker_client.create_model_package.assert_called_once()
    assert result["model_package_arn"] == PACKAGE_ARN


def test_update_approval_status_refuses_without_approval():
    registry = _registry()
    with pytest.raises(ApprovalRequired):
        registry.update_model_approval_status(PACKAGE_ARN, "Approved")
    assert registry.sagemaker_client.mock_calls == []


def test_update_approval_status_updates_when_approved():
    registry = _registry()
    registry.update_model_approval_status(PACKAGE_ARN, "Approved", approved=True)
    registry.sagemaker_client.update_model_package.assert_called_once()


def test_create_model_package_group_refuses_without_approval():
    registry = _registry()
    with pytest.raises(ApprovalRequired):
        registry.create_model_package_group("foo")
    assert registry.sagemaker_client.mock_calls == []


def test_register_task_refuses_without_approval(task_logger):
    aws_registry = MagicMock()
    with pytest.raises(ApprovalRequired):
        training_tasks.register_model_in_registry.fn("job", 60, "s3://m", aws_registry)
    assert aws_registry.mock_calls == []


def test_register_task_registers_when_approved(task_logger):
    aws_registry = MagicMock()
    aws_registry.create_model_package_group.return_value = {"status": "created"}
    aws_registry.register_model_package.return_value = {
        "status": "success",
        "model_package_arn": PACKAGE_ARN,
    }
    result = training_tasks.register_model_in_registry.fn(
        "job", 60, "s3://m", aws_registry, approved=True
    )
    aws_registry.register_model_package.assert_called_once()
    assert result["status"] == "success"


def test_promote_task_builds_no_registry_without_approval(task_logger):
    with patch.object(training_tasks, "AWSModelRegistry") as registry_class:
        with pytest.raises(ApprovalRequired):
            training_tasks.promote_model_in_aws_registry.fn(PACKAGE_ARN)
        registry_class.assert_not_called()


def test_promote_task_approves_when_approved(task_logger):
    with patch.object(training_tasks, "AWSModelRegistry") as registry_class:
        registry_class.return_value.update_model_approval_status.return_value = {
            "status": "success"
        }
        training_tasks.promote_model_in_aws_registry.fn(PACKAGE_ARN, approved=True)
        registry_class.return_value.update_model_approval_status.assert_called_once()


# --- Snowflake and dbt -------------------------------------------------------


def test_save_dataset_refuses_without_approval():
    with (
        patch.object(data_tasks, "create_snowflake_connection") as connect,
        patch.object(data_tasks, "write_pandas") as write,
    ):
        with pytest.raises(ApprovalRequired):
            data_tasks.save_dataset_to_snowflake_impl(pd.DataFrame({"a": [1]}), "X_TRAIN")
        connect.assert_not_called()
        write.assert_not_called()


def test_save_dataset_writes_when_approved():
    with (
        patch.object(data_tasks, "create_snowflake_connection") as connect,
        patch.object(data_tasks, "write_pandas", return_value=(True, 1, 1, [])) as write,
    ):
        assert data_tasks.save_dataset_to_snowflake_impl(
            pd.DataFrame({"a": [1]}), "X_TRAIN", approved=True
        )
        connect.assert_called_once()
        write.assert_called_once()


def test_dbt_build_refuses_without_approval(task_logger):
    with patch.object(data_tasks, "PrefectDbtRunner") as runner_class:
        with pytest.raises(ApprovalRequired):
            data_tasks.run_raw_to_staging_dbt_transformations.fn()
        runner_class.assert_not_called()


# --- Scripts -----------------------------------------------------------------


def test_apply_grants_forces_no_role_and_connects_nowhere_without_approval(monkeypatch):
    from scripts import apply_snowflake_grants

    monkeypatch.delenv("SNOWFLAKE_ROLE", raising=False)
    with patch("box_office.utils.snowflake_connection.create_snowflake_connection") as connect:
        with pytest.raises(ApprovalRequired):
            apply_snowflake_grants.apply_statements(["SELECT 1"])
        connect.assert_not_called()
    assert "SNOWFLAKE_ROLE" not in os.environ


def test_apply_grants_runs_statements_when_approved(monkeypatch):
    from scripts import apply_snowflake_grants

    monkeypatch.delenv("SNOWFLAKE_ROLE", raising=False)
    with patch("box_office.utils.snowflake_connection.create_snowflake_connection") as connect:
        cursor = connect.return_value.cursor.return_value
        cursor.fetchall.return_value = [("ok",)]
        try:
            exit_code = apply_snowflake_grants.apply_statements(["SELECT 1"], approved=True)
        finally:
            os.environ.pop("SNOWFLAKE_ROLE", None)
    assert exit_code == 0
    cursor.execute.assert_called_once_with("SELECT 1")


def test_dataset_loader_refuses_before_any_connection(tmp_path):
    from scripts import load_dataset_to_snowflake as loader_module

    with patch.object(loader_module, "create_snowflake_connection") as connect:
        with pytest.raises(ApprovalRequired):
            loader_module.DatasetLoader("BOX_OFFICE", "RAW").run(tmp_path / "missing.parquet")
        connect.assert_not_called()


def test_dataset_loader_replaces_table_when_approved(tmp_path):
    from scripts import load_dataset_to_snowflake as loader_module

    frame = pd.DataFrame({"tmdb_id": [1], "production_budget": [1.0]})
    with (
        patch.object(loader_module, "read_source_frame", return_value=frame),
        patch.object(loader_module, "build_spot_check_expectations", return_value={1: {}}),
        patch.object(loader_module, "prepare_for_load", return_value=frame),
        patch.object(loader_module, "build_staging_ddl", return_value="CREATE STAGING"),
        patch.object(loader_module.DatasetLoader, "_verify"),
        patch.object(loader_module, "create_snowflake_connection") as connect,
        patch("snowflake.connector.pandas_tools.write_pandas", return_value=(True, 1, 1, [])),
    ):
        loader_module.DatasetLoader("BOX_OFFICE", "RAW").run(tmp_path / "x.parquet", approved=True)
    statements = [
        c.args[0] for c in connect.return_value.cursor.return_value.execute.call_args_list
    ]
    assert any(s.startswith("CREATE OR REPLACE TABLE") for s in statements)


def test_register_local_model_cli_refuses_before_reading_artifacts(monkeypatch):
    from scripts import register_local_model

    monkeypatch.setattr("sys.argv", ["register_local_model.py"])
    with patch.object(register_local_model, "AWSModelRegistry") as registry_class:
        with pytest.raises(ApprovalRequired):
            register_local_model.main()
        registry_class.assert_not_called()


@pytest.mark.parametrize("flag, expected", [([], False), (["--approved"], True)])
def test_pipeline_cli_passes_approved_only_from_the_flag(monkeypatch, flag, expected):
    from box_office.orchestration.flows import ml_pipeline

    monkeypatch.setattr("sys.argv", ["box-office-pipeline", *flag])
    with patch.object(ml_pipeline, "ml_pipeline") as flow:
        ml_pipeline.main()
    assert flow.call_args.kwargs["approved"] is expected
