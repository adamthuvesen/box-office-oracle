"""Pipeline error handling: every catch site either re-raises on
unexpected errors or catches a narrow type with a documented benign-path log.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError


# --------------------------------------------------------------------------- #
# 1.8 — _parse_s3_uri
# --------------------------------------------------------------------------- #


class TestParseS3Uri:
    """Task 1.7 / 1.8: `s3://bucket/key` parser used by model_training.py."""

    def test_valid_uri_returns_bucket_and_key(self):
        from box_office.utils.aws_helpers import parse_s3_uri as _parse_s3_uri

        assert _parse_s3_uri("s3://my-bucket/path/to/key.tar.gz") == (
            "my-bucket",
            "path/to/key.tar.gz",
        )

    def test_missing_scheme_raises(self):
        from box_office.utils.aws_helpers import parse_s3_uri as _parse_s3_uri

        with pytest.raises(ValueError, match="bucket/key"):
            _parse_s3_uri("bucket/key")

    def test_wrong_scheme_raises(self):
        from box_office.utils.aws_helpers import parse_s3_uri as _parse_s3_uri

        with pytest.raises(ValueError, match="https://"):
            _parse_s3_uri("https://bucket/key")

    def test_missing_bucket_raises(self):
        from box_office.utils.aws_helpers import parse_s3_uri as _parse_s3_uri

        with pytest.raises(ValueError):
            _parse_s3_uri("s3:///just-a-key")

    def test_empty_string_raises(self):
        from box_office.utils.aws_helpers import parse_s3_uri as _parse_s3_uri

        with pytest.raises(ValueError):
            _parse_s3_uri("")


# --------------------------------------------------------------------------- #
# 1.6 — snowflake_loader counting helpers re-raise on connection errors
# --------------------------------------------------------------------------- #


def _make_loader_with_cursor(cursor_exc: Exception):
    """Build a SnowflakeLoader whose cursor raises `cursor_exc` on execute."""
    from box_office.utils.snowflake_loader import SnowflakeLoader

    loader = SnowflakeLoader.__new__(SnowflakeLoader)
    loader.database = "BOX_OFFICE"
    loader.schema = "RAW"

    cursor = MagicMock()
    cursor.execute.side_effect = cursor_exc
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []

    @contextmanager_no_io
    def fake_cursor_ctx():
        yield cursor, MagicMock()

    loader._snowflake_cursor = fake_cursor_ctx
    return loader


def contextmanager_no_io(fn):
    """Local stand-in to avoid importing contextlib at module top."""
    from contextlib import contextmanager

    return contextmanager(fn)


class _TableNotFound(Exception):
    """Stand-in for snowflake.connector.errors.ProgrammingError errno 002003."""

    def __init__(self):
        super().__init__(
            "SQL compilation error: Object 'X' does not exist or not authorized."
        )
        self.errno = 2003


class _AuthError(Exception):
    """Stand-in for an auth/network error — must propagate."""

    def __init__(self):
        super().__init__("250001 Could not connect to Snowflake backend.")
        self.errno = 250001


class TestSnowflakeLoaderErrorClassification:
    def test_get_current_count_returns_zero_on_table_not_found(self):
        loader = _make_loader_with_cursor(_TableNotFound())
        assert loader.get_current_count("BOX_OFFICE_V3") == 0

    def test_get_current_count_reraises_on_auth_error(self):
        loader = _make_loader_with_cursor(_AuthError())
        with pytest.raises(_AuthError):
            loader.get_current_count("BOX_OFFICE_V3")

    def test_get_existing_tmdb_ids_returns_empty_on_table_not_found(self):
        loader = _make_loader_with_cursor(_TableNotFound())
        assert loader.get_existing_tmdb_ids("BOX_OFFICE_V3") == set()

    def test_get_existing_tmdb_ids_reraises_on_auth_error(self):
        loader = _make_loader_with_cursor(_AuthError())
        with pytest.raises(_AuthError):
            loader.get_existing_tmdb_ids("BOX_OFFICE_V3")


# --------------------------------------------------------------------------- #
# 1.12 — aws_model_registry classification + raise on hard failure
# --------------------------------------------------------------------------- #


def _client_error(code: str, message: str = "") -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": message}},
        operation_name="TestOp",
    )


class TestModelRegistryErrorClassification:
    def _make_registry(self):
        from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry

        registry = AWSModelRegistry.__new__(AWSModelRegistry)
        registry.region_name = "eu-north-1"
        registry.sagemaker_client = MagicMock()
        registry.s3_client = MagicMock()
        return registry

    def test_create_group_treats_validation_does_not_exist_as_create_path(self):
        registry = self._make_registry()
        registry.sagemaker_client.describe_model_package_group.side_effect = (
            _client_error(
                "ValidationException", "ModelPackageGroup foo does not exist."
            )
        )
        registry.sagemaker_client.create_model_package_group.return_value = {
            "ModelPackageGroupArn": "arn:aws:sagemaker:eu-north-1:123:model-package-group/foo"
        }

        result = registry.create_model_package_group("foo")
        assert result["status"] == "created"

    def test_create_group_propagates_access_denied(self):
        registry = self._make_registry()
        registry.sagemaker_client.describe_model_package_group.side_effect = (
            _client_error("AccessDeniedException", "Not authorized.")
        )
        with pytest.raises(ClientError):
            registry.create_model_package_group("foo")

    def test_register_model_package_raises_on_hard_failure(self):
        registry = self._make_registry()
        # Skip the SHA256 download path by patching it.
        with patch(
            "box_office.ml.model_registry.aws_model_registry._compute_sha256_of_s3_object",
            return_value=("a" * 64, 1024),
        ):
            registry.sagemaker_client.create_model_package.side_effect = _client_error(
                "ThrottlingException", "Rate exceeded"
            )
            with pytest.raises(ClientError) as excinfo:
                registry.register_model_package(
                    model_package_group_name="foo",
                    model_data_url="s3://bucket/key.tar.gz",
                )
            # Cause chain preserved (the `raise` re-raises the same exception).
            assert excinfo.value.response["Error"]["Code"] == "ThrottlingException"

    def test_get_account_id_propagates_sts_failure(self):
        registry = self._make_registry()
        with patch(
            "box_office.ml.model_registry.aws_model_registry.boto3.client"
        ) as mock_client:
            sts = MagicMock()
            sts.get_caller_identity.side_effect = _client_error(
                "ExpiredToken", "Security token expired"
            )
            mock_client.return_value = sts
            with pytest.raises(ClientError):
                registry._get_account_id()


# --------------------------------------------------------------------------- #
# 1.2 / 1.4 — training-task error classification
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_sagemaker_client():
    sm = MagicMock()
    sm.region = "eu-north-1"
    sm.s3_bucket = "test-bucket"
    sm.s3_prefix = "test-prefix"
    return sm


@pytest.fixture
def fake_estimator():
    est = MagicMock()
    est.latest_training_job.name = "job-123"
    est.model_data = "s3://test-bucket/test-prefix/output/job-123/output/model.tar.gz"
    return est


class TestTrainingTaskErrorClassification:
    """The Prefect tasks are imported as plain functions via `.fn`."""

    def _patched_boto(
        self, monkeypatch, s3_side_effect=None, sagemaker_side_effect=None
    ):
        """Patch boto3.client to return mocks whose chosen call raises."""
        from box_office.orchestration.tasks import training_tasks

        s3 = MagicMock()
        if s3_side_effect:
            s3.download_file.side_effect = s3_side_effect

        sm = MagicMock()
        if sagemaker_side_effect:
            sm.describe_training_job.side_effect = sagemaker_side_effect

        cw = MagicMock()

        def client_factory(service, **kwargs):
            return {"s3": s3, "sagemaker": sm, "cloudwatch": cw}[service]

        monkeypatch.setattr(training_tasks.boto3, "client", client_factory)
        return s3, sm

    def test_missing_tarball_returns_empty_metrics(
        self, monkeypatch, fake_sagemaker_client, fake_estimator
    ):
        from box_office.orchestration.tasks import training_tasks

        self._patched_boto(
            monkeypatch,
            s3_side_effect=_client_error(
                "NoSuchKey", "The specified key does not exist."
            ),
            sagemaker_side_effect=_client_error(
                "ValidationException", "TrainingJob job-123 does not exist."
            ),
        )

        # Bypass Prefect's run-context requirement.
        monkeypatch.setattr(training_tasks, "get_run_logger", lambda: MagicMock())

        result = training_tasks.download_and_analyze_results.fn(
            fake_estimator, fake_sagemaker_client
        )
        assert result.get("training_duration_seconds", None) is None
        # NoSuchKey was the *only* swallowed error; we got an empty dict back, not a crash.

    def test_unexpected_s3_error_propagates(
        self, monkeypatch, fake_sagemaker_client, fake_estimator
    ):
        from box_office.orchestration.tasks import training_tasks

        self._patched_boto(
            monkeypatch,
            s3_side_effect=_client_error("AccessDenied", "Access Denied"),
        )
        monkeypatch.setattr(training_tasks, "get_run_logger", lambda: MagicMock())

        with pytest.raises(ClientError) as excinfo:
            training_tasks.download_and_analyze_results.fn(
                fake_estimator, fake_sagemaker_client
            )
        assert excinfo.value.response["Error"]["Code"] == "AccessDenied"

    def test_throttled_describe_propagates(
        self, monkeypatch, fake_sagemaker_client, fake_estimator, tmp_path
    ):
        """Even when the tarball download succeeds, a throttled describe must surface."""
        from box_office.orchestration.tasks import training_tasks

        # Build a valid tarball so the s3 download_file path completes.
        tar_path = tmp_path / "output.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            results_path = tmp_path / "cv_results.json"
            results_path.write_text(
                json.dumps(
                    {
                        "cv_scores": [0.1, 0.2],
                        "mean_cv_mae": 0.5,
                        "std_cv_mae": 0.05,
                        "mean_best_iteration": 100,
                    }
                )
            )
            tar.add(results_path, arcname="cv_results.json")

        s3 = MagicMock()

        def fake_download(bucket, key, dest):
            Path(dest).write_bytes(tar_path.read_bytes())

        s3.download_file.side_effect = fake_download

        sm = MagicMock()
        sm.describe_training_job.side_effect = _client_error(
            "ThrottlingException", "Rate exceeded"
        )
        cw = MagicMock()

        def client_factory(service, **kwargs):
            return {"s3": s3, "sagemaker": sm, "cloudwatch": cw}[service]

        monkeypatch.setattr(training_tasks.boto3, "client", client_factory)
        monkeypatch.setattr(training_tasks, "get_run_logger", lambda: MagicMock())

        with pytest.raises(ClientError) as excinfo:
            training_tasks.download_and_analyze_results.fn(
                fake_estimator, fake_sagemaker_client
            )
        assert excinfo.value.response["Error"]["Code"] == "ThrottlingException"
