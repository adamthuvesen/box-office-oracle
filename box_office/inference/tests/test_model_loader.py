"""
Unit tests for ModelLoader class.

Tests model loading, caching, validation, and registry integration
functionality with comprehensive mocking for AWS services.
"""

import os
import pickle
import tempfile
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from botocore.exceptions import ClientError, NoCredentialsError

# Import the module under test
from box_office.inference.app.model_loader import (
    RegistryModelInfo,
    ModelLoader,
    ModelLoadError,
    ModelValidationError,
)


class MockXGBModel:
    """Mock XGBoost model class that can be pickled."""

    def __init__(self):
        self.n_estimators = 100
        self.learning_rate = 0.1

    def predict(self, X):
        return [100000, 200000, 150000]

    def get_params(self):
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate}


class MockSklearnModel:
    """Mock scikit-learn model that reports as fitted (the production state)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return [1, 2, 3]

    def __sklearn_is_fitted__(self) -> bool:
        return True


class TestModelLoader:
    """Test cases for ModelLoader class."""

    @pytest.fixture
    def mock_aws_clients(self):
        """Mock AWS clients for testing."""
        with patch("boto3.client") as mock_client:
            mock_s3 = Mock()
            mock_sagemaker = Mock()

            def client_factory(service_name, **kwargs):
                if service_name == "s3":
                    return mock_s3
                elif service_name == "sagemaker":
                    return mock_sagemaker
                return Mock()

            mock_client.side_effect = client_factory
            yield mock_s3, mock_sagemaker

    @pytest.fixture
    def mock_model_registry(self):
        """Mock AWSModelRegistry for testing."""
        with patch(
            "box_office.inference.app.model_loader.AWSModelRegistry"
        ) as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            yield mock_registry

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def sample_model_info(self):
        """Sample model information from registry."""
        return {
            "ModelPackageArn": "arn:aws:sagemaker:eu-north-1:123456789012:model-package/test-group/1",
            "ModelPackageStatus": "Completed",
            "ModelPackageGroupName": "test-group",
            "ModelPackageVersion": 1,
            "CreationTime": datetime.now(timezone.utc),
            "ModelApprovalStatus": "Approved",
        }

    @pytest.fixture
    def sample_model_package_details(self):
        """Sample model package details from SageMaker.

        Includes a placeholder ``sha256`` in ``CustomerMetadataProperties``
        so the integrity gate doesn't reject the package. Tests that actually
        exercise the verification path overwrite this with the real digest.
        """
        return {
            "ModelPackageArn": "arn:aws:sagemaker:eu-north-1:123456789012:model-package/test-group/1",
            "ModelPackageStatus": "Completed",
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": "test-image",
                        "ModelDataUrl": "s3://test-bucket/models/model.tar.gz",
                        "Framework": "XGBOOST",
                        "FrameworkVersion": "1.7-1",
                    }
                ]
            },
            "CustomerMetadataProperties": {
                "sha256": "placeholder-overridden-by-tests-that-actually-download",
                "size_bytes": "0",
                "feature_schema_version": "2",
            },
        }

    @pytest.fixture
    def sample_xgboost_model(self):
        """Create a simple test model for testing."""
        model = MockXGBModel()
        model.__class__.__name__ = "XGBRegressor"
        return model

    def test_init_success(self, mock_aws_clients, temp_cache_dir):
        """Test successful ModelLoader initialization."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader(
            model_package_group_name="test-group",
            aws_region="eu-north-1",
            cache_dir=temp_cache_dir,
            cache_ttl_seconds=1800,
        )

        assert loader.model_package_group_name == "test-group"
        assert loader.aws_region == "eu-north-1"
        assert loader.cache_dir == Path(temp_cache_dir)
        assert loader.cache_ttl_seconds == 1800
        assert loader.s3_client == mock_s3
        assert loader.sagemaker_client == mock_sagemaker
        assert loader._current_model is None
        assert loader._current_model_info is None
        assert loader._last_load_time is None

    def test_init_no_credentials(self):
        """Test initialization failure with no AWS credentials."""
        with patch("boto3.client", side_effect=NoCredentialsError()):
            with pytest.raises(ModelLoadError, match="AWS credentials not configured"):
                ModelLoader("test-group")

    def test_init_aws_error(self):
        """Test initialization failure with AWS error."""
        with patch("boto3.client", side_effect=Exception("AWS connection failed")):
            with pytest.raises(
                ModelLoadError, match="Failed to initialize AWS clients"
            ):
                ModelLoader("test-group")

    def test_get_latest_approved_model_info_with_registry(
        self, mock_aws_clients, mock_model_registry, temp_cache_dir, sample_model_info
    ):
        """Test getting latest approved model info using model registry."""
        mock_s3, mock_sagemaker = mock_aws_clients
        mock_model_registry.get_latest_approved_model.return_value = sample_model_info

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        result = loader._get_latest_approved_model_info()

        assert result == sample_model_info
        mock_model_registry.get_latest_approved_model.assert_called_once_with(
            "test-group"
        )

    def test_model_loader_requires_registry(self, mock_aws_clients, temp_cache_dir):
        """ModelLoader always uses AWSModelRegistry (no list_model_packages fallback)."""
        mock_s3, mock_sagemaker = mock_aws_clients
        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        assert loader.model_registry is not None

    def test_get_latest_approved_model_info_not_found(
        self, mock_aws_clients, mock_model_registry, temp_cache_dir
    ):
        """Test getting model info when no approved model exists."""
        mock_s3, mock_sagemaker = mock_aws_clients
        mock_model_registry.get_latest_approved_model.return_value = None

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        result = loader._get_latest_approved_model_info()

        assert result is None

    def test_get_latest_approved_model_info_validation_error(
        self, mock_aws_clients, mock_model_registry, temp_cache_dir
    ):
        """Test handling of validation error when model group doesn't exist."""
        mock_s3, mock_sagemaker = mock_aws_clients
        error_response = {"Error": {"Code": "ValidationException"}}
        mock_model_registry.get_latest_approved_model.side_effect = ClientError(
            error_response, "ListModelPackages"
        )

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        result = loader._get_latest_approved_model_info()

        assert result is None

    def _build_model_tar(self, sample_xgboost_model):
        """Build a tarball containing model.pkl + the two preprocessor stubs the
        loader requires; return ``(tar_path, sha256)``."""
        import hashlib

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_tar:
            tar_path = temp_tar.name

        with tarfile.open(tar_path, "w:gz") as tar:
            for name in ("model.pkl", "feature_preprocessor.pkl", "feature_scaler.pkl"):
                with tempfile.NamedTemporaryFile(
                    suffix=".pkl", delete=False
                ) as temp_pkl:
                    pickle.dump(sample_xgboost_model, temp_pkl)
                    temp_pkl_path = temp_pkl.name
                tar.add(temp_pkl_path, arcname=name)
                os.unlink(temp_pkl_path)

        digest = hashlib.sha256(Path(tar_path).read_bytes()).hexdigest()
        return tar_path, digest

    def test_download_and_load_model_success(
        self,
        mock_aws_clients,
        temp_cache_dir,
        sample_model_info,
        sample_model_package_details,
        sample_xgboost_model,
    ):
        """Loader verifies SHA256 then unpickles the model."""
        from box_office.inference.app.integrity import reset_verified_cache

        reset_verified_cache()

        mock_s3, mock_sagemaker = mock_aws_clients
        tar_path, digest = self._build_model_tar(sample_xgboost_model)
        try:
            details = dict(sample_model_package_details)
            details["CustomerMetadataProperties"] = {
                "sha256": digest,
                "size_bytes": str(Path(tar_path).stat().st_size),
                "feature_schema_version": "2",
            }
            mock_sagemaker.describe_model_package.return_value = details

            def mock_download_file(bucket, key, filename):
                import shutil

                shutil.copy2(tar_path, filename)

            mock_s3.download_file.side_effect = mock_download_file

            loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
            result = loader._download_and_load_model(sample_model_info)

            assert result is not None
            assert hasattr(result, "predict")
            mock_sagemaker.describe_model_package.assert_called_once()
            mock_s3.download_file.assert_called_once()
        finally:
            os.unlink(tar_path)

    def test_download_and_load_model_sha256_mismatch_blocks_load(
        self,
        mock_aws_clients,
        temp_cache_dir,
        sample_model_info,
        sample_model_package_details,
        sample_xgboost_model,
    ):
        """A wrong sha256 in CustomerMetadataProperties must raise ArtifactIntegrityError before unpickling."""
        from box_office.inference.app.integrity import (
            ArtifactIntegrityError,
            reset_verified_cache,
        )

        reset_verified_cache()

        mock_s3, mock_sagemaker = mock_aws_clients
        tar_path, _digest = self._build_model_tar(sample_xgboost_model)
        try:
            details = dict(sample_model_package_details)
            details["CustomerMetadataProperties"] = {
                "sha256": "0" * 64,
                "size_bytes": "0",
                "feature_schema_version": "2",
            }
            mock_sagemaker.describe_model_package.return_value = details

            def mock_download_file(bucket, key, filename):
                import shutil

                shutil.copy2(tar_path, filename)

            mock_s3.download_file.side_effect = mock_download_file

            loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
            with pytest.raises(ArtifactIntegrityError):
                loader._download_and_load_model(sample_model_info)
        finally:
            os.unlink(tar_path)

    def test_download_and_load_model_missing_manifest_blocks_load(
        self,
        mock_aws_clients,
        temp_cache_dir,
        sample_model_info,
        sample_model_package_details,
    ):
        """A model package with no sha256 manifest must raise ArtifactIntegrityError."""
        from box_office.inference.app.integrity import ArtifactIntegrityError

        mock_s3, mock_sagemaker = mock_aws_clients
        details = dict(sample_model_package_details)
        details["CustomerMetadataProperties"] = {}  # no sha256
        mock_sagemaker.describe_model_package.return_value = details

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        with pytest.raises(ArtifactIntegrityError):
            loader._download_and_load_model(sample_model_info)
        # S3 download must NOT be attempted when the manifest is missing.
        mock_s3.download_file.assert_not_called()

    def test_download_and_load_model_no_containers(
        self, mock_aws_clients, temp_cache_dir, sample_model_info
    ):
        """Test model loading failure when no containers in specification."""
        mock_s3, mock_sagemaker = mock_aws_clients
        mock_sagemaker.describe_model_package.return_value = {
            "ModelPackageArn": sample_model_info["ModelPackageArn"],
            "InferenceSpecification": {"Containers": []},
        }

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)

        with pytest.raises(ModelLoadError, match="No containers found"):
            loader._download_and_load_model(sample_model_info)

    def test_download_and_load_model_no_model_url(
        self, mock_aws_clients, temp_cache_dir, sample_model_info
    ):
        """Test model loading failure when no model data URL."""
        mock_s3, mock_sagemaker = mock_aws_clients
        mock_sagemaker.describe_model_package.return_value = {
            "ModelPackageArn": sample_model_info["ModelPackageArn"],
            "InferenceSpecification": {
                "Containers": [{"Image": "test-image"}]  # Missing ModelDataUrl
            },
        }

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)

        with pytest.raises(ModelLoadError, match="No model data URL found"):
            loader._download_and_load_model(sample_model_info)

    def test_download_and_load_model_invalid_s3_url(
        self, mock_aws_clients, temp_cache_dir, sample_model_info
    ):
        """Test model loading failure with invalid S3 URL."""
        mock_s3, mock_sagemaker = mock_aws_clients
        mock_sagemaker.describe_model_package.return_value = {
            "ModelPackageArn": sample_model_info["ModelPackageArn"],
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": "test-image",
                        "ModelDataUrl": "http://invalid-url/model.tar.gz",
                    }
                ]
            },
        }

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)

        with pytest.raises(ModelLoadError, match="Invalid S3 URL format"):
            loader._download_and_load_model(sample_model_info)

    def test_download_and_load_model_s3_error(
        self,
        mock_aws_clients,
        temp_cache_dir,
        sample_model_info,
        sample_model_package_details,
    ):
        """Test model loading failure with S3 download error."""
        mock_s3, mock_sagemaker = mock_aws_clients
        mock_sagemaker.describe_model_package.return_value = (
            sample_model_package_details
        )

        error_response = {"Error": {"Code": "NoSuchKey"}}
        mock_s3.download_file.side_effect = ClientError(error_response, "GetObject")

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)

        with pytest.raises(ModelLoadError, match="Model file not found in S3"):
            loader._download_and_load_model(sample_model_info)

    def test_validate_model_success(
        self, mock_aws_clients, temp_cache_dir, sample_model_info, sample_xgboost_model
    ):
        """Test successful model validation."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        loader._validate_model(sample_xgboost_model, sample_model_info)

    def test_validate_model_none_object(
        self, mock_aws_clients, temp_cache_dir, sample_model_info
    ):
        """Test model validation failure with None model object."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)

        with pytest.raises(ModelValidationError, match="Model object is None"):
            loader._validate_model(None, sample_model_info)

    def test_validate_model_missing_required_fields(
        self, mock_aws_clients, temp_cache_dir, sample_xgboost_model
    ):
        """Test model validation failure with missing required fields."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        invalid_info = {"ModelPackageArn": "test-arn"}

        with pytest.raises(ModelValidationError, match="Missing required field"):
            loader._validate_model(sample_xgboost_model, invalid_info)

    def test_validate_model_wrong_status(
        self, mock_aws_clients, temp_cache_dir, sample_xgboost_model, sample_model_info
    ):
        """Test model validation failure with wrong package status."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        sample_model_info["ModelPackageStatus"] = "Failed"

        with pytest.raises(
            ModelValidationError, match="Model package status is not 'Completed'"
        ):
            loader._validate_model(sample_xgboost_model, sample_model_info)

    def test_validate_xgboost_model(
        self, mock_aws_clients, temp_cache_dir, sample_xgboost_model
    ):
        """Test XGBoost-specific model validation."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        loader._validate_xgboost_model(sample_xgboost_model)
        params = sample_xgboost_model.get_params()
        assert "n_estimators" in params

    def test_validate_sklearn_model(self, mock_aws_clients, temp_cache_dir):
        """A fitted sklearn-compatible model passes validation."""
        mock_s3, mock_sagemaker = mock_aws_clients

        sklearn_model = MockSklearnModel()

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        loader._validate_sklearn_model(sklearn_model)

    def test_validate_sklearn_model_rejects_unfitted(
        self, mock_aws_clients, temp_cache_dir
    ):
        """An unfitted estimator is rejected at load, not at first request."""
        from sklearn.linear_model import LinearRegression

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        with pytest.raises(ModelValidationError, match="not fitted"):
            loader._validate_sklearn_model(LinearRegression())

    def test_validate_model_rejects_object_without_predict(
        self, mock_aws_clients, temp_cache_dir
    ):
        """A model artifact with no predict method cannot serve inference."""
        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        model_info = {"ModelPackageArn": "arn", "ModelPackageStatus": "Completed"}
        with pytest.raises(ModelValidationError, match="predict"):
            loader._validate_model({"not": "a model"}, model_info)

    def test_validate_model_rejects_unfitted_wrapper(
        self, mock_aws_clients, temp_cache_dir
    ):
        """A wrapper that reports is_fitted=False is rejected at load time."""

        class UnfittedWrapper:
            is_fitted = False

            def predict(self, X):  # pragma: no cover - never reached
                raise AssertionError("unfitted model should not predict")

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        model_info = {"ModelPackageArn": "arn", "ModelPackageStatus": "Completed"}
        with pytest.raises(ModelValidationError, match="is_fitted"):
            loader._validate_model(UnfittedWrapper(), model_info)

    def test_md5_cache_methods_removed(self, mock_aws_clients, temp_cache_dir):
        """The legacy MD5-keyed disk cache must no longer exist on the loader."""
        mock_s3, mock_sagemaker = mock_aws_clients
        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)

        for attr in (
            "_cache_model",
            "_get_cached_model",
            "_generate_cache_key",
            "_remove_cache_files",
        ):
            assert not hasattr(
                loader, attr
            ), f"{attr} should be removed from ModelLoader"

    def test_clear_cache_removes_sha_dirs(self, mock_aws_clients, temp_cache_dir):
        mock_s3, mock_sagemaker = mock_aws_clients
        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)

        sha_dir = loader.cache_dir / ("a" * 64)
        sha_dir.mkdir()
        (sha_dir / "cached_model.pkl").write_bytes(b"x" * 16)

        loader.clear_cache()
        assert not sha_dir.exists()

    def test_get_cache_info_counts_sha_dirs(self, mock_aws_clients, temp_cache_dir):
        mock_s3, mock_sagemaker = mock_aws_clients
        loader = ModelLoader(
            "test-group", cache_dir=temp_cache_dir, cache_ttl_seconds=3600
        )

        info = loader.get_cache_info()
        assert info["cached_models"] == 0
        assert info["total_cache_size_bytes"] == 0
        assert info["cache_ttl_seconds"] == 3600

        sha_dir = loader.cache_dir / ("b" * 64)
        sha_dir.mkdir()
        (sha_dir / "cached_model.pkl").write_bytes(b"y" * 32)

        info = loader.get_cache_info()
        assert info["cached_models"] == 1
        assert info["total_cache_size_bytes"] >= 32

    def test_get_current_model_none(self, mock_aws_clients, temp_cache_dir):
        """Test getting current model when none is loaded."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)

        assert loader.get_current_model() is None

    def test_is_model_cache_valid_no_model(self, mock_aws_clients, temp_cache_dir):
        """Test cache validity check when no model is loaded."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)

        assert loader.is_model_cache_valid() is False

    def test_is_model_cache_valid_expired(self, mock_aws_clients, temp_cache_dir):
        """Test cache validity check with expired cache."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader(
            "test-group", cache_dir=temp_cache_dir, cache_ttl_seconds=1
        )
        loader._last_load_time = datetime.now(timezone.utc) - timedelta(seconds=2)

        assert loader.is_model_cache_valid() is False

    def test_is_model_cache_valid_fresh(self, mock_aws_clients, temp_cache_dir):
        """Test cache validity check with fresh cache."""
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader(
            "test-group", cache_dir=temp_cache_dir, cache_ttl_seconds=3600
        )
        loader._last_load_time = datetime.now(timezone.utc)

        assert loader.is_model_cache_valid() is True

    @patch(
        "box_office.inference.app.model_loader.ModelLoader._get_latest_approved_model_info"
    )
    @patch("box_office.inference.app.model_loader.ModelLoader._download_and_load_model")
    @patch("box_office.inference.app.model_loader.ModelLoader._validate_model")
    def test_load_latest_approved_model_success(
        self,
        mock_validate,
        mock_download,
        mock_get_info,
        mock_aws_clients,
        temp_cache_dir,
        sample_model_info,
        sample_xgboost_model,
    ):
        """End-to-end happy path through ``load_latest_approved_model``."""
        mock_s3, mock_sagemaker = mock_aws_clients

        mock_get_info.return_value = sample_model_info
        mock_download.return_value = sample_xgboost_model

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        result = loader.load_latest_approved_model()

        assert result is not None
        assert loader._current_model == sample_xgboost_model
        assert loader._current_model_info == sample_model_info
        assert loader._last_load_time is not None

        mock_get_info.assert_called_once()
        mock_download.assert_called_once_with(sample_model_info)
        mock_validate.assert_called_once_with(sample_xgboost_model, sample_model_info)

    def test_load_latest_approved_model_no_approved(
        self, mock_aws_clients, temp_cache_dir
    ):
        """When the registry returns no approved model, the loader returns None."""
        mock_s3, mock_sagemaker = mock_aws_clients

        with patch.object(
            ModelLoader, "_get_latest_approved_model_info", return_value=None
        ):
            loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
            result = loader.load_latest_approved_model()
            assert result is None

    def test_refresh_model_if_needed_cache_valid(
        self, mock_aws_clients, temp_cache_dir, sample_model_info
    ):
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        loader._current_model_info = sample_model_info
        loader._last_load_time = datetime.now(timezone.utc)

        with patch.object(
            loader, "_get_latest_approved_model_info", return_value=sample_model_info
        ):
            result = loader.refresh_model_if_needed()
            assert result is False

    def test_refresh_model_if_needed_new_version(
        self, mock_aws_clients, temp_cache_dir, sample_model_info
    ):
        mock_s3, mock_sagemaker = mock_aws_clients

        loader = ModelLoader("test-group", cache_dir=temp_cache_dir)
        loader._current_model_info = sample_model_info
        loader._last_load_time = datetime.now(timezone.utc)

        new_model_info = sample_model_info.copy()
        new_model_info["ModelPackageArn"] = (
            "arn:aws:sagemaker:eu-north-1:123456789012:model-package/test-group/2"
        )

        with patch.object(
            loader, "_get_latest_approved_model_info", return_value=new_model_info
        ):
            with patch.object(
                loader,
                "load_latest_approved_model",
                return_value=RegistryModelInfo(new_model_info),
            ) as mock_load:
                result = loader.refresh_model_if_needed()
                assert result is True
                mock_load.assert_called_once()


class TestModelLoaderIntegration:
    """Integration tests for ModelLoader with more realistic scenarios."""

    def test_full_model_loading_workflow(self, tmp_path):
        """Build a real tarball, hash it, advertise that hash via the manifest,
        and verify the loader reads + verifies + extracts + loads it."""
        import hashlib

        from box_office.inference.app.integrity import reset_verified_cache

        reset_verified_cache()

        # The loader validates the model object (must expose predict); the
        # preprocessor and scaler are plain artifacts.
        model_obj = MockXGBModel()
        artifact_data = {"type": "test_artifact", "version": "1.0"}
        artifacts = {
            "model.pkl": model_obj,
            "feature_preprocessor.pkl": artifact_data,
            "feature_scaler.pkl": artifact_data,
        }
        for name, obj in artifacts.items():
            with open(tmp_path / name, "wb") as f:
                pickle.dump(obj, f)

        tar_file = tmp_path / "model.tar.gz"
        with tarfile.open(tar_file, "w:gz") as tar:
            for name in ("model.pkl", "feature_preprocessor.pkl", "feature_scaler.pkl"):
                tar.add(tmp_path / name, arcname=name)

        digest = hashlib.sha256(tar_file.read_bytes()).hexdigest()

        sample_model_info = {
            "ModelPackageArn": "arn:aws:sagemaker:eu-north-1:123456789012:model-package/test-group/1",
            "ModelPackageStatus": "Completed",
            "ModelApprovalStatus": "Approved",
        }

        sample_package_details = {
            "ModelPackageArn": sample_model_info["ModelPackageArn"],
            "InferenceSpecification": {
                "Containers": [{"ModelDataUrl": "s3://test-bucket/models/model.tar.gz"}]
            },
            "CustomerMetadataProperties": {
                "sha256": digest,
                "size_bytes": str(tar_file.stat().st_size),
                "feature_schema_version": "2",
            },
        }

        with patch("boto3.client") as mock_client:
            mock_s3 = Mock()
            mock_sagemaker = Mock()

            def client_factory(service_name, **kwargs):
                if service_name == "s3":
                    return mock_s3
                elif service_name == "sagemaker":
                    return mock_sagemaker
                return Mock()

            mock_client.side_effect = client_factory
            mock_sagemaker.describe_model_package.return_value = sample_package_details

            def mock_download_file(bucket, key, filename):
                import shutil

                shutil.copy2(tar_file, filename)

            mock_s3.download_file.side_effect = mock_download_file

            with patch(
                "box_office.inference.app.model_loader.AWSModelRegistry"
            ) as mock_registry_class:
                mock_registry = Mock()
                mock_registry.get_latest_approved_model.return_value = sample_model_info
                mock_registry_class.return_value = mock_registry

                cache_dir = tmp_path / "cache"
                loader = ModelLoader("test-group", cache_dir=str(cache_dir))
                result = loader.load_latest_approved_model()

                assert result is not None
                assert loader.get_current_model() is not None
                assert isinstance(loader._current_model, MockXGBModel)

                cache_info = loader.get_cache_info()
                assert cache_info["cached_models"] == 1
                assert cache_info["current_model_loaded"] is True
