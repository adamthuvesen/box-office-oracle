"""Tests for AWS model registry components.

Bare imports on purpose: a broken registry module must fail at collection
time, not silently skip.
"""

import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry
from box_office.ml.model_registry.metadata import ModelMetadata


class TestModelMetadata(unittest.TestCase):
    """Test ModelMetadata essential functionality."""

    def test_metadata_creation_and_validation(self):
        metadata = ModelMetadata(
            model_id="model_001",
            version=1,
            training_job_name="test-job",
            model_artifacts_path="s3://test-bucket/model",
            hyperparameters={"n_estimators": 100},
            status="development",
            metrics={"mae": 12.5, "r2": 0.85},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        self.assertEqual(metadata.model_id, "model_001")
        self.assertEqual(metadata.status, "development")
        self.assertEqual(metadata.metrics["mae"], 12.5)

    def test_status_validation(self):
        with self.assertRaises(ValueError):
            ModelMetadata(
                model_id="model_001",
                version=1,
                training_job_name="test-job",
                model_artifacts_path="s3://test-bucket/model",
                hyperparameters={},
                status="invalid_status",  # Invalid
                metrics={},
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )


class TestAWSModelRegistry(unittest.TestCase):
    """Test AWS Model Registry functionality."""

    @patch("boto3.client")
    def test_aws_registry_initialization(self, mock_boto_client):
        mock_sagemaker = MagicMock()
        mock_boto_client.return_value = mock_sagemaker

        registry = AWSModelRegistry(region_name="eu-north-1")

        # Constructor also creates an S3 client for SHA256 manifests, so use
        # assert_any_call rather than assert_called_once_with.
        self.assertIsNotNone(registry.sagemaker_client)
        mock_boto_client.assert_any_call("sagemaker", region_name="eu-north-1")


if __name__ == "__main__":
    unittest.main()
