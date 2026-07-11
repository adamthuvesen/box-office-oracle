"""Tests for AWS model registry components.

Bare imports on purpose: a broken registry module must fail at collection
time, not silently skip.
"""

import unittest
from unittest.mock import MagicMock, patch

from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry


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
