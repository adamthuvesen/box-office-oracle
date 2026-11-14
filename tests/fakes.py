"""Typed test doubles for the AWS-facing classes.

Each fake subclasses the real class and skips its ``__init__`` (which opens AWS sessions), so a
test passes it wherever the real type is expected. The ``*_mock`` attributes expose the
``MagicMock`` behind each client with its mock type, for stubbing and call assertions.
"""

from unittest.mock import MagicMock

from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry
from box_office.sagemaker.sagemaker_train_job import SageMakerClient


class FakeSageMakerClient(SageMakerClient):
    def __init__(self) -> None:
        self.session_mock = MagicMock()
        self.region = "eu-north-1"
        self.role = "arn:aws:iam::123456789012:role/test"
        self.boto_session = MagicMock()
        self.sagemaker_client = MagicMock()
        self.s3_client = MagicMock()
        self.sagemaker_session = self.session_mock
        self.s3_bucket = "bucket"
        self.s3_prefix = "prefix"


class FakeModelRegistry(AWSModelRegistry):
    def __init__(self) -> None:
        self.sagemaker_mock = MagicMock()
        self.s3_mock = MagicMock()
        self.region_name = "eu-north-1"
        self.sagemaker_client = self.sagemaker_mock
        self.s3_client = self.s3_mock
