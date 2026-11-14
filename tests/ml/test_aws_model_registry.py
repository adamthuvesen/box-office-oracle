"""Registering a model package keeps the training-job link next to the other metadata."""

from unittest.mock import patch

from box_office.ml.feature_schema import SCHEMA_VERSION_METADATA_KEY
from tests.fakes import FakeModelRegistry

TRAINING_JOB_ARN = "arn:aws:sagemaker:eu-north-1:123456789012:training-job/job-1"


def test_registered_package_carries_training_job_arn_with_metadata() -> None:
    registry = FakeModelRegistry()
    registry.sagemaker_mock.describe_training_job.return_value = {
        "TrainingJobArn": TRAINING_JOB_ARN
    }
    registry.sagemaker_mock.create_model_package.return_value = {"ModelPackageArn": "arn:pkg"}

    with patch(
        "box_office.ml.model_registry.aws_model_registry._compute_sha256_of_s3_object",
        return_value=("a" * 64, 1024),
    ):
        registry.register_model_package(
            "group",
            "s3://bucket/model.tar.gz",
            inference_specification={"Containers": [{}]},
            metadata={"training_job_name": "job-1"},
            training_job_name="job-1",
            approved=True,
        )

    sent = registry.sagemaker_mock.create_model_package.call_args.kwargs[
        "CustomerMetadataProperties"
    ]
    assert sent["training_job_arn"] == TRAINING_JOB_ARN
    assert sent["training_job_name"] == "job-1"
    assert sent["sha256"] == "a" * 64
    assert SCHEMA_VERSION_METADATA_KEY in sent
