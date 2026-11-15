"""The inference loader must reject artifacts with stale feature schemas."""

from typing import Any

import pytest

from box_office.ml.feature_schema import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    SCHEMA_VERSION_METADATA_KEY,
    FeatureSchemaVersionMismatch,
)
from tests.inference.fakes import FakeModelLoader


def _describe_response(schema_version: str | None) -> dict[str, Any]:
    customer_meta: dict[str, str] = {"sha256": "a" * 64, "size_bytes": "1234"}
    if schema_version is not None:
        customer_meta[SCHEMA_VERSION_METADATA_KEY] = schema_version
    return {
        "InferenceSpecification": {
            "Containers": [{"ModelDataUrl": "s3://bucket/model.tar.gz"}],
        },
        "CustomerMetadataProperties": customer_meta,
    }


def test_rejects_artifact_without_schema_version():
    loader = FakeModelLoader()
    loader.sagemaker_mock.describe_model_package.return_value = _describe_response(None)

    with pytest.raises(FeatureSchemaVersionMismatch, match="feature_schema_version"):
        loader._download_and_load_model({"ModelPackageArn": "arn:aws:test"})


def test_rejects_artifact_with_old_schema_version():
    loader = FakeModelLoader()
    loader.sagemaker_mock.describe_model_package.return_value = _describe_response("1")

    with pytest.raises(FeatureSchemaVersionMismatch, match="leaked"):
        loader._download_and_load_model({"ModelPackageArn": "arn:aws:test"})


def test_current_schema_version_is_v9():
    """Belt-and-suspenders: the contract is v9 (13-feature pre-release set)."""
    assert CURRENT_FEATURE_SCHEMA_VERSION == "9"
