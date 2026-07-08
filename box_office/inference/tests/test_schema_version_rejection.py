"""The inference loader must reject artifacts with stale feature schemas."""

from unittest.mock import MagicMock, patch

import pytest

from box_office.inference.app.model_loader import ModelLoader
from box_office.ml.feature_schema import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    SCHEMA_VERSION_METADATA_KEY,
    FeatureSchemaVersionMismatch,
)


def _make_loader_skipping_aws() -> ModelLoader:
    with patch("boto3.client"):
        loader = ModelLoader.__new__(ModelLoader)
        loader.s3_client = MagicMock()
        loader.sagemaker_client = MagicMock()
        loader.cache_dir = MagicMock()
        loader._extracted_artifacts_cache = {}
        return loader


def _describe_response(schema_version: str | None) -> dict:
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
    loader = _make_loader_skipping_aws()
    loader.sagemaker_client.describe_model_package.return_value = _describe_response(
        None
    )

    with pytest.raises(FeatureSchemaVersionMismatch, match="feature_schema_version"):
        loader._download_and_load_model({"ModelPackageArn": "arn:aws:test"})


def test_rejects_artifact_with_old_schema_version():
    loader = _make_loader_skipping_aws()
    loader.sagemaker_client.describe_model_package.return_value = _describe_response(
        "1"
    )

    with pytest.raises(FeatureSchemaVersionMismatch, match="leaked"):
        loader._download_and_load_model({"ModelPackageArn": "arn:aws:test"})


def test_current_schema_version_is_v9():
    """Belt-and-suspenders: the contract is v9 (13-feature pre-release set)."""
    assert CURRENT_FEATURE_SCHEMA_VERSION == "9"
