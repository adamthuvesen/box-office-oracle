"""Registry and feature-schema constants shared by training and inference."""

from box_office.config import model_registry_group_name
from box_office.ml.feature_schema import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    SCHEMA_VERSION_METADATA_KEY,
    FeatureSchemaVersionMismatch,
)

__all__ = [
    "CURRENT_FEATURE_SCHEMA_VERSION",
    "SCHEMA_VERSION_METADATA_KEY",
    "FeatureSchemaVersionMismatch",
    "model_registry_group_name",
]
