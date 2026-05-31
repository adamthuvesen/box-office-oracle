"""Registry and feature-schema constants shared by training and inference."""

from box_office.ml.feature_schema import (
    CURRENT_FEATURE_SCHEMA_VERSION,
    SCHEMA_VERSION_METADATA_KEY,
    FeatureSchemaVersionMismatch,
)
from box_office.shared.names import model_registry_group_name

__all__ = [
    "CURRENT_FEATURE_SCHEMA_VERSION",
    "SCHEMA_VERSION_METADATA_KEY",
    "FeatureSchemaVersionMismatch",
    "model_registry_group_name",
]
