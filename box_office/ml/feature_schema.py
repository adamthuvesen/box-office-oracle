"""Feature schema version contract between training and inference.

Current runtime schema: v7, the 12-feature pre-release contract.
The inference loader rejects artifacts whose metadata does not match the
runtime schema instead of serving a silent shape mismatch.
"""

CURRENT_FEATURE_SCHEMA_VERSION = "7"
SCHEMA_VERSION_METADATA_KEY = "feature_schema_version"


class FeatureSchemaVersionMismatch(Exception):
    """Raised when an artifact's feature schema version does not match the runtime."""


class FeatureContractMismatch(Exception):
    """Raised at inference when the preprocessor output and the model disagree.

    A loud, typed failure for the case where an artifact's components (scaler /
    model) expect a different feature width than the runtime pipeline produces,
    instead of a cryptic shape error from deep inside ``model.predict``.
    """
