"""Feature schema version contract between training and inference.

Bumped to 2 with the leakage fix: artifacts produced before that fix used
``social_media_buzz`` (synthesized from ``worldwide_gross``) and
``production_budget`` imputed as ``0.4 * worldwide_gross`` plus four derived
features built on top. They are scientifically invalid and the inference loader
rejects them rather than silently serving leaked predictions.
"""

CURRENT_FEATURE_SCHEMA_VERSION = "2"
SCHEMA_VERSION_METADATA_KEY = "feature_schema_version"


class FeatureSchemaVersionMismatch(Exception):
    """Raised when an artifact's feature schema version does not match the runtime."""
