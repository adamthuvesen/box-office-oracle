"""Feature schema version contract between training and inference.

Bumped to 2 with the leakage fix: artifacts produced before that fix used
``social_media_buzz`` (synthesized from ``worldwide_gross``) and
``production_budget`` imputed as ``0.4 * worldwide_gross`` plus four derived
features built on top. They are scientifically invalid and the inference loader
rejects them rather than silently serving leaked predictions.

Bumped to 3 when the pipeline was slimmed to the curated, decorrelated
``SELECTED_FEATURES`` set (see feature_pipeline/constants.py). Wider artifacts
(the old ~66-feature matrix) are rejected at load: their pickled scaler expects
a different feature width than the runtime pipeline now produces.
"""

CURRENT_FEATURE_SCHEMA_VERSION = "3"
SCHEMA_VERSION_METADATA_KEY = "feature_schema_version"


class FeatureSchemaVersionMismatch(Exception):
    """Raised when an artifact's feature schema version does not match the runtime."""


class FeatureContractMismatch(Exception):
    """Raised at inference when the preprocessor output and the model disagree.

    A loud, typed failure for the case where an artifact's components (scaler /
    model) expect a different feature width than the runtime pipeline produces,
    instead of a cryptic shape error from deep inside ``model.predict``.
    """
