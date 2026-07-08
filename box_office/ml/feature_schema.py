"""Feature schema version contract between training and inference.

Current runtime schema: v9, the 13-feature pre-release contract
(``SELECTED_FEATURES``): 10 engineered budget/genre/time/industry features
plus three pre-release IP/franchise inputs — ``IP_TIER`` (ordinal 1-5, 5 =
no pre-sold IP), ``PRIOR_FRANCHISE_GROSS_LOG`` (log1p of the same TMDB
collection's strictly-earlier worldwide gross), and
``IS_FRANCHISE_FOLLOWUP`` (1.0 when any strictly-earlier film exists in the
collection). The inference loader rejects artifacts whose metadata does not
match the runtime schema instead of serving a silent shape mismatch.
"""

CURRENT_FEATURE_SCHEMA_VERSION = "9"
SCHEMA_VERSION_METADATA_KEY = "feature_schema_version"


class FeatureSchemaVersionMismatch(Exception):
    """Raised when an artifact's feature schema version does not match the runtime."""


class FeatureContractMismatch(Exception):
    """Raised at inference when the preprocessor output and the model disagree.

    A loud, typed failure for the case where an artifact's components (scaler /
    model) expect a different feature width than the runtime pipeline produces,
    instead of a cryptic shape error from deep inside ``model.predict``.
    """
