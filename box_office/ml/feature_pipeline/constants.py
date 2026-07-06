"""Shared constants for the feature pipeline: raw-input drop lists, column groups, and interaction specs."""

from __future__ import annotations

# Raw input columns dropped by the final ``_SelectEngineered`` step.
# Core numerical columns are not in this list; ``CoreNumericalTransformer``
# claims them as engineered, type-coerced output.
RAW_INPUT_COLUMNS_TO_DROP: tuple[str, ...] = (
    "RANK",
    "MOVIE_RANK",
    "RELEASE_DATE",
    "RATING",
    "VOTES",
    "MPAA",
    "GENRES",
    "DIRECTOR",
    "PRODUCTION_COMPANY",
    "ACTORS",
    "DOMESTIC_GROSS",
    "FRANCHISE_RATING",
    "SOCIAL_MEDIA_BUZZ",
    # String columns the staging table carries that no transformer reads.
    # Without dropping them here they reach the scaler and explode on the first
    # IMDb id ("tt14454876") it tries to coerce to float.
    "IMDB_ID",
    "TITLE",
    "ORIGINAL_LANGUAGE",
    "PRODUCTION_COUNTRIES",
    "OVERVIEW",
    "TAGLINE",
    "KEYWORDS",
    "LOADED_AT",
    "RELEASE_TYPE",
)

CORE_NUMERICAL_FEATURES: tuple[str, ...] = (
    "RELEASE_YEAR",
    "AD_BUDGET",
    "PRODUCTION_BUDGET",
    "RUNTIME",
)

# Canonical feature contract enforced by ``FeatureSelector`` (the final pipeline
# step) and asserted training↔serving. Changing this list is a feature-contract
# change: bump CURRENT_FEATURE_SCHEMA_VERSION and retrain. Names must match what
# the transformers emit exactly (note ``GENRE_<name>`` keeps a lowercase suffix).
SELECTED_FEATURES: tuple[str, ...] = (
    # Budget, time, release disruption
    "PRODUCTION_BUDGET",
    "IS_COVID_ERA",
    "GENRE_drama",
    "RELEASE_YEAR",
    # Budget x genre interactions
    "LOG_TOTAL_BUDGET_X_HORROR",
    "SUPER_GENRE_ENCODED",
    "LOG1P_LEAD_ACTOR_FREQ",
    "AD_TO_PROD_RATIO",
    "RUNTIME",
    "LOG_TOTAL_BUDGET_X_COMEDY",
    "AD_BUDGET",
    "COMPANY_FREQ",
)

GENRE_VOCABULARY: tuple[str, ...] = (
    "action",
    "comedy",
    "drama",
    "adventure",
    "thriller",
    "horror",
    "romance",
    "science_fiction",
    "animation",
    "fantasy",
    "crime",
    "family",
    "mystery",
    "war",
    "western",
)
