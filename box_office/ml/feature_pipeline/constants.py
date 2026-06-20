"""Shared constants for the feature pipeline: raw-input drop lists, column groups, and interaction specs."""

from __future__ import annotations

# Raw input columns dropped by the final ``_SelectEngineered`` step.
# Core numerical columns are NOT in this list — the ``CoreNumericalTransformer``
# claims them as engineered output (type-coerced versions), matching how the
# old ``FeaturePreprocessorHigh`` reported them.
RAW_INPUT_COLUMNS_TO_DROP: tuple[str, ...] = (
    "RELEASE_DATE",
    "MPAA",
    "GENRES",
    "DIRECTOR",
    "PRODUCTION_COMPANY",
    "ACTORS",
    # Removed in the leakage fix; if a stale upstream still passes it, drop here
    # so the output matrix stays canonical.
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
    "RATING",
    "VOTES",
    "AD_BUDGET",
    "PRODUCTION_BUDGET",
    "FRANCHISE_RATING",
    "RUNTIME",
)

# Canonical feature contract enforced by ``FeatureSelector`` (the final pipeline
# step) and asserted training↔serving. A compact, decorrelated (max |Spearman|
# = 0.57), axis-balanced subset chosen by analysis/feature_selection_study.py,
# then tightened by the depth-3/drop-COVID challenger evaluation. Changing this
# list is a feature-contract change — bump CURRENT_FEATURE_SCHEMA_VERSION and
# retrain. Names must match what the transformers emit exactly (note
# ``GENRE_<name>`` keeps a lowercase suffix).
SELECTED_FEATURES: tuple[str, ...] = (
    # Demand, budget, marketing, IP, time
    "VOTES",
    "PRODUCTION_BUDGET",
    "AD_TO_PROD_RATIO",
    "FRANCHISE_RATING",
    "RELEASE_YEAR",
    # Content rating + studio track record
    "MPAA_ENCODED",
    "COMPANY_FREQ",
    # Genre
    "GENRE_action",
    "GENRE_comedy",
    "SUPER_GENRE_ENCODED",
    # Release-window seasonality
    "IS_JULY_4TH_WEEKEND",
    "IS_WEEKEND_RELEASE",
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
