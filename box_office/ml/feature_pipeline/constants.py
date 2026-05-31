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

# Multiplicative interactions: missing input means "no interaction" → fill 0.
# Ratios: missing denominator must surface, not silently become 0.
_INTERACTION_FILL_ZERO = {
    "BUDGET_SEASONAL_BOOST",
    "SUMMER_BUDGET_INTERACTION",
    "HOLIDAY_BUDGET_INTERACTION",
    "WEEKEND_RATING_BOOST",
    "COVID_BUDGET_IMPACT",
    "COVID_RATING_IMPACT",
    "COVID_VOTES_IMPACT",
    "FRANCHISE_STRENGTH",
    "FRANCHISE_BUDGET_CONFIDENCE",
    "BLOCKBUSTER_BUDGET_MULTIPLIER",
    "ACTION_BUDGET_INTERACTION",
    "COMEDY_BUDGET_EFFICIENCY",
    "DIRECTOR_BUDGET_CONFIDENCE",
    "STAR_POWER_PREMIUM",
}
_INTERACTION_KEEP_NAN = {"HORROR_LOW_BUDGET_ADVANTAGE"}
