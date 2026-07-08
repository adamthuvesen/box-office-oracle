"""Shared training-frame rules: staging shape, quality gate, v9 IP/franchise.

One home for the rules that turn cleaned movie rows into the v9 feature
contract, so the local driver (``scripts/prepare_training_frame.py``) and the
production data phase (``box_office/orchestration/phases/data_phase.py``)
apply the *same* logic — no drift between offline iteration and the SageMaker
run.

Two things legitimately differ between the two callers, and only these two:

- **IP_TIER source.** The local script reads the pre-generated
  ``data/generated/ip/ip_classification_1980_2026.parquet``; production runs
  ``box_office.ip_classification`` in-pipeline against the staging frame plus
  ``data/ip_rules.yml`` / ``data/collection_overrides.yml``. The classifier
  logic is identical; only where it runs differs.
- **Collection links.** The local script keys franchises off the raw backfill
  JSONL; production reads the ``COLLECTION_ID`` / ``COLLECTION_NAME`` columns
  the staging table carries. Both produce ``collection:<id>`` keys and both
  layer ``data/collection_overrides.yml`` on top.

Everything else — the uppercase column mapping, the ACTORS list-literal
shape, the quality-gate drop rules, the collection-keyed franchise features,
and the preprocessor contract check — lives here and is called by both.

Null production budgets stay NaN end to end: no imputation, no zero-fill.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from box_office.franchise_history import (
    DEFAULT_OVERRIDES_PATH,
    load_collection_overrides,
    prior_franchise_stats,
)

NO_IP_TIER = 5.0

# Columns the feature preprocessor consumes (uppercase staging shape).
PREPROCESSOR_INPUT_COLUMNS: tuple[str, ...] = (
    "RELEASE_YEAR",
    "RELEASE_DATE",
    "PRODUCTION_BUDGET",
    "RUNTIME",
    "MPAA",
    "GENRES",
    "DIRECTOR",
    "PRODUCTION_COMPANY",
    "ACTORS",
    "IP_TIER",
    "PRIOR_FRANCHISE_GROSS_LOG",
    "IS_FRANCHISE_FOLLOWUP",
)

# Source (lowercase) -> frame (uppercase) mapping. Metadata columns ride
# along for evaluation joins and variants; the training driver selects only
# PREPROCESSOR_INPUT_COLUMNS before feature engineering.
COLUMN_MAPPING: dict[str, str] = {
    "tmdb_id": "TMDB_ID",
    "imdb_id": "IMDB_ID",
    "title": "TITLE",
    "release_year": "RELEASE_YEAR",
    "release_date": "RELEASE_DATE",
    "production_budget": "PRODUCTION_BUDGET",
    "production_budget_source": "PRODUCTION_BUDGET_SOURCE",
    "production_budget_was_missing": "PRODUCTION_BUDGET_WAS_MISSING",
    "runtime": "RUNTIME",
    "mpaa": "MPAA",
    "genres": "GENRES",
    "director": "DIRECTOR",
    "production_company": "PRODUCTION_COMPANY",
    "actors": "ACTORS",
    "worldwide_gross": "WORLDWIDE_GROSS",
}

MIN_FEATURE_RUNTIME = 60
IMPLAUSIBLE_GROSS_FLOOR = 100_000_000
FIRST_NON_FINAL_YEAR = 2026
EXTREME_GROSS_BUDGET_MULTIPLIER = 50

DEFAULT_IP_RULES_PATH = Path("data/ip_rules.yml")


def actors_to_list_literal(value: object) -> str:
    """Convert a comma-separated actor string to a Python list-literal string.

    ``process_text_list`` parses list-literal strings via ``ast.literal_eval``;
    a plain comma-separated string would become one token (the whole cast as a
    single "actor"). ``repr`` handles quoting for names with apostrophes.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "[]"
    if not isinstance(value, str):
        raise TypeError(f"actors must be a string or NaN; got {type(value).__name__}")
    names = [name.strip() for name in value.split(",") if name.strip()]
    return repr(names)


def map_to_staging_columns(source: pd.DataFrame) -> pd.DataFrame:
    """Select and rename the source columns to the uppercase staging shape."""
    missing = [c for c in COLUMN_MAPPING if c not in source.columns]
    if missing:
        raise ValueError(f"source dataset missing expected columns: {missing}")
    frame = source[list(COLUMN_MAPPING)].rename(columns=COLUMN_MAPPING).copy()
    frame["ACTORS"] = frame["ACTORS"].map(actors_to_list_literal)
    return frame


def add_franchise_features(
    frame: pd.DataFrame,
    collection_map: dict[int, str],
    eligible_as_prior: pd.Series | None = None,
) -> pd.DataFrame:
    """Add the collection-keyed ``PRIOR_FRANCHISE_GROSS_LOG`` / ``IS_FRANCHISE_FOLLOWUP``.

    ``collection_map`` maps ``TMDB_ID`` -> franchise key (``collection:<id>``).
    Only strictly-earlier releases in the same collection contribute; a
    franchise's first film gets 0. Rows without a real ``RELEASE_DATE`` never
    count as priors (mirrors ``box_office/ip_classification.py``).

    ``eligible_as_prior`` (bool, aligned to ``frame.index``) further restricts
    which rows may CONTRIBUTE to other rows' priors — pass the quality-gate
    pass mask so known-bad gross artifacts (e.g. the gross-over-$100M-with-no-
    documented-budget rows) never inflate a kept row's franchise history.
    Every row still RECEIVES its own feature values regardless of the mask,
    so the dropped-rows sidecar keeps the v9 columns.
    """
    out = frame.copy()
    release_dates = pd.to_datetime(out["RELEASE_DATE"], errors="coerce")
    counts_as_prior = release_dates.notna()
    if eligible_as_prior is not None:
        counts_as_prior &= eligible_as_prior.astype(bool)
    prior = prior_franchise_stats(
        pd.DataFrame(
            {
                "franchise_key": out["TMDB_ID"].map(collection_map),
                "release_date": release_dates,
                "worldwide_gross": out["WORLDWIDE_GROSS"].astype(float),
                "counts_as_prior": counts_as_prior,
            },
            index=out.index,
        )
    )
    out["PRIOR_FRANCHISE_GROSS_LOG"] = np.log1p(prior["prior_gross"])
    out["IS_FRANCHISE_FOLLOWUP"] = (prior["prior_count"] > 0).astype(float)
    return out


def quality_gate_reasons(frame: pd.DataFrame) -> pd.Series:
    """Per-row drop reasons (';'-joined) or empty string for kept rows."""
    budget_missing = frame["PRODUCTION_BUDGET"].isna()
    rules = {
        "no_reliable_worldwide_gross": frame["WORLDWIDE_GROSS"].isna(),
        "runtime_under_60_non_feature": frame["RUNTIME"] < MIN_FEATURE_RUNTIME,
        "gross_over_100m_with_no_documented_budget": budget_missing
        & (frame["WORLDWIDE_GROSS"] > IMPLAUSIBLE_GROSS_FLOOR),
        "gross_not_final_future_year": frame["RELEASE_YEAR"] >= FIRST_NON_FINAL_YEAR,
    }
    reasons = pd.Series("", index=frame.index, dtype=str)
    for reason, mask in rules.items():
        reasons[mask] = reasons[mask].where(reasons[mask] == "", reasons[mask] + ";")
        reasons[mask] += reason
    return reasons


def flag_extreme_gross_multiplier(frame: pd.DataFrame) -> pd.Series:
    """The spec's original rule: gross > 50x known budget AND gross > $100M.

    Kept as a report-only flag: a hand check of every hit shows legitimate
    low-budget sleeper hits, so the rule is not used to drop rows.
    """
    budget = frame["PRODUCTION_BUDGET"]
    return (
        budget.notna()
        & (budget > 0)
        & (frame["WORLDWIDE_GROSS"] > EXTREME_GROSS_BUDGET_MULTIPLIER * budget)
        & (frame["WORLDWIDE_GROSS"] > IMPLAUSIBLE_GROSS_FLOOR)
    )


def apply_quality_gate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the frame into (kept, dropped-with-reason)."""
    reasons = quality_gate_reasons(frame)
    dropped = frame[reasons != ""].copy()
    dropped["DROP_REASON"] = reasons[reasons != ""]
    kept = frame[reasons == ""].reset_index(drop=True)
    return kept, dropped


def check_frame_against_preprocessor(frame: pd.DataFrame) -> None:
    """Fail loudly if the frame does not satisfy the v9 feature contract."""
    from box_office.ml.feature_pipeline.constants import SELECTED_FEATURES
    from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
    from box_office.ml.text_utils import process_text_list

    features = FeaturePreprocessorHigh().fit_transform(
        frame[list(PREPROCESSOR_INPUT_COLUMNS)]
    )
    if list(features.columns) != list(SELECTED_FEATURES):
        raise AssertionError(
            f"preprocessor output {list(features.columns)} != SELECTED_FEATURES"
        )
    if len(features) != len(frame):
        raise AssertionError("preprocessor changed the row count")

    budget_missing = frame["PRODUCTION_BUDGET"].isna().to_numpy()
    budget_derived = {"PRODUCTION_BUDGET", "LOG_BUDGET_X_HORROR", "LOG_BUDGET_X_COMEDY"}
    for col in features.columns:
        nan_rows = features[col].isna().to_numpy()
        if col in budget_derived:
            if (nan_rows & ~budget_missing).any():
                raise AssertionError(f"{col} has NaN outside missing-budget rows")
        elif nan_rows.any():
            raise AssertionError(f"non-budget feature {col} has NaN values")

    for col in ("IP_TIER", "PRIOR_FRANCHISE_GROSS_LOG", "IS_FRANCHISE_FOLLOWUP"):
        if frame[col].isna().any():
            raise AssertionError(f"{col} has NaN values in the training frame")
    if not frame["IP_TIER"].between(1, 5).all():
        raise AssertionError("IP_TIER outside the 1-5 ordinal range")

    multi_actor = frame.loc[frame["ACTORS"].str.contains("', '"), "ACTORS"]
    if multi_actor.empty:
        raise AssertionError("no multi-actor rows found; actors conversion is broken")
    parsed = process_text_list(multi_actor.iloc[0])
    if len(parsed) <= 1:
        raise AssertionError(f"multi-actor row parsed as {parsed}; expected > 1 names")


# ---------------------------------------------------------------------------
# Production path: staging frame -> v9 training frame (IP classified in-pipeline)
# ---------------------------------------------------------------------------

_STAGING_TO_MOVIE_COLUMNS: dict[str, str] = {
    "TMDB_ID": "tmdb_id",
    "IMDB_ID": "imdb_id",
    "TITLE": "title",
    "ORIGINAL_TITLE": "original_title",
    "RELEASE_YEAR": "release_year",
    "RELEASE_DATE": "release_date",
    "WORLDWIDE_GROSS": "worldwide_gross",
    "PRODUCTION_COMPANY": "production_company",
    "KEYWORDS": "keywords",
    "OVERVIEW": "overview",
    "TAGLINE": "tagline",
}


def _merged_collection_links(
    staging: pd.DataFrame, overrides_path: Path | None
) -> dict[int, tuple[int, str | None]]:
    """``TMDB_ID`` -> (collection_id, collection_name) from staging + overrides.

    The staging ``COLLECTION_ID`` / ``COLLECTION_NAME`` columns are the base;
    ``data/collection_overrides.yml`` always wins (same precedence as the
    offline ``collection_memberships`` merge).
    """
    links: dict[int, tuple[int, str | None]] = {}
    if "COLLECTION_ID" in staging.columns:
        names = (
            staging["COLLECTION_NAME"]
            if "COLLECTION_NAME" in staging.columns
            else pd.Series([None] * len(staging), index=staging.index)
        )
        for tmdb_id, cid, name in zip(
            staging["TMDB_ID"], staging["COLLECTION_ID"], names, strict=True
        ):
            if pd.notna(cid):
                links[int(tmdb_id)] = (int(cid), None if pd.isna(name) else name)

    if overrides_path is not None:
        for entry in load_collection_overrides(overrides_path):
            links[entry["tmdb_id"]] = (
                entry["collection_id"],
                entry["collection_name"],
            )
    return links


def _classify_ip_tiers(
    staging: pd.DataFrame,
    links: dict[int, tuple[int, str | None]],
    rules_path: Path,
) -> pd.Series:
    """Run ``box_office.ip_classification`` in-pipeline; return IP_TIER per row.

    Missing rows (no classification) fall back to ``NO_IP_TIER`` (5), exactly
    as the offline parquet join does.
    """
    from box_office.ip_classification import classify_movies, load_rules

    present = [c for c in _STAGING_TO_MOVIE_COLUMNS if c in staging.columns]
    movies = staging[present].rename(
        columns={c: _STAGING_TO_MOVIE_COLUMNS[c] for c in present}
    )
    raw_metadata = pd.DataFrame(
        {
            "tmdb_id": staging["TMDB_ID"].astype(int),
            "collection_id": staging["TMDB_ID"].map(
                {t: cid for t, (cid, _) in links.items()}
            ),
            "collection_name": staging["TMDB_ID"].map(
                {t: name for t, (_, name) in links.items()}
            ),
            "wikidata_id": None,
        }
    )
    classification, _audit = classify_movies(
        movies, raw_metadata, load_rules(rules_path)
    )
    tier = classification.drop_duplicates("tmdb_id").set_index("tmdb_id")["ip_tier"]
    return staging["TMDB_ID"].map(tier).fillna(NO_IP_TIER).astype(float)


def build_production_training_frame(
    staging: pd.DataFrame,
    *,
    rules_path: Path = DEFAULT_IP_RULES_PATH,
    overrides_path: Path | None = DEFAULT_OVERRIDES_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn the uppercase staging frame into the v9 training frame.

    Applies the same quality gate and v9 IP/franchise computation the local
    script uses, with IP classified in-pipeline. Returns ``(kept, dropped)``;
    ``kept`` carries ``PREPROCESSOR_INPUT_COLUMNS`` plus ``TMDB_ID`` and
    ``WORLDWIDE_GROSS`` (for the target and parity joins).

    ``ACTORS`` is expected comma-separated (raw source shape carried through
    Snowflake) and is converted to the list-literal shape here.
    """
    frame = staging.copy()
    frame["ACTORS"] = frame["ACTORS"].map(actors_to_list_literal)

    links = _merged_collection_links(frame, overrides_path)
    frame["IP_TIER"] = _classify_ip_tiers(frame, links, rules_path)
    collection_map = {t: f"collection:{cid}" for t, (cid, _) in links.items()}
    # Gate first: rows the quality gate drops (known-bad gross artifacts) must
    # not feed the franchise priors of kept rows. Future-year placeholder rows
    # are harmless either way (priors use strictly-earlier releases), but the
    # bad-gross drops would otherwise inflate priors.
    gate_pass = quality_gate_reasons(frame) == ""
    frame = add_franchise_features(frame, collection_map, eligible_as_prior=gate_pass)

    kept, dropped = apply_quality_gate(frame)
    keep_cols = list(
        dict.fromkeys(["TMDB_ID", *PREPROCESSOR_INPUT_COLUMNS, "WORLDWIDE_GROSS"])
    )
    kept = kept[[c for c in keep_cols if c in kept.columns]].copy()
    return kept, dropped
