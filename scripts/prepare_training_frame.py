"""Build the local training frame from the 1980-2026 TMDB rich backfill.

Reads ``data/generated/tmdb/rich_backfill_1980_2026/
tmdb_budget_wikipedia_5m_1980_2026.parquet`` and writes
``data/generated/training/train_frame_1980_2026.parquet`` in the staging
shape ``FeaturePreprocessorHigh`` expects (uppercase columns, ``ACTORS`` as
Python list-literal strings), plus:

- ``dropped_rows_1980_2026.csv`` — every filtered row with an explicit reason
  (no silent drops).
- ``flagged_kept_rows_1980_2026.csv`` — rows the spec's 50x gross/budget rule
  flags but that a hand check shows are legitimate sleeper hits (E.T., Get
  Out, The Blair Witch Project, ...). Kept in the frame; listed for audit.

Quality gate (rows dropped, all logged to the sidecar):

1. ``runtime < 60`` — non-feature releases (shorts, concert recordings).
2. missing production budget AND worldwide_gross > $100M — every such row in
   the dataset is a known gross artifact (e.g. tmdb 715904 "Metallica:
   WorldWired Tour" at $426.9M, tmdb 168626 "DeAD" at $201M). A real $100M+
   grosser always has a documented budget in TMDB/Wikipedia/Wikidata.
3. ``release_year >= 2026`` — future releases whose gross is a placeholder,
   not a final number.
4. missing ``worldwide_gross`` — source cleanup nulled an unsupported actual.

Curated source fixes run before the quality gate: unsupported financials are
nulled and non-movie rows are excluded with an audit sidecar.

IP/franchise contract-v9 columns are joined/computed here per row:
``IP_TIER`` from ``data/generated/ip/ip_classification_1980_2026.parquet``
(missing -> 5, no pre-sold IP), and collection-keyed
``PRIOR_FRANCHISE_GROSS_LOG`` / ``IS_FRANCHISE_FOLLOWUP`` from
``box_office.franchise_history`` over this frame's own rows (strictly
earlier releases in the same TMDB collection; first films get 0).

Null production budgets stay NaN end to end — no imputation, no zero-fill.

Run:  uv run python scripts/prepare_training_frame.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from box_office.franchise_history import (
    collection_franchise_keys,
    prior_franchise_stats,
)
from box_office.movie_data_quality import clean_movie_source_data

DATASET = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/"
    "tmdb_budget_wikipedia_5m_1980_2026.parquet"
)
OUTPUT_DIR = Path("data/generated/training")
FRAME_PATH = OUTPUT_DIR / "train_frame_1980_2026.parquet"
DROPPED_PATH = OUTPUT_DIR / "dropped_rows_1980_2026.csv"
FLAGGED_PATH = OUTPUT_DIR / "flagged_kept_rows_1980_2026.csv"
SOURCE_QUALITY_PATH = OUTPUT_DIR / "source_quality_rows_1980_2026.csv"
IP_CLASSIFICATION_PATH = Path("data/generated/ip/ip_classification_1980_2026.parquet")
RAW_JSONL_PATH = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/tmdb_rich_raw_5m_1980_2026.jsonl"
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


def add_ip_franchise_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Join IP_TIER and compute the collection-keyed franchise features.

    Runs on the full mapped frame (before the quality gate) so the dropped
    rows sidecar also carries the columns; scripts/score_all_movies.py
    scores those rows through the same v9 preprocessor input.
    """
    if not IP_CLASSIFICATION_PATH.exists():
        raise SystemExit(
            f"IP classification not found: {IP_CLASSIFICATION_PATH}. "
            "Run `uv run python scripts/classify_ip.py` first."
        )
    ip = pd.read_parquet(IP_CLASSIFICATION_PATH).drop_duplicates("tmdb_id")
    out = frame.copy()
    out["IP_TIER"] = (
        out["TMDB_ID"]
        .map(ip.set_index("tmdb_id")["ip_tier"])
        .fillna(NO_IP_TIER)
        .astype(float)
    )

    collection_map = collection_franchise_keys(RAW_JSONL_PATH)
    release_dates = pd.to_datetime(out["RELEASE_DATE"], errors="coerce")
    prior = prior_franchise_stats(
        pd.DataFrame(
            {
                "franchise_key": out["TMDB_ID"].map(collection_map),
                "release_date": release_dates,
                "worldwide_gross": out["WORLDWIDE_GROSS"].astype(float),
                # Rows without a real release_date never count as priors
                # (mirrors box_office/ip_classification.py).
                "counts_as_prior": release_dates.notna(),
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


def main() -> None:
    if not DATASET.exists():
        raise SystemExit(f"dataset not found: {DATASET}")

    source_raw = pd.read_parquet(DATASET)
    source, source_quality = clean_movie_source_data(source_raw)
    frame = add_ip_franchise_features(map_to_staging_columns(source))
    kept, dropped = apply_quality_gate(frame)

    flagged = kept[flag_extreme_gross_multiplier(kept)]

    check_frame_against_preprocessor(kept)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(FRAME_PATH, index=False)
    dropped.to_csv(DROPPED_PATH, index=False)
    flagged.to_csv(FLAGGED_PATH, index=False)
    source_quality.to_csv(SOURCE_QUALITY_PATH, index=False)

    print(f"source rows: {len(source_raw)}")
    print(f"source rows after curated cleanup: {len(source)}")
    print(f"kept rows:   {len(kept)} -> {FRAME_PATH}")
    print(f"dropped:     {len(dropped)} -> {DROPPED_PATH}")
    print(f"source quality fixes: {len(source_quality)} -> {SOURCE_QUALITY_PATH}")
    for reason, count in (
        dropped["DROP_REASON"].str.split(";").explode().value_counts().items()
    ):
        print(f"  {reason}: {count}")
    print(
        f"flagged (50x rule, kept after hand check): {len(flagged)} -> {FLAGGED_PATH}"
    )
    print(
        f"null budgets kept as NaN: {int(kept['PRODUCTION_BUDGET'].isna().sum())} rows",
        file=sys.stderr,
    )
    tier_counts = kept["IP_TIER"].astype(int).value_counts().sort_index().to_dict()
    print(f"IP_TIER counts (kept rows): {tier_counts}")
    print(
        f"franchise follow-ups (kept rows): {int(kept['IS_FRANCHISE_FOLLOWUP'].sum())}"
    )


if __name__ == "__main__":
    main()
