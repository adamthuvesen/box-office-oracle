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
earlier releases in the same TMDB collection; first films get 0). Rows the
quality gate drops never contribute to other rows' priors — their gross is
exactly what the gate distrusts — though they still get their own values
for the sidecar.

Null production budgets stay NaN end to end — no imputation, no zero-fill.

Run:  uv run python scripts/prepare_training_frame.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from box_office.franchise_history import collection_franchise_keys
from box_office.movie_data_quality import clean_movie_source_data

# The staging shape, quality gate, franchise features, and preprocessor check
# are shared with the production data phase (box_office/training_frame.py) so
# the two paths cannot drift. Re-exported names keep this script's public
# surface (and tests/test_prepare_training_frame.py) unchanged.
from box_office.training_frame import (  # noqa: F401
    COLUMN_MAPPING,
    NO_IP_TIER,
    PREPROCESSOR_INPUT_COLUMNS,
    actors_to_list_literal,
    add_franchise_features,
    apply_quality_gate,
    check_frame_against_preprocessor,
    flag_extreme_gross_multiplier,
    map_to_staging_columns,
    quality_gate_reasons,
)

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


def add_ip_franchise_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Join IP_TIER from the local parquet, then the collection-keyed franchise
    features (shared with the production path).

    Runs on the full mapped frame (before the quality gate) so the dropped
    rows sidecar also carries the columns; scripts/score_all_movies.py
    scores those rows through the same v9 preprocessor input. Rows the gate
    drops still RECEIVE their own values but never CONTRIBUTE to other rows'
    priors — a known-bad gross artifact must not inflate a kept row's
    franchise history. Production classifies IP in-pipeline instead of
    reading this parquet
    (box_office.training_frame.build_production_training_frame); the franchise
    computation is the same shared function.
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
    gate_pass = quality_gate_reasons(out) == ""
    return add_franchise_features(
        out,
        collection_franchise_keys(RAW_JSONL_PATH),
        eligible_as_prior=gate_pass,
    )


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
