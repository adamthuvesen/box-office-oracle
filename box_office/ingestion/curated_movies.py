"""Ensure hand-curated must-have movies exist in the local budget dataset.

The TMDB ``/discover`` sweep in ``tmdb_rich_backfill`` has paging holes, so
top-grossing films can be absent from the canonical parquet
(``data/generated/tmdb/rich_backfill_1980_2026/tmdb_budget_wikipedia_5m_1980_2026.parquet``).
This module reads ``data/curated_movies.yml``, fetches any listed movie that
is missing, and appends it — raw JSONL record plus a flat row shaped exactly
like the rest of the dataset — provided it clears the inclusion bar
(>= $5M worldwide gross, >= 60min runtime). Additions are idempotent: a
tmdb_id already in the JSONL or parquet is skipped.

Runs after ``wikipedia_budget_fill`` (the target parquet carries the budget
provenance columns).

Run:  uv run box-office-curated-movies
Needs TMDB_API_TOKEN (see .env).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from box_office.ingestion.tmdb_rich_backfill import (
    TMDB_API_URL,
    BackfillConfig,
    BackfillState,
    extract_release_year,
    flat_record_from_raw,
    load_raw_tmdb_ids,
    raw_record_from_payload,
    request_json,
    utc_now,
    write_jsonl_record,
)
from box_office.movie_data_quality import clean_movie_source_data
from box_office.utils.env_setup import configure_environment

logger = logging.getLogger(__name__)

DEFAULT_CURATED_PATH = Path("data/curated_movies.yml")

DATASET_DIR = Path("data/generated/tmdb/rich_backfill_1980_2026")
SOURCE_PARQUET = DATASET_DIR / "tmdb_budget_wikipedia_5m_1980_2026.parquet"
SOURCE_CSV = DATASET_DIR / "tmdb_budget_wikipedia_5m_1980_2026.csv"
RAW_JSONL = DATASET_DIR / "tmdb_rich_raw_5m_1980_2026.jsonl"
AUDIT_MANIFEST = DATASET_DIR / "curated_movies_manifest_1980_2026.json"

# Columns the budget parquet adds on top of the 37-column flat schema.
BUDGET_EXTRA_COLUMNS = (
    "production_budget_original",
    "production_budget_source",
    "production_budget_was_missing",
    "wikidata_budget_usd",
    "wikidata_item",
    "wikidata_label",
)

MIN_RUNTIME_MINUTES = 60


@dataclass(frozen=True)
class CuratedMovie:
    tmdb_id: int
    title: str
    # Documented production budget (USD), used only when TMDB returns 0/blank.
    documented_budget: int | None = None


def load_curated_movies(path: Path = DEFAULT_CURATED_PATH) -> list[CuratedMovie]:
    """Read ``data/curated_movies.yml``. Malformed entries raise loudly."""
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("movies") or []
    movies: list[CuratedMovie] = []
    seen: set[int] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or "tmdb_id" not in entry
            or "title" not in entry
        ):
            raise ValueError(f"malformed curated movie entry: {entry!r}")
        tmdb_id = int(entry["tmdb_id"])
        if tmdb_id in seen:
            raise ValueError(f"duplicate curated tmdb_id: {tmdb_id}")
        seen.add(tmdb_id)
        budget = entry.get("documented_budget")
        movies.append(
            CuratedMovie(
                tmdb_id=tmdb_id,
                title=str(entry["title"]),
                documented_budget=int(budget) if budget is not None else None,
            )
        )
    if not movies:
        raise ValueError(f"no curated movies found in {path}")
    return movies


def build_flat_row(
    raw: dict[str, Any],
    target_columns: list[str],
    *,
    documented_budget: int | None = None,
) -> dict[str, Any]:
    """Shape one raw record into a full budget-parquet row.

    Base fields come from ``flat_record_from_raw`` (identical list-string
    formatting to every other row); the budget-provenance columns are filled
    from the TMDB budget, falling back to a documented value only when TMDB
    reports nothing.
    """
    flat = flat_record_from_raw(raw)
    tmdb_budget = flat.get("production_budget")

    if tmdb_budget is not None and float(tmdb_budget) > 0:
        budget = int(tmdb_budget)
        flat["production_budget"] = float(budget)
        flat["production_budget_original"] = budget
        flat["production_budget_source"] = "tmdb"
        flat["production_budget_was_missing"] = False
    elif documented_budget is not None and documented_budget > 0:
        budget = int(documented_budget)
        flat["production_budget"] = float(budget)
        flat["production_budget_original"] = budget
        flat["production_budget_source"] = "documented"
        flat["production_budget_was_missing"] = False
    else:
        flat["production_budget"] = None
        flat["production_budget_original"] = 0
        flat["production_budget_source"] = "missing"
        flat["production_budget_was_missing"] = True

    flat["wikidata_budget_usd"] = None
    flat["wikidata_item"] = None
    flat["wikidata_label"] = None

    missing = [column for column in target_columns if column not in flat]
    if missing:
        raise ValueError(f"row is missing target columns: {missing}")
    return {column: flat[column] for column in target_columns}


def append_rows(
    df: pd.DataFrame, rows: list[dict[str, Any]], target_columns: list[str]
) -> pd.DataFrame:
    """Append rows and restore the original per-column dtypes.

    ``pd.concat`` can upcast (e.g. int64 -> float64 when a column gains a
    NaN); we re-cast every column back to the source dtype so the parquet
    schema is byte-for-byte compatible with what dbt already expects.
    """
    if not rows:
        return df
    dtypes = df.dtypes.to_dict()
    additions = pd.DataFrame(rows, columns=target_columns)
    combined = pd.concat([df, additions], ignore_index=True)
    for column, dtype in dtypes.items():
        combined[column] = combined[column].astype(dtype)
    return combined


def validate_row(row: dict[str, Any]) -> None:
    """Run the row through the source-quality expectations.

    A well-formed theatrical row must survive ``clean_movie_source_data``
    without being excluded, and (for a row we vouch for) without having its
    financials corrected.
    """
    frame = pd.DataFrame([row])
    cleaned, audit = clean_movie_source_data(frame)
    if cleaned.empty:
        raise ValueError(
            f"tmdb {row['tmdb_id']} was excluded by clean_movie_source_data: "
            f"{audit.to_dict(orient='records')}"
        )
    if audit.empty:
        return
    corrected = audit[audit["action"].str.contains("set_worldwide_gross", na=False)]
    if not corrected.empty:
        raise ValueError(
            f"tmdb {row['tmdb_id']} had its gross corrected away: "
            f"{corrected.to_dict(orient='records')}"
        )


def fetch_raw_record(
    session: requests.Session,
    tmdb_id: int,
    config: BackfillConfig,
    state: BackfillState,
) -> dict[str, Any]:
    payload = request_json(
        session,
        f"{TMDB_API_URL}/movie/{tmdb_id}",
        params={
            "append_to_response": ",".join(config.append_responses),
            "include_image_language": "en,null",
        },
        config=config,
        state=state,
    )
    year = extract_release_year(payload.get("release_date")) or config.start_year
    return raw_record_from_payload(payload, year=year, page=0)


def add_missing_movies(
    df: pd.DataFrame,
    target_columns: list[str],
    candidates: list[CuratedMovie],
    config: BackfillConfig,
    *,
    write_jsonl: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch and append any candidate not already in JSONL/parquet.

    Returns the updated frame, the rows added, and rows skipped for not
    clearing the inclusion bar.
    """
    raw_ids = load_raw_tmdb_ids(config.raw_jsonl_path)
    parquet_ids = set(df["tmdb_id"].astype(int))
    state = BackfillState(started_at=utc_now())
    session = requests.Session()

    new_rows: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for candidate in candidates:
        tmdb_id = candidate.tmdb_id
        if tmdb_id in parquet_ids:
            logger.info("present already: tmdb %s (%s)", tmdb_id, candidate.title)
            continue

        logger.warning(
            "MISSING from parquet: tmdb %s (%s) -- fetching",
            tmdb_id,
            candidate.title,
        )
        raw = fetch_raw_record(session, tmdb_id, config, state)
        payload = raw.get("payload") or {}
        gross = int(payload.get("revenue") or 0)
        runtime = int(payload.get("runtime") or 0)
        title = payload.get("title")

        if gross < config.min_revenue or runtime < MIN_RUNTIME_MINUTES:
            logger.warning(
                "REJECTED tmdb %s (%s): gross=%s runtime=%s below inclusion bar",
                tmdb_id,
                title,
                gross,
                runtime,
            )
            rejected.append(
                {
                    "tmdb_id": tmdb_id,
                    "title": title,
                    "worldwide_gross": gross,
                    "runtime": runtime,
                }
            )
            continue

        if write_jsonl and tmdb_id not in raw_ids:
            write_jsonl_record(config.raw_jsonl_path, raw)
            raw_ids.add(tmdb_id)
            logger.warning("appended raw JSONL record: tmdb %s (%s)", tmdb_id, title)

        row = build_flat_row(
            raw, target_columns, documented_budget=candidate.documented_budget
        )
        validate_row(row)
        new_rows.append(row)
        added.append(
            {
                "tmdb_id": tmdb_id,
                "title": title,
                "release_year": row["release_year"],
                "worldwide_gross": gross,
                "runtime": runtime,
                "production_budget": row["production_budget"],
                "production_budget_source": row["production_budget_source"],
                "collection_id": (payload.get("belongs_to_collection") or {}).get("id"),
            }
        )
        logger.warning(
            "ADDED tmdb %s (%s): gross=%s budget=%s collection=%s",
            tmdb_id,
            title,
            gross,
            row["production_budget"],
            added[-1]["collection_id"],
        )
        parquet_ids.add(tmdb_id)

    df = append_rows(df, new_rows, target_columns)
    return df, added, rejected


def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["release_year", "worldwide_gross", "tmdb_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def write_manifest(
    path: Path,
    *,
    added: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    row_count_before: int,
    row_count_after: int,
) -> None:
    manifest = {
        "created_at": utc_now(),
        "source_parquet": str(SOURCE_PARQUET),
        "source_csv": str(SOURCE_CSV),
        "raw_jsonl": str(RAW_JSONL),
        "row_count_before": row_count_before,
        "row_count_after": row_count_after,
        "rows_added": len(added),
        "added": added,
        "rejected_below_inclusion_bar": rejected,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch curated must-have movies missing from the budget parquet."
    )
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED_PATH)
    parser.add_argument("--parquet", type=Path, default=SOURCE_PARQUET)
    parser.add_argument("--csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--raw-jsonl", type=Path, default=RAW_JSONL)
    parser.add_argument("--manifest", type=Path, default=AUDIT_MANIFEST)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report, but do not write the parquet/csv/manifest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args(argv)

    config = BackfillConfig(output_dir=args.parquet.parent)
    if args.raw_jsonl != config.raw_jsonl_path:
        raise SystemExit(
            f"raw JSONL path mismatch: {args.raw_jsonl} != {config.raw_jsonl_path}"
        )

    candidates = load_curated_movies(args.curated)
    logger.info("loaded %d curated movies from %s", len(candidates), args.curated)

    df = pd.read_parquet(args.parquet)
    row_count_before = len(df)
    logger.info("loaded %d rows from %s", row_count_before, args.parquet)

    target_columns = list(df.columns)
    df, added, rejected = add_missing_movies(
        df, target_columns, candidates, config, write_jsonl=not args.dry_run
    )

    df = sort_dataset(df)
    row_count_after = len(df)

    logger.warning(
        "additions: %d  rejected: %d  rows %d -> %d",
        len(added),
        len(rejected),
        row_count_before,
        row_count_after,
    )
    for entry in added:
        logger.warning("  + %s (%s)", entry["tmdb_id"], entry["title"])

    if args.dry_run:
        logger.warning("dry run: not writing parquet/csv/manifest")
        return 0

    df.to_parquet(args.parquet, index=False)
    df.to_csv(args.csv, index=False)
    write_manifest(
        args.manifest,
        added=added,
        rejected=rejected,
        row_count_before=row_count_before,
        row_count_after=row_count_after,
    )
    logger.info("wrote %s, %s, %s", args.parquet, args.csv, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
