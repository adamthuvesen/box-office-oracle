#!/usr/bin/env python
"""Replace the production RAW dataset in Snowflake with the local 1980-2026 parquet.

Loads the post-refetch parquet
(``data/generated/tmdb/rich_backfill_1980_2026/tmdb_budget_wikipedia_5m_1980_2026.parquet``)
into ``BOX_OFFICE.RAW.BOX_OFFICE_V4``, the successor to ``BOX_OFFICE_V3``.

The load is atomic: rows go into a transient staging table, that table is
verified (row count, null-budget count, spot checks on known movies), and only
then is ``BOX_OFFICE_V4`` replaced from it in a single ``CREATE OR REPLACE`` so
dbt never reads a half-loaded source. Null ``production_budget`` values land as
SQL NULL, never 0.

Dropping the old ``BOX_OFFICE_V3`` is a separate, explicit step behind
``--drop-old``. Without that flag the script prints what it *would* drop and
leaves the old table alone.

Run the real upload from a developer machine (the agent CLI is SELECT-only):

    uv run python scripts/load_dataset_to_snowflake.py

Preview without writing:

    uv run python scripts/load_dataset_to_snowflake.py --dry-run

Load and drop the old table in one reviewed batch:

    uv run python scripts/load_dataset_to_snowflake.py --drop-old
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from box_office.config import config
from box_office.utils.snowflake_connection import create_snowflake_connection
from box_office.utils.snowflake_loader import (
    fully_qualified_name,
    validate_sql_identifier,
)

logger = logging.getLogger("load_dataset_to_snowflake")

DEFAULT_PARQUET = (
    Path("data/generated/tmdb/rich_backfill_1980_2026")
    / "tmdb_budget_wikipedia_5m_1980_2026.parquet"
)

SOURCES_YML = (
    Path(__file__).resolve().parent.parent
    / "transformations"
    / "models"
    / "sources"
    / "sources.yml"
)

TARGET_TABLE = "BOX_OFFICE_V4"
STAGING_TABLE = "BOX_OFFICE_V4_LOAD_TEMP"
OLD_TABLE = "BOX_OFFICE_V3"

# Column whose NaN values must survive the round trip as SQL NULL (never 0).
BUDGET_COLUMN = "production_budget"

# Known movies for the post-load spot check. Values are asserted row-for-row
# against the staging table before the swap; a mismatch aborts the load. Keyed
# by tmdb_id, which is stable across refetches.
SPOT_CHECK_TMDB_IDS: tuple[int, ...] = (597, 19995, 155, 27205, 329)
SPOT_CHECK_COLUMNS: tuple[str, ...] = (
    "title",
    "release_year",
    "production_budget",
    "worldwide_gross",
)


# --- pure logic (unit-tested without a live Snowflake) -----------------------


def staging_columns_from_sources(sources_path: Path = SOURCES_YML) -> list[str]:
    """Return the RAW.BOX_OFFICE_V4 column names from the dbt sources.yml.

    The column contract lives in one place (sources.yml); parsing it at runtime
    keeps the load-time check in sync automatically instead of duplicating the
    list here. Names are lowercased to match the parquet after
    ``read_source_frame`` normalizes casing.
    """
    if not sources_path.exists():
        raise FileNotFoundError(f"dbt sources.yml not found: {sources_path}")

    doc = yaml.safe_load(sources_path.read_text())
    for source in doc.get("sources", []):
        if source.get("name") != "RAW":
            continue
        for table in source.get("tables", []):
            if table.get("name") != TARGET_TABLE:
                continue
            columns = [col["name"].lower() for col in table.get("columns", [])]
            if not columns:
                raise ValueError(
                    f"sources.yml lists no columns for RAW.{TARGET_TABLE}"
                )
            return columns

    raise ValueError(
        f"sources.yml has no RAW.{TARGET_TABLE} table definition: {sources_path}"
    )


def validate_columns(df: pd.DataFrame, expected: list[str]) -> None:
    """Fail loudly when the parquet columns don't match the staging contract."""
    actual = set(df.columns)
    expected_set = set(expected)
    missing = sorted(expected_set - actual)
    extra = sorted(actual - expected_set)
    if missing or extra:
        raise ValueError(
            f"Source parquet columns do not match the RAW.{TARGET_TABLE} "
            f"contract in sources.yml: missing={missing}, extra={extra}"
        )


def read_source_frame(parquet_path: Path) -> pd.DataFrame:
    """Read the parquet, lowercase columns, and reject a malformed source.

    Loud validation: an empty frame, a missing ``tmdb_id``/budget column, a
    column set that doesn't match the RAW.BOX_OFFICE_V4 contract in sources.yml,
    or duplicate ``tmdb_id`` values all raise here rather than producing a
    silently wrong table.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Source parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df.columns = df.columns.str.lower()

    if df.empty:
        raise ValueError(f"Source parquet has no rows: {parquet_path}")

    for required in ("tmdb_id", BUDGET_COLUMN):
        if required not in df.columns:
            raise ValueError(
                f"Source parquet missing required column {required!r}: {parquet_path}"
            )

    validate_columns(df, staging_columns_from_sources())

    duplicate_ids = int(df["tmdb_id"].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            f"Source parquet has {duplicate_ids} duplicate tmdb_id values; "
            "expected one row per movie"
        )

    return df


def expected_null_budget_count(df: pd.DataFrame) -> int:
    """Count rows whose ``production_budget`` is NaN (must become SQL NULL)."""
    return int(df[BUDGET_COLUMN].isna().sum())


def prepare_for_load(df: pd.DataFrame) -> pd.DataFrame:
    """Return an upload-ready copy: uppercased columns, NaN budgets as ``None``.

    ``write_pandas`` maps a float NaN to SQL NULL, but the budget column is the
    one place where a silent coercion to 0 would corrupt the model, so it is
    made explicit: cast to object and replace NaN with ``None``.
    """
    prepared = df.copy()

    if prepared[BUDGET_COLUMN].isna().any():
        prepared[BUDGET_COLUMN] = (
            prepared[BUDGET_COLUMN]
            .astype(object)
            .where(prepared[BUDGET_COLUMN].notna(), None)
        )

    prepared.columns = prepared.columns.str.upper()
    return prepared


def snowflake_column_type(dtype: Any) -> str:
    """Map a pandas dtype to the Snowflake column type used in the staging DDL."""
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "NUMBER"
    if pd.api.types.is_float_dtype(dtype):
        return "FLOAT"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP_NTZ"
    return "VARCHAR"


def build_staging_ddl(fq_staging: str, df: pd.DataFrame) -> str:
    """Build ``CREATE OR REPLACE TRANSIENT TABLE`` DDL preserving parquet order.

    Types come from the source frame's *original* dtypes (uppercased column
    names), so any new parquet column (e.g. a later collection-id backfill) is
    carried automatically and the budget column keeps its FLOAT type even though
    it later holds ``None`` for null budgets. The schema is never hardcoded.
    """
    columns = ", ".join(
        f"{validate_sql_identifier(name.upper(), 'column')} "
        f"{snowflake_column_type(dtype)}"
        for name, dtype in df.dtypes.items()
    )
    return f"CREATE OR REPLACE TRANSIENT TABLE {fq_staging} ({columns})"


def spot_check_mismatches(
    expected_row: dict[str, Any], actual_row: dict[str, Any]
) -> list[str]:
    """Return human-readable mismatches between an expected and actual spot row.

    Numbers are compared with a tolerance so a float round-trip through Snowflake
    (e.g. ``200000000.0``) does not read as a mismatch.
    """
    mismatches: list[str] = []
    for column in SPOT_CHECK_COLUMNS:
        expected = expected_row.get(column)
        actual = actual_row.get(column)
        if _values_equal(expected, actual):
            continue
        mismatches.append(f"{column}: expected {expected!r}, got {actual!r}")
    return mismatches


def _values_equal(expected: Any, actual: Any) -> bool:
    """Compare two scalar cell values, tolerating NULL and float rounding."""
    expected_null = expected is None or (
        isinstance(expected, float) and pd.isna(expected)
    )
    actual_null = actual is None or (isinstance(actual, float) and pd.isna(actual))
    if expected_null or actual_null:
        return expected_null and actual_null

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= 1.0

    return str(expected) == str(actual)


def build_spot_check_expectations(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Extract expected spot-check rows from the source frame, keyed by tmdb_id.

    Silently skips ids absent from this dataset (the known-id list is a superset
    across refetches); at least one must be present or the caller should treat
    that as a failure.
    """
    expectations: dict[int, dict[str, Any]] = {}
    for tmdb_id in SPOT_CHECK_TMDB_IDS:
        match = df[df["tmdb_id"] == tmdb_id]
        if match.empty:
            continue
        row = match.iloc[0]
        expectations[tmdb_id] = {col: row[col] for col in SPOT_CHECK_COLUMNS}
    return expectations


# --- Snowflake side effects --------------------------------------------------


class DatasetLoader:
    """Load the source parquet into ``RAW.BOX_OFFICE_V4`` with an atomic swap."""

    def __init__(self, database: str, schema: str, use_browser_auth: bool = False):
        self.database = validate_sql_identifier(database, "database")
        self.schema = validate_sql_identifier(schema, "schema")
        self.use_browser_auth = use_browser_auth

    def run(self, parquet_path: Path, drop_old: bool) -> None:
        df = read_source_frame(parquet_path)
        expected_rows = len(df)
        expected_nulls = expected_null_budget_count(df)
        expectations = build_spot_check_expectations(df)
        if not expectations:
            raise ValueError(
                "None of the spot-check tmdb_ids are present in the source; "
                "refusing to load without a verifiable anchor"
            )

        prepared = prepare_for_load(df)

        logger.info(
            "Source %s: %d rows, %d null budgets, %d spot anchors",
            parquet_path,
            expected_rows,
            expected_nulls,
            len(expectations),
        )

        from snowflake.connector.pandas_tools import write_pandas

        fq_staging = fully_qualified_name(self.database, self.schema, STAGING_TABLE)
        fq_target = fully_qualified_name(self.database, self.schema, TARGET_TABLE)

        conn = create_snowflake_connection(
            schema=self.schema, use_browser_auth=self.use_browser_auth
        )
        try:
            cursor = conn.cursor()
            try:
                logger.info("Creating transient staging table %s", fq_staging)
                cursor.execute(build_staging_ddl(fq_staging, df))

                logger.info("Uploading %d rows to staging", expected_rows)
                success, _, nrows, _ = write_pandas(
                    conn=conn,
                    df=prepared,
                    table_name=STAGING_TABLE,
                    database=self.database,
                    schema=self.schema,
                    auto_create_table=False,
                    overwrite=False,
                )
                if not success:
                    raise RuntimeError("write_pandas reported failure loading staging")

                self._verify(cursor, fq_staging, expected_rows, expected_nulls, expectations)

                logger.info("Verification passed; swapping %s into place", fq_target)
                cursor.execute(
                    f"CREATE OR REPLACE TABLE {fq_target} AS SELECT * FROM {fq_staging}"
                )
                cursor.execute(f"DROP TABLE IF EXISTS {fq_staging}")
                logger.info("Load complete: %s now holds %d rows", fq_target, expected_rows)

                self._maybe_drop_old(cursor, drop_old)
            finally:
                cursor.close()
        finally:
            conn.close()

    def _verify(
        self,
        cursor: Any,
        fq_staging: str,
        expected_rows: int,
        expected_nulls: int,
        expectations: dict[int, dict[str, Any]],
    ) -> None:
        """Assert the staging table matches the source; raise on any mismatch."""
        cursor.execute(f"SELECT COUNT(*) FROM {fq_staging}")
        actual_rows = int(cursor.fetchone()[0])
        if actual_rows != expected_rows:
            raise ValueError(
                f"Row count mismatch: parquet has {expected_rows}, "
                f"staging has {actual_rows}"
            )

        cursor.execute(
            f"SELECT COUNT(*) FROM {fq_staging} WHERE PRODUCTION_BUDGET IS NULL"
        )
        actual_nulls = int(cursor.fetchone()[0])
        if actual_nulls != expected_nulls:
            raise ValueError(
                f"Null-budget mismatch: parquet has {expected_nulls}, "
                f"staging has {actual_nulls} (budgets must be NULL, not 0)"
            )

        columns = ", ".join(col.upper() for col in SPOT_CHECK_COLUMNS)
        for tmdb_id, expected_row in expectations.items():
            cursor.execute(
                f"SELECT {columns} FROM {fq_staging} WHERE TMDB_ID = {int(tmdb_id)}"
            )
            fetched = cursor.fetchall()
            if len(fetched) != 1:
                raise ValueError(
                    f"Spot check for tmdb_id {tmdb_id}: expected 1 row, got {len(fetched)}"
                )
            actual_row = dict(zip(SPOT_CHECK_COLUMNS, fetched[0], strict=True))
            mismatches = spot_check_mismatches(expected_row, actual_row)
            if mismatches:
                raise ValueError(
                    f"Spot check for tmdb_id {tmdb_id} failed: {'; '.join(mismatches)}"
                )

        logger.info(
            "Verified: %d rows, %d null budgets, %d spot checks",
            actual_rows,
            actual_nulls,
            len(expectations),
        )

    def _maybe_drop_old(self, cursor: Any, drop_old: bool) -> None:
        """Drop the old table only when explicitly asked; otherwise just report."""
        fq_old = fully_qualified_name(self.database, self.schema, OLD_TABLE)
        if not drop_old:
            logger.info(
                "Old table left in place. Re-run with --drop-old to drop %s", fq_old
            )
            return
        logger.info("Dropping old table %s", fq_old)
        cursor.execute(f"DROP TABLE IF EXISTS {fq_old}")
        logger.info("Dropped %s", fq_old)


def _print_dry_run(database: str, schema: str, parquet_path: Path, drop_old: bool) -> int:
    df = read_source_frame(parquet_path)
    fq_staging = fully_qualified_name(database, schema, STAGING_TABLE)
    fq_target = fully_qualified_name(database, schema, TARGET_TABLE)
    fq_old = fully_qualified_name(database, schema, OLD_TABLE)

    print(f"[dry-run] source parquet    : {parquet_path}")
    print(f"[dry-run] rows              : {len(df)}")
    print(f"[dry-run] null budgets      : {expected_null_budget_count(df)}")
    print(f"[dry-run] columns           : {len(df.columns)}")
    print(f"[dry-run] staging table     : {fq_staging} (transient, verified, then dropped)")
    print(f"[dry-run] target table      : CREATE OR REPLACE {fq_target}")
    if drop_old:
        print(f"[dry-run] would DROP        : {fq_old}")
    else:
        print(f"[dry-run] old table {fq_old} left in place (pass --drop-old to drop it)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Source parquet (default: local rich backfill 1980-2026)",
    )
    parser.add_argument(
        "--database",
        default=config.snowflake.database,
        help="Target Snowflake database (default from config)",
    )
    parser.add_argument(
        "--schema",
        default=config.snowflake.schemas.raw,
        help="Target Snowflake schema (default: RAW from config)",
    )
    parser.add_argument(
        "--drop-old",
        action="store_true",
        help=f"Also drop the old {OLD_TABLE} table after a successful load",
    )
    parser.add_argument(
        "--browser-auth",
        action="store_true",
        help="Use externalbrowser SSO instead of key-pair auth",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without touching Snowflake",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)

    try:
        if args.dry_run:
            return _print_dry_run(
                args.database, args.schema, args.parquet, args.drop_old
            )

        loader = DatasetLoader(
            database=args.database,
            schema=args.schema,
            use_browser_auth=args.browser_auth,
        )
        loader.run(args.parquet, drop_old=args.drop_old)
    except Exception as exc:
        logger.error("Load failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("OK: dataset loaded into", TARGET_TABLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
