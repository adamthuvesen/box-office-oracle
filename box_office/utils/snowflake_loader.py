"""
Snowflake CSV loader with merge/overwrite support.

Loads CSV data to Snowflake RAW schema with schema validation
and incremental merge capabilities using TMDB_ID as primary key.
"""

import logging
import re
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from box_office.config import config
from box_office.utils.snowflake_connection import create_snowflake_connection

# Pattern for valid SQL identifiers (alphanumeric and underscores, starting with letter/underscore)
SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Snowflake errno for "table does not exist" (sqlstate 42S02 maps to errno 002003).
_SNOWFLAKE_TABLE_NOT_FOUND_ERRNO = 2003


def _is_snowflake_table_not_found(exc: Exception) -> bool:
    """True only for the Snowflake "table does not exist" error.

    Checks the snowflake.connector errno when available (canonical) and falls
    back to a tight message check for cases where the exception is wrapped or
    the connector isn't importable in the test environment. Anything else —
    auth, network, throttling — must propagate.
    """
    errno = getattr(exc, "errno", None)
    if errno == _SNOWFLAKE_TABLE_NOT_FOUND_ERRNO:
        return True
    msg = str(exc).lower()
    return "does not exist" in msg and "table" in msg


def validate_sql_identifier(name: str, identifier_type: str = "identifier") -> str:
    """Reject any string that is not a safe Snowflake unquoted identifier.

    Used at the boundary between env-driven config (database/schema/table
    names) and SQL string interpolation. A compromised env var
    (e.g. ``SNOWFLAKE_DATABASE='FOO; DROP TABLE BAR'``) raises here instead
    of becoming a SQL-injection path.
    """
    if not isinstance(name, str) or not SQL_IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"Invalid {identifier_type} name: {name!r}. "
            "Must contain only alphanumeric characters and underscores, "
            "and start with a letter or underscore."
        )
    return name


def fully_qualified_name(database: str, schema: str, table: str) -> str:
    """Build a validated ``database.schema.table`` identifier for SQL interpolation.

    Validates every component through :func:`validate_sql_identifier` as it
    assembles the name, so a caller cannot interpolate an unvalidated identifier
    into a query. Prefer this over hand-built ``f"{db}.{schema}.{table}"`` strings:
    routing all FQN construction through one validated path means no site can
    silently skip the check (the gap that this closes in the SageMaker loader).
    """
    return ".".join(
        (
            validate_sql_identifier(database, "database"),
            validate_sql_identifier(schema, "schema"),
            validate_sql_identifier(table, "table"),
        )
    )


logger = logging.getLogger(__name__)

# Expected columns for the CSV load path into BOX_OFFICE_V4. The RAW source
# (RAW.BOX_OFFICE_V4 in transformations/models/sources/sources.yml) carries the
# wider 1980-2026 parquet schema; the parquet load path is
# scripts/load_dataset_to_snowflake.py.
EXPECTED_COLUMNS = [
    "tmdb_id",
    "imdb_id",
    "title",
    "release_date",
    "original_language",
    "production_countries",
    "genres",
    "production_budget",
    "director",
    "actors",
    "mpaa",
    "release_type",
    "runtime",
    "overview",
    "tagline",
    "keywords",
    "production_company",
    "release_year",
    "worldwide_gross",
]

# Column type mappings for Snowflake
NUMERIC_COLUMNS = [
    "tmdb_id",
    "production_budget",
    "runtime",
    "release_year",
    "worldwide_gross",
]

STRING_COLUMNS = [
    "imdb_id",
    "title",
    "original_language",
    "production_countries",
    "genres",
    "director",
    "actors",
    "mpaa",
    "release_type",
    "overview",
    "tagline",
    "keywords",
    "production_company",
]


class SnowflakeLoader:
    """Load CSV data to Snowflake RAW schema with merge support."""

    def __init__(
        self,
        schema: str = "RAW",
        database: str | None = None,
        use_browser_auth: bool = False,
    ):
        # Validate identifiers up front so a compromised env var (e.g.
        # SNOWFLAKE_DATABASE='BOX_OFFICE; DROP TABLE FOO') can't reach SQL
        # interpolation later in this class. Closes the H12-H16 cluster.
        self.schema = self._validate_identifier(schema, "schema")
        self.database = self._validate_identifier(
            database or config.snowflake.database, "database"
        )
        self.use_browser_auth = use_browser_auth

    @staticmethod
    def _validate_identifier(name: str, identifier_type: str = "table") -> str:
        """Validate a Snowflake identifier with the shared module helper."""
        return validate_sql_identifier(name, identifier_type)

    @contextmanager
    def _snowflake_cursor(self) -> Generator[tuple[Any, Any], None, None]:
        """Yield ``(cursor, conn)`` and close both on exit."""
        conn = create_snowflake_connection(
            schema=self.schema, use_browser_auth=self.use_browser_auth
        )
        try:
            cursor = conn.cursor()
            try:
                yield cursor, conn
            finally:
                cursor.close()
        finally:
            conn.close()

    def load_csv_to_raw(
        self,
        csv_path: str,
        table_name: str = "BOX_OFFICE_V4",
        mode: Literal["merge", "overwrite"] = "merge",
        dry_run: bool = False,
    ) -> dict:
        """Load a CSV to ``self.schema.<table_name>`` via merge or full overwrite."""
        # Validate table name before any SQL interpolation downstream.
        self._validate_identifier(table_name, "table")

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Read {len(df)} rows from CSV")

        df = self.validate_schema(df)
        df = self.transform_columns(df)

        if dry_run:
            logger.info("Dry run - skipping actual load")
            return {
                "status": "dry_run",
                "rows_to_load": len(df),
                "columns": list(df.columns),
                "missing_columns": [],
                "extra_columns": [],
            }

        if mode == "merge":
            return self._merge_data(df, table_name)
        return self._overwrite_data(df, table_name)

    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase, fill missing, drop extra, and reorder to ``EXPECTED_COLUMNS``."""
        df.columns = df.columns.str.lower()

        current_cols = set(df.columns)
        expected_cols = set(EXPECTED_COLUMNS)

        missing_cols = expected_cols - current_cols
        extra_cols = current_cols - expected_cols

        if missing_cols:
            logger.warning(
                f"Missing columns (will be filled with NULL): {missing_cols}"
            )
            for col in missing_cols:
                df[col] = None

        if extra_cols:
            logger.warning(f"Extra columns (will be dropped): {extra_cols}")
            df = df.drop(columns=list(extra_cols))

        df = df[EXPECTED_COLUMNS]

        logger.info(f"Schema validated: {len(EXPECTED_COLUMNS)} columns")
        return df

    def transform_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce numeric/date/string columns into shapes Snowflake will accept."""
        df = df.copy()

        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # ``replace({np.nan: None})`` is dtype-fragile and can leave
                # NaN behind; mask-based assignment is dtype-safe. Only cast
                # to object when NaN is present so clean columns keep their
                # numeric dtype.
                if df[col].isna().any():
                    df[col] = df[col].astype(object).where(df[col].notna(), None)

        if "release_date" in df.columns:
            df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
            df["release_date"] = df["release_date"].dt.strftime("%Y-%m-%d")
            df["release_date"] = df["release_date"].replace({pd.NaT: None, "NaT": None})

        for col in STRING_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
                df[col] = df[col].replace({"nan": "", "None": "", "NaN": ""})

        logger.info("Column types transformed")
        return df

    def _merge_data(self, df: pd.DataFrame, table_name: str) -> dict:
        """Upsert into ``table_name`` keyed on ``TMDB_ID`` via a staging table + MERGE."""
        staging_table = f"STG_{table_name}_TEMP"

        with self._snowflake_cursor() as (cursor, conn):
            full_staging = fully_qualified_name(
                self.database, self.schema, staging_table
            )
            full_target = fully_qualified_name(self.database, self.schema, table_name)

            try:
                logger.info(f"Creating staging table: {staging_table}")
                cursor.execute(f"DROP TABLE IF EXISTS {full_staging}")
                cursor.execute(f"CREATE TABLE {full_staging} LIKE {full_target}")

                from snowflake.connector.pandas_tools import write_pandas

                df_upload = df.copy()
                df_upload.columns = df_upload.columns.str.upper()

                success, nchunks, nrows, _ = write_pandas(
                    conn=conn,
                    df=df_upload,
                    table_name=staging_table,
                    database=self.database,
                    schema=self.schema,
                    auto_create_table=False,
                    overwrite=True,
                )

                logger.info(f"Uploaded {nrows} rows to staging in {nchunks} chunks")

                columns_upper = [c.upper() for c in EXPECTED_COLUMNS]
                col_list = ", ".join(columns_upper)
                update_set = ", ".join(
                    [f"t.{c} = s.{c}" for c in columns_upper if c != "TMDB_ID"]
                )
                insert_vals = ", ".join([f"s.{c}" for c in columns_upper])

                merge_sql = f"""
                    MERGE INTO {full_target} t
                    USING {full_staging} s
                    ON t.TMDB_ID = s.TMDB_ID
                    WHEN MATCHED THEN UPDATE SET {update_set}
                    WHEN NOT MATCHED THEN INSERT ({col_list}) VALUES ({insert_vals})
                """

                logger.info("Executing MERGE statement...")
                cursor.execute(merge_sql)
                cursor.fetchone()

                cursor.execute(f"DROP TABLE IF EXISTS {full_staging}")
                logger.info(f"Merge complete: {nrows} rows processed")

                return {
                    "status": "success",
                    "mode": "merge",
                    "rows_processed": nrows,
                    "table": full_target,
                }

            except Exception as e:
                logger.error(f"Merge failed: {e}")
                # Best-effort cleanup of the staging table; swallow because the
                # original error is what the caller needs to see.
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {full_staging}")
                except Exception:
                    pass
                raise

    def _overwrite_data(self, df: pd.DataFrame, table_name: str) -> dict:
        """Overwrite table with new data (full replace)."""
        from snowflake.connector.pandas_tools import write_pandas

        df_upload = df.copy()
        df_upload.columns = df_upload.columns.str.upper()
        full_target = fully_qualified_name(self.database, self.schema, table_name)

        with self._snowflake_cursor() as (cursor, conn):
            logger.info(f"Overwriting table: {full_target}")

            success, nchunks, nrows, _ = write_pandas(
                conn=conn,
                df=df_upload,
                table_name=table_name,
                database=self.database,
                schema=self.schema,
                auto_create_table=False,
                overwrite=True,
            )

            logger.info(f"Overwrite complete: {nrows} rows in {nchunks} chunks")

            return {
                "status": "success",
                "mode": "overwrite",
                "rows_loaded": nrows,
                "table": full_target,
            }

    def get_current_count(self, table_name: str = "BOX_OFFICE_V4") -> int:
        """Get current row count in target table.

        Returns 0 only when the table genuinely does not exist (Snowflake errno
        002003). Connection / auth / network errors propagate so callers do not
        see a misleading "Net change: N rows" log on a transient outage.
        """
        self._validate_identifier(table_name, "table")
        full_target = fully_qualified_name(self.database, self.schema, table_name)
        try:
            with self._snowflake_cursor() as (cursor, conn):
                cursor.execute(f"SELECT COUNT(*) FROM {full_target}")
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            if _is_snowflake_table_not_found(e):
                logger.warning(
                    f"Table {full_target} does not exist; reporting count as 0"
                )
                return 0
            raise

    def get_existing_tmdb_ids(self, table_name: str = "BOX_OFFICE_V4") -> set:
        """Get set of existing TMDB IDs in target table.

        Returns ``set()`` only when the table genuinely does not exist.
        Connection / auth / network errors propagate.
        """
        self._validate_identifier(table_name, "table")
        full_target = fully_qualified_name(self.database, self.schema, table_name)
        try:
            with self._snowflake_cursor() as (cursor, conn):
                cursor.execute(f"SELECT DISTINCT TMDB_ID FROM {full_target}")
                results = cursor.fetchall()
                return {row[0] for row in results if row[0] is not None}
        except Exception as e:
            if _is_snowflake_table_not_found(e):
                logger.warning(
                    f"Table {full_target} does not exist; returning empty id set"
                )
                return set()
            raise
