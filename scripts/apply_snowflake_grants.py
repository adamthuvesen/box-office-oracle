#!/usr/bin/env python
"""Apply scripts/snowflake_role_grants.sql as ACCOUNTADMIN.

Reconciles the BOX_OFFICE role model so the least-privilege runtime roles
(DBT_RUNNER for the pipeline/dbt, BOX_OFFICE_LOADER for RAW dataset loads) own
the objects they need. The SQL is idempotent and never drops a table or data.

Connects as ACCOUNTADMIN regardless of the ambient ``SNOWFLAKE_ROLE`` (this is
the one administration path that is allowed to run as ACCOUNTADMIN), prints each
statement and its result, and aborts on the first failure.

    uv run python scripts/apply_snowflake_grants.py            # apply
    uv run python scripts/apply_snowflake_grants.py --dry-run  # print only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("apply_snowflake_grants")

SQL_FILE = Path(__file__).with_name("snowflake_role_grants.sql")


def split_statements(sql: str) -> list[str]:
    """Split a SQL script into executable statements.

    Strips ``--`` line comments and blank lines, then splits on ``;``. The
    grants script contains no strings with embedded semicolons, so a plain
    split is correct here.
    """
    lines = [line.split("--", 1)[0] for line in sql.splitlines()]
    body = "\n".join(lines)
    return [stmt.strip() for stmt in body.split(";") if stmt.strip()]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the statements without connecting to Snowflake",
    )
    args = parser.parse_args(argv)

    if not SQL_FILE.exists():
        print(f"ERROR: grants file not found: {SQL_FILE}", file=sys.stderr)
        return 1

    statements = split_statements(SQL_FILE.read_text())
    if not statements:
        print(f"ERROR: no statements parsed from {SQL_FILE}", file=sys.stderr)
        return 1

    if args.dry_run:
        for i, stmt in enumerate(statements, 1):
            print(f"[{i:2d}] {stmt}")
        print(f"\n[dry-run] {len(statements)} statements; nothing executed")
        return 0

    # Administration path: force ACCOUNTADMIN irrespective of the pipeline role.
    os.environ["SNOWFLAKE_ROLE"] = "ACCOUNTADMIN"

    from box_office.utils.snowflake_connection import create_snowflake_connection

    conn = create_snowflake_connection(schema="RAW")
    failures = 0
    try:
        cursor = conn.cursor()
        try:
            for i, stmt in enumerate(statements, 1):
                one_line = " ".join(stmt.split())
                print(f"\n[{i:2d}/{len(statements)}] {one_line}")
                try:
                    cursor.execute(stmt)
                    rows = cursor.fetchall()
                    result = rows[0][0] if rows and rows[0] else "OK"
                    print(f"       -> {result}")
                except Exception as exc:  # noqa: BLE001 - surface and abort
                    failures += 1
                    print(f"       -> FAILED: {exc}", file=sys.stderr)
                    break
        finally:
            cursor.close()
    finally:
        conn.close()

    if failures:
        print(f"\nABORTED after {failures} failure(s)", file=sys.stderr)
        return 1
    print(f"\nOK: applied {len(statements)} statements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
