"""Add COLLECTION_ID / COLLECTION_NAME columns to the canonical local dataset.

Production staging (transformations/) needs collection links per movie, but the
canonical parquet
(data/generated/tmdb/rich_backfill_1980_2026/tmdb_budget_wikipedia_5m_1980_2026.parquet)
carries none — collection membership lives only in the raw backfill JSONL and the
refetch / override gap-fillers. This script materializes the same merged
collection mapping that box_office.ip_classification uses
(box_office.franchise_history.collection_memberships: raw JSONL wins, then the
refetch parquet, then data/collection_overrides.yml) into two new columns on the
parquet and its CSV sibling.

Idempotent: rerunning recomputes both columns from scratch, so it is safe to run
after a refetch refresh. Audited: prints coverage counts before writing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from box_office.franchise_history import (
    DEFAULT_OVERRIDES_PATH,
    DEFAULT_REFETCH_PATH,
    collection_memberships,
)

DATASET_DIR = Path("data/generated/tmdb/rich_backfill_1980_2026")
DEFAULT_PARQUET = DATASET_DIR / "tmdb_budget_wikipedia_5m_1980_2026.parquet"
DEFAULT_CSV = DATASET_DIR / "tmdb_budget_wikipedia_5m_1980_2026.csv"
DEFAULT_JSONL = DATASET_DIR / "tmdb_rich_raw_5m_1980_2026.jsonl"


def add_collection_columns(
    frame: pd.DataFrame,
    memberships: dict[int, tuple[int, str | None]],
) -> pd.DataFrame:
    """Return ``frame`` with ``collection_id`` / ``collection_name`` columns.

    Keyed on ``tmdb_id``. Movies with no collection get null in both columns.
    ``collection_id`` is a pandas nullable ``Int64`` so the absent value is a
    real null rather than a coerced float. Rerunnable: any pre-existing
    collection columns are overwritten from ``memberships``.
    """
    if "tmdb_id" not in frame.columns:
        raise ValueError("frame has no tmdb_id column to key collections on")

    out = frame.copy()
    ids = out["tmdb_id"].astype(int)
    out["collection_id"] = ids.map(
        lambda t: memberships.get(t, (None, None))[0]
    ).astype("Int64")
    out["collection_name"] = ids.map(lambda t: memberships.get(t, (None, None))[1])
    return out


def _audit(frame: pd.DataFrame) -> None:
    total = len(frame)
    linked = int(frame["collection_id"].notna().sum())
    named = int(frame["collection_name"].notna().sum())
    distinct = int(frame["collection_id"].dropna().nunique())
    print(f"rows:               {total}")
    print(f"with collection_id: {linked} ({linked / total:.1%})")
    print(f"with name:          {named}")
    print(f"distinct collections: {distinct}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--refetch", type=Path, default=DEFAULT_REFETCH_PATH)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES_PATH)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and audit without writing the files",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"parquet not found: {args.parquet}")
    if not args.jsonl.exists():
        raise SystemExit(f"raw JSONL not found: {args.jsonl}")

    memberships = collection_memberships(args.jsonl, args.refetch, args.overrides)
    print(f"merged collection links: {len(memberships)}")

    frame = add_collection_columns(pd.read_parquet(args.parquet), memberships)
    _audit(frame)

    if args.dry_run:
        print("dry run: no files written")
        return

    frame.to_parquet(args.parquet, index=False)
    frame.to_csv(args.csv, index=False)
    print(f"wrote {args.parquet}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
