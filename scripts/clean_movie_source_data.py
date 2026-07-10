"""Apply curated movie source-data cleanup to generated local artifacts.

This is intentionally offline: it cleans the existing CSV/parquet produced by
the TMDB + budget-fill pipeline without refetching TMDB or Wikipedia.

Run:  uv run python scripts/clean_movie_source_data.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from box_office.movie_data_quality import clean_movie_source_data

DEFAULT_PARQUET = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/"
    "tmdb_budget_wikipedia_5m_1980_2026.parquet"
)
DEFAULT_CSV = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/tmdb_budget_wikipedia_5m_1980_2026.csv"
)
DEFAULT_AUDIT = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/"
    "tmdb_budget_wikipedia_source_quality_1980_2026.csv"
)
DEFAULT_MANIFEST = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/"
    "tmdb_budget_wikipedia_source_quality_manifest_1980_2026.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean generated movie source data in place."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def _load_source(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise SystemExit(f"missing source input: {parquet_path} or {csv_path}")


def _counts(audit: pd.DataFrame) -> dict[str, Any]:
    if audit.empty:
        return {"actions": {}, "reasons": {}}
    actions = audit["action"].value_counts().to_dict()
    reasons = audit["reason"].value_counts().to_dict()
    return {
        "actions": {str(key): int(value) for key, value in actions.items()},
        "reasons": {str(key): int(value) for key, value in reasons.items()},
    }


def main() -> None:
    args = parse_args()
    source = _load_source(args.parquet, args.csv)
    cleaned, audit = clean_movie_source_data(source)

    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.parquet, index=False)
    cleaned.to_csv(args.csv, index=False)
    audit.to_csv(args.audit, index=False)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "parquet": str(args.parquet),
            "csv": str(args.csv),
        },
        "outputs": {
            "parquet": str(args.parquet),
            "csv": str(args.csv),
            "audit": str(args.audit),
        },
        "row_counts": {
            "source_rows": int(len(source)),
            "cleaned_rows": int(len(cleaned)),
            "excluded_rows": int(len(source) - len(cleaned)),
            "quality_actions": int(len(audit)),
            **_counts(audit),
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["row_counts"], indent=2))


if __name__ == "__main__":
    main()
