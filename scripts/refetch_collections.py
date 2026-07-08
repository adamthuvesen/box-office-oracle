"""Refetch TMDB collection links for every movie in the cleaned source.

The raw rich-backfill JSONL froze ``belongs_to_collection`` at fetch time;
TMDB has since added collection links for some movies, so the collection-
keyed franchise features undercount. This script asks TMDB /movie/{id}
again for every tmdb_id in the cleaned source parquet and records the
current collection link.

The output is a gap-filler, not a replacement: consumers merge it via
``box_office.franchise_history.collection_memberships`` where the JSONL
value wins when present. If TMDB removed a link we already have, the old
one is kept by that merge; this script only logs it as ``lost_vs_jsonl``.

Run:  uv run python scripts/refetch_collections.py
Needs TMDB_API_TOKEN (see .env).

Writes:
- data/generated/tmdb/collections_refetch_1980_2026.parquet
- data/generated/tmdb/collections_refetch_manifest_1980_2026.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from box_office.franchise_history import collection_memberships
from box_office.ingestion.tmdb_rich_backfill import (
    TMDB_API_URL,
    BackfillConfig,
    BackfillState,
    request_json,
    utc_now,
)

logger = logging.getLogger(__name__)

SOURCE_PARQUET = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/"
    "tmdb_budget_wikipedia_5m_1980_2026.parquet"
)
RAW_JSONL = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/tmdb_rich_raw_5m_1980_2026.jsonl"
)
OUTPUT_PARQUET = Path("data/generated/tmdb/collections_refetch_1980_2026.parquet")
OUTPUT_MANIFEST = Path(
    "data/generated/tmdb/collections_refetch_manifest_1980_2026.json"
)

MAX_FAILED_MOVIES = 5


def fetch_collection_rows(
    tmdb_ids: list[int],
    config: BackfillConfig,
    state: BackfillState,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    session = requests.Session()
    for i, tmdb_id in enumerate(tmdb_ids, start=1):
        try:
            payload = request_json(
                session,
                f"{TMDB_API_URL}/movie/{tmdb_id}",
                params=None,
                config=config,
                state=state,
            )
        except requests.RequestException as exc:
            failures.append({"tmdb_id": tmdb_id, "error": str(exc)})
            if len(failures) > MAX_FAILED_MOVIES:
                raise SystemExit(
                    f"aborting: more than {MAX_FAILED_MOVIES} movies failed "
                    f"persistently; last error for tmdb {tmdb_id}: {exc}"
                ) from exc
            continue
        collection = payload.get("belongs_to_collection") or {}
        rows.append(
            {
                "tmdb_id": tmdb_id,
                "collection_id": collection.get("id"),
                "collection_name": collection.get("name"),
            }
        )
        if i % 250 == 0:
            logger.info("fetched %d/%d movies", i, len(tmdb_ids))
        time.sleep(config.base_sleep_seconds)
    return rows, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refetch TMDB collection links for the cleaned source movies."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_PARQUET)
    parser.add_argument("--raw-jsonl", type=Path, default=RAW_JSONL)
    parser.add_argument("--out", type=Path, default=OUTPUT_PARQUET)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument(
        "--limit", type=int, default=None, help="Fetch only the first N movies (smoke)."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()
    tmdb_ids = [
        int(v) for v in pd.read_parquet(args.source, columns=["tmdb_id"])["tmdb_id"]
    ]
    if args.limit is not None:
        tmdb_ids = tmdb_ids[: args.limit]

    # JSONL-only view (no refetch/overrides) to diff against.
    jsonl_map = collection_memberships(
        args.raw_jsonl, refetch_path=None, overrides_path=None
    )

    config = BackfillConfig()
    state = BackfillState(started_at=utc_now())
    rows, failures = fetch_collection_rows(tmdb_ids, config, state)

    frame = pd.DataFrame(rows, columns=["tmdb_id", "collection_id", "collection_name"])
    frame["collection_id"] = frame["collection_id"].astype("Int64")

    has_collection = frame[frame["collection_id"].notna()]
    gained = has_collection[~has_collection["tmdb_id"].isin(jsonl_map)]
    lost_ids = [
        tmdb_id
        for tmdb_id in frame.loc[frame["collection_id"].isna(), "tmdb_id"]
        if tmdb_id in jsonl_map
    ]
    if lost_ids:
        logger.warning(
            "TMDB no longer links %d movies we have collections for; "
            "the merge keeps the old JSONL links: %s",
            len(lost_ids),
            lost_ids,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    manifest = {
        "started_at": state.started_at,
        "finished_at": utc_now(),
        "source": str(args.source),
        "requested": len(tmdb_ids),
        "fetched": len(frame),
        "has_collection": int(len(has_collection)),
        "gained_vs_jsonl": int(len(gained)),
        "lost_vs_jsonl": len(lost_ids),
        "lost_tmdb_ids": lost_ids,
        "failed_movies": failures,
        "request_count": state.request_count,
        "retry_count": state.retry_count,
        "rate_limit_count": state.rate_limit_count,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"fetched {len(frame)}/{len(tmdb_ids)} movies")
    print(f"has_collection: {len(has_collection)}")
    print(f"gained_vs_jsonl: {len(gained)}")
    print(f"lost_vs_jsonl: {len(lost_ids)}")
    print(f"wrote {args.out}")
    print(f"wrote {args.manifest}")
    if failures:
        raise SystemExit(f"{len(failures)} movies failed persistently: {failures}")


if __name__ == "__main__":
    main()
