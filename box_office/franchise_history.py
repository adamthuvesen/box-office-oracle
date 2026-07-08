"""Prior-franchise box-office history, strictly before each movie's release.

Shared by the IP classifier (box_office/ip_classification.py) and the
training-frame builder (scripts/prepare_training_frame.py). For each movie,
only films in the same franchise with a release_date STRICTLY earlier
count; same-day releases do not see each other, and a franchise's first
film gets zero prior history.

Known approximation: ``worldwide_gross`` is each prior film's LIFETIME
gross, which can include re-release money earned after the later movie's
release date. The ordering is time-safe; the gross amounts are time-safe
only up to this lifetime-gross approximation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

# Optional gap-fillers for collection links, merged by collection_memberships.
DEFAULT_REFETCH_PATH = Path("data/generated/tmdb/collections_refetch_1980_2026.parquet")
DEFAULT_OVERRIDES_PATH = Path("data/collection_overrides.yml")


def collection_memberships(
    jsonl_path: Path,
    refetch_path: Path | None = DEFAULT_REFETCH_PATH,
    overrides_path: Path | None = DEFAULT_OVERRIDES_PATH,
) -> dict[int, tuple[int, str | None]]:
    """tmdb_id -> (collection_id, collection_name), merged from three sources.

    Precedence: the raw backfill JSONL is the base; the refetch parquet
    (scripts/refetch_collections.py) only fills tmdb_ids the JSONL has no
    collection for — it never overwrites a present link (so a link TMDB
    later removed is kept); manual overrides (data/collection_overrides.yml)
    always win. Refetch/overrides paths that do not exist are skipped;
    pass None to disable a source explicitly.
    """
    mapping: dict[int, tuple[int, str | None]] = {}
    with jsonl_path.open() as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            collection = row.get("payload", {}).get("belongs_to_collection")
            if collection and collection.get("id") is not None:
                mapping[int(row["tmdb_id"])] = (
                    int(collection["id"]),
                    collection.get("name"),
                )

    if refetch_path is not None and refetch_path.exists():
        refetch = pd.read_parquet(refetch_path)
        linked = refetch[refetch["collection_id"].notna()]
        for tmdb_id, collection_id, name in zip(
            linked["tmdb_id"],
            linked["collection_id"],
            linked["collection_name"],
            strict=True,
        ):
            mapping.setdefault(int(tmdb_id), (int(collection_id), name))

    if overrides_path is not None and overrides_path.exists():
        for entry in _load_overrides(overrides_path):
            mapping[entry["tmdb_id"]] = (
                entry["collection_id"],
                entry["collection_name"],
            )
    return mapping


def _load_overrides(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("overrides") or []
    parsed = []
    for i, entry in enumerate(entries):
        try:
            parsed.append(
                {
                    "tmdb_id": int(entry["tmdb_id"]),
                    "collection_id": int(entry["collection_id"]),
                    "collection_name": str(entry["collection_name"]),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed collection override at {path} entry {i}: {entry!r} "
                "(needs tmdb_id, collection_id, collection_name)"
            ) from exc
    return parsed


def collection_franchise_keys(
    jsonl_path: Path,
    refetch_path: Path | None = DEFAULT_REFETCH_PATH,
    overrides_path: Path | None = DEFAULT_OVERRIDES_PATH,
) -> dict[int, str]:
    """tmdb_id -> franchise key ("collection:<id>") from the merged sources.

    Collection-keyed only: umbrella brands never pool gross across
    collections (see box_office/ip_classification.py ``_franchise_key``).
    """
    return {
        tmdb_id: f"collection:{collection_id}"
        for tmdb_id, (collection_id, _) in collection_memberships(
            jsonl_path, refetch_path, overrides_path
        ).items()
    }


def prior_franchise_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-movie prior-franchise gross and film count.

    ``frame`` needs columns: ``franchise_key`` (nullable str; null = no
    franchise), ``release_date`` (datetime64), ``worldwide_gross`` (float).
    Optional ``counts_as_prior`` (bool): rows with False still receive
    their own stats but never contribute to other rows' priors (used for
    rows whose true release_date is unknown).

    Returns a DataFrame aligned to ``frame.index`` with ``prior_gross``
    (sum of worldwide_gross of strictly earlier films in the same
    franchise) and ``prior_count``.
    """
    out = pd.DataFrame(
        {"prior_gross": 0.0, "prior_count": 0},
        index=frame.index,
    )
    if "counts_as_prior" in frame.columns:
        contributes_all = frame["counts_as_prior"].astype(bool)
    else:
        contributes_all = pd.Series(True, index=frame.index)
    keyed = frame[frame["franchise_key"].notna() & frame["release_date"].notna()]
    for _, group in keyed.groupby("franchise_key"):
        dates = group["release_date"].to_numpy()
        gross = group["worldwide_gross"].astype(float).fillna(0.0).to_numpy()
        contributes = contributes_all.loc[group.index].to_numpy()
        for idx, date in zip(group.index, dates, strict=True):
            earlier = (dates < date) & contributes
            n_prior = int(earlier.sum())
            if n_prior == 0:
                continue
            out.at[idx, "prior_gross"] = float(gross[earlier].sum())
            out.at[idx, "prior_count"] = n_prior
    return out
