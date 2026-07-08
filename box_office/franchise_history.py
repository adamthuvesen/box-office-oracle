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


def collection_franchise_keys(jsonl_path: Path) -> dict[int, str]:
    """tmdb_id -> franchise key ("collection:<id>") from the raw TMDB JSONL.

    Collection-keyed only: umbrella brands never pool gross across
    collections (see box_office/ip_classification.py ``_franchise_key``).
    """
    mapping: dict[int, str] = {}
    with jsonl_path.open() as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            collection = row.get("payload", {}).get("belongs_to_collection")
            if collection and collection.get("id") is not None:
                mapping[int(row["tmdb_id"])] = f"collection:{collection['id']}"
    return mapping


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
