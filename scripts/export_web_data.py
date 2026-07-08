"""Snapshot exporter for the local Next.js app — reads the local dataset.

Writes two JSON files to ``web/data/`` (gitignored):

- ``movies.json`` — every movie in the rich 1980-2026 backfill parquet, with
  the comma-joined string columns (genres/actors/keywords) split into arrays.
  ``worldwide_gross`` is null where the recorded gross is not a real
  theatrical actual (future releases and the documented gross artifacts),
  as flagged by ``scripts/score_all_movies.py``'s predictions parquet.
- ``model_meta.json`` — export timestamp, feature schema version, the
  committed per-year backtest table, and (best effort) live ``/model/info``
  from the inference API.

Oracle predictions (``web/data/predictions.json``) come from
``scripts/score_all_movies.py``, not from here. The legacy
``per_movie_predictions.json`` is removed if present.

Poster/backdrop paths come straight from the parquet — no TMDB API calls,
no Snowflake.

Run:  uv run python scripts/export_web_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from box_office.movie_data_quality import (
    MIN_ROI_BUDGET,
    clean_movie_source_data,
)

# The local-retrain track is canonical (v9 contract); its per-year backtest
# table is the one the current schema version is stamped against.
DEFAULT_PER_YEAR_TABLE = Path("docs/internal/experiments/local_retrain/per_year_table.json")

DEFAULT_INPUT = Path(
    "data/generated/tmdb/rich_backfill_1980_2026/"
    "tmdb_budget_wikipedia_5m_1980_2026.parquet"
)
DEFAULT_PREDICTIONS = Path("data/generated/training/predictions_all_1980_2026.parquet")

# Columns movies.json is built from; anything missing is a hard error.
INPUT_COLUMNS = (
    "tmdb_id",
    "imdb_id",
    "title",
    "release_date",
    "release_year",
    "genres",
    "director",
    "actors",
    "mpaa",
    "runtime",
    "production_budget",
    "production_budget_source",
    "production_company",
    "overview",
    "tagline",
    "keywords",
    "worldwide_gross",
    "poster_path",
    "backdrop_path",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export web/data JSON snapshots for the Next.js app."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Movie dataset parquet (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help=(
            "predictions_all parquet from scripts/score_all_movies.py, used to "
            f"null out unreliable grosses (default: {DEFAULT_PREDICTIONS})"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("web/data"),
        help="Output directory (default: web/data)",
    )
    parser.add_argument(
        "--per-year-table",
        type=Path,
        default=DEFAULT_PER_YEAR_TABLE,
        help=(
            "Per-year backtest table JSON to embed in model_meta "
            f"(default: {DEFAULT_PER_YEAR_TABLE})"
        ),
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Inference API base URL (default: $INFERENCE_API_URL)",
    )
    return parser.parse_args()


def load_movies(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise SystemExit(f"input parquet not found: {input_path}")
    df = pd.read_parquet(input_path)
    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"{input_path} is missing columns: {missing}")
    print(f"loaded {len(df)} movies from {input_path}")
    return df


def unreliable_gross_ids(predictions_path: Path) -> set[int]:
    """tmdb_ids whose recorded gross is a placeholder, not a real actual.

    scripts/score_all_movies.py writes actual_gross as null for future
    releases and the documented gross artifacts; movies.json mirrors that
    by exporting worldwide_gross as null for the same movies.
    """
    if not predictions_path.exists():
        raise SystemExit(
            f"predictions parquet not found: {predictions_path}. "
            "Run `uv run python scripts/score_all_movies.py` first."
        )
    preds = pd.read_parquet(predictions_path)
    ids = {
        int(tmdb_id) for tmdb_id in preds.loc[preds["actual_gross"].isna(), "tmdb_id"]
    }
    print(f"{len(ids)} movies have no reliable gross (exported as null)")
    return ids


def split_comma_list(value: Any) -> list[str]:
    """Split a comma-joined string column into a clean list of strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    parts = (part.strip() for part in str(value).split(","))
    return [part for part in parts if part]


def _json_scalar(value: Any) -> Any:
    """NaN/NaT -> None; numpy scalars -> Python; dates -> ISO strings."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _optional_str(value: Any) -> str | None:
    """Missing strings ('' or NaN) become null."""
    scalar = _json_scalar(value)
    if scalar is None:
        return None
    text = str(scalar).strip()
    return text or None


def _is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def _clean_financials(row: dict[str, Any], no_gross_ids: set[int]) -> tuple[Any, Any]:
    tmdb_id = int(row["tmdb_id"])
    budget = _json_scalar(row["production_budget"])
    gross = _json_scalar(row["worldwide_gross"])

    if tmdb_id in no_gross_ids:
        gross = None

    if budget is not None and budget < MIN_ROI_BUDGET:
        budget = None

    vote_count = row.get("vote_count")
    imdb_id = _optional_str(row.get("imdb_id"))
    has_no_votes = "vote_count" in row and (_is_missing(vote_count) or int(vote_count) == 0)
    if gross is not None and gross >= 1_000_000 and imdb_id is None and has_no_votes:
        gross = None

    return budget, gross


def build_movie_records(
    df: pd.DataFrame, no_gross_ids: set[int]
) -> list[dict[str, Any]]:
    df, _ = clean_movie_source_data(df)
    ordered = df.sort_values("worldwide_gross", ascending=False, na_position="last")

    records = []
    for row in ordered.to_dict(orient="records"):
        tmdb_id = int(row["tmdb_id"])
        budget, gross = _clean_financials(row, no_gross_ids)
        runtime = _json_scalar(row["runtime"])
        records.append(
            {
                "tmdb_id": tmdb_id,
                "imdb_id": _optional_str(row["imdb_id"]),
                "title": _optional_str(row["title"]),
                "release_date": _json_scalar(row["release_date"]),
                "release_year": int(row["release_year"]),
                "genres": split_comma_list(row["genres"]),
                "director": _optional_str(row["director"]),
                "actors": split_comma_list(row["actors"]),
                "mpaa": _optional_str(row["mpaa"]),
                "runtime": runtime if runtime else None,
                "production_budget": budget,
                "production_budget_source": _optional_str(
                    row["production_budget_source"]
                ),
                "production_company": _optional_str(row["production_company"]),
                "overview": _optional_str(row["overview"]),
                "tagline": _optional_str(row["tagline"]),
                "keywords": split_comma_list(row["keywords"]),
                "worldwide_gross": gross,
                "poster_path": _optional_str(row["poster_path"]),
                "backdrop_path": _optional_str(row["backdrop_path"]),
            }
        )
    return records


def fetch_model_info(api_url: str) -> dict[str, Any] | None:
    """GET {api_url}/model/info; any failure means null (offline-friendly)."""
    api_key = os.environ.get("API_KEY") or os.environ.get("INFERENCE_API_KEY")
    headers = {"X-API-Key": api_key} if api_key else {}
    url = f"{api_url.rstrip('/')}/model/info"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload
    except (requests.RequestException, ValueError) as exc:
        print(f"model_info unavailable ({exc}); writing null", file=sys.stderr)
        return None


def build_model_meta(api_url: str | None, per_year_table: Path) -> dict[str, Any]:
    from box_office.ml.feature_schema import CURRENT_FEATURE_SCHEMA_VERSION

    per_year = None
    if per_year_table.exists():
        per_year = json.loads(per_year_table.read_text())
    else:
        print(f"{per_year_table} not found; per_year is null", file=sys.stderr)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "feature_schema_version": CURRENT_FEATURE_SCHEMA_VERSION,
        "per_year": per_year,
        "model_info": fetch_model_info(api_url) if api_url else None,
        "prediction_api_url": api_url,
    }


def write_json(path: Path, payload: Any) -> None:
    # allow_nan=False: a NaN reaching this point is a sanitization bug, and
    # bare NaN would silently break JSON.parse in the web app.
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def main() -> None:
    args = parse_args()

    from box_office.utils.env_setup import configure_environment

    configure_environment()

    api_url = args.api_url or os.environ.get("INFERENCE_API_URL")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_movies(args.input)
    no_gross_ids = unreliable_gross_ids(args.predictions)

    movies = build_movie_records(df, no_gross_ids)
    write_json(out_dir / "movies.json", movies)
    print(f"wrote {out_dir / 'movies.json'} ({len(movies)} movies)")

    legacy = out_dir / "per_movie_predictions.json"
    if legacy.exists():
        legacy.unlink()
        print(f"removed legacy {legacy}")

    write_json(
        out_dir / "model_meta.json", build_model_meta(api_url, args.per_year_table)
    )
    print(f"wrote {out_dir / 'model_meta.json'}")


if __name__ == "__main__":
    main()
