"""Local-only rich TMDB backfill.

This module deliberately stays separate from ``box-office-ingest``. The older
CLI writes the compact 20-column shape used by Snowflake; this command archives
rich TMDB payloads first, then derives flat local analysis files from them.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from box_office.ingestion.data_enrichment import HeuristicEnricher

logger = logging.getLogger(__name__)

TMDB_API_URL = "https://api.themoviedb.org/3"
DEFAULT_APPEND_RESPONSES = (
    "credits",
    "keywords",
    "release_dates",
    "external_ids",
    "images",
    "videos",
    "alternative_titles",
    "translations",
)
EXCLUDED_GENRES = frozenset({"Documentary"})


@dataclass(frozen=True)
class BackfillConfig:
    start_year: int = 1980
    end_year: int = 2026
    min_revenue: int = 5_000_000
    fifty_million_revenue: int = 50_000_000
    page_limit: int = 100
    stop_after_consecutive_empty_pages: int = 5
    request_timeout_seconds: int = 20
    max_retries: int = 5
    base_sleep_seconds: float = 0.35
    retry_sleep_seconds: float = 2.0
    append_responses: tuple[str, ...] = DEFAULT_APPEND_RESPONSES
    existing_web_data: Path = Path("web/data/movies.json")
    output_dir: Path = Path("data/generated/tmdb/rich_backfill_1980_2026")
    seed: int = 42

    @property
    def raw_jsonl_path(self) -> Path:
        return self.output_dir / "tmdb_rich_raw_5m_1980_2026.jsonl"

    @property
    def flat_csv_path(self) -> Path:
        return self.output_dir / "tmdb_flat_5m_1980_2026.csv"

    @property
    def flat_parquet_path(self) -> Path:
        return self.output_dir / "tmdb_flat_5m_1980_2026.parquet"

    @property
    def fifty_million_csv_path(self) -> Path:
        return self.output_dir / "tmdb_flat_50m_1980_2026.csv"

    @property
    def fifty_million_parquet_path(self) -> Path:
        return self.output_dir / "tmdb_flat_50m_1980_2026.parquet"

    @property
    def combined_csv_path(self) -> Path:
        return self.output_dir / "tmdb_combined_flat_5m_1980_2026.csv"

    @property
    def combined_parquet_path(self) -> Path:
        return self.output_dir / "tmdb_combined_flat_5m_1980_2026.parquet"

    @property
    def combined_fifty_million_csv_path(self) -> Path:
        return self.output_dir / "tmdb_combined_flat_50m_1980_2026.csv"

    @property
    def combined_fifty_million_parquet_path(self) -> Path:
        return self.output_dir / "tmdb_combined_flat_50m_1980_2026.parquet"

    @property
    def summary_csv_path(self) -> Path:
        return self.output_dir / "tmdb_summary_1980_2026.csv"

    @property
    def missingness_csv_path(self) -> Path:
        return self.output_dir / "tmdb_missingness_1980_2026.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "tmdb_backfill_manifest_1980_2026.json"


@dataclass
class BackfillState:
    started_at: str
    finished_at: str | None = None
    discovered_count: int = 0
    accepted_count: int = 0
    skipped_existing_count: int = 0
    skipped_below_revenue_count: int = 0
    skipped_language_count: int = 0
    skipped_documentary_count: int = 0
    skipped_missing_id_count: int = 0
    request_count: int = 0
    retry_count: int = 0
    rate_limit_count: int = 0
    failed_pages: list[dict[str, Any]] = field(default_factory=list)
    failed_movies: list[dict[str, Any]] = field(default_factory=list)
    per_year: dict[str, dict[str, int]] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def auth_headers() -> dict[str, str]:
    token = os.getenv("TMDB_API_TOKEN")
    if not token:
        raise RuntimeError("TMDB_API_TOKEN is not set.")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def load_existing_movies_from_web(path: Path) -> tuple[set[int], set[str]]:
    if not path.exists():
        logger.warning("Existing web data not found: %s", path)
        return set(), set()

    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")

    ids: set[int] = set()
    titles: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        tmdb_id = row.get("tmdb_id")
        if tmdb_id is not None:
            ids.add(int(tmdb_id))
        title = row.get("title")
        if title:
            titles.add(str(title).strip().lower())
    return ids, titles


def load_existing_flat_from_web(
    path: Path, config: BackfillConfig
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"Expected {path} to contain a JSON list.")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for column in [
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
        "ad_budget",
        "production_company",
        "overview",
        "tagline",
        "keywords",
        "worldwide_gross",
        "poster_path",
        "backdrop_path",
    ]:
        if column not in df.columns:
            df[column] = None

    df = df.copy()
    df["original_title"] = df.get("original_title", df["title"])
    df["status"] = df.get("status", "Released")
    df["original_language"] = df.get("original_language", "en")
    df["production_countries"] = df.get("production_countries", "")
    df["spoken_languages"] = df.get("spoken_languages", "")
    df["homepage"] = df.get("homepage", None)
    df["adult"] = df.get("adult", None)
    df["popularity"] = df.get("popularity", None)
    df["vote_average"] = df.get("vote_average", None)
    df["vote_count"] = df.get("vote_count", None)
    df["poster_count"] = df["poster_path"].notna().astype(int)
    df["backdrop_count"] = df["backdrop_path"].notna().astype(int)
    df["video_count"] = df.get("video_count", 0)
    df["alternative_title_count"] = df.get("alternative_title_count", 0)
    df["translation_count"] = df.get("translation_count", 0)
    df["release_year_query"] = df["release_year"]
    df["discover_page"] = None
    df["fetched_at"] = None
    df["source_dataset"] = "web_existing"

    for column in ["genres", "actors", "keywords"]:
        df[column] = df[column].apply(jsonish_list_to_text)

    df["worldwide_gross"] = pd.to_numeric(
        df["worldwide_gross"], errors="coerce"
    )
    df = df[
        (df["release_year"] >= config.start_year)
        & (df["release_year"] <= config.end_year)
        & (df["worldwide_gross"] >= config.min_revenue)
    ]
    return df[flat_columns()]


def jsonish_list_to_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(
            str(item).strip() for item in value if str(item).strip()
        )
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def iter_raw_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                message = f"Invalid JSONL at {path}:{line_number}"
                raise ValueError(message) from exc
            if isinstance(value, dict):
                yield value


def load_raw_tmdb_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    for record in iter_raw_records(path):
        tmdb_id = record.get("tmdb_id")
        if tmdb_id is not None:
            ids.add(int(tmdb_id))
    return ids


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None,
    config: BackfillConfig,
    state: BackfillState,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        state.request_count += 1
        try:
            response = session.get(
                url,
                headers=auth_headers(),
                params=params,
                timeout=config.request_timeout_seconds,
            )
            if response.status_code == 429:
                state.rate_limit_count += 1
                retry_after = response.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after else 60.0
                logger.warning(
                    "TMDB rate limit hit; sleeping %.1fs", sleep_for
                )
                time.sleep(sleep_for)
                state.retry_count += 1
                continue
            if 500 <= response.status_code < 600:
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= config.max_retries:
                break
            state.retry_count += 1
            sleep_for = config.retry_sleep_seconds * (2**attempt)
            logger.warning(
                "TMDB request failed on attempt %d/%d: %s; sleeping %.1fs",
                attempt + 1,
                config.max_retries + 1,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)

    assert last_error is not None
    raise last_error


def normalize_keywords(payload: dict[str, Any]) -> str:
    keywords = payload.get("keywords") or {}
    values = keywords.get("keywords") if isinstance(keywords, dict) else []
    return ", ".join(
        str(keyword.get("name", "")).strip()
        for keyword in values
        if isinstance(keyword, dict) and keyword.get("name")
    )


def normalize_credits(payload: dict[str, Any]) -> dict[str, str]:
    credits = payload.get("credits") or {}
    crew = credits.get("crew") if isinstance(credits, dict) else []
    cast = credits.get("cast") if isinstance(credits, dict) else []

    directors = [
        str(member.get("name", "")).strip()
        for member in crew
        if isinstance(member, dict)
        and member.get("job") == "Director"
        and member.get("name")
    ]
    sorted_cast = sorted(
        [member for member in cast if isinstance(member, dict)],
        key=lambda member: member.get("order", 999),
    )
    actors = [
        str(member.get("name", "")).strip()
        for member in sorted_cast[:10]
        if member.get("name")
    ]
    return {"director": ", ".join(directors), "actors": ", ".join(actors)}


def normalize_mpaa(payload: dict[str, Any]) -> str:
    release_dates = payload.get("release_dates") or {}
    results = (
        release_dates.get("results") if isinstance(release_dates, dict) else []
    )
    for country_release in results:
        if not isinstance(country_release, dict):
            continue
        if country_release.get("iso_3166_1") != "US":
            continue
        for release in country_release.get("release_dates", []):
            if not isinstance(release, dict):
                continue
            certification = str(release.get("certification", "")).strip()
            if certification:
                return certification
    return "Not Rated"


def names_from_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(
        str(item.get("name", "")).strip()
        for item in values
        if isinstance(item, dict) and item.get("name")
    )


def country_names(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(
        str(item.get("name", "")).strip()
        for item in values
        if isinstance(item, dict) and item.get("name")
    )


def extract_release_year(release_date: Any) -> int | None:
    if not release_date:
        return None
    parsed = pd.to_datetime(release_date, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.year)


def raw_record_from_payload(
    payload: dict[str, Any], year: int, page: int
) -> dict[str, Any]:
    tmdb_id = payload.get("id")
    return {
        "tmdb_id": int(tmdb_id) if tmdb_id is not None else None,
        "title": payload.get("title"),
        "release_year_query": year,
        "discover_page": page,
        "fetched_at": utc_now(),
        "payload": payload,
    }


def flat_record_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    payload = raw.get("payload") or {}
    credits = normalize_credits(payload)
    external_ids = payload.get("external_ids") or {}
    images = payload.get("images") or {}
    videos = payload.get("videos") or {}
    alternatives = payload.get("alternative_titles") or {}
    translations = payload.get("translations") or {}

    poster_count = (
        len(images.get("posters", [])) if isinstance(images, dict) else 0
    )
    backdrop_count = (
        len(images.get("backdrops", [])) if isinstance(images, dict) else 0
    )
    video_count = (
        len(videos.get("results", [])) if isinstance(videos, dict) else 0
    )
    alternative_title_count = (
        len(alternatives.get("titles", []))
        if isinstance(alternatives, dict)
        else 0
    )
    translation_count = (
        len(translations.get("translations", []))
        if isinstance(translations, dict)
        else 0
    )

    release_date = payload.get("release_date")
    return {
        "tmdb_id": raw.get("tmdb_id"),
        "imdb_id": payload.get("imdb_id") or external_ids.get("imdb_id"),
        "title": payload.get("title"),
        "original_title": payload.get("original_title"),
        "status": payload.get("status"),
        "release_date": release_date,
        "release_year": extract_release_year(release_date),
        "original_language": payload.get("original_language"),
        "production_countries": country_names(
            payload.get("production_countries")
        ),
        "spoken_languages": names_from_list(payload.get("spoken_languages")),
        "genres": names_from_list(payload.get("genres")),
        "production_budget": payload.get("budget"),
        "worldwide_gross": payload.get("revenue"),
        "runtime": payload.get("runtime"),
        "director": credits["director"],
        "actors": credits["actors"],
        "mpaa": normalize_mpaa(payload),
        "overview": payload.get("overview"),
        "tagline": payload.get("tagline"),
        "keywords": normalize_keywords(payload),
        "production_company": names_from_list(
            payload.get("production_companies")
        ),
        "homepage": payload.get("homepage"),
        "poster_path": payload.get("poster_path"),
        "backdrop_path": payload.get("backdrop_path"),
        "adult": payload.get("adult"),
        "popularity": payload.get("popularity"),
        "vote_average": payload.get("vote_average"),
        "vote_count": payload.get("vote_count"),
        "poster_count": poster_count,
        "backdrop_count": backdrop_count,
        "video_count": video_count,
        "alternative_title_count": alternative_title_count,
        "translation_count": translation_count,
        "release_year_query": raw.get("release_year_query"),
        "discover_page": raw.get("discover_page"),
        "fetched_at": raw.get("fetched_at"),
        "source_dataset": "tmdb_backfill",
    }


def flat_columns() -> list[str]:
    return [
        "tmdb_id",
        "imdb_id",
        "title",
        "original_title",
        "status",
        "release_date",
        "release_year",
        "original_language",
        "production_countries",
        "spoken_languages",
        "genres",
        "production_budget",
        "worldwide_gross",
        "runtime",
        "director",
        "actors",
        "mpaa",
        "overview",
        "tagline",
        "keywords",
        "production_company",
        "homepage",
        "poster_path",
        "backdrop_path",
        "adult",
        "popularity",
        "vote_average",
        "vote_count",
        "poster_count",
        "backdrop_count",
        "video_count",
        "alternative_title_count",
        "translation_count",
        "release_year_query",
        "discover_page",
        "fetched_at",
        "source_dataset",
        "ad_budget",
    ]


def is_documentary(payload: dict[str, Any]) -> bool:
    genres = payload.get("genres") or []
    names = {
        str(genre.get("name", "")).strip()
        for genre in genres
        if isinstance(genre, dict)
    }
    return bool(names & EXCLUDED_GENRES)


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def year_stats(state: BackfillState, year: int) -> dict[str, int]:
    key = str(year)
    if key not in state.per_year:
        state.per_year[key] = {
            "discovered": 0,
            "accepted": 0,
            "skipped_existing": 0,
            "skipped_below_revenue": 0,
            "skipped_language": 0,
            "skipped_documentary": 0,
            "failed_pages": 0,
            "failed_movies": 0,
        }
    return state.per_year[key]


def run_backfill(config: BackfillConfig) -> BackfillState:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    existing_ids, existing_titles = load_existing_movies_from_web(
        config.existing_web_data
    )
    raw_ids = load_raw_tmdb_ids(config.raw_jsonl_path)
    skipped_ids = existing_ids | raw_ids

    logger.info("Loaded %d existing web TMDB IDs", len(existing_ids))
    logger.info("Loaded %d already written raw TMDB IDs", len(raw_ids))

    state = BackfillState(started_at=utc_now())
    session = requests.Session()

    for year in range(config.start_year, config.end_year + 1):
        logger.info("Processing %d", year)
        empty_pages = 0
        stats = year_stats(state, year)

        for page in range(1, config.page_limit + 1):
            try:
                discover = request_json(
                    session,
                    f"{TMDB_API_URL}/discover/movie",
                    params={
                        "primary_release_year": year,
                        "with_original_language": "en",
                        "sort_by": "revenue.desc",
                        "include_adult": "false",
                        "page": page,
                    },
                    config=config,
                    state=state,
                )
            except Exception as exc:
                failure = {"year": year, "page": page, "error": str(exc)}
                state.failed_pages.append(failure)
                stats["failed_pages"] += 1
                logger.error("Failed discover page %d/%d: %s", year, page, exc)
                continue

            page_accepted = 0
            results = discover.get("results", [])
            for item in results:
                state.discovered_count += 1
                stats["discovered"] += 1
                tmdb_id = item.get("id") if isinstance(item, dict) else None
                if tmdb_id is None:
                    state.skipped_missing_id_count += 1
                    continue
                tmdb_id = int(tmdb_id)
                if tmdb_id in skipped_ids:
                    state.skipped_existing_count += 1
                    stats["skipped_existing"] += 1
                    continue

                try:
                    payload = request_json(
                        session,
                        f"{TMDB_API_URL}/movie/{tmdb_id}",
                        params={
                            "append_to_response": ",".join(
                                config.append_responses
                            ),
                            "include_image_language": "en,null",
                        },
                        config=config,
                        state=state,
                    )
                except Exception as exc:
                    failure = {
                        "year": year,
                        "page": page,
                        "tmdb_id": tmdb_id,
                        "error": str(exc),
                    }
                    state.failed_movies.append(failure)
                    stats["failed_movies"] += 1
                    logger.error("Failed movie %s: %s", tmdb_id, exc)
                    continue

                if payload.get("original_language") != "en":
                    state.skipped_language_count += 1
                    stats["skipped_language"] += 1
                    continue
                title = str(payload.get("title") or "").strip().lower()
                if title and title in existing_titles:
                    state.skipped_existing_count += 1
                    stats["skipped_existing"] += 1
                    skipped_ids.add(tmdb_id)
                    continue
                revenue = int(payload.get("revenue") or 0)
                if revenue < config.min_revenue:
                    state.skipped_below_revenue_count += 1
                    stats["skipped_below_revenue"] += 1
                    continue
                if is_documentary(payload):
                    state.skipped_documentary_count += 1
                    stats["skipped_documentary"] += 1
                    continue

                record = raw_record_from_payload(payload, year=year, page=page)
                write_jsonl_record(config.raw_jsonl_path, record)
                skipped_ids.add(tmdb_id)
                if title:
                    existing_titles.add(title)
                state.accepted_count += 1
                stats["accepted"] += 1
                page_accepted += 1
                time.sleep(config.base_sleep_seconds)

            logger.info(
                "Year %d page %d: %d accepted, %d results",
                year,
                page,
                page_accepted,
                len(results),
            )

            if page_accepted == 0:
                empty_pages += 1
            else:
                empty_pages = 0
            if page >= int(discover.get("total_pages", 0)):
                break
            if empty_pages >= config.stop_after_consecutive_empty_pages:
                logger.info(
                    "Stopping %d after %d consecutive empty pages",
                    year,
                    empty_pages,
                )
                break
            time.sleep(0.25)

    state.finished_at = utc_now()
    return state


def build_flat_outputs(
    config: BackfillConfig, state: BackfillState | None = None
) -> dict[str, Any]:
    records = [
        flat_record_from_raw(record)
        for record in iter_raw_records(config.raw_jsonl_path)
    ]
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No raw records found at {config.raw_jsonl_path}")

    df = HeuristicEnricher(seed=config.seed).enrich(df)
    if "source_dataset" not in df.columns:
        df["source_dataset"] = "tmdb_backfill"
    df = df.sort_values(
        ["release_year", "worldwide_gross", "tmdb_id"],
        ascending=[True, False, True],
    )
    df = df[flat_columns()]

    duplicate_count = int(df["tmdb_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"Raw output contains {duplicate_count} duplicate TMDB IDs"
        )
    if int((df["worldwide_gross"] < config.min_revenue).sum()):
        raise ValueError(
            "Flat output contains rows below the configured revenue floor"
        )

    subset_50m = df[
        df["worldwide_gross"] >= config.fifty_million_revenue
    ].copy()
    existing_df = load_existing_flat_from_web(config.existing_web_data, config)
    combined_parts = [
        part.dropna(axis=1, how="all")
        for part in [existing_df, df]
        if not part.empty
    ]
    combined = pd.concat(combined_parts, ignore_index=True).reindex(
        columns=flat_columns()
    )
    combined = combined.drop_duplicates(subset=["tmdb_id"], keep="last")
    combined = combined.sort_values(
        ["release_year", "worldwide_gross", "tmdb_id"],
        ascending=[True, False, True],
    )
    combined_50m = combined[
        combined["worldwide_gross"] >= config.fifty_million_revenue
    ].copy()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.flat_csv_path, index=False)
    df.to_parquet(config.flat_parquet_path, index=False)
    subset_50m.to_csv(config.fifty_million_csv_path, index=False)
    subset_50m.to_parquet(config.fifty_million_parquet_path, index=False)
    combined.to_csv(config.combined_csv_path, index=False)
    combined.to_parquet(config.combined_parquet_path, index=False)
    combined_50m.to_csv(config.combined_fifty_million_csv_path, index=False)
    combined_50m.to_parquet(
        config.combined_fifty_million_parquet_path, index=False
    )

    summary = build_summary(combined, combined_50m, config)
    summary.to_csv(config.summary_csv_path, index=False)
    missingness = build_missingness(combined)
    missingness.to_csv(config.missingness_csv_path, index=False)

    manifest = build_manifest(
        config,
        df,
        subset_50m,
        combined,
        combined_50m,
        summary,
        missingness,
        state,
    )
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def build_summary(
    df: pd.DataFrame, subset_50m: pd.DataFrame, config: BackfillConfig
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in range(config.start_year, config.end_year + 1):
        year_df = df[df["release_year"] == year]
        year_50m = subset_50m[subset_50m["release_year"] == year]
        rows.append(
            {
                "release_year": year,
                "row_count_5m": int(len(year_df)),
                "row_count_50m": int(len(year_50m)),
                "min_worldwide_gross": (
                    int(year_df["worldwide_gross"].min())
                    if not year_df.empty
                    else None
                ),
                "max_worldwide_gross": (
                    int(year_df["worldwide_gross"].max())
                    if not year_df.empty
                    else None
                ),
                "missing_imdb_id": int(year_df["imdb_id"].isna().sum()),
                "missing_budget": int(
                    year_df["production_budget"].isna().sum()
                ),
                "zero_budget": int(
                    (year_df["production_budget"].fillna(0) == 0).sum()
                ),
                "missing_runtime": int(year_df["runtime"].isna().sum()),
                "missing_director": int(
                    year_df["director"].fillna("").eq("").sum()
                ),
                "missing_actors": int(
                    year_df["actors"].fillna("").eq("").sum()
                ),
                "missing_mpaa": int(year_df["mpaa"].fillna("").eq("").sum()),
                "missing_poster": int(year_df["poster_path"].isna().sum()),
                "missing_backdrop": int(year_df["backdrop_path"].isna().sum()),
                "missing_keywords": int(
                    year_df["keywords"].fillna("").eq("").sum()
                ),
                "missing_videos": int(
                    year_df["video_count"].fillna(0).eq(0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_missingness(df: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "imdb_id",
        "production_budget",
        "runtime",
        "director",
        "actors",
        "mpaa",
        "poster_path",
        "backdrop_path",
        "video_count",
        "keywords",
    ]
    rows: list[dict[str, Any]] = []
    for field_name in fields:
        series = df[field_name]
        if field_name == "video_count":
            missing = series.fillna(0).eq(0)
        elif series.dtype == object:
            missing = series.isna() | series.astype(str).str.strip().eq("")
        else:
            missing = series.isna()
        rows.append(
            {
                "field": field_name,
                "missing_count": int(missing.sum()),
                "missing_rate": float(missing.mean()),
            }
        )
    return pd.DataFrame(rows)


def build_manifest(
    config: BackfillConfig,
    df: pd.DataFrame,
    subset_50m: pd.DataFrame,
    combined: pd.DataFrame,
    combined_50m: pd.DataFrame,
    summary: pd.DataFrame,
    missingness: pd.DataFrame,
    state: BackfillState | None,
) -> dict[str, Any]:
    coverage_years = sorted(
        int(year) for year in combined["release_year"].dropna().unique()
    )
    missing_years = [
        year for year in range(config.start_year, config.end_year + 1)
        if year not in set(coverage_years)
    ]
    return {
        "created_at": utc_now(),
        "config": {
            **asdict(config),
            "append_responses": list(config.append_responses),
            "existing_web_data": str(config.existing_web_data),
            "output_dir": str(config.output_dir),
        },
        "outputs": {
            "raw_jsonl": str(config.raw_jsonl_path),
            "flat_csv_5m": str(config.flat_csv_path),
            "flat_parquet_5m": str(config.flat_parquet_path),
            "flat_csv_50m": str(config.fifty_million_csv_path),
            "flat_parquet_50m": str(config.fifty_million_parquet_path),
            "combined_csv_5m": str(config.combined_csv_path),
            "combined_parquet_5m": str(config.combined_parquet_path),
            "combined_csv_50m": str(config.combined_fifty_million_csv_path),
            "combined_parquet_50m": str(
                config.combined_fifty_million_parquet_path
            ),
            "summary_csv": str(config.summary_csv_path),
            "missingness_csv": str(config.missingness_csv_path),
        },
        "row_counts": {
            "raw_records": int(len(df)),
            "flat_5m": int(len(df)),
            "flat_50m": int(len(subset_50m)),
            "combined_5m": int(len(combined)),
            "combined_50m": int(len(combined_50m)),
        },
        "quality": {
            "duplicate_tmdb_ids": int(combined["tmdb_id"].duplicated().sum()),
            "below_min_revenue": int(
                (combined["worldwide_gross"] < config.min_revenue).sum()
            ),
            "missing_required": {
                "tmdb_id": int(combined["tmdb_id"].isna().sum()),
                "title": int(combined["title"].isna().sum()),
                "release_date": int(combined["release_date"].isna().sum()),
                "worldwide_gross": int(
                    combined["worldwide_gross"].isna().sum()
                ),
            },
            "coverage_years": coverage_years,
            "missing_years": missing_years,
            "missingness": missingness.to_dict(orient="records"),
        },
        "summary_by_year": summary.to_dict(orient="records"),
        "state": asdict(state) if state else None,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local-only rich TMDB backfill and derived flat exports."
        )
    )
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--min-revenue", type=positive_int, default=5_000_000)
    parser.add_argument("--page-limit", type=positive_int, default=100)
    parser.add_argument("--empty-page-stop", type=positive_int, default=5)
    parser.add_argument(
        "--existing-web-data",
        type=Path,
        default=Path("web/data/movies.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/tmdb/rich_backfill_1980_2026"),
    )
    parser.add_argument(
        "--append-responses",
        default=",".join(DEFAULT_APPEND_RESPONSES),
        help="Comma-separated TMDB append_to_response values.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flatten-only",
        action="store_true",
        help="Build flat outputs from existing raw JSONL without fetching.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> BackfillConfig:
    append_responses = tuple(
        part.strip()
        for part in str(args.append_responses).split(",")
        if part.strip()
    )
    return BackfillConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        min_revenue=args.min_revenue,
        page_limit=args.page_limit,
        stop_after_consecutive_empty_pages=args.empty_page_stop,
        append_responses=append_responses,
        existing_web_data=args.existing_web_data,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = config_from_args(args)
    state = None if args.flatten_only else run_backfill(config)
    manifest = build_flat_outputs(config, state=state)
    logger.info("Backfill complete: %s", config.manifest_path)
    logger.info("Rows: %s", manifest["row_counts"])
    logger.info("Missing years: %s", manifest["quality"]["missing_years"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
