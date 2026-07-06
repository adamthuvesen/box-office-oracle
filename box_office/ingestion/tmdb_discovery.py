"""TMDB movie discovery.

Discovers candidate movies from the TMDB API and enriches each with details,
keywords, credits, and US certification. This is the implementation behind the
``box-office-ingest`` CLI.

Only ``get_existing_ids``, ``discover_movies``, and ``filter_new_movies`` are
public; the per-movie fetch helpers are internal.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

TMDB_API_URL = "https://api.themoviedb.org/3"

MIN_REVENUE_USD = 50_000_000
_EXCLUDED_GENRES = frozenset({"Documentary"})


def _auth_headers() -> Dict[str, str]:
    """Build TMDB auth headers, reading the token lazily so importing this
    module never requires a configured environment."""
    token = os.getenv("TMDB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "TMDB_API_TOKEN is not set; export it before running discovery."
        )
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def get_existing_ids(input_path: str) -> Tuple[Set[int], Set[str]]:
    """Return the (tmdb_id, lowercased-title) sets already present in a dataset."""
    logger.info("Loading existing TMDB IDs from %s", input_path)
    df = pd.read_csv(input_path, usecols=["tmdb_id", "title"])
    df = df.dropna(subset=["tmdb_id"])
    df["tmdb_id"] = df["tmdb_id"].astype(int)
    return set(df["tmdb_id"]), set(df["title"].str.lower())


def get_movie_details(session: requests.Session, tmdb_id: int) -> Dict:
    """Fetch detailed movie information from the TMDB API."""
    url = f"{TMDB_API_URL}/movie/{tmdb_id}"
    try:
        resp = session.get(url, headers=_auth_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error("Error fetching details for movie %s: %s", tmdb_id, e)
        return {}


def get_movie_keywords(session: requests.Session, tmdb_id: int) -> str:
    """Fetch comma-separated movie keywords from the TMDB API."""
    url = f"{TMDB_API_URL}/movie/{tmdb_id}/keywords"
    try:
        resp = session.get(url, headers=_auth_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return ", ".join(kw["name"] for kw in data.get("keywords", []))
    except requests.RequestException as e:
        logger.error("Error fetching keywords for movie %s: %s", tmdb_id, e)
        return ""


def get_movie_credits(session: requests.Session, tmdb_id: int) -> Dict:
    """Fetch the director and top-3 actors for a movie.

    Returns a dict with ``director`` (str) and ``actors`` (comma-separated top 3).
    """
    url = f"{TMDB_API_URL}/movie/{tmdb_id}/credits"
    try:
        resp = session.get(url, headers=_auth_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()

        director = ""
        for crew_member in data.get("crew", []):
            if crew_member.get("job") == "Director":
                director = crew_member.get("name", "")
                break

        cast_sorted = sorted(data.get("cast", []), key=lambda x: x.get("order", 999))
        actors = ", ".join(actor.get("name", "") for actor in cast_sorted[:3])

        return {"director": director, "actors": actors}
    except requests.RequestException as e:
        logger.error("Error fetching credits for movie %s: %s", tmdb_id, e)
        return {"director": "", "actors": ""}


def get_movie_certification(session: requests.Session, tmdb_id: int) -> str:
    """Fetch the US MPAA rating (G, PG, PG-13, R, NC-17) or 'Not Rated'."""
    url = f"{TMDB_API_URL}/movie/{tmdb_id}/release_dates"
    try:
        resp = session.get(url, headers=_auth_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for country_release in data.get("results", []):
            if country_release.get("iso_3166_1") == "US":
                for release in country_release.get("release_dates", []):
                    certification = release.get("certification", "")
                    if certification:
                        return certification

        return "Not Rated"
    except requests.RequestException as e:
        logger.error("Error fetching certification for movie %s: %s", tmdb_id, e)
        return "Not Rated"


def extract_key_fields(
    movie_data: Dict,
    keywords: str = "",
    credits: Dict | None = None,
    mpaa: str = "",
) -> Dict:
    """Flatten TMDB movie data into ingestion column names."""
    if not movie_data:
        return {}

    credits = credits or {"director": "", "actors": ""}

    genres = ", ".join(genre["name"] for genre in movie_data.get("genres", []))
    production_companies = ", ".join(
        company["name"] for company in movie_data.get("production_companies", [])
    )
    production_countries = ", ".join(
        country["name"] for country in movie_data.get("production_countries", [])
    )
    spoken_languages = ", ".join(
        lang["english_name"] for lang in movie_data.get("spoken_languages", [])
    )

    return {
        "id": movie_data.get("id"),
        "title": movie_data.get("title"),
        "status": movie_data.get("status"),
        "release_date": movie_data.get("release_date"),
        "revenue": movie_data.get("revenue"),
        "runtime": movie_data.get("runtime"),
        "adult": movie_data.get("adult"),
        "backdrop_path": movie_data.get("backdrop_path"),
        "budget": movie_data.get("budget"),
        "homepage": movie_data.get("homepage"),
        "imdb_id": movie_data.get("imdb_id"),
        "original_language": movie_data.get("original_language"),
        "original_title": movie_data.get("original_title"),
        "overview": movie_data.get("overview"),
        "poster_path": movie_data.get("poster_path"),
        "tagline": movie_data.get("tagline"),
        "genres": genres,
        "production_companies": production_companies,
        "production_countries": production_countries,
        "spoken_languages": spoken_languages,
        "keywords": keywords,
        "director": credits.get("director", ""),
        "actors": credits.get("actors", ""),
        "mpaa": mpaa,
    }


def discover_movies(
    existing_ids: Set[int],
    start_year: int = 2000,
    end_year: int = 2019,
    page_limit: int = 10,
    min_revenue: int = MIN_REVENUE_USD,
) -> List[Dict]:
    """Discover English-language movies via the TMDB Discover API.

    Fetches details only for movies not already in ``existing_ids``, filters to
    English originals at or above ``min_revenue`` (excluding documentaries), and
    returns the enriched records sorted by revenue descending.
    """
    session = requests.Session()
    all_movies: List[Dict] = []

    for year in range(start_year, end_year + 1):
        logger.info("Processing year %d...", year)
        year_movies: List[Dict] = []

        for page in range(1, page_limit + 1):
            url = f"{TMDB_API_URL}/discover/movie"
            params = {
                "primary_release_year": year,
                "with_original_language": "en",
                "sort_by": "revenue.desc",
                "page": page,
            }

            try:
                resp = session.get(
                    url, headers=_auth_headers(), params=params, timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.error("Discover error for %d page %d: %s", year, page, e)
                break

            page_movies = []
            skipped_existing = 0

            for item in data.get("results", []):
                tmdb_id = item.get("id")
                if not tmdb_id:
                    continue

                if tmdb_id in existing_ids:
                    skipped_existing += 1
                    continue

                movie_details = get_movie_details(session, tmdb_id)
                if not movie_details:
                    continue

                if movie_details.get("original_language") != "en":
                    continue

                revenue = movie_details.get("revenue", 0)
                if revenue < min_revenue:
                    continue

                genre_names = {g["name"] for g in movie_details.get("genres", [])}
                if genre_names & _EXCLUDED_GENRES:
                    continue

                keywords = get_movie_keywords(session, tmdb_id)
                credits = get_movie_credits(session, tmdb_id)
                mpaa = get_movie_certification(session, tmdb_id)

                movie_fields = extract_key_fields(
                    movie_details, keywords=keywords, credits=credits, mpaa=mpaa
                )
                if movie_fields:
                    page_movies.append(movie_fields)

                # Rate limit: several API calls per movie.
                time.sleep(0.35)

            year_movies.extend(page_movies)
            logger.info(
                "  Year %d page %d: %d new, %d skipped (existing)",
                year,
                page,
                len(page_movies),
                skipped_existing,
            )

            if page >= data.get("total_pages", 0):
                break

            time.sleep(0.25)

        year_movies.sort(key=lambda x: x.get("revenue", 0), reverse=True)
        all_movies.extend(year_movies)
        logger.info("Found %d new English movies for %d", len(year_movies), year)

    all_movies.sort(key=lambda x: x.get("revenue", 0), reverse=True)
    return all_movies


def filter_new_movies(
    existing_ids: Set[int], existing_titles: Set[str], candidates: List[Dict]
) -> List[Dict]:
    """Return candidates whose id and lowercased title are both new, deduped by
    id and preserving vote-count order."""
    new: List[Dict] = []
    seen: Set[int] = set()

    for c in candidates:
        mid = c.get("id")
        title_lower = c.get("title", "").lower() if c.get("title") else ""

        if mid not in existing_ids and title_lower not in existing_titles:
            if mid not in seen:
                new.append(c)
                seen.add(mid)

    return new


__all__ = ["get_existing_ids", "discover_movies", "filter_new_movies"]
