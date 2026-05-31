#!/usr/bin/env python3
"""Historical TMDB discovery runner.

Kept as a record of the original data-gathering workflow. The discovery logic
now lives in the maintained package module ``box_office.ingestion.tmdb_discovery``;
this script is a thin CLI over it. The maintained ingestion path is the
``box-office-ingest`` CLI documented in ``data/README.md``.
"""

import csv
import logging
import os

import pandas as pd
from dotenv import load_dotenv

from box_office.ingestion.tmdb_discovery import (
    discover_movies,
    filter_new_movies,
    get_existing_ids,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
INPUT_CSV_PATH = "data/raw/box_office_movies/prod_movie_dataset_enriched_v8.csv"
OUTPUT_CSV_PATH = "data/external/tmdb/suggested_movies_2000_2019_extra.csv"


def save_suggestions(suggestions: list[dict], output_path: str) -> None:
    """Write discovered movies to CSV and log the top 10 by vote count."""
    if not suggestions:
        logger.info("No suggestions to save.")
        return

    df = pd.DataFrame(suggestions)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8-sig")
    logger.info("Saved %d suggested movies to '%s'.", len(suggestions), output_path)

    logger.info("Top 10 most voted new English movies:")
    for i, movie in enumerate(suggestions[:10], 1):
        vote_count = movie.get("vote_count", 0)
        vote_avg = movie.get("vote_average", 0)
        release_year = (movie.get("release_date") or "N/A")[:4]
        logger.info(
            "  %2d. %s (%s) - votes: %s (avg: %.1f)",
            i,
            movie.get("title"),
            release_year,
            f"{vote_count:,}",
            vote_avg,
        )


def main() -> None:
    if not os.getenv("TMDB_API_TOKEN"):
        logger.error("Please set TMDB_API_TOKEN in your environment.")
        return

    existing_ids, existing_titles = get_existing_ids(INPUT_CSV_PATH)
    logger.info(
        "Loaded %d existing TMDB IDs and %d titles.",
        len(existing_ids),
        len(existing_titles),
    )

    candidates = discover_movies(existing_ids)
    logger.info("Discovered %d new English-language movies.", len(candidates))

    new_movies = filter_new_movies(existing_ids, existing_titles, candidates)
    logger.info("Found %d unique new movies after title dedup.", len(new_movies))

    save_suggestions(new_movies, OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
