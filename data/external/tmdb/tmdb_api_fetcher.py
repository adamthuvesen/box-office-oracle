import logging
import os
import time
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class TMDBMovieFetcher:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.themoviedb.org/3"
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {api_token}",
        }

    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Search for a movie by title and optionally year"""
        search_url = f"{self.base_url}/search/movie"
        params = {"query": title, "language": "en-US"}
        if year:
            params["year"] = year

        try:
            response = requests.get(search_url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if data["results"]:
                # Return the first (most relevant) result
                return data["results"][0]
            return None

        except requests.exceptions.RequestException as e:
            logger.error("Error searching for '%s': %s", title, e)
            return None

    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Get detailed movie information by movie ID"""
        detail_url = f"{self.base_url}/movie/{movie_id}"
        params = {"language": "en-US"}

        try:
            response = requests.get(detail_url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error("Error fetching details for movie ID %s: %s", movie_id, e)
            return None

    def fetch_movie_by_title(
        self, title: str, year: Optional[int] = None
    ) -> Optional[Dict]:
        """Search for a movie and fetch its detailed information"""
        logger.info("Searching for: '%s'", title)

        search_result = self.search_movie(title, year)
        if not search_result:
            logger.warning("No results found for '%s'", title)
            return None

        movie_id = search_result["id"]
        found_title = search_result["title"]
        release_date = search_result.get("release_date", "Unknown")

        logger.info("Found '%s' (%s) - ID: %s", found_title, release_date, movie_id)

        details = self.get_movie_details(movie_id)
        if details:
            logger.info("Fetched detailed information for '%s'", found_title)

        return details

    def extract_key_fields(self, movie_data: Dict) -> Dict:
        """Extract key fields from movie data for CSV export"""
        if not movie_data:
            return {}

        # Extract genres as comma-separated string
        genres = ", ".join([genre["name"] for genre in movie_data.get("genres", [])])

        # Extract production companies as comma-separated string
        production_companies = ", ".join(
            [company["name"] for company in movie_data.get("production_companies", [])]
        )

        # Extract production countries as comma-separated string
        production_countries = ", ".join(
            [country["name"] for country in movie_data.get("production_countries", [])]
        )

        # Extract spoken languages as comma-separated string
        spoken_languages = ", ".join(
            [lang["english_name"] for lang in movie_data.get("spoken_languages", [])]
        )

        # Keywords would need to be fetched separately (not included in movie details endpoint)
        # For now, set as empty string
        keywords = ""

        # Return in the exact order specified
        return {
            "id": movie_data.get("id"),
            "title": movie_data.get("title"),
            "vote_average": movie_data.get("vote_average"),
            "vote_count": movie_data.get("vote_count"),
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
            "popularity": movie_data.get("popularity"),
            "poster_path": movie_data.get("poster_path"),
            "tagline": movie_data.get("tagline"),
            "genres": genres,
            "production_companies": production_companies,
            "production_countries": production_countries,
            "spoken_languages": spoken_languages,
            "keywords": keywords,
        }


def main():
    # Get TMDB API token from environment variable
    API_TOKEN = os.environ.get("TMDB_API_TOKEN")
    if not API_TOKEN:
        raise ValueError(
            "TMDB_API_TOKEN environment variable is required. "
            "Please set it with: export TMDB_API_TOKEN='your-token-here'"
        )

    # Movies to fetch
    movies_to_fetch = [
        "Dune: Part Two",
        "Moana 2",
        "Despicable Me 4",
        "Wicked",
        "Kung Fu Panda 4",
        "Venom: The Last Dance",
        "Gladiator II",
    ]

    fetcher = TMDBMovieFetcher(API_TOKEN)

    all_movie_data = []
    successful_fetches = []
    failed_fetches = []

    logger.info("Starting to fetch %d movies from TMDB API...", len(movies_to_fetch))

    for i, title in enumerate(movies_to_fetch, 1):
        logger.info("[%d/%d] Processing: %s", i, len(movies_to_fetch), title)

        movie_data = fetcher.fetch_movie_by_title(title)

        if movie_data:
            all_movie_data.append(fetcher.extract_key_fields(movie_data))
            successful_fetches.append(title)
        else:
            failed_fetches.append(title)

        # Be nice to the API - add a small delay
        time.sleep(0.25)

    if all_movie_data:
        df = pd.DataFrame(all_movie_data)
        df.to_csv("tmdb_movie_details.csv", index=False)
        logger.info("Saved %d movies to 'tmdb_movie_details.csv'", len(all_movie_data))

    logger.info(
        "Fetch summary: %d/%d successful (%.1f%%)",
        len(successful_fetches),
        len(movies_to_fetch),
        len(successful_fetches) / len(movies_to_fetch) * 100,
    )

    if failed_fetches:
        logger.warning("Failed fetches: %s", ", ".join(failed_fetches))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
