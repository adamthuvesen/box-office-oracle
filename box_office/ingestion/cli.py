"""
Unified movie data ingestion CLI.

Orchestrates the full pipeline: TMDB Discovery → Snowflake Load.

Usage:
    # Full pipeline
    box-office-ingest --start-year 2024 --load-to-snowflake

    # Discovery only
    box-office-ingest --discover-only --output movies_2024.csv
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from box_office.ingestion import tmdb_discovery

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _write_csv(df: pd.DataFrame, output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run_discovery(
    start_year: int,
    end_year: int,
    min_revenue: int,
    page_limit: int,
    existing_csv: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Run TMDB discovery to find new movies."""
    get_existing_ids = tmdb_discovery.get_existing_ids
    discover_movies = tmdb_discovery.discover_movies
    filter_new_movies = tmdb_discovery.filter_new_movies

    logger.info(f"Starting TMDB discovery: {start_year}-{end_year}")
    logger.info(f"Min revenue filter: ${min_revenue:,}")

    if existing_csv and Path(existing_csv).exists():
        existing_ids, existing_titles = get_existing_ids(existing_csv)
        logger.info(f"Loaded {len(existing_ids)} existing movies to skip")
    else:
        existing_ids, existing_titles = set(), set()
        logger.info("No existing dataset - will fetch all movies")

    candidates = discover_movies(
        existing_ids=existing_ids,
        start_year=start_year,
        end_year=end_year,
        page_limit=page_limit,
        min_revenue=min_revenue,
    )
    logger.info(f"Discovered {len(candidates)} movies from TMDB")

    new_movies = filter_new_movies(existing_ids, existing_titles, candidates)
    logger.info(f"Found {len(new_movies)} unique new movies")

    df = pd.DataFrame(new_movies)

    if output_path:
        path = _write_csv(df, output_path)
        logger.info(f"Saved discovery results to: {path}")

    return df


def run_snowflake_load(
    csv_path: str,
    table_name: str = "BOX_OFFICE_V4",
    schema: str = "RAW",
    mode: str = "merge",
) -> dict:
    """Load discovery output to Snowflake."""
    from box_office.utils.snowflake_loader import SnowflakeLoader

    logger.info(f"Loading to Snowflake: {schema}.{table_name}")

    loader = SnowflakeLoader(schema=schema)
    result = loader.load_csv_to_raw(csv_path=csv_path, table_name=table_name, mode=mode)

    logger.info(f"Snowflake load complete: {result}")
    return result


def prepare_for_snowflake(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for Snowflake loading by mapping columns.

    Maps TMDB discovery output to expected Snowflake schema.
    """
    column_mapping = {
        "id": "tmdb_id",
        "budget": "production_budget",
        "revenue": "worldwide_gross",
        "production_companies": "production_company",
    }

    df = df.copy()

    # Only rename if target doesn't already exist (preserve pre-mapped cols).
    rename_map = {
        k: v
        for k, v in column_mapping.items()
        if k in df.columns and v not in df.columns
    }
    df = df.rename(columns=rename_map)

    required_cols = [
        "tmdb_id",
        "imdb_id",
        "title",
        "release_date",
        "original_language",
        "production_countries",
        "genres",
        "production_budget",
        "director",
        "actors",
        "mpaa",
        "release_type",
        "runtime",
        "overview",
        "tagline",
        "keywords",
        "production_company",
        "release_year",
        "worldwide_gross",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    for col in missing_cols:
        df[col] = None

    # Extract release_year from release_date if missing
    if "release_year" in df.columns and df["release_year"].isna().all():
        if "release_date" in df.columns:
            df["release_year"] = pd.to_datetime(
                df["release_date"], errors="coerce"
            ).dt.year

    return df[required_cols]


def main():
    """Main entry point for ingestion CLI."""
    parser = argparse.ArgumentParser(
        description="Movie data ingestion pipeline: TMDB → Snowflake",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: discover 2024 movies and load to Snowflake
  box-office-ingest --start-year 2024 --load-to-snowflake

  # Discovery only (save to CSV)
  box-office-ingest --discover-only --start-year 2024 --output movies_2024.csv

  # Custom year range
  box-office-ingest --start-year 2020 --end-year 2023 --load-to-snowflake
        """,
    )

    # Mode selection
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only run TMDB discovery, skip Snowflake loading",
    )

    # Discovery options
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="Start year for movie discovery (default: 2024)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year for discovery (default: current year)",
    )
    parser.add_argument(
        "--min-revenue",
        type=int,
        default=50_000_000,
        help="Minimum worldwide revenue filter (default: 50M)",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=10,
        help="Max pages per year from TMDB API (default: 10)",
    )
    parser.add_argument(
        "--existing-csv", help="Path to existing dataset for deduplication"
    )

    # I/O options
    parser.add_argument("--output", dest="output_csv", help="Output CSV path")

    # Snowflake options
    parser.add_argument(
        "--load-to-snowflake",
        action="store_true",
        help="Load results to Snowflake after discovery",
    )
    parser.add_argument(
        "--table",
        default="BOX_OFFICE_V4",
        help="Target Snowflake table (default: BOX_OFFICE_V4)",
    )
    parser.add_argument(
        "--schema",
        default="RAW",
        help="Target Snowflake schema (default: RAW)",
    )
    parser.add_argument(
        "--mode",
        choices=["merge", "overwrite"],
        default="merge",
        help="Snowflake load mode (default: merge)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default end year to current year
    if args.end_year is None:
        args.end_year = datetime.now().year

    # Generate default output path
    if args.output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_csv = (
            f"data/generated/tmdb/ingested_movies_"
            f"{args.start_year}_{args.end_year}_{timestamp}.csv"
        )

    logger.info("Movie Data Ingestion Pipeline")

    try:
        if args.discover_only:
            logger.info("Mode: Discovery only")
            logger.info(f"Years: {args.start_year}-{args.end_year}")
            logger.info(f"Output: {args.output_csv}")

            df = run_discovery(
                start_year=args.start_year,
                end_year=args.end_year,
                min_revenue=args.min_revenue,
                page_limit=args.page_limit,
                existing_csv=args.existing_csv,
                output_path=args.output_csv,
            )
            logger.info(f"Discovered {len(df)} movies")

        else:
            logger.info("Mode: Full pipeline")
            logger.info(f"Years: {args.start_year}-{args.end_year}")
            logger.info(f"Load to Snowflake: {args.load_to_snowflake}")

            logger.info("\n--- Step 1: TMDB Discovery ---")
            df = run_discovery(
                start_year=args.start_year,
                end_year=args.end_year,
                min_revenue=args.min_revenue,
                page_limit=args.page_limit,
                existing_csv=args.existing_csv,
            )

            if len(df) == 0:
                logger.info("No new movies discovered. Exiting.")
                return 0

            logger.info("\n--- Step 2: Prepare for Snowflake ---")
            df = prepare_for_snowflake(df)

            path = _write_csv(df, args.output_csv)
            logger.info(f"Saved to: {path}")

            if args.load_to_snowflake:
                logger.info("\n--- Step 3: Load to Snowflake ---")
                result = run_snowflake_load(
                    csv_path=args.output_csv,
                    table_name=args.table,
                    schema=args.schema,
                    mode=args.mode,
                )
                logger.info(f"Load result: {result}")

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
