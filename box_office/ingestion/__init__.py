"""Movie data ingestion pipeline.

Modules:
- tmdb_discovery: Discover movies from TMDB API
- data_enrichment: Enrich movies with heuristic data
- cli: Unified ingestion CLI
"""

from box_office.ingestion.data_enrichment import HeuristicEnricher

__all__ = ["HeuristicEnricher"]
