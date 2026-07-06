"""
Heuristic data enrichment for movie datasets.

Fills pre-release-safe fields using rule-based estimates when TMDB data is
incomplete. Outcome-derived and post-release fields are deliberately not
created here.
"""

import logging
import random

import pandas as pd

logger = logging.getLogger(__name__)


class HeuristicEnricher:
    """Fill missing pre-release-safe fields with simple local heuristics."""

    def __init__(self, seed: int = 42):
        """Initialize enricher with a local RNG."""
        self.seed = seed
        self._rng = random.Random(seed)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all heuristic enrichment steps and return the enriched copy."""
        df = df.copy()
        logger.info(f"Enriching {len(df)} movies with heuristics...")

        df = self._fill_ad_budget(df)

        logger.info("Heuristic enrichment complete")
        return df

    def _fill_ad_budget(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ad_budget as a percentage of production_budget."""
        if "ad_budget" not in df.columns:
            df["ad_budget"] = None

        budget_col = self._get_budget_column(df)
        if budget_col is None:
            logger.warning("Budget column not found, skipping ad_budget")
            return df

        missing_mask = df["ad_budget"].isna()
        missing_count = int(missing_mask.sum())

        if missing_count > 0:
            logger.info(f"Generating ad_budget for {missing_count} movies")
            df.loc[missing_mask, "ad_budget"] = df.loc[missing_mask, budget_col].apply(
                self._generate_ad_budget
            )

        return df

    def _get_budget_column(self, df: pd.DataFrame) -> str | None:
        """Find the budget column (production_budget or budget)."""
        for col in ["production_budget", "budget"]:
            if col in df.columns:
                return col
        return None

    def _generate_ad_budget(self, production_budget: float) -> float:
        """Generate ad_budget as a percentage of production_budget."""
        if pd.isna(production_budget) or production_budget <= 0:
            return 0

        if production_budget < 10_000_000:
            multiplier = self._rng.uniform(0.20, 0.35)
        elif production_budget < 50_000_000:
            multiplier = self._rng.uniform(0.30, 0.40)
        elif production_budget < 100_000_000:
            multiplier = self._rng.uniform(0.40, 0.50)
        elif production_budget < 200_000_000:
            multiplier = self._rng.uniform(0.45, 0.55)
        else:
            multiplier = self._rng.uniform(0.5, 0.70)

        return round(production_budget * multiplier)


def enrich_dataframe(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Convenience wrapper around :class:`HeuristicEnricher`."""
    enricher = HeuristicEnricher(seed=seed)
    return enricher.enrich(df)
