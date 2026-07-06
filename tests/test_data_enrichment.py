"""Tests for the heuristic data enrichment module."""

import pytest
import pandas as pd

from box_office.ingestion.data_enrichment import (
    HeuristicEnricher,
    enrich_dataframe,
)


@pytest.fixture
def sample_movie_data():
    return pd.DataFrame(
        {
            "title": [
                "Avengers: Endgame",
                "The Dark Knight",
                "Indie Film",
                "Unknown Movie",
            ],
            "worldwide_gross": [2_797_000_000, 1_005_000_000, 25_000_000, None],
            "revenue": [2_797_000_000, 1_005_000_000, 25_000_000, None],
            "production_budget": [356_000_000, 185_000_000, 0, 50_000_000],
            "budget": [356_000_000, 185_000_000, 0, 50_000_000],
        }
    )


@pytest.fixture
def enricher():
    return HeuristicEnricher(seed=42)


class TestHeuristicEnricher:
    def test_init_with_seed(self):
        enricher = HeuristicEnricher(seed=123)
        assert enricher.seed == 123

    def test_enrich_returns_dataframe(self, enricher, sample_movie_data):
        result = enricher.enrich(sample_movie_data)
        assert isinstance(result, pd.DataFrame)

    def test_enrich_preserves_rows(self, enricher, sample_movie_data):
        result = enricher.enrich(sample_movie_data)
        assert len(result) == len(sample_movie_data)


class TestNoLeakageAtEnrichment:
    """Enrichment must not synthesize post-release features."""

    def test_no_leakage_columns(self, enricher):
        df = pd.DataFrame(
            {
                "title": ["Some Movie"],
                "worldwide_gross": [500_000_000],
                "production_budget": [100_000_000],
            }
        )
        result = enricher.enrich(df)
        assert "social_media_buzz" not in result.columns
        assert "franchise_rating" not in result.columns
        assert "rating" not in result.columns
        assert "votes" not in result.columns

    def test_missing_budget_left_as_is(self, enricher):
        """Missing or zero production_budget is NOT imputed from worldwide_gross."""
        df = pd.DataFrame(
            {
                "title": ["Zero Budget", "Missing Budget"],
                "worldwide_gross": [100_000_000, 200_000_000],
                "production_budget": [0, None],
            }
        )
        result = enricher.enrich(df)
        assert result["production_budget"].iloc[0] == 0
        assert pd.isna(result["production_budget"].iloc[1])


class TestAdBudgetGeneration:
    def test_ad_budget_is_percentage_of_production(self, enricher):
        """ad_budget lands in the 40-55% band for a $100M production budget."""
        df = pd.DataFrame(
            {
                "title": ["Test Movie"],
                "worldwide_gross": [200_000_000],
                "production_budget": [100_000_000],
            }
        )
        result = enricher.enrich(df)
        ad_budget = result["ad_budget"].iloc[0]
        assert 40_000_000 <= ad_budget <= 55_000_000

    def test_zero_budget_gets_zero_ad_budget(self, enricher):
        """Zero production_budget short-circuits ``_generate_ad_budget`` to 0."""
        assert enricher._generate_ad_budget(0) == 0
        assert enricher._generate_ad_budget(0.0) == 0

        df = pd.DataFrame(
            {
                "title": ["No Budget Film"],
                "worldwide_gross": [0],
                "production_budget": [0],
            }
        )
        result = enricher.enrich(df)
        assert result["production_budget"].iloc[0] == 0
        assert result["ad_budget"].iloc[0] == 0

    def test_high_budget_gets_higher_ratio(self, enricher):
        df_low = pd.DataFrame(
            {
                "title": ["Low Budget"],
                "worldwide_gross": [20_000_000],
                "production_budget": [5_000_000],
            }
        )
        df_high = pd.DataFrame(
            {
                "title": ["High Budget"],
                "worldwide_gross": [500_000_000],
                "production_budget": [250_000_000],
            }
        )

        result_low = enricher.enrich(df_low)
        result_high = enricher.enrich(df_high)

        ratio_low = result_low["ad_budget"].iloc[0] / 5_000_000
        ratio_high = result_high["ad_budget"].iloc[0] / 250_000_000

        assert ratio_high > ratio_low


class TestEnrichDataframeFunction:
    def test_enrich_dataframe_works(self, sample_movie_data):
        result = enrich_dataframe(sample_movie_data, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert "ad_budget" in result.columns
        assert "social_media_buzz" not in result.columns
        assert "franchise_rating" not in result.columns

    def test_reproducible_with_seed(self, sample_movie_data):
        result1 = enrich_dataframe(sample_movie_data, seed=42)
        result2 = enrich_dataframe(sample_movie_data, seed=42)

        pd.testing.assert_frame_equal(result1, result2)
