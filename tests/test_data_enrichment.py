"""Tests for the heuristic data enrichment module."""

import pytest
import pandas as pd

from box_office.ingestion.data_enrichment import (
    HeuristicEnricher,
    enrich_dataframe,
    MAJOR_FRANCHISE_KEYWORDS,
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


class TestNoTargetLeakageAtEnrichment:
    """Enrichment must not synthesize features from worldwide_gross."""

    def test_no_social_media_buzz_column(self, enricher):
        df = pd.DataFrame(
            {
                "title": ["Some Movie"],
                "worldwide_gross": [500_000_000],
                "production_budget": [100_000_000],
            }
        )
        result = enricher.enrich(df)
        assert "social_media_buzz" not in result.columns

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


class TestFranchiseRating:
    @pytest.mark.parametrize(
        "title,revenue,budget,expected_rating",
        [
            (
                "Avengers: Infinity War",
                2_000_000_000,
                300_000_000,
                2,
            ),  # major franchise, high revenue
            (
                "X-Men: Dark Phoenix",
                252_000_000,
                200_000_000,
                1,
            ),  # major franchise, lower revenue
            ("Original Standalone Film", 500_000_000, 100_000_000, 0),  # non-franchise
        ],
    )
    def test_franchise_rating_by_type(
        self, enricher, title, revenue, budget, expected_rating
    ):
        df = pd.DataFrame(
            {
                "title": [title],
                "worldwide_gross": [revenue],
                "production_budget": [budget],
            }
        )
        result = enricher.enrich(df)
        assert result["franchise_rating"].iloc[0] == expected_rating

    @pytest.mark.parametrize(
        "title,is_franchise",
        [
            ("Spider-Man: No Way Home", True),
            ("The Dark Knight Rises", True),
            ("Toy Story 4", True),
            ("The Matrix Resurrections", True),
            ("Original Movie Title", False),
            ("Some Random Film", False),
        ],
    )
    def test_franchise_keywords_detection(self, enricher, title, is_franchise):
        rating = enricher._assign_franchise_rating(title, 500_000_000)
        if is_franchise:
            assert rating >= 1, f"Expected {title} to be franchise"
        else:
            assert rating == 0, f"Expected {title} to be non-franchise"


class TestEnrichDataframeFunction:
    def test_enrich_dataframe_works(self, sample_movie_data):
        result = enrich_dataframe(sample_movie_data, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert "ad_budget" in result.columns
        assert "franchise_rating" in result.columns
        assert "social_media_buzz" not in result.columns

    def test_reproducible_with_seed(self, sample_movie_data):
        result1 = enrich_dataframe(sample_movie_data, seed=42)
        result2 = enrich_dataframe(sample_movie_data, seed=42)

        pd.testing.assert_frame_equal(result1, result2)


class TestMajorFranchiseKeywords:
    def test_keywords_list_not_empty(self):
        assert len(MAJOR_FRANCHISE_KEYWORDS) > 0

    def test_major_franchises_included(self):
        expected_franchises = [
            "Star Wars",
            "Marvel",
            "Spider-Man",
            "Batman",
            "Harry Potter",
            "Jurassic",
            "Avengers",
        ]
        keywords_lower = [k.lower() for k in MAJOR_FRANCHISE_KEYWORDS]

        for franchise in expected_franchises:
            found = any(franchise.lower() in k for k in keywords_lower)
            assert found, f"Expected {franchise} in franchise keywords"
