from __future__ import annotations

import pandas as pd

from box_office.movie_data_quality import clean_movie_source_data


def _row(
    tmdb_id: int,
    title: str,
    *,
    imdb_id: str | None = "tt0000001",
    budget: float | None = 1_000_000,
    gross: float | None = 10_000_000,
    vote_count: int | None = 100,
    runtime: int | None = 100,
) -> dict:
    return {
        "tmdb_id": tmdb_id,
        "imdb_id": imdb_id,
        "title": title,
        "release_year": 2024,
        "release_date": "2024-01-01",
        "production_budget": budget,
        "production_budget_source": "tmdb" if budget is not None else "missing",
        "production_budget_was_missing": budget is None,
        "worldwide_gross": gross,
        "runtime": runtime,
        "vote_count": vote_count,
        "popularity": 1.0,
    }


def test_corrects_bad_financials_without_dropping_movie_rows():
    cleaned, audit = clean_movie_source_data(
        pd.DataFrame(
            [
                _row(270650, "Oy Vey! My Son Is Gay!", budget=2, gross=6_000_007),
                _row(1289601, "Life After Fighting", budget=234_325, gross=59_530_099),
            ]
        )
    )

    rows = cleaned.set_index("tmdb_id")
    assert pd.isna(rows.loc[270650, "production_budget"])
    assert rows.loc[270650, "production_budget_source"] == "missing"
    assert rows.loc[270650, "worldwide_gross"] == 89_507
    assert rows.loc[1289601, "production_budget"] == 234_325
    assert rows.loc[1289601, "worldwide_gross"] == 5_727
    assert set(audit["tmdb_id"]) == {270650, 1289601}


def test_excludes_non_movie_rows_from_source():
    cleaned, audit = clean_movie_source_data(
        pd.DataFrame(
            [
                _row(565916, "The Visual Effects of 'Scary Movie 4'"),
                _row(1, "A Movie"),
            ]
        )
    )

    assert cleaned["tmdb_id"].tolist() == [1]
    assert audit.loc[audit["tmdb_id"] == 565916, "action"].iloc[0] == "exclude_row"


def test_nulls_tiny_budget_and_high_gross_without_footprint():
    cleaned, audit = clean_movie_source_data(
        pd.DataFrame(
            [
                _row(2, "Tiny Budget", budget=999, gross=500_000),
                _row(
                    3,
                    "No Footprint Hit",
                    imdb_id=None,
                    budget=20_000,
                    gross=2_000_000,
                    vote_count=0,
                ),
            ]
        )
    )

    rows = cleaned.set_index("tmdb_id")
    assert pd.isna(rows.loc[2, "production_budget"])
    assert rows.loc[2, "worldwide_gross"] == 500_000
    assert rows.loc[3, "production_budget"] == 20_000
    assert pd.isna(rows.loc[3, "worldwide_gross"])
    assert set(audit["tmdb_id"]) == {2, 3}
