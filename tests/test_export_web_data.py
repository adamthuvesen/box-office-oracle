from __future__ import annotations

import math

import pandas as pd

from scripts.export_web_data import build_movie_records


def _row(
    tmdb_id: int,
    title: str,
    *,
    imdb_id: str | None = "tt0000001",
    budget: float | None = 1_000_000,
    gross: float | None = 10_000_000,
    vote_count: int | None = 100,
) -> dict:
    return {
        "tmdb_id": tmdb_id,
        "imdb_id": imdb_id,
        "title": title,
        "release_date": "2024-01-01",
        "release_year": 2024,
        "genres": "Drama",
        "director": "Director",
        "actors": "Actor One, Actor Two",
        "mpaa": "PG-13",
        "runtime": 100,
        "production_budget": budget,
        "production_budget_source": "tmdb",
        "production_company": "Studio",
        "overview": "Overview",
        "tagline": "",
        "keywords": "keyword",
        "worldwide_gross": gross,
        "poster_path": "/poster.jpg",
        "backdrop_path": "/backdrop.jpg",
        "vote_count": vote_count,
    }


def _records(*rows: dict) -> dict[int, dict]:
    return {
        record["tmdb_id"]: record
        for record in build_movie_records(pd.DataFrame(rows), no_gross_ids=set())
    }


def test_financial_corrections_replace_confirmed_bad_grosses():
    records = _records(
        _row(270650, "Oy Vey! My Son Is Gay!", budget=2, gross=6_000_007),
        _row(1289601, "Life After Fighting", budget=234_325, gross=59_530_099),
    )

    assert records[270650]["production_budget"] is None
    assert records[270650]["worldwide_gross"] == 89_507
    assert records[1289601]["production_budget"] == 234_325
    assert records[1289601]["worldwide_gross"] == 5_727


def test_financial_corrections_null_unsupported_extreme_roi_rows():
    records = _records(
        _row(1175807, "Honk", imdb_id=None, budget=5, gross=100_000_000, vote_count=0),
        _row(
            1248416,
            "Ginger Person",
            imdb_id=None,
            budget=1,
            gross=10_000_000,
            vote_count=0,
        ),
        _row(1679323, "Hu Chhu Mr. Shankar", budget=127_613, gross=30_000_000),
        _row(1494947, "Super Vixens 2", budget=100_000, gross=14_500_000),
    )

    assert records[1175807]["production_budget"] is None
    assert records[1175807]["worldwide_gross"] is None
    assert records[1248416]["production_budget"] is None
    assert records[1248416]["worldwide_gross"] is None
    assert records[1679323]["production_budget"] == 127_613
    assert records[1679323]["worldwide_gross"] is None
    assert records[1494947]["production_budget"] == 100_000
    assert records[1494947]["worldwide_gross"] is None


def test_tiny_budget_and_missing_imdb_guard_does_not_touch_normal_movies():
    records = _records(
        _row(1, "Tiny Budget", budget=999, gross=100_000),
        _row(
            2,
            "No Footprint Hit",
            imdb_id=None,
            budget=20_000,
            gross=2_000_000,
            vote_count=0,
        ),
        _row(3, "Normal Movie", budget=20_000, gross=2_000_000, vote_count=100),
    )

    assert records[1]["production_budget"] is None
    assert records[1]["worldwide_gross"] == 100_000
    assert records[2]["production_budget"] == 20_000
    assert records[2]["worldwide_gross"] is None
    assert records[3]["production_budget"] == 20_000
    assert records[3]["worldwide_gross"] == 2_000_000


def test_prediction_no_gross_ids_still_win_over_export_values():
    record = build_movie_records(
        pd.DataFrame([_row(4, "Future Movie", budget=10_000_000, gross=50_000_000)]),
        no_gross_ids={4},
    )[0]

    assert record["production_budget"] == 10_000_000
    assert record["worldwide_gross"] is None
    assert not any(
        isinstance(value, float) and math.isnan(value) for value in record.values()
    )


def test_non_movie_source_exclusions_do_not_reach_web_records():
    records = _records(
        _row(565916, "The Visual Effects of 'Scary Movie 4'"),
        _row(5, "A Movie"),
    )

    assert set(records) == {5}
