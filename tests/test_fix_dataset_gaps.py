"""Pure-logic tests for scripts/fix_dataset_gaps.py (no live TMDB)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from box_office.ingestion.tmdb_rich_backfill import BackfillConfig, flat_columns
from scripts import fix_dataset_gaps as fdg


def _payload(
    tmdb_id: int,
    *,
    title: str = "Test Movie",
    revenue: int = 500_000_000,
    runtime: int = 120,
    budget: int = 100_000_000,
    collection_id: int | None = 1000,
) -> dict[str, Any]:
    collection = (
        {"id": collection_id, "name": "Test Collection"}
        if collection_id is not None
        else None
    )
    return {
        "id": tmdb_id,
        "imdb_id": f"tt{tmdb_id:07d}",
        "title": title,
        "original_title": title,
        "status": "Released",
        "release_date": "2012-05-04",
        "original_language": "en",
        "belongs_to_collection": collection,
        "genres": [{"name": "Action"}, {"name": "Adventure"}],
        "budget": budget,
        "revenue": revenue,
        "runtime": runtime,
        "vote_count": 25_000,
        "vote_average": 7.7,
        "popularity": 100.0,
        "adult": False,
        "overview": "An overview.",
        "tagline": "A tagline.",
        "homepage": "",
        "poster_path": "/poster.jpg",
        "backdrop_path": "/backdrop.jpg",
        "production_companies": [{"name": "Marvel Studios"}],
        "production_countries": [{"name": "United States of America"}],
        "spoken_languages": [{"name": "English"}],
        "credits": {
            "crew": [{"job": "Director", "name": "Joss Whedon"}],
            "cast": [{"order": 0, "name": "Robert Downey Jr."}],
        },
        "keywords": {"keywords": [{"name": "superhero"}]},
        "release_dates": {
            "results": [
                {
                    "iso_3166_1": "US",
                    "release_dates": [{"certification": "PG-13"}],
                }
            ]
        },
        "external_ids": {"imdb_id": f"tt{tmdb_id:07d}"},
        "images": {"posters": [{}], "backdrops": [{}]},
        "videos": {"results": [{}]},
        "alternative_titles": {"titles": []},
        "translations": {"translations": []},
    }


def _raw(tmdb_id: int, **kwargs: Any) -> dict[str, Any]:
    payload = _payload(tmdb_id, **kwargs)
    return {
        "tmdb_id": tmdb_id,
        "title": payload["title"],
        "release_year_query": 2012,
        "discover_page": 0,
        "fetched_at": "2026-07-08T00:00:00+00:00",
        "payload": payload,
    }


def _target_columns() -> list[str]:
    return flat_columns() + list(fdg.BUDGET_EXTRA_COLUMNS)


def test_build_flat_row_uses_tmdb_budget() -> None:
    row = fdg.build_flat_row(_raw(24428, budget=220_000_000), _target_columns())
    assert row["tmdb_id"] == 24428
    assert row["production_budget"] == 220_000_000.0
    assert row["production_budget_original"] == 220_000_000
    assert row["production_budget_source"] == "tmdb"
    assert row["production_budget_was_missing"] is False
    # list-string formatting matches the rest of the dataset
    assert row["genres"] == "Action, Adventure"
    assert row["director"] == "Joss Whedon"
    assert row["mpaa"] == "PG-13"
    assert row["wikidata_budget_usd"] is None


def test_build_flat_row_documented_budget_fallback() -> None:
    row = fdg.build_flat_row(
        _raw(24428, budget=0), _target_columns(), documented_budget=220_000_000
    )
    assert row["production_budget"] == 220_000_000.0
    assert row["production_budget_source"] == "tmdb"
    assert row["production_budget_was_missing"] is False


def test_build_flat_row_missing_budget() -> None:
    row = fdg.build_flat_row(_raw(1, budget=0), _target_columns())
    assert row["production_budget"] is None
    assert row["production_budget_original"] == 0
    assert row["production_budget_source"] == "missing"
    assert row["production_budget_was_missing"] is True


def test_build_flat_row_rejects_missing_target_column() -> None:
    with pytest.raises(ValueError, match="missing target columns"):
        fdg.build_flat_row(_raw(1), _target_columns() + ["does_not_exist"])


def _seed_frame(target_columns: list[str]) -> pd.DataFrame:
    existing = fdg.build_flat_row(_raw(99861, budget=235_000_000), target_columns)
    df = pd.DataFrame([existing], columns=target_columns)
    # match real parquet dtypes on the int/bool columns
    for column in ["runtime", "vote_count", "release_year", "production_budget_original"]:
        df[column] = df[column].astype("int64")
    df["adult"] = df["adult"].astype(bool)
    df["production_budget_was_missing"] = df["production_budget_was_missing"].astype(bool)
    df["production_budget"] = df["production_budget"].astype("float64")
    df["wikidata_budget_usd"] = df["wikidata_budget_usd"].astype("float64")
    return df


def test_append_rows_preserves_dtypes() -> None:
    target_columns = _target_columns()
    df = _seed_frame(target_columns)
    before = df.dtypes.to_dict()
    new = fdg.build_flat_row(_raw(24428), target_columns)
    combined = fdg.append_rows(df, [new], target_columns)
    assert len(combined) == 2
    assert combined.dtypes.to_dict() == before


def test_append_rows_empty_is_noop() -> None:
    df = _seed_frame(_target_columns())
    assert fdg.append_rows(df, [], list(df.columns)) is df


def test_drop_ad_columns() -> None:
    df = pd.DataFrame(
        {"tmdb_id": [1], "ad_budget_original": [0], "ad_budget_source": ["missing"]}
    )
    stripped, dropped = fdg.drop_ad_columns(df)
    assert dropped == ["ad_budget_original", "ad_budget_source"]
    assert list(stripped.columns) == ["tmdb_id"]


def test_drop_ad_columns_absent_is_noop() -> None:
    df = pd.DataFrame({"tmdb_id": [1]})
    stripped, dropped = fdg.drop_ad_columns(df)
    assert dropped == []
    assert list(stripped.columns) == ["tmdb_id"]


def test_validate_row_accepts_clean_row() -> None:
    row = fdg.build_flat_row(_raw(24428, budget=220_000_000), _target_columns())
    fdg.validate_row(row)  # must not raise


def test_add_missing_movies_skips_present_id(monkeypatch, tmp_path) -> None:
    target_columns = _target_columns()
    df = _seed_frame(target_columns)  # contains 99861
    config = BackfillConfig(output_dir=tmp_path)

    def _boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("should not fetch an already-present id")

    monkeypatch.setattr(fdg, "request_json", _boom)
    updated, added, rejected = fdg.add_missing_movies(df, target_columns, [99861], config)
    assert added == []
    assert rejected == []
    assert len(updated) == 1


def test_add_missing_movies_appends_and_writes_jsonl(monkeypatch, tmp_path) -> None:
    target_columns = _target_columns()
    df = _seed_frame(target_columns)
    config = BackfillConfig(output_dir=tmp_path)

    monkeypatch.setattr(
        fdg, "request_json", lambda *a, **k: _payload(24428, title="The Avengers")
    )
    updated, added, rejected = fdg.add_missing_movies(df, target_columns, [24428], config)

    assert rejected == []
    assert [entry["tmdb_id"] for entry in added] == [24428]
    assert added[0]["collection_id"] == 1000
    assert set(updated["tmdb_id"]) == {99861, 24428}
    # raw record was appended to the JSONL for the franchise-feature merge
    assert config.raw_jsonl_path.exists()
    assert config.raw_jsonl_path.read_text().count("24428") >= 1


def test_add_missing_movies_idempotent_on_jsonl(monkeypatch, tmp_path) -> None:
    target_columns = _target_columns()
    df = _seed_frame(target_columns)
    config = BackfillConfig(output_dir=tmp_path)
    # pretend 24428 is already in the JSONL (but not the parquet)
    config.raw_jsonl_path.write_text(
        '{"tmdb_id": 24428, "payload": {"id": 24428}}\n'
    )

    monkeypatch.setattr(fdg, "request_json", lambda *a, **k: _payload(24428))
    fdg.add_missing_movies(df, target_columns, [24428], config)
    # still exactly one JSONL line for 24428 (no duplicate append)
    lines = [
        line for line in config.raw_jsonl_path.read_text().splitlines() if line.strip()
    ]
    assert len(lines) == 1


def test_add_missing_movies_rejects_below_inclusion_bar(monkeypatch, tmp_path) -> None:
    target_columns = _target_columns()
    df = _seed_frame(target_columns)
    config = BackfillConfig(output_dir=tmp_path)

    monkeypatch.setattr(
        fdg,
        "request_json",
        lambda *a, **k: _payload(24428, revenue=1_000_000, runtime=40),
    )
    updated, added, rejected = fdg.add_missing_movies(df, target_columns, [24428], config)
    assert added == []
    assert [entry["tmdb_id"] for entry in rejected] == [24428]
    assert len(updated) == 1
    # nothing written to the JSONL when the candidate is rejected
    assert not config.raw_jsonl_path.exists()
