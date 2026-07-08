"""Tests for the local-only rich TMDB backfill."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from box_office.ingestion import tmdb_rich_backfill as rich


def _payload(
    tmdb_id: int,
    *,
    title: str = "Test Movie",
    revenue: int = 20_000_000,
    language: str = "en",
    genres: list[dict] | None = None,
) -> dict:
    return {
        "id": tmdb_id,
        "imdb_id": f"tt{tmdb_id:07d}",
        "title": title,
        "original_title": title,
        "status": "Released",
        "release_date": "2020-05-01",
        "original_language": language,
        "production_countries": [{"name": "United States of America"}],
        "spoken_languages": [{"english_name": "English"}],
        "genres": genres if genres is not None else [{"name": "Action"}],
        "budget": 10_000_000,
        "revenue": revenue,
        "runtime": 110,
        "overview": "Overview",
        "tagline": "Tagline",
        "homepage": "https://example.test",
        "poster_path": "/poster.jpg",
        "backdrop_path": "/backdrop.jpg",
        "adult": False,
        "popularity": 10.5,
        "vote_average": 7.2,
        "vote_count": 100,
        "production_companies": [{"name": "Studio"}],
        "credits": {
            "crew": [{"job": "Director", "name": "Director One"}],
            "cast": [
                {"order": 1, "name": "Lead Two"},
                {"order": 0, "name": "Lead One"},
            ],
        },
        "keywords": {"keywords": [{"name": "space"}, {"name": "quest"}]},
        "release_dates": {
            "results": [
                {
                    "iso_3166_1": "US",
                    "release_dates": [{"certification": "PG-13"}],
                }
            ]
        },
        "external_ids": {"imdb_id": f"tt{tmdb_id:07d}"},
        "images": {"posters": [{"file_path": "/poster.jpg"}], "backdrops": []},
        "videos": {"results": [{"key": "abc"}]},
        "alternative_titles": {"titles": [{"title": "Alt"}]},
        "translations": {"translations": [{"iso_639_1": "sv"}]},
    }


def _write_web_data(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def test_run_backfill_filters_and_writes_raw_jsonl(tmp_path, monkeypatch):
    web_data = tmp_path / "web" / "movies.json"
    _write_web_data(web_data, [{"tmdb_id": 1, "title": "Existing Movie"}])
    out_dir = tmp_path / "out"
    config = rich.BackfillConfig(
        start_year=2020,
        end_year=2020,
        min_revenue=5_000_000,
        page_limit=1,
        base_sleep_seconds=0,
        existing_web_data=web_data,
        output_dir=out_dir,
    )

    payloads = {
        2: _payload(2, revenue=4_000_000),
        3: _payload(3, genres=[{"name": "Documentary"}]),
        4: _payload(4, language="fr"),
        5: _payload(5, title="Accepted Movie"),
    }

    def fake_request_json(session, url, *, params, config, state):
        if url.endswith("/discover/movie"):
            return {
                "results": [
                    {"id": 1},
                    {"id": 2},
                    {"id": 3},
                    {"id": 4},
                    {"id": 5},
                ],
                "total_pages": 1,
            }
        tmdb_id = int(url.rsplit("/", 1)[1])
        return payloads[tmdb_id]

    monkeypatch.setattr(rich, "request_json", fake_request_json)
    state = rich.run_backfill(config)

    raw_rows = list(rich.iter_raw_records(config.raw_jsonl_path))
    assert state.accepted_count == 1
    assert raw_rows[0]["tmdb_id"] == 5
    assert state.skipped_existing_count == 1
    assert state.skipped_below_revenue_count == 1
    assert state.skipped_documentary_count == 1
    assert state.skipped_language_count == 1


def test_flat_outputs_include_rich_fields_and_50m_subset(tmp_path):
    web_data = tmp_path / "web" / "movies.json"
    _write_web_data(
        web_data,
        [
            {
                "tmdb_id": 1,
                "imdb_id": "tt0000001",
                "title": "Existing Movie",
                "release_date": "2020-01-01",
                "release_year": 2020,
                "genres": ["Comedy"],
                "director": "Existing Director",
                "actors": ["Actor One"],
                "mpaa": "PG",
                "runtime": 90,
                "production_budget": 1_000_000,
                "ad_budget": 250_000,
                "production_company": "Existing Studio",
                "overview": "Existing",
                "tagline": "Existing",
                "keywords": ["existing"],
                "worldwide_gross": 6_000_000,
                "poster_path": "/existing.jpg",
                "backdrop_path": "/existing_backdrop.jpg",
            }
        ],
    )
    config = rich.BackfillConfig(
        start_year=2020,
        end_year=2020,
        min_revenue=5_000_000,
        existing_web_data=web_data,
        output_dir=tmp_path / "out",
    )
    config.output_dir.mkdir(parents=True)
    rows = [
        rich.raw_record_from_payload(
            _payload(10, revenue=60_000_000), 2020, 1
        ),
        rich.raw_record_from_payload(
            _payload(11, revenue=6_000_000), 2020, 2
        ),
    ]
    for row in rows:
        rich.write_jsonl_record(config.raw_jsonl_path, row)

    manifest = rich.build_flat_outputs(config)

    flat = pd.read_csv(config.flat_csv_path)
    subset = pd.read_csv(config.fifty_million_csv_path)
    combined = pd.read_csv(config.combined_csv_path)
    combined_subset = pd.read_csv(config.combined_fifty_million_csv_path)
    missingness = pd.read_csv(config.missingness_csv_path)

    assert len(flat) == 2
    assert len(subset) == 1
    assert len(combined) == 3
    assert len(combined_subset) == 1
    assert set(subset["tmdb_id"]) == {10}
    assert set(combined["source_dataset"]) == {"web_existing", "tmdb_backfill"}
    assert (
        flat.loc[flat["tmdb_id"] == 10, "director"].iloc[0]
        == "Director One"
    )
    assert (
        flat.loc[flat["tmdb_id"] == 10, "keywords"].iloc[0]
        == "space, quest"
    )
    assert flat.loc[flat["tmdb_id"] == 10, "mpaa"].iloc[0] == "PG-13"
    assert config.flat_parquet_path.exists()
    assert config.fifty_million_parquet_path.exists()
    assert config.combined_parquet_path.exists()
    assert config.combined_fifty_million_parquet_path.exists()
    assert set(missingness["field"]) >= {
        "imdb_id",
        "poster_path",
        "video_count",
    }
    assert manifest["row_counts"] == {
        "raw_records": 2,
        "flat_5m": 2,
        "flat_50m": 1,
        "combined_5m": 3,
        "combined_50m": 1,
    }
    assert manifest["quality"]["missing_years"] == []


def test_request_json_retries_after_rate_limit(monkeypatch):
    class FakeResponse:
        def __init__(
            self,
            status_code: int,
            payload: dict,
            headers: dict | None = None,
        ):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    class FakeSession:
        def __init__(self):
            self.responses = [
                FakeResponse(429, {}, {"Retry-After": "0"}),
                FakeResponse(200, {"ok": True}),
            ]

        def get(self, *args, **kwargs):
            return self.responses.pop(0)

    monkeypatch.setenv("TMDB_API_TOKEN", "token")
    monkeypatch.setattr(rich.time, "sleep", lambda *_: None)
    state = rich.BackfillState(started_at=rich.utc_now())

    result = rich.request_json(
        FakeSession(),
        "https://example.test",
        params={},
        config=rich.BackfillConfig(max_retries=1),
        state=state,
    )

    assert result == {"ok": True}
    assert state.rate_limit_count == 1
    assert state.retry_count == 1
