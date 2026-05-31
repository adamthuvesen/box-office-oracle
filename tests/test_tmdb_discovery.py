"""Unit tests for box_office.ingestion.tmdb_discovery.

The TMDB HTTP calls are mocked; these tests exercise the discovery filtering
logic (notably that the min_revenue threshold is honoured) without hitting the
network.
"""

from unittest.mock import Mock, patch

from box_office.ingestion import tmdb_discovery


def _fake_session(revenue: int) -> Mock:
    """A requests.Session whose .get routes by URL to canned TMDB payloads for a
    single English movie (id=1) with the given revenue."""
    payloads = {
        "/discover/movie": {"results": [{"id": 1}], "total_pages": 1},
        "/movie/1/keywords": {"keywords": []},
        "/movie/1/credits": {"crew": [], "cast": []},
        "/movie/1/release_dates": {"results": []},
        "/movie/1": {
            "id": 1,
            "title": "Test Movie",
            "original_language": "en",
            "revenue": revenue,
            "genres": [],
            "vote_count": 100,
        },
    }

    def get(url, **kwargs):
        # Longest matching suffix wins so "/movie/1/keywords" beats "/movie/1".
        key = max((k for k in payloads if url.endswith(k)), key=len)
        resp = Mock()
        resp.json.return_value = payloads[key]
        resp.raise_for_status.return_value = None
        return resp

    session = Mock()
    session.get.side_effect = get
    return session


@patch("box_office.ingestion.tmdb_discovery.time.sleep", lambda *_: None)
@patch("box_office.ingestion.tmdb_discovery._auth_headers", lambda: {})
def test_min_revenue_threshold_is_honoured():
    """A movie below min_revenue is excluded; lowering the threshold includes it."""
    with patch(
        "box_office.ingestion.tmdb_discovery.requests.Session",
        return_value=_fake_session(revenue=20_000_000),
    ):
        excluded = tmdb_discovery.discover_movies(
            existing_ids=set(),
            start_year=2020,
            end_year=2020,
            page_limit=1,
            min_revenue=50_000_000,
        )
        assert excluded == []

    with patch(
        "box_office.ingestion.tmdb_discovery.requests.Session",
        return_value=_fake_session(revenue=20_000_000),
    ):
        included = tmdb_discovery.discover_movies(
            existing_ids=set(),
            start_year=2020,
            end_year=2020,
            page_limit=1,
            min_revenue=10_000_000,
        )
        assert len(included) == 1
        assert included[0]["id"] == 1
