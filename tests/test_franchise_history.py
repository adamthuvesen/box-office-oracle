"""Merged collection-link loading (JSONL + refetch parquet + overrides)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from box_office.franchise_history import (
    collection_franchise_keys,
    collection_memberships,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _jsonl_row(tmdb_id: int, collection: dict | None) -> dict:
    return {
        "tmdb_id": tmdb_id,
        "payload": {"id": tmdb_id, "belongs_to_collection": collection},
    }


def _write_refetch(path: Path, rows: list[dict]) -> Path:
    frame = pd.DataFrame(rows, columns=["tmdb_id", "collection_id", "collection_name"])
    frame["collection_id"] = frame["collection_id"].astype("Int64")
    frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def jsonl(tmp_path: Path) -> Path:
    return _write_jsonl(
        tmp_path / "raw.jsonl",
        [
            _jsonl_row(1, {"id": 100, "name": "Alpha Collection"}),
            _jsonl_row(2, None),
        ],
    )


def test_jsonl_only(jsonl: Path) -> None:
    memberships = collection_memberships(jsonl, refetch_path=None, overrides_path=None)
    assert memberships == {1: (100, "Alpha Collection")}


def test_refetch_fills_gap_but_never_overwrites(jsonl: Path, tmp_path: Path) -> None:
    refetch = _write_refetch(
        tmp_path / "refetch.parquet",
        [
            # Gap: tmdb 2 has no JSONL collection -> filled.
            {"tmdb_id": 2, "collection_id": 200, "collection_name": "Beta Collection"},
            # Present in JSONL as collection 100 -> refetch value ignored.
            {"tmdb_id": 1, "collection_id": 999, "collection_name": "Wrong"},
        ],
    )
    memberships = collection_memberships(jsonl, refetch, overrides_path=None)
    assert memberships == {
        1: (100, "Alpha Collection"),
        2: (200, "Beta Collection"),
    }


def test_refetch_null_collection_keeps_jsonl_link(jsonl: Path, tmp_path: Path) -> None:
    refetch = _write_refetch(
        tmp_path / "refetch.parquet",
        [{"tmdb_id": 1, "collection_id": None, "collection_name": None}],
    )
    memberships = collection_memberships(jsonl, refetch, overrides_path=None)
    assert memberships[1] == (100, "Alpha Collection")


def test_override_wins_over_both(jsonl: Path, tmp_path: Path) -> None:
    refetch = _write_refetch(
        tmp_path / "refetch.parquet",
        [{"tmdb_id": 1, "collection_id": 200, "collection_name": "Beta"}],
    )
    overrides = tmp_path / "overrides.yml"
    overrides.write_text(
        "overrides:\n"
        "  - tmdb_id: 1\n"
        "    collection_id: 300\n"
        '    collection_name: "Gamma Collection"\n'
    )
    memberships = collection_memberships(jsonl, refetch, overrides)
    assert memberships[1] == (300, "Gamma Collection")


def test_missing_optional_sources_are_skipped(jsonl: Path, tmp_path: Path) -> None:
    memberships = collection_memberships(
        jsonl, tmp_path / "absent.parquet", tmp_path / "absent.yml"
    )
    assert memberships == {1: (100, "Alpha Collection")}


def test_malformed_override_fails_loudly(jsonl: Path, tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.yml"
    overrides.write_text("overrides:\n  - tmdb_id: 1\n    collection_id: 300\n")
    with pytest.raises(ValueError, match="Malformed collection override"):
        collection_memberships(jsonl, None, overrides)


def test_franchise_keys_use_merged_sources(jsonl: Path, tmp_path: Path) -> None:
    refetch = _write_refetch(
        tmp_path / "refetch.parquet",
        [{"tmdb_id": 2, "collection_id": 200, "collection_name": "Beta"}],
    )
    keys = collection_franchise_keys(jsonl, refetch, overrides_path=None)
    assert keys == {1: "collection:100", 2: "collection:200"}
