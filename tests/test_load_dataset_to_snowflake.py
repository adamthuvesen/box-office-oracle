"""Tests for the RAW dataset replacement loader (pure logic, no live Snowflake)."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.load_dataset_to_snowflake import (
    DatasetLoader,
    build_spot_check_expectations,
    build_staging_ddl,
    expected_null_budget_count,
    prepare_for_load,
    read_source_frame,
    snowflake_column_type,
    spot_check_mismatches,
    staging_columns_from_sources,
    validate_columns,
)


@pytest.fixture
def source_frame() -> pd.DataFrame:
    # Full RAW.BOX_OFFICE_V4 column contract (from sources.yml) so the frame
    # passes read_source_frame's column check; only the columns the tests read
    # carry meaningful values, the rest are null placeholders.
    columns = staging_columns_from_sources()
    data: dict[str, list] = {col: [None, None, None] for col in columns}
    data["tmdb_id"] = [597, 19995, 999]
    data["imdb_id"] = ["tt0120338", "tt0499549", "tt0000000"]
    data["title"] = ["Titanic", "Avatar", "No Budget"]
    data["release_year"] = [1997, 2009, 2001]
    data["production_budget"] = [200000000.0, 237000000.0, np.nan]
    data["production_budget_source"] = ["wikidata", "wikidata", None]
    data["worldwide_gross"] = [2264162353.0, 2923706026.0, 100.0]
    data["adult"] = [False, False, True]
    return pd.DataFrame(data, columns=columns)


def _write_parquet(df: pd.DataFrame, tmp_path) -> "object":
    path = tmp_path / "source.parquet"
    df.to_parquet(path)
    return path


# --- read_source_frame -------------------------------------------------------


def test_read_source_frame_lowercases_and_reads(source_frame, tmp_path):
    upper = source_frame.rename(columns=str.upper)
    path = _write_parquet(upper, tmp_path)

    df = read_source_frame(path)

    assert list(df.columns) == [c.lower() for c in upper.columns]
    assert len(df) == 3


def test_read_source_frame_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_source_frame(tmp_path / "nope.parquet")


def test_read_source_frame_rejects_empty(source_frame, tmp_path):
    path = _write_parquet(source_frame.iloc[0:0], tmp_path)
    with pytest.raises(ValueError, match="no rows"):
        read_source_frame(path)


def test_read_source_frame_rejects_missing_budget_column(source_frame, tmp_path):
    path = _write_parquet(source_frame.drop(columns=["production_budget"]), tmp_path)
    with pytest.raises(ValueError, match="production_budget"):
        read_source_frame(path)


def test_read_source_frame_rejects_duplicate_tmdb_id(source_frame, tmp_path):
    dup = pd.concat([source_frame, source_frame.iloc[[0]]], ignore_index=True)
    path = _write_parquet(dup, tmp_path)
    with pytest.raises(ValueError, match="duplicate tmdb_id"):
        read_source_frame(path)


def test_read_source_frame_rejects_extra_column(source_frame, tmp_path):
    path = _write_parquet(source_frame.assign(surprise=[1, 2, 3]), tmp_path)
    with pytest.raises(ValueError, match="extra=\\['surprise'\\]"):
        read_source_frame(path)


def test_read_source_frame_rejects_missing_contract_column(source_frame, tmp_path):
    path = _write_parquet(source_frame.drop(columns=["collection_name"]), tmp_path)
    with pytest.raises(ValueError, match="missing=\\['collection_name'\\]"):
        read_source_frame(path)


# --- column contract from sources.yml ----------------------------------------


def test_staging_columns_from_sources_returns_contract():
    columns = staging_columns_from_sources()
    assert columns == [c.lower() for c in columns]  # all lowercased
    assert {"tmdb_id", "production_budget", "collection_name"} <= set(columns)
    assert len(columns) == len(set(columns))  # no duplicates


def test_validate_columns_accepts_matching(source_frame):
    validate_columns(source_frame, staging_columns_from_sources())


def test_validate_columns_reports_missing_and_extra(source_frame):
    frame = source_frame.drop(columns=["title"]).assign(bogus=[1, 2, 3])
    with pytest.raises(ValueError) as exc:
        validate_columns(frame, staging_columns_from_sources())
    message = str(exc.value)
    assert "missing=['title']" in message
    assert "extra=['bogus']" in message


def test_staging_columns_from_sources_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        staging_columns_from_sources(tmp_path / "nope.yml")


# --- null handling -----------------------------------------------------------


def test_expected_null_budget_count(source_frame):
    assert expected_null_budget_count(source_frame) == 1


def test_prepare_for_load_nan_budget_becomes_none(source_frame):
    prepared = prepare_for_load(source_frame)

    assert list(prepared.columns) == [c.upper() for c in source_frame.columns]
    budgets = list(prepared["PRODUCTION_BUDGET"])
    assert budgets[0] == 200000000.0
    assert budgets[2] is None  # NaN -> None, never 0
    assert not any(b == 0 for b in budgets if b is not None)


def test_prepare_for_load_does_not_mutate_source(source_frame):
    before = source_frame["production_budget"].copy()
    prepare_for_load(source_frame)
    pd.testing.assert_series_equal(source_frame["production_budget"], before)


# --- schema mapping ----------------------------------------------------------


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pd.Series([True, False]), "BOOLEAN"),
        (pd.Series([1, 2], dtype="int64"), "NUMBER"),
        (pd.Series([1.5, 2.5]), "FLOAT"),
        (pd.Series(["a", "b"]), "VARCHAR"),
        (pd.Series(pd.to_datetime(["2020-01-01", "2020-02-01"])), "TIMESTAMP_NTZ"),
    ],
)
def test_snowflake_column_type(series, expected):
    assert snowflake_column_type(series.dtype) == expected


def test_build_staging_ddl_carries_all_columns(source_frame):
    # DDL is built from the original-typed frame: the budget column stays FLOAT
    # even though prepare_for_load later casts its NaN to None.
    ddl = build_staging_ddl("DB.RAW.STG", source_frame)

    assert ddl.startswith("CREATE OR REPLACE TRANSIENT TABLE DB.RAW.STG (")
    assert "TMDB_ID NUMBER" in ddl
    assert "PRODUCTION_BUDGET FLOAT" in ddl
    assert "TITLE VARCHAR" in ddl
    assert "ADULT BOOLEAN" in ddl
    # Every source column is represented (uppercased).
    for col in source_frame.columns:
        assert col.upper() in ddl


# --- spot checks -------------------------------------------------------------


def test_build_spot_check_expectations_skips_absent_ids(source_frame):
    expectations = build_spot_check_expectations(source_frame)
    # 597 and 19995 are in SPOT_CHECK_TMDB_IDS and present; 999 is not checked.
    assert set(expectations) == {597, 19995}
    assert expectations[597]["title"] == "Titanic"


def test_spot_check_mismatches_all_match():
    expected = {
        "title": "Titanic",
        "release_year": 1997,
        "production_budget": 200000000.0,
        "worldwide_gross": 2264162353.0,
    }
    # Float round-trip within tolerance and int/float parity both pass.
    actual = dict(expected, production_budget=200000000.0)
    assert spot_check_mismatches(expected, actual) == []


def test_spot_check_mismatches_reports_bad_value():
    expected = {
        "title": "Titanic",
        "release_year": 1997,
        "production_budget": 200000000.0,
        "worldwide_gross": 2264162353.0,
    }
    actual = dict(expected, production_budget=0.0)  # the exact bug we guard against
    mismatches = spot_check_mismatches(expected, actual)
    assert len(mismatches) == 1
    assert "production_budget" in mismatches[0]


def test_spot_check_mismatches_null_vs_value():
    expected = {"title": "X", "release_year": None, "production_budget": None,
                "worldwide_gross": None}
    actual = {"title": "X", "release_year": None, "production_budget": 5.0,
              "worldwide_gross": None}
    mismatches = spot_check_mismatches(expected, actual)
    assert len(mismatches) == 1
    assert "production_budget" in mismatches[0]


# --- verification against a mocked cursor ------------------------------------


def _cursor_for(rows: int, nulls: int, spot_rows: dict[int, tuple]) -> MagicMock:
    cursor = MagicMock()
    calls: list[object] = []

    def execute(sql, *_args):
        calls.append(sql)

    def fetchone():
        last = calls[-1].upper()
        if "COUNT(*)" in last and "IS NULL" in last:
            return (nulls,)
        if "COUNT(*)" in last:
            return (rows,)
        return None

    def fetchall():
        last = calls[-1]
        for tmdb_id, row in spot_rows.items():
            if f"TMDB_ID = {tmdb_id}" in last:
                return [row]
        return []

    cursor.execute.side_effect = execute
    cursor.fetchone.side_effect = fetchone
    cursor.fetchall.side_effect = fetchall
    return cursor


def _loader() -> DatasetLoader:
    return DatasetLoader(database="BOX_OFFICE", schema="RAW")


def test_verify_passes_on_match():
    expectations = {597: {"title": "Titanic", "release_year": 1997,
                          "production_budget": 200000000.0,
                          "worldwide_gross": 2264162353.0}}
    cursor = _cursor_for(
        rows=3, nulls=1,
        spot_rows={597: ("Titanic", 1997, 200000000.0, 2264162353.0)},
    )
    _loader()._verify(cursor, "DB.RAW.STG", 3, 1, expectations)


def test_verify_row_count_mismatch_raises():
    cursor = _cursor_for(rows=2, nulls=1, spot_rows={})
    with pytest.raises(ValueError, match="Row count mismatch"):
        _loader()._verify(cursor, "DB.RAW.STG", 3, 1, {})


def test_verify_null_budget_mismatch_raises():
    cursor = _cursor_for(rows=3, nulls=0, spot_rows={})
    with pytest.raises(ValueError, match="Null-budget mismatch"):
        _loader()._verify(cursor, "DB.RAW.STG", 3, 1, {})


def test_verify_spot_check_mismatch_raises():
    expectations = {597: {"title": "Titanic", "release_year": 1997,
                          "production_budget": 200000000.0,
                          "worldwide_gross": 2264162353.0}}
    cursor = _cursor_for(
        rows=3, nulls=1,
        spot_rows={597: ("Titanic", 1997, 0.0, 2264162353.0)},  # budget zeroed
    )
    with pytest.raises(ValueError, match="Spot check for tmdb_id 597 failed"):
        _loader()._verify(cursor, "DB.RAW.STG", 3, 1, expectations)


def test_verify_spot_check_missing_row_raises():
    expectations = {597: {"title": "Titanic", "release_year": 1997,
                          "production_budget": 200000000.0,
                          "worldwide_gross": 2264162353.0}}
    cursor = _cursor_for(rows=3, nulls=1, spot_rows={})  # no row returned
    with pytest.raises(ValueError, match="expected 1 row, got 0"):
        _loader()._verify(cursor, "DB.RAW.STG", 3, 1, expectations)


# --- drop-old gating ---------------------------------------------------------


def test_maybe_drop_old_default_does_not_drop():
    cursor = MagicMock()
    _loader()._maybe_drop_old(cursor, drop_old=False)
    cursor.execute.assert_not_called()


def test_maybe_drop_old_drops_when_flagged():
    cursor = MagicMock()
    _loader()._maybe_drop_old(cursor, drop_old=True)
    executed = " ".join(str(c.args[0]) for c in cursor.execute.call_args_list)
    assert "DROP TABLE IF EXISTS" in executed
    assert "BOX_OFFICE_V3" in executed
