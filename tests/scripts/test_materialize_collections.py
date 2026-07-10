import pandas as pd

from scripts.materialize_collections import add_collection_columns


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tmdb_id": [1, 2, 3],
            "title": ["Alpha", "Beta", "Gamma"],
        }
    )


def test_adds_collection_columns_keyed_on_tmdb_id():
    memberships = {1: (10, "Alpha Collection"), 3: (30, "Gamma Collection")}

    out = add_collection_columns(_frame(), memberships)

    assert out.loc[out["tmdb_id"] == 1, "collection_id"].item() == 10
    assert out.loc[out["tmdb_id"] == 1, "collection_name"].item() == "Alpha Collection"
    assert out.loc[out["tmdb_id"] == 3, "collection_id"].item() == 30


def test_movie_without_collection_gets_nulls():
    out = add_collection_columns(_frame(), {1: (10, "Alpha Collection")})

    row = out.loc[out["tmdb_id"] == 2]
    assert pd.isna(row["collection_id"].item())
    assert pd.isna(row["collection_name"].item())


def test_collection_id_is_nullable_integer():
    out = add_collection_columns(_frame(), {1: (10, "Alpha Collection")})

    assert out["collection_id"].dtype == "Int64"
    # No float coercion of the present id.
    assert out.loc[out["tmdb_id"] == 1, "collection_id"].item() == 10


def test_collection_id_present_but_name_missing():
    out = add_collection_columns(_frame(), {2: (20, None)})

    row = out.loc[out["tmdb_id"] == 2]
    assert row["collection_id"].item() == 20
    assert pd.isna(row["collection_name"].item())


def test_idempotent_when_columns_already_exist():
    first = add_collection_columns(_frame(), {1: (10, "Alpha Collection")})
    second = add_collection_columns(first, {1: (10, "Alpha Collection")})

    pd.testing.assert_frame_equal(first, second)


def test_recompute_overwrites_stale_columns():
    stale = _frame()
    stale["collection_id"] = pd.Series([99, 99, 99], dtype="Int64")
    stale["collection_name"] = ["stale", "stale", "stale"]

    out = add_collection_columns(stale, {1: (10, "Alpha Collection")})

    assert out.loc[out["tmdb_id"] == 1, "collection_id"].item() == 10
    assert pd.isna(out.loc[out["tmdb_id"] == 2, "collection_id"].item())


def test_missing_tmdb_id_column_raises():
    import pytest

    with pytest.raises(ValueError, match="tmdb_id"):
        add_collection_columns(pd.DataFrame({"title": ["x"]}), {})
