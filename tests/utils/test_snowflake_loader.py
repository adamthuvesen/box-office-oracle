"""Tests for validated Snowflake identifier construction."""

import pytest

from box_office.utils.snowflake_loader import (
    fully_qualified_name,
    validate_sql_identifier,
)


@pytest.mark.parametrize(
    "name",
    ["BOX_OFFICE", "my_table", "_private_table", "Table123"],
)
def test_validate_sql_identifier_accepts_safe_names(name):
    assert validate_sql_identifier(name, "table") == name


@pytest.mark.parametrize(
    "name",
    [
        "table; DROP TABLE users--",
        "table' OR '1'='1",
        "table\nDROP TABLE",
        "123_starts_with_number",
        "has spaces",
        "has-dashes",
        "",
        None,
    ],
)
def test_validate_sql_identifier_rejects_unsafe_names(name):
    with pytest.raises(ValueError, match="Invalid table name"):
        validate_sql_identifier(name, "table")


def test_fully_qualified_name_joins_valid_components():
    assert (
        fully_qualified_name("BOX_OFFICE", "ML_TRAINING", "X_TRAIN")
        == "BOX_OFFICE.ML_TRAINING.X_TRAIN"
    )


@pytest.mark.parametrize(
    ("database", "schema", "table", "bad_part"),
    [
        ("BOX_OFFICE", "ML_TRAINING", "X_TRAIN; DROP TABLE Y--", "table"),
        ("BOX_OFFICE", "ML_TRAINING' OR '1'='1", "X_TRAIN", "schema"),
        ("BOX_OFFICE; DROP DATABASE X", "ML_TRAINING", "X_TRAIN", "database"),
        ("BOX_OFFICE", "ML_TRAINING.PUBLIC", "X_TRAIN", "schema"),
    ],
)
def test_fully_qualified_name_rejects_unsafe_components(
    database, schema, table, bad_part
):
    with pytest.raises(ValueError, match=f"Invalid {bad_part} name"):
        fully_qualified_name(database, schema, table)
