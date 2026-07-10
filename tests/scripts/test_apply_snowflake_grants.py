"""Tests for the SQL statement splitter in scripts/apply_snowflake_grants.py."""

import pytest

from scripts.apply_snowflake_grants import SQL_FILE, split_statements


def test_split_strips_comments_and_blank_lines():
    sql = """
    -- a leading comment
    USE ROLE ACCOUNTADMIN;

    GRANT USAGE ON DATABASE BOX_OFFICE TO ROLE DBT_RUNNER;  -- trailing comment
    """
    statements = split_statements(sql)
    assert statements == [
        "USE ROLE ACCOUNTADMIN",
        "GRANT USAGE ON DATABASE BOX_OFFICE TO ROLE DBT_RUNNER",
    ]


def test_split_keeps_string_literal_intact():
    sql = "CREATE ROLE IF NOT EXISTS DBT_RUNNER COMMENT = 'runs dbt';"
    assert split_statements(sql) == [
        "CREATE ROLE IF NOT EXISTS DBT_RUNNER COMMENT = 'runs dbt'"
    ]


def test_split_refuses_to_strip_comment_from_a_string_literal():
    # A `--` inside a quoted string is not a comment; stripping it would corrupt
    # the value, so the splitter must fail loudly instead.
    sql = "CREATE ROLE R COMMENT = 'owns a--b schema';"
    with pytest.raises(ValueError, match="string literal"):
        split_statements(sql)


def test_real_grants_file_splits_cleanly():
    statements = split_statements(SQL_FILE.read_text())
    assert statements
    assert statements[0] == "USE ROLE ACCOUNTADMIN"
    # Both runtime roles are created before they are granted.
    assert any("CREATE ROLE IF NOT EXISTS DBT_RUNNER" in s for s in statements)
    assert any("CREATE ROLE IF NOT EXISTS BOX_OFFICE_LOADER" in s for s in statements)
