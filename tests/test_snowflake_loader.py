"""Tests for the Snowflake CSV loader module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from box_office.utils.snowflake_loader import (
    SnowflakeLoader,
    EXPECTED_COLUMNS,
    NUMERIC_COLUMNS,
    STRING_COLUMNS,
)


@pytest.fixture
def sample_csv_data():
    return pd.DataFrame(
        {
            "tmdb_id": [12345, 67890],
            "imdb_id": ["tt1234567", "tt7654321"],
            "rank": [1, 2],
            "title": ["Test Movie 1", "Test Movie 2"],
            "release_date": ["2024-01-15", "2024-02-20"],
            "rating": [7.5, 8.2],
            "votes": [1000, 2000],
            "original_language": ["en", "en"],
            "production_countries": ["United States", "United Kingdom"],
            "genres": ["Action, Adventure", "Drama, Comedy"],
            "production_budget": [100000000, 50000000],
            "director": ["Director A", "Director B"],
            "actors": ["Actor 1, Actor 2", "Actor 3, Actor 4"],
            "mpaa": ["PG-13", "R"],
            "social_media_buzz": [7, 5],
            "release_type": ["wide", "limited"],
            "franchise_rating": [2, 0],
            "runtime": [120, 95],
            "overview": ["A test movie overview", "Another overview"],
            "tagline": ["Tagline 1", "Tagline 2"],
            "keywords": ["action, blockbuster", "drama, indie"],
            "ad_budget": [50000000, 25000000],
            "production_company": ["Studio A", "Studio B"],
            "release_year": [2024, 2024],
            "release_type_encoded": [1, 0],
            "production_company_encoded": [1, 2],
            "mpaa_encoded": [2, 3],
            "worldwide_gross": [500000000, 75000000],
        }
    )


@pytest.fixture
def loader():
    return SnowflakeLoader(schema="RAW")


class TestSnowflakeLoaderInit:
    def test_default_schema(self):
        loader = SnowflakeLoader()
        assert loader.schema == "RAW"

    def test_custom_schema(self):
        loader = SnowflakeLoader(schema="STAGING")
        assert loader.schema == "STAGING"


class TestSchemaValidation:
    def test_validate_schema_normalizes_columns(self, loader, sample_csv_data):
        df = sample_csv_data.copy()
        df.columns = df.columns.str.upper()

        result = loader.validate_schema(df)
        assert all(c.islower() for c in result.columns)

    def test_validate_schema_adds_missing_columns(self, loader):
        df = pd.DataFrame(
            {
                "tmdb_id": [12345],
                "title": ["Test Movie"],
            }
        )

        result = loader.validate_schema(df)

        assert set(result.columns) == set(EXPECTED_COLUMNS)
        assert result["director"].iloc[0] is None

    def test_validate_schema_drops_extra_columns(self, loader, sample_csv_data):
        df = sample_csv_data.copy()
        df["extra_column"] = "should be dropped"

        result = loader.validate_schema(df)

        assert "extra_column" not in result.columns
        assert set(result.columns) == set(EXPECTED_COLUMNS)

    def test_validate_schema_column_order(self, loader, sample_csv_data):
        result = loader.validate_schema(sample_csv_data)
        assert list(result.columns) == EXPECTED_COLUMNS


class TestColumnTransformation:
    def test_transform_numeric_columns(self, loader):
        df = pd.DataFrame(
            {
                "tmdb_id": ["12345", "67890"],
                "rating": ["7.5", "8.2"],
                "votes": ["1000", "2000"],
                "production_budget": ["100000000", "50000000"],
                "worldwide_gross": ["500000000", "75000000"],
            }
        )
        df = loader.validate_schema(df)
        result = loader.transform_columns(df)

        for col in [
            "tmdb_id",
            "rating",
            "votes",
            "production_budget",
            "worldwide_gross",
        ]:
            assert (
                pd.api.types.is_numeric_dtype(result[col]) or result[col].isna().all()
            )

    def test_transform_handles_invalid_numeric(self, loader):
        df = pd.DataFrame(
            {
                "tmdb_id": ["12345", "invalid"],
                "rating": ["7.5", "not_a_number"],
            }
        )
        df = loader.validate_schema(df)
        result = loader.transform_columns(df)

        assert result["tmdb_id"].iloc[1] is None
        assert result["rating"].iloc[1] is None

    def test_transform_date_column(self, loader):
        df = pd.DataFrame(
            {
                "release_date": ["2024-01-15", "2024/02/20", "January 15, 2024"],
            }
        )
        df = loader.validate_schema(df)
        result = loader.transform_columns(df)

        assert result["release_date"].iloc[0] == "2024-01-15"

    def test_transform_string_columns(self, loader):
        df = pd.DataFrame(
            {
                "title": ["Test Movie", None, np.nan],
                "director": ["Director A", "nan", "None"],
            }
        )
        df = loader.validate_schema(df)
        result = loader.transform_columns(df)

        assert result["title"].iloc[1] == ""
        assert result["director"].iloc[1] == ""
        assert result["director"].iloc[2] == ""


class TestExpectedColumns:
    def test_expected_columns_count(self):
        assert len(EXPECTED_COLUMNS) == 28

    def test_primary_key_in_columns(self):
        assert "tmdb_id" in EXPECTED_COLUMNS

    def test_target_column_in_columns(self):
        assert "worldwide_gross" in EXPECTED_COLUMNS

    def test_all_numeric_columns_in_expected(self):
        assert set(NUMERIC_COLUMNS).issubset(set(EXPECTED_COLUMNS))

    def test_all_string_columns_in_expected(self):
        assert set(STRING_COLUMNS).issubset(set(EXPECTED_COLUMNS))


class TestDryRun:
    @patch("box_office.utils.snowflake_loader.pd.read_csv")
    def test_dry_run_returns_validation_info(
        self, mock_read_csv, loader, sample_csv_data, tmp_path
    ):
        mock_read_csv.return_value = sample_csv_data

        csv_file = tmp_path / "test.csv"
        csv_file.write_text("dummy")

        result = loader.load_csv_to_raw(csv_path=str(csv_file), dry_run=True)

        assert result["status"] == "dry_run"
        assert "rows_to_load" in result
        assert "columns" in result


class TestFileValidation:
    def test_nonexistent_file_raises_error(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_csv_to_raw("/nonexistent/path/file.csv")


class TestSQLInjectionPrevention:
    @pytest.mark.parametrize(
        "invalid_name",
        [
            "table; DROP TABLE users--",
            "table' OR '1'='1",
            "table\nDROP TABLE",
            "123_starts_with_number",
            "has spaces",
            "has-dashes",
        ],
    )
    def test_invalid_table_names_rejected(self, loader, invalid_name):
        with pytest.raises(ValueError, match="Invalid table name"):
            loader._validate_identifier(invalid_name, "table")

    @pytest.mark.parametrize(
        "valid_name",
        [
            "BOX_OFFICE_V3",
            "my_table",
            "_private_table",
            "Table123",
        ],
    )
    def test_valid_table_names_accepted(self, loader, valid_name):
        result = loader._validate_identifier(valid_name, "table")
        assert result == valid_name


class TestLoadModes:
    @patch("snowflake.connector.pandas_tools.write_pandas")
    @patch("box_office.utils.snowflake_loader.create_snowflake_connection")
    @patch("box_office.utils.snowflake_loader.pd.read_csv")
    def test_merge_mode_uses_merge_statement(
        self, mock_read_csv, mock_conn, mock_write, loader, sample_csv_data, tmp_path
    ):
        mock_read_csv.return_value = sample_csv_data.head(1)
        mock_write.return_value = (True, 1, 1, None)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]
        mock_conn.return_value.cursor.return_value = mock_cursor

        csv_file = tmp_path / "test.csv"
        sample_csv_data.head(1).to_csv(csv_file, index=False)

        loader.load_csv_to_raw(csv_path=str(csv_file), mode="merge")

        merge_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "MERGE INTO" in str(call)
        ]
        assert len(merge_calls) > 0
        merge_sql = str(merge_calls[0])
        assert "BOX_OFFICE_V3" in merge_sql, "MERGE must target the BOX_OFFICE_V3 table"
        assert "TMDB_ID" in merge_sql, "MERGE must key on TMDB_ID"

    @patch("snowflake.connector.pandas_tools.write_pandas")
    @patch("box_office.utils.snowflake_loader.create_snowflake_connection")
    @patch("box_office.utils.snowflake_loader.pd.read_csv")
    def test_overwrite_mode_uses_write_pandas(
        self, mock_read_csv, mock_conn, mock_write, loader, sample_csv_data, tmp_path
    ):
        mock_read_csv.return_value = sample_csv_data.head(1)
        mock_write.return_value = (True, 1, 1, None)

        csv_file = tmp_path / "test.csv"
        sample_csv_data.head(1).to_csv(csv_file, index=False)

        loader.load_csv_to_raw(csv_path=str(csv_file), mode="overwrite")

        assert mock_write.called
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs.get("overwrite") is True
