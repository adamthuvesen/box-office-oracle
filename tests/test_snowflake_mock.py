"""Snowflake operations with mocked connections.

No module-level ``sys.modules`` mock for ``snowflake`` — it leaks across
the session. ``snowflake-connector-python`` imports cleanly without
contacting a warehouse.
"""

import pandas as pd
import pytest


class TestSnowflakeConnection:
    """Test Snowflake connection utilities."""

    def test_snowflake_connection_module_importable(self):
        from box_office.utils import snowflake_connection

        assert snowflake_connection is not None
        assert hasattr(snowflake_connection, "create_snowflake_connection")
        assert hasattr(snowflake_connection, "enforce_data_types")
        assert hasattr(snowflake_connection, "load_private_key_from_file")


class TestSnowflakeDataOperations:
    def test_data_tasks_module_importable(self):
        from box_office.orchestration.tasks import data_tasks

        assert data_tasks is not None
        assert hasattr(data_tasks, "save_dataset_to_snowflake_impl")
        assert hasattr(data_tasks, "validate_snowflake_tables")
        assert hasattr(data_tasks, "load_staging_box_office_from_snowflake")


class TestEnforceDataTypes:
    def test_enforce_data_types_training(self):
        from box_office.utils.snowflake_connection import enforce_data_types

        # Mixed string/numeric types simulating Snowflake output.
        df = pd.DataFrame(
            {
                "RELEASE_YEAR": ["2020", "2021", "2022"],
                "AD_BUDGET": [500000.0, 600000.0, 700000.0],
                "PRODUCTION_BUDGET": ["1000000", "2000000", "1500000"],
                "RUNTIME": ["120", "130", "110"],
                "GENRE_ACTION": ["1", "0", "1"],
                "IS_SUMMER_RELEASE": ["1", "0", "1"],
            }
        )

        result = enforce_data_types(df, table_type="training")

        assert pd.api.types.is_numeric_dtype(result["RELEASE_YEAR"])
        assert pd.api.types.is_numeric_dtype(result["AD_BUDGET"])
        assert pd.api.types.is_numeric_dtype(result["PRODUCTION_BUDGET"])
        assert pd.api.types.is_numeric_dtype(result["RUNTIME"])

    def test_enforce_data_types_handles_null_strings(self):
        from box_office.utils.snowflake_connection import enforce_data_types

        df = pd.DataFrame(
            {
                "RELEASE_YEAR": ["2020", "NULL", "2022"],
                "RUNTIME": ["120", "null", "110"],
            }
        )

        result = enforce_data_types(df, table_type="training")

        assert pd.api.types.is_numeric_dtype(result["RELEASE_YEAR"])
        assert pd.isna(result["RELEASE_YEAR"].iloc[1])
        assert pd.isna(result["RUNTIME"].iloc[1])


class TestSnowflakePrivateKeyLoading:
    def test_load_private_key_file_not_found(self):
        from box_office.utils.snowflake_connection import load_private_key_from_file

        with pytest.raises(FileNotFoundError):
            load_private_key_from_file("/path/that/does/not/exist.p8")

    def test_private_key_loader_is_callable(self):
        from box_office.utils.snowflake_connection import load_private_key_from_file

        assert callable(load_private_key_from_file)
