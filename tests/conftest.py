"""Pytest configuration and shared fixtures.

Test environment variables (AWS region, Snowflake host, etc.) must be set
*before* ``box_office.config`` is imported, because the config layer reads
them at first attribute access during collection. We can't satisfy that
from inside a fixture (fixtures run after collection imports).

To stay hermetic anyway, we use a module-scoped ``MonkeyPatch`` instance
that records every override we install, and undo it in
``pytest_unconfigure`` so the host shell's environment is restored at the
end of the session.
"""

import os

import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch

# Module-scoped MonkeyPatch instance. Bound below before any test imports.
_test_env_mp = MonkeyPatch()

_test_env = {
    # AWS Configuration (required)
    "AWS_REGION": "eu-north-1",
    "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789:role/test-sagemaker-role",
    "AWS_S3_BUCKET": "box-office-test-bucket",
    # Snowflake Configuration (required)
    "SNOWFLAKE_USER": "test_user",
    "SNOWFLAKE_ACCOUNT": "test_account",
    "SNOWFLAKE_DATABASE": "TEST_DB",
    "SNOWFLAKE_WAREHOUSE": "TEST_WH",
    # Snowflake Schemas (optional but good to have)
    "SNOWFLAKE_SCHEMA_RAW": "RAW",
    "SNOWFLAKE_SCHEMA_STAGING": "STAGING",
    "SNOWFLAKE_SCHEMA_ML_TRAINING": "ML_TRAINING",
    "SNOWFLAKE_SCHEMA_FEATURE_STORE": "FEATURE_STORE",
}

# Apply env vars now so ``box_office.config`` gets sane values during the
# very first import. ``MonkeyPatch.setenv`` records the previous value (or
# absence) so ``undo()`` will restore the host shell at session teardown.
for _key, _value in _test_env.items():
    if not os.environ.get(_key):
        _test_env_mp.setenv(_key, _value)


def pytest_unconfigure(config):  # noqa: ARG001 - pytest hook signature
    """Restore the original environment after the session ends."""
    _test_env_mp.undo()


@pytest.fixture
def sample_movie_data():
    """10-row sample movie dataset with all columns required by the preprocessor."""
    return pd.DataFrame(
        {
            "RELEASE_YEAR": [
                2019,
                2020,
                2020,
                2021,
                2021,
                2022,
                2022,
                2023,
                2023,
                2024,
            ],
            "RELEASE_DATE": pd.to_datetime(
                [
                    "2019-05-15",
                    "2020-02-14",
                    "2020-07-04",
                    "2021-11-25",
                    "2021-12-25",
                    "2022-03-10",
                    "2022-06-15",
                    "2023-01-05",
                    "2023-08-20",
                    "2024-05-01",
                ]
            ),
            "PRODUCTION_BUDGET": [
                50000000,
                100000000,
                30000000,
                70000000,
                150000000,
                45000000,
                25000000,
                80000000,
                60000000,
                120000000,
            ],
            "RUNTIME": [120, 140, 95, 130, 155, 110, 90, 135, 125, 145],
            "DIRECTOR": [
                "Director A",
                "Director B",
                "Director C",
                "Director A",
                "Director B",
                "Director D",
                "Director C",
                "Director A",
                "Director E",
                "Director B",
            ],
            "PRODUCTION_COMPANY": [
                "Warner Bros",
                "Disney",
                "Universal",
                "Warner Bros",
                "Disney",
                "Paramount",
                "Universal",
                "Warner Bros",
                "20th Century",
                "Disney",
            ],
            "ACTORS": [
                "Actor A, Actor B",
                "Actor C, Actor D",
                "Actor E",
                "Actor A, Actor F",
                "Actor C, Actor G",
                "Actor H",
                "Actor E, Actor I",
                "Actor A",
                "Actor J",
                "Actor C",
            ],
            "MPAA": [
                "PG-13",
                "PG-13",
                "R",
                "PG",
                "PG-13",
                "R",
                "PG",
                "PG-13",
                "R",
                "PG-13",
            ],
            "GENRES": [
                "Action, Adventure",
                "Animation, Comedy, Family",
                "Drama, Thriller",
                "Action, Science Fiction",
                "Animation, Family",
                "Horror, Thriller",
                "Comedy",
                "Action, Adventure, Fantasy",
                "Drama",
                "Action, Science Fiction, Adventure",
            ],
            "WORLDWIDE_GROSS": [
                800000000,
                1200000000,
                150000000,
                650000000,
                1500000000,
                300000000,
                100000000,
                750000000,
                450000000,
                1100000000,
            ],
        }
    )


@pytest.fixture
def expected_feature_names():
    """
    Canonical engineered feature names sourced from the live preprocessor.

    Deriving names from ``get_feature_names()`` makes feature additions or
    removals fail loudly across the suite.
    """
    from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

    return FeaturePreprocessorHigh().get_feature_names()


@pytest.fixture
def expected_feature_count(expected_feature_names):
    """Engineered feature count derived from the live preprocessor."""
    return len(expected_feature_names)


@pytest.fixture
def sample_features_data(expected_feature_names):
    """
    Sample preprocessed features for testing downstream components.

    Columns match ``FeaturePreprocessorHigh.get_feature_names()`` so the
    fixture stays in sync with the production preprocessor.
    """
    import numpy as np

    n_rows = 10
    n_features = len(expected_feature_names)

    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((n_rows, n_features))

    return pd.DataFrame(data, columns=expected_feature_names)


@pytest.fixture
def mock_snowflake_connection():
    """MagicMock that simulates snowflake.connector.connect()."""
    from unittest.mock import MagicMock

    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.execute.return_value = None
    mock_cursor.fetchone.return_value = (100,)

    mock_cursor.fetch_pandas_all.return_value = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    )

    return mock_conn


def pytest_collection_modifyitems(config, items):
    """
    Ensure tests are consistently marked for CI selection.

    - Files with "integration" in the nodeid are marked as integration tests.
    - All remaining unmarked tests are treated as unit tests.
    """
    for item in items:
        nodeid_lower = item.nodeid.lower()
        if "integration" in nodeid_lower and "integration" not in item.keywords:
            item.add_marker(pytest.mark.integration)

        if not any(
            marker in item.keywords for marker in ("unit", "integration", "slow")
        ):
            item.add_marker(pytest.mark.unit)
