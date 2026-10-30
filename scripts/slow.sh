#!/usr/bin/env bash
# The slow gate: the full test suite and an offline dbt parse. Needs network
# access to fetch dbt packages, and no Snowflake credentials.
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync --locked --extra dev --extra inference --extra train --quiet
uv run --no-sync pytest tests/

export SNOWFLAKE_ACCOUNT="${SNOWFLAKE_ACCOUNT:-offline}"
export SNOWFLAKE_USER="${SNOWFLAKE_USER:-offline}"
export SNOWFLAKE_ROLE="${SNOWFLAKE_ROLE:-DBT_RUNNER}"
export SNOWFLAKE_DATABASE="${SNOWFLAKE_DATABASE:-BOX_OFFICE}"
export SNOWFLAKE_WAREHOUSE="${SNOWFLAKE_WAREHOUSE:-COMPUTE_WH}"
export SNOWFLAKE_SCHEMA_STAGING="${SNOWFLAKE_SCHEMA_STAGING:-STAGING}"
export SNOWFLAKE_PRIVATE_KEY_PATH="${SNOWFLAKE_PRIVATE_KEY_PATH:-/nonexistent/offline-snowflake-key.p8}"

cd transformations
uv run --no-sync dbt deps --profiles-dir=.
uv run --no-sync dbt parse --profiles-dir=.
