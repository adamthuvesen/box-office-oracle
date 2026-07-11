"""
Snowflake connection utilities with key-pair authentication support.

This module provides helper functions for creating Snowflake connections
using both password and private key authentication methods.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from box_office.config import config

logger = logging.getLogger(__name__)


def load_private_key_from_file(
    private_key_path: str, passphrase: str | None = None
) -> bytes:
    """Load a .p8 private key and return DER/PKCS8 bytes as expected by snowflake-connector-python."""
    if not os.path.exists(private_key_path):
        raise FileNotFoundError(f"Private key file not found: {private_key_path}")

    try:
        with open(private_key_path, "rb") as key_file:
            private_key_data = key_file.read()

        password_bytes = None
        if passphrase and passphrase.strip():
            password_bytes = passphrase.encode()

        private_key = load_pem_private_key(private_key_data, password=password_bytes)

        private_key_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        return private_key_bytes

    except Exception as e:
        raise ValueError(
            f"Failed to load private key from {private_key_path}: {str(e)}"
        ) from e


def create_snowflake_connection(
    schema: str | None = None,
    use_private_key: bool = True,
    private_key_passphrase: str | None = None,
    use_browser_auth: bool = False,
    **additional_params,
) -> snowflake.connector.SnowflakeConnection:
    """Create a Snowflake connection using key-pair (preferred) or password auth.

    Credentials resolve through ``box_office.config.config``, which layers
    environment variables and ``.env`` over baked-in defaults.
    """
    if schema is None:
        schema = config.snowflake.schemas.staging

    connection_params = {
        "user": config.snowflake.user,
        "account": config.snowflake.account,
        "warehouse": config.snowflake.warehouse,
        "database": config.snowflake.database,
        "schema": schema,
        "disable_ocsp_checks": True,
    }

    # Without an explicit role we'd authenticate as the user's default — which
    # in CI is a high-privilege role with no STAGING write grants. dbt picks up
    # SNOWFLAKE_ROLE via profiles.yml; do the same here so the Python pipeline
    # writes through the same least-privilege role.
    snowflake_role = os.environ.get("SNOWFLAKE_ROLE")
    if snowflake_role:
        connection_params["role"] = snowflake_role

    if use_private_key:
        # Prefer the env var so a redeploy can override the key path without a code change.
        private_key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
        config_private_key_path = config.snowflake.private_key_path

        logger.info("Attempting private key authentication...")
        logger.info(
            f"SNOWFLAKE_PRIVATE_KEY_PATH env var: {'***MASKED***' if private_key_path else 'Not set'}"
        )
        logger.info(
            f"Private key path from config: {'***MASKED***' if config_private_key_path else 'Not set'}"
        )

        if not private_key_path:
            private_key_path = config_private_key_path
            if private_key_path:
                logger.info("Using private key path from config: ***MASKED***")
            else:
                logger.warning(
                    "Private key path not configured in environment or config, falling back to password authentication"
                )
                use_private_key = False
        else:
            logger.info(
                "Using private key path from environment variable: ***MASKED***"
            )

        if private_key_path and use_private_key:
            try:
                key_path = Path(private_key_path)
                if not key_path.is_absolute():
                    key_path = (Path(config.paths.project_root) / key_path).resolve()

                private_key_path = str(key_path)

                if not key_path.exists():
                    logger.error(f"Private key file not found: {private_key_path}")
                    raise FileNotFoundError(
                        f"Private key file not found: {private_key_path}"
                    )

                logger.info("Private key file path: ***MASKED***")

                if private_key_passphrase is None:
                    private_key_passphrase = os.environ.get(
                        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE"
                    )

                # Treat empty / whitespace passphrase as None.
                if (
                    private_key_passphrase is not None
                    and not private_key_passphrase.strip()
                ):
                    private_key_passphrase = None

                private_key = load_private_key_from_file(
                    private_key_path, private_key_passphrase
                )
                connection_params["private_key"] = private_key
                auth_method = (
                    "private key (encrypted)"
                    if private_key_passphrase
                    else "private key (unencrypted)"
                )
                logger.info(
                    f"Successfully loaded private key - using {auth_method} authentication"
                )
            except FileNotFoundError as e:
                # Legitimate fallback: the configured key path simply doesn't
                # exist on this host (e.g. local dev without the secret). Drop
                # to password / browser auth.
                logger.warning(
                    f"Private key file missing, falling back to password auth: {e}"
                )
                use_private_key = False
            except Exception:
                # Decryption errors, permission errors, malformed PEM, etc. —
                # silently swapping identity here was the C-tier root of H20.
                # Re-raise so the caller surfaces the real problem instead of
                # silently authenticating as a different role.
                logger.error(
                    "Failed to load private key (not a missing-file case); refusing to fall back"
                )
                raise

    if not use_private_key:
        if (
            use_browser_auth
            or os.environ.get("SNOWFLAKE_AUTHENTICATOR") == "externalbrowser"
        ):
            connection_params["authenticator"] = "externalbrowser"
            logger.info(
                "Using browser-based SSO authentication for Snowflake connection"
            )
        else:
            password = config.snowflake.password
            if not password:
                raise ValueError(
                    "Neither private key nor password is configured for Snowflake authentication"
                )
            connection_params["password"] = password
            logger.info("Using password authentication for Snowflake connection")

    # Caller-supplied params take precedence over our defaults.
    connection_params.update(additional_params)

    logger.info(
        f"Connecting to Snowflake - Account: {config.snowflake.account}, "
        f"User: {config.snowflake.user}, Database: {config.snowflake.database}, "
        f"Schema: {schema}"
    )

    try:
        conn = snowflake.connector.connect(**connection_params)
        logger.info("Snowflake connection established successfully")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise


def enforce_data_types(df: pd.DataFrame, table_type: str = "training") -> pd.DataFrame:
    """Coerce expected columns to numeric/datetime types after Snowflake load."""
    logger.info(f"Enforcing data types for {table_type} data...")

    numerical_columns = {
        "training": [
            "RELEASE_YEAR",
            "PRODUCTION_BUDGET",
            "RUNTIME",
        ],
        "features": [
            "RELEASE_YEAR",
            "PRODUCTION_BUDGET",
            "RUNTIME",
            "MPAA_ENCODED",
            "SUPER_GENRE_ENCODED",
            "RELEASE_MONTH",
            "RELEASE_WEEK",
            "RELEASE_QUARTER",
            "DIRECTOR_FREQ",
            "COMPANY_FREQ",
            "LEAD_ACTOR_FREQ",
            "AVG_ACTOR_FREQ",
            "MAX_ACTOR_FREQ",
            "YEARS_SINCE_2000",
        ],
        "target": ["GROSS_LOG", "WORLDWIDE_GROSS"],
    }

    expected_numerical = numerical_columns.get(table_type, [])

    df_converted = df.copy()
    converted_count = 0

    for col in df_converted.columns:
        original_dtype = df_converted[col].dtype

        if col in expected_numerical or col.startswith(
            ("GENRE_", "IS_", "BUDGET_", "RATING_", "COVID_")
        ):
            try:
                # Snowflake sometimes returns NULLs as strings; normalize before numeric coerce.
                if df_converted[col].dtype == "object":
                    df_converted[col] = df_converted[col].replace(
                        ["NULL", "null", "None", "NaN", ""], np.nan
                    )

                df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce")

                if original_dtype != df_converted[col].dtype:
                    converted_count += 1
                    logger.debug(f"{col}: {original_dtype}  {df_converted[col].dtype}")

            except Exception as e:
                logger.warning(f"Could not convert {col} to numeric: {e}")

        elif col in ["RELEASE_DATE", "release_date"]:
            try:
                if df_converted[col].dtype == "object":
                    df_converted[col] = pd.to_datetime(
                        df_converted[col], errors="coerce"
                    )
                    if original_dtype != df_converted[col].dtype:
                        converted_count += 1
                        logger.debug(f"{col}: {original_dtype}  datetime64")
            except Exception as e:
                logger.warning(f"Could not convert {col} to datetime: {e}")

    logger.info(f"Data type enforcement complete: {converted_count} columns converted")

    numeric_cols = df_converted.select_dtypes(include=[np.number]).columns
    logger.info(
        f"Final numeric columns: {len(numeric_cols)}/{len(df_converted.columns)}"
    )

    return df_converted
