import os
import joblib
import pandas as pd
from contextlib import contextmanager
from prefect import task, get_run_logger
from prefect.tasks import exponential_backoff
from prefect_dbt import PrefectDbtRunner, PrefectDbtSettings
from sklearn.preprocessing import StandardScaler

from box_office.config import config
from box_office.utils.snowflake_connection import (
    create_snowflake_connection,
    enforce_data_types,
)
from box_office.utils.snowflake_loader import (
    fully_qualified_name,
    validate_sql_identifier,
)
from box_office.ml.artifacts import FEATURE_PREPROCESSOR_PKL, FEATURE_SCALER_PKL
from box_office.ml.data_prep import TargetTransformer
from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
from snowflake.connector.pandas_tools import write_pandas


@contextmanager
def _scoped_env(updates: dict):
    """Apply ``updates`` to ``os.environ`` for the duration of the with block.

    Restores the prior values (or removes the key if it was previously absent)
    on exit. Concurrent Prefect tasks on the same worker were leaking env vars
    between runs because the previous code used ``os.environ.setdefault`` /
    direct assignment without ever undoing the change.
    """
    sentinel = object()
    previous = {k: os.environ.get(k, sentinel) for k in updates}
    try:
        for k, v in updates.items():
            if v is not None:
                os.environ[k] = v
        yield
    finally:
        for k, prior in previous.items():
            if prior is sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prior


@task(retries=3, retry_delay_seconds=exponential_backoff(backoff_factor=5))
def run_raw_to_staging_dbt_transformations():
    """
    Run dbt transformations to create staging models from raw data.
    Uses PrefectDbtRunner for better integration with Prefect flows.
    """
    logger = get_run_logger()
    logger.info(
        "Running dbt transformations from raw to staging using PrefectDbtRunner"
    )

    # Get the dbt project directory (transformations folder)
    dbt_project_dir = os.path.join(
        config.paths.project_root, config.paths.transformations_dir
    )

    # Build the dbt env contribution from config without mutating os.environ
    # at module level. ``_scoped_env`` will install these for the duration of
    # the dbt invocation and restore the prior values on exit so concurrent
    # tasks don't leak state.
    dbt_env: dict = {}
    if "SNOWFLAKE_DATABASE" not in os.environ:
        dbt_env["SNOWFLAKE_DATABASE"] = config.snowflake.database
    if "SNOWFLAKE_WAREHOUSE" not in os.environ:
        dbt_env["SNOWFLAKE_WAREHOUSE"] = config.snowflake.warehouse
    if "SNOWFLAKE_SCHEMA_STAGING" not in os.environ:
        dbt_env["SNOWFLAKE_SCHEMA_STAGING"] = config.snowflake.schemas.staging

    # Resolve SNOWFLAKE_PRIVATE_KEY_PATH to an absolute path for the dbt run
    # (dbt resolves env vars at parse time and a relative path breaks when its
    # cwd differs from ours). Keep the resolved value scoped to the with-block.
    if "SNOWFLAKE_PRIVATE_KEY_PATH" in os.environ:
        key_path = os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"]
        if not os.path.isabs(key_path):
            absolute_key_path = os.path.abspath(
                os.path.join(config.paths.project_root, key_path)
            )
            dbt_env["SNOWFLAKE_PRIVATE_KEY_PATH"] = absolute_key_path
            logger.info(
                "Resolved SNOWFLAKE_PRIVATE_KEY_PATH to absolute path: ***MASKED***"
            )
        else:
            logger.info("Key path is already absolute: ***MASKED***")

    try:
        profiles_path = os.path.join(dbt_project_dir, "profiles.yml")
        if not os.path.exists(profiles_path):
            raise FileNotFoundError(f"profiles.yml not found at: {profiles_path}")

        logger.info(f"Using profiles.yml at: {profiles_path}")
        logger.info(f"dbt project directory: {dbt_project_dir}")

        # Create .dbt directory if it doesn't exist
        dbt_home_dir = os.path.expanduser("~/.dbt")
        if not os.path.exists(dbt_home_dir):
            os.makedirs(dbt_home_dir, exist_ok=True)
            logger.info(f"Created dbt home directory: {dbt_home_dir}")

        # Initialize PrefectDbtRunner with project directory
        # Use the local profiles.yml in the transformations directory
        runner = PrefectDbtRunner(
            settings=PrefectDbtSettings(
                project_dir=dbt_project_dir,
                profiles_dir=dbt_project_dir,
            )
        )

        with _scoped_env(dbt_env):
            logger.info("Installing dbt dependencies...")
            # Fail fast: a broken packages.yml or unreachable hub silently masking
            # itself as a "warning" produces confusing downstream `dbt run` errors.
            # Let the error bubble so the caller sees the dbt-deps failure directly.
            runner.invoke(["deps"])
            logger.info("dbt deps completed successfully")

            logger.info("Testing dbt connection...")
            try:
                runner.invoke(["debug"])
                logger.info("dbt connection test successful")
            except Exception as e:
                logger.error(f"dbt debug failed: {e}")
                raise RuntimeError(f"dbt connection test failed: {e}")

            logger.info("Running dbt transformations for staging models...")
            run_result = runner.invoke(["run", "--select", "staging"])
            logger.info("dbt transformations completed successfully")

        # Log result details
        if hasattr(run_result, "results") and run_result.results:
            logger.info(
                f"dbt run completed: {len(run_result.results)} models processed"
            )

        return run_result

    except Exception as e:
        logger.error(f"Error running dbt transformations: {e}")
        raise


@task(retries=3, retry_delay_seconds=exponential_backoff(backoff_factor=5))
def load_staging_box_office_from_snowflake():
    """Load staging data from Snowflake."""
    logger = get_run_logger()
    logger.info("Loading staging data from Snowflake")

    key_path_env = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    key_path_config = getattr(config.snowflake, "private_key_path", None)
    logger.debug(
        f"SNOWFLAKE_PRIVATE_KEY_PATH env var: {'***MASKED***' if key_path_env else 'Not set'}"
    )
    logger.debug(
        f"Config private_key_path: {'***MASKED***' if key_path_config else 'Not set'}"
    )

    # Validate identifiers before any SQL string interpolation. A compromised
    # SNOWFLAKE_DATABASE / SNOWFLAKE_SCHEMA_STAGING env var would otherwise
    # become a SQL-injection path through the f-string below.
    database = validate_sql_identifier(config.snowflake.database, "database")
    schema = validate_sql_identifier(config.snowflake.schemas.staging, "schema")

    try:
        conn = create_snowflake_connection(schema=schema, use_private_key=True)

        try:
            with conn.cursor() as cursor:
                table = fully_qualified_name(database, schema, "STG_BOX_OFFICE")
                query = f"SELECT * FROM {table}"
                cursor.execute(query)
                df = cursor.fetch_pandas_all()
        finally:
            conn.close()

        # Enforce proper data types
        df = enforce_data_types(df, table_type="training")

        logger.info(f"Successfully fetched {len(df)} rows from Snowflake.")
        logger.info(f"Column names: {list(df.columns)}")

        return df

    except Exception as e:
        logger.error(f"Error loading data from Snowflake: {e}")
        raise


@task
def split_data(df, target_column, split_year=None):
    """Split the data into training and validation sets based on RELEASE_YEAR."""
    logger = get_run_logger()

    if split_year is None:
        split_year = config.model.cross_validation.end_year

    logger.info("Splitting data into train/validation sets based on RELEASE_YEAR")
    logger.info(f"Training set: RELEASE_YEAR < {split_year}")
    logger.info(f"Validation set: RELEASE_YEAR >= {split_year}")

    if "RELEASE_YEAR" not in df.columns:
        logger.error("RELEASE_YEAR column not found in dataframe")
        raise ValueError("RELEASE_YEAR column is required for temporal split")

    train_mask = df["RELEASE_YEAR"] < split_year
    val_mask = df["RELEASE_YEAR"] >= split_year

    train_df = df[train_mask].copy()
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    val_df = df[val_mask].copy()
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    logger.info(
        f"Training set: {X_train.shape[0]} samples (RELEASE_YEAR < {split_year})"
    )
    logger.info(
        f"Validation set: {X_val.shape[0]} samples (RELEASE_YEAR >= {split_year})"
    )

    if len(X_train) > 0:
        train_year_range = (
            f"{train_df['RELEASE_YEAR'].min()}-{train_df['RELEASE_YEAR'].max()}"
        )
        logger.info(f"Training year range: {train_year_range}")

    if len(X_val) > 0:
        val_year_range = (
            f"{val_df['RELEASE_YEAR'].min()}-{val_df['RELEASE_YEAR'].max()}"
        )
        logger.info(f"Validation year range: {val_year_range}")

    return X_train, X_val, y_train, y_val


@task
def apply_feature_engineering(X_train, X_val):
    """Apply feature engineering to the training and validation sets."""
    logger = get_run_logger()
    logger.info("Starting feature preprocessing...")

    processor = FeaturePreprocessorHigh()
    processor.fit(X_train)

    X_train_processed = processor.transform(X_train)
    X_val_processed = processor.transform(X_val)

    logger.info(
        f"Feature engineering complete. Total features: {X_train_processed.shape[1]}"
    )

    return X_train_processed, X_val_processed, processor


@task
def scale_features(X_train_processed, X_val_processed):
    """Apply standard scaling to the processed features."""
    logger = get_run_logger()
    logger.info("Applying standard scaling to features...")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_processed),
        columns=X_train_processed.columns,
        index=X_train_processed.index,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_processed),
        columns=X_val_processed.columns,
        index=X_val_processed.index,
    )

    logger.info("Scaling complete")

    return X_train_scaled, X_val_scaled, scaler


@task
def transform_targets(y_train, y_val):
    """Apply log transformation to the target variables."""
    logger = get_run_logger()
    logger.info("Applying log transformation to target variables...")

    y_train_log, y_val_log = TargetTransformer.log_transform(y_train, y_val)

    y_train_log = pd.Series(y_train_log, name="GROSS_LOG", index=y_train.index)
    y_val_log = pd.Series(y_val_log, name="GROSS_LOG", index=y_val.index)

    logger.info("Target transformation complete")

    return y_train_log, y_val_log


@task
def save_artifacts(processor, scaler, artifact_dir=None):
    """Save the fitted processor and scaler objects."""
    logger = get_run_logger()

    if artifact_dir is None:
        artifact_dir = config.model.artifacts_dir

    logger.info(f"Saving preprocessing artifacts to {artifact_dir}/")

    os.makedirs(artifact_dir, exist_ok=True)

    processor_path = os.path.join(artifact_dir, FEATURE_PREPROCESSOR_PKL)
    scaler_path = os.path.join(artifact_dir, FEATURE_SCALER_PKL)

    joblib.dump(processor, processor_path)
    joblib.dump(scaler, scaler_path)

    logger.info("Artifacts saved successfully")

    return processor_path, scaler_path


def save_dataset_to_snowflake_impl(
    df: pd.DataFrame, table_name: str, schema: str | None = None
) -> bool:
    """Save a DataFrame to Snowflake (plain function for batch saves and tests)."""
    if schema is None:
        schema = config.snowflake.schemas.ml_training

    database = validate_sql_identifier(config.snowflake.database, "database")
    schema = validate_sql_identifier(schema, "schema")
    validate_sql_identifier(table_name, "table")

    conn = create_snowflake_connection(schema=schema)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"USE DATABASE {database}")
            cursor.execute(f"USE SCHEMA {schema}")

        df_clean = df.reset_index(drop=False)
        df_clean = df_clean.rename(columns={"index": "ROW_ID"})

        success, _nchunks, _nrows, _ = write_pandas(
            conn=conn,
            df=df_clean,
            table_name=table_name.upper(),
            schema=schema,
            auto_create_table=True,
            overwrite=True,
        )
        return success
    finally:
        conn.close()


@task
def create_feature_metadata(feature_names, processor_path, scaler_path):
    """Create feature metadata DataFrame."""
    logger = get_run_logger()
    logger.info("Creating feature metadata...")

    feature_metadata = pd.DataFrame(
        {
            "FEATURE_NAME": feature_names,
            "FEATURE_INDEX": range(len(feature_names)),
            "CREATED_AT": pd.Timestamp.now(),
            "PROCESSOR_PATH": processor_path,
            "SCALER_PATH": scaler_path,
        }
    )

    logger.info(f"Created metadata for {len(feature_names)} features")
    return feature_metadata


@task
def validate_snowflake_tables(expected_tables, schema=None):
    """Validate that all expected tables exist in Snowflake with data."""
    logger = get_run_logger()
    logger.info("Validating saved datasets in Snowflake...")

    if schema is None:
        schema = config.snowflake.schemas.ml_training

    # Validate every identifier before SQL interpolation. Caller-supplied
    # ``expected_tables`` strings are validated per-iteration so a single
    # bad entry doesn't poison the others.
    database = validate_sql_identifier(config.snowflake.database, "database")
    schema = validate_sql_identifier(schema, "schema")

    try:
        conn = create_snowflake_connection(schema=schema, use_private_key=True)

        try:
            with conn.cursor() as cursor:
                cursor.execute(f"USE DATABASE {database}")
                cursor.execute(f"USE SCHEMA {schema}")

                validation_results = {}

                for table in expected_tables:
                    try:
                        validate_sql_identifier(table, "table")
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        row_count = cursor.fetchone()[0]
                        validation_results[table] = row_count
                        logger.info(f"{table}: {row_count} rows")
                    except Exception as e:
                        validation_results[table] = f"Error: {e}"
                        logger.error(f"{table}: {e}")
        finally:
            conn.close()

        return validation_results

    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return {"error": str(e)}
