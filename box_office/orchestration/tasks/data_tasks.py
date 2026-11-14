import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import joblib
import pandas as pd
from dbt.artifacts.schemas.run import RunExecutionResult
from dbt.cli.main import dbtRunnerResult
from prefect import get_run_logger, task
from prefect.tasks import exponential_backoff
from prefect_dbt import PrefectDbtRunner, PrefectDbtSettings
from sklearn.preprocessing import StandardScaler

from box_office.config import config
from box_office.eval_windows import CONFIRMATION_START_YEAR
from box_office.ml.artifacts import FEATURE_PREPROCESSOR_PKL, FEATURE_SCALER_PKL
from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh
from box_office.ml.target import to_log_target
from box_office.utils.approval import require_approval
from box_office.utils.snowflake_connection import (
    create_snowflake_connection,
    enforce_data_types,
)
from box_office.utils.snowflake_loader import (
    fully_qualified_name,
    validate_sql_identifier,
)


@contextmanager
def _scoped_env(updates: dict[str, str]) -> Generator[None]:
    """Apply ``updates`` to ``os.environ`` for the duration of the with block.

    Restores the prior values (or removes the key if it was previously absent)
    on exit so concurrent Prefect tasks on the same worker do not leak env vars
    between runs.
    """
    previous = {k: os.environ.get(k) for k in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for k, prior in previous.items():
            if prior is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prior


@task(retries=3, retry_delay_seconds=exponential_backoff(backoff_factor=5))
def run_raw_to_staging_dbt_transformations(approved: bool = False) -> dbtRunnerResult:
    """
    Run dbt transformations to create staging models from raw data.
    Uses PrefectDbtRunner for better integration with Prefect flows.
    """
    require_approval("building dbt staging models in Snowflake", approved)
    logger = get_run_logger()
    logger.info("Running dbt transformations from raw to staging using PrefectDbtRunner")

    # Get the dbt project directory (transformations folder)
    dbt_project_dir = Path(config.paths.project_root) / config.paths.transformations_dir

    # Build the dbt env contribution from config without mutating os.environ
    # at module level. ``_scoped_env`` will install these for the duration of
    # the dbt invocation and restore the prior values on exit so concurrent
    # tasks don't leak state.
    dbt_env: dict[str, str] = {}
    if "SNOWFLAKE_DATABASE" not in os.environ:
        dbt_env["SNOWFLAKE_DATABASE"] = config.snowflake.database
    if "SNOWFLAKE_WAREHOUSE" not in os.environ:
        dbt_env["SNOWFLAKE_WAREHOUSE"] = config.snowflake.warehouse
    if "SNOWFLAKE_SCHEMA_STAGING" not in os.environ:
        dbt_env["SNOWFLAKE_SCHEMA_STAGING"] = config.snowflake.schemas.staging

    # Resolve SNOWFLAKE_PRIVATE_KEY_PATH to an absolute path for the dbt build
    # (dbt resolves env vars at parse time and a relative path breaks when its
    # cwd differs from ours). Keep the resolved value scoped to the with-block.
    if "SNOWFLAKE_PRIVATE_KEY_PATH" in os.environ:
        key_path = Path(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"])
        if not key_path.is_absolute():
            absolute_key_path = (Path(config.paths.project_root) / key_path).resolve()
            dbt_env["SNOWFLAKE_PRIVATE_KEY_PATH"] = str(absolute_key_path)
            logger.info("Resolved SNOWFLAKE_PRIVATE_KEY_PATH to absolute path: ***MASKED***")
        else:
            logger.info("Key path is already absolute: ***MASKED***")

    profiles_path = dbt_project_dir / "profiles.yml"
    if not profiles_path.exists():
        raise FileNotFoundError(f"profiles.yml not found at: {profiles_path}")

    logger.info(f"Using profiles.yml at: {profiles_path}")
    logger.info(f"dbt project directory: {dbt_project_dir}")

    # Create .dbt directory if it doesn't exist
    dbt_home_dir = Path("~/.dbt").expanduser()
    if not dbt_home_dir.exists():
        dbt_home_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created dbt home directory: {dbt_home_dir}")

    # Initialize PrefectDbtRunner with project directory
    # Use the local profiles.yml in the transformations directory
    runner = PrefectDbtRunner(
        settings=PrefectDbtSettings(
            project_dir=dbt_project_dir,
            profiles_dir=dbt_project_dir,
        ),
        raise_on_failure=True,
    )

    with _scoped_env(dbt_env):
        logger.info("Installing dbt dependencies...")
        # Fail fast: a broken packages.yml or unreachable hub silently masking
        # itself as a "warning" produces confusing downstream dbt errors.
        # Let the error bubble so the caller sees the dbt-deps failure directly.
        runner.invoke(["deps"])
        logger.info("dbt deps completed successfully")

        logger.info("Testing dbt connection...")
        try:
            runner.invoke(["debug"])
            logger.info("dbt connection test successful")
        except ValueError as e:
            logger.error(f"dbt debug failed: {e}")
            raise RuntimeError(f"dbt connection test failed: {e}") from e

        logger.info("Building and testing dbt staging models...")
        run_result = runner.invoke(["build", "--select", "staging"])
        logger.info("dbt staging build completed successfully")

    # Log result details
    build_result = run_result.result
    if isinstance(build_result, RunExecutionResult) and build_result.results:
        logger.info(f"dbt build completed: {len(build_result.results)} nodes processed")

    return run_result


@task(retries=3, retry_delay_seconds=exponential_backoff(backoff_factor=5))
def load_staging_box_office_from_snowflake():
    """Load staging data from Snowflake."""
    logger = get_run_logger()
    logger.info("Loading staging data from Snowflake")

    key_path_env = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    key_path_config = config.snowflake.private_key_path
    logger.debug(
        f"SNOWFLAKE_PRIVATE_KEY_PATH env var: {'***MASKED***' if key_path_env else 'Not set'}"
    )
    logger.debug(f"Config private_key_path: {'***MASKED***' if key_path_config else 'Not set'}")

    # Validate identifiers before any SQL string interpolation. A compromised
    # SNOWFLAKE_DATABASE / SNOWFLAKE_SCHEMA_STAGING env var would otherwise
    # become a SQL-injection path through the f-string below.
    database = validate_sql_identifier(config.snowflake.database, "database")
    schema = validate_sql_identifier(config.snowflake.schemas.staging, "schema")

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


@task
def split_data(
    df: pd.DataFrame, target_column: str, split_year: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the frame at the confirmation set: training rows are the years before it."""
    logger = get_run_logger()

    if split_year is None:
        split_year = CONFIRMATION_START_YEAR

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

    logger.info(f"Training set: {X_train.shape[0]} samples (RELEASE_YEAR < {split_year})")
    logger.info(f"Validation set: {X_val.shape[0]} samples (RELEASE_YEAR >= {split_year})")

    if len(X_train) > 0:
        train_year_range = f"{train_df['RELEASE_YEAR'].min()}-{train_df['RELEASE_YEAR'].max()}"
        logger.info(f"Training year range: {train_year_range}")

    if len(X_val) > 0:
        val_year_range = f"{val_df['RELEASE_YEAR'].min()}-{val_df['RELEASE_YEAR'].max()}"
        logger.info(f"Validation year range: {val_year_range}")

    return X_train, X_val, y_train, y_val


@task
def apply_feature_engineering(
    X_train: pd.DataFrame, X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, FeaturePreprocessorHigh]:
    """Apply feature engineering to the training and validation sets."""
    logger = get_run_logger()
    logger.info("Starting feature preprocessing...")

    processor = FeaturePreprocessorHigh()
    X_train_processed = processor.fit_transform(X_train)
    X_val_processed = processor.transform(X_val)

    logger.info(f"Feature engineering complete. Total features: {X_train_processed.shape[1]}")

    return X_train_processed, X_val_processed, processor


@task
def scale_features(
    X_train_processed: pd.DataFrame, X_val_processed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
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
def transform_targets(y_train: pd.Series, y_val: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Apply log transformation to the target variables."""
    logger = get_run_logger()
    logger.info("Applying log transformation to target variables...")

    y_train_log = pd.Series(to_log_target(y_train), name="GROSS_LOG", index=y_train.index)
    y_val_log = pd.Series(to_log_target(y_val), name="GROSS_LOG", index=y_val.index)

    logger.info("Target transformation complete")

    return y_train_log, y_val_log


@task
def save_artifacts(
    processor: FeaturePreprocessorHigh,
    scaler: StandardScaler,
    artifact_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Save the fitted processor and scaler objects."""
    logger = get_run_logger()

    if artifact_dir is None:
        artifact_dir = config.model.artifacts_dir

    logger.info(f"Saving preprocessing artifacts to {artifact_dir}/")

    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    processor_path = str(Path(artifact_dir) / FEATURE_PREPROCESSOR_PKL)
    scaler_path = str(Path(artifact_dir) / FEATURE_SCALER_PKL)

    joblib.dump(processor, processor_path)
    joblib.dump(scaler, scaler_path)

    logger.info("Artifacts saved successfully")

    return processor_path, scaler_path
