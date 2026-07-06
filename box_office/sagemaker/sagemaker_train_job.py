import logging
import os
import tempfile
import time

import boto3
import pandas as pd

import sagemaker
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.inputs import TrainingInput

from box_office.config import config
from box_office.utils.aws_helpers import BOTO3_CONFIG
from box_office.utils.snowflake_connection import (
    create_snowflake_connection,
    enforce_data_types,
)
from box_office.utils.snowflake_loader import fully_qualified_name

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging - call explicitly from main execution to avoid import-time side effects."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class SageMakerClient:
    def __init__(self, region=None, role=None):
        logger.info("Initializing SageMaker client...")
        # Use configuration system with fallback to parameters
        self.region = region or config.aws.region
        self.role = role or config.aws.sagemaker_role_arn
        self.boto_session = boto3.Session(region_name=self.region)
        # Initialize AWS clients with explicit timeout configuration
        self.sagemaker_client = self.boto_session.client(
            "sagemaker", config=BOTO3_CONFIG
        )
        self.s3_client = self.boto_session.client("s3", config=BOTO3_CONFIG)
        self.sagemaker_session = sagemaker.Session(
            boto_session=self.boto_session,
            sagemaker_client=self.sagemaker_client,
            default_bucket=config.aws.s3_bucket,
        )
        self.s3_bucket = config.aws.s3_bucket
        self.s3_prefix = config.sagemaker.s3_prefix

        logger.info("SageMaker client initialized successfully")
        logger.info(f"Region: {self.region}")
        logger.info(f"S3 Bucket: {self.s3_bucket}")
        logger.info(f"S3 Prefix: {self.s3_prefix}")


def load_processed_data_from_snowflake():
    try:
        logger.info("Connecting to Snowflake...")
        logger.info(f"Account: {config.snowflake.account}")
        logger.info(f"User: {config.snowflake.user}")
        logger.info(f"Database: {config.snowflake.database}")
        logger.info(f"Schema: {config.snowflake.schemas.ml_training}")

        # Release both objects if a query fails mid-fetch.
        conn = create_snowflake_connection(
            schema=config.snowflake.schemas.ml_training, use_private_key=True
        )
        try:
            with conn.cursor() as cursor:
                logger.info(
                    "Loading training data with JOIN on ROW_ID to ensure alignment..."
                )
                # Validate the env-driven database/schema identifiers before
                # interpolating them into the query (the table names are
                # literals). Mirrors the hardening already applied in
                # snowflake_loader.py and data_tasks.py so no execute site
                # interpolates an unvalidated identifier.
                database = config.snowflake.database
                schema = config.snowflake.schemas.ml_training
                x_train_scaled = fully_qualified_name(database, schema, "X_TRAIN_SCALED")
                x_train = fully_qualified_name(database, schema, "X_TRAIN")
                y_train_log = fully_qualified_name(database, schema, "Y_TRAIN_LOG")

                # Single query with JOIN on ROW_ID guarantees perfect row alignment
                query = f"""
            SELECT
                xs.*,
                x.RELEASE_YEAR as RELEASE_YEAR_ORIGINAL,
                y.GROSS_LOG
            FROM {x_train_scaled} xs
            INNER JOIN {x_train} x
                ON xs.ROW_ID = x.ROW_ID
            INNER JOIN {y_train_log} y
                ON xs.ROW_ID = y.ROW_ID
            ORDER BY xs.ROW_ID
        """
                cursor.execute(query)
                combined_data = cursor.fetch_pandas_all()
            combined_data = enforce_data_types(combined_data, table_type="features")

            logger.info(
                f"Combined data loaded: {combined_data.shape[0]:,} rows x {combined_data.shape[1]} columns (aligned by ROW_ID)"
            )
        finally:
            conn.close()

        # Split into features and target, using original RELEASE_YEAR from unscaled table
        X_train_with_dates = combined_data.drop(columns=["GROSS_LOG"])
        X_train_with_dates["RELEASE_YEAR"] = combined_data["RELEASE_YEAR_ORIGINAL"]

        y_train = combined_data[["GROSS_LOG"]].copy()

        logger.info("Successfully loaded all training data from Snowflake!")
        logger.info(f"Total dataset size: {X_train_with_dates.shape[0]:,} samples")
        logger.info(f"Feature dimensions: {X_train_with_dates.shape[1]} features")
        logger.info(
            f"Date range: {X_train_with_dates['RELEASE_YEAR'].min()} - {X_train_with_dates['RELEASE_YEAR'].max()}"
        )
        logger.info(
            "Target column: GROSS_LOG (log-transformed worldwide gross revenue)"
        )

    except Exception as e:
        logger.error(f"Error loading data from Snowflake: {e}")
        raise e

    return X_train_with_dates, y_train


def upload_processed_data_to_s3(sagemaker_client, X_train, y_train):
    logger.info(
        f"Using S3 bucket: {sagemaker_client.s3_bucket} with prefix: {sagemaker_client.s3_prefix}"
    )

    logger.info("Combining features, dates, and target data...")
    train_data = pd.concat([X_train, y_train], axis=1)
    logger.info(
        f"Combined dataset: {train_data.shape[0]:,} rows x {train_data.shape[1]} columns"
    )
    logger.info("Includes RELEASE_YEAR for time series CV")

    # Save to temporary file first, then upload to S3.
    # ``delete=False`` is required so we can hand the path to upload_data after
    # the with-block; the unlink lives in a finally so an upload failure does
    # not leak the temp file.
    logger.info("Creating temporary parquet file...")
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            train_data.to_parquet(tmp_path, index=False)
        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        logger.info(f"Temporary file created: {file_size_mb:.2f} MB")

        # Upload to S3 and get the actual S3 URI where the file was uploaded
        logger.info(f"Uploading to S3 (prefix: {sagemaker_client.s3_prefix}/train)...")
        start_upload = time.time()
        s3_train_uri = sagemaker_client.sagemaker_session.upload_data(
            path=tmp_path,
            bucket=sagemaker_client.s3_bucket,
            key_prefix=f"{sagemaker_client.s3_prefix}/train",
        )
        upload_time = time.time() - start_upload

        logger.info(f"Upload completed in {upload_time:.2f} seconds")
        logger.info(f"S3 URI: {s3_train_uri}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                logger.info("Temporary file cleaned up")
            except OSError as e:
                logger.warning(f"Could not unlink temp file {tmp_path}: {e}")

    train_input = TrainingInput(s3_train_uri, content_type="application/x-parquet")
    logger.info("Training data successfully prepared for SageMaker!")

    return train_input


def _validate_hyperparameters(hyperparameters, logger):
    """Validate hyperparameters and log any validation errors clearly."""
    validation_errors = []

    # Define expected parameter ranges and types
    validation_rules = {
        "learning_rate": {"type": (int, float), "min": 0.001, "max": 1.0},
        "max_depth": {"type": int, "min": 1, "max": 20},
        "min_child_weight": {"type": (int, float), "min": 0, "max": 100},
        "reg_alpha": {"type": (int, float), "min": 0, "max": 100},
        "reg_lambda": {"type": (int, float), "min": 0, "max": 100},
        "early_stopping_rounds": {"type": int, "min": 1, "max": 1000},
        "n_estimators": {"type": int, "min": 1, "max": 10000},
        "cv_folds": {"type": int, "min": 2, "max": 20},
        "start_eval_year": {"type": int, "min": 1900, "max": 2030},
        "end_year": {"type": int, "min": 1900, "max": 2030},
    }

    # Validate each parameter
    for param, value in hyperparameters.items():
        if param in validation_rules:
            rules = validation_rules[param]

            # Check type
            if not isinstance(value, rules["type"]):
                validation_errors.append(
                    f"{param}: Expected {rules['type']}, got {type(value).__name__} ({value})"
                )
                continue

            # Check range
            if "min" in rules and value < rules["min"]:
                validation_errors.append(
                    f"{param}: Value {value} is below minimum {rules['min']}"
                )
            if "max" in rules and value > rules["max"]:
                validation_errors.append(
                    f"{param}: Value {value} is above maximum {rules['max']}"
                )

    # Check logical constraints
    if "start_eval_year" in hyperparameters and "end_year" in hyperparameters:
        if hyperparameters["start_eval_year"] >= hyperparameters["end_year"]:
            validation_errors.append("start_eval_year must be less than end_year")

    # Log validation results
    if validation_errors:
        logger.error("Hyperparameter validation failed:")
        for error in validation_errors:
            logger.error(f"- {error}")
        raise ValueError(f"Invalid hyperparameters: {'; '.join(validation_errors)}")
    else:
        logger.info("Hyperparameter validation passed")


def _log_hyperparameters_block(hyperparameters, logger):
    """Log hyperparameters as structured block instead of individual messages."""
    logger.info("Model and CV hyperparameters:")

    # Group related parameters logically
    model_params = {}
    cv_params = {}
    other_params = {}

    for param, value in hyperparameters.items():
        if param in [
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "early_stopping_rounds",
            "n_estimators",
        ]:
            model_params[param] = value
        elif param in ["cv_folds", "start_eval_year", "end_year"]:
            cv_params[param] = value
        else:
            other_params[param] = value

    # Log model parameters
    if model_params:
        logger.info("Model Parameters:")
        for param, value in model_params.items():
            logger.info(f"{param}: {value}")

    # Log CV parameters
    if cv_params:
        logger.info("Cross-Validation Parameters:")
        for param, value in cv_params.items():
            logger.info(f"{param}: {value}")

    # Log other parameters
    if other_params:
        logger.info("Other Parameters:")
        for param, value in other_params.items():
            logger.info(f"{param}: {value}")


def train_xgboost_model_with_timeseries_cv_framework(
    sagemaker_client,
    train_input,
    X_train_shape,
    hyperparameters,
    experiment_config=None,
):
    """Launch a SageMaker training job using the XGBoost Framework Estimator,
    running the in-container time-series CV training script."""

    XGBOOST_PARAM_KEYS = {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "early_stopping_rounds",
    }

    # Separate XGBoost hyperparameters from custom script parameters
    xgboost_params = {
        k: str(v)
        for k, v in hyperparameters.items()
        if k in XGBOOST_PARAM_KEYS and v is not None
    }

    # Only pass custom parameters as environment variables (not XGBoost hyperparameters)
    custom_params = {
        k: str(v) if v is not None else "None"
        for k, v in hyperparameters.items()
        if k not in XGBOOST_PARAM_KEYS
    }

    # Validate hyperparameters first
    _validate_hyperparameters(hyperparameters, logger)

    # Log hyperparameters in structured format - use original for logging
    _log_hyperparameters_block(hyperparameters, logger)

    logger.info("Configuring SageMaker XGBoost Framework Estimator...")
    logger.info(f"XGBoost hyperparameters: {xgboost_params}")
    logger.info(f"Custom script parameters (env vars): {custom_params}")

    # The source_dir needs to include the full box_office package structure
    # so that imports like "from box_office.ml.model" work correctly.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    xgb_estimator = XGBoost(
        entry_point="box_office/ml/model_training.py",
        source_dir=project_root,
        role=sagemaker_client.role,
        instance_type=config.sagemaker.instance_type,
        instance_count=1,
        framework_version=config.sagemaker.framework_version,
        py_version="py3",  # Required for framework mode
        output_path=f"s3://{sagemaker_client.s3_bucket}/{sagemaker_client.s3_prefix}/output",
        sagemaker_session=sagemaker_client.sagemaker_session,
        hyperparameters=xgboost_params,  # XGBoost hyperparameters
        environment=custom_params,  # Custom parameters for training script
        max_run=1800,  # 30 minutes
    )

    logger.info(
        "Starting SageMaker XGBoost training job: %s samples x %s features, %s-fold CV (%s-%s)",
        f"{X_train_shape[0]:,}",
        X_train_shape[1],
        hyperparameters["cv_folds"],
        hyperparameters["start_eval_year"],
        hyperparameters["end_year"],
    )

    start_time = time.time()
    if experiment_config:
        xgb_estimator.fit({"train": train_input}, experiment_config=experiment_config)
    else:
        xgb_estimator.fit({"train": train_input})
    training_time = time.time() - start_time

    logger.info(
        "XGBoost training job %s completed in %.1f s (%.2f min)",
        xgb_estimator.latest_training_job.name,
        training_time,
        training_time / 60,
    )

    return xgb_estimator
