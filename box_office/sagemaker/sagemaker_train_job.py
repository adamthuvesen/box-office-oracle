import logging
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import boto3
import pandas as pd
from pydantic import ValidationError
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.session import Session
from sagemaker.workflow.entities import PipelineVariable

from box_office.config import CrossValidationWindow, ModelHyperparameters, config
from box_office.utils.approval import require_approval
from box_office.utils.aws_helpers import BOTO3_CONFIG

logger = logging.getLogger(__name__)

XGBOOST_PARAM_KEYS = frozenset(ModelHyperparameters.model_fields)


class SageMakerClient:
    def __init__(
        self, region: str | None = None, role: str | None = None, approved: bool = False
    ) -> None:
        require_approval("opening a SageMaker session", approved)
        logger.info("Initializing SageMaker client...")
        # Use configuration system with fallback to parameters
        self.region = region or config.aws.region
        self.role = role or config.aws.sagemaker_role_arn
        self.boto_session = boto3.Session(region_name=self.region)
        # Initialize AWS clients with explicit timeout configuration
        self.sagemaker_client = self.boto_session.client("sagemaker", config=BOTO3_CONFIG)
        self.s3_client = self.boto_session.client("s3", config=BOTO3_CONFIG)
        self.sagemaker_session = Session(
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


def upload_processed_data_to_s3(
    sagemaker_client: SageMakerClient,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    approved: bool = False,
) -> TrainingInput:
    require_approval("uploading training data to S3", approved)
    logger.info(
        f"Using S3 bucket: {sagemaker_client.s3_bucket} with prefix: {sagemaker_client.s3_prefix}"
    )

    logger.info("Combining features, dates, and target data...")
    train_data = pd.concat([X_train, y_train], axis=1)
    logger.info(f"Combined dataset: {train_data.shape[0]:,} rows x {train_data.shape[1]} columns")
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
        file_size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
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
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
                logger.info("Temporary file cleaned up")
            except OSError as e:
                logger.warning(f"Could not unlink temp file {tmp_path}: {e}")

    train_input = TrainingInput(s3_train_uri, content_type="application/x-parquet")
    logger.info("Training data successfully prepared for SageMaker!")

    return train_input


def _validate_hyperparameters(
    hyperparameters: Mapping[str, object], logger: logging.Logger
) -> None:
    """Validate hyperparameters and log any validation errors clearly."""
    validation_errors = []

    try:
        ModelHyperparameters.model_validate(
            {k: v for k, v in hyperparameters.items() if k in XGBOOST_PARAM_KEYS}
        )
    except ValidationError as e:
        validation_errors.extend(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
        )

    cv_keys = set(CrossValidationWindow.model_fields)
    try:
        CrossValidationWindow.model_validate(
            {k: v for k, v in hyperparameters.items() if k in cv_keys}
        )
    except ValidationError as e:
        validation_errors.extend(
            f"{'.'.join(map(str, err['loc'])) or 'cross_validation'}: {err['msg']}"
            for err in e.errors()
        )

    # Log validation results
    if validation_errors:
        logger.error("Hyperparameter validation failed:")
        for error in validation_errors:
            logger.error(f"- {error}")
        raise ValueError(f"Invalid hyperparameters: {'; '.join(validation_errors)}")
    else:
        logger.info("Hyperparameter validation passed")


def _log_hyperparameters_block(
    hyperparameters: Mapping[str, object], logger: logging.Logger
) -> None:
    """Log hyperparameters as structured block instead of individual messages."""
    logger.info("Model and CV hyperparameters:")

    # Group related parameters logically
    model_params: dict[str, object] = {}
    cv_params: dict[str, object] = {}
    other_params: dict[str, object] = {}

    for param, value in hyperparameters.items():
        if param in XGBOOST_PARAM_KEYS:
            model_params[param] = value
        elif param in CrossValidationWindow.model_fields:
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
    sagemaker_client: SageMakerClient,
    train_input: TrainingInput,
    X_train_shape: tuple[int, int],
    hyperparameters: Mapping[str, object],
    experiment_config: dict[str, str] | None = None,
    approved: bool = False,
) -> Estimator:
    """Launch a SageMaker training job on the repository's training image.

    The image (infrastructure/docker/training) runs
    ``box_office.ml.model_training``: time-series CV and the final fit.
    """
    require_approval("launching a SageMaker training job", approved)
    image_uri = config.sagemaker.training_image_uri
    if not image_uri:
        raise ValueError(
            "SAGEMAKER_TRAINING_IMAGE_URI is not set. Build and push the training "
            "image first (docs/training-image.md)."
        )

    # Separate XGBoost hyperparameters from custom script parameters
    xgboost_params: dict[str, str | PipelineVariable] = {
        k: str(v) for k, v in hyperparameters.items() if k in XGBOOST_PARAM_KEYS and v is not None
    }

    # Only pass custom parameters as environment variables (not XGBoost hyperparameters)
    custom_params: dict[str, str | PipelineVariable] = {
        k: str(v) if v is not None else "None"
        for k, v in hyperparameters.items()
        if k not in XGBOOST_PARAM_KEYS
    }

    # Validate hyperparameters first
    _validate_hyperparameters(hyperparameters, logger)

    # Log hyperparameters in structured format - use original for logging
    _log_hyperparameters_block(hyperparameters, logger)

    logger.info("Configuring SageMaker Estimator on image %s", image_uri)
    logger.info(f"XGBoost hyperparameters: {xgboost_params}")
    logger.info(f"Custom script parameters (env vars): {custom_params}")

    xgb_estimator = Estimator(
        image_uri=image_uri,
        role=sagemaker_client.role,
        instance_type=config.sagemaker.instance_type,
        instance_count=1,
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
    training_job = xgb_estimator.latest_training_job
    if training_job is None:
        raise RuntimeError("SageMaker fit returned no training job")

    logger.info(
        "XGBoost training job %s completed in %.1f s (%.2f min)",
        training_job.name,
        training_time,
        training_time / 60,
    )

    return xgb_estimator
