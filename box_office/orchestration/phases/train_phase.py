"""SageMaker training phase (in-memory data handoff)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from box_office.config import config
from box_office.orchestration.phases.data_phase import (
    DataPhaseResult,
    sagemaker_training_frames,
)
from box_office.orchestration.tasks.training_tasks import (
    download_and_analyze_results,
    train_model,
    upload_preprocessing_artifacts_to_s3,
    upload_processed_data_to_s3,
)
from box_office.sagemaker import sagemaker_train_job


@dataclass
class TrainPhaseResult:
    training_metrics: dict[str, Any]
    estimator: Any


def run_train_phase(data: DataPhaseResult, logger) -> TrainPhaseResult:
    """Upload in-memory training data to S3, train, and extract metrics."""
    X_train, y_train = sagemaker_training_frames(data)

    role_arn = config.aws.sagemaker_role_arn
    if role_arn and ":role/" in role_arn:
        role_name = role_arn.split(":role/")[-1]
        account_id = role_arn.split(":")[4]
        masked_role = f"arn:aws:iam::{account_id}:role/{role_name[:20]}***"
    else:
        masked_role = "***MASKED***"

    logger.info("Using SageMaker Role: %s", masked_role)
    logger.info("Using AWS Region: %s", config.aws.region)

    sagemaker_client = sagemaker_train_job.SageMakerClient(
        config.aws.region, config.aws.sagemaker_role_arn
    )
    train_input = upload_processed_data_to_s3(sagemaker_client, X_train, y_train)
    artifacts_s3_uris = upload_preprocessing_artifacts_to_s3(
        sagemaker_client, data.processor_path, data.scaler_path
    )

    estimator = train_model(
        sagemaker_client,
        train_input,
        X_train.shape,
        artifacts_s3_uris,
    )

    training_metrics = download_and_analyze_results(estimator, sagemaker_client)

    return TrainPhaseResult(
        training_metrics=training_metrics,
        estimator=estimator,
    )
