from prefect import task, get_run_logger
from prefect.cache_policies import NONE as NO_CACHE
from prefect.tasks import exponential_backoff
import json
import boto3
from botocore.exceptions import ClientError
import time
from datetime import datetime, timezone
import pandas as pd

from box_office.config import config
from box_office.sagemaker import sagemaker_train_job
from box_office.ml.model_registry.aws_model_registry import (
    AWSModelRegistry,
    ModelRegistryRegistrationError,
)
from box_office.utils.aws_helpers import BOTO3_CONFIG
from box_office.utils.format_helpers import safe_format

# S3 error codes that indicate a missing object (legitimate "no metrics yet" path).
_S3_NOT_FOUND_CODES = frozenset({"NoSuchKey", "404", "NoSuchBucket"})

# SageMaker describe_training_job error codes that indicate the job legitimately
# does not exist; everything else (throttling, IAM, network) must propagate.
_SAGEMAKER_VALIDATION_CODE = "ValidationException"


@task(
    cache_policy=NO_CACHE,
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
)
def upload_processed_data_to_s3(sagemaker_client, X_train, y_train):
    logger = get_run_logger()
    logger.info("Uploading processed data to S3...")

    start_time = time.time()
    train_input = sagemaker_train_job.upload_processed_data_to_s3(
        sagemaker_client, X_train, y_train
    )
    upload_time = time.time() - start_time

    # Calculate data size metrics
    combined_data = pd.concat([X_train, y_train], axis=1)
    data_size_mb = combined_data.memory_usage(deep=True).sum() / (1024 * 1024)

    logger.info("S3 Upload Metrics:")
    logger.info(f"Upload time: {upload_time:.2f} seconds")
    logger.info(f"Data size: {data_size_mb:.2f} MB")
    upload_speed = data_size_mb / upload_time if upload_time > 0.001 else float("inf")
    logger.info(f"Upload speed: {upload_speed:.2f} MB/s")
    logger.info(
        f"S3 location: {train_input.config['DataSource']['S3DataSource']['S3Uri']}"
    )

    return train_input


@task(cache_policy=NO_CACHE)
def train_model(
    sagemaker_client,
    train_input,
    X_train_shape,
    artifacts_s3_uris=None,
    experiment_name="box-office-predictions",
):
    logger = get_run_logger()
    logger.info("Training XGBoost model with comprehensive metrics logging...")

    # Log pre-training metrics
    logger.info("Pre-Training Metrics:")
    logger.info(f"Training samples: {X_train_shape[0]:,}")
    logger.info(f"Feature count: {X_train_shape[1]}")
    logger.info(f"Instance type: {config.sagemaker.instance_type}")
    logger.info(f"Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    training_start = time.time()

    # XGBoost hyperparameters using framework mode
    xgboost_hyperparameters = {
        "n_estimators": config.model.hyperparameters.n_estimators,
        "learning_rate": config.model.hyperparameters.learning_rate,
        "max_depth": config.model.hyperparameters.max_depth,
        "min_child_weight": config.model.hyperparameters.min_child_weight,
        "subsample": config.model.hyperparameters.subsample,
        "colsample_bytree": config.model.hyperparameters.colsample_bytree,
        "reg_alpha": config.model.hyperparameters.reg_alpha,
        "reg_lambda": config.model.hyperparameters.reg_lambda,
        "early_stopping_rounds": config.model.hyperparameters.early_stopping_rounds,
    }

    # Custom script parameters for the training script
    script_parameters = {
        "cv_folds": config.model.cross_validation.cv_folds,
        "start_eval_year": config.model.cross_validation.start_eval_year,
        "end_year": config.model.cross_validation.end_year,
    }

    # Add preprocessing artifacts S3 URIs
    if artifacts_s3_uris:
        script_parameters.update(
            {
                "processor_s3_uri": artifacts_s3_uris["processor_s3_uri"],
                "scaler_s3_uri": artifacts_s3_uris["scaler_s3_uri"],
            }
        )
        logger.info("Added preprocessing artifacts S3 URIs to script parameters")
    else:
        logger.warning(
            "No preprocessing artifacts S3 URIs provided - training script will create default artifacts"
        )

    # Combine for the training function call
    hyperparameters = {**xgboost_hyperparameters, **script_parameters}

    logger.info(f"XGBoost hyperparameters: {len(xgboost_hyperparameters)} parameters")
    logger.info(f"Custom script parameters: {len(script_parameters)} parameters")
    logger.info(f"Total parameters: {len(hyperparameters)} parameters")

    # Build SageMaker Experiment configuration to attach the training job directly
    experiment_config = {"ExperimentName": experiment_name}

    # Launch training job
    estimator = sagemaker_train_job.train_xgboost_model_with_timeseries_cv_framework(
        sagemaker_client,
        train_input,
        X_train_shape,
        hyperparameters,
        experiment_config=experiment_config,
    )

    training_duration = time.time() - training_start

    # Log post-training metrics
    logger.info("Post-Training Metrics:")
    logger.info(
        f"Total training time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)"
    )
    logger.info(f"Job name: {estimator.latest_training_job.name}")
    logger.info(f"Model artifacts: {estimator.model_data}")
    logger.info(
        f"Estimated cost: ~${(training_duration/3600) * 0.269:.4f} ({config.sagemaker.instance_type})"
    )

    return estimator


@task(
    cache_policy=NO_CACHE,
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=5),
)
def download_and_analyze_results(estimator, sagemaker_client):
    """Download and analyze training results with comprehensive metrics logging."""
    from box_office.sagemaker.training_output import build_training_metrics

    logger = get_run_logger()
    logger.info("Downloading and analyzing training results with detailed metrics...")

    job_name = estimator.latest_training_job.name
    bucket = sagemaker_client.s3_bucket
    output_prefix = f"{sagemaker_client.s3_prefix}/output"

    logger.info("Training Job Analysis:")
    logger.info(f"Job name: {job_name}")
    logger.info(f"Output path: s3://{bucket}/{output_prefix}")

    model_data_url = estimator.model_data if hasattr(estimator, "model_data") else None
    performance_metrics = build_training_metrics(
        job_name=job_name,
        region=sagemaker_client.region,
        bucket=bucket,
        output_prefix=output_prefix,
        model_data_url=model_data_url,
    )

    if "cv_results" in performance_metrics:
        cv = performance_metrics["cv_results"]
        logger.info("Cross-Validation Results:")
        logger.info(f"Mean CV MAE: {safe_format(cv.get('mean_cv_mae', 'N/A'), '.4f')}")
        if "cv_scores" in cv:
            for i, score in enumerate(cv["cv_scores"]):
                logger.info(f"Fold {i+1}: {score:.4f}")

    if "oof_results" in performance_metrics:
        oof = performance_metrics["oof_results"]
        logger.info("Out-of-Fold Evaluation:")
        logger.info(f"OOF R²: {safe_format(oof.get('oof_r2', 'N/A'), '.4f')}")

    if performance_metrics.get("training_duration_seconds"):
        duration = performance_metrics["training_duration_seconds"]
        logger.info(f"Duration: {duration:.0f} seconds ({duration/60:.2f} minutes)")

    logger.info("Training metrics extraction finished (see job logs and S3 artifacts).")
    return performance_metrics


@task(cache_policy=NO_CACHE)
def register_model_in_registry(
    job_name,
    duration,
    model_data_url,
    aws_registry,
    performance_metrics=None,
    environment="dev",
):
    """Register the trained model in AWS Model Registry with comprehensive metrics."""
    logger = get_run_logger()
    logger.info("Registering model in AWS SageMaker Model Registry...")

    try:
        if performance_metrics is None:
            performance_metrics = {}

        logger.info(f"Training Job: {job_name}")
        logger.info(f"Model Artifacts: {model_data_url}")
        logger.info(f"Training Duration: {duration / 60:.2f} minutes")

        # Get the model group name from the registry helper using static method
        group_name = AWSModelRegistry.get_model_group_name(environment=environment)
        logger.info(f"Using model group name: {group_name}")

        # Create model package group (if not exists)
        group_result = aws_registry.create_model_package_group(group_name)
        if group_result.get("status") == "error":
            raise ModelRegistryRegistrationError(
                f"Failed to create model package group: {group_result.get('error')}"
            )

        logger.info(f"Model package group: {group_name}")
        logger.info(f"Total metrics for registration: {len(performance_metrics)}")

        # Create metadata for model registry
        metadata = {
            "training_job_name": job_name,
            "framework_version": config.sagemaker.framework_version,
            "instance_type": config.sagemaker.instance_type,
            "training_duration_minutes": round(duration / 60, 4),
        }

        # Register the model package
        aws_result = aws_registry.register_model_package(
            model_package_group_name=group_name,
            model_data_url=model_data_url,
            framework="XGBOOST",
            framework_version=config.sagemaker.framework_version,
            model_approval_status="PendingManualApproval",
            metrics=performance_metrics,
            metadata=metadata,
            training_job_name=job_name,
        )

        if aws_result.get("status") == "success":
            logger.info("Successfully registered model in AWS Model Registry")
            logger.info(f"- Model ARN: {aws_result.get('model_package_arn')}")
            return {"aws_result": aws_result, "status": "success"}
        else:
            error_msg = aws_result.get("error", "Unknown error")
            logger.error(f"AWS registration failed: {error_msg}")
            raise ModelRegistryRegistrationError(
                f"AWS Model Registry registration failed: {error_msg}"
            )

    except (ModelRegistryRegistrationError, ClientError, OSError, ValueError) as e:
        logger.error("Model registration failed in AWS Model Registry", exc_info=True)
        return {"aws_result": {"error": str(e)}, "status": "failed"}


@task(cache_policy=NO_CACHE)
def upload_preprocessing_artifacts_to_s3(sagemaker_client, processor_path, scaler_path):
    """Upload fitted preprocessing artifacts to S3 for SageMaker training job."""
    logger = get_run_logger()
    logger.info("Uploading preprocessing artifacts to S3...")

    try:
        # Upload processor
        processor_s3_uri = sagemaker_client.sagemaker_session.upload_data(
            path=processor_path,
            bucket=sagemaker_client.s3_bucket,
            key_prefix=f"{sagemaker_client.s3_prefix}/artifacts",
        )

        # Upload scaler
        scaler_s3_uri = sagemaker_client.sagemaker_session.upload_data(
            path=scaler_path,
            bucket=sagemaker_client.s3_bucket,
            key_prefix=f"{sagemaker_client.s3_prefix}/artifacts",
        )

        logger.info("Preprocessing artifacts uploaded successfully:")
        logger.info(f"Processor: {processor_s3_uri}")
        logger.info(f"Scaler: {scaler_s3_uri}")

        return {"processor_s3_uri": processor_s3_uri, "scaler_s3_uri": scaler_s3_uri}

    except Exception as e:
        logger.error(f"Failed to upload preprocessing artifacts: {e}")
        raise


@task
def validate_model_for_promotion(model_package_arn, min_r2_score=None):
    """
    Validate if a model meets the criteria for production promotion using AWS Model Registry.
    """
    logger = get_run_logger()
    logger.info("Validating model package for production promotion...")
    logger.info(f"Model Package ARN: {model_package_arn}")

    start_time = time.time()

    # Use configuration default if not provided
    if min_r2_score is None:
        min_r2_score = config.model.promotion_threshold

    try:
        # Get AWS region configuration
        aws_region = config.aws.region

        # Get model package details
        sagemaker_client = boto3.client(
            "sagemaker", region_name=aws_region, config=BOTO3_CONFIG
        )

        try:
            model_package = sagemaker_client.describe_model_package(
                ModelPackageName=model_package_arn
            )
        except Exception as e:
            return {
                "promote": False,
                "reason": f"Model package not found: {str(e)}",
                "validation_time_seconds": time.time() - start_time,
            }

        logger.info(
            f"Model Package Status: {model_package.get('ModelPackageStatus', 'Unknown')}"
        )
        logger.info(f"Created: {model_package.get('CreationTime', 'Unknown')}")

        # Check if auto-approval is enabled
        auto_approve = getattr(config.model, "auto_approve_models", False)

        if auto_approve:
            logger.info("Auto-approving model (bypass metric validation)")
            logger.info("Auto-approval enabled in configuration")
            return {
                "promote": True,
                "validation_details": {
                    "auto_approved": True,
                    "model_package_arn": model_package_arn,
                    "model_package_status": model_package.get(
                        "ModelPackageStatus", "Unknown"
                    ),
                    "reason": "Auto-approval enabled - bypassing metric validation",
                },
                "validation_time_seconds": time.time() - start_time,
            }

        # Perform actual validation using OOF R2 score from model package metrics
        # Metrics are stored in CustomerMetadataProperties (not ModelMetrics)
        customer_metadata = model_package.get("CustomerMetadataProperties", {})
        oof_r2 = None

        # Try to extract R2 from customer metadata (stored as string)
        if "oof_r2" in customer_metadata:
            try:
                oof_r2 = float(customer_metadata["oof_r2"])
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse oof_r2 from metadata: {customer_metadata.get('oof_r2')}"
                )

        # Fallback: try ModelMetrics if CustomerMetadataProperties doesn't have it
        if oof_r2 is None:
            metrics = model_package.get("ModelMetrics", {}).get("ModelQuality", {})
            if "Value" in metrics:
                try:
                    metrics_dict = (
                        json.loads(metrics["Value"])
                        if isinstance(metrics["Value"], str)
                        else metrics["Value"]
                    )
                    oof_r2 = metrics_dict.get("oof_r2")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

        if oof_r2 is None:
            logger.warning("Could not extract OOF R² from model package metrics")
            logger.info(
                "Consider enabling auto_approve_models in config for development"
            )
            return {
                "promote": False,
                "validation_details": {
                    "auto_approved": False,
                    "model_package_arn": model_package_arn,
                    "reason": "Could not extract R² score from model metrics",
                    "min_required": min_r2_score,
                },
                "validation_time_seconds": time.time() - start_time,
            }

        # Validate R2 score
        meets_threshold = oof_r2 >= min_r2_score
        logger.info(f"OOF R² Score: {oof_r2:.4f}")
        logger.info(f"Threshold: {min_r2_score:.4f}")

        if meets_threshold:
            logger.info("Model meets promotion criteria")
        else:
            logger.info("Model does not meet promotion criteria")

        return {
            "promote": meets_threshold,
            "validation_details": {
                "auto_approved": False,
                "r2_score": oof_r2,
                "min_required": min_r2_score,
                "model_package_arn": model_package_arn,
                "model_package_status": model_package.get(
                    "ModelPackageStatus", "Unknown"
                ),
            },
            "validation_time_seconds": time.time() - start_time,
        }

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return {
            "promote": False,
            "reason": f"Validation error: {str(e)}",
            "validation_time_seconds": time.time() - start_time,
        }


@task
def promote_model_in_aws_registry(model_package_arn, approval_description=None):
    """
    Promote a model package to Approved status in AWS SageMaker Model Registry.
    """
    logger = get_run_logger()
    logger.info("Promoting model package to Approved status...")
    logger.info(f"Model Package ARN: {model_package_arn}")

    start_time = time.time()

    try:
        # Get AWS region configuration
        aws_region = config.aws.region

        # Initialize AWS Model Registry
        aws_registry = AWSModelRegistry(region_name=aws_region)

        # Update approval status
        result = aws_registry.update_model_approval_status(
            model_package_arn=model_package_arn,
            approval_status="Approved",
            approval_description=approval_description
            or f"Automatically approved based on performance criteria - {datetime.now(timezone.utc).isoformat()}",
        )

        promotion_time = time.time() - start_time

        if result["status"] == "success":
            logger.info("Model package approved successfully!")
            logger.info(f"ARN: {model_package_arn}")
            logger.info(f"Promotion Time: {promotion_time:.2f} seconds")

            return {
                "status": "success",
                "model_package_arn": model_package_arn,
                "approval_status": "Approved",
                "promotion_time_seconds": promotion_time,
            }
        else:
            logger.error(f"Model package promotion failed: {result.get('error')}")
            return {
                "status": "error",
                "model_package_arn": model_package_arn,
                "error": result.get("error"),
                "promotion_time_seconds": promotion_time,
            }

    except Exception as e:
        promotion_time = time.time() - start_time
        logger.error(f"AWS Model Registry promotion failed: {e}")
        return {
            "status": "error",
            "model_package_arn": model_package_arn,
            "error": str(e),
            "promotion_time_seconds": promotion_time,
        }
