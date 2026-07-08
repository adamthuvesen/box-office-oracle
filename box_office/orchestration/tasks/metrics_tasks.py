import json
import time
from datetime import datetime
from typing import Any

from prefect import get_run_logger, task

from box_office.ml.metrics_models import (
    DataProcessingMetrics,
    FeatureEngineeringMetrics,
    PipelineExecutionSummary,
    PipelineStartMetrics,
)
from box_office.utils.format_helpers import safe_format


@task
def log_pipeline_start_metrics():
    """Log comprehensive pipeline start metrics."""
    logger = get_run_logger()

    start_time = datetime.now()
    logger.info("Pipeline started at %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    return PipelineStartMetrics(
        pipeline_start_time=start_time.isoformat(),
        pipeline_id=f"box-office-ml-{int(time.time())}",
    ).model_dump()


@task
def log_data_processing_metrics(
    X_train_shape: tuple, X_val_shape: tuple, feature_count: int, target_column: str
):
    """Log comprehensive data processing metrics."""
    logger = get_run_logger()

    logger.info("DATA PROCESSING METRICS SUMMARY")
    logger.info(f"Target variable: {target_column}")
    logger.info(f"Training set: {X_train_shape[0]:,} samples")
    logger.info(f"Validation set: {X_val_shape[0]:,} samples")
    logger.info(f"Features engineered: {feature_count}")
    total_samples = X_train_shape[0] + X_val_shape[0]
    if total_samples > 0:
        train_pct = (X_train_shape[0] / total_samples) * 100
        val_pct = (X_val_shape[0] / total_samples) * 100
        logger.info(f"Train/Val split: {train_pct:.1f}% / {val_pct:.1f}%")
    else:
        logger.warning("Empty dataset: no samples in train or validation sets")

    return DataProcessingMetrics(
        training_samples=X_train_shape[0],
        validation_samples=X_val_shape[0],
        feature_count=feature_count,
        target_column=target_column,
        train_val_ratio=(
            X_train_shape[0] / X_val_shape[0] if X_val_shape[0] > 0 else float("inf")
        ),
    ).model_dump()


@task
def log_feature_engineering_metrics(processor, feature_names: list[str]):
    """Log detailed feature engineering metrics."""
    logger = get_run_logger()

    logger.info("FEATURE ENGINEERING METRICS")
    logger.info(f"Processor type: {type(processor).__name__}")
    logger.info(f"Total features created: {len(feature_names)}")

    # Categorize features by type
    feature_categories = {
        "genre": [f for f in feature_names if "genre" in f.lower()],
        "temporal": [
            f
            for f in feature_names
            if any(
                t in f.lower() for t in ["year", "month", "season", "holiday", "covid"]
            )
        ],
        "financial": [
            f
            for f in feature_names
            if any(t in f.lower() for t in ["budget", "gross", "revenue", "cost"])
        ],
        "categorical": [
            f
            for f in feature_names
            if any(t in f.lower() for t in ["mpaa", "director", "company", "actor"])
        ],
        "interaction": [
            f for f in feature_names if "_x_" in f.lower() or "interaction" in f.lower()
        ],
    }

    for category, features in feature_categories.items():
        if features:
            logger.info(f"{category.title()} features: {len(features)}")
            if len(features) <= 5:
                logger.info(f"{', '.join(features)}")
            else:
                logger.info(
                    f"{', '.join(features[:3])} ... (+{len(features) - 3} more)"
                )

    return FeatureEngineeringMetrics(
        total_features=len(feature_names),
        feature_categories={k: len(v) for k, v in feature_categories.items()},
        feature_names=feature_names,
    ).model_dump()


@task()
def log_model_training_summary(training_metrics: dict[str, Any]):
    """Log comprehensive model training summary."""
    logger = get_run_logger()

    logger.info("MODEL TRAINING SUMMARY")

    # Training configuration
    if "hyperparameters" in training_metrics:
        logger.info("Hyperparameters:")
        for param, value in training_metrics["hyperparameters"].items():
            logger.info(f"{param}: {value}")

    # Performance metrics
    if "cv_results" in training_metrics:
        cv = training_metrics["cv_results"]
        logger.info("Cross-Validation Performance:")
        logger.info(f"Mean MAE: {safe_format(cv.get('mean_cv_mae', 'N/A'), '.4f')}")
        logger.info(f"Std MAE: ±{safe_format(cv.get('std_cv_mae', 'N/A'), '.4f')}")
        logger.info(
            f"Best iteration: {safe_format(cv.get('mean_best_iteration', 'N/A'), '.0f')}"
        )
        logger.info(f"CV folds: {len(cv.get('cv_scores', []))}")

    # Resource usage
    if "training_time" in training_metrics:
        duration = training_metrics["training_time"]
        logger.info(
            f"Training duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)"
        )

    if "estimated_cost" in training_metrics:
        logger.info(f"Estimated cost: ${training_metrics['estimated_cost']:.4f}")

    return training_metrics


@task
def log_pipeline_completion_metrics(
    pipeline_start_metrics: dict[str, Any],
    data_metrics: dict[str, Any],
    feature_metrics: dict[str, Any],
    training_metrics: dict[str, Any],
    model_registry_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log comprehensive pipeline completion summary."""
    logger = get_run_logger()

    end_time = datetime.now()
    start_time = datetime.fromisoformat(pipeline_start_metrics["pipeline_start_time"])
    total_duration = (end_time - start_time).total_seconds()

    logger.info("ML PIPELINE EXECUTION COMPLETED")
    logger.info("FINAL PIPELINE SUMMARY:")
    logger.info(
        f"Total duration: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)"
    )
    logger.info(f"Completion time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Pipeline ID: {pipeline_start_metrics['pipeline_id']}")

    _log_completion_data_summary(logger, data_metrics, feature_metrics)
    _log_completion_training_summary(logger, training_metrics)
    _log_model_registry_summary(logger, model_registry_metrics)

    return _build_pipeline_execution_summary(
        pipeline_start_metrics=pipeline_start_metrics,
        data_metrics=data_metrics,
        feature_metrics=feature_metrics,
        training_metrics=training_metrics,
        model_registry_metrics=model_registry_metrics,
        end_time=end_time,
        total_duration=total_duration,
    )


def _log_completion_data_summary(
    logger: Any, data_metrics: dict[str, Any], feature_metrics: dict[str, Any]
) -> None:
    logger.info("\n DATA PROCESSING SUMMARY:")
    logger.info(f"Training samples: {data_metrics['training_samples']:,}")
    logger.info(f"Validation samples: {data_metrics['validation_samples']:,}")
    logger.info(f"Features created: {feature_metrics['total_features']}")
    logger.info(f"Target: {data_metrics['target_column']}")


def _log_completion_training_summary(
    logger: Any, training_metrics: dict[str, Any]
) -> None:
    logger.info("\n MODEL PERFORMANCE SUMMARY:")
    if "cv_results" in training_metrics:
        cv = training_metrics["cv_results"]
        logger.info(
            f"Cross-validation MAE: {safe_format(cv.get('mean_cv_mae', 'N/A'), '.4f')} ±{safe_format(cv.get('std_cv_mae', 'N/A'), '.4f')}"
        )
        logger.info(
            f"Optimal iterations: {safe_format(cv.get('mean_best_iteration', 'N/A'), '.0f')}"
        )
        logger.info(f"CV folds completed: {len(cv.get('cv_scores', []))}")

    if "oof_results" in training_metrics:
        oof = training_metrics["oof_results"]
        logger.info(f"Out-of-fold R²: {safe_format(oof.get('oof_r2', 'N/A'), '.4f')}")
        logger.info(
            f"Out-of-fold MAE: ${safe_format(oof.get('oof_mae', 'N/A'), ',.0f')}"
        )

    logger.info("\n RESOURCE UTILIZATION:")
    if "training_time" in training_metrics:
        logger.info(f"Training time: {training_metrics['training_time']:.2f} seconds")
    if "estimated_cost" in training_metrics:
        logger.info(f"Estimated cost: ${training_metrics['estimated_cost']:.4f}")


def _log_model_registry_summary(
    logger: Any, model_registry_metrics: dict[str, Any] | None
) -> None:
    if model_registry_metrics:
        logger.info("\n MODEL REGISTRY SUMMARY:")
        _log_model_registration(logger, model_registry_metrics)
        _log_promotion_validation(logger, model_registry_metrics)
        _log_aws_promotion(logger, model_registry_metrics)

    _log_production_readiness(logger, model_registry_metrics)


def _log_model_registration(
    logger: Any, model_registry_metrics: dict[str, Any]
) -> None:
    if "model_registration" not in model_registry_metrics:
        return

    reg_result = model_registry_metrics["model_registration"]

    if reg_result and "aws_result" in reg_result:
        aws_result = reg_result["aws_result"]
        if aws_result and aws_result.get("status") == "success":
            arn = aws_result.get("model_package_arn", "Unknown ARN")
            logger.info("AWS Model Registry: Package registered")
            logger.info(f"Model Package ARN: ...{arn[-20:] if len(arn) > 20 else arn}")
            logger.info(
                f"Approval Status: {aws_result.get('approval_status', 'Unknown')}"
            )
        else:
            logger.info(
                f"AWS Model Registry failed: {aws_result.get('error', 'Unknown error') if aws_result else 'No result'}"
            )

    aws_status = reg_result.get("status", "unknown")
    if aws_status == "success":
        logger.info("Overall: Successfully registered in AWS Model Registry")
    else:
        logger.info("Overall: AWS Model Registry registration failed")


def _log_promotion_validation(
    logger: Any, model_registry_metrics: dict[str, Any]
) -> None:
    if "model_promotion_validation" not in model_registry_metrics:
        return

    promo_result = model_registry_metrics["model_promotion_validation"]
    if promo_result and promo_result.get("promote"):
        logger.info("Model approved for promotion to production")
    else:
        logger.info("Model does not meet promotion criteria")

    if promo_result and "validation_details" in promo_result:
        _log_validation_r2(logger, promo_result["validation_details"])


def _log_validation_r2(logger: Any, details: dict[str, Any]) -> None:
    if "r2_score" not in details:
        return
    logger.info(
        f"R² Score: {safe_format(details.get('r2_score', 'N/A'), '.4f')} (threshold: {details.get('min_required', 0.75)})"
    )


def _log_aws_promotion(logger: Any, model_registry_metrics: dict[str, Any]) -> None:
    if (
        "aws_promotion" not in model_registry_metrics
        or model_registry_metrics["aws_promotion"] is None
    ):
        return

    aws_promo_result = model_registry_metrics["aws_promotion"]
    if aws_promo_result.get("status") == "success":
        logger.info("AWS Model Package promoted to Approved status")
        logger.info(
            f"Promotion time: {aws_promo_result.get('promotion_time_seconds', 0):.2f}s"
        )
    else:
        logger.info(
            f"AWS promotion failed: {aws_promo_result.get('error', 'Unknown error')}"
        )


def _log_production_readiness(
    logger: Any, model_registry_metrics: dict[str, Any] | None
) -> None:
    if (
        model_registry_metrics
        and model_registry_metrics.get("model_registration", {}).get("status")
        == "success"
    ):
        logger.info("Model registered in AWS Model Registry")

        # Check AWS promotion status
        if (
            model_registry_metrics.get("aws_promotion") is not None
            and model_registry_metrics.get("aws_promotion", {}).get("status")
            == "success"
        ):
            logger.info("Model approved in AWS Model Registry - ready for deployment!")
        elif model_registry_metrics.get("model_promotion_validation", {}).get(
            "promote"
        ):
            logger.info("Model ready for production promotion")
            aws_registry_success = (
                model_registry_metrics.get("model_registration", {}).get("status")
                == "success"
            )
            if aws_registry_success:
                logger.info(
                    "AWS Model Package can be manually promoted to Approved status"
                )
            else:
                logger.info("Model meets promotion criteria")
        else:
            logger.info("Model requires performance improvements for production")
    else:
        logger.info("Check S3 output for detailed results")


def _build_pipeline_execution_summary(
    pipeline_start_metrics: dict[str, Any],
    data_metrics: dict[str, Any],
    feature_metrics: dict[str, Any],
    training_metrics: dict[str, Any],
    model_registry_metrics: dict[str, Any] | None,
    end_time: datetime,
    total_duration: float,
) -> dict[str, Any]:
    final_summary = PipelineExecutionSummary(
        pipeline_id=pipeline_start_metrics["pipeline_id"],
        execution_time={
            "start": pipeline_start_metrics["pipeline_start_time"],
            "end": end_time.isoformat(),
            "duration_seconds": total_duration,
            "duration_minutes": total_duration / 60,
        },
        data_summary=data_metrics,
        feature_summary=feature_metrics,
        training_summary=training_metrics,
        model_registry_summary=model_registry_metrics,
    ).model_dump()

    return final_summary


@task
def save_metrics_to_json(
    metrics_summary: dict[str, Any], output_path: str = "pipeline_metrics.json"
):
    """Save comprehensive metrics summary to JSON file."""
    logger = get_run_logger()

    try:
        with open(output_path, "w") as f:
            json.dump(metrics_summary, f, indent=2, default=str)

        logger.info(f"Metrics summary saved to: {output_path}")
        logger.info("File contains complete pipeline execution metrics")

        return output_path

    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        return None
