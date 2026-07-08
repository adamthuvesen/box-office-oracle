import logging
import os
from datetime import datetime

from prefect import flow, get_run_logger

from box_office.config import config
from box_office.orchestration.phases.data_phase import run_data_phase
from box_office.orchestration.phases.registry_phase import run_registry_phase
from box_office.orchestration.phases.train_phase import run_train_phase
from box_office.orchestration.tasks.metrics_tasks import (
    log_pipeline_completion_metrics,
    log_pipeline_start_metrics,
    save_metrics_to_json,
)
from box_office.utils.env_setup import configure_environment


def get_logger():
    """Get appropriate logger - always use Prefect logger when available."""
    try:
        return get_run_logger()
    except (RuntimeError, ImportError):
        logger = logging.getLogger("ml_pipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


def run_ml_pipeline_logic(
    environment: str = "dev",
    experiment_name: str = "box-office-predictions",
    logger=None,
):
    """
    Core ML pipeline: data phase, SageMaker training (in-memory handoff), registry.
    """
    if logger is None:
        logger = get_logger()

    logger.info(
        "Running ML Pipeline with environment=%r experiment=%r",
        environment,
        experiment_name,
    )

    pipeline_start_metrics = log_pipeline_start_metrics()

    data = run_data_phase(logger)
    train = run_train_phase(data, logger)
    registry = run_registry_phase(
        train.training_metrics, environment=environment, logger=logger
    )

    final_metrics = log_pipeline_completion_metrics(
        pipeline_start_metrics,
        data.data_metrics,
        data.feature_metrics,
        train.training_metrics,
        registry.model_registry_metrics,
    )

    metrics_dir = os.path.join(config.model.artifacts_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(
        metrics_dir, f"pipeline_execution_metrics_{timestamp}.json"
    )
    save_metrics_to_json(final_metrics, metrics_path)
    logger.info("Metrics saved to: %s", metrics_path)

    estimator = train.estimator
    return {
        "pipeline_status": "completed",
        "target_column": data.target_column,
        "feature_count": len(data.feature_names),
        "training_samples": data.X_train_shape[0],
        "validation_samples": data.X_val_shape[0],
        "processor_path": data.processor_path,
        "scaler_path": data.scaler_path,
        "save_results": data.save_results,
        "validation_results": data.validation_results,
        "feature_names": data.feature_names,
        "model_registry": {
            "registration_result": registry.registration_result,
            "promotion_result": registry.promotion_result,
            "aws_promotion_result": registry.aws_promotion_result,
        },
        "estimator": (
            estimator.latest_training_job.name
            if hasattr(estimator, "latest_training_job")
            else None
        ),
    }


@flow(
    name="Box Office ML Pipeline",
    description="End-to-end ML pipeline for box office prediction using dbt, Snowflake, and SageMaker",
)
def ml_pipeline(
    environment: str = "dev", experiment_name: str = "box-office-predictions"
):
    """Complete ML pipeline from data loading to preprocessing, training and saving model artifacts."""
    configure_environment()
    logger = get_logger()
    logger.info("Starting ML Pipeline...")
    result = run_ml_pipeline_logic(environment, experiment_name, logger)

    logger.info("ML Pipeline completed!")
    if result and isinstance(result, dict):
        model_registry_result = result.get("model_registry")
        if model_registry_result and isinstance(model_registry_result, dict):
            promotion_result = model_registry_result.get("promotion_result")
            if promotion_result:
                logger.info(
                    "- Model evaluation: %s",
                    "Approved" if promotion_result.get("promote") else "Not approved",
                )
            registration_result = model_registry_result.get("registration_result", {})
            logger.info(
                "- Model registration: %s",
                (
                    "Success"
                    if registration_result.get("status") == "success"
                    else "Failed"
                ),
            )

    return result


def main():
    """Main entry point for console script execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Box Office ML Pipeline")
    parser.add_argument(
        "--environment",
        default="dev",
        choices=["dev", "staging", "prod"],
        help="Environment to run the pipeline in",
    )
    parser.add_argument(
        "--experiment-name",
        default="box-office-predictions",
        help="Name of the SageMaker experiment",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the pipeline in serve mode (Prefect server)",
    )

    args = parser.parse_args()

    if args.serve:
        ml_pipeline.serve(
            name="box-office-ml-pipeline",
            parameters={
                "environment": args.environment,
                "experiment_name": args.experiment_name,
            },
        )
    else:
        result = ml_pipeline(
            environment=args.environment, experiment_name=args.experiment_name
        )
        get_logger().info("Pipeline completed with result: %s", result)


if __name__ == "__main__":
    main()
