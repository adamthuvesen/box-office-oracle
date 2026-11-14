import logging
from typing import Any

from prefect import flow, get_run_logger

from box_office.orchestration.phases.data_phase import run_data_phase
from box_office.orchestration.phases.registry_phase import run_registry_phase
from box_office.orchestration.phases.train_phase import run_train_phase
from box_office.orchestration.run_logger import RunLogger
from box_office.sagemaker.sagemaker_train_job import require_training_image
from box_office.utils.approval import require_approval
from box_office.utils.env_setup import configure_environment


def get_logger() -> RunLogger:
    """Get appropriate logger - always use Prefect logger when available."""
    try:
        return get_run_logger()
    except RuntimeError, ImportError:
        logger = logging.getLogger("ml_pipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


def run_ml_pipeline_logic(
    environment: str = "dev",
    logger: RunLogger | None = None,
    approved: bool = False,
) -> dict[str, Any]:
    """
    Core ML pipeline: data phase, SageMaker training (in-memory handoff), registry.
    """
    if logger is None:
        logger = get_logger()

    require_approval("running the ML pipeline", approved)
    require_training_image()

    logger.info("Running ML Pipeline with environment=%r", environment)

    data = run_data_phase(logger, approved=approved)
    train = run_train_phase(data, logger, approved=approved)
    registry = run_registry_phase(
        train.training_metrics,
        environment=environment,
        logger=logger,
        data_fingerprint=data.data_fingerprint,
        approved=approved,
    )

    training_job = train.estimator.latest_training_job
    return {
        "pipeline_status": "completed",
        "target_column": data.target_column,
        "feature_count": len(data.feature_names),
        "training_samples": data.X_train_shape[0],
        "validation_samples": data.X_val_shape[0],
        "processor_path": data.processor_path,
        "scaler_path": data.scaler_path,
        "feature_names": data.feature_names,
        "model_registry": {
            "registration_result": registry.registration_result,
            "promotion_result": registry.promotion_result,
            "aws_promotion_result": registry.aws_promotion_result,
        },
        "estimator": training_job.name if training_job is not None else None,
    }


@flow(
    name="Box Office ML Pipeline",
    description="End-to-end ML pipeline for box office prediction using dbt, Snowflake, and SageMaker",
)
def ml_pipeline(environment: str = "dev", approved: bool = False) -> dict[str, Any]:
    """Complete ML pipeline from data loading to preprocessing, training and saving model artifacts."""
    configure_environment()
    logger = get_logger()
    logger.info("Starting ML Pipeline...")
    result = run_ml_pipeline_logic(environment, logger, approved=approved)

    logger.info("ML Pipeline completed!")
    model_registry_result = result["model_registry"]
    promotion_result = model_registry_result["promotion_result"]
    if promotion_result:
        logger.info(
            "- Model evaluation: %s",
            "Approved" if promotion_result.get("promote") else "Not approved",
        )
    registration_result = model_registry_result["registration_result"]
    logger.info(
        "- Model registration: %s",
        ("Success" if registration_result.get("status") == "success" else "Failed"),
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
        "--approved",
        action="store_true",
        help="Approve the paid and production-writing steps of this run",
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
            parameters={"environment": args.environment, "approved": args.approved},
        )
    else:
        result = ml_pipeline(environment=args.environment, approved=args.approved)
        get_logger().info("Pipeline completed with result: %s", result)


if __name__ == "__main__":
    main()
