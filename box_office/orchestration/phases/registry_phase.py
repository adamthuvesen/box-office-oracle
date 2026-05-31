"""Model registry registration and promotion phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from box_office.config import config
from box_office.ml.model_registry.aws_model_registry import AWSModelRegistry
from box_office.orchestration.tasks.training_tasks import (
    promote_model_in_aws_registry,
    register_model_in_registry,
    validate_model_for_promotion,
)


@dataclass
class RegistryPhaseResult:
    registration_result: Dict[str, Any]
    promotion_result: Optional[Dict[str, Any]]
    aws_promotion_result: Optional[Dict[str, Any]]

    @property
    def model_registry_metrics(self) -> Dict[str, Any]:
        return {
            "model_registration": self.registration_result,
            "model_promotion_validation": self.promotion_result,
            "aws_promotion": self.aws_promotion_result,
        }


def run_registry_phase(
    training_metrics: Dict[str, Any],
    environment: str,
    logger,
) -> RegistryPhaseResult:
    """Register model package and optionally promote to Approved."""
    aws_registry = AWSModelRegistry(config.aws.region)
    registration_result = register_model_in_registry(
        job_name=training_metrics["job_name"],
        duration=training_metrics["duration"],
        model_data_url=training_metrics["model_data_url"],
        aws_registry=aws_registry,
        performance_metrics=training_metrics,
        environment=environment,
    )

    promotion_result = None
    aws_promotion_result = None

    if registration_result.get("status") == "success":
        aws_model_package_arn = registration_result["aws_result"]["model_package_arn"]
        promotion_result = validate_model_for_promotion(
            model_package_arn=aws_model_package_arn,
            min_r2_score=config.model.promotion_threshold,
        )

        if promotion_result.get("promote"):
            logger.info("Model meets criteria - promoting in AWS Model Registry...")
            validation_details = promotion_result.get("validation_details", {})
            if validation_details.get("auto_approved"):
                approval_description = f"Auto-approved: {validation_details.get('reason', 'Bypassing metric validation')}"
            else:
                r2_score = validation_details.get("r2_score", 0.0)
                approval_description = f"Automatically approved: R² = {r2_score:.4f} (>= {config.model.promotion_threshold})"

            aws_promotion_result = promote_model_in_aws_registry(
                model_package_arn=aws_model_package_arn,
                approval_description=approval_description,
            )

            if aws_promotion_result.get("status") == "success":
                logger.info("Model promoted to Approved status in AWS Model Registry!")
            else:
                logger.warning(
                    "AWS promotion failed: %s",
                    aws_promotion_result.get("error"),
                )
        else:
            logger.info("Model does not meet promotion criteria")
    else:
        logger.warning(
            "Skipping model promotion validation due to AWS registration failure"
        )

    return RegistryPhaseResult(
        registration_result=registration_result,
        promotion_result=promotion_result,
        aws_promotion_result=aws_promotion_result,
    )
