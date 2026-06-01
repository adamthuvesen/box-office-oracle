"""AWS SageMaker Model Registry client for model package lifecycle management."""

import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from box_office.utils.aws_helpers import compute_sha256_stream, parse_s3_uri

# Validation constants
VALID_APPROVAL_STATUSES = {"Approved", "Rejected", "PendingManualApproval"}
ARN_PATTERN = re.compile(r"^arn:aws:sagemaker:[a-z0-9-]+:\d{12}:model-package/.+$")

# SageMaker error codes that indicate the resource genuinely does not exist.
# We must classify by code (and validate the message text on the generic
# ValidationException), not by string-matching `'does not exist' in str(e)` —
# AWS message wording is not a stable API.
_NOT_FOUND_CODES = frozenset({"ResourceNotFound", "ResourceNotFoundException"})
_VALIDATION_CODE = "ValidationException"


def _is_resource_not_found(exc: ClientError) -> bool:
    """True only when the error confirms the resource does not exist."""
    err = exc.response.get("Error", {}) if hasattr(exc, "response") else {}
    code = err.get("Code", "")
    if code in _NOT_FOUND_CODES:
        return True
    # SageMaker often returns ValidationException for not-found; check message.
    if code == _VALIDATION_CODE and "does not exist" in err.get("Message", "").lower():
        return True
    return False


logger = logging.getLogger(__name__)


class ModelRegistryRegistrationError(RuntimeError):
    """Creating or registering a SageMaker model package or group failed."""


def _compute_sha256_of_s3_object(s3_client, model_data_url: str) -> Tuple[str, int]:
    """Download the object at ``model_data_url`` to a temp file and stream-hash it."""
    bucket, key = parse_s3_uri(model_data_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
        tmp_path = Path(tmp.name)
    try:
        s3_client.download_file(bucket, key, str(tmp_path))
        digest = compute_sha256_stream(tmp_path)
        size = tmp_path.stat().st_size
        return digest, size
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class AWSModelRegistry:
    """AWS SageMaker Model Registry client for managing model packages."""

    def __init__(self, region_name: str | None = None):
        self.region_name = region_name
        self.sagemaker_client = boto3.client("sagemaker", region_name=region_name)
        self.s3_client = boto3.client("s3", region_name=region_name)

    @staticmethod
    def get_model_group_name(
        environment: str | None = None, project_name: str | None = None
    ) -> str:
        import os

        from box_office.shared.names import model_registry_group_name as _group_name

        if environment is None:
            environment = os.environ.get("ENVIRONMENT", "dev")
        if project_name is None:
            project_name = os.environ.get("PROJECT_NAME", "box-office")

        return _group_name(project_name=project_name, environment=environment)

    def create_model_package_group(
        self, group_name: str, description: str | None = None
    ) -> Dict[str, Any]:
        """Create the package group, or return ``status="exists"`` if it already does.

        Stays idempotent on purpose so callers can use it as "ensure exists",
        but any non-not-found ClientError (AccessDenied, throttling) propagates
        instead of being downgraded to a status dict.
        """
        # ResourceNotFound is the legitimate "create it" path; any other
        # ClientError (auth, throttling) means infra is broken and propagates.
        try:
            response = self.sagemaker_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            logger.info(f"Model package group '{group_name}' already exists")
            return {
                "status": "exists",
                "group_name": group_name,
                "arn": response["ModelPackageGroupArn"],
            }
        except ClientError as e:
            if not _is_resource_not_found(e):
                raise

        create_response = self.sagemaker_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=description
            or f"Model package group for {group_name}",
        )

        logger.info(f"Created model package group '{group_name}'")
        return {
            "status": "created",
            "group_name": group_name,
            "arn": create_response["ModelPackageGroupArn"],
        }

    def register_model_package(
        self,
        model_package_group_name: str,
        model_data_url: str,
        framework: str = "XGBOOST",
        framework_version: str = "1.7-1",
        model_approval_status: str = "PendingManualApproval",
        inference_specification: Dict[str, Any] | None = None,
        metrics: Dict[str, float] | None = None,
        metadata: Dict[str, Any] | None = None,
        training_job_name: str | None = None,
    ) -> Dict[str, Any]:
        """
        Register a model package in the AWS Model Registry.

        Args:
            model_package_group_name: Name of the model package group
            model_data_url: S3 URL to the model artifacts
            framework: ML framework (XGBOOST, SKLEARN, etc.)
            framework_version: Framework version
            model_approval_status: Approval status (PendingManualApproval, Approved, Rejected)
            inference_specification: Inference container specification
            metrics: Model performance metrics
            metadata: Additional metadata

        Returns:
            Dictionary with registration result
        """
        if inference_specification is None:
            inference_specification = self._get_default_inference_spec(
                framework, framework_version
            )

        if (
            "Containers" in inference_specification
            and len(inference_specification["Containers"]) > 0
        ):
            inference_specification["Containers"][0]["ModelDataUrl"] = model_data_url

        model_package_request = {
            "ModelPackageGroupName": model_package_group_name,
            "ModelApprovalStatus": model_approval_status,
            "InferenceSpecification": inference_specification,
            "ModelPackageDescription": f"Box office prediction model - {datetime.now(timezone.utc).isoformat()}",
        }

        # Optionally link back to the training job so the Studio UI shows the connection.
        # The describe-job lookup is best-effort: a missing job is benign (we still
        # register), but auth/throttling errors propagate so the operator sees them.
        if training_job_name:
            try:
                tj_resp = self.sagemaker_client.describe_training_job(
                    TrainingJobName=training_job_name
                )
                training_job_arn = tj_resp.get("TrainingJobArn")
                if training_job_arn:
                    if "CustomerMetadataProperties" not in model_package_request:
                        model_package_request["CustomerMetadataProperties"] = {}
                    model_package_request["CustomerMetadataProperties"][
                        "training_job_arn"
                    ] = training_job_arn
            except ClientError as e:
                if _is_resource_not_found(e):
                    logger.warning(
                        f"Training job '{training_job_name}' not found; "
                        f"registering package without training_job_arn"
                    )
                else:
                    raise

        # AWS ModelMetrics has strict format requirements, so all custom data
        # (metrics + free-form metadata) goes into CustomerMetadataProperties,
        # which only accepts string values up to 256 chars per field.
        if metadata:
            model_package_request["CustomerMetadataProperties"] = {
                key: str(value) for key, value in metadata.items()
            }

        if metrics:
            if "CustomerMetadataProperties" not in model_package_request:
                model_package_request["CustomerMetadataProperties"] = {}

            # Allowlist essential metrics; long URIs and hyperparameters would
            # exceed the 256-char-per-field cap.
            essential_metrics = [
                "oof_r2",
                "oof_mae",
                "oof_rmsle",
                "oof_num_samples",
                "cv_mean_mae",
                "cv_std_mae",
                "cv_mean_rmsle",
                "cv_std_rmsle",
                "cv_mean_best_iteration",
                "training_duration_minutes",
            ]

            for metric_name, metric_value in metrics.items():
                if metric_name in essential_metrics and isinstance(
                    metric_value, (int, float)
                ):
                    short_value = (
                        f"{metric_value:.6f}"
                        if isinstance(metric_value, float)
                        else str(metric_value)
                    )
                    if len(short_value) <= 250:
                        model_package_request["CustomerMetadataProperties"][
                            metric_name
                        ] = short_value

            framework_info = f"{framework}_{framework_version}"
            if len(framework_info) <= 250:
                model_package_request["CustomerMetadataProperties"][
                    "framework"
                ] = framework_info

        # Compute SHA256 + size_bytes of the uploaded model artifact and store
        # them in CustomerMetadataProperties. The inference loader verifies
        # these BEFORE unpickling — closes the bucket-write -> RCE surface.
        sha256, size_bytes = _compute_sha256_of_s3_object(
            self.s3_client, model_data_url
        )
        from box_office.ml.feature_schema import (
            CURRENT_FEATURE_SCHEMA_VERSION,
            SCHEMA_VERSION_METADATA_KEY,
        )

        model_package_request.setdefault("CustomerMetadataProperties", {})
        model_package_request["CustomerMetadataProperties"]["sha256"] = sha256
        model_package_request["CustomerMetadataProperties"]["size_bytes"] = str(
            size_bytes
        )
        model_package_request["CustomerMetadataProperties"][
            SCHEMA_VERSION_METADATA_KEY
        ] = CURRENT_FEATURE_SCHEMA_VERSION
        logger.info(
            "Model artifact manifest: sha256=%s size_bytes=%d", sha256[:16], size_bytes
        )

        # Create the model package. Hard failures (auth, throttling, validation
        # other than already-exists) MUST raise — callers used to read
        # `result['model_package_arn']` and KeyError with no context when this
        # silently returned `{'status': 'error'}`.
        try:
            response = self.sagemaker_client.create_model_package(
                **model_package_request
            )
        except ClientError as e:
            logger.error(f"Failed to register model package: {e}", exc_info=True)
            raise

        model_package_arn = response["ModelPackageArn"]
        logger.info(f"Successfully registered model package: {model_package_arn}")

        return {
            "status": "success",
            "model_package_arn": model_package_arn,
            "approval_status": model_approval_status,
            "group_name": model_package_group_name,
        }

    def update_model_approval_status(
        self,
        model_package_arn: str,
        approval_status: str,
        approval_description: str | None = None,
    ) -> Dict[str, Any]:
        """
        Update the approval status of a model package.

        Args:
            model_package_arn: ARN of the model package to update
            approval_status: New status (Approved, Rejected, or PendingManualApproval)
            approval_description: Optional description for the status change

        Returns:
            Dict with status and details

        Raises:
            ValueError: If ARN format is invalid or approval_status is not valid
        """
        # Validate ARN format
        if not ARN_PATTERN.match(model_package_arn):
            raise ValueError(
                f"Invalid model package ARN format: {model_package_arn}. "
                f"Expected format: arn:aws:sagemaker:REGION:ACCOUNT:model-package/NAME"
            )

        # Validate approval status
        if approval_status not in VALID_APPROVAL_STATUSES:
            raise ValueError(
                f"Invalid approval status: {approval_status}. "
                f"Valid values: {VALID_APPROVAL_STATUSES}"
            )

        try:
            update_request = {
                "ModelPackageArn": model_package_arn,
                "ModelApprovalStatus": approval_status,
            }

            if approval_description:
                update_request["ApprovalDescription"] = approval_description

            self.sagemaker_client.update_model_package(**update_request)

            logger.info(f"Updated model package approval status to: {approval_status}")
            return {
                "status": "success",
                "model_package_arn": model_package_arn,
                "approval_status": approval_status,
            }

        except Exception as e:
            logger.error(f"Failed to update model approval status: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    def list_model_packages(
        self,
        model_package_group_name: str | None = None,
        approval_status: str | None = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:

        try:
            list_request = {
                "MaxResults": max_results,
                "SortBy": "CreationTime",
                "SortOrder": "Descending",
            }

            if model_package_group_name:
                list_request["ModelPackageGroupName"] = model_package_group_name

            if approval_status:
                list_request["ModelApprovalStatus"] = approval_status

            response = self.sagemaker_client.list_model_packages(**list_request)

            return response.get("ModelPackageSummaryList", [])

        except Exception as e:
            logger.error(f"Failed to list model packages: {e}", exc_info=True)
            return []

    def get_latest_approved_model(
        self, model_package_group_name: str
    ) -> Optional[Dict[str, Any]]:

        approved_models = self.list_model_packages(
            model_package_group_name=model_package_group_name,
            approval_status="Approved",
            max_results=1,
        )

        return approved_models[0] if approved_models else None

    def _get_default_inference_spec(
        self, framework: str, framework_version: str
    ) -> Dict[str, Any]:
        """Build a default inference spec via SageMaker SDK image-URI lookup, with an ECR-URL fallback."""
        try:
            from sagemaker import image_uris

            framework_mapping = {"XGBOOST": "xgboost", "SKLEARN": "sklearn"}
            sagemaker_framework = framework_mapping.get(framework.upper(), "xgboost")

            image_uri = image_uris.retrieve(
                framework=sagemaker_framework,
                region=self.region_name,
                version=framework_version,
            )
        except Exception as e:
            logger.warning(
                f"Could not retrieve image URI dynamically: {e}. Using fallback."
            )
            account_id = self._get_account_id()
            framework_images = {
                "XGBOOST": f"{account_id}.dkr.ecr.{self.region_name}.amazonaws.com/sagemaker-xgboost:{framework_version}",
                "SKLEARN": f"{account_id}.dkr.ecr.{self.region_name}.amazonaws.com/sagemaker-scikit-learn:{framework_version}",
            }
            image_uri = framework_images.get(
                framework.upper(), framework_images["XGBOOST"]
            )

        return {
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": "",  # Filled in by the caller.
                    "Framework": framework.upper(),
                    "FrameworkVersion": framework_version,
                }
            ],
            "SupportedContentTypes": ["text/csv", "application/json"],
            "SupportedResponseMIMETypes": ["text/csv", "application/json"],
        }

    def get_training_job_metrics(self, training_job_name: str) -> Dict[str, float]:
        try:
            response = self.sagemaker_client.describe_training_job(
                TrainingJobName=training_job_name
            )

            metrics = {}
            if "FinalMetricDataList" in response:
                for metric in response["FinalMetricDataList"]:
                    metric_name = metric["MetricName"]
                    metric_value = metric["Value"]
                    metrics[metric_name] = metric_value
                    logger.info(
                        f"Retrieved training metric: {metric_name} = {metric_value}"
                    )
            else:
                logger.info("No FinalMetricDataList found in training job description")

            return metrics

        except Exception as e:
            logger.warning(f"Could not retrieve training job metrics: {e}")
            return {}

    def _get_account_id(self) -> str:
        """Get AWS account ID via STS.

        Raises ``ClientError`` on STS failure. The previous behavior of
        returning the literal string ``"unknown"`` produced invalid ECR URLs
        (``unknown.dkr.ecr.<region>.amazonaws.com/...``) that surfaced as
        cryptic image-pull errors much later.
        """
        sts = boto3.client("sts", region_name=self.region_name)
        return sts.get_caller_identity()["Account"]
