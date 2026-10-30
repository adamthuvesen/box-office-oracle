"""Parse SageMaker training job output artifacts."""

from __future__ import annotations

import contextlib
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from box_office.ml.metrics_models import CrossValidationResults, OofEvaluation, TrainingMetrics
from box_office.utils.aws_helpers import BOTO3_CONFIG
from box_office.utils.safe_tarfile import extractall_data_filter

logger = logging.getLogger(__name__)

_S3_NOT_FOUND_CODES = frozenset({"NoSuchKey", "404", "NoSuchBucket"})
_SAGEMAKER_VALIDATION_CODE = "ValidationException"


def parse_training_output_tarball(tarball_path: Path) -> dict[str, Any]:
    """Extract CV and OOF metrics from a SageMaker ``output.tar.gz``."""
    metrics: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as extract_dir:
        with tarfile.open(tarball_path, "r:gz") as tar:
            extractall_data_filter(tar, extract_dir)

        cv_path = Path(extract_dir) / "cv_results.json"
        if cv_path.exists():
            cv_results = CrossValidationResults.model_validate_json(
                cv_path.read_text(encoding="utf-8")
            )
            metrics.update(
                {
                    "cv_mean_mae": cv_results.mean_cv_mae,
                    "cv_std_mae": cv_results.std_cv_mae,
                    "cv_mean_rmsle": cv_results.mean_cv_rmsle,
                    "cv_std_rmsle": cv_results.std_cv_rmsle,
                    "cv_mean_best_iteration": cv_results.mean_best_iteration,
                    "cv_results": cv_results.model_dump(),
                }
            )
        else:
            logger.warning("cv_results.json not found in output tarball")

        oof_path = Path(extract_dir) / "oof_evaluation.json"
        if oof_path.exists():
            oof_results = OofEvaluation.model_validate_json(oof_path.read_text(encoding="utf-8"))
            metrics.update(
                {
                    "oof_r2": oof_results.oof_r2,
                    "oof_mae": oof_results.oof_mae,
                    "oof_rmsle": oof_results.oof_rmsle,
                    "oof_num_samples": oof_results.num_oof_samples,
                    "oof_results": oof_results.model_dump(),
                }
            )
        else:
            logger.warning("oof_evaluation.json not found in output tarball")

    return metrics


def fetch_training_job_duration(job_name: str, region: str) -> dict[str, float] | None:
    """Return training duration fields from ``describe_training_job``."""
    sagemaker_boto = boto3.client("sagemaker", region_name=region, config=BOTO3_CONFIG)
    try:
        job_details = sagemaker_boto.describe_training_job(TrainingJobName=job_name)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        message = e.response.get("Error", {}).get("Message", "")
        if code == _SAGEMAKER_VALIDATION_CODE and "does not exist" in message.lower():
            logger.warning(
                "Training job %r not visible to describe_training_job yet",
                job_name,
            )
            return None
        raise

    start_time = job_details["TrainingStartTime"]
    end_time = job_details["TrainingEndTime"]
    duration = (end_time - start_time).total_seconds()
    return {
        "training_duration_seconds": duration,
        "training_duration_minutes": duration / 60,
        "training_job_status": job_details.get("TrainingJobStatus"),
    }


def download_and_parse_training_output(
    bucket: str,
    output_tarball_key: str,
    region: str,
) -> dict[str, Any]:
    """Download output tarball from S3 and parse metrics (raises on missing S3 if not benign)."""
    s3_client = boto3.client("s3", region_name=region, config=BOTO3_CONFIG)
    tarball_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tarball_path = tmp.name
        s3_client.download_file(bucket, output_tarball_key, tarball_path)
        return parse_training_output_tarball(Path(tarball_path))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in _S3_NOT_FOUND_CODES:
            logger.warning(
                "Output tarball not yet present (code=%s); skipping metrics extraction",
                code,
            )
            return {}
        raise
    finally:
        if tarball_path and Path(tarball_path).exists():
            with contextlib.suppress(OSError):
                Path(tarball_path).unlink()


def build_training_metrics(
    job_name: str,
    region: str,
    bucket: str,
    output_prefix: str,
    model_data_url: str | None,
) -> dict[str, Any]:
    """Full metrics dict for registry tasks."""
    output_tarball_key = f"{output_prefix}/{job_name}/output/output.tar.gz"
    performance = download_and_parse_training_output(bucket, output_tarball_key, region)
    duration_info = fetch_training_job_duration(job_name, region)
    if duration_info:
        performance.update(duration_info)

    performance.update(
        {
            "job_name": job_name,
            "duration": performance.get("training_duration_seconds", 0),
            "model_data_url": model_data_url,
        }
    )
    return TrainingMetrics.from_performance_dict(performance).to_performance_dict()
