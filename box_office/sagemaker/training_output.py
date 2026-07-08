"""Parse SageMaker training job output artifacts."""

from __future__ import annotations

import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from box_office.ml.metrics_models import TrainingMetrics
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

        cv_path = os.path.join(extract_dir, "cv_results.json")
        if os.path.exists(cv_path):
            with open(cv_path, encoding="utf-8") as f:
                cv_results = json.load(f)
            metrics.update(
                {
                    "cv_mean_mae": cv_results.get("mean_cv_mae"),
                    "cv_std_mae": cv_results.get("std_cv_mae"),
                    "cv_mean_rmsle": cv_results.get("mean_cv_rmsle"),
                    "cv_std_rmsle": cv_results.get("std_cv_rmsle"),
                    "cv_mean_best_iteration": cv_results.get("mean_best_iteration"),
                }
            )
            metrics["cv_results"] = cv_results
        else:
            logger.warning("cv_results.json not found in output tarball")

        oof_path = os.path.join(extract_dir, "oof_evaluation.json")
        if os.path.exists(oof_path):
            with open(oof_path, encoding="utf-8") as f:
                oof_results = json.load(f)
            metrics.update(
                {
                    "oof_r2": oof_results.get("oof_r2"),
                    "oof_mae": oof_results.get("oof_mae"),
                    "oof_rmsle": oof_results.get("oof_rmsle"),
                    "oof_num_samples": oof_results.get("num_oof_samples"),
                }
            )
            metrics["oof_results"] = oof_results
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
        if tarball_path and os.path.exists(tarball_path):
            try:
                os.unlink(tarball_path)
            except OSError:
                pass


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
