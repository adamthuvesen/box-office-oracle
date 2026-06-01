"""Check the SageMaker model registry and decide whether to deploy.

The script writes human-readable logs to stdout/stderr and machine-readable
``key=value`` lines only to the file referenced by ``$GITHUB_OUTPUT``.

The ``deploy_model`` decision rule: deploy iff the most recently created
``Approved`` model package in the configured group is younger than
``--max-age-hours`` (default 1).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("check_model")


def _write_outputs(github_output: str | None, **outputs: str) -> None:
    """Append ``key=value`` lines to ``$GITHUB_OUTPUT`` when set.

    No-op when ``$GITHUB_OUTPUT`` is unset (e.g. local invocation).
    """
    if not github_output:
        logger.info("GITHUB_OUTPUT not set; would have written: %s", outputs)
        return
    path = Path(github_output)
    with path.open("a", encoding="utf-8") as fh:
        for key, value in outputs.items():
            fh.write(f"{key}={value}\n")


def decide_deploy(
    *,
    model_group_name: str,
    region: str,
    max_age_hours: float,
    now: datetime | None = None,
    sagemaker_client=None,
) -> dict[str, str]:
    """Return the GITHUB_OUTPUT key/value mapping for the deploy decision.

    Pure-ish function: takes a SageMaker client (so unit tests can inject a
    fake) and returns the mapping rather than writing to disk. The CLI entry
    point handles I/O.
    """
    if sagemaker_client is None:
        import boto3  # local import — keeps this module importable in tests

        sagemaker_client = boto3.client("sagemaker", region_name=region)

    if now is None:
        now = datetime.now(timezone.utc)

    logger.info("Checking for approved models in group: %s", model_group_name)

    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = response.get("ModelPackageSummaryList", [])

    if not packages:
        logger.warning("No approved models found in registry")
        return {"deploy_model": "false", "model_package_arn": ""}

    latest = packages[0]
    arn = latest["ModelPackageArn"]
    creation_time = latest["CreationTime"]
    status = latest.get("ModelPackageStatus", "")

    age = now - creation_time
    deploy = age < timedelta(hours=max_age_hours)
    age_hours = age.total_seconds() / 3600.0

    logger.info("Found approved model: %s (status=%s)", arn, status)
    logger.info(
        "Model age: %.2f hours; threshold: %.2f hours", age_hours, max_age_hours
    )
    logger.info("Deploy decision: %s", deploy)

    return {
        "deploy_model": "true" if deploy else "false",
        "model_package_arn": arn,
        "model_status": status,
        "model_age_hours": f"{age_hours:.1f}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-group-name", required=True)
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-north-1"))
    parser.add_argument("--max-age-hours", type=float, default=1.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        outputs = decide_deploy(
            model_group_name=args.model_group_name,
            region=args.region,
            max_age_hours=args.max_age_hours,
        )
    except Exception as exc:  # pragma: no cover — surfaced via stderr in CI
        logger.error("Error checking model registry: %s", exc)
        _write_outputs(
            os.environ.get("GITHUB_OUTPUT"),
            deploy_model="false",
            model_package_arn="",
        )
        return 1

    _write_outputs(os.environ.get("GITHUB_OUTPUT"), **outputs)

    # Empty registry is a hard failure for the workflow (matches prior behavior).
    if outputs.get("model_package_arn", "") == "":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
