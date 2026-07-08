"""
One-shot backfill script: writes SHA256 + size_bytes for every existing model
package in a given Model Package Group into its CustomerMetadataProperties so
the new ArtifactIntegrity verification path can validate before unpickling.

Usage:
    uv run python scripts/backfill_model_manifests.py \
        --group-name box-office-dev-box-office-models \
        --region eu-north-1

    uv run python scripts/backfill_model_manifests.py --dry-run ...

The script is idempotent: it skips packages that already have matching `sha256`
metadata (re-writing only when the stored value is missing or stale relative
to the actual S3 object).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("backfill_model_manifests")

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def compute_sha256(path: Path) -> tuple[str, int]:
    """Stream-hash the file and return (hex_digest, size_bytes)."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def parse_s3_url(url: str) -> tuple[str, str]:
    if not url.startswith("s3://"):
        raise ValueError(f"Not an s3:// URL: {url}")
    bucket, _, key = url[len("s3://") :].partition("/")
    if not bucket or not key:
        raise ValueError(f"Malformed s3:// URL: {url}")
    return bucket, key


def iter_packages(sagemaker, group_name: str) -> Iterable[dict]:
    paginator = sagemaker.get_paginator("list_model_packages")
    for page in paginator.paginate(ModelPackageGroupName=group_name):
        yield from page.get("ModelPackageSummaryList", [])


def backfill_one(
    *,
    sagemaker,
    s3,
    arn: str,
    dry_run: bool,
) -> dict:
    details = sagemaker.describe_model_package(ModelPackageName=arn)
    existing_meta = details.get("CustomerMetadataProperties", {}) or {}

    containers = (details.get("InferenceSpecification") or {}).get("Containers", [])
    if not containers:
        return {"arn": arn, "status": "skipped", "reason": "no containers"}
    model_data_url = containers[0].get("ModelDataUrl")
    if not model_data_url:
        return {"arn": arn, "status": "skipped", "reason": "no ModelDataUrl"}

    bucket, key = parse_s3_url(model_data_url)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
        tmp_path = Path(tmp.name)
    try:
        s3.download_file(bucket, key, str(tmp_path))
        sha256, size = compute_sha256(tmp_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if existing_meta.get("sha256") == sha256 and existing_meta.get("size_bytes") == str(
        size
    ):
        return {"arn": arn, "status": "already_current", "sha256": sha256}

    new_meta = dict(existing_meta)
    new_meta["sha256"] = sha256
    new_meta["size_bytes"] = str(size)

    if dry_run:
        return {
            "arn": arn,
            "status": "would_update",
            "sha256": sha256,
            "size_bytes": size,
        }

    sagemaker.update_model_package(
        ModelPackageArn=arn,
        CustomerMetadataProperties=new_meta,
    )
    return {"arn": arn, "status": "updated", "sha256": sha256, "size_bytes": size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-name", required=True, help="Model Package Group name")
    parser.add_argument("--region", required=True, help="AWS region")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute hashes but skip update_model_package",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    sagemaker = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    results = {
        "updated": 0,
        "already_current": 0,
        "would_update": 0,
        "skipped": 0,
        "errored": 0,
    }
    for summary in iter_packages(sagemaker, args.group_name):
        arn = summary["ModelPackageArn"]
        try:
            result = backfill_one(
                sagemaker=sagemaker, s3=s3, arn=arn, dry_run=args.dry_run
            )
            logger.info("%s -> %s", arn, result["status"])
            results[result["status"]] = results.get(result["status"], 0) + 1
        except ClientError as exc:
            logger.error("AWS error processing %s: %s", arn, exc)
            results["errored"] += 1
        except Exception as exc:  # noqa: BLE001 - want a wide net for one-shot tooling
            logger.exception("Unexpected error processing %s: %s", arn, exc)
            results["errored"] += 1

    logger.info("Done. Summary: %s", results)
    return 0 if results["errored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
