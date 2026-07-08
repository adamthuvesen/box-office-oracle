"""Shared AWS / boto3 constants and parsing helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

from botocore.config import Config

# 60 s connect / 300 s read with 3 adaptive retries. Tuned for SageMaker
# describe / list / download patterns where p99 control-plane latency can
# exceed the boto3 default (15 s) under throttling.
BOTO3_CONFIG = Config(
    connect_timeout=60,
    read_timeout=300,
    retries={"max_attempts": 3, "mode": "adaptive"},
)

# Streaming chunk size for SHA-256 hashing of model artifacts.
SHA256_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an ``s3://bucket/key`` URI into ``(bucket, key)``.

    Raises ``ValueError`` on missing scheme, missing bucket, or missing key —
    callers reach this from boto3 download paths where any of those three
    failures surface much later as a confusing botocore error.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI (scheme must be 's3'): {uri!r}")
    if not parsed.netloc:
        raise ValueError(f"Invalid S3 URI (missing bucket): {uri!r}")
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"Invalid S3 URI (missing key): {uri!r}")
    return parsed.netloc, key


def resolve_aws_region(default: str = "eu-north-1") -> str:
    """Return the AWS region from env vars, mirroring ``config.aws.region``.

    For use in modules where importing the project ``config`` is awkward
    (e.g. the SageMaker entry script that runs inside the training container).
    """
    return (
        os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or default
    )


def compute_sha256_stream(path: Path) -> str:
    """Stream-hash the file at ``path`` and return its hex digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(SHA256_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
