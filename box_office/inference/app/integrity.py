"""
Artifact integrity verification for model packages loaded from S3.

The inference Lambda must verify the SHA256 of any downloaded model artifact
against the trusted manifest stored in the SageMaker Model Package's
``CustomerMetadataProperties`` BEFORE invoking ``pickle.load`` /
``joblib.load``. Bucket-write does not equal metadata-write, so this closes
the RCE surface where a compromised S3 key would otherwise become arbitrary
code execution inside the Lambda.

The module also provides a process-local "already-verified" cache keyed on
``(path, mtime, size)`` so that warm Lambda invocations don't re-hash the
same files on every request. The cache is intentionally per-process — Lambda
``/tmp`` is shared across invocations on the same worker but not across
workers, and we re-verify on cold start.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

from box_office.utils.aws_helpers import compute_sha256_stream

logger = logging.getLogger(__name__)


class ArtifactIntegrityError(Exception):
    """Raised when an artifact's SHA256 does not match the trusted manifest."""


VerifyKey = Tuple[str, float, int]
_verified_paths: Dict[VerifyKey, str] = {}


def _verify_key(path: Path) -> VerifyKey:
    stat = path.stat()
    return (str(path), stat.st_mtime, stat.st_size)


def compute_sha256(path: Path) -> str:
    """Stream-hash the file at ``path`` and return its hex digest."""
    return compute_sha256_stream(path)


def verify_artifact(path: Path, expected_sha256: str) -> str:
    """
    Verify that ``path`` hashes to ``expected_sha256``.

    Returns the verified hex digest. Raises ``ArtifactIntegrityError`` on
    mismatch or if ``expected_sha256`` is missing/empty.

    A successful verification is recorded in a process-local cache keyed on
    ``(path, mtime, size)``; re-verification of an unchanged file is a
    constant-time dict lookup.
    """
    if not expected_sha256:
        raise ArtifactIntegrityError(
            f"No expected SHA256 provided for {path}; refusing to load unverified artifact"
        )

    expected = expected_sha256.lower()
    cache_key = _verify_key(path)
    cached = _verified_paths.get(cache_key)
    if cached == expected:
        logger.debug("artifact integrity: cache hit for %s", path)
        return cached

    actual = compute_sha256(path)
    if actual != expected:
        # Do NOT log the file contents or full path of the bad artifact at
        # WARNING+ levels in a way that would surface to operator dashboards;
        # the digest pair is enough to triage.
        logger.error(
            "artifact integrity mismatch: path=%s expected=%s actual=%s",
            path,
            expected,
            actual,
        )
        raise ArtifactIntegrityError(
            f"SHA256 mismatch for {path}: expected={expected[:16]}... actual={actual[:16]}..."
        )

    _verified_paths[cache_key] = actual
    logger.debug("artifact integrity: verified %s sha256=%s", path, actual[:16])
    return actual


def reset_verified_cache() -> None:
    """Clear the process-local verified-paths cache (test hook)."""
    _verified_paths.clear()
