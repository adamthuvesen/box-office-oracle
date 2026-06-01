"""Tests for box_office.inference.app.integrity."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path

import pytest

from box_office.utils.safe_tarfile import extractall_data_filter

from box_office.inference.app.integrity import (
    ArtifactIntegrityError,
    compute_sha256,
    reset_verified_cache,
    verify_artifact,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_verified_cache()
    yield
    reset_verified_cache()


@pytest.fixture
def sample_file(tmp_path: Path) -> tuple[Path, str]:
    payload = b"box-office-test-artifact-payload-" * 100
    expected = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    return path, expected


def test_compute_sha256_matches_hashlib(sample_file):
    path, expected = sample_file
    assert compute_sha256(path) == expected


def test_verify_artifact_accepts_match(sample_file):
    path, expected = sample_file
    assert verify_artifact(path, expected) == expected


def test_verify_artifact_is_case_insensitive(sample_file):
    path, expected = sample_file
    assert verify_artifact(path, expected.upper()) == expected


def test_verify_artifact_raises_on_mismatch(sample_file):
    path, _ = sample_file
    bogus = "0" * 64
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact(path, bogus)


def test_verify_artifact_raises_on_empty_expected(sample_file):
    path, _ = sample_file
    with pytest.raises(ArtifactIntegrityError):
        verify_artifact(path, "")


def test_verify_artifact_caches_unchanged_files(sample_file, monkeypatch):
    path, expected = sample_file
    assert verify_artifact(path, expected) == expected

    # If the file is unchanged, a second verify must not re-hash. We assert
    # that by patching compute_sha256 to raise — the cache must short-circuit.
    from box_office.inference.app import integrity as integrity_mod

    def _explode(_path):
        raise AssertionError("compute_sha256 should not be called for cached path")

    monkeypatch.setattr(integrity_mod, "compute_sha256", _explode)
    assert verify_artifact(path, expected) == expected


def test_verify_artifact_rehashes_after_mtime_change(sample_file):
    path, expected = sample_file
    assert verify_artifact(path, expected) == expected

    # Touch mtime forward; cache key changes; verify must re-hash and still pass.
    new_mtime = os.stat(path).st_mtime + 10
    os.utime(path, (new_mtime, new_mtime))
    assert verify_artifact(path, expected) == expected


# ---------------------------------------------------------------------------
# Tar extraction safety (covers spec scenario "Malicious tar member is rejected")
# ---------------------------------------------------------------------------


def _make_tar(
    tmp_path: Path, members: list[tarfile.TarInfo], payloads: dict[str, bytes]
) -> Path:
    """Build a .tar.gz containing the given members."""
    tar_path = tmp_path / "evil.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for info in members:
            data = payloads.get(info.name, b"")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return tar_path


def test_tar_data_filter_rejects_parent_traversal(tmp_path: Path):
    if sys.version_info < (3, 12):
        pytest.skip("PEP 706 data filter requires Python 3.12+")

    info = tarfile.TarInfo(name="../escaped.txt")
    info.type = tarfile.REGTYPE
    tar_path = _make_tar(tmp_path, [info], {"../escaped.txt": b"hi"})

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()

    with tarfile.open(tar_path, "r:gz") as tar:
        with pytest.raises(
            (
                tarfile.OutsideDestinationError,
                tarfile.AbsolutePathError,
                tarfile.LinkOutsideDestinationError,
            )
        ):
            extractall_data_filter(tar, extract_dir)

    # And nothing escaped:
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_data_filter_rejects_symlink_escape(tmp_path: Path):
    if sys.version_info < (3, 12):
        pytest.skip("PEP 706 data filter requires Python 3.12+")

    sym = tarfile.TarInfo(name="link")
    sym.type = tarfile.SYMTYPE
    sym.linkname = "../../etc"
    tar_path = _make_tar(tmp_path, [sym], {})

    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()

    with tarfile.open(tar_path, "r:gz") as tar:
        with pytest.raises(
            (
                tarfile.OutsideDestinationError,
                tarfile.LinkOutsideDestinationError,
                tarfile.AbsoluteLinkError,
            )
        ):
            extractall_data_filter(tar, extract_dir)
