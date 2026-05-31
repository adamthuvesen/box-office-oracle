"""Tests for box_office.utils.safe_tarfile."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from box_office.utils.safe_tarfile import extractall_data_filter


def test_extractall_data_filter_writes_member(tmp_path: Path) -> None:
    tar_path = tmp_path / "a.tar.gz"
    payload = b"hello"
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo(name="hello.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    out = tmp_path / "out"
    out.mkdir()
    with tarfile.open(tar_path, "r:gz") as tar:
        extractall_data_filter(tar, out)

    assert (out / "hello.txt").read_bytes() == payload
