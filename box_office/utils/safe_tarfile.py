"""Tar extraction with PEP 706 ``filter='data'`` when supported."""

from __future__ import annotations

import sys
import tarfile
from pathlib import Path


def extractall_data_filter(tar: tarfile.TarFile, path: str | Path) -> None:
    """Extract members to ``path``; use the ``data`` filter on Python 3.12+.

    On Python 3.11, :meth:`tarfile.TarFile.extractall` has no ``filter``
    argument — extraction proceeds without the hardening (call sites should
    run on 3.12+ in production: Lambda inference image, CI).
    """
    path = Path(path)
    if sys.version_info >= (3, 12):
        tar.extractall(path, filter="data")
    else:
        tar.extractall(path)
