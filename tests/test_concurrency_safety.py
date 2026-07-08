"""Concurrency safety: races between pipeline runs over shared filesystem/registry state."""

from __future__ import annotations

import os

import pytest

# --------------------------------------------------------------------------- #
# 4.4 — _scoped_env restores os.environ
# --------------------------------------------------------------------------- #


class TestScopedEnv:
    def test_scoped_env_restores_prior_values(self, monkeypatch):
        from box_office.orchestration.tasks.data_tasks import _scoped_env

        monkeypatch.setenv("FOO_PRESENT", "original")
        monkeypatch.delenv("FOO_ABSENT", raising=False)

        snapshot_before = dict(os.environ)

        with _scoped_env({"FOO_PRESENT": "scoped", "FOO_ABSENT": "scoped"}):
            assert os.environ["FOO_PRESENT"] == "scoped"
            assert os.environ["FOO_ABSENT"] == "scoped"

        assert os.environ["FOO_PRESENT"] == "original"
        assert "FOO_ABSENT" not in os.environ
        assert dict(os.environ) == snapshot_before

    def test_scoped_env_restores_on_exception(self, monkeypatch):
        from box_office.orchestration.tasks.data_tasks import _scoped_env

        monkeypatch.delenv("FOO_RAISE", raising=False)

        with pytest.raises(RuntimeError):
            with _scoped_env({"FOO_RAISE": "x"}):
                assert os.environ["FOO_RAISE"] == "x"
                raise RuntimeError("boom")

        assert "FOO_RAISE" not in os.environ
