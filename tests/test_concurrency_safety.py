"""Concurrency safety: races between pipeline runs over shared filesystem/registry state."""

from __future__ import annotations

import os
import random

import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# 4.2 — HeuristicEnricher uses a local Random instance
# --------------------------------------------------------------------------- #


class TestEnricherLocalRng:
    def test_enricher_does_not_perturb_global_random(self):
        from box_office.ingestion.data_enrichment import HeuristicEnricher

        random.seed(0)
        baseline = random.random()

        random.seed(0)
        HeuristicEnricher(seed=42)  # Must not call random.seed
        observed = random.random()

        assert (
            observed == baseline
        ), "HeuristicEnricher() perturbed the global RNG — expected local Random()"

    def test_two_enrichers_with_same_seed_produce_identical_output(self):
        from box_office.ingestion.data_enrichment import HeuristicEnricher

        df = pd.DataFrame(
            {
                "title": ["Movie A", "Movie B", "Movie C", "Movie D"] * 5,
                "production_budget": [10_000_000, 50_000_000, 100_000_000, 200_000_000]
                * 5,
                "worldwide_gross": [50_000_000, 200_000_000, 500_000_000, 1_500_000_000]
                * 5,
            }
        )

        a = HeuristicEnricher(seed=42).enrich(df.copy())
        b = HeuristicEnricher(seed=42).enrich(df.copy())
        pd.testing.assert_frame_equal(a, b)


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
