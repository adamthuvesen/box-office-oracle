"""Tests for the `ingestion-cli-imports` capability.

Covers task 7.4: confirm the CLI imports tmdb_discovery via the package and
not via `importlib.util` against a `__file__`-relative path.
"""

from __future__ import annotations

import inspect


def test_cli_does_not_use_importlib_util_for_tmdb_discovery():
    """The CLI source must not contain `importlib.util.spec_from_file_location`.

    The dynamic-import ceremony lives in the dedicated
    box_office.ingestion.tmdb_discovery module so the CLI stays oblivious to
    the upstream script's path-on-disk.
    """
    from box_office.ingestion import cli

    src = inspect.getsource(cli)
    assert "spec_from_file_location" not in src, (
        "cli.py still uses importlib.util.spec_from_file_location — "
        "import via box_office.ingestion.tmdb_discovery instead."
    )


def test_tmdb_discovery_module_exposes_expected_symbols():
    """The encapsulating module exposes the three callables the CLI needs."""
    from box_office.ingestion import tmdb_discovery

    assert callable(tmdb_discovery.get_existing_ids)
    assert callable(tmdb_discovery.discover_movies)
    assert callable(tmdb_discovery.filter_new_movies)
    assert set(tmdb_discovery.__all__) == {
        "get_existing_ids",
        "discover_movies",
        "filter_new_movies",
    }
