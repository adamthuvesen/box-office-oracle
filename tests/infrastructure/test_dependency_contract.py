"""Checks for packages imported directly by production code."""

import tomllib
from pathlib import Path


def test_direct_runtime_dependencies_are_declared() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    )
    dependencies = {
        requirement.split("==", 1)[0]
        for requirement in pyproject["project"]["dependencies"]
    }

    assert {"pydantic", "requests"} <= dependencies
