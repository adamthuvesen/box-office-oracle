"""Checks for packages imported directly by production code, and the serving closure."""

import tomllib
from pathlib import Path

PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())


def _names(requirements: list[str]) -> set[str]:
    return {r.split("==", 1)[0].split("[", 1)[0].lower() for r in requirements}


def test_direct_runtime_dependencies_are_declared() -> None:
    assert "pydantic" in _names(PYPROJECT["project"]["dependencies"])
    assert "requests" in _names(PYPROJECT["project"]["optional-dependencies"]["train"])


def test_serving_closure_has_no_training_packages() -> None:
    serving = _names(PYPROJECT["project"]["dependencies"]) | _names(
        PYPROJECT["project"]["optional-dependencies"]["inference"]
    )
    training_only = {"prefect", "prefect-dbt", "sagemaker", "snowflake-connector-python"}
    training_only |= {name for name in serving if name.startswith("dbt-")}
    assert not serving & training_only
