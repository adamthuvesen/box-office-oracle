"""The dbt profile resolves its Snowflake role from the environment.

The IAM posture checks for ``infrastructure/terraform`` are checkov policies
in ``infrastructure/checkov`` (``scripts/lint_infra.sh``).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_YML = REPO_ROOT / "transformations" / "profiles.yml"


def test_profiles_yml_does_not_hardcode_accountadmin() -> None:
    assert PROFILES_YML.exists(), f"missing {PROFILES_YML}"
    text = PROFILES_YML.read_text()
    # Strip YAML comments
    active = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    assert "ACCOUNTADMIN" not in active.upper().replace("ACCOUNT_ADMIN", "ACCOUNTADMIN").replace(
        "THE ACCOUNT", ""
    ), (
        "profiles.yml must not hardcode ACCOUNTADMIN. "
        "Use {{ env_var('SNOWFLAKE_ROLE') }} and provision DBT_RUNNER."
    )


def test_profiles_yml_role_is_env_driven() -> None:
    text = PROFILES_YML.read_text()
    assert "env_var('SNOWFLAKE_ROLE')" in text, "dbt profile must resolve role from SNOWFLAKE_ROLE."
