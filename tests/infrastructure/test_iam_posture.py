"""IAM posture tests for ``infrastructure/terraform/iam.tf``.

Text-parse against forbidden patterns (``AmazonSageMakerFullAccess``,
IAM mutation actions, hardcoded ``ACCOUNTADMIN``). ``terraform plan
-json`` would be stronger but needs S3 backend init, unavailable in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
IAM_TF = REPO_ROOT / "infrastructure" / "terraform" / "iam.tf"
PROFILES_YML = REPO_ROOT / "transformations" / "profiles.yml"


@pytest.fixture(scope="module")
def iam_tf_text() -> str:
    assert IAM_TF.exists(), f"missing {IAM_TF}"
    return IAM_TF.read_text()


def _strip_comments(text: str) -> str:
    """Remove single-line ``#`` and ``//`` comments and ``/* */`` blocks
    so we only inspect the *active* HCL configuration, not justification
    notes about removed actions."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    out = []
    for line in text.splitlines():
        stripped = line.split("#", 1)[0]
        stripped = stripped.split("//", 1)[0]
        out.append(stripped)
    return "\n".join(out)


def test_sagemaker_role_does_not_attach_full_access(iam_tf_text: str) -> None:
    active = _strip_comments(iam_tf_text)
    assert "AmazonSageMakerFullAccess" not in active, (
        "SageMaker execution role must not attach the "
        "AmazonSageMakerFullAccess managed policy. A justification "
        "comment may mention it; the active configuration must not."
    )


FORBIDDEN_IAM_MUTATION_ACTIONS = (
    "iam:CreateRole",
    "iam:DeleteRole",
    "iam:UpdateRole",
    "iam:PutRolePolicy",
    "iam:DeleteRolePolicy",
    "iam:AttachRolePolicy",
    "iam:DetachRolePolicy",
    "iam:CreateOpenIDConnectProvider",
    "iam:DeleteOpenIDConnectProvider",
    "iam:UpdateOpenIDConnectProviderThumbprint",
    "iam:CreateServiceLinkedRole",
)


@pytest.mark.parametrize("action", FORBIDDEN_IAM_MUTATION_ACTIONS)
def test_no_iam_mutation_action_in_active_config(iam_tf_text: str, action: str) -> None:
    active = _strip_comments(iam_tf_text)
    assert action not in active, (
        f"'{action}' must not appear in the active IAM configuration of the "
        "github_actions role. IAM mutation belongs in a separate admin apply."
    )


def test_pass_role_is_scoped_not_wildcarded(iam_tf_text: str) -> None:
    """``iam:PassRole`` is allowed (we need to pass the SageMaker role to
    SageMaker), but only when scoped to that specific role's ARN — never on
    Resource = "*"."""
    active = _strip_comments(iam_tf_text)
    # Find every block containing iam:PassRole and confirm none of them
    # use Resource = "*" within the same statement object.
    statements = re.split(r"\{", active)
    for block in statements:
        if "iam:PassRole" in block:
            # The same statement must reference a specific ARN, not "*"
            # and not just a Resource = "*" line. The current correct
            # pattern is `Resource = aws_iam_role.sagemaker_execution_role.arn`.
            if 'Resource = "*"' in block or 'Resource = ["*"]' in block:
                pytest.fail(
                    "iam:PassRole appears in a statement with Resource = '*'. "
                    "Scope it to the specific role ARN."
                )


def test_profiles_yml_does_not_hardcode_accountadmin() -> None:
    assert PROFILES_YML.exists(), f"missing {PROFILES_YML}"
    text = PROFILES_YML.read_text()
    # Strip YAML comments
    active = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    assert "ACCOUNTADMIN" not in active.upper().replace(
        "ACCOUNT_ADMIN", "ACCOUNTADMIN"
    ).replace("THE ACCOUNT", ""), (
        "profiles.yml must not hardcode ACCOUNTADMIN. "
        "Use {{ env_var('SNOWFLAKE_ROLE') }} and provision DBT_RUNNER."
    )


def test_profiles_yml_role_is_env_driven() -> None:
    text = PROFILES_YML.read_text()
    assert (
        "env_var('SNOWFLAKE_ROLE')" in text
    ), "dbt profile must resolve role from SNOWFLAKE_ROLE."
