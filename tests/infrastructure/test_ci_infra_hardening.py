"""CI/infra hardening assertions — text/YAML parse only, no AWS or terraform init."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TERRAFORM_DIR = REPO_ROOT / "infrastructure" / "terraform"
SERVERLESS_INFER_DIR = TERRAFORM_DIR / "modules" / "serverless-inference"

MAIN_TF = TERRAFORM_DIR / "main.tf"
VARS_TF = TERRAFORM_DIR / "variables.tf"
SERVERLESS_MAIN_TF = SERVERLESS_INFER_DIR / "main.tf"
SERVERLESS_IAM_TF = SERVERLESS_INFER_DIR / "iam.tf"
PROD_TFVARS = TERRAFORM_DIR / "environments" / "prod.tfvars"
DEV_TFVARS = TERRAFORM_DIR / "environments" / "dev.tfvars"
BACKEND_TF = TERRAFORM_DIR / "backend.tf"

DOCKER_DIR = REPO_ROOT / "infrastructure" / "docker" / "inference"
DOCKERFILE = DOCKER_DIR / "Dockerfile"
DOCKERIGNORE = DOCKER_DIR / ".dockerignore"

WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CI_YML = WORKFLOWS_DIR / "ci.yml"
ML_PIPELINE_YML = WORKFLOWS_DIR / "ml-pipeline.yml"
CD_INFRA_YML = WORKFLOWS_DIR / "cd-infrastructure.yml"

DATA_TASKS_PY = REPO_ROOT / "box_office" / "orchestration" / "tasks" / "data_tasks.py"


def _strip_hcl_comments(text: str) -> str:
    """Strip ``#``/``//`` line comments and ``/* */`` blocks from HCL."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    out = []
    for line in text.splitlines():
        stripped = line.split("#", 1)[0]
        stripped = stripped.split("//", 1)[0]
        out.append(stripped)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3.1 / Spec: CD does not push :latest
# ---------------------------------------------------------------------------


def test_cd_workflow_does_not_push_latest_tag() -> None:
    text = CD_INFRA_YML.read_text()
    # The build step must not tag or push :latest. We look for both patterns
    # in the active YAML (comments stay in the file as justification).
    # Strip YAML comments first so a "no :latest" comment can't trigger us.
    active = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    assert "repository_uri }}:latest" not in active, (
        "Release hardening M13: cd-infrastructure.yml must not tag or push :latest. "
        "Use only the SHA-derived tag."
    )
    assert "docker push ${{ steps.ecr.outputs.repository_uri }}:latest" not in active


# ---------------------------------------------------------------------------
# 3.2 / Spec: SageMaker artifacts bucket has noncurrent-version expiration
# ---------------------------------------------------------------------------


def test_sagemaker_artifacts_bucket_has_lifecycle_rule() -> None:
    text = _strip_hcl_comments(MAIN_TF.read_text())
    assert (
        'resource "aws_s3_bucket_lifecycle_configuration" "sagemaker_artifacts"' in text
    ), (
        "Release hardening M3: SageMaker artifacts bucket must declare a lifecycle "
        "configuration."
    )
    assert "noncurrent_version_expiration" in text, (
        "Release hardening M3: lifecycle must include a noncurrent_version_expiration "
        "block."
    )
    # Confirm the noncurrent-days value is present and <= 90 (we set 90).
    m = re.search(r"noncurrent_days\s*=\s*(\d+)", text)
    assert m is not None, "noncurrent_days must be set"
    assert int(m.group(1)) <= 90, "noncurrent_days must be <= 90"


# ---------------------------------------------------------------------------
# 3.3 / Spec: terraform_state_bucket_name has no default
# ---------------------------------------------------------------------------


def test_terraform_state_bucket_has_no_account_id_default() -> None:
    text = _strip_hcl_comments(VARS_TF.read_text())
    # Find the variable block for terraform_state_bucket_name and confirm no
    # `default = "..."` line inside it.
    m = re.search(
        r'variable\s+"terraform_state_bucket_name"\s*\{(?P<body>[^}]*)\}',
        text,
        flags=re.DOTALL,
    )
    assert m is not None, "variable block missing"
    body = m.group("body")
    assert "default" not in body, (
        "Release hardening M4: terraform_state_bucket_name must not have a default — "
        "it must be supplied per environment via tfvars."
    )
    # No 12-digit AWS account-ID strings anywhere as a default value.
    assert not re.search(r"default\s*=\s*\".*\d{12}.*\"", text), (
        "No AWS account ID should appear as a default in variables.tf."
    )


def test_tfvars_set_terraform_state_bucket_name() -> None:
    for tfvars in (PROD_TFVARS, DEV_TFVARS):
        text = tfvars.read_text()
        active = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        assert re.search(
            r'^\s*terraform_state_bucket_name\s*=\s*".+"',
            active,
            flags=re.MULTILINE,
        ), f"{tfvars.name} must set terraform_state_bucket_name explicitly"


def test_no_aws_account_id_in_tracked_terraform() -> None:
    """No 12-digit AWS account ID may appear in backend.tf or the tfvars.

    The variable-default guard above missed these files, which is how a real
    account id reached the public repo. backend.tf must be public-safe (config
    injected at init) and tfvars must use example/placeholder bucket names.
    """
    for path in (BACKEND_TF, PROD_TFVARS, DEV_TFVARS):
        active = "\n".join(
            line.split("#", 1)[0] for line in path.read_text().splitlines()
        )
        assert not re.search(r"\d{12}", active), (
            f"{path.name} contains a 12-digit AWS account id; keep state/bucket "
            "values out of tracked files (inject at init / via untracked override)."
        )


# ---------------------------------------------------------------------------
# Production alarm SNS topic ARN is clean (empty or a real ARN, no placeholder)
# ---------------------------------------------------------------------------


def test_prod_alarm_sns_topic_arn_is_clean() -> None:
    """alarm_sns_topic_arn must be either empty (alarms disabled — the safe
    default the module handles gracefully) or a real SNS ARN. A placeholder
    ARN containing <ACCOUNT_ID> is not allowed: if applied unedited it would
    point alarm actions at a malformed target."""
    text = PROD_TFVARS.read_text()
    active = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    m = re.search(
        r'^\s*alarm_sns_topic_arn\s*=\s*"(?P<arn>[^"]*)"',
        active,
        flags=re.MULTILINE,
    )
    assert m is not None, "prod.tfvars must declare alarm_sns_topic_arn"
    arn = m.group("arn")
    assert arn == "" or (arn.startswith("arn:aws:sns:") and "<" not in arn), (
        "alarm_sns_topic_arn must be empty or a real SNS ARN with no <placeholder>."
    )


# ---------------------------------------------------------------------------
# 3.6 / Spec: .dockerignore excludes the required entries
# ---------------------------------------------------------------------------


REQUIRED_DOCKERIGNORE_ENTRIES = (
    ".venv/",
    "keys/",
    ".git/",
    ".github/",
    "tests/",
    "**/__pycache__",
    "*.pyc",
    "openspec/",
    "data/",
    "transformations/target/",
)


@pytest.mark.parametrize("entry", REQUIRED_DOCKERIGNORE_ENTRIES)
def test_dockerignore_excludes_entry(entry: str) -> None:
    assert DOCKERIGNORE.exists(), f"missing {DOCKERIGNORE}"
    lines = {line.strip() for line in DOCKERIGNORE.read_text().splitlines()}
    assert entry in lines, (
        f"Release hardening M10: .dockerignore must exclude '{entry}' so it is never "
        "sent to the Docker daemon."
    )


def test_dockerfile_runtime_uses_dedicated_target_dir() -> None:
    text = DOCKERFILE.read_text()
    assert "uv pip install --target" in text, (
        "Release hardening M10: builder stage must install into a dedicated target "
        "directory (not the system site-packages)."
    )
    # And the runtime stage must NOT wholesale-copy the builder's
    # /var/lang/lib/python3.12/site-packages tree.
    assert (
        "/var/lang/lib/python3.12/site-packages /var/lang/lib/python3.12/site-packages"
        not in text
    ), (
        "Release hardening M10: runtime stage must not copy the entire builder "
        "site-packages tree."
    )


# ---------------------------------------------------------------------------
# 3.7 / Spec: every job in the four workflows has an explicit timeout-minutes
# ---------------------------------------------------------------------------


WORKFLOW_FILES = [CI_YML, ML_PIPELINE_YML, CD_INFRA_YML]


@pytest.mark.parametrize("workflow", WORKFLOW_FILES, ids=lambda p: p.name)
def test_every_job_has_timeout_minutes(workflow: Path) -> None:
    assert workflow.exists(), f"missing {workflow}"
    data = yaml.safe_load(workflow.read_text())
    jobs = data.get("jobs", {})
    assert jobs, f"no jobs found in {workflow}"
    missing = [
        name
        for name, body in jobs.items()
        if not isinstance(body, dict) or "timeout-minutes" not in body
    ]
    assert not missing, (
        f"Release hardening M11: jobs in {workflow.name} missing timeout-minutes: "
        f"{missing}. Every job must declare an explicit cap so a hung job "
        "cannot run to GitHub's 360-minute default."
    )


def test_ml_pipeline_training_is_manual_only() -> None:
    data = yaml.safe_load(ML_PIPELINE_YML.read_text())
    triggers = data.get("on") or data.get(True)

    assert isinstance(triggers, dict), "ml-pipeline.yml must declare workflow triggers"
    assert "workflow_dispatch" in triggers, (
        "Full SageMaker training must stay available as a manual workflow."
    )
    assert "push" not in triggers, (
        "Full SageMaker training must not run automatically on main pushes."
    )
    assert "schedule" not in triggers, (
        "Full SageMaker training must not run on a schedule without fresh-data "
        "automation."
    )


def test_cd_infrastructure_deploy_is_manual_only() -> None:
    data = yaml.safe_load(CD_INFRA_YML.read_text())
    triggers = data.get("on") or data.get(True)

    assert isinstance(triggers, dict), (
        "cd-infrastructure.yml must declare workflow triggers"
    )
    assert "workflow_dispatch" in triggers, (
        "Infrastructure deploy must stay available as a manual workflow."
    )
    assert "push" not in triggers, (
        "Infrastructure deploy must not run automatically on main pushes in a "
        "public repo."
    )
    assert "schedule" not in triggers, (
        "Infrastructure deploy must not run on a schedule."
    )


# ---------------------------------------------------------------------------
# 3.10 / Spec: dbt CI runs compile (catches Jinja + ref/source errors)
#
# Parse + compile catches Jinja, ref, and source regressions without requiring
# CI to hold CREATE TABLE on the STAGING schema.
# ---------------------------------------------------------------------------


def test_dbt_ci_includes_compile_step() -> None:
    text = CI_YML.read_text()
    assert "dbt compile" in text, (
        "Release hardening M14: ci.yml must run `dbt compile` (catches Jinja and "
        "ref errors that `dbt parse` alone misses)."
    )


def test_ci_enforces_python_lint_and_formatting() -> None:
    data = yaml.safe_load(CI_YML.read_text())
    steps = data["jobs"]["test"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)

    assert "ruff check box_office tests scripts" in commands
    assert "ruff format --check box_office tests scripts" in commands


def test_ci_enforces_web_lint_and_build() -> None:
    data = yaml.safe_load(CI_YML.read_text())
    steps = data["jobs"]["web-quality"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)

    assert "pnpm --dir web install --frozen-lockfile" in commands
    assert "pnpm --dir web lint" in commands
    assert "pnpm --dir web build" in commands


def test_dbt_ci_parses_without_snowflake_credentials() -> None:
    data = yaml.safe_load(CI_YML.read_text())
    job = data["jobs"]["dbt-test"]
    steps = {step["name"]: step for step in job["steps"] if "name" in step}

    assert "if" not in steps["dbt deps"]
    assert "if" not in steps["dbt parse (validate models)"]
    assert steps["dbt compile (catch jinja and ref errors)"]["if"] == (
        "steps.creds.outputs.has_creds == 'true'"
    )
    assert "offline" in job["env"]["SNOWFLAKE_ACCOUNT"]


# ---------------------------------------------------------------------------
# 3.12 / Spec: save_dataset_to_snowflake validates table_name first
# ---------------------------------------------------------------------------


def test_save_dataset_to_snowflake_validates_identifier() -> None:
    text = DATA_TASKS_PY.read_text()
    # Find the save_dataset_to_snowflake function body.
    m = re.search(
        r"^def save_dataset_to_snowflake_impl\([\s\S]*?\) -> bool:(?P<body>[\s\S]*?)(?=^def )",
        text,
        flags=re.MULTILINE,
    )
    assert m is not None, "save_dataset_to_snowflake_impl function not found"
    body = m.group("body")
    # The validate call for table_name MUST come before write_pandas.
    validate_idx = body.find('validate_sql_identifier(table_name, "table")')
    write_idx = body.find("write_pandas(")
    assert validate_idx != -1, (
        "Release hardening M66: save_dataset_to_snowflake must call "
        'validate_sql_identifier(table_name, "table") before write_pandas.'
    )
    assert write_idx != -1, "write_pandas call missing"
    assert validate_idx < write_idx, (
        "validate_sql_identifier(table_name, ...) must precede write_pandas."
    )
