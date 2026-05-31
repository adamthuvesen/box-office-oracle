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
CLAUDE_YML = WORKFLOWS_DIR / "claude.yml"

CHECK_MODEL_PY = REPO_ROOT / "scripts" / "check_model.py"
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
        "Bug-hunt M13: cd-infrastructure.yml must not tag or push :latest. "
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
        "Bug-hunt M3: SageMaker artifacts bucket must declare a lifecycle "
        "configuration."
    )
    assert "noncurrent_version_expiration" in text, (
        "Bug-hunt M3: lifecycle must include a noncurrent_version_expiration " "block."
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
        "Bug-hunt M4: terraform_state_bucket_name must not have a default — "
        "it must be supplied per environment via tfvars."
    )
    # No 12-digit AWS account-ID strings anywhere as a default value.
    assert not re.search(
        r"default\s*=\s*\".*\d{12}.*\"", text
    ), "No AWS account ID should appear as a default in variables.tf."


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
    assert arn == "" or (
        arn.startswith("arn:aws:sns:") and "<" not in arn
    ), "alarm_sns_topic_arn must be empty or a real SNS ARN with no <placeholder>."


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
        f"Bug-hunt M10: .dockerignore must exclude '{entry}' so it is never "
        "sent to the Docker daemon."
    )


def test_dockerfile_runtime_uses_dedicated_target_dir() -> None:
    text = DOCKERFILE.read_text()
    assert "uv pip install --target" in text, (
        "Bug-hunt M10: builder stage must install into a dedicated target "
        "directory (not the system site-packages)."
    )
    # And the runtime stage must NOT wholesale-copy the builder's
    # /var/lang/lib/python3.12/site-packages tree.
    assert (
        "/var/lang/lib/python3.12/site-packages /var/lang/lib/python3.12/site-packages"
        not in text
    ), (
        "Bug-hunt M10: runtime stage must not copy the entire builder "
        "site-packages tree."
    )


# ---------------------------------------------------------------------------
# 3.7 / Spec: every job in the four workflows has an explicit timeout-minutes
# ---------------------------------------------------------------------------


WORKFLOW_FILES = [CI_YML, ML_PIPELINE_YML, CD_INFRA_YML, CLAUDE_YML]


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
        f"Bug-hunt M11: jobs in {workflow.name} missing timeout-minutes: "
        f"{missing}. Every job must declare an explicit cap so a hung job "
        "cannot run to GitHub's 360-minute default."
    )


def test_ml_pipeline_training_is_manual_only() -> None:
    data = yaml.safe_load(ML_PIPELINE_YML.read_text())
    triggers = data.get("on") or data.get(True)

    assert isinstance(triggers, dict), "ml-pipeline.yml must declare workflow triggers"
    assert (
        "workflow_dispatch" in triggers
    ), "Full SageMaker training must stay available as a manual workflow."
    assert (
        "push" not in triggers
    ), "Full SageMaker training must not run automatically on main pushes."
    assert "schedule" not in triggers, (
        "Full SageMaker training must not run on a schedule without fresh-data "
        "automation."
    )


# ---------------------------------------------------------------------------
# 3.8 / Spec: ml-pipeline delegates to scripts/check_model.py
# ---------------------------------------------------------------------------


def test_check_model_script_exists_and_writes_github_output_correctly() -> None:
    assert CHECK_MODEL_PY.exists(), "Bug-hunt M12: scripts/check_model.py must exist."
    text = CHECK_MODEL_PY.read_text()
    assert (
        'f"{key}={value}\\n"' in text or "f'{key}={value}\\n'" in text
    ), "scripts/check_model.py must write GITHUB_OUTPUT in key=value\\n format."


# scripts/check_model.py currently has no caller; kept for the next
# deploy-model rewire. The assertion above pins its correctness.


# ---------------------------------------------------------------------------
# 3.10 / Spec: dbt CI runs compile (catches Jinja + ref/source errors)
#
# An earlier draft also gated `dbt build --select staging --empty` on
# SNOWFLAKE_PRIVATE_KEY availability, but the warehouse role used in CI
# (DBT_RUNNER) doesn't hold CREATE TABLE on the STAGING schema, so the
# build step failed with a real-world 003001 access-control error. For a
# solo project, parse + compile catches every regression we actually care
# about (Jinja, refs, sources) without forcing a CI-specific role grant.
# ---------------------------------------------------------------------------


def test_dbt_ci_includes_compile_step() -> None:
    text = CI_YML.read_text()
    assert "dbt compile" in text, (
        "Bug-hunt M14: ci.yml must run `dbt compile` (catches Jinja and "
        "ref errors that `dbt parse` alone misses)."
    )


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
        "Bug-hunt M66: save_dataset_to_snowflake must call "
        'validate_sql_identifier(table_name, "table") before write_pandas.'
    )
    assert write_idx != -1, "write_pandas call missing"
    assert (
        validate_idx < write_idx
    ), "validate_sql_identifier(table_name, ...) must precede write_pandas."


# ---------------------------------------------------------------------------
# Unit test: scripts/check_model.py decision logic
# ---------------------------------------------------------------------------


def test_check_model_decide_deploy_writes_correct_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke test: the decision function returns key=value pairs that the
    CLI then writes to GITHUB_OUTPUT. We exercise the file write end-to-end
    by invoking the CLI's ``_write_outputs`` helper with a tmp path."""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from scripts import check_model  # type: ignore[import-not-found]

    output_file = tmp_path / "github_output"
    check_model._write_outputs(
        str(output_file),
        deploy_model="true",
        model_package_arn="arn:aws:sagemaker:eu-north-1:123:model-package/x/1",
    )
    content = output_file.read_text()
    assert "deploy_model=true\n" in content
    assert (
        "model_package_arn=arn:aws:sagemaker:eu-north-1:123:model-package/x/1\n"
        in content
    )


def test_check_model_decide_deploy_recent_model() -> None:
    """The decide function returns deploy_model=true for a fresh model."""
    import sys
    from datetime import datetime, timedelta, timezone

    sys.path.insert(0, str(REPO_ROOT))
    from scripts import check_model  # type: ignore[import-not-found]

    now = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)
    fake_creation = now - timedelta(minutes=10)

    class FakeSageMaker:
        def list_model_packages(self, **kwargs):  # noqa: ANN003
            return {
                "ModelPackageSummaryList": [
                    {
                        "ModelPackageArn": "arn:aws:sagemaker:eu-north-1:1:model-package/g/1",
                        "CreationTime": fake_creation,
                        "ModelPackageStatus": "Completed",
                    }
                ]
            }

    out = check_model.decide_deploy(
        model_group_name="g",
        region="eu-north-1",
        max_age_hours=1.0,
        now=now,
        sagemaker_client=FakeSageMaker(),
    )
    assert out["deploy_model"] == "true"
    assert out["model_package_arn"].startswith("arn:aws:sagemaker:")


def test_check_model_decide_deploy_stale_model() -> None:
    """A model older than the threshold yields deploy_model=false."""
    import sys
    from datetime import datetime, timedelta, timezone

    sys.path.insert(0, str(REPO_ROOT))
    from scripts import check_model  # type: ignore[import-not-found]

    now = datetime(2026, 4, 27, 12, 0, 0, tzinfo=timezone.utc)
    fake_creation = now - timedelta(hours=5)

    class FakeSageMaker:
        def list_model_packages(self, **kwargs):  # noqa: ANN003
            return {
                "ModelPackageSummaryList": [
                    {
                        "ModelPackageArn": "arn:aws:sagemaker:eu-north-1:1:model-package/g/2",
                        "CreationTime": fake_creation,
                        "ModelPackageStatus": "Completed",
                    }
                ]
            }

    out = check_model.decide_deploy(
        model_group_name="g",
        region="eu-north-1",
        max_age_hours=1.0,
        now=now,
        sagemaker_client=FakeSageMaker(),
    )
    assert out["deploy_model"] == "false"


def test_check_model_decide_deploy_empty_registry() -> None:
    """No approved models means deploy_model=false and arn empty."""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from scripts import check_model  # type: ignore[import-not-found]

    class FakeSageMaker:
        def list_model_packages(self, **kwargs):  # noqa: ANN003
            return {"ModelPackageSummaryList": []}

    out = check_model.decide_deploy(
        model_group_name="g",
        region="eu-north-1",
        max_age_hours=1.0,
        sagemaker_client=FakeSageMaker(),
    )
    assert out["deploy_model"] == "false"
    assert out["model_package_arn"] == ""
