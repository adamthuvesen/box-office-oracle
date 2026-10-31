"""Repository checkov policies for the IAM and S3 posture.

Run by `make slow` and the `infra-lint` CI job:
checkov --framework terraform --external-checks-dir infrastructure/checkov --check CKV_BO_*
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from typing import Any

from checkov.common.models.enums import CheckCategories, CheckResult
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck

_POLICY_RESOURCES = ["aws_iam_policy", "aws_iam_role_policy", "aws_iam_user_policy"]
_JSONENCODE = re.compile(r"\$\{jsonencode\((?P<document>.*)\)\}", re.DOTALL)

IAM_MUTATION_ACTIONS = frozenset(
    {
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
    }
)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _statements(conf: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The policy's statements. A root module keeps ``jsonencode`` unevaluated."""
    document = conf.get("policy", [None])[0]
    if isinstance(document, str):
        match = _JSONENCODE.fullmatch(document)
        if match is None:
            raise ValueError(f"cannot read IAM policy document: {document[:80]}")
        document = ast.literal_eval(match.group("document"))
    if not isinstance(document, dict):
        raise ValueError(f"cannot read IAM policy document: {document!r}")
    yield from _as_list(document.get("Statement", []))


class NoSageMakerFullAccess(BaseResourceCheck):
    def __init__(self) -> None:
        super().__init__(
            name="No role attaches AmazonSageMakerFullAccess",
            id="CKV_BO_1",
            categories=[CheckCategories.IAM],
            supported_resources=["aws_iam_role_policy_attachment", "aws_iam_policy_attachment"],
        )

    def scan_resource_conf(self, conf: dict[str, Any]) -> CheckResult:
        arn = str(conf.get("policy_arn", [""])[0])
        return CheckResult.FAILED if "AmazonSageMakerFullAccess" in arn else CheckResult.PASSED


class NoIamMutationActions(BaseResourceCheck):
    def __init__(self) -> None:
        super().__init__(
            name="No policy grants IAM mutation actions; IAM changes need an elevated apply",
            id="CKV_BO_2",
            categories=[CheckCategories.IAM],
            supported_resources=_POLICY_RESOURCES,
        )

    def scan_resource_conf(self, conf: dict[str, Any]) -> CheckResult:
        for statement in _statements(conf):
            if IAM_MUTATION_ACTIONS & set(_as_list(statement.get("Action", []))):
                return CheckResult.FAILED
        return CheckResult.PASSED


class PassRoleIsScoped(BaseResourceCheck):
    def __init__(self) -> None:
        super().__init__(
            name="iam:PassRole names a role ARN, never Resource '*'",
            id="CKV_BO_3",
            categories=[CheckCategories.IAM],
            supported_resources=_POLICY_RESOURCES,
        )

    def scan_resource_conf(self, conf: dict[str, Any]) -> CheckResult:
        for statement in _statements(conf):
            if "iam:PassRole" in _as_list(statement.get("Action", [])) and "*" in _as_list(
                statement.get("Resource", [])
            ):
                return CheckResult.FAILED
        return CheckResult.PASSED


class NoncurrentVersionsExpireWithin90Days(BaseResourceCheck):
    def __init__(self) -> None:
        super().__init__(
            name="S3 lifecycle expires noncurrent versions within 90 days",
            id="CKV_BO_4",
            categories=[CheckCategories.GENERAL_SECURITY],
            supported_resources=["aws_s3_bucket_lifecycle_configuration"],
        )

    def scan_resource_conf(self, conf: dict[str, Any]) -> CheckResult:
        days = [
            expiration.get("noncurrent_days", [None])[0]
            for rule in conf.get("rule", [])
            for expiration in rule.get("noncurrent_version_expiration", [])
        ]
        if days and all(isinstance(d, int) and d <= 90 for d in days):
            return CheckResult.PASSED
        return CheckResult.FAILED


check_full_access = NoSageMakerFullAccess()
check_mutation = NoIamMutationActions()
check_pass_role = PassRoleIsScoped()
check_lifecycle = NoncurrentVersionsExpireWithin90Days()
