#!/usr/bin/env bash
# Pinned workflow and Terraform linters. `make slow` and the infra-lint CI job run this.
# uvx fetches each pinned tool once; later runs use the uv cache.
set -euo pipefail

cd "$(dirname "$0")/.."

# actionlint's shellcheck integration stays off: shellcheck is not pinned here.
uvx --from actionlint-py==1.7.12.25 actionlint -shellcheck=

# Repository policies in infrastructure/checkov (CKV_BO_*): IAM and S3 posture.
uvx --from checkov==3.4.4 checkov \
  --directory infrastructure/terraform \
  --framework terraform \
  --external-checks-dir infrastructure/checkov \
  --check CKV_BO_1,CKV_BO_2,CKV_BO_3,CKV_BO_4 \
  --compact --quiet
