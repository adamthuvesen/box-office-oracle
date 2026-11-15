#!/usr/bin/env bash
# Boundary lane of `make check`: the import contracts on the real tree, then a
# proof that each detector still fires on a planted violation. A detector that
# is disabled or emptied fails here.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
bin="$root/.venv/bin"
cd "$root"

"$bin/lint-imports"

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
rsync -a --exclude __pycache__ pyproject.toml box_office scripts tests "$scratch/"

cat >"$scratch/box_office/ml/planted_layer_violation.py" <<'PY'
from box_office.inference.app import main  # noqa: F401
PY
cat >"$scratch/box_office/utils/planted_module_violation.py" <<'PY'
import boto3  # noqa: F401
import requests  # noqa: F401
import subprocess  # noqa: F401
PY
cat >"$scratch/box_office/inference/planted_registry_violation.py" <<'PY'
from box_office.ml.model_registry import aws_model_registry  # noqa: F401
PY
cat >"$scratch/tests/inference/planted_train_only_import.py" <<'PY'
import requests  # noqa: F401
import sagemaker  # noqa: F401
from box_office.utils.snowflake_connection import create_snowflake_connection  # noqa: F401
from tests.fakes import FakeSageMakerClient  # noqa: F401
PY
cat >"$scratch/box_office/utils/planted_process_call.py" <<'PY'
import asyncio
import os

os.system("true")
asyncio.create_subprocess_exec("true")
PY

contracts="$(grep -c '^\[\[tool.importlinter.contracts\]\]' pyproject.toml)"
cd "$scratch"

contracts_output="$("$bin/lint-imports" 2>&1 || true)"
broken="$(grep -c ' BROKEN' <<<"$contracts_output" || true)"
if [ "$broken" != "$contracts" ]; then
  echo "FAIL  a planted violation did not break every import contract ($broken of $contracts broke)" >&2
  echo "$contracts_output" >&2
  exit 1
fi

ruff_output="$("$bin/ruff" check --no-cache box_office 2>&1 || true)"
for planted in "os.system" "asyncio.create_subprocess_exec"; do
  if ! grep -q "TID251.*\`$planted\`" <<<"$ruff_output"; then
    echo "FAIL  ruff did not ban the planted $planted call" >&2
    echo "$ruff_output" >&2
    exit 1
  fi
done
