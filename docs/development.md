# Development playbook

Rules, commands, and layout: [AGENTS.md](../AGENTS.md). Architecture: [architecture.md](architecture.md). Terms: [GLOSSARY.md](../GLOSSARY.md).

## Snowflake auth

Local dev uses key-pair auth. Set in `.env`:

```
SNOWFLAKE_USER, SNOWFLAKE_ACCOUNT, SNOWFLAKE_DATABASE, SNOWFLAKE_WAREHOUSE
SNOWFLAKE_PRIVATE_KEY_PATH, SNOWFLAKE_PRIVATE_KEY_PASSPHRASE (optional)
SNOWFLAKE_ROLE=DBT_RUNNER
```

`SNOWFLAKE_ROLE` is required and must NOT be `ACCOUNTADMIN`. The dbt profile resolves it via `env_var('SNOWFLAKE_ROLE')` and fails fast if unset.

### Which role runs what

| Task                                       | `SNOWFLAKE_ROLE`    |
| ------------------------------------------ | ------------------- |
| Pipeline (`box-office-pipeline`) + dbt     | `DBT_RUNNER`        |
| RAW dataset load (`load_dataset_to_snowflake.py`) | `BOX_OFFICE_LOADER` |
| Role/permission administration             | `ACCOUNTADMIN`      |

`DBT_RUNNER` owns `STAGING` (read-only on `RAW`); `BOX_OFFICE_LOADER` owns `RAW`. Keep `.env` set to `SNOWFLAKE_ROLE=DBT_RUNNER` for normal work; only override it inline when loading RAW:

```bash
SNOWFLAKE_ROLE=BOX_OFFICE_LOADER uv run python scripts/load_dataset_to_snowflake.py
```

If a schema/table ever ends up owned by the wrong role (symptom: a runtime role can't recreate an object an old `ACCOUNTADMIN` run created), re-apply the idempotent reconciliation:

```bash
uv run python scripts/apply_snowflake_grants.py   # connects as ACCOUNTADMIN
```

Every command that spends money or writes to production refuses until the caller passes the approval flag: `box-office-pipeline`, `load_dataset_to_snowflake.py`, `apply_snowflake_grants.py`, `register_local_model.py`, and `make training-image`. The refusal message names the flag. Agents never pass it. CI passes it only in the manual `ml-pipeline.yml` run. `--dry-run` prints the plan and needs no approval. In code, the same gate is the `approved` argument of the function that spends (`box_office/utils/approval.py`).

## Test discipline

- Tests live in [`tests/`](../tests), in subdirectories that mirror the `box_office/` package layout (plus `tests/scripts/` for `scripts/` and `tests/infrastructure/` for Terraform/CI posture). Name `test_*.py`.
- **Never** mutate `sys.modules` at module import. Real `boto3` / `snowflake-connector-python` / `sagemaker` import cleanly without contacting AWS — no mocking layer needed.
- **Never** wrap a unit-under-test in `try/except: pass` and pass regardless. Use `pytest.raises` or assert a concrete value.
- Random fixtures must be seeded (`np.random.default_rng(42)`).
- Use `monkeypatch` for env-var changes; `tests/conftest.py` already restores host env at session end.

Pull requests run `make check` first. Then separate CI jobs run the infra linters (actionlint, checkov), the lean inference tests, the suite with coverage, the dbt tests, and the web build. `make slow` runs the same lanes locally.

Run one test file:

```bash
uv run pytest tests/orchestration/test_pipeline_failure_semantics.py -v
```

## Commands

Python 3.14 (`.python-version`) and Node 24 (`.nvmrc`). Install uv and pnpm first.

```bash
make install-dev      # dev, inference, and train extras (locked)
make web-install      # web dependencies from the lockfile
make check            # the fast gate: ruff, basedpyright, import contracts, fast tests, web lint/format/types/tests
make fix              # ruff --fix, ruff format, regenerate the web contract, oxfmt
make verify ARGS=all  # drive the app on synthetic fixtures; ARGS=list shows the drives
make slow             # full suite, actionlint, checkov, dbt parse, web build
make test-inference-lean  # inference tests with only the serving dependencies

box-office-pipeline   # three-phase Prefect flow (refuses without the approval flag)

uv run uvicorn box_office.inference.app.main:app --reload   # the inference API locally

cd transformations && uv run dbt build --select staging --profiles-dir=.
```

The `train` extra holds Prefect, dbt, SageMaker, and Snowflake. The Lambda image installs only `.[inference]`, and `test-inference-lean` checks that closure.
