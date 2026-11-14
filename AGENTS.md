# AGENTS.md

**box-office-oracle** predicts movie box-office revenue: Snowflake and dbt data, an XGBoost model on the v9 feature contract, SageMaker Model Registry, Lambda serving, Terraform infra, and a Next.js app. Python 3.14 via **uv**; Node 24 for `web/`. Human overview: [README.md](README.md). Terms: [GLOSSARY.md](GLOSSARY.md). Decisions: [docs/adr/](docs/adr/).

## Commands

```bash
make install-dev           # setup: dev, inference, and train extras
make web-install           # setup: web dependencies
make check                 # lint, format, types, import contracts, fast tests, web lanes; CI runs the same
make fix                   # apply safe fixes and regenerate the web contract
make verify ARGS=all       # drive the app on synthetic fixtures (ARGS=list shows the drives)
make slow                  # full suite, infra linters (actionlint, checkov), dbt parse, web build
```

Prove user-facing changes with the `verify-box-office-oracle` skill in `.agents/skills/verify-box-office-oracle`.

## Layout

- `box_office/`: Python package (`ingestion/`, `ml/`, `orchestration/`, `sagemaker/`, `inference/`, `utils/`)
- `transformations/`: dbt staging models
- `infrastructure/`: Terraform, Docker images, checkov policies
- `scripts/`: loaders, local training, scoring, registry ([scripts/README.md](scripts/README.md))
- `tests/`: hermetic; subdirectories mirror `box_office/`
- `web/`: Next.js app ([web/AGENTS.md](web/AGENTS.md))
- `docs/`: [architecture](docs/architecture.md), [development](docs/development.md), [training image](docs/training-image.md). `docs/internal/` is git-ignored and local only.

## Enforced Rules

| Rule | Enforced by |
| --- | --- |
| Spending money or writing to production refuses without approval | `box_office/utils/approval.py::require_approval`, `tests/orchestration/test_spend_gates.py::test_training_job_refuses_without_approval` |
| No module starts a process | `pyproject.toml::Process modules are banned everywhere` |
| Only the ingestion owners use network clients | `pyproject.toml::Network clients only in the ingestion owners` |
| Only the owner modules import cloud and warehouse SDKs | `pyproject.toml::Cloud and warehouse SDKs only in their owner modules` |
| Module layers import in one direction | `pyproject.toml::Layers: config, utils, ml, then sagemaker and inference, then orchestration` |
| The Lambda path does not import the registry writer | `pyproject.toml::The Lambda path does not import the model registry writer` |
| Only `box_office/ml/target.py` converts the log target | `pyproject.toml::numpy.expm1` |
| Tuning and CV never score the confirmation years | `box_office/eval_windows.py::refuse_confirmation_years`, `tests/ml/test_eval_windows.py::test_cv_refuses_a_window_that_reaches_the_confirmation_set` |
| Serving scores match training scores | `tests/inference/test_train_serve_parity.py::test_engine_frame_scores_match_the_training_path` |
| Features never correlate with the target or carry post-release names | `tests/ml/test_feature_leakage_guard.py::test_no_feature_correlates_perfectly_with_target` |
| The runtime refuses a model with a different feature contract version | `box_office/ml/feature_schema.py::CURRENT_FEATURE_SCHEMA_VERSION`, `tests/inference/test_schema_version_rejection.py::test_rejects_artifact_with_old_schema_version` |
| A model records which training data it saw | `box_office/ml/data_fingerprint.py::DataFingerprint`, `tests/ml/test_data_fingerprint.py::test_any_change_to_the_rows_changes_the_hash` |
| The web predict contract is generated from the Pydantic models | `tests/scripts/test_generate_web_contract.py::test_committed_web_contract_is_current` |
| Snowflake identifiers are validated before use in SQL | `box_office/utils/snowflake_loader.py::validate_sql_identifier`, `tests/utils/test_snowflake_loader.py::test_validate_sql_identifier_rejects_unsafe_names` |
| Terraform IAM stays scoped (no SageMaker full access, no IAM mutation, scoped PassRole) | `infrastructure/checkov/box_office_iam_checks.py::NoSageMakerFullAccess`, `infrastructure/checkov/box_office_iam_checks.py::NoIamMutationActions`, `infrastructure/checkov/box_office_iam_checks.py::PassRoleIsScoped` |
| Env files, keys, and certificates stay out of git | `.gitignore::.env`, `.gitignore::*.pem` |

## Limits code cannot enforce

- The Snowflake CLI runs read-only `SELECT`, with `LIMIT` on unknown tables. Data changes go through the committed scripts.
- Use `DBT_RUNNER` for the pipeline and dbt, `BOX_OFFICE_LOADER` for RAW loads, and `ACCOUNTADMIN` for administration only ([docs/architecture.md](docs/architecture.md)).
- Never pass the approval flag or `approved=True` on your own. The user approves.
- Put model metrics and run notes in `docs/internal/`, never in public docs.
- If a doc disagrees with the code, fix the doc in the same change.
