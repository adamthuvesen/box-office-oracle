# AGENTS.md

box-office-oracle predicts movie box-office revenue: XGBoost (v9 feature contract), Snowflake + dbt data, SageMaker Model Registry, Lambda serving, Terraform infra. Human overview: [README.md](README.md).

This file holds project facts only. User-level guidance (tone, git etiquette) lives in the user's own agent config.

## Layout

```
box_office/        Python package: ingestion, training frame, ML, inference
transformations/   dbt (Snowflake staging)
infrastructure/    Terraform (SageMaker, Lambda, IAM, S3)
scripts/           Loaders, training, scoring, registry, grants
tests/             Pytest, hermetic (no cloud access)
web/               Next.js frontend — has its own AGENTS.md
docs/              Public docs; docs/internal/ is gitignored, local-only
```

## Quickstart

```bash
make install-dev                                  # uv sync --extra dev
uv run pytest tests/ box_office/inference/tests/  # CI gate
make lint                                         # ruff
```

Web app: `pnpm --dir web dev` (regenerate data first with `make web-data` after a retrain). Live /predict needs `INFERENCE_API_URL` + `INFERENCE_API_KEY` in `web/.env.local`; without them it runs in mock mode.

## Rules

- **Snowflake CLI is read-only `SELECT`** (with `LIMIT` on unknown tables). Data changes go through the committed scripts (`scripts/load_dataset_to_snowflake.py`, `scripts/apply_snowflake_grants.py`) or the UI — never ad-hoc write SQL.
- **Roles**: `DBT_RUNNER` for pipeline/dbt, `BOX_OFFICE_LOADER` for RAW loads, `ACCOUNTADMIN` for admin only. Role table: [docs/architecture.md](docs/architecture.md).
- **Eval discipline**: iterate against years ≤2023 only. 2024–2025 is a spent confirmation set — never tune against it.
- **No secrets in git** — `.env`, `keys/`, `*.pem`, `*.p8`. CI secrets go to `${RUNNER_TEMP}` only.
- **No push / PR / force-push / `--no-verify`** without explicit user permission.
- **No IAM mutation from CI** — `github_actions_role` is restricted; IAM changes need a manual elevated `terraform apply`.
- **No historical narration** in code, comments, or docs — describe what is, not what was.

## Read next

Load these only when the task needs them:

| Task touches…                        | Read                                                        |
| ------------------------------------ | ----------------------------------------------------------- |
| Snowflake / dbt / pipeline / serving | [docs/architecture.md](docs/architecture.md)                 |
| Tests, commands, style, git          | [docs/development.md](docs/development.md)                   |
| Model detail, experiments, runbooks  | `docs/internal/` (local-only: model.md, operations.md)       |
| Frontend                             | [web/AGENTS.md](web/AGENTS.md)                               |

If a doc disagrees with code, fix the doc in the same change.
