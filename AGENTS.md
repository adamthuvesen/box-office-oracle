# AGENTS.md

box-office-oracle is an end-to-end MLOps pipeline that predicts movie box-office revenue: XGBoost trained on SageMaker, served from Lambda, with Snowflake/dbt features and Terraform infra. Human overview: [README.md](README.md).

User-level guidance (tone, principles, git etiquette) lives in `~/.claude/CLAUDE.md` and `~/dotfiles/agents/AGENTS.md` and is *not* duplicated here. This file is for project-specific facts.

## Layout

```
box_office/        Python package: ingestion, training, inference (serving)
transformations/   dbt project (Snowflake staging + ML feature models)
infrastructure/    Terraform (SageMaker, Lambda, IAM, S3)
tests/             Pytest suite mirroring box_office/
docs/              Deeper docs — see Read next
```

## Quickstart

```bash
make install-dev                                  # uv sync --extra dev
uv run pytest tests/ box_office/inference/tests/  # CI gate (excludes integration/slow)
make lint                                         # flake8 + mypy
```

## Critical rules

- **Read [docs/architecture.md](docs/architecture.md) first** before non-trivial changes — Snowflake / dbt / Prefect / SageMaker / Lambda are tightly coupled and not obvious from one file.
- **Never run destructive Snowflake commands** via CLI: no `ALTER`, `CREATE`, `DROP`, `TRUNCATE`, `INSERT`, `UPDATE`, `DELETE`, `MERGE`, `GRANT`, `REVOKE`. Use the Snowflake UI for data changes. CLI: read-only `SELECT` only, with `LIMIT` on unknown tables.
- **Never commit secrets** — `.env`, `keys/`, `*.pem` / `*.p8`. CI writes secrets to `${RUNNER_TEMP}` only; never to `./keys/` in the workspace.
- **Do not push, open PRs, force-push, use `--no-verify`, or skip hooks** without explicit user permission.
- **Do not grant IAM mutations from CI.** `github_actions_role` is restricted; IAM changes need a manual elevated `terraform apply` from a developer machine.

## Read next

| Topic | Doc |
| --- | --- |
| Architecture, data flow, security | [docs/architecture.md](docs/architecture.md) |
| Snowflake env, tests, commands, style, git, subagents | [docs/development.md](docs/development.md) |
| Index of docs | [docs/README.md](docs/README.md) |

If a doc disagrees with code, fix the doc in the same change.
