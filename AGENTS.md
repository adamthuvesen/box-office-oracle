# AGENTS.md

Conventions for AI coding agents (Claude, Codex, Cursor, etc.) in this repo. Human overview: [README.md](README.md).

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
