# Spend refusal

Every entry point that spends money or writes to production refuses without `--approved`, prints the refusal token, and touches no external tool.

Source: box_office/utils/approval.py, box_office/orchestration/flows/ml_pipeline.py, scripts/load_dataset_to_snowflake.py, scripts/apply_snowflake_grants.py, scripts/register_local_model.py, scripts/build_training_image.sh
Inventory-id: box-office-pipeline, scripts/load_dataset_to_snowflake.py, scripts/apply_snowflake_grants.py, scripts/register_local_model.py, make training-image

## Sub-features

- `box-office-pipeline`, `load_dataset_to_snowflake`, `apply_snowflake_grants`, `register_local_model`, `training-image`: one per entry point.

## How to get to it (user POV)

- `uv run box-office-pipeline`, the three scripts, and `make training-image`, each run without the approval flag.

## Driving it with verify

Preconditions: none.

- **Drive.** `make verify ARGS=spend-refusal`. Each entry point exits non-zero and prints `REFUSAL_TOKEN` (imported from `box_office/utils/approval.py`). The five run in parallel. The tripwires on `PATH` (`aws`, `snow`, `terraform`, `dbt`, `docker`, `curl`) must log no call.
- **Proof.** `spend-refusal.*-<entry>.json` per entry point and a manifest block.

## Gotchas

- Python clients do not use `PATH`, so the drive also sets a dead proxy and disables the AWS metadata endpoint. The temp HOME holds no credentials.
- The pipeline refuses in the flow body, before any task runs.
