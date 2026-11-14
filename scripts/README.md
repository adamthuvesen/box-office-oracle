# scripts/

Entry points invoked as `uv run python scripts/<name>.py`. They stay flat because tests import them as the `scripts` package; this table says what each one is for.

## Dataset pipeline (local canonical dataset → training frame)

Roughly in dependency order. Commands that spend money or write to production refuse without the approval flag; see [docs/development.md](../docs/development.md).

| Script                        | Purpose                                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| `clean_movie_source_data.py`  | Apply curated source-data cleanup to the generated local dataset |
| `refetch_collections.py`      | Refetch TMDB collection links for the cleaned source             |
| `materialize_collections.py`  | Add collection id/name columns to the canonical dataset          |
| `classify_ip.py`              | CLI wrapper for movie IP classification                          |
| `prepare_training_frame.py`   | Build the local training frame from the rich backfill            |

## Training, evaluation, registry

| Script                        | Purpose                                                             |
| ----------------------------- | ------------------------------------------------------------------- |
| `train_local.py`              | Local training driver (offline CV + final fit)                      |
| `evaluate_local_retrain.py`   | Leakage-free CV evaluation of the local retrain                     |
| `register_local_model.py`     | Register the locally-trained model in the SageMaker Model Registry  |

## Snowflake administration (sanctioned write path)

| Script                       | Purpose                                                          |
| ---------------------------- | ---------------------------------------------------------------- |
| `load_dataset_to_snowflake.py` | Replace the production RAW dataset with the local parquet      |
| `apply_snowflake_grants.py`  | Apply `snowflake_role_grants.sql` as ACCOUNTADMIN (idempotent)   |
| `snowflake_role_grants.sql`  | Role/permission reconciliation SQL, applied by the script above  |

## Web app data

| Script               | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| `score_all_movies.py` | Predict every movie in the dataset for the web app  |
| `export_web_data.py` | Write JSON snapshots to `web/data/` (gitignored)     |

Both run via `make web-data`.

## Contract and tooling

| Script                      | Purpose                                                                   |
| --------------------------- | ------------------------------------------------------------------------- |
| `generate_web_contract.py`  | Write the web predict contract from the Python models (`make fix` runs it) |
| `check.sh`                  | The fast gate behind `make check`                                         |
| `slow.sh`                   | The slow lanes behind `make slow`                                         |
| `check_boundaries.sh`       | Run the import contracts                                                  |
| `lint_infra.sh`             | Run the pinned actionlint and checkov policies                            |
| `build_training_image.sh`   | Build and push the SageMaker training image (refuses without approval; [docs/training-image.md](../docs/training-image.md)) |
| `no_emoji_in_logs.py`       | Pre-commit hook: reject emoji in `logger.*` / `print()`                   |
