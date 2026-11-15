# box-office-oracle features

## Baseline preconditions

- Locked environment synced (`make verify` does it) and `web/node_modules` installed.
- The temp copy comes from `git ls-files -co --exclude-standard`, so untracked files that are not ignored are included.
- Fixtures: `fixtures.py`, `SEED = 20261112`. Source row ids start at 9,000,001, above any real TMDB id.
- Doctor passes before the first drive. `verify` runs it itself.

## Driving conventions

Drives share one temp copy per run. `train-local` produces the artifacts that `serve` and `web-data` read, and each of them trains first when it runs alone. `spend-refusal` is independent.

## Inventory

`make verify ARGS=list` prints the drives. The user-facing entry points are:

- CLI scripts: `scripts/prepare_training_frame.py`, `train_local.py`, `register_local_model.py`, `score_all_movies.py`, `export_web_data.py`, `generate_web_contract.py`, `load_dataset_to_snowflake.py`, `apply_snowflake_grants.py`.
- Console script: `box-office-pipeline` (`pyproject.toml` `[project.scripts]`).
- Make target: `training-image`.
- HTTP: `GET /health`, `GET /`, `POST /predict`, `GET /model/info` in `box_office/inference/app/main.py`.
- Web data files: `web/data/movies.json`, `predictions.json`, `model_meta.json`.

Unmapped:

- `box-office-rich-backfill`, `box-office-curated-movies`, `box-office-wikipedia-budget-fill`, `scripts/refetch_collections.py`, `scripts/classify_ip.py`, `scripts/materialize_collections.py`, `scripts/clean_movie_source_data.py`, `scripts/evaluate_local_retrain.py`: need TMDB, Wikipedia, or a long-running server, or overlap with the mapped drives.
- `GET /health`, `GET /`: no behavior beyond what `POST /predict` already exercises.
- Next.js pages: only the data loaders are driven; rendering stays with the web tests.
- `infrastructure/`, `transformations/`: need AWS, Terraform, or a Snowflake account.

## Proof, skip, and flake reporting

Every drive writes a manifest block and per-command JSON in the run directory. A skipped drive FAILs. Report an unreachable prerequisite (for example a missing `web/node_modules`) as FAIL with the command attempted.

## Control

`make verify ARGS=control` negates `from_log_target` in the temp copy after a clean training, and expects `serve` to fail with `prediction-not-positive`.

## Features

| Feature | File | Inventory ids | Last verified |
| --- | --- | --- | --- |
| Local training | [train-local.md](train-local.md) | `scripts/prepare_training_frame.py`, `scripts/train_local.py`, `scripts/register_local_model.py` | |
| Serving | [serve.md](serve.md) | `POST /predict`, `GET /model/info` | |
| Web data | [web-data.md](web-data.md) | `scripts/score_all_movies.py`, `scripts/export_web_data.py`, `scripts/generate_web_contract.py` | |
| Spend refusal | [spend-refusal.md](spend-refusal.md) | `box-office-pipeline`, `scripts/load_dataset_to_snowflake.py`, `scripts/apply_snowflake_grants.py`, `scripts/register_local_model.py`, `make training-image` | |
