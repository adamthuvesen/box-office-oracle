# Box Office Oracle

![License](https://img.shields.io/github/license/adamthuvesen/box-office-oracle) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

An end-to-end MLOps pipeline predicting worldwide box-office revenue from TMDB and IMDb data. The model is an XGBoost regressor trained on ~6,080 movies (kept from 6,152 in the 1980-2026 dataset).

## Architecture

Data flows from Snowflake through dbt into a scikit-learn feature pipeline, training an XGBoost model on SageMaker. Artifacts are tracked in SageMaker Model Registry and served via an AWS Lambda FastAPI container. Prefect orchestrates the runs; Terraform manages the infrastructure.

| Layer | Tech |
| --- | --- |
| **Warehouse** | Snowflake (`RAW` → `STAGING` → `ML_TRAINING`) |
| **Transform** | dbt-core + dbt-snowflake |
| **Features** | scikit-learn `Pipeline`: pre-release raw columns → 52 engineered → 13 selected |
| **Training** | XGBoost on SageMaker `ml.m5.large` |
| **Registry** | SageMaker Model Registry (SHA256-verified) |
| **Serving** | AWS Lambda (FastAPI + Mangum) |
| **Orchestration** | Prefect |
| **Infrastructure**| Terraform |
| **CI/CD** | GitHub Actions (OIDC to AWS, key-pair to Snowflake) |

**Scale:** Offline jobs assume Snowflake result sets and pandas DataFrames fit in memory (a few thousand rows). Scaling up would require chunked reads, unload-to-S3, or Snowpark—not larger `fetch_pandas_all()` pulls. The inference API runs `predict` in a worker thread to keep FastAPI responsive under concurrency.

## Quick Start

The full pipeline needs private services — Snowflake, AWS, and TMDB
credentials — and the repo ships no warehouse data or model artifacts. The
test suite runs without any of them:

```bash
make install-dev
make test
```

Run the pipeline (requires AWS and Snowflake credentials in `.env`):

```bash
box-office-pipeline --environment dev --experiment-name "smoke"
```

Run the inference API locally:

```bash
uv run uvicorn box_office.inference.app.main:app --reload
```

## Features

The pipeline expands pre-release raw columns into 52 engineered features (temporal
windows, genre vectors, frequency-encoded industry signals, CPI-adjusted
financials, safe budget interactions, and pre-release IP/franchise
inputs), then selects 13 (`SELECTED_FEATURES`).
The current contract excludes outcome-derived and post-release signals such as
popularity scores, user ratings, ranking fields, and franchise scores derived
from revenue. Revenue is log-transformed before training.

Leakage guard: `tests/test_feature_leakage_guard.py` fails the build if any feature
correlates >0.99 with the log-target on synthetic data. It also blocks forbidden
post-release field names from entering the feature contract.

## Evaluation

On a leakage-free expanding-window backtest (each fold refits its preprocessor
on train-year rows only), the pre-release model beats a log-budget baseline
on log R² in every fold from 2015-2023. Pooled over 2015-2023 (1,159 movies):
median APE 51.8%, mean log-R² 0.603. Sorted into the nine revenue classes
standard in the box-office literature (flop to blockbuster), the model hits
the exact class 33% of the time and lands within one class 70% of the time,
out of sample.

## Configuration

Settings are loaded from environment variables and a `.env` file via Pydantic,
falling back to code defaults. Required variables:

```text
SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_DATABASE, SNOWFLAKE_WAREHOUSE
SNOWFLAKE_SCHEMA_{RAW,STAGING,ML_TRAINING,FEATURE_STORE}
SNOWFLAKE_PRIVATE_KEY_PATH, SNOWFLAKE_ROLE
AWS_REGION, SAGEMAKER_ROLE_ARN, AWS_S3_BUCKET
TMDB_API_TOKEN
```

`SNOWFLAKE_ROLE` is a least-privilege role (`DBT_RUNNER` for the pipeline), not `ACCOUNTADMIN`.

## Schema Versioning

Every artifact carries a `feature_schema_version` in its registry metadata; the runtime
pins `v9` (the 13-feature pre-release contract). A model with a different feature width is
rejected at cold start with `FeatureSchemaVersionMismatch` rather than served with a silent
shape mismatch; retraining is required. The same load path verifies a SHA256
manifest against the artifact tarball *before* unpickling, so a bucket write can't turn into
remote code execution.

## Web App

A local Next.js frontend under `web/` — "the screening room": a WebGL
constellation of every movie, a poster explorer, box-office statistics, the
model's backtest report card, and an interactive prediction oracle.

```bash
make web-data       # export gitignored JSON snapshots to web/data/ (local dataset, no Snowflake)
pnpm --dir web install
pnpm --dir web dev  # http://localhost:3000
```

The catalog comes from the local 1980-2026 parquet under
`data/generated/tmdb/rich_backfill_1980_2026/`; poster/backdrop paths are
already in it, so no TMDB API calls are made. Oracle predictions
(`web/data/predictions.json`) come from `scripts/score_all_movies.py`. Live
predictions on /predict need `INFERENCE_API_URL` and `INFERENCE_API_KEY` in
`web/.env.local`; without them the app runs the oracle in mock mode.

## Project Layout

- **`box_office/orchestration/`**: Prefect flows
- **`box_office/ml/`**: Feature engineering and backtesting
- **`box_office/inference/`**: Lambda FastAPI app
- **`transformations/`**: dbt models
- **`infrastructure/terraform/`**: IaC
- **`web/`**: Next.js frontend (local-only)
- **[`docs/architecture.md`](docs/architecture.md)**: Architecture deep-dive
- **[`AGENTS.md`](AGENTS.md)**: AI coding agent conventions

## Tests

```bash
uv run pytest
```

The suite is hermetic — no AWS or Snowflake access required. It includes the leakage guard
described above and an offline IAM-posture check (`tests/infrastructure/`) that parses the
Terraform and dbt profiles to assert no `ACCOUNTADMIN`, no `SageMakerFullAccess`, and no IAM
mutation from CI.

## Data Sources & Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.
Movie metadata is sourced from [The Movie Database (TMDB)](https://www.themoviedb.org/).
Box office figures are sourced from public Box Office Mojo data.
