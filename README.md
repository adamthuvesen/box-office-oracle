# Box Office Oracle

An end-to-end MLOps pipeline predicting worldwide box-office revenue from TMDB and IMDb data. The model is an XGBoost regressor trained on ~2,400 movies.

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

Public demo boundary:

- The production pipeline needs private or external services: Snowflake, AWS SageMaker/S3/Lambda, Prefect, and TMDB credentials.
- The checked-in repo does not include the private warehouse snapshot, trained model artifacts, or cloud resources.
- Public users can run the no-secret smoke path below and inspect the committed backtest result table.
- Do not treat AWS, Snowflake, TMDB, or headline metric reproduction as public unless you have the same services and data.

```bash
make install-dev
make test
```

The smoke path is local and hermetic. It does not call AWS, Snowflake, or TMDB.

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

The pre-release model beats a budget-only baseline on log R² in every
non-2020 fold. In the data-rich 2015-2019 window, dollar-space R² lands at
`0.54-0.63`; on the log scale used for training, the yearly gain is `+0.12` to
`+0.29` R².

The committed result table is the visual proof for public review: [`results/per_year_table.md`](results/per_year_table.md). It is an artifact, not a promise that the private training run can be reproduced without the production data and services.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 0.671 | 0.502 | +0.169 | 0.781 | 0.779 | 0.541 | 40.7% |
| 2016 | 119 | 0.634 | 0.423 | +0.211 | 0.790 | 0.734 | 0.587 | 40.4% |
| 2017 | 108 | 0.597 | 0.431 | +0.166 | 0.744 | 0.711 | 0.596 | 46.4% |
| 2018 | 111 | 0.640 | 0.349 | +0.291 | 0.765 | 0.652 | 0.635 | 38.9% |
| 2019 | 104 | 0.650 | 0.531 | +0.119 | 0.777 | 0.741 | 0.536 | 45.2% |
| 2021 | 92 | 0.450 | 0.121 | +0.329 | 0.588 | 0.615 | 0.262 | 63.0% |
| 2022 | 96 | 0.540 | 0.326 | +0.214 | 0.670 | 0.624 | 0.395 | 66.1% |
| 2023 | 117 | 0.599 | 0.319 | +0.280 | 0.763 | 0.653 | 0.296 | 62.6% |

COVID-2020 is excluded from the headline. The full table, including 2020, is committed in [`results/per_year_table.md`](results/per_year_table.md) and [`results/per_year_table.json`](results/per_year_table.json).

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

*Note: `SNOWFLAKE_ROLE` must be a least-privilege role (e.g., `DBT_RUNNER`), not `ACCOUNTADMIN`.*

## Schema Versioning

Every artifact carries a `feature_schema_version` in its registry metadata; the runtime
pins `v9` (the 13-feature pre-release contract). A model with a different feature width is
rejected at cold start with `FeatureSchemaVersionMismatch` rather than served with a silent
shape mismatch; retraining is required. The same load path verifies a SHA256
manifest against the artifact tarball *before* unpickling, so a bucket write can't turn into
remote code execution.

## Local Data

CSVs and Parquet snapshots under `analysis/datasets_*/` are git-ignored. Pull fresh data from Snowflake:

```bash
make datasets
```

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
