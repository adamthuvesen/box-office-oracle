# Box Office Oracle

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

The pre-release model beats a log-budget baseline on log R² in every fold. Over
the data-rich 2015-2019 window, dollar-space R² lands at `0.64-0.72`; on the log
scale used for training, the per-year gain over the baseline is `+0.14` to
`+0.28` R². 2020 is the COVID low point (dollar-space R² `0.46`).

The table below is the leakage-free v9 backtest on the 1980-2026 dataset:
expanding-window CV where each fold refits its preprocessor on train-year rows
only. It is an artifact, not a promise that the private training run can be
reproduced without the production data and services. Full record:
[`results/local_retrain/iteration_report.md`](results/local_retrain/iteration_report.md).

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 158 | 0.676 | 0.393 | +0.283 | 0.797 | 0.695 | 0.660 | 46.2% |
| 2016 | 181 | 0.634 | 0.497 | +0.137 | 0.781 | 0.734 | 0.669 | 50.5% |
| 2017 | 165 | 0.655 | 0.432 | +0.224 | 0.767 | 0.654 | 0.643 | 54.5% |
| 2018 | 163 | 0.659 | 0.438 | +0.221 | 0.779 | 0.694 | 0.722 | 46.5% |
| 2019 | 134 | 0.655 | 0.435 | +0.220 | 0.764 | 0.662 | 0.653 | 45.1% |
| 2020 | 53 | 0.437 | 0.174 | +0.263 | 0.684 | 0.558 | 0.457 | 60.5% |
| 2021 | 89 | 0.508 | 0.284 | +0.224 | 0.683 | 0.553 | 0.407 | 58.7% |
| 2022 | 100 | 0.583 | 0.425 | +0.157 | 0.725 | 0.670 | 0.422 | 62.3% |
| 2023 | 116 | 0.621 | 0.405 | +0.216 | 0.786 | 0.655 | 0.252 | 52.5% |

Pooled over 2015-2023 (1,159 movies): median APE 51.8%, mean log-R² 0.603.

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
