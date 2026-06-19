# Box Office Oracle

An end-to-end MLOps pipeline predicting worldwide box-office revenue from TMDB and IMDb data. The model is an XGBoost regressor trained on ~2,400 movies.

## Architecture

Data flows from Snowflake through dbt into a scikit-learn feature pipeline, training an XGBoost model on SageMaker. Artifacts are tracked in SageMaker Model Registry and served via an AWS Lambda FastAPI container. Prefect orchestrates the runs; Terraform manages the infrastructure.

| Layer | Tech |
| --- | --- |
| **Warehouse** | Snowflake (`RAW` → `STAGING` → `ML_TRAINING`) |
| **Transform** | dbt-core + dbt-snowflake |
| **Features** | scikit-learn `Pipeline`: 12 raw columns → ~66 engineered → 13 selected (max pairwise \|Spearman\|=0.57) |
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

The pipeline expands 12 raw columns into ~66 engineered features (temporal windows,
genre vectors, frequency-encoded industry signals, CPI-adjusted financials, interactions),
then selects 13. The selection is importance-ranked greedy with a Spearman ceiling
(`analysis/feature_selection_study.py`): the 13-feature set matches the full model within
~1% backtest R² but drops the multicollinearity. Revenue is log-transformed before training.

An earlier version leaned on a `social_media_buzz` feature synthesized from the target and
a budget imputation conditioned on revenue — both leaks. Removing them bumped the schema to
`v2`; the 13-feature slim is `v3`. `tests/test_feature_leakage_guard.py` now fails the build
if any feature correlates >0.99 with the log-target on synthetic data.

## Evaluation

The 13-feature production model beats a budget-only baseline in every valid yearly fold. In the data-rich 2015-2019 window, dollar-space R² lands at `0.76-0.82` versus the baseline's `0.17-0.22`; on the log scale used for training, the yearly gain is `+0.27` to `+0.52` R².

The committed result table is the visual proof for public review: [`results/per_year_table.md`](results/per_year_table.md). It is an artifact, not a promise that the private training run can be reproduced without the production data and services.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 0.799 | 0.502 | +0.297 | 0.861 | 0.779 | 0.755 | 33.9% |
| 2016 | 119 | 0.835 | 0.423 | +0.412 | 0.908 | 0.734 | 0.820 | 29.2% |
| 2017 | 108 | 0.811 | 0.431 | +0.380 | 0.891 | 0.711 | 0.802 | 29.2% |
| 2018 | 111 | 0.787 | 0.349 | +0.438 | 0.867 | 0.652 | 0.779 | 27.5% |
| 2019 | 104 | 0.804 | 0.531 | +0.273 | 0.859 | 0.741 | 0.778 | 33.2% |
| 2021 | 92 | 0.636 | 0.121 | +0.515 | 0.737 | 0.615 | 0.551 | 44.5% |
| 2022 | 96 | 0.662 | 0.326 | +0.336 | 0.763 | 0.624 | 0.621 | 54.4% |
| 2023 | 117 | 0.700 | 0.319 | +0.380 | 0.865 | 0.653 | 0.467 | 49.6% |

The model adds the most for franchises, summer tentpoles, and hype-driven releases. The hardest cases are low-budget breakouts and foreign-language crossovers. COVID-2020 is excluded from the headline as a market-wide regime break; the full table, including 2020, is committed in [`results/per_year_table.md`](results/per_year_table.md) and [`results/per_year_table.json`](results/per_year_table.json).

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
pins `v3` (the 13-feature contract). A model trained against an older feature width is
rejected at cold start with `FeatureSchemaVersionMismatch` rather than served with a silent
shape mismatch — to run an old model you retrain. The same load path verifies a SHA256
manifest against the artifact tarball *before* unpickling, so a bucket write can't turn into
remote code execution.

## Local Data

CSVs and Parquet snapshots under `analysis/datasets_*/` are git-ignored. Pull fresh data from Snowflake:

```bash
make datasets
```

## Project Layout

- **`box_office/orchestration/`**: Prefect flows
- **`box_office/ml/`**: Feature engineering and backtesting
- **`box_office/inference/`**: Lambda FastAPI app
- **`transformations/`**: dbt models
- **`infrastructure/terraform/`**: IaC
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
