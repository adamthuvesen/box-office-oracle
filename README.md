# Box Office Oracle

An end-to-end MLOps pipeline predicting worldwide box-office revenue from TMDB and IMDb data. The model is an XGBoost regressor trained on ~2,400 movies.

## Architecture

Data flows from Snowflake through dbt into a scikit-learn feature pipeline, training an XGBoost model on SageMaker. Artifacts are tracked in SageMaker Model Registry and served via an AWS Lambda FastAPI container. Prefect orchestrates the runs; Terraform manages the infrastructure.

| Layer | Tech |
| --- | --- |
| **Warehouse** | Snowflake (`RAW` → `STAGING` → `ML_TRAINING`) |
| **Transform** | dbt-core + dbt-snowflake |
| **Features** | scikit-learn `Pipeline`: 12 raw columns → ~66 engineered → 12 selected (max pairwise \|Spearman\|=0.57) |
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
then selects 12. The v3 selection was importance-ranked greedy with a Spearman ceiling
(`analysis/feature_selection_study.py`); the v4 contract removes `IS_COVID_ERA`
after a same-split challenger backtest improved log-scale validation and 2023 holdout
R² with shallower XGBoost trees. Revenue is log-transformed before training.

An earlier version leaned on a `social_media_buzz` feature synthesized from the target and
a budget imputation conditioned on revenue — both leaks. Removing them bumped the schema to
`v2`; the curated slim was `v3`; the depth-3/drop-COVID contract is `v4`.
`tests/test_feature_leakage_guard.py` now fails the build if any feature correlates
>0.99 with the log-target on synthetic data.

## Evaluation

The 12-feature production model beats a budget-only baseline in every valid yearly fold. In the data-rich 2015-2019 window, dollar-space R² lands at `0.72-0.80` versus the baseline's `0.17-0.22`; on the log scale used for training, the yearly gain is `+0.30` to `+0.52` R².

The committed result table is the visual proof for public review: [`results/per_year_table.md`](results/per_year_table.md). The challenger decision is documented in [`results/drop_covid_challenger_comparison.md`](results/drop_covid_challenger_comparison.md). These are artifacts, not promises that the private training run can be reproduced without the production data and services.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 0.797 | 0.502 | +0.296 | 0.866 | 0.779 | 0.724 | 31.5% |
| 2016 | 119 | 0.844 | 0.423 | +0.422 | 0.915 | 0.734 | 0.799 | 26.7% |
| 2017 | 108 | 0.810 | 0.431 | +0.380 | 0.894 | 0.711 | 0.785 | 33.4% |
| 2018 | 111 | 0.782 | 0.349 | +0.433 | 0.864 | 0.652 | 0.775 | 27.3% |
| 2019 | 104 | 0.823 | 0.531 | +0.292 | 0.889 | 0.741 | 0.752 | 28.0% |
| 2021 | 92 | 0.683 | 0.121 | +0.562 | 0.800 | 0.615 | 0.529 | 35.4% |
| 2022 | 96 | 0.650 | 0.326 | +0.323 | 0.774 | 0.624 | 0.503 | 50.0% |
| 2023 | 117 | 0.711 | 0.319 | +0.392 | 0.865 | 0.653 | 0.614 | 52.4% |

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
pins `v4` (the 12-feature contract). A model trained against an older feature width is
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
