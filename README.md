# Box Office Oracle

An end-to-end MLOps pipeline predicting worldwide box-office revenue from TMDB and IMDb data. The model is an XGBoost regressor trained on ~2,400 movies.

## Architecture

Data flows from Snowflake through dbt into a scikit-learn feature pipeline, training an XGBoost model on SageMaker. Artifacts are tracked in SageMaker Model Registry and served via an AWS Lambda FastAPI container. Prefect orchestrates the runs; Terraform manages the infrastructure.

| Layer | Tech |
| --- | --- |
| **Warehouse** | Snowflake (`RAW` → `STAGING` → `ML_TRAINING`) |
| **Transform** | dbt-core + dbt-snowflake |
| **Features** | scikit-learn `Pipeline` (69 features from 12 raw columns) |
| **Training** | XGBoost on SageMaker `ml.m5.large` |
| **Registry** | SageMaker Model Registry (SHA256-verified) |
| **Serving** | AWS Lambda (FastAPI + Mangum) |
| **Orchestration** | Prefect |
| **Infrastructure**| Terraform |
| **CI/CD** | GitHub Actions (OIDC to AWS, key-pair to Snowflake) |

**Scale:** Offline jobs assume Snowflake result sets and pandas DataFrames fit in memory (a few thousand rows). Scaling up would require chunked reads, unload-to-S3, or Snowpark—not larger `fetch_pandas_all()` pulls. The inference API runs `predict` in a worker thread to keep FastAPI responsive under concurrency.

## Quick Start

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

## Evaluation

We evaluate using a **per-year expanding-window backtest**. The model's performance is compared side-by-side with a simple log-budget baseline (revenue ≈ a · budget^b). The baseline is the honest floor—any real lift must come from features other than budget.

Generate the backtest report:

```bash
python -m box_office.ml.backtest_report \
  --raw-data data/box_office_raw.parquet \
  --cv-results artifacts/per_year_model_metrics.json \
  --output artifacts/per_year_table
```

### Reality Check

- **What works:** Established franchises, summer tentpoles, and films with massive pre-release hype. The model interpolates well in data-rich regions.
- **What fails:** Low-budget breakout hits (*Get Out*, *Everything Everywhere All at Once*), foreign-language crossovers, and post-2020 COVID anomalies. The baseline often beats the model here.
- **No causal claims:** Features like `RATING_VOTES_INTERACTION` correlate with revenue but don't cause it. This is an estimation tool, not a causal inference engine.

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

Artifacts carry a `feature_schema_version`. The current runtime requires `v2` (which drops leaky social-buzz features and uses real CPI for inflation adjustment). **`v1` artifacts will not load**—the inference container will throw a `FeatureSchemaVersionMismatch` on cold start. To serve a `v1` model, you must retrain.

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
- **`agents/docs/architecture.md`**: Architecture deep-dive
- **`AGENTS.md`**: AI coding agent conventions

## Tests

```bash
uv run pytest
```

The test suite is hermetic. A leakage guard (`tests/test_feature_leakage_guard.py`) automatically fails any feature with a >0.99 correlation to the log-target on synthetic data.

## Data Sources & Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.
Movie metadata is sourced from [The Movie Database (TMDB)](https://www.themoviedb.org/).
Box office figures are sourced from public Box Office Mojo data.
