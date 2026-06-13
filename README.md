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

**On 2019 releases the model scores 0.78 dollar-space R² against a 0.21 log-budget baseline —
a +0.57 R² lift from non-budget features — and the gain holds at +0.55 to +0.62 per year
across the data-rich 2015–2019 interior.** COVID-2020 is the honest exception: the model goes
*negative* (−0.42 R², 200% median APE) on a 56-film year the training window never anticipated,
and the 2021–2023 recovery trails off (+0.19 to +0.33 gain). The table is per-year on purpose —
a single aggregate R² would paper over that regime shift, which is exactly the kind of leaky
"0.70–0.85" headline this repo retired.

Per-year expanding-window backtest (train on `<2015`, score 2015; then `<2016`, score 2016;
and so on through 2023). Every fold is compared against a log-budget baseline (revenue ≈ a · budget^b)
fit on the same window — that baseline is the floor, so any lift has to come from features other
than budget. Folds report dollar-space R² and median APE, not just the log-space loss.

| Year | n | Baseline R² | Model R² | Gain | Model RMSLE | Median APE |
|---|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 0.215 | 0.769 | +0.553 | 0.470 | 30.9% |
| 2016 | 119 | 0.201 | 0.812 | +0.611 | 0.410 | 28.9% |
| 2017 | 108 | 0.174 | 0.797 | +0.623 | 0.509 | 31.4% |
| 2018 | 111 | 0.189 | 0.742 | +0.553 | 0.476 | 31.8% |
| 2019 | 104 | 0.211 | 0.779 | +0.568 | 0.494 | 26.5% |
| 2020 | 56 | 0.114 | -0.425 | -0.539 | 1.180 | 200.2% |
| 2021 | 92 | 0.297 | 0.620 | +0.323 | 0.771 | 42.8% |
| 2022 | 96 | 0.352 | 0.679 | +0.326 | 0.845 | 53.5% |
| 2023 | 117 | 0.315 | 0.506 | +0.191 | 0.858 | 52.9% |

Generated offline from the local training snapshot by [`scripts/run_backtest.py`](scripts/run_backtest.py)
(no SageMaker, no AWS round-trip; numbers also in [`results/per_year_table.json`](results/per_year_table.json)).
That snapshot predates the v2 leakage fix, so the driver re-applies **both** leak controls before
scoring: it drops the target-synthesized `social_media_buzz` feature family (`social_media_buzz`,
`viral_potential`, `social_buzz_to_budget`, `buzz_to_votes_ratio`, `marketing_efficiency`) and the
6 rows carrying the `production_budget = 0.4 · worldwide_gross` imputation signature. Scores are for
the de-leaked snapshot feature set; the deployed model further slims it to the 13-feature v3 contract
(within ~1% backtest R², see [Features](#features)). The snapshot CSVs are gitignored, so only the
derived table — not the data — is committed.

Where it holds up: established franchises, summer tentpoles, hype-driven releases — the
data-rich pre-COVID interior. Where it doesn't: the COVID-2020 shock (negative R² above), low-budget
breakouts (*Get Out*, *Everything Everywhere All at Once*), and foreign-language crossovers.

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
