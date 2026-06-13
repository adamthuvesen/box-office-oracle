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

**Across all nine backtested years (2015–2023) the model beats the log-budget baseline on the
metric it is trained against — R² on `log1p(worldwide_gross)` — by +0.29 to +0.53 per year, and it
ranks films more accurately than budget alone in every year (Spearman 0.75–0.92 vs 0.55–0.78), the
COVID-2020 shutdown included.** In normal markets that lift also lands in dollar-space R² (0.51–0.81,
beating the baseline by +0.19 to +0.62). Dollar-space R² goes negative only in COVID-2020, where a
~3× market-wide revenue collapse — absent from any pre-2020 training window — breaks absolute
calibration even though the model still orders that year's films correctly (ρ=0.83 vs the baseline's
0.55). That column stays in the table rather than being dropped.

Per-year expanding-window backtest (train on `<2015`, score 2015; then `<2016`, score 2016; and so on
through 2023). Each fold is scored against a log-budget baseline (revenue ≈ a · budget^b) fit on the
same window — the floor any feature beyond budget has to clear. Log-space R² leads because it matches
the training objective and is robust to the heavy revenue tail; rank correlation (ρ) measures ordering
quality; dollar-space R² and median APE show absolute calibration.

| Year | n | Model R² (log) | Baseline R² (log) | Gain (log) | Model ρ | Baseline ρ | Model R² ($) | Median APE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 110 | 0.803 | 0.502 | +0.301 | 0.856 | 0.779 | 0.769 | 30.9% |
| 2016 | 119 | 0.836 | 0.423 | +0.413 | 0.918 | 0.734 | 0.812 | 28.9% |
| 2017 | 108 | 0.804 | 0.431 | +0.373 | 0.893 | 0.711 | 0.797 | 31.4% |
| 2018 | 111 | 0.800 | 0.349 | +0.451 | 0.875 | 0.652 | 0.742 | 31.8% |
| 2019 | 104 | 0.817 | 0.531 | +0.286 | 0.869 | 0.741 | 0.779 | 26.5% |
| 2020 | 56 | 0.040 | -0.385 | +0.425 | 0.827 | 0.555 | -0.425 | 200.2% |
| 2021 | 92 | 0.646 | 0.121 | +0.525 | 0.748 | 0.615 | 0.620 | 42.8% |
| 2022 | 96 | 0.662 | 0.326 | +0.336 | 0.780 | 0.624 | 0.679 | 53.5% |
| 2023 | 117 | 0.655 | 0.319 | +0.336 | 0.839 | 0.653 | 0.506 | 52.9% |

Generated offline from the local training snapshot by [`scripts/run_backtest.py`](scripts/run_backtest.py)
(no SageMaker, no AWS round-trip; full per-metric numbers in [`results/per_year_table.json`](results/per_year_table.json)).
The snapshot predates the v2 leakage fix, so the driver re-applies both controls from that fix before
scoring: it drops the target-synthesized `social_media_buzz` feature family (`social_media_buzz`,
`viral_potential`, `social_buzz_to_budget`, `buzz_to_votes_ratio`, `marketing_efficiency`) and the
6 rows carrying the `production_budget = 0.4 · worldwide_gross` imputation signature. Scores are for
the de-leaked snapshot feature set; the deployed model further slims it to the 13-feature v3 contract
(within ~1% backtest R², see [Features](#features)). The snapshot CSVs are gitignored, so only the
derived table — not the data — is committed.

Where budget alone is weakest and the model adds the most: established franchises, summer tentpoles,
and hype-driven releases in the data-rich interior. Hardest cases: the COVID-2020 demand collapse
(dollar-space calibration above), low-budget breakouts (*Get Out*, *Everything Everywhere All at
Once*), and foreign-language crossovers.

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
