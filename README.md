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

**Scored on the 13-feature production model, in the data-rich pre-COVID interior (2015–2019) it beats
the budget-only baseline by +0.54 to +0.63 in dollar-space R² — model 0.76–0.82 vs the baseline's
0.17–0.22 — and it clears the baseline in every valid backtested year on the log scale it is trained
against (+0.27 to +0.52 R²) and on rank correlation (Spearman 0.74–0.91 vs 0.62–0.78).** Budget alone
is a weak predictor; the lift over it is what the other 12 features buy.

Per-year expanding-window backtest (train on `<Y`, score year `Y`, walking forward). Each fold is
scored against a log-budget baseline (revenue ≈ a · budget^b) fit on the same window — the floor any
feature beyond budget has to clear. Log-space R² matches the training objective and is robust to the
heavy revenue tail; rank correlation (ρ) measures ordering quality; dollar-space R² and median APE
show absolute calibration.

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

**2020 is held out of the headline as a regime break, not a model result.** COVID theater closures
severed the budget→revenue relationship market-wide, so *no* budget-based predictor can model that
year — the log-budget baseline fails too (dollar-space R² 0.11, log-space R² −0.39). The fold is still
computed and committed for completeness: model log-space R² 0.013, dollar-space R² −0.29, median APE
179%, rank ρ 0.79. The model still *orders* 2020 films correctly (ρ 0.79 vs the baseline's 0.55); it
just cannot anticipate the market-wide collapse in the level. The full nine-year table (2020
included) is in [`results/per_year_table.md`](results/per_year_table.md) and
[`results/per_year_table.json`](results/per_year_table.json).

Generated offline from the local training snapshot by [`scripts/run_backtest.py`](scripts/run_backtest.py)
(no SageMaker, no AWS round-trip). It scores the 13-feature production model — the v3 contract from
[Features](#features): `votes`, `production_budget`, `ad_to_prod_ratio`, `franchise_rating`,
`release_year`, `mpaa_encoded`, `company_freq`, `genre_action`, `genre_comedy`, `super_genre_encoded`,
`is_covid_era`, `is_july_4th_weekend`, `is_weekend_release` — and drops the 6 snapshot rows where
`production_budget` was imputed as a fixed 0.4 × `worldwide_gross` (a budget-column artifact) before
scoring. The snapshot CSVs are gitignored, so only the derived table — not the data — is committed.

Where budget alone is weakest and the model adds the most: established franchises, summer tentpoles,
and hype-driven releases in the data-rich interior. Hardest cases: low-budget breakouts (*Get Out*,
*Everything Everywhere All at Once*) and foreign-language crossovers — the COVID-2020 demand collapse
is the regime break noted above, not a typical miss.

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
