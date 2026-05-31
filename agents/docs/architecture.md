# Architecture

Production ML system for box office prediction. Snowflake → dbt → feature engineering → XGBoost on SageMaker → Model Registry → Lambda inference. Orchestrated by Prefect, deployed via Terraform + GitHub Actions.

## Stack


| Layer               | Tool                         | Notes                                                                                                               |
| ------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Data warehouse      | Snowflake                    | RAW → STAGING → ML_TRAINING schemas                                                                                 |
| Transformations     | dbt-core + dbt-snowflake     | `[transformations/](../../transformations/)`                                                                           |
| Feature engineering | scikit-learn + pandas        | Single `Pipeline` built by [`box_office/ml/feature_pipeline/`](../../box_office/ml/feature_pipeline/)                 |
| Training            | XGBoost on SageMaker         | `ml.m5.large`, time-series CV                                                                                       |
| Model registry      | AWS SageMaker Model Registry | manual approval gate at R² ≥ 0.75                                                                                   |
| Inference           | Lambda (container image)     | `[box_office/inference/](../../box_office/inference)`                                                                  |
| Orchestration       | Prefect                      | Three-phase flow in [`box_office/orchestration/flows/ml_pipeline.py`](../../box_office/orchestration/flows/ml_pipeline.py) |
| Infra               | Terraform                    | `[infrastructure/terraform/](../../infrastructure/terraform)`                                                          |
| CI/CD               | GitHub Actions               | OIDC → AWS, key-pair → Snowflake                                                                                    |


Operational cost: ~$50/month. Training run: 2-5 min on ~2,400 movies. The per-year R² table is produced by the expanding-window backtest (see [`box_office/ml/backtest.py`](../../box_office/ml/backtest.py)) — never quoted as a single CV number, because the previous codebase's "0.70–0.85" figure leaned on a leaky synthetic feature.

## Data flow

```
ingestion (TMDB/IMDb) → Snowflake RAW
                            ↓
                       dbt run (staging)
                            ↓
                  Snowflake STAGING.stg_box_office
                            ↓
              build_feature_pipeline() (69 features)
                            ↓
                   Snowflake ML_TRAINING.{X,Y}_TRAIN
                            ↓
                       upload to S3
                            ↓
                  SageMaker training (XGBoost)
                            ↓
                  SageMaker Model Registry
                            ↓ (R² ≥ 0.75 gate)
                       Approved
                            ↓
                Lambda inference (loads from registry)
```

## Pipeline (three phases)

`[box_office/orchestration/flows/ml_pipeline.py](../../box_office/orchestration/flows/ml_pipeline.py)` delegates to phase modules:

| Phase | Module | Responsibility |
| ----- | ------ | -------------- |
| Data | [`phases/data_phase.py`](../../box_office/orchestration/phases/data_phase.py) | dbt → staging load → temporal split → features → scale → targets → artifacts → batch Snowflake save → `ML_TRAINING` validation |
| Train | [`phases/train_phase.py`](../../box_office/orchestration/phases/train_phase.py) | In-memory `(X_TRAIN_SCALED, Y_TRAIN_LOG)` upload to S3 → SageMaker training → parse output metrics |
| Registry | [`phases/registry_phase.py`](../../box_office/orchestration/phases/registry_phase.py) | Register model package → R² gate → optional promotion |

Snowflake `ML_TRAINING` tables remain the **audit/canonical store** after each data phase. SageMaker upload uses **in-memory frames** after successful saves (no Snowflake reload in the default flow). For manual/debug reload, use `sagemaker_train_job.load_processed_data_from_snowflake()`.

Prefect `@task` wrappers in [`data_tasks`](../../box_office/orchestration/tasks/data_tasks.py) and [`training_tasks`](../../box_office/orchestration/tasks/training_tasks.py) retain retries and logging.

## Feature engineering

`build_feature_pipeline()` ([`box_office/ml/feature_pipeline/`](../../box_office/ml/feature_pipeline/)) returns a single sklearn `Pipeline` of six augmenting transformers plus a final raw-column strip. Turns 12 raw columns into **69 engineered features** in ~3 sec on 2,400 rows.

| Step                  | Adds | What it captures                                                                              |
| --------------------- | ---- | --------------------------------------------------------------------------------------------- |
| Core numerical        | 8    | Numeric pass-through with type coercion + missing-column fill                                 |
| Temporal              | 18   | Release timing — summer/holiday/Oscar windows, COVID era, day-of-week                         |
| Genre                 | 9    | Binary genre vectors + super-genre encoding                                                   |
| Industry              | 6    | Frequency encoding of director / studio / actor / MPAA                                        |
| Financial             | 8    | Total budget, ad/prod ratio, CPI-adjusted budget (real CPI table; no proxy)                   |
| Interactions          | 20   | Rating × votes, COVID × budget, seasonal × budget, blockbuster multipliers                    |

`TransformerMixin`-style augmenting steps: each one returns the input frame plus its new columns, so the pipeline is debuggable via `set_output(transform="pandas")`. The final `_SelectEngineered` step drops the raw input columns the transformers consumed.

**Removed in the leakage cleanup**: any feature derived from `social_media_buzz`, plus the `_fill_missing_budget` target-conditional imputation. `tests/test_feature_leakage_guard.py` rejects any column whose absolute Pearson correlation with the log-target exceeds 0.99 on a synthetic year.

## Training

XGBoost regressor with an **expanding-window per-year backtest** (forward chaining by `RELEASE_YEAR`):

```
Fold 1: Train on <2020, evaluate on 2020
Fold 2: Train on <2021, evaluate on 2021
Fold 3: Train on <2022, evaluate on 2022
Fold 4: Train on <2023, evaluate on 2023
Fold 5: Train on <2024, evaluate on 2024     # default: --backtest-years 5
```

Hyperparameters: `n_estimators=2000`, `learning_rate=0.05`, `max_depth=6`, early stopping at 50 rounds. Targets are log-transformed for stability — model.py exposes the loss as `rmse_on_log_scale`, deliberately not as `root_mean_squared_log_error` (it's the RMSE of `log(y)`, not `RMSLE`).

Each fold reports both **dollar-space R²** and **median APE** alongside the log-space loss. The per-fold output is merged with a `LogBudgetBaseline` (revenue ≈ a · budget^b, fit on the same training window) by `box_office.ml.backtest_report` to produce the final per-year table the README quotes.

Per-fold failures are caught and logged; the loop raises `CrossValidationFailed` only if every fold fails (with `__cause__` chained to the last exception).

## Model registry

Production model lifecycle:

1. **Development** — newly registered, `PendingManualApproval`
2. **Validation** — R² ≥ 0.75 OOF threshold
3. **Promotion** — automated approval if validation passes
4. **Production** — Lambda loads latest `Approved` package on cold start

Each registration writes a SHA256 manifest of the artifact tarball into `CustomerMetadataProperties`. The inference loader verifies the SHA256 against the manifest **before** any `pickle.load` / `joblib.load` — closing the bucket-write → RCE surface. The same metadata carries `feature_schema_version`; the loader rejects artifacts whose version doesn't match the runtime (currently `v2`) with a `FeatureSchemaVersionMismatch` exception.

CLI: `[box_office/ml/model_registry/aws_model_registry_cli.py](../../box_office/ml/model_registry/aws_model_registry_cli.py)`.

## Snowflake schema

```
BOX_OFFICE database
├── RAW              (source data ingested via box_office.ingestion)
│   └── BOX_OFFICE_V3
├── STAGING          (dbt-transformed)
│   └── STG_BOX_OFFICE
├── ML_TRAINING      (processed datasets)
│   ├── X_TRAIN / X_TRAIN_SCALED
│   └── Y_TRAIN / Y_TRAIN_LOG
└── FEATURE_STORE    (feature metadata + lineage)
```

dbt runs as a least-privilege `DBT_RUNNER` role (not `ACCOUNTADMIN`). The role owns the `STAGING` and `ML_TRAINING` schemas so it can recreate dbt models; it has read-only access to `RAW`.

## Configuration

`box_office/config.py` is a `pydantic-settings.BaseSettings` model. Sources, in priority order:

1. Environment variables (`pydantic-settings` handles type coercion)
2. `.env`
3. `config.yaml` (legacy; settings inherit defaults from this file when set)
4. Defaults declared on the model

The `config` singleton exposes nested *frozen-dataclass* views (`config.aws`, `config.snowflake`, `config.model`) so call sites read like `config.x.y.z` — but the underlying source of truth is the flat `Settings` model, not a bespoke proxy. `tests/test_config.py::test_every_documented_env_var_is_known` keeps the README env-var table honest.

```python
from box_office.config import config

config.aws.region                          # eu-north-1
config.snowflake.database                  # BOX_OFFICE
config.snowflake.schemas.staging           # STAGING
config.model.promotion_threshold           # 0.75
config.model.hyperparameters.n_estimators  # 2000
```

## Inference

Lambda container (`[infrastructure/docker/inference/Dockerfile](../../infrastructure/docker/inference/Dockerfile)`) loads the latest approved model from the registry, verifies SHA256 against the manifest, then serves predictions via FastAPI + Mangum at the Function URL.

Cold-start budget includes one `describe_model_package`, one S3 download, one SHA256 verify, one tar extract (with the `data` filter), and joblib loads of model + preprocessor + scaler. Warm invocations skip everything via the process-local `_extracted_artifacts_cache` dict.

The Lambda runs as `USER 1001` (non-root) inside the container.

The Function URL uses AWS `authorization_type = NONE` (public HTTPS); access control is **application-layer** via optional `X-API-Key` when `API_KEY` is set on the Lambda (merge `API_KEY` into `additional_environment_variables` in Terraform so the same key is not hardcoded; `/health` stays unauthenticated for probes).

## Batch throughput and scale assumptions

The training and orchestration code paths assume **warehouse-sized tables that comfortably fit in the Python driver process** (this project’s historic volume is on the order of a few thousand movies): Snowflake reads use `fetch_pandas_all()` / full-result pulls, and the Prefect flow may round-trip processed frames through Snowflake after already holding them in memory. That is intentional simplicity at this dataset scale — **not** the pattern you would keep for multi-million-row fact tables without chunked reads, **COPY INTO** / unload-to-S3 staging, or Snowpark pushdown.

The FastAPI `/predict` handler runs CPU-bound `PredictionEngine.predict` in **`asyncio.to_thread`** so concurrent requests do not block the event loop while the model scores a batch.

## Security posture

- **No `ACCOUNTADMIN`** for runtime: dbt uses `DBT_RUNNER` with scoped grants.
- **No `AmazonSageMakerFullAccess`** on the SageMaker execution role: scoped inline policy only.
- **No IAM mutation** in the GitHub Actions role: any IAM change must come from a manual elevated apply.
- **Manifest verification**: every `pickle.load` is preceded by SHA256 check against the trusted Model Package metadata.
- **CI secrets**: written to `${RUNNER_TEMP}` via `printf`, `chmod 600`, never to `./keys/` in the workspace.
- **Action pinning**: `astral-sh/setup-uv` pinned to a specific uv version; `claude-code-action` pinned to a commit SHA.
- **Fork-PR guard**: secret-handling jobs in `claude-code-review.yml` skip on fork PRs.

A `tests/infrastructure/` suite asserts the posture offline by parsing `iam.tf` and `profiles.yml` — runs without `terraform init` or AWS access.

## Where to look next

- For day-to-day commands: [README.md](../../README.md)
- For AI-agent entry rules: [AGENTS.md](../../AGENTS.md)
- For tests, style, and git workflow: [development.md](development.md)
- For the actual code, the table above maps every layer to its module.
