"""Project configuration: ``pydantic-settings`` over a flat env surface.

All callers use the module-level ``config`` instance and reach through
nested views (e.g. ``config.aws.region``, ``config.snowflake.schemas.staging``).
Env vars are loaded once at import time from ``os.environ`` plus an optional
``.env`` file in the project root.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class _AwsView:
    region: str
    s3_bucket: str
    account_id: str | None
    sagemaker_role_arn: str


@dataclass(frozen=True)
class _SnowflakeSchemas:
    raw: str
    staging: str
    feature_store: str
    ml_training: str


@dataclass(frozen=True)
class _SnowflakeView:
    user: str
    account: str
    database: str
    warehouse: str
    password: str | None
    private_key_path: str | None
    private_key_passphrase: str | None
    schemas: _SnowflakeSchemas


@dataclass(frozen=True)
class _ModelHyperparams:
    n_estimators: int = 1500
    learning_rate: float = 0.04
    max_depth: int = 4
    min_child_weight: int = 2
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.01
    reg_lambda: float = 0.2
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class _ModelCV:
    cv_folds: int = 8
    start_eval_year: int = 2015
    end_year: int = 2024


@dataclass(frozen=True)
class _ModelView:
    # Minimum pooled OOF R² (dollar space) for promotion, calibrated against
    # the leakage-free local backtest
    # (results/local_retrain/iteration_report.md).
    promotion_threshold: float = 0.55
    auto_approve_models: bool = False
    artifacts_dir: str = "artifacts"
    hyperparameters: _ModelHyperparams = _ModelHyperparams()
    cross_validation: _ModelCV = _ModelCV()


@dataclass(frozen=True)
class _FeatureEngineeringView:
    max_genre_features: int = 8
    enable_covid_features: bool = True


@dataclass(frozen=True)
class _PathsView:
    data_dir: str = "data"
    transformations_dir: str = "transformations"
    project_root: str = "."


@dataclass(frozen=True)
class _SagemakerView:
    instance_type: str = "ml.m5.large"
    framework_version: str = "1.7-1"
    s3_prefix: str = "box-office"


@dataclass(frozen=True)
class _TmdbView:
    start_year: int
    end_year: int | None
    min_revenue: int
    page_limit: int


@dataclass(frozen=True)
class _HeuristicsView:
    enabled: bool = True
    seed: int = 42


@dataclass(frozen=True)
class _IngestionView:
    tmdb: _TmdbView
    heuristics: _HeuristicsView = _HeuristicsView()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    AWS_REGION: str = Field(
        default="eu-north-1",
        validation_alias=AliasChoices("AWS_REGION", "AWS_DEFAULT_REGION"),
    )
    AWS_S3_BUCKET: str = Field(
        validation_alias=AliasChoices(
            "AWS_S3_BUCKET", "S3_BUCKET_NAME", "SAGEMAKER_BUCKET"
        ),
    )
    AWS_ACCOUNT_ID: str | None = None
    SAGEMAKER_ROLE_ARN: str

    SNOWFLAKE_USER: str
    SNOWFLAKE_PASSWORD: str | None = None
    SNOWFLAKE_PRIVATE_KEY_PATH: str | None = None
    SNOWFLAKE_PRIVATE_KEY_PASSPHRASE: str | None = None
    SNOWFLAKE_ACCOUNT: str
    SNOWFLAKE_WAREHOUSE: str = "COMPUTE_WH"
    SNOWFLAKE_DATABASE: str
    SNOWFLAKE_SCHEMA_RAW: str = "RAW"
    SNOWFLAKE_SCHEMA_STAGING: str = "STAGING"
    SNOWFLAKE_SCHEMA_FEATURE_STORE: str = "FEATURE_STORE"
    SNOWFLAKE_SCHEMA_ML_TRAINING: str = "ML_TRAINING"

    TMDB_START_YEAR: int = 2024
    TMDB_END_YEAR: int | None = None
    TMDB_MIN_REVENUE: int = 50_000_000
    TMDB_PAGE_LIMIT: int = 10

    def model_post_init(self, __context) -> None:
        """Pre-compute the nested view dataclasses once.

        The views materialise at ``Settings()`` construction and subsequent
        reads are pure attribute lookups; ``config.aws is config.aws`` holds.
        """
        object.__setattr__(
            self,
            "_aws",
            _AwsView(
                region=self.AWS_REGION,
                s3_bucket=self.AWS_S3_BUCKET,
                account_id=self.AWS_ACCOUNT_ID,
                sagemaker_role_arn=self.SAGEMAKER_ROLE_ARN,
            ),
        )
        object.__setattr__(
            self,
            "_snowflake",
            _SnowflakeView(
                user=self.SNOWFLAKE_USER,
                account=self.SNOWFLAKE_ACCOUNT,
                database=self.SNOWFLAKE_DATABASE,
                warehouse=self.SNOWFLAKE_WAREHOUSE,
                password=self.SNOWFLAKE_PASSWORD,
                private_key_path=self.SNOWFLAKE_PRIVATE_KEY_PATH,
                private_key_passphrase=self.SNOWFLAKE_PRIVATE_KEY_PASSPHRASE,
                schemas=_SnowflakeSchemas(
                    raw=self.SNOWFLAKE_SCHEMA_RAW,
                    staging=self.SNOWFLAKE_SCHEMA_STAGING,
                    feature_store=self.SNOWFLAKE_SCHEMA_FEATURE_STORE,
                    ml_training=self.SNOWFLAKE_SCHEMA_ML_TRAINING,
                ),
            ),
        )
        object.__setattr__(self, "_model", _ModelView())
        object.__setattr__(self, "_feature_engineering", _FeatureEngineeringView())
        object.__setattr__(self, "_paths", _PathsView())
        object.__setattr__(self, "_sagemaker", _SagemakerView())
        object.__setattr__(
            self,
            "_ingestion",
            _IngestionView(
                tmdb=_TmdbView(
                    start_year=self.TMDB_START_YEAR,
                    end_year=self.TMDB_END_YEAR,
                    min_revenue=self.TMDB_MIN_REVENUE,
                    page_limit=self.TMDB_PAGE_LIMIT,
                ),
            ),
        )

    @property
    def aws(self) -> _AwsView:
        return self._aws

    @property
    def snowflake(self) -> _SnowflakeView:
        return self._snowflake

    @property
    def model(self) -> _ModelView:
        return self._model

    @property
    def feature_engineering(self) -> _FeatureEngineeringView:
        return self._feature_engineering

    @property
    def paths(self) -> _PathsView:
        return self._paths

    @property
    def sagemaker(self) -> _SagemakerView:
        return self._sagemaker

    @property
    def ingestion(self) -> _IngestionView:
        return self._ingestion


config = Settings()
