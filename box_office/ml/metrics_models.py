"""Typed pipeline and training metrics (Pydantic)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineStartMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    pipeline_start_time: str
    pipeline_id: str
    framework: str = "prefect-xgboost-sagemaker"


class DataProcessingMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    training_samples: int
    validation_samples: int
    feature_count: int
    target_column: str
    train_val_ratio: float


class FeatureEngineeringMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_features: int
    feature_categories: dict[str, int]
    feature_names: list[str]


class CvResultsMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    mean_cv_mae: float | None = None
    std_cv_mae: float | None = None
    mean_cv_rmsle: float | None = None
    std_cv_rmsle: float | None = None
    mean_best_iteration: float | None = None
    cv_scores: list[float] = Field(default_factory=list)


class OofResultsMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    oof_r2: float | None = None
    oof_mae: float | None = None
    oof_rmsle: float | None = None
    num_oof_samples: int | None = None


class TrainingMetrics(BaseModel):
    """Metrics extracted from a SageMaker training job and output tarball."""

    model_config = ConfigDict(extra="allow")

    job_name: str = ""
    duration: float = 0.0
    model_data_url: str | None = None
    training_duration_seconds: float | None = None
    training_duration_minutes: float | None = None
    cv_mean_mae: float | None = None
    cv_std_mae: float | None = None
    cv_mean_rmsle: float | None = None
    cv_std_rmsle: float | None = None
    cv_mean_best_iteration: float | None = None
    oof_r2: float | None = None
    oof_mae: float | None = None
    oof_rmsle: float | None = None
    oof_num_samples: int | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    cv_results: CvResultsMetrics | None = None
    oof_results: OofResultsMetrics | None = None
    training_time: float | None = None
    estimated_cost: float | None = None

    @classmethod
    def from_performance_dict(cls, data: dict[str, Any]) -> TrainingMetrics:
        """Build from the flat dict produced by training output parsing."""
        return cls.model_validate(data)

    def to_performance_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class AwsRegistrationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    model_package_arn: str | None = None
    approval_status: str | None = None
    error: str | None = None


class ModelRegistrationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    aws_result: AwsRegistrationResult | None = None

    @classmethod
    def from_task_dict(cls, data: dict[str, Any]) -> ModelRegistrationResult:
        aws = data.get("aws_result")
        if isinstance(aws, dict):
            data = {**data, "aws_result": AwsRegistrationResult.model_validate(aws)}
        return cls.model_validate(data)


class PromotionValidationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    promote: bool = False
    validation_details: dict[str, Any] = Field(default_factory=dict)


class AwsPromotionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    error: str | None = None
    promotion_time_seconds: float | None = None


class ModelRegistryMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_registration: ModelRegistrationResult | None = None
    model_promotion_validation: PromotionValidationResult | None = None
    aws_promotion: AwsPromotionResult | None = None

    @classmethod
    def from_task_dict(cls, data: dict[str, Any] | None) -> ModelRegistryMetrics | None:
        if data is None:
            return None
        return cls.model_validate(data)


class PipelineExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    pipeline_id: str
    execution_time: dict[str, Any]
    data_summary: dict[str, Any]
    feature_summary: dict[str, Any]
    training_summary: dict[str, Any]
    model_registry_summary: dict[str, Any] | None = None
    status: str = "completed_successfully"


class PipelineMetrics(BaseModel):
    """Alias for the final JSON-serializable pipeline summary."""

    model_config = ConfigDict(extra="allow")

    summary: PipelineExecutionSummary

    def to_dict(self) -> dict[str, Any]:
        return self.summary.model_dump()
