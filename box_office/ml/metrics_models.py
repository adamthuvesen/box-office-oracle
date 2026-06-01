"""Typed pipeline and training metrics (Pydantic)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
    feature_categories: Dict[str, int]
    feature_names: List[str]


class CvResultsMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    mean_cv_mae: Optional[float] = None
    std_cv_mae: Optional[float] = None
    mean_cv_rmsle: Optional[float] = None
    std_cv_rmsle: Optional[float] = None
    mean_best_iteration: Optional[float] = None
    cv_scores: List[float] = Field(default_factory=list)


class OofResultsMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    oof_r2: Optional[float] = None
    oof_mae: Optional[float] = None
    oof_rmsle: Optional[float] = None
    num_oof_samples: Optional[int] = None


class TrainingMetrics(BaseModel):
    """Metrics extracted from a SageMaker training job and output tarball."""

    model_config = ConfigDict(extra="allow")

    job_name: str = ""
    duration: float = 0.0
    model_data_url: Optional[str] = None
    training_duration_seconds: Optional[float] = None
    training_duration_minutes: Optional[float] = None
    cv_mean_mae: Optional[float] = None
    cv_std_mae: Optional[float] = None
    cv_mean_rmsle: Optional[float] = None
    cv_std_rmsle: Optional[float] = None
    cv_mean_best_iteration: Optional[float] = None
    oof_r2: Optional[float] = None
    oof_mae: Optional[float] = None
    oof_rmsle: Optional[float] = None
    oof_num_samples: Optional[int] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    cv_results: Optional[CvResultsMetrics] = None
    oof_results: Optional[OofResultsMetrics] = None
    training_time: Optional[float] = None
    estimated_cost: Optional[float] = None

    @classmethod
    def from_performance_dict(cls, data: Dict[str, Any]) -> "TrainingMetrics":
        """Build from the flat dict produced by training output parsing."""
        return cls.model_validate(data)

    def to_performance_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class AwsRegistrationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    model_package_arn: Optional[str] = None
    approval_status: Optional[str] = None
    error: Optional[str] = None


class ModelRegistrationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    aws_result: Optional[AwsRegistrationResult] = None

    @classmethod
    def from_task_dict(cls, data: Dict[str, Any]) -> "ModelRegistrationResult":
        aws = data.get("aws_result")
        if isinstance(aws, dict):
            data = {**data, "aws_result": AwsRegistrationResult.model_validate(aws)}
        return cls.model_validate(data)


class PromotionValidationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    promote: bool = False
    validation_details: Dict[str, Any] = Field(default_factory=dict)


class AwsPromotionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    error: Optional[str] = None
    promotion_time_seconds: Optional[float] = None


class ModelRegistryMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_registration: Optional[ModelRegistrationResult] = None
    model_promotion_validation: Optional[PromotionValidationResult] = None
    aws_promotion: Optional[AwsPromotionResult] = None

    @classmethod
    def from_task_dict(
        cls, data: Optional[Dict[str, Any]]
    ) -> Optional["ModelRegistryMetrics"]:
        if data is None:
            return None
        return cls.model_validate(data)


class PipelineExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    pipeline_id: str
    execution_time: Dict[str, Any]
    data_summary: Dict[str, Any]
    feature_summary: Dict[str, Any]
    training_summary: Dict[str, Any]
    model_registry_summary: Optional[Dict[str, Any]] = None
    status: str = "completed_successfully"


class PipelineMetrics(BaseModel):
    """Alias for the final JSON-serializable pipeline summary."""

    model_config = ConfigDict(extra="allow")

    summary: PipelineExecutionSummary

    def to_dict(self) -> Dict[str, Any]:
        return self.summary.model_dump()
