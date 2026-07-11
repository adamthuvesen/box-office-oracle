"""Typed SageMaker training-output metrics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
